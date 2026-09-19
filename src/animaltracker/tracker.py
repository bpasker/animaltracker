"""Object tracking for consistent species identification across frames.

Uses ByteTrack via supervision library to assign persistent IDs to detected
objects, accumulating classifications to pick the best identification.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

from .detector import Detection
from .species_names import pick_species_by_lineage, species_lineage, species_rank

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackClassification:
    """A single classification for a tracked object."""
    species: str
    confidence: float
    taxonomy: Optional[str] = None
    bbox: Optional[List[float]] = None
    frame_idx: int = 0


@dataclass
class TrackInfo:
    """Accumulated information for a tracked object."""
    track_id: int
    classifications: List[TrackClassification] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    best_frame: Optional[np.ndarray] = None
    best_confidence: float = 0.0
    best_bbox: Optional[List[float]] = None
    # Store best frame per species for better key frame selection after merging
    species_best_frames: Dict[str, Tuple[np.ndarray, float, List[float]]] = field(default_factory=dict)
    
    def add_classification(
        self, 
        species: str, 
        confidence: float, 
        taxonomy: Optional[str],
        bbox: Optional[List[float]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None
    ) -> None:
        """Add a classification to this track."""
        self.classifications.append(TrackClassification(
            species=species,
            confidence=confidence,
            taxonomy=taxonomy,
            bbox=bbox,
            frame_idx=frame_idx,
        ))
        self.last_seen_frame = frame_idx
        
        # Keep the best frame overall (highest confidence)
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = bbox
            if frame is not None:
                self.best_frame = frame.copy()
        
        # Also track best frame per species (for key frame extraction)
        if frame is not None:
            if species not in self.species_best_frames:
                self.species_best_frames[species] = (frame.copy(), confidence, bbox)
            else:
                existing_conf = self.species_best_frames[species][1]
                if confidence > existing_conf:
                    self.species_best_frames[species] = (frame.copy(), confidence, bbox)
    
    def get_best_species(self) -> Tuple[str, float, Optional[str]]:
        """Determine the best species based on accumulated classifications.

        The votes are read as a walk down the taxonomy (see
        ``species_names.pick_species_by_lineage``): a specific label beats its
        own generic ancestors even on a handful of frames at lower confidence
        (canidae > carnivorous mammal > mammal > animal), while labels that
        contradict each other are settled by their votes. Ranking on
        specificity alone let one misread frame rename a track of forty.

        Returns:
            (species, confidence, taxonomy) tuple
        """
        if not self.classifications:
            return "", 0.0, None
        
        # Group by species
        species_data: Dict[str, Dict] = {}
        
        for c in self.classifications:
            if c.species not in species_data:
                species_data[c.species] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'max_confidence': 0.0,
                    'taxonomy': c.taxonomy,
                    'specificity': self._calculate_specificity(c.species),
                }
            
            species_data[c.species]['count'] += 1
            species_data[c.species]['total_confidence'] += c.confidence
            species_data[c.species]['max_confidence'] = max(
                species_data[c.species]['max_confidence'],
                c.confidence
            )
        
        # Log candidates for debugging
        candidates_str = [f"{s}({d['max_confidence']:.1%})" for s, d in species_data.items()]
        LOGGER.debug("Track %d candidates: %s", self.track_id, candidates_str)
        
        # Walk down the taxonomy by votes: count first, then confidence.
        best_species = pick_species_by_lineage({
            s: (d['count'], d['max_confidence']) for s, d in species_data.items()
        })

        LOGGER.debug("Track %d selected '%s' (specificity=%d) from %d candidates: %s",
                    self.track_id, best_species, species_data[best_species]['specificity'],
                    len(species_data), list(species_data.keys()))

        return (
            best_species,
            species_data[best_species]['max_confidence'],
            species_data[best_species]['taxonomy']
        )
    
    def get_best_frame(self) -> Optional[Tuple[np.ndarray, float, List[float]]]:
        """Get the best frame for this track (highest confidence detection).
        
        Returns:
            (frame, confidence, bbox) tuple or None if no frame stored
        """
        if self.best_frame is None:
            return None
        return (self.best_frame, self.best_confidence, self.best_bbox)
    
    def get_best_frame_for_species(self, target_species: str) -> Optional[Tuple[np.ndarray, float, List[float]]]:
        """Get the best frame for a specific species in this track.
        
        After track merging, we may want the best frame specifically for the
        selected species (e.g., bovidae), not the overall highest confidence
        frame (which might be a generic "animal" detection).
        
        Args:
            target_species: The species name to find the best frame for
            
        Returns:
            (frame, confidence, bbox) tuple or None if no matching frame stored
        """
        # Check species_best_frames first (preferred)
        if target_species in self.species_best_frames:
            return self.species_best_frames[target_species]
        
        # Fall back to overall best frame if species not found
        if self.best_frame is not None:
            return (self.best_frame, self.best_confidence, self.best_bbox)
        
        return None
    
    def _calculate_specificity(self, species: str) -> int:
        """How specific a species name is: its taxonomy depth.

        0 "animal", 1 class ("mammalia_mammal", "bird"), 2 order
        ("mammalia_rodentia_rodent"), 3 family ("mammalia_rodentia_sciuridae"),
        4+ below family. Every label at one level gets the same score; see
        ``species_names.species_rank``.
        """
        return species_rank(species)


class ObjectTracker:
    """Tracks objects across frames and accumulates species classifications."""
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 120,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 15,
    ):
        """Initialize the object tracker.
        
        Args:
            track_activation_threshold: Min confidence to start a track
            lost_track_buffer: Frames to keep lost tracks alive. Higher values
                              help maintain identity through detection gaps.
                              Default 120 handles ~8s gaps at 15fps.
            minimum_matching_threshold: Highest matching cost ByteTrack accepts
                              when continuing a track, where the cost is
                              1 - IoU x detection confidence. Higher is MORE
                              permissive. 0.8 is the supervision default. The
                              0.1 used until 2026-09-09 (meant as an IoU floor)
                              demanded IoU x confidence >= 0.9, so no detection
                              ever continued a track and every track was a
                              single frame that the merge passes had to stitch.
            frame_rate: Expected frame rate (for buffer calculations)
        """
        if not SUPERVISION_AVAILABLE:
            raise RuntimeError(
                "supervision library not installed. Run: pip install supervision>=0.19"
            )
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        
        # track_id -> TrackInfo
        self.tracks: Dict[int, TrackInfo] = {}
        self.frame_count = 0
    
    def update(
        self, 
        detections: List[Detection], 
        frame: Optional[np.ndarray] = None,
        frame_idx: Optional[int] = None
    ) -> Dict[int, Detection]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detections from the detector
            frame: Current frame (optional, for storing best frames)
            frame_idx: Actual video frame index (optional, defaults to internal counter)
            
        Returns:
            Dict mapping track_id -> Detection for this frame
        """
        self.frame_count += 1
        
        # Use provided frame_idx or fall back to internal counter
        actual_frame_idx = frame_idx if frame_idx is not None else self.frame_count
        
        if not detections:
            # Still update ByteTrack with empty detections so its internal
            # frame counter advances. This is critical for:
            # 1. Kalman filter predictions staying accurate over time
            # 2. lost_track_buffer expiring correctly
            # 3. Proper track association when detections resume
            sv_empty = sv.Detections.empty()
            self.tracker.update_with_detections(sv_empty)
            return {}
        
        # Convert to supervision format
        bboxes = np.array([d.bbox for d in detections])
        confidences = np.array([d.confidence for d in detections])
        
        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
        )
        
        # Update tracker
        tracked = self.tracker.update_with_detections(sv_detections)

        # Map results back and accumulate classifications
        result: Dict[int, Detection] = {}

        if tracked.tracker_id is None:
            return result

        def _iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
            union = area_a + area_b - inter
            if union <= 0:
                return 0.0
            return inter / union

        used_indices: set[int] = set()
        for i, track_id in enumerate(tracked.tracker_id):
            if track_id is None:
                continue

            track_id = int(track_id)

            # Match the tracked output to its source Detection by IoU.
            # ByteTrack's Kalman filter can shift the bbox by more than 1px,
            # so np.allclose(atol=1) misses real matches; the i-th index
            # fallback is only correct when ByteTrack returns one output per
            # input in the same order, which is not guaranteed.
            tracked_bbox = tracked.xyxy[i]
            best_iou = 0.0
            best_idx = -1
            for j, det in enumerate(detections):
                if j in used_indices:
                    continue
                iou = _iou(det.bbox, tracked_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx < 0 or best_iou < 0.3:
                # Couldn't confidently match this tracked output to any input
                # detection. Skip rather than risk attaching the wrong id.
                LOGGER.debug(
                    "Track %d: no input detection matched tracked bbox (best IoU=%.2f)",
                    track_id, best_iou
                )
                continue

            used_indices.add(best_idx)
            original_det = detections[best_idx]

            # Initialize track if new
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackInfo(
                    track_id=track_id,
                    first_seen_frame=actual_frame_idx,
                )

            # Add classification to track
            self.tracks[track_id].add_classification(
                species=original_det.species,
                confidence=original_det.confidence,
                taxonomy=original_det.taxonomy,
                bbox=original_det.bbox,
                frame_idx=actual_frame_idx,
                frame=frame,
            )

            # Stamp the persistent id back onto the detection so downstream
            # consumers (PTZ tracker lock) can use it directly.
            original_det.track_id = track_id

            result[track_id] = original_det

        return result
    
    def get_track_species(self, track_id: int) -> Tuple[str, float, Optional[str]]:
        """Get the best species classification for a track.
        
        Returns:
            (species, confidence, taxonomy) tuple
        """
        if track_id not in self.tracks:
            return "", 0.0, None
        
        return self.tracks[track_id].get_best_species()
    
    def get_all_species(self) -> Dict[str, Dict]:
        """Get best species for all tracked objects.
        
        Returns:
            Dict mapping track_id -> {species, confidence, taxonomy, track_info}
        """
        results = {}
        
        for track_id, track_info in self.tracks.items():
            species, confidence, taxonomy = track_info.get_best_species()
            if species:
                results[track_id] = {
                    'species': species,
                    'confidence': confidence,
                    'taxonomy': taxonomy,
                    'classification_count': len(track_info.classifications),
                    'frames_visible': track_info.last_seen_frame - track_info.first_seen_frame + 1,
                    'best_frame': track_info.best_frame,
                    'best_bbox': track_info.best_bbox,
                }
        
        return results
    
    def get_unique_species(self) -> List[Tuple[str, float]]:
        """Get list of unique species across all tracks.
        
        Returns:
            List of (species, max_confidence) tuples, deduplicated
        """
        species_best: Dict[str, float] = {}
        
        for track_id, track_info in self.tracks.items():
            species, confidence, _ = track_info.get_best_species()
            if species:
                if species not in species_best or confidence > species_best[species]:
                    species_best[species] = confidence
        
        return [(s, c) for s, c in species_best.items()]
    
    def reset(self) -> None:
        """Reset the tracker state for a new event."""
        self.tracker.reset()
        self.tracks.clear()
        self.frame_count = 0
    
    def _get_species_hierarchy(self, species: str) -> tuple:
        """Get the hierarchy category and specificity of a species.

        Returns:
            (category, specificity) where category is 'bird', 'mammal',
            'reptile', 'amphibian' or 'animal' (no class known), and
            specificity is the label's taxonomy depth, on the scale
            ``TrackInfo`` and the post-processor use
            (``species_names.species_rank``): 0 "animal", 1 class, 2 order,
            3 family, 4+ below family.

        The scale used to come from keyword lists of its own, which scored a
        listed family (felidae) above an unlisted one at the same level
        (mephitidae, bovidae). ``merge_hierarchical_tracks`` absorbs the less
        specific of two tracks, so it folded a skunk into a cat and a
        twenty-detection cow into a four-detection deer.
        """
        lineage = species_lineage(species)
        categories = {
            'mammalia': 'mammal', 'bird': 'bird',
            'reptile': 'reptile', 'amphibian': 'amphibian',
        }
        category = categories.get(lineage[0], 'animal') if lineage else 'animal'
        return (category, species_rank(species))

    def _species_compatible(self, species1: str, species2: str) -> bool:
        """Check if two species are compatible for merging.
        
        Species are compatible if:
        1. One is more generic than the other (e.g., "animal" and "canidae")
        2. They're in the same category hierarchy (both mammals, both birds, etc.)
        
        Returns:
            True if species can be merged (one subsumes the other)
        """
        cat1, spec1 = self._get_species_hierarchy(species1)
        cat2, spec2 = self._get_species_hierarchy(species2)
        
        # "animal" is compatible with everything
        if cat1 == 'animal' or cat2 == 'animal':
            return True
        
        # Same category - compatible (e.g., both mammals)
        if cat1 == cat2:
            return True
        
        # Different categories (bird vs mammal) - not compatible
        return False
    
    def merge_similar_tracks(self, max_frame_gap: int = 60) -> int:
        """Merge tracks that likely represent the same animal.
        
        Tracks are merged if they:
        1. Have the same best species classification
        2. Don't have overlapping frame ranges (not two animals at once)
        3. Are temporally close (within max_frame_gap of each other)
        
        Args:
            max_frame_gap: Maximum gap between track end and next track start
                          to consider them the same animal.
        
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Group tracks by their best species
        species_tracks: Dict[str, List[int]] = {}
        for track_id, track_info in self.tracks.items():
            species, _, _ = track_info.get_best_species()
            if species:
                if species not in species_tracks:
                    species_tracks[species] = []
                species_tracks[species].append(track_id)
        
        merged_count = 0
        tracks_to_remove = set()
        
        for species, track_ids in species_tracks.items():
            if len(track_ids) <= 1:
                continue
            
            # Sort tracks by first_seen_frame
            track_ids_sorted = sorted(
                track_ids, 
                key=lambda tid: self.tracks[tid].first_seen_frame
            )
            
            # Check for non-overlapping tracks that can be merged
            primary_track_id = track_ids_sorted[0]
            primary = self.tracks[primary_track_id]
            
            for other_id in track_ids_sorted[1:]:
                if other_id in tracks_to_remove:
                    continue
                    
                other = self.tracks[other_id]
                
                # Check for ACTUAL detection frame overlap (not just range overlap)
                # After spatial merges, ranges can overlap even though detections don't
                primary_frames = {c.frame_idx for c in primary.classifications}
                other_frames = {c.frame_idx for c in other.classifications}
                actual_overlap = primary_frames & other_frames
                
                if actual_overlap:
                    # These have detections at the same frames - might be two different animals
                    # Update primary to be the one with more detections
                    if len(other.classifications) > len(primary.classifications):
                        primary_track_id = other_id
                        primary = other
                    continue
                
                # Check if gap is small enough (using actual detection frames)
                primary_max = max(primary_frames) if primary_frames else primary.last_seen_frame
                other_min = min(other_frames) if other_frames else other.first_seen_frame
                gap = other_min - primary_max
                
                if gap <= max_frame_gap and gap >= 0:
                    # Merge other into primary
                    LOGGER.info("Merging Track %d into Track %d (same %s, gap=%d frames)",
                               other_id, primary_track_id, species, gap)
                    
                    # Copy all classifications
                    primary.classifications.extend(other.classifications)
                    
                    # Update frame range (must update BOTH first and last)
                    primary.first_seen_frame = min(primary.first_seen_frame, other.first_seen_frame)
                    primary.last_seen_frame = max(primary.last_seen_frame, other.last_seen_frame)
                    
                    # Update best frame if other's is better OR if primary has no frame
                    if other.best_frame is not None:
                        if primary.best_frame is None or other.best_confidence > primary.best_confidence:
                            primary.best_confidence = other.best_confidence
                            primary.best_bbox = other.best_bbox
                            primary.best_frame = other.best_frame
                    
                    # Merge species_best_frames (keep best confidence per species, or copy if missing)
                    for sp, frame_data in other.species_best_frames.items():
                        if sp not in primary.species_best_frames:
                            primary.species_best_frames[sp] = frame_data
                        elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                            existing = primary.species_best_frames[sp]
                            # Copy if primary has no frame for this species, or if other has better confidence
                            if existing[0] is None or frame_data[1] > existing[1]:
                                primary.species_best_frames[sp] = frame_data
                    
                    tracks_to_remove.add(other_id)
                    merged_count += 1
                else:
                    # Gap too large, other becomes the new primary for subsequent tracks
                    primary_track_id = other_id
                    primary = other
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Merged %d tracks (same species), %d tracks remaining", merged_count, len(self.tracks))
        
        return merged_count
    
    def merge_hierarchical_tracks(self, max_frame_gap: int = 120, min_specific_detections: int = 2) -> int:
        """Merge generic tracks into more specific compatible tracks.
        
        This is a second-pass merge that handles cases like:
        - "animal" track getting absorbed into "canidae" track
        - "mammalia_mammal" track absorbed into "canidae" track
        
        Only merges when:
        1. Tracks don't overlap in time (not two animals at once)
        2. Tracks are temporally adjacent (within max_frame_gap)
        3. The specific track has enough detections to be reliable (min_specific_detections)
        4. Species are hierarchically compatible (same animal type family)
        
        Args:
            max_frame_gap: Maximum gap between tracks to consider merging
            min_specific_detections: Minimum detections in specific track to be merge target
        
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Build list of (track_id, species, specificity, category, track_info)
        track_data = []
        for track_id, track_info in self.tracks.items():
            species, confidence, _ = track_info.get_best_species()
            if species:
                category, specificity = self._get_species_hierarchy(species)
                track_data.append({
                    'track_id': track_id,
                    'species': species,
                    'specificity': specificity,
                    'category': category,
                    'info': track_info,
                    'detections': len(track_info.classifications),
                })
        
        # Sort by specificity (most specific first) then by detection count
        track_data.sort(key=lambda x: (-x['specificity'], -x['detections']))
        
        merged_count = 0
        tracks_to_remove = set()
        
        # For each specific track, try to absorb nearby generic tracks
        for specific in track_data:
            if specific['track_id'] in tracks_to_remove:
                continue
            
            # Skip if not specific enough or not enough detections
            if specific['specificity'] < 2:
                continue
            if specific['detections'] < min_specific_detections:
                continue
            
            specific_info = specific['info']
            
            # Look for generic tracks to absorb
            for generic in track_data:
                if generic['track_id'] in tracks_to_remove:
                    continue
                if generic['track_id'] == specific['track_id']:
                    continue
                
                # Only absorb less specific tracks
                if generic['specificity'] >= specific['specificity']:
                    continue
                
                # Check species compatibility
                if not self._species_compatible(specific['species'], generic['species']):
                    LOGGER.debug("Skipping merge: %s and %s not compatible", 
                                specific['species'], generic['species'])
                    continue
                
                generic_info = generic['info']
                
                # Check for ACTUAL detection frame overlap (not just range overlap)
                # After spatial merges, ranges can overlap even though detections don't
                specific_frames = {c.frame_idx for c in specific_info.classifications}
                generic_frames = {c.frame_idx for c in generic_info.classifications}
                actual_overlap = specific_frames & generic_frames
                
                if actual_overlap:
                    LOGGER.debug("Skipping merge: Track %d and %d have %d overlapping detection frames",
                                specific['track_id'], generic['track_id'], len(actual_overlap))
                    continue
                
                # Check temporal proximity (using actual detection frames, not just ranges)
                specific_max = max(specific_frames) if specific_frames else specific_info.last_seen_frame
                specific_min = min(specific_frames) if specific_frames else specific_info.first_seen_frame
                generic_max = max(generic_frames) if generic_frames else generic_info.last_seen_frame
                generic_min = min(generic_frames) if generic_frames else generic_info.first_seen_frame
                
                if generic_min > specific_max:
                    gap = generic_min - specific_max
                else:
                    gap = specific_min - generic_max
                
                if gap > max_frame_gap:
                    LOGGER.debug("Skipping merge: Track %d and %d too far apart (gap=%d)",
                                specific['track_id'], generic['track_id'], gap)
                    continue
                
                # Merge generic into specific
                LOGGER.info("Hierarchical merge: Track %d (%s, %d det) <- Track %d (%s, %d det), gap=%d",
                           specific['track_id'], specific['species'], specific['detections'],
                           generic['track_id'], generic['species'], generic['detections'], gap)
                
                # Copy all classifications from generic to specific
                specific_info.classifications.extend(generic_info.classifications)
                
                # Update frame range
                specific_info.first_seen_frame = min(specific_info.first_seen_frame, 
                                                     generic_info.first_seen_frame)
                specific_info.last_seen_frame = max(specific_info.last_seen_frame, 
                                                    generic_info.last_seen_frame)
                
                # Update best frame if generic's is better OR if specific has no frame
                if generic_info.best_frame is not None:
                    if specific_info.best_frame is None or generic_info.best_confidence > specific_info.best_confidence:
                        specific_info.best_confidence = generic_info.best_confidence
                        specific_info.best_bbox = generic_info.best_bbox
                        specific_info.best_frame = generic_info.best_frame
                
                # Merge species_best_frames (keep best confidence per species, or copy if missing)
                for sp, frame_data in generic_info.species_best_frames.items():
                    if sp not in specific_info.species_best_frames:
                        specific_info.species_best_frames[sp] = frame_data
                    elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                        existing = specific_info.species_best_frames[sp]
                        # Copy if specific has no frame for this species, or if generic has better confidence
                        if existing[0] is None or frame_data[1] > existing[1]:
                            specific_info.species_best_frames[sp] = frame_data
                
                tracks_to_remove.add(generic['track_id'])
                merged_count += 1
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Hierarchical merge: absorbed %d generic tracks, %d tracks remaining", 
                       merged_count, len(self.tracks))
        
        return merged_count
    
    def merge_non_overlapping_tracks(self) -> int:
        """Aggressively merge all non-overlapping tracks into the most confident one.
        
        This is for single-animal videos where we're confident there's only one subject.
        All tracks that don't overlap in time get merged into whichever track has
        the most specific species identification with highest confidence.
        
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Find all tracks and score them
        track_scores = []
        for track_id, track_info in self.tracks.items():
            species, confidence, _ = track_info.get_best_species()
            if species:
                _, specificity = self._get_species_hierarchy(species)
                # Score: prioritize specificity, then confidence, then detection count
                score = (specificity * 100) + confidence + (len(track_info.classifications) * 0.01)
                track_scores.append({
                    'track_id': track_id,
                    'species': species,
                    'confidence': confidence,
                    'specificity': specificity,
                    'detections': len(track_info.classifications),
                    'score': score,
                    'info': track_info,
                })
        
        if not track_scores:
            return 0
        
        # Sort by score (highest first)
        track_scores.sort(key=lambda x: -x['score'])
        
        # The primary track is the highest scored one
        primary = track_scores[0]
        primary_info = primary['info']
        
        merged_count = 0
        tracks_to_remove = set()
        
        for other in track_scores[1:]:
            other_info = other['info']
            
            # Check for time overlap
            overlaps = (
                primary_info.first_seen_frame <= other_info.last_seen_frame and
                other_info.first_seen_frame <= primary_info.last_seen_frame
            )
            
            if overlaps:
                LOGGER.debug("Cannot merge Track %d: overlaps with primary Track %d",
                            other['track_id'], primary['track_id'])
                continue
            
            # Merge into primary
            LOGGER.info("Non-overlapping merge: Track %d (%s) <- Track %d (%s)",
                       primary['track_id'], primary['species'],
                       other['track_id'], other['species'])
            
            # Copy classifications
            primary_info.classifications.extend(other_info.classifications)
            
            # Update frame range
            primary_info.first_seen_frame = min(primary_info.first_seen_frame, 
                                                other_info.first_seen_frame)
            primary_info.last_seen_frame = max(primary_info.last_seen_frame, 
                                               other_info.last_seen_frame)
            
            # Update best frame if other's is better OR if primary has no frame
            if other_info.best_frame is not None:
                if primary_info.best_frame is None or other_info.best_confidence > primary_info.best_confidence:
                    primary_info.best_confidence = other_info.best_confidence
                    primary_info.best_bbox = other_info.best_bbox
                    primary_info.best_frame = other_info.best_frame
            
            # Merge species_best_frames (keep best confidence per species, or copy if missing)
            for sp, frame_data in other_info.species_best_frames.items():
                if sp not in primary_info.species_best_frames:
                    primary_info.species_best_frames[sp] = frame_data
                elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                    existing = primary_info.species_best_frames[sp]
                    if existing[0] is None or frame_data[1] > existing[1]:
                        primary_info.species_best_frames[sp] = frame_data
            
            tracks_to_remove.add(other['track_id'])
            merged_count += 1
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Non-overlapping merge: combined %d tracks into 1, %d tracks remaining",
                       merged_count, len(self.tracks))
        
        return merged_count
    
    def merge_overlapping_same_location_tracks(self, iou_threshold: float = 0.3) -> int:
        """Merge tracks that overlap in time but are spatially the same object.
        
        ByteTrack sometimes creates spurious parallel tracks when an object is temporarily
        lost and re-detected. This merges tracks that:
        1. Overlap in time (both active at same frames)
        2. Have high spatial IoU during the overlap period
        3. The smaller track gets absorbed into the larger one
        
        Args:
            iou_threshold: Minimum IoU during overlap to consider same object
            
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Build track data with frame->bbox mapping for overlap detection
        track_data = []
        for track_id, track_info in self.tracks.items():
            frame_bboxes = {}
            for c in track_info.classifications:
                if c.bbox:
                    frame_bboxes[c.frame_idx] = c.bbox
            
            if not frame_bboxes:
                continue
            
            species, confidence, _ = track_info.get_best_species()
            track_data.append({
                'track_id': track_id,
                'info': track_info,
                'first_frame': track_info.first_seen_frame,
                'last_frame': track_info.last_seen_frame,
                'frame_bboxes': frame_bboxes,
                'species': species,
                'confidence': confidence,
                'detections': len(track_info.classifications),
            })
        
        if len(track_data) <= 1:
            return 0
        
        # Sort by number of detections (merge smaller into larger)
        track_data.sort(key=lambda x: x['detections'], reverse=True)
        
        merged_count = 0
        tracks_to_remove = set()
        
        for i, larger in enumerate(track_data):
            if larger['track_id'] in tracks_to_remove:
                continue
            
            for j, smaller in enumerate(track_data[i+1:], i+1):
                if smaller['track_id'] in tracks_to_remove:
                    continue
                
                # Check for time overlap
                overlaps_in_time = (
                    larger['first_frame'] <= smaller['last_frame'] and
                    smaller['first_frame'] <= larger['last_frame']
                )
                
                if not overlaps_in_time:
                    continue
                
                # Find frames where both tracks have detections and check IoU
                overlap_ious = []
                for frame, bbox1 in larger['frame_bboxes'].items():
                    if frame in smaller['frame_bboxes']:
                        bbox2 = smaller['frame_bboxes'][frame]
                        iou = self._calculate_iou(bbox1, bbox2)
                        overlap_ious.append(iou)
                
                # If they never have detections at the same frame, check adjacent frames
                if not overlap_ious:
                    # Look for frames within a small window
                    for frame1, bbox1 in larger['frame_bboxes'].items():
                        for frame2, bbox2 in smaller['frame_bboxes'].items():
                            if abs(frame1 - frame2) <= 6:  # Within 6 frames
                                iou = self._calculate_iou(bbox1, bbox2)
                                if iou >= iou_threshold:
                                    overlap_ious.append(iou)
                                    break
                        if overlap_ious:
                            break
                
                if not overlap_ious:
                    continue
                
                # Check if average IoU meets threshold
                avg_iou = sum(overlap_ious) / len(overlap_ious)
                if avg_iou >= iou_threshold:
                    LOGGER.info(
                        "Overlap merge: Track %d (%s, %d det) <- Track %d (%s, %d det), "
                        "avg_iou=%.2f over %d shared frames",
                        larger['track_id'], larger['species'], larger['detections'],
                        smaller['track_id'], smaller['species'], smaller['detections'],
                        avg_iou, len(overlap_ious)
                    )
                    
                    larger_info = larger['info']
                    smaller_info = smaller['info']
                    
                    # Merge smaller into larger
                    larger_info.classifications.extend(smaller_info.classifications)
                    larger_info.first_seen_frame = min(larger_info.first_seen_frame,
                                                       smaller_info.first_seen_frame)
                    larger_info.last_seen_frame = max(larger_info.last_seen_frame,
                                                      smaller_info.last_seen_frame)
                    
                    # Update best frame if smaller's is better OR if larger has no frame
                    if smaller_info.best_frame is not None:
                        if larger_info.best_frame is None or smaller_info.best_confidence > larger_info.best_confidence:
                            larger_info.best_confidence = smaller_info.best_confidence
                            larger_info.best_bbox = smaller_info.best_bbox
                            larger_info.best_frame = smaller_info.best_frame
                    
                    # Merge species_best_frames (keep best confidence per species, or copy if missing)
                    for sp, frame_data in smaller_info.species_best_frames.items():
                        if sp not in larger_info.species_best_frames:
                            larger_info.species_best_frames[sp] = frame_data
                        elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                            existing = larger_info.species_best_frames[sp]
                            if existing[0] is None or frame_data[1] > existing[1]:
                                larger_info.species_best_frames[sp] = frame_data
                    
                    # Update larger's frame_bboxes for subsequent comparisons
                    larger['frame_bboxes'].update(smaller['frame_bboxes'])
                    larger['last_frame'] = larger_info.last_seen_frame
                    larger['first_frame'] = larger_info.first_seen_frame
                    larger['detections'] += smaller['detections']
                    
                    tracks_to_remove.add(smaller['track_id'])
                    merged_count += 1
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Overlap merge: merged %d spurious parallel tracks, %d tracks remaining",
                       merged_count, len(self.tracks))
        
        return merged_count

    def merge_gap_filling_tracks(
        self,
        max_detections: Optional[int] = None,
        iou_threshold: float = 0.3,
        reach: float = 1.0,
        reach_frames: int = 30,
    ) -> int:
        """Merge tracks that fill gaps in larger tracks' detection timelines.
        
        If Track A has detections at frames [100-200, 300-400] (with a gap 200-300)
        and Track B exists entirely within that gap (e.g., frames 220-280),
        Track B likely represents the same animal that was briefly lost and re-detected
        with a new track ID.

        Being sandwiched in time is not enough, though: a second animal that
        visits while the first is out of sight is sandwiched too. This pass
        took any such track, wherever it was in the frame and whatever it was
        (a cardinal in the far corner disappeared into a dog with a three
        second gap, and the clip reported one animal). So the fragment must
        also be where the larger track plausibly was: its first box near the
        larger track's last box before the gap, or its last box near the
        first one after it (``_fits_gap_spatially``). No species agreement is
        asked for, as in the other spatial passes: a dog read as a cat for a
        few frames on the dog's own path is still the dog.

        Args:
            max_detections: When set, only tracks with at most this many
                detections are absorbed, so a real second animal that stayed
                for a while keeps its own track.
            iou_threshold: Overlap that counts as "the same place".
            reach: Centre-to-centre distance, in body lengths, that counts as
                a continuation across ``reach_frames`` frames; 0 disables the
                distance test and leaves the overlap as the only criterion.
            reach_frames: The frame gap ``reach`` is meant for. A longer wait
                allows proportionally more movement, up to
                ``_GAP_FILL_MAX_REACH`` times ``reach``.
        
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Build track data with detection frame sets
        track_data = []
        for track_id, track_info in self.tracks.items():
            detection_frames = set()
            for c in track_info.classifications:
                detection_frames.add(c.frame_idx)
            
            if not detection_frames:
                continue
            
            species, confidence, _ = track_info.get_best_species()
            track_data.append({
                'track_id': track_id,
                'info': track_info,
                'first_frame': track_info.first_seen_frame,
                'last_frame': track_info.last_seen_frame,
                'detection_frames': detection_frames,
                'species': species,
                'confidence': confidence,
                'detections': len(track_info.classifications),
            })
        
        if len(track_data) <= 1:
            return 0
        
        # Sort by number of detections (merge smaller into larger)
        track_data.sort(key=lambda x: x['detections'], reverse=True)
        
        merged_count = 0
        tracks_to_remove = set()
        
        for i, larger in enumerate(track_data):
            if larger['track_id'] in tracks_to_remove:
                continue
            
            # Find gaps in the larger track's detections
            if len(larger['detection_frames']) < 2:
                continue

            def open_gaps() -> list:
                """(start_frame, end_frame) of every gap still open in ``larger``."""
                frames = sorted(larger['detection_frames'])
                return [
                    (frames[j], frames[j + 1]) for j in range(len(frames) - 1)
                    if frames[j + 1] - frames[j] > 6  # larger than typical frame skip
                ]

            gaps = open_gaps()
            if not gaps:
                continue
            
            for smaller in track_data[i+1:]:
                if smaller['track_id'] in tracks_to_remove:
                    continue
                
                if max_detections is not None and smaller['detections'] > max_detections:
                    continue

                # Check if smaller track is entirely within one of larger's gaps
                smaller_first = smaller['first_frame']
                smaller_last = smaller['last_frame']
                
                for gap_start, gap_end in gaps:
                    # Smaller track must be entirely within the gap
                    # (with some tolerance - within 3 frames of gap boundaries)
                    if (smaller_first >= gap_start - 3 and 
                        smaller_last <= gap_end + 3 and
                        smaller_first > gap_start and
                        smaller_last < gap_end):

                        if not self._fits_gap_spatially(
                            larger['info'], smaller['info'], gap_start, gap_end,
                            iou_threshold, reach, reach_frames,
                        ):
                            LOGGER.debug(
                                "Gap-fill skipped: Track %d (%s) sits in a gap of Track %d (%s) "
                                "but nowhere near it",
                                smaller['track_id'], smaller['species'],
                                larger['track_id'], larger['species'],
                            )
                            break
                        
                        LOGGER.info(
                            "Gap-fill merge: Track %d (%s, %d det, frames %d-%d) <- "
                            "Track %d (%s, %d det, frames %d-%d) fills gap %d-%d",
                            larger['track_id'], larger['species'], larger['detections'],
                            larger['first_frame'], larger['last_frame'],
                            smaller['track_id'], smaller['species'], smaller['detections'],
                            smaller_first, smaller_last,
                            gap_start, gap_end
                        )
                        
                        larger_info = larger['info']
                        smaller_info = smaller['info']
                        
                        # Merge smaller into larger
                        larger_info.classifications.extend(smaller_info.classifications)
                        larger_info.first_seen_frame = min(larger_info.first_seen_frame,
                                                           smaller_info.first_seen_frame)
                        larger_info.last_seen_frame = max(larger_info.last_seen_frame,
                                                          smaller_info.last_seen_frame)
                        
                        # Update best frame if smaller's is better OR if larger has no frame
                        if smaller_info.best_frame is not None:
                            if larger_info.best_frame is None or smaller_info.best_confidence > larger_info.best_confidence:
                                larger_info.best_confidence = smaller_info.best_confidence
                                larger_info.best_bbox = smaller_info.best_bbox
                                larger_info.best_frame = smaller_info.best_frame
                        
                        # Merge species_best_frames (keep best confidence per species, or copy if missing)
                        for sp, frame_data in smaller_info.species_best_frames.items():
                            if sp not in larger_info.species_best_frames:
                                larger_info.species_best_frames[sp] = frame_data
                            elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                                existing = larger_info.species_best_frames[sp]
                                if existing[0] is None or frame_data[1] > existing[1]:
                                    larger_info.species_best_frames[sp] = frame_data
                        
                        # The gap just got smaller (or split in two): work
                        # the rest out against what is actually still open.
                        # The list used to be built once per larger track, so
                        # two fragments that overlapped each other in time
                        # could both land in the same, by then filled, gap.
                        larger['detection_frames'].update(smaller['detection_frames'])
                        gaps = open_gaps()
                        
                        tracks_to_remove.add(smaller['track_id'])
                        merged_count += 1
                        break  # Move to next smaller track
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Gap-fill merge: absorbed %d tracks that filled detection gaps, %d tracks remaining",
                       merged_count, len(self.tracks))
        
        return merged_count

    # A fragment found after a long wait may have moved further than one after
    # a short one; this caps how much further, in multiples of ``reach``.
    _GAP_FILL_MAX_REACH = 4.0

    def _fits_gap_spatially(
        self,
        larger: TrackInfo,
        smaller: TrackInfo,
        gap_start: int,
        gap_end: int,
        iou_threshold: float,
        reach: float,
        reach_frames: int,
    ) -> bool:
        """Whether ``smaller`` is where ``larger`` plausibly was during its gap.

        Compares the fragment's first box with the larger track's last box
        before the gap, and its last box with the first one after it; either
        side is enough. "Near" is the spatial merge's test, overlap or centre
        distance in body lengths, with the distance allowed to grow with the
        frames that passed between the two boxes. Tracks without boxes give
        nothing to go on and are treated as fitting, as before.
        """
        def box_at(info: TrackInfo, frame_idx: int) -> Optional[List[float]]:
            for c in info.classifications:
                if c.frame_idx == frame_idx and c.bbox:
                    return c.bbox
            return None

        boxed = [c for c in smaller.classifications if c.bbox]
        if not boxed:
            return True
        first = min(boxed, key=lambda c: c.frame_idx)
        last = max(boxed, key=lambda c: c.frame_idx)
        before = box_at(larger, gap_start)
        after = box_at(larger, gap_end)
        if before is None and after is None:
            return True

        def near(box_a: Optional[List[float]], box_b: List[float], frames_between: int) -> bool:
            if box_a is None:
                return False
            if self._calculate_iou(box_a, box_b) >= iou_threshold:
                return True
            if reach <= 0:
                return False
            scale = max(1.0, frames_between / max(reach_frames, 1))
            allowed = reach * min(scale, self._GAP_FILL_MAX_REACH)
            return self._center_distance_ratio(box_a, box_b) <= allowed

        return (near(before, first.bbox, first.frame_idx - gap_start)
                or near(after, last.bbox, gap_end - last.frame_idx))

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1, bbox2: Bounding boxes as [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0  # No intersection
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def _center_distance_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Distance between two boxes' centres, in body lengths.

        The body length is the longest side of either box, so the same
        pixel move counts for more the smaller (further away) the animal is.
        0.0 means the boxes share a centre; a degenerate box gives ``inf``.

        Args:
            bbox1, bbox2: Bounding boxes as [x1, y1, x2, y2]

        Returns:
            Centre-to-centre distance divided by the longest box side
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        body_length = max(x2_1 - x1_1, y2_1 - y1_1, x2_2 - x1_2, y2_2 - y1_2)
        if body_length <= 0:
            return float("inf")
        dx = (x1_1 + x2_1) / 2 - (x1_2 + x2_2) / 2
        dy = (y1_1 + y2_1) / 2 - (y1_2 + y2_2) / 2
        return math.hypot(dx, dy) / body_length
    
    def merge_spatially_adjacent_tracks(
        self,
        iou_threshold: float = 0.3,
        max_frame_gap: int = 30,
        reach: float = 1.0,
    ) -> int:
        """Merge tracks that end and start in similar spatial locations.
        
        This is a simpler, more robust merge strategy:
        - If Track A ends at frame N with bounding box at position P
        - And Track B starts at frame N+gap with bounding box at position Q
        - And P and Q overlap (IoU) or Q's centre lies within ``reach``
          body lengths of P's centre
        - Then they're probably the same animal
        
        This works regardless of species labels - pure spatial continuity.

        IoU alone misses an animal that is walking away or towards the
        camera: its box shrinks or grows between the two fragments, so the
        overlap falls below the threshold although the animal barely moved.
        A dog walking away from the door (2026-09-11) ended one track with a
        175 px box and started the next a third of a body length away with a
        112 px box; the IoU was 0.34, under the configured 0.6, and the far
        fragment stood as a separate "squirrel". The centre-distance test
        measures the move in units of the longer side of either box (one
        body length), so it scales with how close the animal is.
        
        Args:
            iou_threshold: Minimum IoU between ending/starting bboxes to merge (0.3 = 30% overlap)
            max_frame_gap: Maximum frame gap to consider for spatial matching
            reach: Maximum centre-to-centre distance, in body lengths, for
                the later track's first box to count as a continuation of
                the earlier track's last box; 0 disables the distance test
                and leaves IoU as the only criterion
            
        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0
        
        # Build track data with ending/starting bbox info
        track_data = []
        for track_id, track_info in self.tracks.items():
            # Get bbox from last frame of this track
            last_bbox = None
            first_bbox = None
            last_frame = track_info.last_seen_frame
            first_frame = track_info.first_seen_frame
            
            # Find the actual bboxes at track boundaries
            for c in track_info.classifications:
                if c.bbox:
                    if c.frame_idx == last_frame or last_bbox is None:
                        last_bbox = c.bbox
                    if c.frame_idx == first_frame or first_bbox is None:
                        first_bbox = c.bbox
            
            if last_bbox is None and first_bbox is None:
                continue
                
            species, confidence, _ = track_info.get_best_species()
            
            track_data.append({
                'track_id': track_id,
                'info': track_info,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'first_bbox': first_bbox or last_bbox,
                'last_bbox': last_bbox or first_bbox,
                'species': species,
                'confidence': confidence,
                'detections': len(track_info.classifications),
            })
        
        if len(track_data) <= 1:
            return 0
        
        # Sort by first frame (chronological order)
        track_data.sort(key=lambda x: x['first_frame'])
        
        merged_count = 0
        tracks_to_remove = set()
        
        # For each track, look for later tracks that start near where this one ended
        for i, earlier in enumerate(track_data):
            if earlier['track_id'] in tracks_to_remove:
                continue
            
            earlier_info = earlier['info']
            
            for later in track_data[i+1:]:
                if later['track_id'] in tracks_to_remove:
                    continue
                
                # Check frame gap
                frame_gap = later['first_frame'] - earlier['last_frame']
                if frame_gap <= 0:
                    # Tracks overlap in time - skip. Zero counts: both tracks
                    # have a box in that frame, so they are two objects (a doe
                    # and the fawn beside her), not one that moved. Duplicate
                    # boxes of one animal are the overlap merge's job.
                    continue
                if frame_gap > max_frame_gap:
                    # Too far apart temporally
                    continue
                
                # Check spatial continuity between the end of earlier and the
                # start of later: overlap, or a centre within reach
                iou = self._calculate_iou(earlier['last_bbox'], later['first_bbox'])
                distance = self._center_distance_ratio(earlier['last_bbox'], later['first_bbox'])
                within_reach = reach > 0 and distance <= reach
                
                if iou >= iou_threshold or within_reach:
                    LOGGER.info(
                        "Spatial merge: Track %d (%s, frames %d-%d) + Track %d (%s, frames %d-%d), "
                        "IoU=%.2f, centre distance=%.2f body lengths, gap=%d frames",
                        earlier['track_id'], earlier['species'], 
                        earlier['first_frame'], earlier['last_frame'],
                        later['track_id'], later['species'],
                        later['first_frame'], later['last_frame'],
                        iou, distance, frame_gap
                    )
                    
                    later_info = later['info']
                    
                    # Merge later into earlier
                    earlier_info.classifications.extend(later_info.classifications)
                    
                    # Update frame range (must update BOTH first and last)
                    earlier_info.first_seen_frame = min(earlier_info.first_seen_frame,
                                                        later_info.first_seen_frame)
                    earlier_info.last_seen_frame = max(earlier_info.last_seen_frame, 
                                                       later_info.last_seen_frame)
                    
                    # Update best frame if later's is better OR if earlier has no frame
                    if later_info.best_frame is not None:
                        if earlier_info.best_frame is None or later_info.best_confidence > earlier_info.best_confidence:
                            earlier_info.best_confidence = later_info.best_confidence
                            earlier_info.best_bbox = later_info.best_bbox
                            earlier_info.best_frame = later_info.best_frame
                    
                    # Merge species_best_frames (keep best confidence per species, or copy if missing)
                    for sp, frame_data in later_info.species_best_frames.items():
                        if sp not in earlier_info.species_best_frames:
                            earlier_info.species_best_frames[sp] = frame_data
                        elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                            existing = earlier_info.species_best_frames[sp]
                            # Copy if earlier has no frame for this species, or if later has better confidence
                            if existing[0] is None or frame_data[1] > existing[1]:
                                earlier_info.species_best_frames[sp] = frame_data
                    
                    # Update the earlier track's last_bbox for chaining
                    earlier['last_bbox'] = later['last_bbox']
                    earlier['last_frame'] = later['last_frame']
                    
                    tracks_to_remove.add(later['track_id'])
                    merged_count += 1
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Spatial merge: merged %d tracks based on location continuity, %d tracks remaining",
                       merged_count, len(self.tracks))
        
        return merged_count
    
    def merge_weak_tracks(self, min_detections: int = 2, iou_threshold: float = 0.1) -> int:
        """Fold tracks with too few detections into the longer track spanning them.

        A track with fewer than ``min_detections`` classifications is a frame
        or two of evidence. When another track with at least ``min_detections``
        classifications was on screen before and after it (its detection
        frames bracket the weak track's), names a hierarchically compatible
        species (``_species_compatible``), and has a box near in time that
        overlaps the weak track's box by at least ``iou_threshold``, the weak
        track is almost always that same animal read differently for a frame:
        a dog called "felidae" once, an "animal" between two "canidae" frames.
        Its classifications join the host as minority votes instead of
        standing as a separate species with a thumbnail of its own.

        Weak tracks nobody spans, or that sit elsewhere in the frame, are left
        alone: this pass never invents continuity.

        Args:
            min_detections: Tracks below this many classifications are weak;
                tracks at or above it may host them.
            iou_threshold: Minimum overlap between the weak track's box and the
                host's nearest-in-time box.

        Returns:
            Number of tracks merged
        """
        if len(self.tracks) <= 1:
            return 0

        hosts = [(tid, info) for tid, info in self.tracks.items()
                 if len(info.classifications) >= min_detections]
        weak = [(tid, info) for tid, info in self.tracks.items()
                if 0 < len(info.classifications) < min_detections]
        if not hosts or not weak:
            return 0

        merged_count = 0
        tracks_to_remove = set()

        for weak_id, weak_info in weak:
            weak_species, _, _ = weak_info.get_best_species()
            weak_frames = sorted({c.frame_idx for c in weak_info.classifications})
            best_host = None
            best_iou = 0.0

            for host_id, host_info in hosts:
                host_frames = sorted({c.frame_idx for c in host_info.classifications})
                if not (host_frames[0] < weak_frames[0] and weak_frames[-1] < host_frames[-1]):
                    continue
                host_species, _, _ = host_info.get_best_species()
                if not self._species_compatible(host_species, weak_species):
                    continue
                iou = self._nearest_box_iou(host_info, weak_info)
                if iou >= iou_threshold and iou > best_iou:
                    best_host, best_iou = (host_id, host_info), iou

            if best_host is None:
                continue

            host_id, host_info = best_host
            LOGGER.info(
                "Weak-track merge: Track %d (%s, %d det) <- Track %d (%s, %d det, frames %d-%d), IoU=%.2f",
                host_id, host_info.get_best_species()[0], len(host_info.classifications),
                weak_id, weak_species, len(weak_info.classifications),
                weak_frames[0], weak_frames[-1], best_iou,
            )
            self._absorb_track(host_info, weak_info)
            tracks_to_remove.add(weak_id)
            merged_count += 1

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        if merged_count > 0:
            LOGGER.info("Weak-track merge: absorbed %d tracks, %d tracks remaining",
                        merged_count, len(self.tracks))

        return merged_count

    def _nearest_box_iou(self, host: TrackInfo, weak: TrackInfo) -> float:
        """Best IoU between the weak track's boxes and the host boxes nearest in time."""
        host_boxes = sorted(((c.frame_idx, c.bbox) for c in host.classifications if c.bbox),
                            key=lambda fb: fb[0])
        if not host_boxes:
            return 0.0

        best = 0.0
        for c in weak.classifications:
            if not c.bbox:
                continue
            before = [fb for fb in host_boxes if fb[0] <= c.frame_idx]
            after = [fb for fb in host_boxes if fb[0] >= c.frame_idx]
            for neighbour in (before[-1] if before else None, after[0] if after else None):
                if neighbour is not None:
                    best = max(best, self._calculate_iou(neighbour[1], c.bbox))
        return best

    @staticmethod
    def _absorb_track(host: TrackInfo, other: TrackInfo) -> None:
        """Move every classification and key frame of ``other`` into ``host``."""
        host.classifications.extend(other.classifications)
        host.first_seen_frame = min(host.first_seen_frame, other.first_seen_frame)
        host.last_seen_frame = max(host.last_seen_frame, other.last_seen_frame)

        if other.best_frame is not None:
            if host.best_frame is None or other.best_confidence > host.best_confidence:
                host.best_confidence = other.best_confidence
                host.best_bbox = other.best_bbox
                host.best_frame = other.best_frame

        # Keep the best frame per species, or copy it if the host has none
        for sp, frame_data in other.species_best_frames.items():
            if sp not in host.species_best_frames:
                host.species_best_frames[sp] = frame_data
            elif frame_data[0] is not None:  # frame_data = (frame, confidence, bbox)
                existing = host.species_best_frames[sp]
                if existing[0] is None or frame_data[1] > existing[1]:
                    host.species_best_frames[sp] = frame_data

    @property
    def active_track_count(self) -> int:
        """Number of tracks being tracked."""
        return len(self.tracks)


def create_tracker(
    enabled: bool = True,
    frame_rate: int = 15,
    lost_track_buffer: int = 120,
) -> Optional[ObjectTracker]:
    """Create an object tracker if available and enabled.
    
    Args:
        enabled: Whether tracking is enabled
        frame_rate: Expected frame rate
        lost_track_buffer: How many frames to keep a "lost" track alive.
                          Should be high enough to handle gaps in detections.
                          Default 120 = ~8 seconds at 15fps.
        
    Returns:
        ObjectTracker instance or None if not available/disabled
    """
    if not enabled:
        LOGGER.info("Object tracking is disabled")
        return None
    
    if not SUPERVISION_AVAILABLE:
        LOGGER.warning("supervision library not installed, tracking disabled")
        return None
    
    try:
        tracker = ObjectTracker(frame_rate=frame_rate, lost_track_buffer=lost_track_buffer)
        LOGGER.info("Object tracking enabled (ByteTrack, lost_buffer=%d)", lost_track_buffer)
        return tracker
    except Exception as e:
        LOGGER.error("Failed to create tracker: %s", e, exc_info=True)
        return None
