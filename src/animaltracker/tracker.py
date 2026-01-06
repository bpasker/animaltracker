"""Object tracking for consistent species identification across frames.

Uses ByteTrack via supervision library to assign persistent IDs to detected
objects, accumulating classifications to pick the best identification.
"""
from __future__ import annotations

import logging
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
        
        # Keep the best frame (highest confidence)
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = bbox
            if frame is not None:
                self.best_frame = frame.copy()
    
    def get_best_species(self) -> Tuple[str, float, Optional[str]]:
        """Determine the best species based on accumulated classifications.
        
        Uses a hierarchy: specific species > bird/mammal > animal.
        More specific classifications are preferred even at lower confidence.
        
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
        
        # Find most specific candidates (highest specificity score)
        max_specificity = max(d['specificity'] for d in species_data.values())
        best_candidates = {
            s: d for s, d in species_data.items() 
            if d['specificity'] == max_specificity
        }
        
        # Among equally-specific candidates, pick by count then confidence
        best_species = max(
            best_candidates.keys(),
            key=lambda s: (
                best_candidates[s]['count'],
                best_candidates[s]['max_confidence']
            )
        )
        
        LOGGER.debug("Track %d selected '%s' (specificity=%d) from %d candidates: %s",
                    self.track_id, best_species, max_specificity, len(species_data), 
                    list(species_data.keys()))
        
        return (
            best_species, 
            best_candidates[best_species]['max_confidence'],
            best_candidates[best_species]['taxonomy']
        )
    
    def get_best_frame(self) -> Optional[Tuple[np.ndarray, float, List[float]]]:
        """Get the best frame for this track (highest confidence detection).
        
        Returns:
            (frame, confidence, bbox) tuple or None if no frame stored
        """
        if self.best_frame is None:
            return None
        return (self.best_frame, self.best_confidence, self.best_bbox)
    
    def _calculate_specificity(self, species: str) -> int:
        """Calculate how specific a species name is.
        
        Hierarchy (higher = more specific):
        - 0: "animal" (most generic)
        - 1: "bird", "mammal", "reptile" (class-level)
        - 2: "hawk", "owl", "songbird" (order/family level)
        - 3+: "red-tailed_hawk", "barred_owl" (species level)
        
        This ensures "bird" always beats "animal", and specific species
        always beat generic class labels.
        """
        species_lower = species.lower().replace('-', '_').strip()
        
        # Most generic - just "animal"
        if species_lower in {'animal', 'unknown'}:
            return 0
        
        # Class level - we know what type of animal
        class_level = {'bird', 'mammal', 'aves', 'mammalia', 'reptile', 'reptilia', 
                       'amphibian', 'amphibia', 'fish'}
        if species_lower in class_level:
            return 1
        
        # Order/family level - more specific groupings
        order_level = {'hawk', 'owl', 'eagle', 'falcon', 'duck', 'goose', 'songbird',
                       'sparrow', 'finch', 'warbler', 'woodpecker', 'heron', 'gull',
                       'crow', 'jay', 'dove', 'pigeon', 'hummingbird',
                       'deer', 'bear', 'cat', 'dog', 'fox', 'coyote', 'wolf',
                       'rabbit', 'squirrel', 'mouse', 'rat', 'raccoon', 'skunk',
                       'rodent', 'carnivore', 'ungulate'}
        if species_lower in order_level:
            return 2
        
        # Species level - has underscore suggesting binomial or compound name
        words = species_lower.split('_')
        if len(words) >= 2:
            # Binomial names (genus_species) or compound common names
            return 3 + len(words)  # More parts = more specific
        
        # Single word that's not in our known categories - likely a common name
        return 2


class ObjectTracker:
    """Tracks objects across frames and accumulates species classifications."""
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 120,
        minimum_matching_threshold: float = 0.3,
        frame_rate: int = 15,
    ):
        """Initialize the object tracker.
        
        Args:
            track_activation_threshold: Min confidence to start a track
            lost_track_buffer: Frames to keep lost tracks alive. Higher values
                              help maintain identity through detection gaps.
                              Default 120 handles ~8s gaps at 15fps.
            minimum_matching_threshold: IoU threshold for matching detections to 
                              existing tracks. Lower = more forgiving of movement.
                              Default 0.3 is very permissive for fast-moving animals.
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
        frame: Optional[np.ndarray] = None
    ) -> Dict[int, Detection]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detections from the detector
            frame: Current frame (optional, for storing best frames)
            
        Returns:
            Dict mapping track_id -> Detection for this frame
        """
        self.frame_count += 1
        
        if not detections:
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
        
        for i, track_id in enumerate(tracked.tracker_id):
            if track_id is None:
                continue
            
            track_id = int(track_id)
            
            # Find original detection by matching bbox
            original_det = None
            tracked_bbox = tracked.xyxy[i]
            for det in detections:
                if np.allclose(det.bbox, tracked_bbox, atol=1.0):
                    original_det = det
                    break
            
            if original_det is None:
                # Fallback: use index if available
                if i < len(detections):
                    original_det = detections[i]
                else:
                    continue
            
            # Initialize track if new
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackInfo(
                    track_id=track_id,
                    first_seen_frame=self.frame_count,
                )
            
            # Add classification to track
            self.tracks[track_id].add_classification(
                species=original_det.species,
                confidence=original_det.confidence,
                taxonomy=original_det.taxonomy,
                bbox=original_det.bbox,
                frame_idx=self.frame_count,
                frame=frame,
            )
            
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
            (category, specificity) where category is 'bird', 'mammal', 'animal', etc.
            and specificity is how specific the identification is (higher = more specific).
            
        Taxonomy levels (higher = more specific):
            0 = "animal" (most generic)
            1 = Class level: "mammal", "bird", "reptile"
            2 = Order level: "rodent", "carnivora", "passeriformes"
            3 = Family level: "sciuridae", "canidae", "felidae", "corvidae"
            4+ = Genus/species level: specific species names
        """
        species_lower = species.lower().replace('-', '_').strip()
        
        # Most generic
        if species_lower in {'animal', 'unknown'}:
            return ('animal', 0)
        
        # Class level (specificity 1)
        if species_lower in {'bird', 'aves', 'mammal', 'mammalia', 'mammalia_mammal', 
                             'reptile', 'reptilia', 'amphibian', 'amphibia'}:
            if 'mammal' in species_lower or species_lower == 'mammalia':
                return ('mammal', 1)
            elif species_lower in {'bird', 'aves'}:
                return ('bird', 1)
            elif species_lower in {'reptile', 'reptilia'}:
                return ('reptile', 1)
            return ('animal', 1)
        
        # Determine category from taxonomy string
        category = 'animal'
        if 'mammalia' in species_lower or 'mammal' in species_lower:
            category = 'mammal'
        elif 'aves' in species_lower or 'bird' in species_lower:
            category = 'bird'
        elif 'reptilia' in species_lower:
            category = 'reptile'
        
        # Family level keywords (specificity 3) - FAMILIES are more specific than orders
        family_keywords = {
            # Mammal families
            'sciuridae', 'canidae', 'felidae', 'cervidae', 'ursidae', 'mustelidae',
            'procyonidae', 'leporidae', 'muridae', 'cricetidae', 'didelphidae',
            # Bird families  
            'corvidae', 'accipitridae', 'strigidae', 'anatidae', 'columbidae',
            'picidae', 'trochilidae', 'turdidae', 'fringillidae', 'passeridae',
        }
        
        # Order level keywords (specificity 2)
        order_keywords = {
            # Mammal orders
            'rodent', 'rodentia', 'carnivora', 'carnivore', 'artiodactyla', 
            'lagomorpha', 'chiroptera', 'didelphimorphia',
            # Bird orders
            'passeriformes', 'passerine', 'accipitriformes', 'strigiformes',
            'anseriformes', 'columbiformes', 'piciformes', 'apodiformes',
        }
        
        # Check for family-level match (specificity 3)
        for family in family_keywords:
            if family in species_lower:
                # Add bonus for additional taxonomy depth
                return (category, 3 + species_lower.count('_'))
        
        # Check for order-level match (specificity 2)
        for order in order_keywords:
            if order in species_lower:
                return (category, 2 + max(0, species_lower.count('_') - 1))
        
        # Fallback: count underscores as proxy for taxonomy depth
        underscore_count = species_lower.count('_')
        if underscore_count >= 3:
            # Likely genus_species or more specific
            return (category, 4 + underscore_count)
        elif underscore_count >= 1:
            return (category, 2 + underscore_count)
        
        return (category, 1)
    
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
                
                # Check if tracks overlap in time (both visible at same frame)
                overlaps = (
                    primary.first_seen_frame <= other.last_seen_frame and
                    other.first_seen_frame <= primary.last_seen_frame
                )
                
                if overlaps:
                    # These might be two different animals - don't merge
                    # Update primary to be the one with more detections
                    if len(other.classifications) > len(primary.classifications):
                        primary_track_id = other_id
                        primary = other
                    continue
                
                # Check if gap is small enough
                gap = other.first_seen_frame - primary.last_seen_frame
                if gap <= max_frame_gap:
                    # Merge other into primary
                    LOGGER.info("Merging Track %d into Track %d (same %s, gap=%d frames)",
                               other_id, primary_track_id, species, gap)
                    
                    # Copy all classifications
                    primary.classifications.extend(other.classifications)
                    
                    # Update frame range (must update BOTH first and last)
                    primary.first_seen_frame = min(primary.first_seen_frame, other.first_seen_frame)
                    primary.last_seen_frame = max(primary.last_seen_frame, other.last_seen_frame)
                    
                    # Update best frame if other's is better
                    if other.best_confidence > primary.best_confidence:
                        primary.best_confidence = other.best_confidence
                        primary.best_bbox = other.best_bbox
                        primary.best_frame = other.best_frame
                    
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
                
                # Check for time overlap (would indicate two different animals)
                overlaps = (
                    specific_info.first_seen_frame <= generic_info.last_seen_frame and
                    generic_info.first_seen_frame <= specific_info.last_seen_frame
                )
                
                if overlaps:
                    LOGGER.debug("Skipping merge: Track %d and %d overlap in time",
                                specific['track_id'], generic['track_id'])
                    continue
                
                # Check temporal proximity
                if generic_info.first_seen_frame > specific_info.last_seen_frame:
                    gap = generic_info.first_seen_frame - specific_info.last_seen_frame
                else:
                    gap = specific_info.first_seen_frame - generic_info.last_seen_frame
                
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
                
                # Update best frame if generic's is better
                if generic_info.best_confidence > specific_info.best_confidence:
                    specific_info.best_confidence = generic_info.best_confidence
                    specific_info.best_bbox = generic_info.best_bbox
                    specific_info.best_frame = generic_info.best_frame
                
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
            
            tracks_to_remove.add(other['track_id'])
            merged_count += 1
        
        # Remove merged tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if merged_count > 0:
            LOGGER.info("Non-overlapping merge: combined %d tracks into 1, %d tracks remaining",
                       merged_count, len(self.tracks))
        
        return merged_count
    
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
    
    def merge_spatially_adjacent_tracks(self, iou_threshold: float = 0.3, max_frame_gap: int = 30) -> int:
        """Merge tracks that end and start in similar spatial locations.
        
        This is a simpler, more robust merge strategy:
        - If Track A ends at frame N with bounding box at position P
        - And Track B starts at frame N+gap with bounding box at position Q
        - And P and Q have high IoU (spatial overlap)
        - Then they're probably the same animal
        
        This works regardless of species labels - pure spatial continuity.
        
        Args:
            iou_threshold: Minimum IoU between ending/starting bboxes to merge (0.3 = 30% overlap)
            max_frame_gap: Maximum frame gap to consider for spatial matching
            
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
                if frame_gap < 0:
                    # Tracks overlap in time - skip
                    continue
                if frame_gap > max_frame_gap:
                    # Too far apart temporally
                    continue
                
                # Check spatial overlap (IoU between end of earlier and start of later)
                iou = self._calculate_iou(earlier['last_bbox'], later['first_bbox'])
                
                if iou >= iou_threshold:
                    LOGGER.info(
                        "Spatial merge: Track %d (%s, frames %d-%d) + Track %d (%s, frames %d-%d), "
                        "IoU=%.2f, gap=%d frames",
                        earlier['track_id'], earlier['species'], 
                        earlier['first_frame'], earlier['last_frame'],
                        later['track_id'], later['species'],
                        later['first_frame'], later['last_frame'],
                        iou, frame_gap
                    )
                    
                    later_info = later['info']
                    
                    # Merge later into earlier
                    earlier_info.classifications.extend(later_info.classifications)
                    
                    # Update frame range (must update BOTH first and last)
                    earlier_info.first_seen_frame = min(earlier_info.first_seen_frame,
                                                        later_info.first_seen_frame)
                    earlier_info.last_seen_frame = max(earlier_info.last_seen_frame, 
                                                       later_info.last_seen_frame)
                    
                    # Update best frame if later's is better
                    if later_info.best_confidence > earlier_info.best_confidence:
                        earlier_info.best_confidence = later_info.best_confidence
                        earlier_info.best_bbox = later_info.best_bbox
                        earlier_info.best_frame = later_info.best_frame
                    
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
        LOGGER.error("Failed to create tracker: %s", e)
        return None
