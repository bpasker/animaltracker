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
        
        Uses voting weighted by confidence and frequency.
        
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
        
        # Filter out generic terms if we have specific ones
        generic_terms = {'animal', 'bird', 'mammal', 'aves'}
        specific_species = [s for s in species_data if s.lower() not in generic_terms]
        
        if specific_species:
            candidates = {s: species_data[s] for s in specific_species}
        else:
            candidates = species_data
        
        if not candidates:
            return "", 0.0, None
        
        # Score: specificity first, then count, then max confidence
        best_species = max(
            candidates.keys(),
            key=lambda s: (
                candidates[s]['specificity'],
                candidates[s]['count'],
                candidates[s]['max_confidence']
            )
        )
        
        return (
            best_species, 
            candidates[best_species]['max_confidence'],
            candidates[best_species]['taxonomy']
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
        """Calculate how specific a species name is."""
        generic_terms = {'animal', 'bird', 'mammal', 'aves'}
        words = species.lower().replace('-', '_').split('_')
        specificity = len([w for w in words if w and w not in generic_terms])
        
        # Bonus for binomial names (genus_species)
        if '_' in species and len(species.split('_')) >= 2:
            specificity += 2
        
        return specificity


class ObjectTracker:
    """Tracks objects across frames and accumulates species classifications."""
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 15,
    ):
        """Initialize the object tracker.
        
        Args:
            track_activation_threshold: Min confidence to start a track
            lost_track_buffer: Frames to keep lost tracks
            minimum_matching_threshold: IoU threshold for matching
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
    
    @property
    def active_track_count(self) -> int:
        """Number of tracks being tracked."""
        return len(self.tracks)


def create_tracker(
    enabled: bool = True,
    frame_rate: int = 15,
) -> Optional[ObjectTracker]:
    """Create an object tracker if available and enabled.
    
    Args:
        enabled: Whether tracking is enabled
        frame_rate: Expected frame rate
        
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
        tracker = ObjectTracker(frame_rate=frame_rate)
        LOGGER.info("Object tracking enabled (ByteTrack)")
        return tracker
    except Exception as e:
        LOGGER.error("Failed to create tracker: %s", e)
        return None
