"""Post-processing module for improving clip classifications.

This module provides functionality to reanalyze saved video clips using
the detector to get more accurate species classifications and better
detection thumbnails.

This is the UNIFIED classification engine - both auto-processing after
clip save and manual reanalysis use this same module.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from .detector import BaseDetector, Detection, create_detector
from .tracker import ObjectTracker, create_tracker

LOGGER = logging.getLogger(__name__)

# Default configuration values
DEFAULT_SAMPLE_RATE = 3  # Analyze every Nth frame (lower = more accurate tracking)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for specific species
DEFAULT_GENERIC_CONFIDENCE = 0.5  # Minimum confidence for generic categories
MAX_KEY_FRAMES_PER_SPECIES = 3


@dataclass
class ProcessingSettings:
    """Unified settings for video classification processing.
    
    These settings control how the classification engine analyzes video clips.
    The same settings structure is used for both auto-processing and manual reanalysis.
    """
    # Detection settings
    sample_rate: int = DEFAULT_SAMPLE_RATE  # Analyze every Nth frame
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD  # Min confidence for species
    generic_confidence: float = DEFAULT_GENERIC_CONFIDENCE  # Min confidence for "animal", "bird"
    
    # Tracking settings
    tracking_enabled: bool = True
    lost_track_buffer: int = 120  # Frames to keep lost track alive
    
    # Merge settings - for consolidating fragmented tracks
    merge_enabled: bool = True
    same_species_merge_gap: int = 120  # Max frame gap for same-species merge
    spatial_merge_enabled: bool = True  # Merge tracks in similar locations (best!)
    spatial_merge_iou: float = 0.3  # Min IoU overlap to consider same object
    spatial_merge_gap: int = 30  # Max frame gap for spatial matching
    hierarchical_merge_enabled: bool = True  # Merge generic→specific (animal→canidae)
    hierarchical_merge_gap: int = 120  # Max frame gap for hierarchical merge
    min_specific_detections: int = 2  # Min detections for specific track to absorb generic
    single_animal_mode: bool = False  # Aggressive merge: assume only 1 animal in video
    
    # Output settings
    max_thumbnails: int = MAX_KEY_FRAMES_PER_SPECIES
    save_processing_log: bool = True
    
    def to_dict(self) -> Dict:
        """Convert settings to dictionary for JSON serialization."""
        return {
            "sample_rate": self.sample_rate,
            "confidence_threshold": self.confidence_threshold,
            "generic_confidence": self.generic_confidence,
            "tracking_enabled": self.tracking_enabled,
            "lost_track_buffer": self.lost_track_buffer,
            "merge_enabled": self.merge_enabled,
            "same_species_merge_gap": self.same_species_merge_gap,
            "spatial_merge_enabled": self.spatial_merge_enabled,
            "spatial_merge_iou": self.spatial_merge_iou,
            "spatial_merge_gap": self.spatial_merge_gap,
            "hierarchical_merge_enabled": self.hierarchical_merge_enabled,
            "hierarchical_merge_gap": self.hierarchical_merge_gap,
            "min_specific_detections": self.min_specific_detections,
            "single_animal_mode": self.single_animal_mode,
            "max_thumbnails": self.max_thumbnails,
            "save_processing_log": self.save_processing_log,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProcessingSettings":
        """Create settings from dictionary, using defaults for missing keys."""
        return cls(
            sample_rate=data.get("sample_rate", DEFAULT_SAMPLE_RATE),
            confidence_threshold=data.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            generic_confidence=data.get("generic_confidence", DEFAULT_GENERIC_CONFIDENCE),
            tracking_enabled=data.get("tracking_enabled", True),
            lost_track_buffer=data.get("lost_track_buffer", 120),
            merge_enabled=data.get("merge_enabled", True),
            same_species_merge_gap=data.get("same_species_merge_gap", 120),
            spatial_merge_enabled=data.get("spatial_merge_enabled", True),
            spatial_merge_iou=data.get("spatial_merge_iou", 0.3),
            spatial_merge_gap=data.get("spatial_merge_gap", 30),
            hierarchical_merge_enabled=data.get("hierarchical_merge_enabled", True),
            hierarchical_merge_gap=data.get("hierarchical_merge_gap", 120),
            min_specific_detections=data.get("min_specific_detections", 2),
            single_animal_mode=data.get("single_animal_mode", False),
            max_thumbnails=data.get("max_thumbnails", MAX_KEY_FRAMES_PER_SPECIES),
            save_processing_log=data.get("save_processing_log", True),
        )


@dataclass
class SpeciesResult:
    """Result of species detection from post-processing."""
    species: str
    confidence: float
    count: int  # Number of times detected
    specificity: int  # Higher = more specific identification
    taxonomy: Optional[str] = None
    # Top detection frames for this species: (frame, confidence, bbox)
    key_frames: List[Tuple] = field(default_factory=list)


@dataclass
class ProcessingLogEntry:
    """A single log entry from processing."""
    frame_idx: int
    event: str  # "detection", "filtered", "tracked", "selected", etc.
    species: str
    confidence: float
    reason: Optional[str] = None  # Why filtered, selected, etc.
    track_id: Optional[int] = None
    bbox: Optional[List[float]] = None


@dataclass 
class PostProcessResult:
    """Result of post-processing a video clip."""
    original_path: Path
    new_path: Optional[Path]  # None if unchanged
    original_species: str
    new_species: str
    confidence: float
    species_results: Dict[str, SpeciesResult]
    thumbnails_saved: List[Path]
    frames_analyzed: int
    total_frames: int
    raw_detections: int = 0  # Total detections before filtering
    filtered_detections: int = 0  # Detections filtered out
    processing_log: List[ProcessingLogEntry] = field(default_factory=list)
    tracking_summary: Optional[Dict] = None  # Track consolidation info
    settings_used: Optional[ProcessingSettings] = None  # Settings that produced this result
    tracks_detected: int = 0  # Number of unique animals/tracks
    success: bool = True
    error: Optional[str] = None


class ClipPostProcessor:
    """Unified post-processor for analyzing and classifying video clips.
    
    This is the SINGLE engine used for both:
    - Auto-processing after a clip is saved from real-time detection
    - Manual reanalysis via the web UI "Reanalyze" button
    
    Using the same engine ensures consistent results regardless of how
    processing is triggered.
    """
    
    def __init__(
        self,
        detector: BaseDetector,
        storage_root: Path,
        settings: Optional[ProcessingSettings] = None,
        # Legacy parameters for backward compatibility
        sample_rate: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        generic_confidence: Optional[float] = None,
        tracking_enabled: Optional[bool] = None,
    ):
        """Initialize the post-processor.
        
        Args:
            detector: Detection backend to use for analysis
            storage_root: Root directory for clip storage
            settings: ProcessingSettings instance (preferred)
            
            Legacy parameters (deprecated, use settings instead):
            sample_rate: Analyze every Nth frame
            confidence_threshold: Minimum confidence for specific species
            generic_confidence: Minimum confidence for generic categories
            tracking_enabled: Use object tracking
        """
        self.detector = detector
        self.storage_root = storage_root
        
        # Use provided settings or create from legacy parameters
        if settings:
            self.settings = settings
        else:
            # Build settings from legacy parameters with defaults
            self.settings = ProcessingSettings(
                sample_rate=sample_rate if sample_rate is not None else DEFAULT_SAMPLE_RATE,
                confidence_threshold=confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD,
                generic_confidence=generic_confidence if generic_confidence is not None else DEFAULT_GENERIC_CONFIDENCE,
                tracking_enabled=tracking_enabled if tracking_enabled is not None else True,
            )
        
        # Expose settings as instance attributes for backward compatibility
        self.sample_rate = self.settings.sample_rate
        self.tracking_enabled = self.settings.tracking_enabled
        self.confidence_threshold = self.settings.confidence_threshold
        self.generic_confidence = self.settings.generic_confidence
        
        # Terms to filter out
        self.invalid_terms = {
            'unknown', 'blank', 'empty', 'no cv result', 'no_cv_result', 
            'vehicle', 'human', 'person'
        }
        self.generic_terms = {'animal', 'bird', 'mammal', 'aves', 'rodent'}
    
    def process_clip(
        self, 
        clip_path: Path,
        update_filename: bool = True,
        regenerate_thumbnails: bool = True,
    ) -> PostProcessResult:
        """Process a single video clip to improve its classification.
        
        Args:
            clip_path: Path to the video file
            update_filename: Whether to rename the file if species changes
            regenerate_thumbnails: Whether to create new detection thumbnails
            
        Returns:
            PostProcessResult with details of the processing
        """
        LOGGER.info("Post-processing clip: %s", clip_path)
        LOGGER.info("Settings: confidence_threshold=%.2f, generic_confidence=%.2f, tracking=%s, sample_rate=%d",
                   self.confidence_threshold, self.generic_confidence, self.tracking_enabled, self.sample_rate)
        
        if not clip_path.exists():
            return PostProcessResult(
                original_path=clip_path,
                new_path=None,
                original_species="",
                new_species="",
                confidence=0.0,
                species_results={},
                thumbnails_saved=[],
                frames_analyzed=0,
                total_frames=0,
                success=False,
                error=f"File not found: {clip_path}"
            )
        
        # Parse original species from filename
        original_species = self._parse_species_from_filename(clip_path.name)
        
        # Open video and analyze frames
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return PostProcessResult(
                original_path=clip_path,
                new_path=None,
                original_species=original_species,
                new_species=original_species,
                confidence=0.0,
                species_results={},
                thumbnails_saved=[],
                frames_analyzed=0,
                total_frames=0,
                success=False,
                error=f"Could not open video: {clip_path}"
            )
        
        processing_log: List[ProcessingLogEntry] = []
        tracking_summary: Optional[Dict] = None
        video_metadata: Dict = {}
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            species_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata = \
                self._analyze_video(cap, total_frames)
            frames_analyzed = (total_frames + self.sample_rate - 1) // self.sample_rate
            
        finally:
            cap.release()
        
        LOGGER.info(
            "Analysis: %d raw detections, %d filtered out, %d valid species",
            raw_detection_count, filtered_count, len(species_results)
        )
        
        # Determine best species classification
        new_species, confidence = self._select_best_species(species_results)
        
        if not new_species:
            new_species = original_species
            confidence = 0.0
        
        # Update filename if species changed
        new_path = None
        if update_filename and new_species != original_species and new_species:
            new_path = self._rename_clip(clip_path, new_species)
            working_path = new_path if new_path else clip_path
        else:
            working_path = clip_path
        
        # Regenerate thumbnails with best detection frames
        thumbnails_saved = []
        if regenerate_thumbnails:
            if species_results:
                thumbnails_saved = self._save_thumbnails(working_path, species_results)
            else:
                # No valid detections - extract sample frames from video as fallback
                LOGGER.info("No valid detections found, extracting sample frames as fallback")
                thumbnails_saved = self._extract_sample_frames(working_path, num_samples=3)
        
        # Save processing log as JSON alongside the clip
        self._save_processing_log(working_path, processing_log, tracking_summary, video_metadata)
        
        # Count unique tracks (animals) detected
        tracks_detected = tracking_summary.get("total_tracks", 0) if tracking_summary else len(species_results)
        
        LOGGER.info(
            "Post-processing complete: %s -> %s (%.1f%% confidence, %d tracks, %d species found)",
            original_species, new_species, confidence * 100, tracks_detected, len(species_results)
        )
        
        return PostProcessResult(
            original_path=clip_path,
            new_path=new_path,
            original_species=original_species,
            new_species=new_species,
            confidence=confidence,
            species_results=species_results,
            thumbnails_saved=thumbnails_saved,
            frames_analyzed=frames_analyzed,
            total_frames=total_frames,
            raw_detections=raw_detection_count,
            filtered_detections=filtered_count,
            processing_log=processing_log,
            tracking_summary=tracking_summary,
            settings_used=self.settings,
            tracks_detected=tracks_detected,
            success=True,
        )
    
    def _save_processing_log(
        self,
        clip_path: Path,
        processing_log: List[ProcessingLogEntry],
        tracking_summary: Optional[Dict],
        video_metadata: Optional[Dict] = None,
    ) -> None:
        """Save processing log as JSON file alongside clip."""
        import json
        from dataclasses import asdict
        
        log_path = clip_path.with_suffix('.log.json')
        
        try:
            log_data = {
                "clip": str(clip_path.name),
                "timestamp": str(Path(clip_path.stem).name.split('_')[0]) if '_' in clip_path.stem else "",
                "settings": self.settings.to_dict(),  # Full settings object
                "detector_type": type(self.detector).__name__,
                "video": video_metadata or {},
                "tracking_summary": tracking_summary,
                "log_entries": [asdict(entry) for entry in processing_log],
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            LOGGER.info("Saved processing log: %s", log_path)
        except Exception as e:
            LOGGER.warning("Failed to save processing log: %s", e)

    def _analyze_video(
        self, 
        cap: cv2.VideoCapture, 
        total_frames: int
    ) -> Tuple[Dict[str, SpeciesResult], int, int, List[ProcessingLogEntry], Optional[Dict], Dict]:
        """Analyze video frames and collect species detections.
        
        Uses object tracking (if enabled) to consolidate classifications for 
        the same animal across frames, producing one species per tracked object.
        
        Returns:
            (species_results dict, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata)
        """
        # Get video FPS and calculate sample rate
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        
        # When tracking is enabled, sample more frequently for better track continuity
        # Without tracking, we can sample less frequently since we just need species votes
        if self.tracking_enabled:
            # For tracking: analyze every 3rd frame (~5-10 fps effective)
            # This gives ByteTrack enough visual overlap to track moving animals
            actual_sample_rate = 3
        else:
            # For non-tracking: ~1 frame per second is fine
            smart_sample_rate = max(1, min(int(fps), 30))
            actual_sample_rate = self.sample_rate if self.sample_rate > 1 else smart_sample_rate
        
        effective_fps = fps / actual_sample_rate
        
        # Build video metadata for logging
        video_metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "actual_sample_rate": actual_sample_rate,
            "effective_fps": effective_fps,
            "duration_seconds": total_frames / fps if fps > 0 else 0,
            "frames_to_analyze": (total_frames + actual_sample_rate - 1) // actual_sample_rate,
        }
        
        LOGGER.info("Video fps=%.1f, sample_rate=%d (effective %.1f fps, tracking=%s)", 
                   fps, actual_sample_rate, effective_fps, self.tracking_enabled)
        
        # Create tracker if enabled, using settings for buffer size
        tracker = create_tracker(
            enabled=self.tracking_enabled, 
            frame_rate=effective_fps,
            lost_track_buffer=self.settings.lost_track_buffer
        )
        if tracker:
            LOGGER.info("Object tracking enabled for post-processing (lost_buffer=%d)", 
                       self.settings.lost_track_buffer)
        
        species_results: Dict[str, SpeciesResult] = {}
        frame_idx = 0
        raw_detection_count = 0
        filtered_count = 0
        all_frames_data: List[Tuple[int, any, List[Detection]]] = []  # For tracking
        processing_log: List[ProcessingLogEntry] = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame (using smart sample rate)
            if frame_idx % actual_sample_rate == 0:
                try:
                    # Request filtered detections to log them
                    infer_result = self.detector.infer(
                        frame, 
                        conf_threshold=self.confidence_threshold,
                        generic_confidence=self.generic_confidence,
                        return_filtered=True,
                    )
                    
                    # Handle both return formats (with/without filtered)
                    if isinstance(infer_result, tuple):
                        detections, detector_filtered = infer_result
                    else:
                        detections = infer_result
                        detector_filtered = []
                    
                    raw_detection_count += len(detections) + len(detector_filtered)
                    
                    # Log detector-level filtering (exotic species, etc.)
                    for det, reason in detector_filtered:
                        processing_log.append(ProcessingLogEntry(
                            frame_idx=frame_idx,
                            event="detector_filtered",
                            species=det.species,
                            confidence=det.confidence,
                            reason=reason,
                            bbox=det.bbox,
                        ))
                        filtered_count += 1
                    
                    if tracker and detections:
                        # Update tracker with detections
                        tracked = tracker.update(detections, frame)
                        all_frames_data.append((frame_idx, frame.copy(), detections))
                        
                        # Log tracking assignments
                        for track_id, det in tracked.items():
                            processing_log.append(ProcessingLogEntry(
                                frame_idx=frame_idx,
                                event="tracked",
                                species=det.species,
                                confidence=det.confidence,
                                track_id=track_id,
                                reason=f"Assigned to track {track_id}",
                            ))
                    
                    # Still process detections for non-tracked fallback
                    valid, filtered, filter_log = self._process_detections_with_log(
                        detections, frame, species_results, frame_idx
                    )
                    filtered_count += filtered
                    processing_log.extend(filter_log)
                    
                except Exception as e:
                    LOGGER.warning("Detection failed on frame %d: %s", frame_idx, e)
                    processing_log.append(ProcessingLogEntry(
                        frame_idx=frame_idx,
                        event="error",
                        species="",
                        confidence=0.0,
                        reason=str(e),
                    ))
            
            frame_idx += 1
        
        # Build tracking summary
        tracking_summary = None
        
        # If tracking was used and we have tracked objects, build results from tracks
        if tracker and tracker.active_track_count > 0:
            # FIRST pass: Spatial merge - most reliable!
            # If an object appears in the same location across frames, it's the same object
            # regardless of species classification changes
            if self.settings.spatial_merge_enabled:
                spatial_merged = tracker.merge_spatially_adjacent_tracks(
                    iou_threshold=self.settings.spatial_merge_iou,
                    max_frame_gap=self.settings.spatial_merge_gap
                )
                if spatial_merged > 0:
                    processing_log.append(ProcessingLogEntry(
                        frame_idx=-1,
                        event="spatial_merge",
                        species="",
                        confidence=0.0,
                        reason=f"Merged {spatial_merged} tracks based on spatial continuity (IoU≥{self.settings.spatial_merge_iou}, gap≤{self.settings.spatial_merge_gap})",
                    ))
            
            # Second pass: Merge fragmented tracks with the SAME species
            # This handles cases where ByteTrack loses a track due to movement
            # but later detections are clearly the same species
            if self.settings.merge_enabled:
                merged_count = tracker.merge_similar_tracks(
                    max_frame_gap=self.settings.same_species_merge_gap
                )
                if merged_count > 0:
                    processing_log.append(ProcessingLogEntry(
                        frame_idx=-1,
                        event="tracks_merged",
                        species="",
                        confidence=0.0,
                        reason=f"Merged {merged_count} fragmented tracks with same species (gap≤{self.settings.same_species_merge_gap})",
                    ))
            
                # Third pass: Merge GENERIC tracks into more SPECIFIC tracks
                # E.g., "animal" track absorbed into "canidae" track if temporally adjacent
                # This only merges hierarchically compatible species (animal->mammal->canidae)
                if self.settings.hierarchical_merge_enabled:
                    hierarchical_merged = tracker.merge_hierarchical_tracks(
                        max_frame_gap=self.settings.hierarchical_merge_gap,
                        min_specific_detections=self.settings.min_specific_detections
                    )
                    if hierarchical_merged > 0:
                        processing_log.append(ProcessingLogEntry(
                            frame_idx=-1,
                            event="hierarchical_merge",
                            species="",
                            confidence=0.0,
                            reason=f"Absorbed {hierarchical_merged} generic tracks into specific species tracks (gap≤{self.settings.hierarchical_merge_gap})",
                        ))
                
                # Third pass: Single animal mode - aggressively merge ALL non-overlapping tracks
                # Use this when you're sure there's only one animal in the video
                if self.settings.single_animal_mode:
                    non_overlap_merged = tracker.merge_non_overlapping_tracks()
                    if non_overlap_merged > 0:
                        processing_log.append(ProcessingLogEntry(
                            frame_idx=-1,
                            event="single_animal_merge",
                            species="",
                            confidence=0.0,
                            reason=f"Single-animal mode: merged {non_overlap_merged} non-overlapping tracks into 1",
                        ))
            
            tracked_results, track_log, tracking_summary = self._build_tracked_species_results_with_log(
                tracker, all_frames_data
            )
            processing_log.extend(track_log)
            
            if tracked_results:
                LOGGER.info("Tracking consolidated %d detections into %d tracked objects",
                           raw_detection_count, len(tracked_results))
                return tracked_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata
        
        # Fallback to non-tracked results
        return species_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata
    
    def _build_tracked_species_results(
        self,
        tracker: ObjectTracker,
        frames_data: List[Tuple[int, any, List[Detection]]],
    ) -> Dict[str, SpeciesResult]:
        """Build species results from tracked objects.
        
        Each tracked object votes for its best species based on all classifications
        it received across frames. This consolidates Dog/Mammal/Cat into one species.
        """
        species_results: Dict[str, SpeciesResult] = {}
        
        # Get unique species from all tracked objects (one per track)
        tracked_species = tracker.get_unique_species()
        
        for species, confidence in tracked_species:
            # Filter invalid species
            species_lower = species.lower()
            if species_lower in self.invalid_terms:
                continue
            if 'no cv result' in species_lower:
                continue
            if len(species) > 30 and species.count('-') >= 3:
                continue
            
            specificity = self._calculate_specificity(species)
            
            if species not in species_results:
                species_results[species] = SpeciesResult(
                    species=species,
                    confidence=confidence,
                    count=1,
                    specificity=specificity,
                    taxonomy=None,
                    key_frames=[],
                )
            else:
                # Multiple tracks with same species - aggregate
                species_results[species].count += 1
                species_results[species].confidence = max(
                    species_results[species].confidence, confidence
                )
        
        # Get key frames for each tracked species
        for track_id, track_info in tracker.tracks.items():
            best_species_data = track_info.get_best_species()
            if not best_species_data or not best_species_data[0]:
                continue
            
            best_species = best_species_data[0]  # (species, confidence, taxonomy)
            if best_species not in species_results:
                continue
            
            result = species_results[best_species]
            # Get the best frame for this track
            best_frame_data = track_info.get_best_frame()
            if best_frame_data:
                frame, conf, bbox = best_frame_data
                self._update_key_frames(result, frame, conf, bbox)
        
        return species_results
    
    def _build_tracked_species_results_with_log(
        self,
        tracker: ObjectTracker,
        frames_data: List[Tuple[int, any, List[Detection]]],
    ) -> Tuple[Dict[str, SpeciesResult], List[ProcessingLogEntry], Dict]:
        """Build species results from tracked objects with detailed logging."""
        species_results: Dict[str, SpeciesResult] = {}
        log_entries: List[ProcessingLogEntry] = []
        
        # Build tracking summary
        tracking_summary = {
            "total_tracks": len(tracker.tracks),
            "tracks": [],
        }
        
        # Get unique species from all tracked objects (one per track)
        tracked_species = tracker.get_unique_species()
        
        for species, confidence in tracked_species:
            # Filter invalid species
            species_lower = species.lower()
            if species_lower in self.invalid_terms:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=-1, event="track_filtered", species=species,
                    confidence=confidence, reason="Invalid term"
                ))
                continue
            if 'no cv result' in species_lower:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=-1, event="track_filtered", species=species,
                    confidence=confidence, reason="No CV result"
                ))
                continue
            if len(species) > 30 and species.count('-') >= 3:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=-1, event="track_filtered", species=species,
                    confidence=confidence, reason="UUID-like identifier"
                ))
                continue
            
            specificity = self._calculate_specificity(species)
            
            if species not in species_results:
                species_results[species] = SpeciesResult(
                    species=species,
                    confidence=confidence,
                    count=1,
                    specificity=specificity,
                    taxonomy=None,
                    key_frames=[],
                )
            else:
                species_results[species].count += 1
                species_results[species].confidence = max(
                    species_results[species].confidence, confidence
                )
        
        # Get key frames and build track details
        for track_id, track_info in tracker.tracks.items():
            best_species_data = track_info.get_best_species()
            if not best_species_data or not best_species_data[0]:
                continue
            
            best_species, best_conf, taxonomy = best_species_data
            
            # Collect all classifications for this track
            all_classifications = {}
            for c in track_info.classifications:
                if c.species not in all_classifications:
                    all_classifications[c.species] = {"count": 0, "max_conf": 0}
                all_classifications[c.species]["count"] += 1
                all_classifications[c.species]["max_conf"] = max(
                    all_classifications[c.species]["max_conf"], c.confidence
                )
            
            track_detail = {
                "track_id": track_id,
                "best_species": best_species,
                "best_confidence": round(best_conf, 3),
                "first_frame": track_info.first_seen_frame,
                "last_frame": track_info.last_seen_frame,
                "frames_seen": track_info.last_seen_frame - track_info.first_seen_frame + 1,
                "classification_count": len(track_info.classifications),
                "all_classifications": all_classifications,
            }
            tracking_summary["tracks"].append(track_detail)
            
            log_entries.append(ProcessingLogEntry(
                frame_idx=-1,
                event="track_consolidated",
                species=best_species,
                confidence=best_conf,
                track_id=track_id,
                reason=f"Selected from {len(all_classifications)} candidates: {list(all_classifications.keys())}"
            ))
            
            if best_species in species_results:
                result = species_results[best_species]
                # Use get_best_frame_for_species to get a frame that matches the selected species
                # This is important after track merging where the overall best frame might be
                # from a different (more generic) classification like "animal"
                best_frame_data = track_info.get_best_frame_for_species(best_species)
                if not best_frame_data:
                    # Fall back to overall best frame if no species-specific frame
                    best_frame_data = track_info.get_best_frame()
                if best_frame_data:
                    frame, conf, bbox = best_frame_data
                    self._update_key_frames(result, frame, conf, bbox)
        
        return species_results, log_entries, tracking_summary
    
    def _process_detections_with_log(
        self,
        detections: List[Detection],
        frame,
        species_results: Dict[str, SpeciesResult],
        frame_idx: int,
    ) -> Tuple[int, int, List[ProcessingLogEntry]]:
        """Process detections from a single frame with logging.
        
        Returns:
            (valid_count, filtered_count, log_entries)
        """
        valid_count = 0
        filtered_count = 0
        log_entries: List[ProcessingLogEntry] = []
        
        for det in detections:
            species = det.species
            species_lower = species.lower()
            
            # Filter out invalid detections
            if species_lower in self.invalid_terms:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=frame_idx, event="filtered", species=species,
                    confidence=det.confidence, reason="Invalid term",
                    bbox=det.bbox
                ))
                filtered_count += 1
                continue
            if 'no cv result' in species_lower:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=frame_idx, event="filtered", species=species,
                    confidence=det.confidence, reason="No CV result",
                    bbox=det.bbox
                ))
                filtered_count += 1
                continue
            # Skip UUID-like strings
            if len(species) > 30 and species.count('-') >= 3:
                log_entries.append(ProcessingLogEntry(
                    frame_idx=frame_idx, event="filtered", species=species,
                    confidence=det.confidence, reason="UUID-like identifier",
                    bbox=det.bbox
                ))
                filtered_count += 1
                continue
            
            valid_count += 1
            log_entries.append(ProcessingLogEntry(
                frame_idx=frame_idx, event="accepted", species=species,
                confidence=det.confidence, bbox=det.bbox
            ))
            
            # Calculate specificity score
            specificity = self._calculate_specificity(species)
            
            if species not in species_results:
                species_results[species] = SpeciesResult(
                    species=species,
                    confidence=det.confidence,
                    count=1,
                    specificity=specificity,
                    taxonomy=det.taxonomy,
                    key_frames=[],
                )
            else:
                result = species_results[species]
                result.count += 1
                result.confidence = max(result.confidence, det.confidence)
            
            # Track key frames for this species
            result = species_results[species]
            self._update_key_frames(result, frame, det.confidence, det.bbox)
        
        return valid_count, filtered_count, log_entries

    def _process_detections(
        self,
        detections: List[Detection],
        frame,
        species_results: Dict[str, SpeciesResult],
    ) -> Tuple[int, int]:
        """Process detections from a single frame and update results.
        
        Returns:
            (valid_count, filtered_count)
        """
        valid_count = 0
        filtered_count = 0
        
        for det in detections:
            species = det.species
            species_lower = species.lower()
            
            # Filter out invalid detections
            if species_lower in self.invalid_terms:
                LOGGER.debug("Filtered out invalid term: %s", species)
                filtered_count += 1
                continue
            if 'no cv result' in species_lower:
                LOGGER.debug("Filtered out 'no cv result': %s", species)
                filtered_count += 1
                continue
            # Skip UUID-like strings
            if len(species) > 30 and species.count('-') >= 3:
                LOGGER.debug("Filtered out UUID-like: %s", species)
                filtered_count += 1
                continue
            
            valid_count += 1
            
            # Calculate specificity score
            specificity = self._calculate_specificity(species)
            
            if species not in species_results:
                species_results[species] = SpeciesResult(
                    species=species,
                    confidence=det.confidence,
                    count=1,
                    specificity=specificity,
                    taxonomy=det.taxonomy,
                    key_frames=[],
                )
            else:
                result = species_results[species]
                result.count += 1
                result.confidence = max(result.confidence, det.confidence)
            
            # Track key frames for this species
            result = species_results[species]
            self._update_key_frames(result, frame, det.confidence, det.bbox)
        
        return valid_count, filtered_count
    
    def _update_key_frames(
        self,
        result: SpeciesResult,
        frame,
        confidence: float,
        bbox: List[float],
    ) -> None:
        """Update key frames for a species, keeping top N by confidence."""
        frames = result.key_frames
        
        # Check if this detection is better than what we have
        min_conf = min((f[1] for f in frames), default=0.0)
        
        if len(frames) < MAX_KEY_FRAMES_PER_SPECIES or confidence > min_conf:
            frames.append((frame.copy(), confidence, bbox))
            # Sort by confidence descending and keep top N
            frames.sort(key=lambda x: x[1], reverse=True)
            result.key_frames = frames[:MAX_KEY_FRAMES_PER_SPECIES]
    
    def _calculate_specificity(self, species: str) -> int:
        """Calculate specificity score for a species name.
        
        Hierarchy (higher = more specific):
        - 0: "animal" (most generic)
        - 1: "bird", "mammal" (class-level)
        - 2: "hawk", "sparrow", "deer" (family/order level)
        - 3+: "red-tailed_hawk", "white-tailed_deer" (species level)
        
        This ensures "bird" beats "animal", and specific species beat generic labels.
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
                       'crow', 'jay', 'dove', 'pigeon', 'hummingbird', 'cardinal',
                       'chickadee', 'nuthatch', 'titmouse', 'wren', 'robin', 'bluebird',
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
    
    def _select_best_species(
        self, 
        species_results: Dict[str, SpeciesResult]
    ) -> Tuple[str, float]:
        """Select the best species classification from results.
        
        Prioritizes:
        1. Specificity (more specific names are better - bird > animal)
        2. Detection count (more detections = more reliable)
        3. Confidence (higher confidence = more certain)
        
        Returns (species_name, confidence).
        """
        if not species_results:
            return "", 0.0
        
        # Find the maximum specificity among all candidates
        max_specificity = max(r.specificity for r in species_results.values())
        
        # Only consider candidates at the highest specificity level
        best_candidates = {
            name: result for name, result in species_results.items()
            if result.specificity == max_specificity
        }
        
        if not best_candidates:
            return "", 0.0
        
        # Among equally-specific candidates, score by count then confidence
        def score_species(name: str) -> Tuple[int, float]:
            result = best_candidates[name]
            return (result.count, result.confidence)
        
        best_species = max(best_candidates.keys(), key=score_species)
        best_result = best_candidates[best_species]
        
        LOGGER.debug("Selected '%s' (specificity=%d, count=%d, conf=%.2f) from %d candidates",
                    best_species, best_result.specificity, best_result.count, 
                    best_result.confidence, len(species_results))
        
        return best_species, best_result.confidence
    
    def _parse_species_from_filename(self, filename: str) -> str:
        """Extract species from clip filename."""
        # Remove extension
        name = filename.rsplit('.', 1)[0]
        
        # Format: timestamp_species.mp4
        parts = name.split('_', 1)
        if len(parts) < 2:
            return 'Unknown'
        
        return parts[1]
    
    def _rename_clip(self, clip_path: Path, new_species: str) -> Optional[Path]:
        """Rename clip file with new species classification.
        
        Returns new path if renamed, None if failed.
        """
        try:
            # Parse timestamp from original filename
            old_name = clip_path.stem
            parts = old_name.split('_', 1)
            if not parts:
                return None
            
            timestamp = parts[0]
            
            # Clean species name for filename
            clean_species = new_species.replace(' ', '_').replace('/', '_')
            
            new_name = f"{timestamp}_{clean_species}{clip_path.suffix}"
            new_path = clip_path.parent / new_name
            
            # Don't overwrite existing files
            if new_path.exists() and new_path != clip_path:
                LOGGER.warning("Cannot rename: %s already exists", new_path)
                return None
            
            # Rename the clip
            clip_path.rename(new_path)
            LOGGER.info("Renamed clip: %s -> %s", clip_path.name, new_name)
            
            # Also rename any existing thumbnails
            self._rename_thumbnails(clip_path, new_path)
            
            return new_path
            
        except Exception as e:
            LOGGER.error("Failed to rename clip %s: %s", clip_path, e)
            return None
    
    def _rename_thumbnails(self, old_clip_path: Path, new_clip_path: Path) -> None:
        """Rename existing thumbnails to match new clip name."""
        old_stem = old_clip_path.stem
        new_stem = new_clip_path.stem
        clip_dir = old_clip_path.parent
        
        for thumb in clip_dir.glob(f"{old_stem}_thumb_*.jpg"):
            # Extract the thumbnail suffix (species info)
            suffix = thumb.name[len(old_stem):]
            new_thumb_name = f"{new_stem}{suffix}"
            new_thumb_path = clip_dir / new_thumb_name
            
            try:
                thumb.rename(new_thumb_path)
            except Exception as e:
                LOGGER.warning("Failed to rename thumbnail %s: %s", thumb, e)
    
    def _save_thumbnails(
        self,
        clip_path: Path,
        species_results: Dict[str, SpeciesResult],
    ) -> List[Path]:
        """Save detection thumbnails for all species.
        
        Returns list of saved thumbnail paths.
        """
        saved = []
        clip_stem = clip_path.stem
        
        LOGGER.info("Saving thumbnails for clip: %s (stem=%s, parent=%s)", 
                   clip_path, clip_stem, clip_path.parent)
        
        # First, remove old thumbnails for this clip
        for old_thumb in clip_path.parent.glob(f"{clip_stem}_thumb_*.jpg"):
            try:
                old_thumb.unlink()
                LOGGER.debug("Removed old thumbnail: %s", old_thumb)
            except Exception as e:
                LOGGER.warning("Failed to remove old thumbnail %s: %s", old_thumb, e)
        
        # Save new thumbnails
        for species, result in species_results.items():
            if not result.key_frames:
                continue
            
            # Clean species name for filename
            clean_species = species.replace(' ', '_').replace('/', '_').lower()
            
            for idx, (frame, confidence, bbox) in enumerate(result.key_frames):
                # Build thumbnail path
                if idx == 0:
                    thumb_name = f"{clip_stem}_thumb_{clean_species}.jpg"
                else:
                    thumb_name = f"{clip_stem}_thumb_{clean_species}_{idx}.jpg"
                
                thumb_path = clip_path.parent / thumb_name
                LOGGER.debug("Attempting to save thumbnail: %s", thumb_path)
                
                try:
                    # Draw bounding box on frame
                    annotated = self._annotate_frame(
                        frame, species, confidence, bbox, 
                        idx + 1 if len(result.key_frames) > 1 else None
                    )
                    
                    # Save thumbnail
                    cv2.imwrite(str(thumb_path), annotated)
                    
                    # Force sync to disk (important for NFS mounts)
                    if thumb_path.exists():
                        import os
                        # Open file and fsync to ensure it's written to NFS server
                        with open(thumb_path, 'r+b') as f:
                            os.fsync(f.fileno())
                        
                        saved.append(thumb_path)
                        LOGGER.info("Saved thumbnail: %s (size: %d bytes)", 
                                   thumb_path, thumb_path.stat().st_size)
                    else:
                        LOGGER.error("Failed to save thumbnail %s - file doesn't exist after imwrite", 
                                    thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save thumbnail %s: %s", thumb_path, e)
        
        LOGGER.info("Saved %d thumbnails for %s", len(saved), clip_path.name)
        return saved
    
    def _extract_sample_frames(
        self,
        clip_path: Path,
        num_samples: int = 3,
    ) -> List[Path]:
        """Extract sample frames from video when no detections are found.
        
        This provides fallback thumbnails so the user can see what's in the video.
        Frames are taken at 25%, 50%, and 75% through the video.
        """
        saved = []
        clip_stem = clip_path.stem
        
        # First, remove old thumbnails for this clip
        for old_thumb in clip_path.parent.glob(f"{clip_stem}_thumb_*.jpg"):
            try:
                old_thumb.unlink()
            except Exception as e:
                LOGGER.warning("Failed to remove old thumbnail %s: %s", old_thumb, e)
        
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            LOGGER.error("Could not open video for frame extraction: %s", clip_path)
            return saved
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                LOGGER.warning("Video has no frames: %s", clip_path)
                return saved
            
            # Sample at 25%, 50%, 75% of video
            sample_positions = [0.25, 0.5, 0.75]
            
            for idx, pos in enumerate(sample_positions[:num_samples]):
                frame_num = int(total_frames * pos)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Build thumbnail path
                thumb_name = f"{clip_stem}_thumb_sample_{idx}.jpg"
                thumb_path = clip_path.parent / thumb_name
                
                try:
                    # Add a label indicating this is a sample frame (no detection)
                    annotated = frame.copy()
                    label = f"Frame {frame_num}/{total_frames} (no detection)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    # Put label at bottom of frame
                    h, w = frame.shape[:2]
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Semi-transparent background
                    cv2.rectangle(
                        annotated,
                        (0, h - text_height - 10),
                        (text_width + 10, h),
                        (0, 0, 0),
                        -1
                    )
                    cv2.putText(
                        annotated, label, (5, h - 5),
                        font, font_scale, (255, 255, 255), thickness
                    )
                    
                    # Save thumbnail
                    cv2.imwrite(str(thumb_path), annotated)
                    
                    # Force sync to disk (important for NFS mounts)
                    if thumb_path.exists():
                        with open(thumb_path, 'r+b') as f:
                            os.fsync(f.fileno())
                        
                        saved.append(thumb_path)
                        LOGGER.info("Saved sample frame thumbnail: %s (size: %d bytes)", 
                                   thumb_path, thumb_path.stat().st_size)
                    else:
                        LOGGER.error("Failed to save sample thumbnail %s - file doesn't exist after imwrite", 
                                    thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save sample thumbnail %s: %s", thumb_path, e)
        
        finally:
            cap.release()
        
        LOGGER.info("Extracted %d sample frames for %s", len(saved), clip_path.name)
        return saved
    
    def _annotate_frame(
        self,
        frame,
        species: str,
        confidence: float,
        bbox: Optional[List[float]],
        detection_num: Optional[int] = None,
    ):
        """Annotate frame with bounding box and label."""
        annotated = frame.copy()
        
        if bbox:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Build label
            if detection_num:
                label = f"{species} #{detection_num} ({confidence:.0%})"
            else:
                label = f"{species} ({confidence:.0%})"
            
            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 4, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 5),
                font, font_scale, (0, 0, 0), thickness
            )
        
        return annotated


def process_all_clips(
    storage_root: Path,
    detector: BaseDetector,
    camera_filter: Optional[List[str]] = None,
    update_filenames: bool = True,
    regenerate_thumbnails: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> List[PostProcessResult]:
    """Process all clips in storage to improve classifications.
    
    Args:
        storage_root: Root storage directory
        detector: Detection backend
        camera_filter: Only process clips from these cameras (None = all)
        update_filenames: Whether to rename files when species changes
        regenerate_thumbnails: Whether to regenerate detection thumbnails
        sample_rate: Analyze every Nth frame
        
    Returns:
        List of PostProcessResult for each clip processed
    """
    clips_dir = storage_root / 'clips'
    if not clips_dir.exists():
        LOGGER.warning("Clips directory not found: %s", clips_dir)
        return []
    
    processor = ClipPostProcessor(
        detector=detector,
        storage_root=storage_root,
        sample_rate=sample_rate,
    )
    
    results = []
    
    # Find all clip files
    for clip_path in clips_dir.rglob('*.mp4'):
        # Skip if camera filter is set and this camera isn't included
        if camera_filter:
            # Camera ID is first directory under clips/
            rel_path = clip_path.relative_to(clips_dir)
            camera_id = rel_path.parts[0] if rel_path.parts else None
            if camera_id and camera_id not in camera_filter:
                continue
        
        result = processor.process_clip(
            clip_path,
            update_filename=update_filenames,
            regenerate_thumbnails=regenerate_thumbnails,
        )
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    updated = sum(1 for r in results if r.new_path is not None)
    LOGGER.info(
        "Post-processing complete: %d/%d clips processed, %d updated",
        successful, len(results), updated
    )
    
    return results
