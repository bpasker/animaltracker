"""Post-processing module for improving clip classifications.

This module provides functionality to reanalyze saved video clips using
the detector to get more accurate species classifications and better
detection thumbnails.
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

# Configuration
DEFAULT_SAMPLE_RATE = 5  # Analyze every Nth frame
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to catch more candidates
MAX_KEY_FRAMES_PER_SPECIES = 3


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
    success: bool = True
    error: Optional[str] = None


class ClipPostProcessor:
    """Post-processor for analyzing and improving clip classifications."""
    
    def __init__(
        self,
        detector: BaseDetector,
        storage_root: Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        generic_confidence: float = 0.5,
        tracking_enabled: bool = True,
    ):
        """Initialize the post-processor.
        
        Args:
            detector: Detection backend to use for analysis
            storage_root: Root directory for clip storage
            sample_rate: Analyze every Nth frame (lower = more thorough but slower)
            confidence_threshold: Minimum confidence for specific species detections
            generic_confidence: Minimum confidence for generic categories (animal, bird)
            tracking_enabled: Use object tracking to consolidate species across frames
        """
        self.detector = detector
        self.storage_root = storage_root
        self.sample_rate = sample_rate
        self.tracking_enabled = tracking_enabled
        self.confidence_threshold = confidence_threshold
        self.generic_confidence = generic_confidence
        
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
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            species_results, raw_detection_count, filtered_count = self._analyze_video(cap, total_frames)
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
        
        LOGGER.info(
            "Post-processing complete: %s -> %s (%.1f%% confidence, %d species found)",
            original_species, new_species, confidence * 100, len(species_results)
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
            success=True,
        )
    
    def _analyze_video(
        self, 
        cap: cv2.VideoCapture, 
        total_frames: int
    ) -> Tuple[Dict[str, SpeciesResult], int, int]:
        """Analyze video frames and collect species detections.
        
        Uses object tracking (if enabled) to consolidate classifications for 
        the same animal across frames, producing one species per tracked object.
        
        Returns:
            (species_results dict, raw_detection_count, filtered_count)
        """
        # Get video FPS and calculate smart sample rate (~1 frame per second)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        # Sample roughly 1 frame per second, with min/max bounds
        smart_sample_rate = max(1, min(int(fps), 30))  # Cap at 30fps videos
        actual_sample_rate = self.sample_rate if self.sample_rate > 1 else smart_sample_rate
        effective_fps = fps / actual_sample_rate
        
        LOGGER.info("Video fps=%.1f, sample_rate=%d (effective %.1f fps)", 
                   fps, actual_sample_rate, effective_fps)
        
        # Create tracker if enabled
        tracker = create_tracker(enabled=self.tracking_enabled, frame_rate=effective_fps)
        if tracker:
            LOGGER.info("Object tracking enabled for post-processing")
        
        species_results: Dict[str, SpeciesResult] = {}
        frame_idx = 0
        raw_detection_count = 0
        filtered_count = 0
        all_frames_data: List[Tuple[int, any, List[Detection]]] = []  # For tracking
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame (using smart sample rate)
            if frame_idx % actual_sample_rate == 0:
                try:
                    detections = self.detector.infer(
                        frame, 
                        conf_threshold=self.confidence_threshold,
                        generic_confidence=self.generic_confidence
                    )
                    raw_detection_count += len(detections)
                    
                    if tracker and detections:
                        # Update tracker with detections
                        tracker.update(detections, frame)
                        all_frames_data.append((frame_idx, frame.copy(), detections))
                    
                    # Still process detections for non-tracked fallback
                    valid, filtered = self._process_detections(detections, frame, species_results)
                    filtered_count += filtered
                except Exception as e:
                    LOGGER.warning("Detection failed on frame %d: %s", frame_idx, e)
            
            frame_idx += 1
        
        # If tracking was used and we have tracked objects, build results from tracks
        if tracker and tracker.active_track_count > 0:
            tracked_results = self._build_tracked_species_results(tracker, all_frames_data)
            if tracked_results:
                LOGGER.info("Tracking consolidated %d detections into %d tracked objects",
                           raw_detection_count, len(tracked_results))
                return tracked_results, raw_detection_count, filtered_count
        
        # Fallback to non-tracked results
        return species_results, raw_detection_count, filtered_count
    
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
        
        Higher scores indicate more specific identifications.
        """
        words = species.lower().replace('-', '_').split('_')
        # Count meaningful words (not generic terms)
        specificity = len([w for w in words if w and w not in self.generic_terms])
        
        # Bonus for binomial names (genus_species format)
        if '_' in species and len(species.split('_')) >= 2:
            specificity += 2
        
        return specificity
    
    def _select_best_species(
        self, 
        species_results: Dict[str, SpeciesResult]
    ) -> Tuple[str, float]:
        """Select the best species classification from results.
        
        Prioritizes:
        1. Specificity (more specific names are better)
        2. Detection count (more detections = more reliable)
        3. Confidence (higher confidence = more certain)
        
        Returns (species_name, confidence).
        """
        if not species_results:
            return "", 0.0
        
        # Filter out generic categories if we have specific ones
        specific_species = {
            name: result for name, result in species_results.items()
            if name.lower() not in self.generic_terms
        }
        
        candidates = specific_species if specific_species else species_results
        
        if not candidates:
            return "", 0.0
        
        # Score and rank candidates
        def score_species(name: str) -> Tuple[int, int, float]:
            result = candidates[name]
            return (result.specificity, result.count, result.confidence)
        
        best_species = max(candidates.keys(), key=score_species)
        best_result = candidates[best_species]
        
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
                    
                    if thumb_path.exists():
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
                    
                    if thumb_path.exists():
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
