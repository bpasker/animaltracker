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

from .detector import BaseDetector, Detection, create_detector, cleanup_gpu_memory, NON_ANIMAL_REASON_PREFIX
from .tracker import ObjectTracker, create_tracker
from .species_names import pick_species_by_lineage, species_rank

LOGGER = logging.getLogger(__name__)


def _bbox_iou(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes (0.0 when unusable)."""
    try:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
    except (TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class NonAnimalBoxes:
    """Person/vehicle boxes the detector reported, indexed for the shadow test.

    A person who sits still yields hundreds of near-identical boxes, so the
    whole-clip test runs against a deduplicated list (boxes within IoU 0.95 of
    one already kept are folded into it); the windowed test uses the per-frame
    index.
    """

    DEDUP_IOU = 0.95

    def __init__(self) -> None:
        self.by_frame: Dict[int, List[List[float]]] = {}
        self.unique: List[List[float]] = []
        self.count = 0

    def add(self, frame_idx: int, bbox) -> None:
        box = list(bbox)
        self.by_frame.setdefault(frame_idx, []).append(box)
        self.count += 1
        if not any(_bbox_iou(box, kept) >= self.DEDUP_IOU for kept in self.unique):
            self.unique.append(box)

    def __bool__(self) -> bool:
        return self.count > 0

    def overlaps(self, bbox, frame_idx: int, iou_threshold: float, window: int) -> bool:
        """Does ``bbox`` sit on a person/vehicle box? ``window`` <= 0 means
        anywhere in the clip; otherwise only within that many raw frames."""
        if window <= 0:
            return any(_bbox_iou(bbox, other) >= iou_threshold for other in self.unique)
        for f in range(frame_idx - window, frame_idx + window + 1):
            for other in self.by_frame.get(f, ()):
                if _bbox_iou(bbox, other) >= iou_threshold:
                    return True
        return False


def _verified_write(path: "Path", write_fn, *, min_bytes: int = 1, label: str = "file") -> bool:
    """Run ``write_fn()`` then verify ``path`` exists with at least ``min_bytes``.

    If verification fails, retry once after a short pause. This works around
    occasional silent loss of small writes on some NFS shares (where the
    write call returns success but the file is not durable yet because of
    attribute-cache / negative-dentry races shortly after a rename).

    ``write_fn`` may return ``False`` to signal it failed (e.g. ``cv2.imwrite``);
    any other return value is treated as success.

    Returns ``True`` if the file is present and non-empty after the attempt(s).
    """
    import time

    for attempt in (1, 2):
        wrote_ok = True
        try:
            result = write_fn()
            if result is False:
                wrote_ok = False
                LOGGER.warning(
                    "%s write returned False (attempt %d): %s", label, attempt, path
                )
        except Exception as e:
            wrote_ok = False
            LOGGER.warning(
                "%s write raised on attempt %d for %s: %s", label, attempt, path, e
            )

        if wrote_ok:
            try:
                size = path.stat().st_size
                if size >= min_bytes:
                    if attempt > 1:
                        LOGGER.info(
                            "%s verified after retry: %s (%d bytes)",
                            label, path, size,
                        )
                    return True
                LOGGER.warning(
                    "%s wrote %d bytes (< %d) on attempt %d: %s",
                    label, size, min_bytes, attempt, path,
                )
            except FileNotFoundError:
                LOGGER.warning(
                    "%s missing after write (attempt %d): %s", label, attempt, path
                )
            except Exception as e:
                LOGGER.warning(
                    "%s stat failed (attempt %d) for %s: %s",
                    label, attempt, path, e,
                )

        if attempt == 1:
            time.sleep(0.1)  # brief settle before retry (NFS attr cache)

    LOGGER.error("%s FAILED to persist after retry: %s", label, path)
    return False


def _drop_directory_cache(directory: Path) -> None:
    """Discard the kernel's cached listing of ``directory``.

    On the production NFS mount a listing fetched between the clip rename and
    the sidecar writes can stay cached for good: everything lands in one
    server clock tick, so the directory's change attribute never moves and
    the client keeps a listing that predates the thumbnails. The files open
    fine by name, but every glob (the web UI's key frames, the alert photo)
    misses them. POSIX_FADV_DONTNEED on the directory drops those pages so
    the next listing is fetched from the server. No-op where unsupported.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError as e:
        LOGGER.debug("Could not open %s to drop its listing cache: %s", directory, e)
        return
    try:
        fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError as e:
        LOGGER.debug("Could not drop the listing cache of %s: %s", directory, e)
    finally:
        os.close(fd)


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
    # A track that starts within this many body lengths (longest box side)
    # of where the previous one ended continues it, whatever the IoU: an
    # animal walking away shrinks faster than it moves. 0 disables.
    spatial_merge_reach: float = 1.0
    hierarchical_merge_enabled: bool = True  # Merge generic→specific (animal→canidae)
    hierarchical_merge_gap: int = 120  # Max frame gap for hierarchical merge
    min_specific_detections: int = 2  # Min detections for specific track to absorb generic
    single_animal_mode: bool = False  # Aggressive merge: assume only 1 animal in video

    # Person-shadow filter. SpeciesNet sometimes labels a person "bird" or
    # "rodent" — for a frame or two, or for ten seconds at a stretch — and
    # those boxes sit exactly where it called the same object a person at
    # another moment of the clip. A detection is a "shadow" when its box
    # overlaps a person/vehicle box from the clip by at least
    # ``person_shadow_iou``; ``person_shadow_window`` = 0 looks at the whole
    # clip (a person and an animal never trade the exact same box within one
    # event), > 0 restricts the match to that many raw frames either side. A
    # track that is mostly shadows is dropped before the species vote, and so
    # is a per-frame species whose every detection was one.
    person_shadow_enabled: bool = True
    person_shadow_iou: float = 0.6
    person_shadow_window: int = 0
    person_shadow_min_fraction: float = 0.5
    
    # Output settings
    max_thumbnails: int = MAX_KEY_FRAMES_PER_SPECIES
    thumbnail_cropped: bool = True  # Crop to detection area (True) or full frame with bbox (False)
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
            "spatial_merge_reach": self.spatial_merge_reach,
            "hierarchical_merge_enabled": self.hierarchical_merge_enabled,
            "hierarchical_merge_gap": self.hierarchical_merge_gap,
            "min_specific_detections": self.min_specific_detections,
            "single_animal_mode": self.single_animal_mode,
            "person_shadow_enabled": self.person_shadow_enabled,
            "person_shadow_iou": self.person_shadow_iou,
            "person_shadow_window": self.person_shadow_window,
            "person_shadow_min_fraction": self.person_shadow_min_fraction,
            "max_thumbnails": self.max_thumbnails,
            "thumbnail_cropped": self.thumbnail_cropped,
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
            spatial_merge_reach=data.get("spatial_merge_reach", 1.0),
            hierarchical_merge_enabled=data.get("hierarchical_merge_enabled", True),
            hierarchical_merge_gap=data.get("hierarchical_merge_gap", 120),
            min_specific_detections=data.get("min_specific_detections", 2),
            single_animal_mode=data.get("single_animal_mode", False),
            person_shadow_enabled=data.get("person_shadow_enabled", True),
            person_shadow_iou=data.get("person_shadow_iou", 0.6),
            person_shadow_window=data.get("person_shadow_window", 0),
            person_shadow_min_fraction=data.get("person_shadow_min_fraction", 0.5),
            max_thumbnails=data.get("max_thumbnails", MAX_KEY_FRAMES_PER_SPECIES),
            thumbnail_cropped=data.get("thumbnail_cropped", True),
            save_processing_log=data.get("save_processing_log", True),
        )


def build_processing_settings(clip_cfg, overrides: Optional[Dict] = None) -> ProcessingSettings:
    """The post-processor settings the configuration asks for.

    One mapping for every caller — a live event, a reanalysis from the clip
    page and the recovery sweep — so a clip analysed later comes out the same
    as it would have at the time. ``clip_cfg`` is ``general.clip`` (any object
    with those attributes, or None for the defaults); ``overrides`` are
    ``ProcessingSettings`` field names, as the reanalysis dialog sends them.
    """
    def get(name, default):
        return getattr(clip_cfg, name, default) if clip_cfg is not None else default

    merge_gap = get('track_merge_gap', 120)
    values = {
        'sample_rate': get('sample_rate', DEFAULT_SAMPLE_RATE),
        'confidence_threshold': get('post_analysis_confidence', DEFAULT_CONFIDENCE_THRESHOLD),
        'generic_confidence': get('post_analysis_generic_confidence', DEFAULT_GENERIC_CONFIDENCE),
        'tracking_enabled': get('tracking_enabled', True),
        'merge_enabled': True,
        'same_species_merge_gap': merge_gap,
        'spatial_merge_enabled': get('spatial_merge_enabled', True),
        'spatial_merge_iou': get('spatial_merge_iou', 0.3),
        'spatial_merge_gap': 30,
        'spatial_merge_reach': get('spatial_merge_reach', 1.0),
        'hierarchical_merge_enabled': get('hierarchical_merge_enabled', True),
        'hierarchical_merge_gap': merge_gap,
        'single_animal_mode': get('single_animal_mode', False),
        'thumbnail_cropped': get('thumbnail_cropped', True),
    }
    for key, value in (overrides or {}).items():
        values[key] = value
    _check_processing_values(values)
    return ProcessingSettings.from_dict(values)


def _check_processing_values(values: Dict) -> None:
    """Refuse values an analysis could only fail on, before it starts.

    A reanalysis can run for most of an hour; a bad override from the clip
    page's dialog should cost a 400 now, not a 500 then.
    """
    rate = values.get('sample_rate')
    if isinstance(rate, bool) or not isinstance(rate, int) or rate < 0:
        raise ValueError(f"sample_rate must be a whole number, 0 (automatic) or more, not {rate!r}")
    for name in ('confidence_threshold', 'generic_confidence'):
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a number between 0 and 1, not {value!r}")


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
    # Sampled frames on which the detector raised. A result with any of these
    # is no proof that the clip is empty: the caller must not delete on it.
    inference_errors: int = 0
    # Sampled frames with a detection that stands behind a species in this
    # result: frames of the tracks that survived (or, without tracking, of
    # the accepted detections that are not a person's shadow). This is what
    # ``min_detection_frames`` is measured against. None from code that does
    # not compute it.
    detection_frames: Optional[int] = None
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
        self.thumbnail_cropped = self.settings.thumbnail_cropped
        
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
        LOGGER.info("Settings: confidence_threshold=%.2f, generic_confidence=%.2f, tracking=%s, sample_rate=%d, spatial_iou=%.2f",
                   self.confidence_threshold, self.generic_confidence, self.tracking_enabled, self.sample_rate, self.settings.spatial_merge_iou)
        
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
        tracker = None
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            species_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata, tracker = \
                self._analyze_video(cap, total_frames)
            # The frames the detector was really shown. This used to be worked
            # out from the configured sample rate, which the analysis itself
            # treats as "choose for me" when it is 0 (and, without tracking,
            # when it is 1): a rate of 0 ran the whole analysis and then died
            # here on a division by zero, with nothing written, and without
            # tracking the figure overstated what had been looked at, which
            # the false-positive gates downstream rely on.
            frames_analyzed = int(video_metadata.get("frames_inferred") or 0)
            
        finally:
            cap.release()

        # A frame the detector raised on says nothing about what was in it.
        # When that is most of the clip the analysis has not happened: report
        # it as failed and touch nothing, so the clip keeps its name and its
        # old key frames, no sidecar claims it is finished, and the recovery
        # sweep tries it again. This used to come back as a successful "no
        # animal", which the live path acts on by deleting the clip: choosing
        # YOLO as the post-processing detector did it to every event, and a
        # CUDA out-of-memory error did it to whichever clip it hit.
        inference_errors = int(video_metadata.get("inference_errors", 0) or 0)
        frames_inferred = int(video_metadata.get("frames_inferred", 0) or 0)
        if inference_errors and inference_errors * 2 >= frames_inferred:
            error = (
                f"the detector failed on {inference_errors} of {frames_inferred} sampled frames"
                f" ({video_metadata.get('first_inference_error', 'unknown error')})"
            )
            LOGGER.error("Post-processing of %s failed: %s", clip_path.name, error)
            return PostProcessResult(
                original_path=clip_path,
                new_path=None,
                original_species=original_species,
                new_species=original_species,
                confidence=0.0,
                species_results={},
                thumbnails_saved=[],
                frames_analyzed=frames_inferred,
                total_frames=total_frames,
                processing_log=processing_log,
                settings_used=self.settings,
                inference_errors=inference_errors,
                success=False,
                error=error,
            )
        if inference_errors:
            LOGGER.warning(
                "Post-processing of %s: the detector failed on %d of %d sampled frames (%s); "
                "the result stands on the rest",
                clip_path.name, inference_errors, frames_inferred,
                video_metadata.get("first_inference_error", "unknown error"),
            )
        
        # Count how many frames had no animal detected (blank frames)
        blank_frame_count = sum(1 for entry in processing_log 
                                if entry.event == "detector_filtered" and entry.reason == "no_animal_detected")
        
        if raw_detection_count == 0 and blank_frame_count > 0:
            LOGGER.info(
                "Analysis: %d raw detections (SpeciesNet found no animals in %d/%d sampled frames)",
                raw_detection_count, blank_frame_count, frames_analyzed
            )
        else:
            LOGGER.info(
                "Analysis: %d raw detections, %d filtered out, %d valid species",
                raw_detection_count, filtered_count, len(species_results)
            )
        
        # PTZ decisions the pipeline parked in a sidecar of this clip (a
        # failed first analysis leaves nothing else there; a reanalysis finds
        # them at the end of the old log). Read before any rename: they are
        # under the old name.
        carried_ptz_decisions = self._read_ptz_decisions(clip_path)

        # Determine best species classification
        new_species, confidence = self._select_best_species(species_results)
        
        if not new_species:
            new_species = original_species
            confidence = 0.0
        
        # Where the clip is going, if it is going anywhere. The key frames and
        # the log are written under that name FIRST and the clip is renamed
        # LAST, because the rename is what tells the recovery sweep this clip
        # is finished. Renaming first left a window of a few seconds in which
        # a restart (every deploy is one, and a restart abandons a running
        # analysis) produced a clip named for its species with no key frames
        # and no sidecar: the sweep only looks at clips that still carry the
        # generic name, so it showed "No frame" for good. Killed the other way
        # round, the clip still has its generic name and no sidecar of its
        # own, which is exactly what the sweep picks up and redoes.
        rename_to = None
        if update_filename and new_species != original_species and new_species:
            rename_to = self._planned_clip_path(clip_path, new_species)
        working_path = rename_to if rename_to is not None else clip_path
        
        # Regenerate thumbnails with best detection frames
        thumbnails_saved = []
        if regenerate_thumbnails:
            if tracker and tracker.active_track_count > 0:
                # Use per-track thumbnails when tracking is enabled
                # This ensures each track gets its own thumbnail with correct time range
                thumbnails_saved = self._save_track_thumbnails(working_path, tracker, video_metadata.get('fps', 15.0))
            elif species_results:
                # Fallback to species-based thumbnails when tracking is disabled
                thumbnails_saved = self._save_thumbnails(working_path, species_results)
            else:
                # No valid detections - extract sample frames from video as fallback
                LOGGER.info("No valid detections found, extracting sample frames as fallback")
                thumbnails_saved = self._extract_sample_frames(
                    working_path, num_samples=3, source=clip_path)
        
        # Save processing log as JSON alongside the clip. It names the
        # thumbnails written above so readers can find them without trusting
        # a directory listing (see _drop_directory_cache).
        self._save_processing_log(
            working_path, processing_log, tracking_summary, video_metadata,
            thumbnails=thumbnails_saved if regenerate_thumbnails else None,
            ptz_decisions=carried_ptz_decisions,
        )

        # Everything this run produced is on disk under the new name: commit.
        new_path = None
        if rename_to is not None:
            new_path = self._rename_clip(clip_path, new_species)
            if new_path is None:
                # The outputs sit under a name the clip did not take. The clip
                # keeps its generic name and has no sidecar of its own, so the
                # sweep will run it again and write over them.
                LOGGER.warning(
                    "Analysis of %s finished but the clip could not be renamed to %s; "
                    "leaving it for the recovery sweep",
                    clip_path.name, working_path.name,
                )
                working_path = clip_path
            else:
                # The log under the old name describes a file that no longer
                # exists, and everything in it has been carried over.
                try:
                    clip_path.with_suffix('.log.json').unlink(missing_ok=True)
                except OSError as e:
                    LOGGER.warning("Failed to remove superseded processing log for %s: %s",
                                   clip_path.name, e)
        _drop_directory_cache(working_path.parent)
        
        # Count unique tracks (animals) detected
        # Tracks that resolved to an animal. (A person's shadow track is gone by
        # now; anything else non-animal is counted in total_tracks only.)
        if tracking_summary:
            tracks_detected = tracking_summary.get("animal_tracks", tracking_summary.get("total_tracks", 0))
        else:
            tracks_detected = len(species_results)
        
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
            inference_errors=inference_errors,
            detection_frames=video_metadata.get("detection_frames"),
            success=True,
        )
    
    @staticmethod
    def _read_ptz_decisions(clip_path: Path) -> Optional[list]:
        """The PTZ decisions recorded in the clip's current sidecar, if any."""
        import json

        log_path = clip_path.with_suffix('.log.json')
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        decisions = data.get('ptz_decisions') if isinstance(data, dict) else None
        return decisions if isinstance(decisions, list) and decisions else None

    def _save_processing_log(
        self,
        clip_path: Path,
        processing_log: List[ProcessingLogEntry],
        tracking_summary: Optional[Dict],
        video_metadata: Optional[Dict] = None,
        thumbnails: Optional[List[Path]] = None,
        ptz_decisions: Optional[list] = None,
    ) -> None:
        """Save processing log as JSON file alongside clip.

        ``thumbnails`` are the files written for this clip in this run; their
        names go into the log's ``thumbnails`` list. ``None`` (thumbnails not
        regenerated) leaves the key out so a stale list is never recorded.
        ``ptz_decisions`` are the event's PTZ decisions from an earlier
        sidecar; they go last, where the pipeline appends them, so writing
        the log again does not lose them.
        """
        import json
        from dataclasses import asdict
        
        log_path = clip_path.with_suffix('.log.json')
        
        # Calculate summary statistics from processing log. A frame can carry
        # several entries (one per box SpeciesNet judged, one per track), so
        # count frames, not entries.
        blank_frames = len({
            e.frame_idx for e in processing_log
            if e.event == "detector_filtered" and e.reason == "no_animal_detected"
        })
        detection_frames = len({
            e.frame_idx for e in processing_log if e.event in ("tracked", "detection")
        })
        filtered_frames = len({
            e.frame_idx for e in processing_log
            if e.event == "detector_filtered" and e.reason != "no_animal_detected"
        })
        non_animal_frames = len({
            e.frame_idx for e in processing_log
            if e.event == "detector_filtered" and (e.reason or "").startswith(NON_ANIMAL_REASON_PREFIX)
        })
        
        try:
            log_data = {
                "clip": str(clip_path.name),
                "timestamp": str(Path(clip_path.stem).name.split('_')[0]) if '_' in clip_path.stem else "",
                "settings": self.settings.to_dict(),  # Full settings object
                "detector_type": type(self.detector).__name__,
                "video": video_metadata or {},
                "analysis_summary": {
                    "frames_analyzed": video_metadata.get("frames_to_analyze", 0) if video_metadata else 0,
                    "frames_with_detections": detection_frames,
                    "frames_with_no_animal": blank_frames,
                    "frames_filtered_other": filtered_frames,
                    "frames_with_non_animal": non_animal_frames,
                    "detection_rate_pct": round(100 * detection_frames / max(1, video_metadata.get("frames_to_analyze", 1)), 1) if video_metadata else 0,
                },
                "tracking_summary": tracking_summary,
            }
            if thumbnails is not None:
                log_data["thumbnails"] = [{"file": Path(p).name} for p in thumbnails]
            log_data["log_entries"] = [asdict(entry) for entry in processing_log]
            if ptz_decisions:
                log_data["ptz_decisions"] = ptz_decisions
            
            def _do_write():
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2, default=str)
                return True

            if _verified_write(log_path, _do_write, min_bytes=2, label="processing log"):
                try:
                    size = log_path.stat().st_size
                except Exception:
                    size = -1
                LOGGER.info("Saved processing log: %s (%d bytes)", log_path, size)
        except Exception as e:
            LOGGER.warning("Failed to save processing log: %s", e)

    def _analyze_video(
        self, 
        cap: cv2.VideoCapture, 
        total_frames: int
    ) -> Tuple[Dict[str, SpeciesResult], int, int, List[ProcessingLogEntry], Optional[Dict], Dict, Optional['ObjectTracker']]:
        """Analyze video frames and collect species detections.
        
        Uses object tracking (if enabled) to consolidate classifications for 
        the same animal across frames, producing one species per tracked object.
        
        Returns:
            (species_results dict, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata, tracker)
        """
        # Get video FPS and calculate sample rate
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        
        # When tracking is enabled, sample more frequently for better track continuity
        # Without tracking, we can sample less frequently since we just need species votes
        if self.tracking_enabled:
            # For tracking, default to every 3rd frame (~5-10 fps effective) to give
            # ByteTrack enough visual overlap. Honor an explicit config override so
            # users can trade tracking quality for speed (e.g. sample_rate=5 cuts
            # SpeciesNet inference cost ~40% with negligible quality loss for normal
            # animal motion).
            actual_sample_rate = self.sample_rate if self.sample_rate and self.sample_rate > 0 else 3
        else:
            # For non-tracking: ~1 frame per second is fine
            smart_sample_rate = max(1, min(int(fps), 30))
            actual_sample_rate = self.sample_rate if self.sample_rate > 1 else smart_sample_rate
        
        effective_fps = fps / actual_sample_rate
        
        # Build video metadata for logging
        video_metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
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
        processing_log: List[ProcessingLogEntry] = []
        non_animal_boxes = NonAnimalBoxes()  # person/vehicle boxes, for the shadow test
        frames_inferred = 0   # sampled frames handed to the detector
        inference_errors = 0  # ...on which it raised
        first_error: Optional[str] = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame (using smart sample rate)
            if frame_idx % actual_sample_rate == 0:
                frames_inferred += 1
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
                        if reason.startswith(NON_ANIMAL_REASON_PREFIX) and det.bbox:
                            non_animal_boxes.add(frame_idx, det.bbox)
                    
                    if tracker:
                        # ALWAYS update tracker, even with empty detections.
                        # This keeps ByteTrack's internal frame counter synchronized
                        # so lost_track_buffer and Kalman filter predictions work correctly.
                        # Without this, blank frames don't advance the counter,
                        # corrupting track association for subsequent detections.
                        tracked = tracker.update(detections, frame, frame_idx=frame_idx)
                        # No copy of the frame is kept here: TrackInfo already
                        # holds each track's best frames, and a list of every
                        # sampled frame grew to tens of GB on a long clip.
                        
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
                    # Counted, not just logged: a frame the detector could not
                    # judge is not a frame without an animal (see process_clip).
                    inference_errors += 1
                    if first_error is None:
                        first_error = f"{type(e).__name__}: {e}"
                        LOGGER.warning("Detection failed on frame %d: %s", frame_idx, e, exc_info=True)
                    else:
                        LOGGER.warning("Detection failed on frame %d: %s", frame_idx, e)
                    processing_log.append(ProcessingLogEntry(
                        frame_idx=frame_idx,
                        event="error",
                        species="",
                        confidence=0.0,
                        reason=str(e),
                    ))
            
            frame_idx += 1

        video_metadata["frames_inferred"] = frames_inferred
        video_metadata["inference_errors"] = inference_errors
        if first_error is not None:
            video_metadata["first_inference_error"] = first_error
        
        # Build tracking summary
        tracking_summary = None

        # Person-shadow filter first, so no merge below can fold a shadow
        # fragment into a real animal's track.
        shadow_tracks = 0
        shadow_track_boxes: set = set()   # (frame_idx, bbox) of every detection in a dropped track
        if self.settings.person_shadow_enabled and non_animal_boxes:
            if tracker and tracker.active_track_count > 0:
                shadow_tracks, shadow_log = self._drop_person_shadow_tracks(
                    tracker, non_animal_boxes, dropped_boxes=shadow_track_boxes)
                processing_log.extend(shadow_log)
            dropped_species = self._drop_person_shadow_species(
                species_results, processing_log, non_animal_boxes, cap)
            if dropped_species:
                processing_log.append(ProcessingLogEntry(
                    frame_idx=-1, event="person_shadow", species=", ".join(dropped_species),
                    confidence=0.0,
                    reason=(f"Dropped {len(dropped_species)} per-frame species whose every "
                            f"detection sat on a person/vehicle box"),
                ))

        # If tracking was used and we have tracked objects, build results from tracks
        if tracker and tracker.active_track_count > 0:
            processing_log.extend(self._merge_tracks(tracker))

            tracked_results, track_log, tracking_summary = self._build_tracked_species_results_with_log(
                tracker
            )
            processing_log.extend(track_log)
            
            tracking_summary["person_shadow_tracks"] = shadow_tracks

            if tracked_results:
                LOGGER.info("Tracking consolidated %d detections into %d tracked objects",
                           raw_detection_count, len(tracked_results))
                video_metadata["detection_frames"] = self._evidence_frames(
                    processing_log, non_animal_boxes, shadow_track_boxes)
                return tracked_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata, tracker
            # Every track resolved to a non-animal. The tracker has explained
            # each detection, so the per-frame votes below must not resurrect
            # them as a species.
            LOGGER.info("Tracking found %d object(s) but none is an animal", tracker.active_track_count)
            video_metadata["detection_frames"] = 0
            return {}, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata, None

        if shadow_tracks:
            # Every track was a person's shadow; leave that on record.
            tracking_summary = {"total_tracks": 0, "animal_tracks": 0, "tracks": [],
                                "person_shadow_tracks": shadow_tracks}

        # Fallback to non-tracked results
        video_metadata["detection_frames"] = self._evidence_frames(
            processing_log, non_animal_boxes, shadow_track_boxes)
        return species_results, raw_detection_count, filtered_count, processing_log, tracking_summary, video_metadata, None
    
    def _merge_tracks(self, tracker: ObjectTracker) -> List[ProcessingLogEntry]:
        """Stitch ByteTrack's fragments back into one track per animal.

        Runs the merge passes in order and returns a log entry for each pass
        that changed something. The order matters: the spatial passes need
        no species agreement; the same-species and hierarchical passes build
        the long tracks; and only once those exist can a one-frame fragment
        sitting inside one of them (a dog read as "felidae" for a single
        frame) be seen for what it is, a minority vote and not a second
        animal. The two closing passes absorb only tracks with fewer than
        ``min_specific_detections`` classifications, so a real second animal
        that stayed for a while keeps its own track.
        """
        log: List[ProcessingLogEntry] = []

        def note(event: str, reason: str) -> None:
            log.append(ProcessingLogEntry(
                frame_idx=-1, event=event, species="", confidence=0.0, reason=reason,
            ))

        settings = self.settings
        weak_limit = max(1, int(settings.min_specific_detections))

        if settings.spatial_merge_enabled:
            # FIRST pass: Spatial merge - most reliable!
            # If an object appears in the same location across frames, it's the same object
            # regardless of species classification changes
            spatial_merged = tracker.merge_spatially_adjacent_tracks(
                iou_threshold=settings.spatial_merge_iou,
                max_frame_gap=settings.spatial_merge_gap,
                reach=settings.spatial_merge_reach,
            )
            if spatial_merged > 0:
                note("spatial_merge",
                     f"Merged {spatial_merged} tracks based on spatial continuity "
                     f"(IoU≥{settings.spatial_merge_iou} or centre within "
                     f"{settings.spatial_merge_reach} body lengths, gap≤{settings.spatial_merge_gap})")

            # Merge spurious parallel tracks (overlapping in time but same location)
            # ByteTrack sometimes creates duplicate tracks when briefly losing an object
            overlap_merged = tracker.merge_overlapping_same_location_tracks(
                iou_threshold=settings.spatial_merge_iou
            )
            if overlap_merged > 0:
                note("overlap_merge",
                     f"Merged {overlap_merged} spurious parallel tracks (same location, overlapping time)")

            # Merge tracks that fill gaps in larger tracks' detection timelines
            # If a smaller track exists entirely within a gap of a larger track, merge them
            gap_merged = tracker.merge_gap_filling_tracks(
                iou_threshold=settings.spatial_merge_iou,
                reach=settings.spatial_merge_reach,
                reach_frames=settings.spatial_merge_gap,
            )
            if gap_merged > 0:
                note("gap_fill_merge",
                     f"Merged {gap_merged} tracks that filled detection gaps in larger tracks")

        if settings.merge_enabled:
            # Second pass: Merge fragmented tracks with the SAME species
            # This handles cases where ByteTrack loses a track due to movement
            # but later detections are clearly the same species
            merged_count = tracker.merge_similar_tracks(
                max_frame_gap=settings.same_species_merge_gap
            )
            if merged_count > 0:
                note("tracks_merged",
                     f"Merged {merged_count} fragmented tracks with same species "
                     f"(gap≤{settings.same_species_merge_gap})")

            # Third pass: Merge GENERIC tracks into more SPECIFIC tracks
            # E.g., "animal" track absorbed into "canidae" track if temporally adjacent
            # This only merges hierarchically compatible species (animal->mammal->canidae)
            if settings.hierarchical_merge_enabled:
                hierarchical_merged = tracker.merge_hierarchical_tracks(
                    max_frame_gap=settings.hierarchical_merge_gap,
                    min_specific_detections=settings.min_specific_detections,
                )
                if hierarchical_merged > 0:
                    note("hierarchical_merge",
                         f"Absorbed {hierarchical_merged} generic tracks into specific species tracks "
                         f"(gap≤{settings.hierarchical_merge_gap})")

            # Fourth pass: the gap-fill again, now that the passes above have
            # built the long tracks. A fragment inside a gap of one of them is
            # the same animal read differently for that moment. Only fragments
            # below min_specific_detections are absorbed.
            if settings.spatial_merge_enabled:
                late_gap_merged = tracker.merge_gap_filling_tracks(
                    max_detections=weak_limit - 1,
                    iou_threshold=settings.spatial_merge_iou,
                    reach=settings.spatial_merge_reach,
                    reach_frames=settings.spatial_merge_gap,
                )
                if late_gap_merged > 0:
                    note("gap_fill_merge",
                         f"Merged {late_gap_merged} tracks with fewer than {weak_limit} detections "
                         f"that filled gaps in the stitched tracks")

            # Fifth pass: a fragment the gap-fill cannot reach (no gap around
            # it, or a parallel duplicate) that a longer compatible track spans
            # in time and overlaps in space becomes a minority vote of that
            # track instead of a species of its own.
            weak_merged = tracker.merge_weak_tracks(min_detections=weak_limit)
            if weak_merged > 0:
                note("weak_track_merge",
                     f"Absorbed {weak_merged} tracks with fewer than {weak_limit} detections "
                     f"into the longer compatible track spanning them")

            # Last: Single animal mode - aggressively merge ALL non-overlapping tracks
            # Use this when you're sure there's only one animal in the video
            if settings.single_animal_mode:
                non_overlap_merged = tracker.merge_non_overlapping_tracks()
                if non_overlap_merged > 0:
                    note("single_animal_merge",
                         f"Single-animal mode: merged {non_overlap_merged} non-overlapping tracks into 1")

        return log

    @staticmethod
    def _tracked_species_votes(tracker: ObjectTracker) -> Dict[str, Tuple[int, float]]:
        """``{species: (detections, best confidence)}`` over the tracks.

        Each track resolves to one species (``TrackInfo.get_best_species``)
        and brings all of its detections to it, so two dog tracks of 40 and
        36 detections are 76 detections of a dog. The count used to come from
        ``get_unique_species()``, which has already folded the tracks into
        one entry per species: every count was 1, and the clip's species fell
        to whichever track had the single most confident frame, a
        two-detection misread included.
        """
        votes: Dict[str, Tuple[int, float]] = {}
        for track_info in tracker.tracks.values():
            species, confidence, _ = track_info.get_best_species()
            if not species:
                continue
            count, best = votes.get(species, (0, 0.0))
            votes[species] = (count + len(track_info.classifications), max(best, confidence))
        return votes

    def _build_tracked_species_results(
        self,
        tracker: ObjectTracker,
    ) -> Dict[str, SpeciesResult]:
        """Build species results from tracked objects.
        
        Each tracked object votes for its best species based on all classifications
        it received across frames. This consolidates Dog/Mammal/Cat into one species.
        """
        species_results: Dict[str, SpeciesResult] = {}
        
        # One entry per species the tracks resolved to, with the detections
        # behind it (see _tracked_species_votes).
        for species, (count, confidence) in self._tracked_species_votes(tracker).items():
            # Filter invalid species
            species_lower = species.lower()
            if species_lower in self.invalid_terms:
                continue
            if 'no cv result' in species_lower:
                continue
            if len(species) > 30 and species.count('-') >= 3:
                continue
            
            species_results[species] = SpeciesResult(
                species=species,
                confidence=confidence,
                count=count,
                specificity=self._calculate_specificity(species),
                taxonomy=None,
                key_frames=[],
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
    ) -> Tuple[Dict[str, SpeciesResult], List[ProcessingLogEntry], Dict]:
        """Build species results from tracked objects with detailed logging."""
        species_results: Dict[str, SpeciesResult] = {}
        log_entries: List[ProcessingLogEntry] = []
        
        # Build tracking summary
        tracking_summary = {
            "total_tracks": len(tracker.tracks),
            "tracks": [],
        }
        
        # One entry per species the tracks resolved to, with the detections
        # behind it (see _tracked_species_votes).
        for species, (count, confidence) in self._tracked_species_votes(tracker).items():
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
            
            species_results[species] = SpeciesResult(
                species=species,
                confidence=confidence,
                count=count,
                specificity=self._calculate_specificity(species),
                taxonomy=None,
                key_frames=[],
            )
        
        # Get key frames and build track details
        animal_tracks = 0
        for track_id, track_info in tracker.tracks.items():
            best_species_data = track_info.get_best_species()
            if not best_species_data or not best_species_data[0]:
                continue
            
            best_species, best_conf, taxonomy = best_species_data
            if best_species in species_results:
                animal_tracks += 1
            
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
        
        tracking_summary["animal_tracks"] = animal_tracks
        return species_results, log_entries, tracking_summary
    
    # ------------------------------------------------------------------
    # Person-shadow filter
    # ------------------------------------------------------------------

    def _is_person_shadow(
        self,
        bbox: Optional[List[float]],
        frame_idx: Optional[int],
        non_animal_boxes: NonAnimalBoxes,
    ) -> bool:
        """True when an animal box overlaps a person/vehicle box from the clip
        (the whole clip, or ``person_shadow_window`` raw frames either side)."""
        if not bbox or frame_idx is None or frame_idx < 0 or not non_animal_boxes:
            return False
        return non_animal_boxes.overlaps(
            bbox, frame_idx,
            float(self.settings.person_shadow_iou),
            int(self.settings.person_shadow_window),
        )

    def _shadow_scope(self) -> str:
        window = int(self.settings.person_shadow_window)
        return "anywhere in the clip" if window <= 0 else f"within {window} frames"

    def _drop_person_shadow_tracks(
        self,
        tracker: ObjectTracker,
        non_animal_boxes: NonAnimalBoxes,
        dropped_boxes: Optional[set] = None,
    ) -> Tuple[int, List[ProcessingLogEntry]]:
        """Remove tracks that are mostly a person's (or a vehicle's) shadow.

        Returns (tracks_removed, log_entries). ``dropped_boxes``, when given,
        receives ``(frame_idx, bbox)`` for every detection of the removed
        tracks, the ones that did not overlap the person included, so none of
        them is later counted as evidence of an animal (``_evidence_frames``).
        """
        removed = 0
        log_entries: List[ProcessingLogEntry] = []
        min_fraction = float(self.settings.person_shadow_min_fraction)
        for track_id in list(tracker.tracks.keys()):
            info = tracker.tracks[track_id]
            total = 0
            shadows = 0
            for c in info.classifications:
                if not c.bbox:
                    continue
                total += 1
                if self._is_person_shadow(c.bbox, c.frame_idx, non_animal_boxes):
                    shadows += 1
            if not total or shadows / total < min_fraction:
                continue
            species, confidence, _ = info.get_best_species()
            reason = (
                f"person shadow: {shadows}/{total} detections sit on a person/vehicle box "
                f"(IoU>={self.settings.person_shadow_iou}, {self._shadow_scope()})"
            )
            LOGGER.info("Dropping track %d (%s): %s", track_id, species, reason)
            log_entries.append(ProcessingLogEntry(
                frame_idx=-1, event="track_filtered", species=species,
                confidence=confidence, track_id=track_id, reason=reason,
            ))
            if dropped_boxes is not None:
                dropped_boxes.update(
                    (c.frame_idx, tuple(c.bbox)) for c in info.classifications if c.bbox)
            del tracker.tracks[track_id]
            removed += 1
        return removed, log_entries

    def _evidence_frames(
        self,
        processing_log: List[ProcessingLogEntry],
        non_animal_boxes: NonAnimalBoxes,
        shadow_track_boxes: set,
    ) -> int:
        """Sampled frames with an accepted detection that may be an animal.

        This is what ``min_detection_frames`` is measured against. It is the
        count the live path used to take from the log itself, every frame with
        an accepted detection, less the detections that are a person's shadow
        and the ones that belonged to a track dropped as one. With those left
        in, someone walking past (a dozen shadow frames) plus one stray
        "squirrel" frame elsewhere cleared a minimum of two and alerted as a
        squirrel. A track's first detection is counted like the rest, although
        ByteTrack only reports a track from its second match, so a visit of
        exactly the minimum length still passes.
        """
        frames = set()
        for entry in processing_log:
            if entry.event != "accepted" or entry.frame_idx < 0:
                continue
            if entry.bbox and (entry.frame_idx, tuple(entry.bbox)) in shadow_track_boxes:
                continue
            if self._is_person_shadow(entry.bbox, entry.frame_idx, non_animal_boxes):
                continue
            frames.add(entry.frame_idx)
        return len(frames)

    def _drop_person_shadow_species(
        self,
        species_results: Dict[str, SpeciesResult],
        processing_log: List[ProcessingLogEntry],
        non_animal_boxes: NonAnimalBoxes,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> List[str]:
        """Rebuild the per-frame (non-tracked) species votes without shadows.

        A species whose every detection sat on a person is removed. One that
        keeps some detections is rebuilt from those alone: its count, its
        confidence and its key frames. Only the count used to be corrected,
        so a bird seen once beside a person who was read as "bird" a dozen
        times went out at the shadow's 0.93, with three crops of the person
        as its key frames.

        Returns the species that had no detection left and were removed.
        """
        kept: Dict[str, List[ProcessingLogEntry]] = {}
        shadows = 0
        for entry in processing_log:
            if entry.event != "accepted" or entry.frame_idx < 0 or not entry.bbox:
                continue
            if self._is_person_shadow(entry.bbox, entry.frame_idx, non_animal_boxes):
                shadows += 1
                continue
            kept.setdefault(entry.species, []).append(entry)
        dropped: List[str] = []
        for species in list(species_results.keys()):
            entries = kept.get(species, [])
            if not entries:
                del species_results[species]
                dropped.append(species)
                continue
            result = species_results[species]
            result.count = len(entries)
            if shadows:
                result.confidence = max(e.confidence for e in entries)
                self._rebuild_key_frames(result, entries, non_animal_boxes, cap)
        return dropped

    def _rebuild_key_frames(
        self,
        result: SpeciesResult,
        entries: List[ProcessingLogEntry],
        non_animal_boxes: NonAnimalBoxes,
        cap: Optional[cv2.VideoCapture],
    ) -> None:
        """Key frames of ``result`` from its non-shadow detections only.

        The frames kept while the clip was read are the most confident ones,
        shadows included, and which detections are shadows is only known once
        the whole clip has been seen. The real detections' frames may have
        been displaced by then, so they are read again from the clip.
        """
        def is_shadow(bbox) -> bool:
            # Key frames carry no frame index; any of the species' shadow
            # boxes identifies one.
            return bool(bbox) and non_animal_boxes.overlaps(
                bbox, 0, float(self.settings.person_shadow_iou), 0)

        frames = [kf for kf in result.key_frames if not is_shadow(kf[2])]
        # Each surviving key frame accounts for one detection with its box
        # and confidence; the others are read again, best first.
        unused = sorted(entries, key=lambda e: e.confidence, reverse=True)
        for kf in frames:
            match = next((e for e in unused
                          if list(e.bbox) == list(kf[2] or []) and e.confidence == kf[1]), None)
            if match is not None:
                unused.remove(match)
        if cap is not None and len(frames) < MAX_KEY_FRAMES_PER_SPECIES:
            for entry in unused:
                if len(frames) >= MAX_KEY_FRAMES_PER_SPECIES:
                    break
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, entry.frame_idx)
                    ok, frame = cap.read()
                except Exception as e:  # noqa: BLE001 - a seek that fails costs a key frame, not the analysis
                    LOGGER.debug("Could not re-read frame %d for a key frame: %s", entry.frame_idx, e)
                    continue
                if ok and frame is not None:
                    frames.append((frame, entry.confidence, list(entry.bbox)))
        frames.sort(key=lambda kf: kf[1], reverse=True)
        result.key_frames = frames[:MAX_KEY_FRAMES_PER_SPECIES]

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
        """Specificity score for a species name: its taxonomy depth.

        0 "animal", 1 class ("bird", "mammalia_mammal"), 2 order
        ("mammalia_rodentia_rodent"), 3 family ("mammalia_carnivora_canidae"),
        4+ below family. The same scale the tracker uses
        (``species_names.species_rank``); this one used to count the words in
        the label, which put an order rollup level with a family.
        """
        return species_rank(species)
    
    def _select_best_species(
        self, 
        species_results: Dict[str, SpeciesResult]
    ) -> Tuple[str, float]:
        """Select the best species classification from results.

        The results vote with their detection counts, and the votes are read
        as a walk down the taxonomy (``species_names.pick_species_by_lineage``):

        1. a specific label beats its own generic ancestors (canidae > mammal
           > animal), however few detections it has;
        2. labels that contradict each other are settled by detection count,
           then confidence: a deer seen in thirty frames beats a lid read as
           a duck in four, whatever the duck's confidence.

        Returns (species_name, confidence).
        """
        if not species_results:
            return "", 0.0

        best_species = pick_species_by_lineage({
            name: (result.count, result.confidence)
            for name, result in species_results.items()
        })
        if not best_species:
            return "", 0.0
        best_result = species_results[best_species]

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
    
    @staticmethod
    def _planned_clip_path(clip_path: Path, new_species: str) -> Optional[Path]:
        """Where ``_rename_clip`` would put this clip, or None if it would not.

        Used to write the key frames and the log under the clip's final name
        before the rename commits to it (see ``process_clip``).
        """
        parts = clip_path.stem.split('_', 1)
        if not parts or not parts[0]:
            return None
        clean_species = new_species.replace(' ', '_').replace('/', '_')
        new_path = clip_path.parent / f"{parts[0]}_{clean_species}{clip_path.suffix}"
        if new_path == clip_path:
            return None
        try:
            if new_path.exists():
                LOGGER.warning("Cannot rename: %s already exists", new_path)
                return None
        except OSError:
            return None
        return new_path

    def _rename_clip(self, clip_path: Path, new_species: str) -> Optional[Path]:
        """Rename clip file with new species classification.
        
        Returns new path if renamed, None if failed.
        """
        try:
            new_path = self._planned_clip_path(clip_path, new_species)
            if new_path is None:
                return None
            
            # Rename the clip
            clip_path.rename(new_path)
            LOGGER.info("Renamed clip: %s -> %s", clip_path.name, new_path.name)
            
            # Thumbnails this run wrote are already under the new name (they
            # are written before the rename); this moves any an earlier run
            # left under the old one.
            self._rename_thumbnails(clip_path, new_path)
            
            return new_path
            
        except Exception as e:
            LOGGER.error("Failed to rename clip %s: %s", clip_path, e, exc_info=True)
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
                if new_thumb_path.exists():
                    # This run already wrote a key frame there (they go under
                    # the clip's new name before the rename). A leftover from
                    # an earlier run must not replace a fresh one.
                    thumb.unlink(missing_ok=True)
                    continue
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
        clip_dir = clip_path.parent
        
        # First, remove old thumbnails for this clip
        for old_thumb in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            try:
                old_thumb.unlink()
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
                
                try:
                    # Crop or annotate based on config
                    if self.thumbnail_cropped:
                        output_frame = self._crop_to_detection(frame, bbox)
                    else:
                        output_frame = self._annotate_frame(
                            frame, species, confidence, bbox, 
                            idx + 1 if len(result.key_frames) > 1 else None
                        )
                    
                    # Save thumbnail with verify+retry (NFS-safe)
                    if _verified_write(
                        thumb_path,
                        lambda: cv2.imwrite(str(thumb_path), output_frame),
                        min_bytes=512,
                        label="thumbnail",
                    ):
                        saved.append(thumb_path)
                        LOGGER.info("Saved thumbnail: %s", thumb_path)
                    else:
                        LOGGER.error("Failed to save thumbnail: %s", thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save thumbnail %s: %s", thumb_path, e, exc_info=True)
        
        LOGGER.info("Saved %d thumbnails for %s", len(saved), clip_path.name)
        return saved
    
    def _save_track_thumbnails(
        self,
        clip_path: Path,
        tracker: 'ObjectTracker',
        fps: float = 15.0,
    ) -> List[Path]:
        """Save one thumbnail per track with track-specific naming.
        
        This ensures each tracked animal gets its own thumbnail, even if
        multiple tracks have the same species. The thumbnail filename includes
        the track index to keep them unique.
        
        Args:
            clip_path: Path to the video clip
            tracker: Object tracker with track data
            fps: Video FPS for time calculations
            
        Returns:
            List of saved thumbnail paths
        """
        saved = []
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        
        # First, remove old thumbnails for this clip
        for old_thumb in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            try:
                old_thumb.unlink()
            except Exception as e:
                LOGGER.warning("Failed to remove old thumbnail %s: %s", old_thumb, e)
        
        # Get all tracks sorted by first_seen_frame for consistent ordering
        tracks_sorted = sorted(
            tracker.tracks.items(),
            key=lambda x: x[1].first_seen_frame
        )
        
        # Save one thumbnail per track
        for track_idx, (track_id, track_info) in enumerate(tracks_sorted):
            best_species_data = track_info.get_best_species()
            if not best_species_data or not best_species_data[0]:
                LOGGER.debug("Track %d has no best species, skipping thumbnail", track_id)
                continue
            
            best_species, best_conf, taxonomy = best_species_data
            
            # Get the best frame for this species
            best_frame_data = track_info.get_best_frame_for_species(best_species)
            if not best_frame_data:
                best_frame_data = track_info.get_best_frame()
            
            if not best_frame_data:
                LOGGER.warning("Track %d (%s) has no frame data for thumbnail", track_id, best_species)
                continue
            
            frame, conf, bbox = best_frame_data
            
            # Validate frame data
            if frame is None:
                LOGGER.warning("Track %d (%s) has None frame, skipping thumbnail", track_id, best_species)
                continue
            
            # Clean species name for filename
            clean_species = best_species.replace(' ', '_').replace('/', '_').lower()
            
            # Include track index in filename to keep tracks separate
            # Format: clipname_thumb_species_t0.jpg, clipname_thumb_species_t1.jpg
            thumb_name = f"{clip_stem}_thumb_{clean_species}_t{track_idx}.jpg"
            thumb_path = clip_dir / thumb_name
            
            try:
                # Crop or annotate based on config
                if self.thumbnail_cropped:
                    output_frame = self._crop_to_detection(frame, bbox)
                else:
                    output_frame = self._annotate_frame(
                        frame, best_species, conf, bbox,
                        detection_num=track_idx + 1 if len(tracks_sorted) > 1 else None
                    )
                
                if output_frame is None:
                    LOGGER.error("Failed to process frame for track %d (%s)", track_id, best_species)
                    continue
                
                # Save thumbnail with verify+retry (some NFS shares silently
                # drop tiny writes that occur right after a rename in the same
                # directory; verify the file landed and retry once if not).
                if _verified_write(
                    thumb_path,
                    lambda: cv2.imwrite(str(thumb_path), output_frame),
                    min_bytes=512,
                    label="track thumbnail",
                ):
                    saved.append(thumb_path)
                    LOGGER.info(
                        "Saved track thumbnail: %s (track %d, frames %d-%d)",
                        thumb_path, track_id,
                        track_info.first_seen_frame, track_info.last_seen_frame,
                    )
                else:
                    LOGGER.error(
                        "Failed to save track thumbnail: %s (output shape: %s, dtype: %s)",
                        thumb_path,
                        output_frame.shape if output_frame is not None else 'None',
                        output_frame.dtype if output_frame is not None else 'None',
                    )
                    
            except Exception as e:
                LOGGER.error("Failed to save track thumbnail %s: %s (frame shape: %s, bbox: %s)", 
                            thumb_path, e, 
                            frame.shape if frame is not None else 'None',
                            bbox, exc_info=True)
        
        LOGGER.info("Saved %d track thumbnails for %s", len(saved), clip_path.name)
        return saved

    def _extract_sample_frames(
        self,
        clip_path: Path,
        num_samples: int = 3,
        source: Optional[Path] = None,
    ) -> List[Path]:
        """Extract sample frames from video when no detections are found.
        
        This provides fallback thumbnails so the user can see what's in the video.
        Frames are taken at 25%, 50%, and 75% through the video.

        ``clip_path`` names the thumbnails; ``source`` is the file to read them
        from, which is not the same thing while an analysis is writing its
        outputs under the name the clip is about to take (see ``process_clip``).
        """
        saved = []
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        read_from = source if source is not None else clip_path
        
        # First, remove old thumbnails for this clip
        for old_thumb in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            try:
                old_thumb.unlink()
            except Exception as e:
                LOGGER.warning("Failed to remove old thumbnail %s: %s", old_thumb, e)
        
        cap = cv2.VideoCapture(str(read_from))
        if not cap.isOpened():
            LOGGER.error("Could not open video for frame extraction: %s", read_from)
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
                    
                    # Save thumbnail with verify+retry (NFS-safe)
                    if _verified_write(
                        thumb_path,
                        lambda: cv2.imwrite(str(thumb_path), annotated),
                        min_bytes=512,
                        label="sample thumbnail",
                    ):
                        saved.append(thumb_path)
                        LOGGER.info("Saved sample frame thumbnail: %s", thumb_path)
                    else:
                        LOGGER.error("Failed to save sample thumbnail: %s", thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save sample thumbnail %s: %s", thumb_path, e, exc_info=True)
        
        finally:
            cap.release()
        
        LOGGER.info("Extracted %d sample frames for %s", len(saved), clip_path.name)
        return saved
    
    def _crop_to_detection(
        self,
        frame,
        bbox: Optional[List[float]],
        padding_percent: float = 0.25,
        min_size: int = 200,
    ):
        """Crop frame to detection bounding box with padding.
        
        Args:
            frame: The video frame to crop
            bbox: Bounding box [x1, y1, x2, y2] or None
            padding_percent: How much padding to add around the detection (0.25 = 25%)
            min_size: Minimum width/height of cropped area in pixels (expands if smaller)
            
        Returns:
            Cropped frame, or original frame if no valid bbox
        """
        if frame is None:
            LOGGER.error("_crop_to_detection received None frame")
            return None
        
        if not hasattr(frame, 'copy') or not hasattr(frame, 'shape'):
            LOGGER.error("_crop_to_detection received invalid frame type: %s", type(frame))
            return None
            
        if len(frame.shape) < 2:
            LOGGER.error("_crop_to_detection received invalid frame shape: %s", frame.shape)
            return None
        
        frame_height, frame_width = frame.shape[:2]
        
        if not bbox:
            # No bbox, return full frame
            return frame
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Calculate padding based on bbox size
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        pad_x = int(bbox_width * padding_percent)
        pad_y = int(bbox_height * padding_percent)
        
        # Apply padding while staying within frame bounds
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_width, x2 + pad_x)
        crop_y2 = min(frame_height, y2 + pad_y)
        
        # Calculate current crop size
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        # Expand to minimum size if needed (centered on detection)
        if crop_width < min_size:
            extra = min_size - crop_width
            crop_x1 = max(0, crop_x1 - extra // 2)
            crop_x2 = min(frame_width, crop_x1 + min_size)
            # Adjust x1 if x2 hit the boundary
            if crop_x2 - crop_x1 < min_size:
                crop_x1 = max(0, crop_x2 - min_size)
        
        if crop_height < min_size:
            extra = min_size - crop_height
            crop_y1 = max(0, crop_y1 - extra // 2)
            crop_y2 = min(frame_height, crop_y1 + min_size)
            # Adjust y1 if y2 hit the boundary
            if crop_y2 - crop_y1 < min_size:
                crop_y1 = max(0, crop_y2 - min_size)
        
        # Ensure we still have a valid crop area
        if (crop_x2 - crop_x1) < 10 or (crop_y2 - crop_y1) < 10:
            # Something went wrong, use full frame
            return frame
        
        # Crop the frame to the detection area
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        return cropped

    def _annotate_frame(
        self,
        frame,
        species: str,
        confidence: float,
        bbox: Optional[List[float]],
        detection_num: Optional[int] = None,
    ):
        """Annotate frame with bounding box and label."""
        if frame is None:
            LOGGER.error("_annotate_frame received None frame")
            return None
        
        if not hasattr(frame, 'copy') or not hasattr(frame, 'shape'):
            LOGGER.error("_annotate_frame received invalid frame type: %s", type(frame))
            return None
            
        if len(frame.shape) < 2:
            LOGGER.error("_annotate_frame received invalid frame shape: %s", frame.shape)
            return None
            
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
    clips_processed = 0

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
        clips_processed += 1

        # Periodically clear GPU memory to prevent VRAM accumulation
        if clips_processed % 10 == 0:
            cleanup_gpu_memory()
            LOGGER.debug("GPU memory cleaned after %d clips", clips_processed)
    
    # Final GPU cleanup after batch processing
    cleanup_gpu_memory()

    # Summary
    successful = sum(1 for r in results if r.success)
    updated = sum(1 for r in results if r.new_path is not None)
    LOGGER.info(
        "Post-processing complete: %d/%d clips processed, %d updated",
        successful, len(results), updated
    )

    return results
