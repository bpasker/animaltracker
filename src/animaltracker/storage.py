"""Storage management for clips and logs."""
from __future__ import annotations

import logging
import os
import re
import glob
import json
import shutil
import subprocess
import threading
import time
import uuid
import cv2
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .species_names import get_common_name

LOGGER = logging.getLogger(__name__)

# An interval longer than this between two captured frames is a stall of the
# stream, not the camera's frame rate, and is left out of the measurement.
_MAX_FRAME_INTERVAL = 1.0


def measured_frame_rate(timestamps: Sequence[float]) -> Optional[float]:
    """Frames per second over a run of capture times; None if it cannot tell.

    Clips were stamped 15 fps whatever the camera delivered, so cam1's 20 fps
    played 1.33x slow and Otteson1's 24 fps 1.6x, and every time measured in
    clip seconds was stretched to match. The pre-roll buffer keeps each
    frame's capture time, which is the rate the clip's frames really arrived
    at. Stalls are left out; a rate within 2% of a whole number is snapped to
    it, since cameras deliver whole-number rates and read-time jitter is not
    worth putting in the file.
    """
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])
                 if 0 <= b - a <= _MAX_FRAME_INTERVAL]
    span = sum(intervals)
    if len(intervals) < 10 or span < 1.0:
        return None
    rate = len(intervals) / span
    whole = round(rate)
    if whole and abs(rate - whole) <= 0.02 * rate:
        rate = float(whole)
    rate = round(rate, 2)
    return rate if 1.0 <= rate <= 120.0 else None


class StreamingClipWriter:
    """Streams frames straight to a temp MJPG AVI as they arrive.

    Replaces the previous approach of buffering every event frame in a
    Python list (which could grow to ~28 GB per camera at 1080p15 with a
    300 s ``max_event_seconds``). The MJPG AVI is later transcoded to a
    browser-friendly MP4 by ``StorageManager.transcode_avi_to_mp4``.

    Encoding runs on a dedicated writer thread. ``seed()`` and ``write()``
    only queue frame references and return immediately, so the asyncio
    stream loop that calls them never waits on cv2: seeding a 10 s 1080p
    pre-roll inline used to block that loop -- and every camera worker
    gathered on it -- for ~5 s at each event start. The writer thread owns
    the ``cv2.VideoWriter`` (opened lazily from the first frame's size,
    released when the thread exits), so no two threads ever touch it.

    ``close()`` drains the queue and joins the thread; call it from an
    executor, not from the event loop.

    ``fps`` is the rate frames are offered at, the camera's capture rate.
    With ``stride`` N only every Nth frame is kept, seed and live alike on
    one count so the spacing holds across the seam, and the file is stamped
    ``fps / N``: a clip plays at the speed it was recorded either way. That
    is for an encoder that cannot keep up. Shedding frames at random when
    the backlog fills (``dropped_frames``) leaves gaps a truthful stamp
    would play fast.
    """

    def __init__(self, temp_path: Path, fps: float = 15, max_pending: int = 300,
                 stride: int = 1) -> None:
        self.temp_path = temp_path
        self.capture_fps = float(fps)
        self.stride = max(1, int(stride))
        self.fps = self.capture_fps / self.stride   # what the file is stamped
        # Bound on queued *live* frames. Seed frames are exempt: the
        # pre-roll is the point of the clip and its frames are already
        # alive in the rolling buffer. Only binds when the encoder is
        # slower than capture; each queued 1080p BGR frame is ~6 MB.
        self.max_pending = max(1, int(max_pending))
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None  # (width, height)
        self.frame_count: int = 0
        self.live_frames: int = 0      # live frames kept by the stride: queued or dropped
        self.dropped_frames: int = 0
        self._offered: int = 0         # every frame seen, seed and live, for the stride
        self.write_errors: int = 0
        self._failed: bool = False
        self._pending: deque[np.ndarray] = deque()
        self._cv = threading.Condition()
        self._closing = False
        self._thread = threading.Thread(
            target=self._run, name=f"clip-writer-{temp_path.stem}", daemon=True
        )
        self._thread.start()

    # -- producer side (event loop thread) ---------------------------------

    def _takes_next(self) -> bool:
        """Whether the stride keeps the next frame offered; call under the lock."""
        keep = self._offered % self.stride == 0
        self._offered += 1
        return keep

    def seed(self, frames: Iterable[np.ndarray]) -> int:
        """Queue pre-roll frames ahead of any live frame; returns how many."""
        queued = 0
        with self._cv:
            if self._closing:
                return 0
            for frame in frames:
                if frame is not None and self._takes_next():
                    self._pending.append(frame)
                    queued += 1
            if queued:
                self._cv.notify()
        return queued

    def write(self, frame: np.ndarray) -> None:
        """Queue one live frame.

        Sheds the frame (and counts it) rather than block the caller when
        the backlog exceeds ``max_pending``.
        """
        if frame is None:
            return
        with self._cv:
            if self._closing or not self._takes_next():
                return
            self.live_frames += 1
            if len(self._pending) >= self.max_pending:
                self.dropped_frames += 1
                return
            self._pending.append(frame)
            self._cv.notify()

    def close(self) -> Optional[Path]:
        """Drain queued frames, release the encoder and return the temp AVI path.

        Returns None on failure or if nothing was written. Blocks until the
        writer thread has exited, so run it in an executor.
        """
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(timeout=120.0)
        if self._thread.is_alive():
            LOGGER.error(
                "Streaming writer for %s did not finish draining within 120s; leaving temp file in place",
                self.temp_path,
            )
            return None
        if self.dropped_frames:
            LOGGER.warning(
                "Streaming writer for %s dropped %d of %d live frames: encoder fell behind "
                "capture at %.1f fps (max_pending=%d)",
                self.temp_path.name, self.dropped_frames, self.live_frames, self.fps, self.max_pending,
            )
        if self._failed or self.frame_count == 0:
            self.discard()
            return None
        if not self.temp_path.exists() or self.temp_path.stat().st_size == 0:
            return None
        return self.temp_path

    def discard(self) -> None:
        """Best-effort delete of the temp AVI."""
        try:
            if self.temp_path.exists():
                self.temp_path.unlink()
        except OSError:
            pass

    # -- writer thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._pending and not self._closing:
                        self._cv.wait()
                    if not self._pending:
                        break  # closing and fully drained
                    frame = self._pending.popleft()
                self._encode(frame)
        finally:
            self._release()

    def _encode(self, frame: np.ndarray) -> None:
        try:
            if not self._ensure_open(frame):
                return
            # Guard against camera resolution change mid-event
            h, w = frame.shape[:2]
            if self._size != (w, h):
                return
            self._writer.write(frame)
            self.frame_count += 1
        except Exception as e:
            # Keep draining so close() still completes and whatever was
            # written before the error survives to be transcoded.
            self.write_errors += 1
            log = LOGGER.warning if self.write_errors == 1 else LOGGER.debug
            log("Streaming writer error for %s (%d so far): %s",
                self.temp_path.name, self.write_errors, e)

    def _ensure_open(self, frame: np.ndarray) -> bool:
        if self._writer is not None:
            return True
        if self._failed:
            return False
        height, width = frame.shape[:2]
        self.temp_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(self.temp_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            LOGGER.error("Failed to open streaming MJPG writer for %s", self.temp_path)
            self._failed = True
            return False
        self._writer = writer
        self._size = (width, height)
        return True

    def _release(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:  # pragma: no cover
                LOGGER.warning("Error releasing streaming writer: %s", e)
            self._writer = None

# Default storage thresholds
DEFAULT_MIN_FREE_BYTES = 500 * 1024 * 1024  # 500 MB minimum free space
DEFAULT_MAX_UTILIZATION_PCT = 80  # Don't use more than 80% of disk


CLIP_SUFFIXES = (".mp4", ".avi", ".mkv")

# Half-finished work, not clips: the AVI an event streams to before its
# transcode, and the MP4 that transcode writes before it renames.
_INTERMEDIATE_SUFFIXES = (".temp.avi", ".tmp.mp4")


def _is_intermediate(path: Path) -> bool:
    return path.name.endswith(_INTERMEDIATE_SUFFIXES)

# The event epoch a clip's name starts with; the archive dates clips the same way.
_CLIP_EPOCH_RE = re.compile(r"^(\d{9,10})_")


def _is_companion(name: str) -> bool:
    """A key frame or a processing log: all the pipeline writes beside a clip."""
    return name.endswith(".log.json") or (name.endswith(".jpg") and "_thumb" in name)


@dataclass
class LeftoverGroup:
    """One event's key frames and log, still on disk after its clip went."""
    directory: Path
    camera: str
    epoch: int
    files: List[Path]
    size: int


@dataclass
class PruneReport:
    """What a retention pass did, or would do with ``dry_run``."""
    dry_run: bool = True
    examined: int = 0
    protected: int = 0            # younger than retention.min_days
    deleted: List[Path] = field(default_factory=list)
    deleted_set: set = field(default_factory=set)
    failed: List[Path] = field(default_factory=list)
    files_removed: int = 0        # clips plus their key frames and logs
    interrupted_seen: int = 0     # *.temp.avi with no clip beside them
    interrupted_removed: int = 0  # ...of those, the ones past max_days
    leftovers_seen: int = 0       # key frames and logs whose clip is gone, per event
    leftovers_removed: int = 0    # ...of those, the ones past max_days
    leftover_files: int = 0
    leftover_bytes: int = 0       # not in freed_bytes, which counts clips
    freed_bytes: int = 0
    by_camera: Dict[str, int] = field(default_factory=dict)
    oldest: Optional[float] = None
    newest: Optional[float] = None
    utilization_pct: Optional[float] = None
    still_over_ceiling: bool = False

    @property
    def kept(self) -> int:
        return self.examined - len(self.deleted)


# ``build_event_temp_avi`` names: <camera>_<event epoch>_<8 hex>.temp.avi. The
# camera id may itself contain underscores, so the pattern is anchored on the
# two fixed fields at the end.
_EVENT_TEMP_RE = re.compile(r"^(?P<camera>.+)_(?P<ts>\d{6,12})_(?P<tag>[0-9a-f]{8})\.temp\.avi$")


@dataclass
class StorageManager:
    storage_root: Path
    logs_root: Path
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    max_utilization_pct: int = DEFAULT_MAX_UTILIZATION_PCT

    def __post_init__(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        # Nothing under ``event_temp`` is touched here. Those files are the
        # only copy of an event until it is transcoded, and any process may
        # build a StorageManager on this config: the ``cleanup`` command did,
        # and with it deleted the running service's open recordings. What a
        # previous run left behind is the pipeline's to deal with, once, at
        # its own startup (``list_orphan_event_temps``).

    def build_clip_path(self, camera_id: str, species: str, event_ts: float, ext: str = "mp4") -> Path:
        ts = time.strftime("%Y/%m/%d", time.localtime(event_ts))
        directory = self.storage_root / "clips" / camera_id / ts
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{int(event_ts)}_{species}.{ext}"
        return directory / filename

    def build_thumbnail_path(self, clip_path: Path, species: str, index: int = 0) -> Path:
        """Build path for a detection thumbnail associated with a clip.
        
        Thumbnails are stored alongside clips with format: 
        {clip_name}_thumb_{species}.jpg (for index 0)
        {clip_name}_thumb_{species}_{index}.jpg (for index > 0)
        """
        # Create thumbnail filename based on clip name
        clip_stem = clip_path.stem  # e.g., "1766587074_animal+bird"
        if index == 0:
            thumb_filename = f"{clip_stem}_thumb_{species}.jpg"
        else:
            thumb_filename = f"{clip_stem}_thumb_{species}_{index}.jpg"
        return clip_path.parent / thumb_filename

    def save_detection_thumbnails(
        self, 
        clip_path: Path, 
        species_frames: dict,
        padding_percent: float = 0.25
    ) -> List[Path]:
        """Save detection thumbnails for each species detected in a clip.
        
        Thumbnails are cropped to the detection bounding box with padding.
        
        Args:
            clip_path: Path to the clip file
            species_frames: Dict mapping species names to a list of 
                (frame, confidence, bbox) tuples, where bbox is [x1, y1, x2, y2] or None.
                Multiple detections per species are supported.
            padding_percent: How much padding to add around the detection (0.25 = 25% of bbox size)
                
        Returns:
            List of saved thumbnail paths
        """
        saved_paths = []
        
        for species, detections in species_frames.items():
            # Handle both old format (single tuple) and new format (list of tuples)
            if not isinstance(detections, list):
                detections = [detections]
            
            for idx, detection in enumerate(detections):
                frame, confidence, bbox = detection
                if frame is None:
                    continue
                    
                thumb_path = self.build_thumbnail_path(clip_path, species, idx)
                
                try:
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Crop to detection area if bbox is available
                    if bbox:
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
                        
                        # Crop the frame to the detection area
                        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        
                        # Ensure minimum size (at least 50x50 pixels)
                        if cropped_frame.shape[0] >= 10 and cropped_frame.shape[1] >= 10:
                            frame_to_save = cropped_frame
                        else:
                            # Bbox too small, use full frame
                            frame_to_save = frame
                    else:
                        # No bbox, use full frame
                        frame_to_save = frame
                    
                    # Save thumbnail
                    cv2.imwrite(str(thumb_path), frame_to_save)
                    saved_paths.append(thumb_path)
                    LOGGER.info("Saved detection thumbnail: %s", thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save thumbnail for %s #%d: %s", species, idx, e, exc_info=True)
                
        return saved_paths

    def get_clip_thumbnails(self, clip_path: Path) -> List[dict]:
        """Get all thumbnails associated with a clip.
        
        Returns:
            List of dicts with 'path', 'species', 'rel_path', and optional 'track_index' for each thumbnail
        """
        thumbnails = []
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        
        # Look for thumbnails matching this clip
        for thumb_file in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            # Extract species from filename
            # Format: {timestamp}_{original_species}_thumb_{specific_species}.jpg
            # or: {timestamp}_{original_species}_thumb_{specific_species}_{index}.jpg
            # or: {timestamp}_{original_species}_thumb_{specific_species}_t{track_index}.jpg (new format)
            parts = thumb_file.stem.split("_thumb_")
            if len(parts) >= 2:
                species_part = parts[-1]
                track_index = None
                
                # Check for track index suffix (e.g., "corvidae_t0" or "corvidae_t1")
                track_match = re.match(r'^(.+?)_t(\d+)$', species_part)
                if track_match:
                    species_name = track_match.group(1)
                    track_index = int(track_match.group(2))
                    species = get_common_name(species_name)
                else:
                    # Check if there's a legacy index suffix (e.g., "cardinal_1" or "cardinal_2")
                    match = re.match(r'^(.+?)(?:_(\d+))?$', species_part)
                    if match:
                        species_name = match.group(1)
                        detection_num = match.group(2)
                        species = get_common_name(species_name)
                        if detection_num:
                            species = f"{species} #{int(detection_num) + 1}"
                    else:
                        species = get_common_name(species_part)
            else:
                species = "Unknown"
                track_index = None
            
            thumb_data = {
                'path': thumb_file,
                'species': species,
                'rel_path': thumb_file.relative_to(self.storage_root / "clips")
            }
            if track_index is not None:
                thumb_data['track_index'] = track_index
                
            thumbnails.append(thumb_data)
        
        # Sort by track_index if present, to maintain consistent order
        thumbnails.sort(key=lambda x: (x.get('track_index', 999), str(x['path'])))
        
        return thumbnails

    def save_snapshot(self, camera_id: str, frame) -> Path:
        import cv2
        path = self.logs_root / f"startup_{camera_id}.jpg"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Saved startup snapshot to %s", path)
        return path

    def build_event_temp_avi(self, camera_id: str, event_ts: float) -> Path:
        """Build a path for the streaming temp AVI for an in-progress event.

        Stored under ``logs_root/event_temp/`` so they sit on local fast
        storage (not the NFS clip mount) while the event is recording.
        Cleaned up after transcode.
        """
        tmp_dir = self.logs_root / "event_temp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Add a uuid suffix so a previous crashed event can't collide with a
        # new one from the same (camera, second).
        return tmp_dir / f"{camera_id}_{int(event_ts)}_{uuid.uuid4().hex[:8]}.temp.avi"

    def list_orphan_event_temps(self) -> List[Path]:
        """Streaming temp AVIs in ``event_temp``, oldest event first.

        Only meaningful before this process records anything: call it once
        at pipeline startup, when every file there belongs to a run that is
        gone. Later the directory also holds the live events' recordings.
        """
        tmp_dir = self.logs_root / "event_temp"
        if not tmp_dir.is_dir():
            return []

        def event_ts(path: Path) -> int:
            match = _EVENT_TEMP_RE.match(path.name)
            return int(match.group("ts")) if match else 0

        return sorted(tmp_dir.glob("*.temp.avi"), key=lambda path: (event_ts(path), path.name))

    def recover_orphan_event_temp(self, temp_avi: Path, label: str) -> Optional[Path]:
        """Save the clip of an event whose process died before transcoding it.

        A restart used to delete these files. They are the whole recording of
        an event that closed and was waiting for a post-processing slot, or
        of one still open when the service stopped, so the clip and its
        alert were lost with them. The name carries the camera and the
        event's start (``build_event_temp_avi``), which is all it takes to
        put the clip where the live path would have: ``label`` is the
        unclassified label, so the recovery sweep analyses it like any clip
        a restart interrupted.

        Returns the new clip, or None when there is nothing to save: the
        name is not ours, the file is empty, the event's clip already exists
        (the old process died between the rename and the unlink) or the
        recording cannot be read. The temp file is gone either way.
        """
        match = _EVENT_TEMP_RE.match(temp_avi.name)
        if match is None:
            LOGGER.warning("Removing event temp file with an unexpected name: %s", temp_avi)
            self._unlink_quietly(temp_avi)
            return None

        camera_id = match.group("camera")
        event_ts = int(match.group("ts"))
        clip_path = self.build_clip_path(camera_id, label, event_ts)
        saved = [
            existing for existing in clip_path.parent.glob(f"{event_ts}_*.mp4")
            if not existing.name.endswith(".tmp.mp4")
        ]
        if saved:
            LOGGER.info("Removing orphan event recording %s: its clip %s was already saved",
                        temp_avi.name, saved[0].name)
            self._unlink_quietly(temp_avi)
            return None

        LOGGER.info("Recovering the recording of an event the last run never saved: %s", temp_avi.name)
        if not self.transcode_avi_to_mp4(temp_avi, clip_path):
            LOGGER.warning("Orphan event recording %s could not be recovered", temp_avi.name)
            return None
        return clip_path

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:
            path.unlink()
        except OSError as e:
            LOGGER.warning("Failed to remove %s: %s", path, e)

    def transcode_avi_to_mp4(self, temp_avi: Path, output_path: Path) -> bool:
        """Transcode a finished MJPG AVI into a browser-friendly MP4.

        Counterpart to ``StreamingClipWriter``; the AVI is written
        frame-by-frame in the streaming loop, then this is invoked once at
        event close. Always deletes the temp AVI on the way out.

        Returns True on success.
        """
        if not temp_avi.exists() or temp_avi.stat().st_size == 0:
            LOGGER.error("Temp AVI missing or empty for %s", output_path)
            try:
                if temp_avi.exists():
                    temp_avi.unlink()
            except OSError:
                pass
            return False

        tmp_mp4 = output_path.with_suffix(".tmp.mp4")
        ok = False
        try:
            if shutil.which("ffmpeg") is not None:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", str(temp_avi),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-crf", "23",
                    str(tmp_mp4),
                ]
                LOGGER.info("Transcoding clip to %s", output_path)
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                        tmp_mp4.rename(output_path)
                        LOGGER.info(
                            "Saved clip %s (%d bytes)", output_path, output_path.stat().st_size
                        )
                        ok = True
                    else:
                        LOGGER.error("FFmpeg produced empty file for %s", output_path)
                except subprocess.CalledProcessError as e:
                    err_msg = e.stderr.decode() if e.stderr else str(e)
                    LOGGER.error("FFmpeg failed: %s", err_msg)
            else:
                LOGGER.warning(
                    "ffmpeg not found; falling back to OpenCV avc1 transcode for %s",
                    output_path,
                )
                cap = cv2.VideoCapture(str(temp_avi))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 15
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (width, height))
                if not out.isOpened():
                    LOGGER.error("Failed to open fallback VideoWriter for %s", tmp_mp4)
                    cap.release()
                else:
                    try:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)
                    finally:
                        cap.release()
                        out.release()
                    if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                        tmp_mp4.rename(output_path)
                        LOGGER.info("Saved clip %s (fallback encoding)", output_path)
                        ok = True
        finally:
            try:
                if temp_avi.exists():
                    temp_avi.unlink()
            except OSError as e:
                LOGGER.warning("Failed to remove temp AVI %s: %s", temp_avi, e)
            if not ok and tmp_mp4.exists():
                try:
                    tmp_mp4.unlink()
                except OSError:
                    pass
        return ok

    def write_clip(self, frames: List, output_path: Path, fps: float = 15) -> bool:
        """Encode ``(timestamp, frame)`` pairs to ``output_path``; True if it exists after.

        Two steps, for browser compatibility: an MJPG AVI, then an H.264 MP4
        through ffmpeg. ``fps`` is the rate the frames were captured at;
        a manual clip used to be stamped 15 like the old event clips and
        played slow motion for the same reason (see ``measured_frame_rate``).
        Space is checked, never made: freeing it is retention's job, with
        the ``min_days`` floor the settings page promises.
        """
        if not frames:
            LOGGER.warning("No frames available for clip %s; skipping", output_path)
            return False
        
        estimated_size = self.estimate_clip_size(frames, fps)
        if not self.ensure_space_for_clip(estimated_size):
            LOGGER.error("Skipping clip %s due to insufficient storage space", output_path)
            return False
            
        height, width = frames[0][1].shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 1. Write to temporary AVI using MJPG (fast, safe, widely supported by OpenCV)
        temp_avi = output_path.with_suffix(".temp.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))
        
        if not out.isOpened():
            LOGGER.error("Failed to open MJPG VideoWriter for %s", temp_avi)
            return False

        try:
            for _, frame in frames:
                out.write(frame)
        finally:
            out.release()
            
        if not temp_avi.exists() or temp_avi.stat().st_size == 0:
            LOGGER.error("Failed to create intermediate AVI %s", temp_avi)
            if temp_avi.exists(): temp_avi.unlink()
            return False

        # 2. Convert to browser-friendly MP4 (H.264 + YUV420p) using FFmpeg CLI
        # This avoids the 'malloc' crash from piping raw frames and ensures web compatibility
        tmp_mp4 = output_path.with_suffix(".tmp.mp4")
        
        # Check if ffmpeg is available
        if shutil.which("ffmpeg") is None:
            LOGGER.warning("ffmpeg not found; falling back to OpenCV H.264 encoding (may not play in all browsers)")
            # Fallback: Try to write directly with OpenCV 'avc1'
            # We reuse the frames we already have, or re-read the AVI? 
            # Re-reading AVI is safer than keeping frames in memory if they were large.
            # But here we already wrote them to AVI. Let's just try to convert AVI -> MP4 using OpenCV
            
            cap = cv2.VideoCapture(str(temp_avi))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (width, height))
            
            if not out.isOpened():
                LOGGER.error("Failed to open fallback VideoWriter")
                temp_avi.unlink()
                return False
                
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
            finally:
                cap.release()
                out.release()
                temp_avi.unlink()
                
            if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                tmp_mp4.rename(output_path)
                LOGGER.info("Saved clip %s (fallback encoding)", output_path)
            return output_path.exists()

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(temp_avi),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # Critical for browser playback
            "-preset", "veryfast",
            "-crf", "23",
            str(tmp_mp4)
        ]
        
        LOGGER.info("Transcoding clip to %s", output_path)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                tmp_mp4.rename(output_path)
                LOGGER.info("Saved clip %s (%d bytes)", output_path, output_path.stat().st_size)
            else:
                LOGGER.error("FFmpeg produced empty file for %s", output_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            err_msg = e.stderr.decode() if isinstance(e, subprocess.CalledProcessError) else str(e)
            LOGGER.error("FFmpeg failed: %s", err_msg)
        finally:
            # Cleanup intermediate file
            if temp_avi.exists():
                temp_avi.unlink()
            if tmp_mp4.exists(): # If rename failed
                tmp_mp4.unlink()
        return output_path.exists()

    def disk_usage_pct(self) -> float:
        stat = shutil.disk_usage(self.storage_root)
        if stat.total == 0:
            return 0.0
        return stat.used / stat.total * 100

    def clip_event_time(self, clip: Path, stat=None) -> float:
        """When the event began: the epoch in the clip's name, else its mtime.

        Pruning by mtime alone gets a clip's age wrong in both directions. The
        mtime is when the transcode finished, minutes after a long event; and
        a reanalysis rewrites the sidecar and the key frames without touching
        the video, so a clip's files disagree about their own age. The epoch
        in the name is the first detection and never moves. Same rule the
        archive uses to date a clip.
        """
        if stat is None:
            stat = clip.stat()
        match = _CLIP_EPOCH_RE.match(clip.name)
        if match:
            epoch = int(match.group(1))
            # Sanity-checked against the clock, not against the file: a clip
            # whose mtime disagrees with its name is exactly the case this
            # exists for. Only a prefix that cannot be a time at all (before
            # 2001, or in the future) falls back.
            if 1_000_000_000 <= epoch <= time.time() + 86400:
                return float(epoch)
        return float(stat.st_mtime)

    def clip_companions(self, clip: Path) -> List[Path]:
        """A clip's key frames and processing log, which go when it goes.

        Deleting the video alone leaves files the archive never lists (it
        lists videos) and nothing else prunes. Names come from the sidecar's
        own list as well as a directory listing, because on the NFS archive a
        listing can be stale while the files open fine.
        """
        stem = clip.stem
        found: List[Path] = []
        log = clip.with_suffix(".log.json")
        names = set()
        try:
            with open(log, "r") as fh:
                entries = json.load(fh).get("thumbnails")
            if isinstance(entries, list):
                for entry in entries:
                    name = entry.get("file") if isinstance(entry, dict) else None
                    if (isinstance(name, str) and name.startswith(stem + "_thumb_")
                            and name.endswith(".jpg") and Path(name).name == name):
                        names.add(name)
        except (OSError, ValueError, AttributeError):
            pass
        try:
            names.update(p.name for p in clip.parent.glob(f"{glob.escape(stem)}_thumb*.jpg"))
        except OSError:
            pass
        found.extend(clip.parent / name for name in sorted(names))
        found.append(log)
        return found

    def find_clips(self) -> List[Path]:
        """Every saved clip, wherever it sits under ``clips/``.

        ``clips/<camera>/<YYYY>/<MM>/<DD>/<file>``, plus manual clips in the
        root. The old glob stopped one level short and matched the day
        directories, so nothing it returned was ever a file and retention has
        deleted nothing since the first commit.
        """
        root = self.storage_root / "clips"
        if not root.is_dir():
            return []
        return [
            p for p in root.rglob("*")
            if p.suffix.lower() in CLIP_SUFFIXES and not _is_intermediate(p) and p.is_file()
        ]

    def find_interrupted_recordings(self) -> List[Path]:
        """Streaming temp files left in the archive by an event that never finished.

        A ``.temp.avi`` with no MP4 beside it is the only copy of an event
        whose transcode did not complete. One from a recent crash may still
        be worth turning into a clip, so it is kept and reported rather than
        swept away; one already past ``max_days`` is as stale as any clip of
        its age and goes with them, logged for what it is.
        """
        root = self.storage_root / "clips"
        if not root.is_dir():
            return []
        return sorted(
            p for p in root.rglob("*")
            if _is_intermediate(p) and p.is_file()
            and not p.with_name(p.name.split(".")[0] + ".mp4").exists()
        )

    def find_leftovers(self) -> List[LeftoverGroup]:
        """Key frames and processing logs whose clip is gone, one group per event.

        A delete that removed only the video left these behind (the web
        delete did, until e236509), and nothing else ever finds them: the
        archive and retention both start from the videos. On production they
        came to 954 events, 4,370 files.

        A clip and everything written for it share a directory and the event
        epoch their names begin with, but not always the label after it: the
        post-processor writes key frames and the log under the name the clip
        is about to take and renames the video last, and a rename used to
        leave the old name's log behind. So files are grouped by directory
        and epoch, never by name, and a group is a leftover only when nothing
        in it is a video, finished or half-written, and its event is not
        still recording to ``event_temp``. A group holding any file the
        pipeline does not write beside a clip is not ours and is left alone.
        """
        root = self.storage_root / "clips"
        if not root.is_dir():
            return []
        groups: Dict[tuple, dict] = {}
        for path in root.rglob("*"):
            if path.parent == root:
                continue  # manual clips, named without an event epoch
            match = _CLIP_EPOCH_RE.match(path.name)
            if not match:
                continue
            group = groups.setdefault((path.parent, int(match.group(1))),
                                      {"files": [], "size": 0, "held": False})
            if not _is_companion(path.name):
                # Decided by name, before any stat: a video that is being
                # renamed while this listing runs must still hold its group.
                group["held"] = True
                continue
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError:
                continue  # gone between the listing and now
            group["files"].append(path)
            group["size"] += size

        recording = self._events_recording()
        leftovers = []
        for (directory, epoch), group in groups.items():
            if group["held"] or not group["files"]:
                continue
            camera = directory.relative_to(root).parts[0]
            if (camera, epoch) in recording:
                continue
            leftovers.append(LeftoverGroup(directory, camera, epoch,
                                           sorted(group["files"]), group["size"]))
        leftovers.sort(key=lambda g: (g.epoch, str(g.directory)))
        return leftovers

    def _events_recording(self) -> set:
        """``(camera, epoch)`` of every event with a recording in ``event_temp``.

        Live or left by a crash, that recording becomes the event's clip (the
        pipeline recovers orphans at startup), so the event's key frames are
        not leftovers though its video has not been written yet. Unlike
        ``list_orphan_event_temps`` this is safe to ask at any time.
        """
        events = set()
        try:
            names = [p.name for p in (self.logs_root / "event_temp").glob("*.temp.avi")]
        except OSError:
            return events
        for name in names:
            match = _EVENT_TEMP_RE.match(name)
            if match:
                events.add((match.group("camera"), int(match.group("ts"))))
        return events

    def prune_clips(
        self,
        max_days: int,
        min_days: int = 0,
        max_utilization_pct: Optional[int] = None,
        dry_run: bool = True,
    ) -> "PruneReport":
        """Remove clips past ``max_days``, then more if the disk is over its ceiling.

        Never touches a clip younger than ``min_days``, whatever the disk is
        doing: that floor is the one guarantee the settings page makes, and it
        is what stops a full disk erasing this morning's visitors. Each clip
        goes with its key frames and its log, and key frames and logs whose
        clip is already gone (``find_leftovers``) go once they pass the same
        age. Nothing is removed when ``dry_run`` is set, which is how the
        report is meant to be read first.
        """
        report = PruneReport(dry_run=dry_run)
        now = time.time()
        if min_days > 0 and max_days > 0 and min_days > max_days:
            LOGGER.warning(
                "retention.min_days (%d) is past retention.max_days (%d); keeping %d days",
                min_days, max_days, min_days,
            )
        keep_after = now - min_days * 86400 if min_days > 0 else None
        too_old = now - max_days * 86400 if max_days > 0 else None

        interrupted = set(self.find_interrupted_recordings())
        # Listed before anything is removed, so the dry run reports exactly
        # what the real run removes: a clip pruned below cannot turn a log it
        # left under an old name into a leftover halfway through this pass.
        # That log is one on the next.
        leftovers = self.find_leftovers()
        aged: List[tuple] = []
        for clip in list(self.find_clips()) + sorted(interrupted):
            try:
                stat = clip.stat()
            except OSError:
                continue
            if clip in interrupted:
                # An intermediate is aged by when it was last written, never by
                # the epoch in its name. A transcode running right now may well
                # be finishing a months-old event (the recovery sweep does
                # exactly that), and dating it by the event would let a
                # concurrent pass delete the file mid-write.
                started = float(stat.st_mtime)
                report.interrupted_seen += 1
            else:
                started = self.clip_event_time(clip, stat)
                report.examined += 1
            if keep_after is not None and started >= keep_after:
                report.protected += 1
                continue
            aged.append((started, clip, stat.st_size))
        aged.sort(key=lambda row: row[0])

        for started, clip, size in aged:
            if too_old is None or started >= too_old:
                continue
            if clip in interrupted:
                report.interrupted_removed += 1
                self._prune_one(clip, started, size, report, dry_run,
                                "interrupted recording, never finished")
            else:
                self._prune_one(clip, started, size, report, dry_run,
                                "older than %d days" % max_days)

        # Aged by the event, like the clip they belonged to. Not in the disk
        # pass below: they are small, and a full disk is no reason to hurry
        # files that go at max_days anyway.
        for group in leftovers:
            report.leftovers_seen += 1
            if not 1_000_000_000 <= group.epoch <= now + 86400:
                continue  # not a time, so no age to judge it by
            started = float(group.epoch)
            if keep_after is not None and started >= keep_after:
                continue
            if too_old is None or started >= too_old:
                continue
            self._prune_leftover(group, report, dry_run)

        if max_utilization_pct:
            def assumed_freed() -> int:
                return report.freed_bytes + report.leftover_bytes if dry_run else 0

            remaining = [row for row in aged if row[1] not in report.deleted_set]
            for started, clip, size in remaining:
                used_pct = self._utilization_pct(assumed_freed())
                if used_pct is None or used_pct <= max_utilization_pct:
                    break
                self._prune_one(clip, started, size, report, dry_run,
                                "disk at %.0f%%, ceiling %d%%" % (used_pct, max_utilization_pct))
            report.utilization_pct = self._utilization_pct(assumed_freed())
            if (report.utilization_pct is not None
                    and report.utilization_pct > max_utilization_pct):
                report.still_over_ceiling = True
        return report

    def _prune_one(self, clip: Path, started: float, size: int,
                   report: "PruneReport", dry_run: bool, reason: str) -> None:
        files = [clip] + self.clip_companions(clip)
        freed = 0
        for path in files:
            try:
                freed += path.stat().st_size
            except OSError:
                continue
        LOGGER.info("%s %s (%s)", "Would remove" if dry_run else "Removing", clip.name, reason)
        if not dry_run:
            for path in files:
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    LOGGER.warning("Failed to remove %s: %s", path.name, e)
                    report.failed.append(path)
        report.deleted.append(clip)
        report.deleted_set.add(clip)
        report.freed_bytes += freed
        report.files_removed += len(files)
        camera = clip.parent.parts[-4] if len(clip.parent.parts) >= 4 else "manual"
        report.by_camera[camera] = report.by_camera.get(camera, 0) + 1
        if report.oldest is None or started < report.oldest:
            report.oldest = started
        if report.newest is None or started > report.newest:
            report.newest = started

    def _prune_leftover(self, group: LeftoverGroup, report: "PruneReport", dry_run: bool) -> None:
        where = group.directory.relative_to(self.storage_root / "clips")
        LOGGER.info("%s %d file(s) of %s/%d_* (clip already gone)",
                    "Would remove" if dry_run else "Removing", len(group.files), where, group.epoch)
        if not dry_run:
            for path in group.files:
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    LOGGER.warning("Failed to remove %s: %s", path.name, e)
                    report.failed.append(path)
        report.leftovers_removed += 1
        report.leftover_files += len(group.files)
        report.leftover_bytes += group.size

    def _utilization_pct(self, assume_freed: int = 0) -> Optional[float]:
        try:
            stat = shutil.disk_usage(self.storage_root)
        except OSError:
            return None
        used = max(0, stat.used - assume_freed)
        return (used / stat.total) * 100 if stat.total else None

    def cleanup(self, retention_days: int, dry_run: bool = False) -> list[Path]:
        """Back-compatible wrapper: prune by age alone."""
        return self.prune_clips(retention_days, dry_run=dry_run).deleted

    def get_free_space(self) -> int:
        """Get free space in bytes on the storage volume."""
        stat = shutil.disk_usage(self.storage_root)
        return stat.free

    def get_total_space(self) -> int:
        """Get total space in bytes on the storage volume."""
        stat = shutil.disk_usage(self.storage_root)
        return stat.total

    def estimate_clip_size(self, frames: List, fps: int = 15) -> int:
        """Estimate the size of a clip based on frame count and resolution.
        
        Uses empirical estimates for H.264 compression ratios.
        Returns estimated size in bytes.
        """
        if not frames:
            return 0
        
        # Get frame dimensions from first frame
        height, width = frames[0][1].shape[:2]
        frame_count = len(frames)
        
        # Estimate bytes per frame for H.264 at CRF 23 (medium quality)
        # Rough estimate: 0.1 bits per pixel for decent quality H.264
        bits_per_pixel = 0.1
        bits_per_frame = width * height * bits_per_pixel
        bytes_per_frame = bits_per_frame / 8
        
        # Add 20% overhead for container, headers, etc.
        estimated_size = int(frame_count * bytes_per_frame * 1.2)
        
        # Minimum estimate of 100KB
        return max(estimated_size, 100 * 1024)

    def has_sufficient_space(self, required_bytes: int) -> bool:
        """Check if there's enough free space for a new clip.
        
        Considers both absolute free space and utilization percentage.
        """
        stat = shutil.disk_usage(self.storage_root)
        
        # Check absolute free space
        if stat.free < self.min_free_bytes + required_bytes:
            return False
        
        # Check utilization percentage after writing
        used_after = stat.used + required_bytes
        utilization_after = (used_after / stat.total) * 100
        if utilization_after > self.max_utilization_pct:
            return False
        
        return True

    def ensure_space_for_clip(self, required_bytes: int) -> bool:
        """Whether a clip of ``required_bytes`` fits; it frees nothing.

        This used to delete the oldest clips until the new one fit, with no
        regard for ``retention.min_days``, taking the video alone and leaving
        its key frames and log. Its glob was one level short of the archive
        layout, so it never actually found one; had it, a full disk would
        have eaten this morning's visitors to save thirty seconds of the
        live view. Making room is retention's job (``prune_clips``, daily),
        with the floor the settings page promises.
        """
        if self.has_sufficient_space(required_bytes):
            return True
        LOGGER.error(
            "Not enough space for a %.1f MB clip: %.1f MB free, %.0f MB must stay free "
            "and the disk may not pass %d%%. Retention runs daily; run `cleanup` to make room now.",
            required_bytes / 1_048_576, self.get_free_space() / 1_048_576,
            self.min_free_bytes / 1_048_576, self.max_utilization_pct,
        )
        return False
