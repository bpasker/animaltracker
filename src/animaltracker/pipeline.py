"""Streaming pipeline orchestration."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np

from .camera_registry import CameraRegistry
from .clip_buffer import ClipBuffer
from .config import CameraConfig, RuntimeConfig
from .detector import Detection, BaseDetector, create_detector, create_realtime_detector, create_postprocess_detector, cleanup_gpu_memory
from .notification import NotificationContext, PushoverNotifier
from .storage import StorageManager
from .onvif_client import OnvifClient
from .tracker import ObjectTracker, create_tracker  # noqa: F401
from .ptz_tracker import PTZTracker, create_ptz_tracker
from .web import WebServer

LOGGER = logging.getLogger(__name__)


# Serializes setting OPENCV_FFMPEG_CAPTURE_OPTIONS (process-wide env var) and
# the immediately-following cv2.VideoCapture() call so concurrent worker
# starts with different transport/hwaccel options can't race each other.
_CAPTURE_OPEN_LOCK = threading.Lock()


def build_ffmpeg_uri(rtsp_uri: str, transport: str = "tcp", hwaccel: bool = False) -> str:
    """Build RTSP URI with FFmpeg env options for OpenCV CAP_FFMPEG backend.
    
    Args:
        rtsp_uri: RTSP stream URL
        transport: tcp or udp
        hwaccel: If True, enable CUDA hardware decoding (requires FFmpeg with CUDA support)
    """
    import os
    
    if hwaccel:
        # Enable CUDA hardware decoding via FFmpeg
        # Format: "key1;value1|key2;value2"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|"
            f"stimeout;5000000|"            # 5s socket timeout (was 30s default)
            f"timeout;5000000|"             # 5s read timeout
            f"reconnect;1|"                 # auto-reconnect on EOF/error
            f"reconnect_streamed;1|"        # reconnect for streamed media
            f"reconnect_delay_max;2|"       # max 2s between reconnect attempts
            f"buffer_size;1048576|"         # 1MB UDP socket buffer
            f"hwaccel;cuda|"
            f"hwaccel_output_format;cuda"
        )
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|"
            f"stimeout;5000000|"
            f"timeout;5000000|"
            f"reconnect;1|"
            f"reconnect_streamed;1|"
            f"reconnect_delay_max;2|"
            f"buffer_size;1048576"
        )

    return rtsp_uri


# GStreamer pipelines for systems with NVIDIA GPU hardware decoding
def build_gstreamer_pipeline(rtsp_uri: str, transport: str = "tcp", latency_ms: int = 0) -> str:
    """Build GStreamer pipeline string for NVDEC hardware decoding.
    
    Uses NVIDIA's nvv4l2decoder for hardware H264/H265 decoding on GTX/RTX GPUs.
    Falls back to nvdec if nvv4l2decoder not available.
    """
    protocols = "tcp" if transport == "tcp" else "udp"
    latency = max(latency_ms, 100)  # Minimum 100ms for stable streaming
    
    # GStreamer pipeline for NVDEC hardware decoding
    # Works with GTX 1080, RTX series, and other NVIDIA GPUs
    pipeline = (
        f"rtspsrc location={rtsp_uri} protocols={protocols} latency={latency} ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! "
        f"video/x-raw,format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=0"
    )
    return pipeline


def build_gstreamer_pipeline_nvdec(rtsp_uri: str, transport: str = "tcp", latency_ms: int = 0) -> str:
    """Alternative GStreamer pipeline using nvdec (for desktop GPUs without nvv4l2decoder)."""
    protocols = "tcp" if transport == "tcp" else "udp"
    latency = max(latency_ms, 100)
    
    pipeline = (
        f"rtspsrc location={rtsp_uri} protocols={protocols} latency={latency} ! "
        f"rtph264depay ! h264parse ! "
        f"nvdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=0"
    )
    return pipeline


# Maximum number of key frames to keep per species
MAX_KEY_FRAMES_PER_SPECIES = 3


@dataclass
class EventState:
    camera: CameraConfig
    start_ts: float
    species: set[str]
    max_confidence: float
    last_detection_ts: float
    frames: List[tuple[float, np.ndarray]] = field(default_factory=list)
    # Track top N detection frames for each species (for thumbnails)
    # species -> list of (frame, confidence, bbox) tuples, sorted by confidence desc
    species_key_frames: dict = field(default_factory=dict)
    # Object tracker for this event
    tracker: Optional[ObjectTracker] = None

    def update(self, detections: List[Detection], frame_ts: float, frame: np.ndarray, frame_idx: Optional[int] = None) -> None:
        self.last_detection_ts = frame_ts

        # NOTE: The ObjectTracker is now driven by StreamWorker._process_frame
        # BEFORE the PTZ tracker call, so detections already carry track_ids
        # by the time we get here. We only consume tracker state for species
        # accumulation; we never call tracker.update() here.
        if self.tracker is not None:
            for det in detections:
                tid = getattr(det, 'track_id', None)
                if tid is None:
                    continue
                best_species, best_conf, _ = self.tracker.get_track_species(tid)
                if best_species:
                    self.species.add(best_species)
                    if best_conf > self.max_confidence:
                        self.max_confidence = best_conf
        else:
            # Fallback: use detections directly (no tracking)
            for det in detections:
                self.species.add(det.species)
                if det.confidence > self.max_confidence:
                    self.max_confidence = det.confidence
        
        # Track top N frames for each species (highest confidence)
        for det in detections:
            if det.species not in self.species_key_frames:
                self.species_key_frames[det.species] = []
            
            frames_list = self.species_key_frames[det.species]
            
            # Check if this detection is better than what we have
            # or if we have room for more
            min_confidence = min((f[1] for f in frames_list), default=0.0)
            
            if len(frames_list) < MAX_KEY_FRAMES_PER_SPECIES or det.confidence > min_confidence:
                # Add this detection
                frames_list.append((
                    frame.copy(),
                    det.confidence,
                    det.bbox
                ))
                
                # Sort by confidence descending and keep only top N
                frames_list.sort(key=lambda x: x[1], reverse=True)
                self.species_key_frames[det.species] = frames_list[:MAX_KEY_FRAMES_PER_SPECIES]
        
        # Frames are now appended in the main loop to ensure full framerate
        # self.frames.append((frame_ts, frame))

    def get_tracked_species_label(self) -> str:
        """Get species label using tracked object classifications."""
        if self.tracker:
            unique_species = self.tracker.get_unique_species()
            if unique_species:
                # Sort by confidence descending
                unique_species.sort(key=lambda x: x[1], reverse=True)
                return "+".join(s[0] for s in unique_species)
        return self.species_label
    
    def get_tracked_key_frames(self) -> dict:
        """Get key frames using tracked object data."""
        if self.tracker:
            tracked_data = self.tracker.get_all_species()
            key_frames = {}
            for track_id, data in tracked_data.items():
                species = data['species']
                if species and data.get('best_frame') is not None:
                    if species not in key_frames:
                        key_frames[species] = []
                    key_frames[species].append((
                        data['best_frame'],
                        data['confidence'],
                        data.get('best_bbox'),
                    ))
            # Sort each species by confidence
            for species in key_frames:
                key_frames[species].sort(key=lambda x: x[1], reverse=True)
                key_frames[species] = key_frames[species][:MAX_KEY_FRAMES_PER_SPECIES]
            return key_frames
        return self.species_key_frames

    @property
    def species_label(self) -> str:
        return "+".join(sorted(self.species))

    @property
    def duration(self) -> float:
        return self.last_detection_ts - self.start_ts


class StreamWorker:
    # Class-level semaphore to limit concurrent post-processing across ALL cameras
    # This prevents RAM explosion when multiple clips finish simultaneously
    # Initialized by PipelineOrchestrator based on config (default: 2 concurrent jobs
    # so two cameras finishing simultaneously don't serialize on the GPU).
    _postprocess_semaphore: threading.Semaphore = None
    _postprocess_limit: int = 2

    @classmethod
    def set_postprocess_limit(cls, limit: int) -> None:
        """Set the maximum concurrent post-processing jobs (called by PipelineOrchestrator)."""
        cls._postprocess_limit = max(1, min(8, limit))
        cls._postprocess_semaphore = threading.Semaphore(cls._postprocess_limit)
        LOGGER.info("Post-processing concurrency limit set to %d", cls._postprocess_limit)

    @classmethod
    def _ensure_postprocess_semaphore(cls) -> threading.Semaphore:
        """Ensure semaphore is initialized (fallback if not set by orchestrator)."""
        if cls._postprocess_semaphore is None:
            cls._postprocess_semaphore = threading.Semaphore(cls._postprocess_limit)
        return cls._postprocess_semaphore

    def __init__(
        self,
        camera: CameraConfig,
        runtime: RuntimeConfig,
        detector: BaseDetector,
        notifier: PushoverNotifier,
        storage: StorageManager,
        tracking_enabled: bool = True,
    ) -> None:
        self.camera = camera
        self.runtime = runtime
        self.detector = detector
        self.notifier = notifier
        self.storage = storage
        self.tracking_enabled = tracking_enabled
        # Ensure buffer is at least 30s for manual clips
        clip_seconds = max(30.0, runtime.general.clip.pre_seconds + runtime.general.clip.post_seconds)
        self.clip_buffer = ClipBuffer(max_seconds=clip_seconds, fps=15)
        self.event_state: Optional[EventState] = None
        self.pending_detection_start_ts: Optional[float] = None
        self.pending_detection_count: int = 0  # Consecutive frames with detections
        self.pending_detection_gap: int = 0  # Frames without detection during pending period
        self._snapshot_taken = False
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_ts: float = 0.0  # Wall-clock time the latest frame was received
        self.stream_connected: bool = False  # True when an RTSP capture is currently open
        self.latest_detections: List[Detection] = []  # Current detections for live view overlay
        self.latest_detection_ts: float = 0.0  # Timestamp of latest detections
        self.latest_frame_size: tuple = (0, 0)  # (width, height) of latest frame

        # Initialize ONVIF client if configured (with timeout to prevent blocking)
        self.onvif_client: Optional[OnvifClient] = None
        self.onvif_profile_token: Optional[str] = None
        self.ptz_tracker: Optional[PTZTracker] = None

        # Multi-camera PTZ tracking: shared detection cache for cameras that contribute
        # to the same PTZ tracker. Maps camera_id -> worker (for accessing detections)
        self._ptz_detection_sources: Optional[Dict[str, 'StreamWorker']] = None
        self._ptz_source_camera_id: Optional[str] = None  # Source camera (wide-angle)
        self._ptz_target_camera_id: Optional[str] = None  # Target camera (PTZ/zoom)

        # Whether this worker should drive the PTZ tracker (call .update on it).
        # When a tracker is shared across workers for *logging only*, the
        # non-driving worker must not push its own (different-camera-frame)
        # detections into the tracker -- doing so corrupts the tracker's lock
        # state, smoothing, and pixel->PTZ math.
        self.ptz_drives_tracking: bool = False

        # Persistent per-worker ObjectTracker. We run this BEFORE the PTZ
        # tracker each frame so detections carry stable track_ids by the time
        # PTZ lock-persistence logic looks at them. (Previously the tracker
        # was only created inside an event and ran AFTER the PTZ tracker, so
        # PTZ never saw track_ids on the same frame and lock persistence was
        # structurally broken.)
        self.tracker: Optional[ObjectTracker] = create_tracker(
            enabled=tracking_enabled, frame_rate=15
        )

        # Cached post-process detector (lazy initialization to avoid loading if not needed)
        # This prevents creating a new SpeciesNet model for every clip (VRAM leak fix)
        self._postprocess_detector: Optional[BaseDetector] = None
        self._postprocess_detector_lock = threading.Lock()
        
        if camera.onvif and camera.onvif.host:
            user, password = camera.onvif.credentials()
            if user and password:
                try:
                    import concurrent.futures
                    
                    def init_onvif():
                        client = OnvifClient(
                            host=camera.onvif.host,
                            port=camera.onvif.port,
                            username=user,
                            password=password
                        )
                        profiles = client.get_profiles()
                        return client, profiles
                    
                    # Use ThreadPoolExecutor with timeout to prevent blocking
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(init_onvif)
                        try:
                            self.onvif_client, profiles = future.result(timeout=10)  # 10 second timeout
                        except concurrent.futures.TimeoutError:
                            LOGGER.warning(f"ONVIF initialization timed out for {camera.id} (10s)")
                            profiles = []
                    
                    if profiles:
                        available_tokens = [p.metadata.get('token') for p in profiles]
                        LOGGER.info(f"ONVIF {camera.id}: Available profiles: {available_tokens}")
                        
                        # Try to find the profile specified in config
                        target_profile = camera.onvif.profile
                        if target_profile:
                            for p in profiles:
                                token = p.metadata.get("token", "")
                                # Match by token or by checking if profile name contains target
                                if token == target_profile or target_profile.lower() in token.lower():
                                    self.onvif_profile_token = token
                                    LOGGER.info(f"ONVIF {camera.id}: Using profile '{token}' (matched '{target_profile}')")
                                    break
                        
                        # Fallback to first profile if no match
                        if not self.onvif_profile_token:
                            self.onvif_profile_token = profiles[0].metadata.get("token")
                            LOGGER.info(f"ONVIF {camera.id}: Using first profile '{self.onvif_profile_token}'")
                except Exception as e:
                    LOGGER.warning(f"Failed to initialize ONVIF for {camera.id}: {e}")

    def _get_postprocess_detector(self) -> BaseDetector:
        """Get or create the cached post-process detector (thread-safe).

        This lazily initializes a SpeciesNet detector for post-processing clips.
        The detector is cached to avoid loading a new model for every clip,
        which would cause VRAM accumulation/leak on GPU.
        """
        if self._postprocess_detector is None:
            with self._postprocess_detector_lock:
                # Double-check after acquiring lock
                if self._postprocess_detector is None:
                    detector_cfg = self.runtime.general.detector
                    LOGGER.info("Initializing cached post-process detector for %s", self.camera.id)
                    self._postprocess_detector = create_postprocess_detector(detector_cfg)
                    LOGGER.info("Post-process detector ready: %s", self._postprocess_detector.backend_name)
        return self._postprocess_detector

    def save_manual_clip(self) -> Optional[str]:
        """Save the last 30 seconds of video buffer as a manual clip."""
        frames = self.clip_buffer.dump()
        if not frames:
            return None
            
        # Filter for last 30 seconds
        now = time.time()
        cutoff = now - 30.0
        recent_frames = [f for f in frames if f[0] >= cutoff]
        
        if not recent_frames:
            return None
            
        filename = f"manual_{self.camera.id}_{int(now)}.mp4"
        path = self.storage.storage_root / "clips" / filename
        
        # Run in thread to avoid blocking loop
        threading.Thread(
            target=self.storage.write_clip,
            args=(recent_frames, path)
        ).start()
        
        return filename

    async def run(self, stop_event: asyncio.Event) -> None:
        LOGGER.info("Starting worker for %s", self.camera.id)
        
        # Hardware decoding via FFmpeg CUDA (works with standard OpenCV + FFmpeg with CUDA)
        use_hwaccel = self.camera.rtsp.hwaccel
        rtsp_uri = build_ffmpeg_uri(
            self.camera.rtsp.uri, 
            self.camera.rtsp.transport,
            hwaccel=use_hwaccel
        )
        capture_backend = cv2.CAP_FFMPEG
        
        if use_hwaccel:
            LOGGER.info("Using FFmpeg CUDA hardware decoding for %s", self.camera.id)
        else:
            LOGGER.info("Using FFmpeg software decoding for %s", self.camera.id)
        
        # Frame skipping counter
        frame_count = 0
        skip_factor = max(1, self.camera.rtsp.frame_skip)  # Treat 0 as 1 (no skipping)
        
        # Track if inference is in progress (for non-blocking detection)
        inference_task: Optional[asyncio.Task] = None
        pending_frame: Optional[tuple[np.ndarray, float]] = None

        while not stop_event.is_set():
            loop = asyncio.get_running_loop()

            def _open_capture(uri: str, hw: bool) -> 'cv2.VideoCapture':
                # Hold the global lock across env-var write + VideoCapture
                # construction so a concurrent worker can't flip the env var
                # mid-open.
                with _CAPTURE_OPEN_LOCK:
                    resolved = build_ffmpeg_uri(
                        self.camera.rtsp.uri, self.camera.rtsp.transport, hwaccel=hw
                    )
                    return cv2.VideoCapture(resolved, capture_backend)

            cap = await loop.run_in_executor(None, _open_capture, rtsp_uri, use_hwaccel)

            if not cap.isOpened():
                if use_hwaccel:
                    # Fall back to software decoding if CUDA failed
                    LOGGER.warning("CUDA hardware decoding failed for %s, falling back to software", self.camera.id)
                    use_hwaccel = False
                    cap = await loop.run_in_executor(None, _open_capture, rtsp_uri, False)
                    if not cap.isOpened():
                        LOGGER.error("Unable to open RTSP stream for %s; retrying in 5s", self.camera.id)
                        self.stream_connected = False
                        await asyncio.sleep(5)
                        continue
                else:
                    LOGGER.error("Unable to open RTSP stream for %s; retrying in 5s", self.camera.id)
                    self.stream_connected = False
                    await asyncio.sleep(5)
                    continue

            LOGGER.info("Connected to stream for %s", self.camera.id)
            self.stream_connected = True
            try:
                while not stop_event.is_set():
                    # Offload blocking OpenCV read to thread to keep web server responsive
                    ret, frame = await loop.run_in_executor(None, cap.read)
                    
                    if not ret:
                        LOGGER.warning("Stream lost for %s; reconnecting...", self.camera.id)
                        self.stream_connected = False
                        break
                    
                    self.latest_frame = frame
                    self.latest_frame_ts = time.time()
                    
                    if not self._snapshot_taken:
                        # Offload snapshot saving to thread
                        loop.run_in_executor(None, self.storage.save_snapshot, self.camera.id, frame.copy())
                        self._snapshot_taken = True

                    frame_ts = time.time()
                    self.clip_buffer.push(frame_ts, frame)
                    
                    # Capture every frame for active events (even if inference is skipped).
                    # Copy because OpenCV may reuse the underlying buffer for the next read,
                    # and downstream consumers (clip writer, key-frame storage) read these
                    # frames asynchronously.
                    if self.event_state is not None:
                        self.event_state.frames.append((frame_ts, frame.copy()))

                        # Force-close events that exceed max duration (prevents memory leak)
                        max_duration = self.runtime.general.clip.max_event_seconds
                        event_elapsed = frame_ts - self.event_state.start_ts
                        if event_elapsed > max_duration:
                            LOGGER.warning(
                                "Event for %s exceeded max duration (%.1fs > %.1fs, %d frames) - force closing to prevent memory leak",
                                self.camera.id, event_elapsed, max_duration, len(self.event_state.frames)
                            )
                            await self._maybe_close_event(frame_ts, force=True)

                    # Skip inference on some frames to save CPU
                    frame_count += 1
                    if frame_count % skip_factor != 0:
                        continue

                    if self.camera.detect_enabled:
                        # Non-blocking inference: if previous inference is done, process result
                        # and start new one. If still running, drop this frame for inference.
                        if inference_task is not None:
                            if inference_task.done():
                                # Process completed inference result
                                try:
                                    await inference_task
                                except Exception as e:
                                    LOGGER.error("Inference error for %s: %s", self.camera.id, e)
                                inference_task = None
                            else:
                                # Still running - drop this frame for inference (but it's still buffered)
                                continue
                        
                        # Start new inference task
                        inference_task = asyncio.create_task(
                            self._process_frame(frame.copy(), frame_ts, frame_count)
                        )
            finally:
                # Cancel any pending inference
                if inference_task and not inference_task.done():
                    inference_task.cancel()
                # Offload release to thread
                await loop.run_in_executor(None, cap.release)
                self.stream_connected = False
            
            if not stop_event.is_set():
                await asyncio.sleep(1)  # Brief pause before reconnect

    @staticmethod
    def _compute_blur_score(frame: np.ndarray) -> float:
        """Compute Laplacian variance as a blur metric.
        
        Higher value = sharper image. Blurry/motion-blurred frames score low.
        Typical values:
        - Sharp outdoor scene: 200-1000+
        - Slightly blurry: 50-200
        - Heavy motion blur (PTZ moving): 5-50
        - Completely out of focus: <5
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    async def _process_frame(self, frame: np.ndarray, ts: float, frame_idx: int = 0) -> None:
        loop = asyncio.get_running_loop()

        # --- Blur detection: skip processing blurry frames (PTZ motion blur) ---
        blur_threshold = self.camera.thresholds.blur_threshold
        if blur_threshold > 0:
            blur_score = self._compute_blur_score(frame)
            if blur_score < blur_threshold:
                LOGGER.debug(
                    "[BLUR_SKIP] %s: frame too blurry (score=%.1f < threshold=%.1f), skipping detection",
                    self.camera.id, blur_score, blur_threshold
                )
                # Tick the tracker with no detections so its Kalman / lost-track
                # buffers stay aligned with wall-clock; otherwise a long blur
                # window desynchronizes ByteTrack from frame time and reacquired
                # tracks get fresh IDs (breaking the PTZ lock).
                if self.tracker is not None:
                    try:
                        await loop.run_in_executor(
                            None, self.tracker.update, [], frame, frame_idx,
                        )
                    except Exception:
                        pass
                await self._maybe_close_event(ts)
                return

        # --- PTZ settle delay: skip detections while camera is stabilizing ---
        # Only applies to the camera physically being moved by the PTZ tracker.
        # Without this guard, a wide-angle source camera that triggers PTZ
        # commands on a separate zoom camera would suppress its own detections
        # every time it issued a move, causing tracking to never engage.
        ptz_settle_time = self.camera.thresholds.ptz_settle_time
        if (
            ptz_settle_time > 0
            and self.ptz_tracker is not None
            and self.onvif_client is not None
            and self.ptz_tracker.onvif_client is self.onvif_client
        ):
            if self.ptz_tracker.is_settling(ptz_settle_time):
                LOGGER.debug(
                    "[PTZ_SETTLE] %s: PTZ still settling (moved %.2fs ago, need %.2fs), skipping detection",
                    self.camera.id,
                    ts - self.ptz_tracker.get_last_move_time(),
                    ptz_settle_time
                )
                # Invalidate this worker's published detections while we're
                # in the settle window. Otherwise other workers reading
                # `self.latest_detections` (multi-cam path) would treat the
                # pre-move detections as still-current after the camera has
                # physically moved, jerking the PTZ on stale data.
                self.latest_detections = []
                self.latest_detection_ts = ts
                # Tick the tracker on settle skips too -- same reasoning as
                # the blur-skip branch above.
                if self.tracker is not None:
                    try:
                        await loop.run_in_executor(
                            None, self.tracker.update, [], frame, frame_idx,
                        )
                    except Exception:
                        pass
                await self._maybe_close_event(ts)
                return

        detections = await loop.run_in_executor(
            None,
            lambda: self.detector.infer(
                frame,
                conf_threshold=self.camera.thresholds.confidence,
                generic_confidence=self.camera.thresholds.generic_confidence
            )
        )

        # Log raw detections from MegaDetector (before species filtering)
        if detections:
            det_summary = ', '.join(f"{d.species}:{d.confidence:.0%}" for d in detections[:3])
            if len(detections) > 3:
                det_summary += f" (+{len(detections)-3} more)"
            LOGGER.info(
                "[REALTIME] %s: %d raw detections: %s",
                self.camera.id, len(detections), det_summary
            )

        filtered = self._filter_detections(detections)
        
        # Filter small detections and leaf-like shapes BEFORE event triggering
        # This prevents false positives from triggering clip recording
        frame_h, frame_w = frame.shape[:2]
        filtered = self._filter_false_positives(filtered, frame_w, frame_h)

        # Log if detections were filtered out
        if detections and not filtered:
            LOGGER.debug(
                "[REALTIME] %s: %d detections filtered out by species rules",
                self.camera.id, len(detections)
            )
        elif detections and len(filtered) < len(detections):
            LOGGER.debug(
                "[REALTIME] %s: %d of %d detections passed species filter",
                self.camera.id, len(filtered), len(detections)
            )

        # Store for live view overlay
        frame_h, frame_w = frame.shape[:2]
        self.latest_frame_size = (frame_w, frame_h)

        # Run object tracker BEFORE publishing latest_detections so that any
        # other worker reading our list (multi-cam PTZ path) sees detections
        # whose track_id has already been stamped. Otherwise that worker can
        # observe a detection with track_id=None microseconds before our
        # executor stamps it -- breaking lock persistence on the receiver.
        # The tracker also needs to be ticked on empty frames so its internal
        # Kalman filter / lost-track buffer advance consistently with
        # wall-clock; ObjectTracker.update handles [].
        if self.tracker is not None:
            try:
                await loop.run_in_executor(
                    None,
                    self.tracker.update,
                    filtered,
                    frame,
                    frame_idx,
                )
            except Exception as e:
                LOGGER.warning("ObjectTracker.update failed for %s: %s", self.camera.id, e)

        self.latest_detections = filtered
        self.latest_detection_ts = ts

        # PTZ auto-tracking: always call update (handles patrol when no detections).
        # Only the worker designated as the tracker driver may push detections;
        # otherwise a second camera with a *different* coordinate space would
        # corrupt the tracker's lock state and pixel->PTZ math.
        if self.ptz_tracker and self.ptz_drives_tracking:
            # Debug: log what we're sending to PTZ tracker
            if filtered:
                LOGGER.info(
                    "PTZ update: %s sending %d detections (track=%s, patrol=%s, mode=%s)",
                    self.camera.id, len(filtered),
                    self.ptz_tracker.is_track_enabled(),
                    self.ptz_tracker.is_patrol_enabled(),
                    self.ptz_tracker.get_mode()
                )

            # Check if we should use multi-camera tracking
            if (self._ptz_detection_sources
                and self._ptz_source_camera_id
                and self._ptz_target_camera_id):
                # Build detection dict from all contributing cameras
                camera_detections = {}

                # Use a staleness window strictly less than the worker's
                # ptz_settle_time so a recently-moved camera's stale
                # detections cannot drive PTZ. (Settle gate already clears
                # latest_detections for the moved camera, but the window
                # protects the source camera too if its inference happens
                # to lag.) When settle is disabled (<=0), there is nothing
                # to protect against -- use a generous window so single
                # ~66ms inter-frame gaps don't spuriously discard fresh
                # detections.
                settle = self.camera.thresholds.ptz_settle_time
                if settle <= 0:
                    staleness_window = 0.5
                else:
                    staleness_window = max(0.1, min(0.5, settle * 0.5))

                for cam_id, worker in self._ptz_detection_sources.items():
                    detection_age = ts - worker.latest_detection_ts
                    if detection_age < staleness_window:
                        w, h = worker.latest_frame_size
                        if w > 0 and h > 0:
                            camera_detections[cam_id] = (
                                worker.latest_detections,
                                w,
                                h
                            )
                            LOGGER.debug(
                                "Multi-cam PTZ: %s contributing %d detections (age=%.3fs)",
                                cam_id, len(worker.latest_detections), detection_age
                            )
                    else:
                        LOGGER.debug(
                            "Multi-cam PTZ: %s detections too old (age=%.3fs > %.3fs)",
                            cam_id, detection_age, staleness_window
                        )

                # Log which cameras are contributing
                if camera_detections:
                    contrib = ', '.join(f"{k}:{len(v[0])}" for k, v in camera_detections.items())
                    LOGGER.debug("Multi-cam PTZ update from %s: %s", self.camera.id, contrib)
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                self.ptz_tracker.update_multi_camera,
                                camera_detections,
                                self._ptz_source_camera_id,
                                self._ptz_target_camera_id,
                            ),
                            timeout=3.0,
                        )
                    except asyncio.TimeoutError:
                        LOGGER.warning(
                            "[PTZ_TIMEOUT] update_multi_camera exceeded 3s on %s; skipping frame",
                            self.camera.id,
                        )
                else:
                    # No recent detections from any camera
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                self.ptz_tracker.update,
                                [], frame_w, frame_h,
                            ),
                            timeout=3.0,
                        )
                    except asyncio.TimeoutError:
                        LOGGER.warning(
                            "[PTZ_TIMEOUT] update([]) exceeded 3s on %s; skipping frame",
                            self.camera.id,
                        )
            else:
                # Single-camera tracking mode
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            self.ptz_tracker.update,
                            filtered, frame_w, frame_h,
                        ),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    LOGGER.warning(
                        "[PTZ_TIMEOUT] update exceeded 3s on %s; skipping frame",
                        self.camera.id,
                    )
            # Periodically trim old decisions to prevent unbounded memory growth
            # Keep last 5 minutes of decisions (enough for any reasonable clip)
            cutoff = ts - 300  # 5 minutes
            self.ptz_tracker.trim_old_decisions(cutoff)

        if not filtered:
            # Allow small gaps without resetting the pending counter.
            # Real animals may be briefly undetected (occlusion, motion blur)
            # but leaves tend to flicker on/off every other frame.
            if self.pending_detection_start_ts is not None:
                self.pending_detection_gap += 1
                # Only reset if gap exceeds tolerance (3 consecutive empty frames)
                if self.pending_detection_gap > 3:
                    self.pending_detection_start_ts = None
                    self.pending_detection_count = 0
                    self.pending_detection_gap = 0
            await self._maybe_close_event(ts)
            return
        
        # Reset gap counter on detection
        self.pending_detection_gap = 0
        
        # Use all filtered detections to update state
        primary = filtered[0]
        if self.event_state is None:
            # Check for minimum duration AND minimum frame count
            if self.pending_detection_start_ts is None:
                self.pending_detection_start_ts = ts
                self.pending_detection_count = 1
                return # Wait for more frames
            
            self.pending_detection_count += 1
            
            duration_met = (ts - self.pending_detection_start_ts) >= self.camera.thresholds.min_duration
            frames_met = self.pending_detection_count >= self.camera.thresholds.min_frames
            
            if not (duration_met and frames_met):
                return # Still waiting for both conditions
            
            # Duration AND frame count met, start event
            self.pending_detection_start_ts = None
            self.pending_detection_count = 0
            self.pending_detection_gap = 0
            LOGGER.info("Started tracking %s on %s (%.2f)", primary.species, self.camera.id, primary.confidence)
            
            # Create tracker for this event if enabled.
            # We share the persistent worker tracker so PTZ lock-persistence
            # and event-scoped species accumulation see the same track_ids.
            event_tracker = self.tracker if self.tracking_enabled else None
            if event_tracker:
                LOGGER.debug("Object tracking enabled for event on %s", self.camera.id)
            
            self.event_state = EventState(
                camera=self.camera,
                start_ts=ts,
                species={d.species for d in filtered},
                max_confidence=max(d.confidence for d in filtered),
                last_detection_ts=ts,
                tracker=event_tracker,
            )
            # Add pre-event frames from buffer, filtered by pre_seconds
            buffered = self.clip_buffer.dump()
            cutoff = ts - self.runtime.general.clip.pre_seconds
            self.event_state.frames.extend([f for f in buffered if f[0] >= cutoff])
            
        self.event_state.update(filtered, ts, frame, frame_idx=frame_idx)

    def _normalize_species(self, species: str) -> str:
        """Normalize species name for comparison (lowercase, underscores)."""
        return species.lower().replace(' ', '_').replace('-', '_').strip()

    def _species_matches_exclude(self, detection_species: str, excludes: set) -> bool:
        """Check if a detection species matches any exclude pattern.
        
        Supports:
        - Exact match: "mammalia_rodentia_sciuridae" matches "mammalia_rodentia_sciuridae"
        - Prefix match: "mammalia_rodentia_sciuridae_sciurus" matches "mammalia_rodentia_sciuridae"
        - Token match: "sciuridae" matches a label with token "sciuridae" (not arbitrary substrings)
        - Common name match: "mammalia_rodentia_sciuridae" matches "squirrel"
        """
        from .species_names import get_common_name
        
        normalized = self._normalize_species(detection_species)
        norm_tokens = normalized.split('_')
        
        for exclude in excludes:
            exclude_norm = self._normalize_species(exclude)
            
            # Exact match
            if normalized == exclude_norm:
                return True
            
            # Detection starts with exclude (hierarchical match)
            # e.g., "mammalia_rodentia_sciuridae_sciurus" starts with "mammalia_rodentia_sciuridae"
            if normalized.startswith(exclude_norm + '_'):
                return True
            
            # Exclude starts with detection (broader exclusion)
            # e.g., excluding "mammalia_rodentia" should exclude "mammalia_rodentia_sciuridae"
            if exclude_norm.startswith(normalized + '_'):
                return True
            
            # Token match (was substring; substring matched too aggressively,
            # e.g. excluding "bear" would also drop "bearded_dragon").
            if exclude_norm in norm_tokens:
                return True
        
        # Also check common name match
        common_name = get_common_name(detection_species).lower()
        if common_name in excludes:
            return True
        
        return False

    def _filter_detections(self, detections: List[Detection]) -> List[Detection]:
        includes = set(self._normalize_species(s) for s in self.camera.include_species)
        excludes = set(self._normalize_species(s) for s in self.camera.exclude_species)
        global_excludes = set(self._normalize_species(s) for s in self.runtime.general.exclusion_list)
        all_excludes = excludes | global_excludes

        def _matches_include(label: str, inc: str) -> bool:
            # Exact match
            if label == inc:
                return True
            # Hierarchical: label is a more-specific child of inc
            #   inc='mammalia_carnivora_ursidae' matches 'mammalia_carnivora_ursidae_ursus'
            if label.startswith(inc + '_'):
                return True
            # Component match: inc appears as a whole token of the label
            #   inc='bear' matches 'mammalia_carnivora_ursidae_bear' but NOT 'bearded_dragon'
            #   inc='deer' matches 'odocoileus_deer' but NOT 'musk_deer_meadow'-style false hits
            tokens = label.split('_')
            if inc in tokens:
                return True
            return False

        filtered: List[Detection] = []
        for det in detections:
            label = self._normalize_species(det.species)

            # Check includes (if specified, only allow listed species)
            if includes and not any(_matches_include(label, inc) for inc in includes):
                continue

            # Check excludes (skip if matches any exclude pattern)
            if all_excludes and self._species_matches_exclude(det.species, all_excludes):
                LOGGER.debug("Excluding detection: %s (matches exclude list)", det.species)
                continue

            filtered.append(det)
        return filtered

    def _filter_false_positives(
        self, detections: List[Detection], frame_width: int, frame_height: int
    ) -> List[Detection]:
        """Filter out detections that are likely false positives (leaves, branches, noise).
        
        Applies geometric filters that are model-agnostic:
        1. Minimum area: tiny detections are usually noise/leaves
        2. Extreme aspect ratio: very long/thin shapes are branches/leaves, not animals
        
        These filters run BEFORE event triggering to prevent false clips.
        """
        if not detections:
            return detections
        
        # Use threshold-level min_detection_area (applies to all cameras).
        min_area_frac = self.camera.thresholds.min_detection_area
        frame_area = frame_width * frame_height
        min_area_pixels = min_area_frac * frame_area
        
        # Animals have aspect ratios roughly between 1:4 and 4:1
        # Leaves/branches tend to be much thinner (1:8 or more)
        max_aspect_ratio = 5.0
        
        filtered = []
        for det in detections:
            bbox_w = det.bbox[2] - det.bbox[0]
            bbox_h = det.bbox[3] - det.bbox[1]
            det_area = bbox_w * bbox_h
            
            # Filter tiny detections
            if det_area < min_area_pixels:
                LOGGER.debug(
                    "[FP_FILTER] %s: too small (%.1fpx², %.2f%% of frame, min=%.2f%%)",
                    self.camera.id, det_area, (det_area / frame_area) * 100,
                    min_area_frac * 100
                )
                continue
            
            # Filter extreme aspect ratios (likely branches/leaves)
            if bbox_w > 0 and bbox_h > 0:
                aspect = max(bbox_w / bbox_h, bbox_h / bbox_w)
                if aspect > max_aspect_ratio:
                    LOGGER.debug(
                        "[FP_FILTER] %s: extreme aspect ratio %.1f (likely leaf/branch), "
                        "bbox=%.0fx%.0f, species=%s",
                        self.camera.id, aspect, bbox_w, bbox_h, det.species
                    )
                    continue
            
            filtered.append(det)
        
        if len(filtered) < len(detections):
            LOGGER.debug(
                "[FP_FILTER] %s: filtered %d/%d detections (size/shape)",
                self.camera.id, len(detections) - len(filtered), len(detections)
            )
        
        return filtered

    async def _maybe_close_event(self, ts: float, force: bool = False) -> None:
        if self.event_state is None:
            return
        idle = ts - self.event_state.last_detection_ts
        if not force and idle < self.runtime.general.clip.post_seconds:
            return
        
        # Offload clip writing, post-analysis, and notification to thread
        loop = asyncio.get_running_loop()
        post_analysis_enabled = self.runtime.general.clip.post_analysis
        post_analysis_frames = self.runtime.general.clip.post_analysis_frames
        # Check if unified post-processor should be used (new approach)
        use_unified_processor = getattr(self.runtime.general.clip, 'unified_post_processing', False)
        
        # Capture detector reference for unified processor
        # For split-model architecture, post-processor will create its own SpeciesNet detector
        detector = self.detector
        detector_cfg = self.runtime.general.detector  # Pass config so post-processor can create SpeciesNet
        storage_root = self.storage.storage_root
        
        # Capture web_base_url for notification links
        web_base_url = getattr(self.runtime.general.notification, 'web_base_url', None)
        
        # Capture PTZ decisions if tracker is active
        # Use time-windowed approach to avoid clearing decisions that other
        # cameras sharing this tracker may need
        ptz_log = None
        if self.ptz_tracker:
            event_start = self.event_state.start_ts
            event_end = ts  # Current time (event closing)
            window_start = event_start - self.runtime.general.clip.pre_seconds
            # Add buffer for pre-event decisions that may be relevant
            ptz_log = self.ptz_tracker.get_decisions_in_window(
                window_start,
                event_end
            )
            if ptz_log:
                LOGGER.info("Captured %d PTZ decisions for event (window: %.1fs to %.1fs)",
                            len(ptz_log), window_start, event_end)
            else:
                # Log why no decisions were captured - helps debug PTZ issues
                total_decisions = len(self.ptz_tracker._decision_log)
                # Show timestamps of available decisions if any
                if total_decisions > 0:
                    oldest = min(e.timestamp for e in self.ptz_tracker._decision_log)
                    newest = max(e.timestamp for e in self.ptz_tracker._decision_log)
                    LOGGER.info(
                        "No PTZ decisions in event window [%.1f - %.1f] for %s "
                        "(tracker has %d decisions from %.1f to %.1f - outside window)",
                        window_start, event_end, self.camera.id,
                        total_decisions, oldest, newest
                    )
                else:
                    LOGGER.info(
                        "No PTZ decisions in event window [%.1f - %.1f] for %s "
                        "(tracker has 0 total decisions, track_enabled=%s, mode=%s)",
                        window_start, event_end, self.camera.id,
                        self.ptz_tracker.is_track_enabled(),
                        self.ptz_tracker.get_mode()
                    )
        else:
            LOGGER.debug(
                "Event closing for %s but no PTZ tracker assigned",
                self.camera.id
            )
        
        # Capture exclusion lists for post-processing check
        # (species may be reclassified by post-processor to an excluded species)
        camera_excludes = set(self._normalize_species(s) for s in self.camera.exclude_species)
        global_excludes = set(self._normalize_species(s) for s in self.runtime.general.exclusion_list)
        all_excludes = camera_excludes | global_excludes
        
        # Capture reference to exclusion check method
        species_matches_exclude = self._species_matches_exclude
        
        def finalize_event(frames, camera_id, start_ts, clip_format, ctx_base, priority, sound, species_key_frames, ptz_decisions, detector_config):
            """Finalize event with optional post-clip species analysis.

            Split-model architecture:
            - Real-time: YOLO was used for fast detection/PTZ tracking
            - Post-processing: SpeciesNet is used for accurate species identification

            Uses class-level semaphore to limit concurrent post-processing and prevent
            RAM explosion when multiple clips finish simultaneously.
            """
            # Acquire semaphore to limit concurrent post-processing (prevents RAM explosion)
            # This will block if too many clips are already being processed
            with StreamWorker._ensure_postprocess_semaphore():
                LOGGER.debug("Post-processing started for %s (%d frames)",
                            camera_id, len(frames))

                final_species = ctx_base['species']
                final_confidence = ctx_base['confidence']
                tracks_count = 1  # Default assumption

                # Step 1: Run post-clip analysis if enabled (OLD approach - analyze in memory)
                if post_analysis_enabled and not use_unified_processor:
                    refined_species, refined_confidence = self._analyze_clip_frames(
                        frames, num_samples=post_analysis_frames
                    )
                    # Use refined species if we got a more specific identification
                    if refined_species:
                        final_species = refined_species
                        final_confidence = max(refined_confidence, ctx_base['confidence'])

                # Step 2: Build clip path with (potentially refined) species
                clip_path = self.storage.build_clip_path(
                    camera_id,
                    final_species,
                    start_ts,
                    clip_format,
                )

                # Step 3: Write the clip
                self.storage.write_clip(frames, clip_path)

                # Step 4: Save detection thumbnails for each species (skip if unified processor will regenerate)
                if species_key_frames and not use_unified_processor:
                    self.storage.save_detection_thumbnails(clip_path, species_key_frames)

                # MEMORY LEAK FIX: release the in-memory frame list (and key
                # frame buffer) as soon as everything that needs raw frames
                # has run. Subsequent steps (unified post-processor,
                # notification, log writing) all operate on the *saved* clip
                # file, not the in-memory frames. With max_event_seconds=300
                # at 15 fps these lists can hold ~4500 numpy frames
                # (~25 GB at 1080p), which previously stayed alive through
                # the entire SpeciesNet analysis under the post-process
                # semaphore -- pinning tens of GB of RSS for the duration of
                # post-processing and stacking up across concurrent jobs.
                frames_count_for_log = len(frames)
                frames = None
                species_key_frames = None
                import gc as _gc
                _gc.collect()
                LOGGER.debug(
                    "Released in-memory frames after clip write for %s (%d frames freed)",
                    camera_id, frames_count_for_log,
                )

                # Step 5: Run UNIFIED post-processor if enabled (NEW approach - analyze saved file)
                # Uses SpeciesNet for accurate species identification (split-model architecture)
                if use_unified_processor:
                    try:
                        from .postprocess import ClipPostProcessor, ProcessingSettings

                        # Use cached detector to avoid VRAM leak (was creating new model per clip)
                        postprocess_detector = self._get_postprocess_detector()
                        LOGGER.debug("Post-processing with %s detector (cached)", postprocess_detector.backend_name)

                        # Use settings from config
                        clip_cfg = self.runtime.general.clip
                        settings = ProcessingSettings(
                            sample_rate=getattr(clip_cfg, 'sample_rate', 3),
                            confidence_threshold=getattr(clip_cfg, 'post_analysis_confidence', 0.3),
                            generic_confidence=getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5),
                            tracking_enabled=getattr(clip_cfg, 'tracking_enabled', True),
                            merge_enabled=True,
                            same_species_merge_gap=getattr(clip_cfg, 'track_merge_gap', 120),
                            spatial_merge_enabled=getattr(clip_cfg, 'spatial_merge_enabled', True),
                            spatial_merge_iou=getattr(clip_cfg, 'spatial_merge_iou', 0.3),
                            spatial_merge_gap=30,
                            hierarchical_merge_enabled=getattr(clip_cfg, 'hierarchical_merge_enabled', True),
                            hierarchical_merge_gap=getattr(clip_cfg, 'track_merge_gap', 120),
                            single_animal_mode=getattr(clip_cfg, 'single_animal_mode', False),
                            thumbnail_cropped=getattr(clip_cfg, 'thumbnail_cropped', True),
                        )
                        processor = ClipPostProcessor(
                            detector=postprocess_detector,
                            storage_root=storage_root,
                            settings=settings,
                        )
                        result = processor.process_clip(
                            clip_path,
                            update_filename=True,
                            regenerate_thumbnails=True,
                        )
                        if result.success:
                            final_species = result.new_species
                            final_confidence = result.confidence
                            tracks_count = result.tracks_detected
                            # Update clip_path if file was renamed
                            if result.new_path:
                                clip_path = result.new_path
                            LOGGER.info("Post-processing complete (%s): %s (%.1f%%, %d tracks, %d raw detections)",
                                       postprocess_detector.backend_name,
                                       final_species, final_confidence * 100, tracks_count, result.raw_detections)

                            # Check if post-processing found NO animal (false positive from real-time detector)
                            # If delete_if_no_animal is enabled, clean up and skip notification
                            # Note: raw_detections counts blank frames too, so check species_results instead
                            delete_if_no_animal = getattr(clip_cfg, 'delete_if_no_animal', True)
                            min_detection_frames = max(1, int(getattr(clip_cfg, 'min_detection_frames', 2) or 1))
                            # Count distinct sampled frames where a detection survived filtering.
                            # "tracked" = detection assigned to a ByteTrack track (tracking path)
                            # "accepted" = detection passed filters (non-tracking fallback path)
                            detection_frame_idxs = {
                                e.frame_idx for e in (result.processing_log or [])
                                if e.event in ("tracked", "accepted") and e.frame_idx >= 0
                            }
                            detection_frame_count = len(detection_frame_idxs)
                            no_species = (not result.species_results or len(result.species_results) == 0) and result.tracks_detected == 0
                            too_few_detections = (
                                detection_frame_count < min_detection_frames
                                and result.frames_analyzed >= min_detection_frames
                            )
                            no_animal_found = no_species or too_few_detections
                            if delete_if_no_animal and no_animal_found:
                                LOGGER.info(
                                    "FALSE POSITIVE CLEANUP: Post-processing found no real animal in clip for %s "
                                    "(species_results=%d, tracks=%d, detection_frames=%d/%d, min_required=%d, raw_detections=%d) - deleting clip and skipping notification",
                                    camera_id, len(result.species_results) if result.species_results else 0,
                                    result.tracks_detected, detection_frame_count, result.frames_analyzed,
                                    min_detection_frames, result.raw_detections
                                )
                                # Delete the clip file and any associated files.
                                # Use unlink(missing_ok=True) and log success/failure
                                # at INFO level -- previously we used `if exists():`
                                # which is racy on NFS (stale attribute cache can
                                # return False) and silently skipped the delete,
                                # leaving orphan .mp4 files on disk.
                                cleaned = []
                                failures = []
                                try:
                                    clip_path.unlink(missing_ok=True)
                                    cleaned.append(clip_path.name)
                                except Exception as e:
                                    failures.append((clip_path.name, str(e)))
                                # Delete associated thumbnails
                                thumb_pattern = clip_path.stem + "_thumb*.jpg"
                                for thumb_file in clip_path.parent.glob(thumb_pattern):
                                    try:
                                        thumb_file.unlink(missing_ok=True)
                                        cleaned.append(thumb_file.name)
                                    except Exception as e:
                                        failures.append((thumb_file.name, str(e)))
                                # Delete processing log
                                log_file = clip_path.with_suffix('.log.json')
                                try:
                                    log_file.unlink(missing_ok=True)
                                    cleaned.append(log_file.name)
                                except Exception as e:
                                    failures.append((log_file.name, str(e)))
                                # Verify the clip really is gone (NFS attribute
                                # cache can lie about delete success). Retry once
                                # if it still resolves.
                                try:
                                    import os as _os
                                    if _os.path.lexists(str(clip_path)):
                                        time.sleep(0.2)
                                        try:
                                            clip_path.unlink(missing_ok=True)
                                        except Exception as e:
                                            failures.append((clip_path.name, f"retry: {e}"))
                                        if _os.path.lexists(str(clip_path)):
                                            failures.append((clip_path.name, "still present after retry"))
                                except Exception:
                                    pass
                                if failures:
                                    LOGGER.warning(
                                        "False-positive cleanup left %d file(s) for %s: %s",
                                        len(failures), camera_id, failures,
                                    )
                                else:
                                    LOGGER.info(
                                        "False-positive cleanup removed %d file(s) for %s",
                                        len(cleaned), camera_id,
                                    )
                                return  # Skip notification - no animal detected
                    except Exception as e:
                        LOGGER.error("Unified post-processing failed: %s", e)
                    finally:
                        # Periodically clear GPU memory to prevent VRAM accumulation
                        cleanup_gpu_memory()

                # Step 5.5: Append PTZ decisions to the processing log if available
                if ptz_decisions:
                    try:
                        import json
                        log_path = clip_path.with_suffix('.log.json')
                        if log_path.exists():
                            # Merge with existing log
                            with open(log_path, 'r') as f:
                                log_data = json.load(f)
                            log_data['ptz_decisions'] = ptz_decisions
                            with open(log_path, 'w') as f:
                                json.dump(log_data, f, indent=2, default=str)
                            LOGGER.debug("Added %d PTZ decisions to processing log", len(ptz_decisions))
                        else:
                            # Create new log file with just PTZ data
                            log_data = {
                                'clip': str(clip_path.name),
                                'ptz_decisions': ptz_decisions,
                            }
                            with open(log_path, 'w') as f:
                                json.dump(log_data, f, indent=2, default=str)
                            LOGGER.debug("Created PTZ log with %d decisions", len(ptz_decisions))
                    except Exception as e:
                        LOGGER.warning("Failed to save PTZ decisions: %s", e)

                # Step 6: Check if final species is excluded (post-processing may have reclassified)
                # If excluded, delete the clip and skip notification
                if all_excludes and species_matches_exclude(final_species, all_excludes):
                    LOGGER.info(
                        "Post-processing identified excluded species '%s' for %s - deleting clip and skipping notification",
                        final_species, camera_id
                    )
                    # Delete the clip file and any associated thumbnails
                    try:
                        if clip_path.exists():
                            clip_path.unlink()
                            LOGGER.debug("Deleted excluded clip: %s", clip_path)
                        # Delete associated thumbnails (same base name with _thumb*.jpg pattern)
                        thumb_pattern = clip_path.stem + "_thumb*.jpg"
                        for thumb_file in clip_path.parent.glob(thumb_pattern):
                            thumb_file.unlink()
                            LOGGER.debug("Deleted thumbnail: %s", thumb_file)
                        # Delete processing log if it exists
                        log_file = clip_path.with_suffix('.log.json')
                        if log_file.exists():
                            log_file.unlink()
                            LOGGER.debug("Deleted log file: %s", log_file)
                    except Exception as e:
                        LOGGER.warning("Failed to clean up excluded clip files: %s", e)
                    return  # Skip notification for excluded species

                # Step 7: Find thumbnail for notification
                thumbnail_path = None
                try:
                    thumb_pattern = clip_path.stem + "_thumb_*.jpg"
                    thumb_files = list(clip_path.parent.glob(thumb_pattern))
                    if thumb_files:
                        # Use the first thumbnail found (usually the main species)
                        thumbnail_path = str(thumb_files[0])
                        LOGGER.debug("Found thumbnail for notification: %s", thumbnail_path)
                except Exception as e:
                    LOGGER.warning("Failed to find thumbnail: %s", e)

                # Step 8: Send notification with refined info
                ctx = NotificationContext(
                    species=final_species,
                    confidence=final_confidence,
                    camera_id=ctx_base['camera_id'],
                    camera_name=ctx_base['camera_name'],
                    clip_path=str(clip_path),
                    event_started_at=ctx_base['event_started_at'],
                    event_duration=ctx_base['event_duration'],
                    thumbnail_path=thumbnail_path,
                    storage_root=str(storage_root),
                    web_base_url=web_base_url,
                )
                self.notifier.send(ctx, priority=priority, sound=sound)
                LOGGER.info("Event for %s closed; clip at %s (species: %s, %d tracks)",
                           ctx.camera_id, clip_path, final_species, tracks_count)

        # Use tracked species if available (more accurate than raw detections)
        tracked_species = self.event_state.get_tracked_species_label()
        if tracked_species and self.event_state.tracker:
            LOGGER.info("Tracked species for event: %s (from %d tracked objects)", 
                       tracked_species, self.event_state.tracker.active_track_count)
        
        ctx_base = {
            'species': tracked_species or self.event_state.species_label,
            'confidence': self.event_state.max_confidence,
            'camera_id': self.camera.id,
            'camera_name': self.camera.name,
            'event_started_at': self.event_state.start_ts,
            'event_duration': self.event_state.duration,
        }
        
        # Use tracked key frames if available
        species_key_frames = self.event_state.get_tracked_key_frames()
        
        loop.run_in_executor(
            None, 
            finalize_event, 
            self.event_state.frames,
            self.camera.id,
            self.event_state.start_ts,
            self.runtime.general.clip.format,
            ctx_base, 
            self.camera.notification.priority, 
            self.camera.notification.sound,
            species_key_frames,
            ptz_log,
            detector_cfg,  # Pass detector config for split-model post-processing
        )

        self.event_state = None

        # Event boundary cleanup: reset the persistent ObjectTracker so the
        # next event starts with fresh IDs (otherwise stale TrackInfo from a
        # previous event leaks into the new clip's species accumulation).
        # Also clear the PTZ lock so a recycled ByteTrack id of 1/2/... cannot
        # be silently treated as a continuation of the previous event's lock.
        if self.tracker is not None:
            try:
                self.tracker.reset()
            except Exception as e:
                LOGGER.debug("tracker.reset failed for %s: %s", self.camera.id, e)
        if self.ptz_tracker is not None and self.ptz_drives_tracking:
            try:
                self.ptz_tracker.clear_lock()
            except Exception:
                pass

    def _analyze_clip_frames(self, frames: List[tuple], num_samples: int = None) -> tuple[str, float]:
        """Analyze frames from a clip to get the most specific species identification.
        
        Uses object tracking to follow the same animal across frames and
        accumulate classifications for more accurate identification.
        
        Args:
            frames: List of (timestamp, frame) tuples
            num_samples: Number of frames to analyze (uses config default if None)
            
        Returns:
            (species_label, confidence) - best identification found
        """
        if not frames:
            return "", 0.0
        
        # Get post-analysis settings from config
        clip_cfg = self.runtime.general.clip
        conf_threshold = getattr(clip_cfg, 'post_analysis_confidence', 0.3)
        generic_conf = getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5)
        
        # Auto-calculate frames to analyze: ~1 frame per second of clip
        # Estimate clip duration from frame count (assume ~15 fps from buffer)
        if num_samples is None:
            total_frames = len(frames)
            estimated_fps = 15  # Default buffer fps
            clip_duration_secs = total_frames / estimated_fps
            # 1 frame per second, min 5, max 60
            num_samples = max(5, min(60, int(clip_duration_secs)))
            LOGGER.debug("Auto-calculated post-analysis: %d frames for ~%.1fs clip",
                        num_samples, clip_duration_secs)

        # Sample frames for analysis
        # When tracking is enabled, we need more frequent samples to maintain track continuity
        total_frames = len(frames)
        
        if self.tracking_enabled:
            # For tracking, sample every 3rd frame to maintain visual continuity
            # This gives ByteTrack enough overlap to track moving objects
            tracking_sample_rate = 3  # Analyze every 3rd frame (~5fps at 15fps source)
            sample_indices = list(range(0, total_frames, tracking_sample_rate))
            LOGGER.debug("Tracking mode: analyzing %d frames (every %dth frame)", 
                        len(sample_indices), tracking_sample_rate)
        elif total_frames <= num_samples:
            sample_indices = list(range(total_frames))
        else:
            step = total_frames / num_samples
            sample_indices = [int(i * step) for i in range(num_samples)]
        
        # Create a tracker for post-analysis to track objects across sampled frames
        effective_fps = 15 / (3 if self.tracking_enabled else max(1, total_frames / len(sample_indices)))
        analysis_tracker = create_tracker(enabled=self.tracking_enabled, frame_rate=int(effective_fps))
        
        # Collect all detections across sampled frames, using tracker if available
        all_detections = []
        for idx in sample_indices:
            _, frame = frames[idx]
            try:
                # Use configured thresholds for post-analysis
                detections = self.detector.infer(
                    frame, 
                    conf_threshold=conf_threshold,
                    generic_confidence=generic_conf,
                )
                
                # Update tracker if available
                if analysis_tracker and detections:
                    analysis_tracker.update(detections, frame, frame_idx=idx)
                
                all_detections.extend(detections)
            except Exception as e:
                LOGGER.warning("Post-clip analysis failed on frame %d: %s", idx, e)
                continue
        
        # If tracking was used, get the best species from tracked objects
        if analysis_tracker and analysis_tracker.active_track_count > 0:
            # Merge fragmented tracks that are likely the same animal
            merged = analysis_tracker.merge_similar_tracks(max_frame_gap=30)
            if merged > 0:
                LOGGER.debug("Merged %d fragmented tracks", merged)
            
            tracked_species = analysis_tracker.get_unique_species()
            if tracked_species:
                # Pick the species with highest confidence
                best_species, best_conf = max(tracked_species, key=lambda x: x[1])
                if best_species:
                    LOGGER.info("Post-clip analysis (tracked): %d objects -> %s (%.2f)",
                               analysis_tracker.active_track_count, best_species, best_conf)
                    return best_species, best_conf
        
        # Fallback to non-tracked analysis if tracking didn't produce results
        if not all_detections:
            return "", 0.0
        
        # Filter out invalid/generic detections before scoring
        invalid_terms = {'unknown', 'blank', 'empty', 'no cv result', 'no_cv_result', 'vehicle'}
        generic_terms = {'animal', 'bird', 'mammal', 'aves'}
        
        valid_detections = []
        for det in all_detections:
            species_lower = det.species.lower()
            # Skip invalid species
            if species_lower in invalid_terms or 'no cv result' in species_lower:
                continue
            # Skip if looks like UUID (wasn't cleaned properly)
            if len(det.species) > 30 and det.species.count('-') >= 3:
                continue
            valid_detections.append(det)
        
        if not valid_detections:
            return "", 0.0
        
        # Score each unique species by specificity and confidence
        # More specific = longer taxonomy (full species name vs just class)
        species_scores: dict[str, dict] = {}
        for det in valid_detections:
            species = det.species
            if species not in species_scores:
                # Calculate specificity: count words, penalize generic terms
                words = species.lower().replace('-', '_').split('_')
                specificity = len([w for w in words if w and w not in generic_terms])
                # Bonus for having a full binomial name (genus_species)
                if '_' in species and len(species.split('_')) >= 2:
                    specificity += 2

                species_scores[species] = {
                    'max_confidence': det.confidence,
                    'count': 1,
                    'specificity': specificity,
                    'taxonomy': det.taxonomy or species,
                }
            else:
                species_scores[species]['count'] += 1
                species_scores[species]['max_confidence'] = max(
                    species_scores[species]['max_confidence'], 
                    det.confidence
                )
        
        # Filter out generic categories if we have more specific ones
        specific_species = [s for s in species_scores.keys() 
                          if s.lower() not in generic_terms]
        
        if specific_species:
            # Use only specific species
            candidates = {s: species_scores[s] for s in specific_species}
        else:
            candidates = species_scores
        
        if not candidates:
            return "", 0.0

        # Pick the best: prioritize specificity, then count, then confidence
        best_species = max(
            candidates.keys(),
            key=lambda s: (
                candidates[s]['specificity'],
                candidates[s]['count'],
                candidates[s]['max_confidence']
            )
        )

        best_confidence = candidates[best_species]['max_confidence']
        LOGGER.info("Post-clip analysis: %d valid detections across %d frames -> %s (%.2f)",
                   len(valid_detections), len(sample_indices), best_species, best_confidence)

        return best_species, best_confidence


class PipelineOrchestrator:
    def __init__(
        self,
        runtime: RuntimeConfig,
        model_path: str = "yolov8n.pt",
        camera_filter: Optional[List[str]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.runtime = runtime
        self.config_path = config_path
        
        # Split-model architecture:
        # - realtime_detector: YOLO for fast streaming/PTZ tracking (~50-150ms)
        # - postprocess_detector: SpeciesNet for accurate clip analysis (~200-500ms)
        detector_cfg = runtime.general.detector
        
        # Override model path if provided
        if model_path:
            detector_cfg.model_path = model_path
        
        # Create realtime detector (YOLO by default for speed)
        self.detector = create_realtime_detector(detector_cfg)
        LOGGER.info(f"Realtime detector: {self.detector.backend_name} (for streaming/PTZ)")
        
        # Store detector config for post-processor to create its own SpeciesNet instance
        # This avoids loading SpeciesNet at startup if not needed immediately
        self.detector_cfg = detector_cfg
        
        # Log the split-model configuration
        realtime_backend = getattr(detector_cfg, 'realtime_backend', detector_cfg.backend)
        postprocess_backend = getattr(detector_cfg, 'postprocess_backend', detector_cfg.backend)
        if realtime_backend != postprocess_backend:
            LOGGER.info(
                "Split-model architecture: %s (realtime) + %s (post-processing)",
                realtime_backend, postprocess_backend
            )

        self.notifier = PushoverNotifier(
            runtime.general.notification.pushover_app_token_env,
            runtime.general.notification.pushover_user_key_env,
        )
        self.storage = StorageManager(
            storage_root=Path(self.runtime.general.storage_root),
            logs_root=Path(self.runtime.general.logs_root),
            max_utilization_pct=self.runtime.general.retention.max_utilization_pct,
        )
        cameras = runtime.cameras
        if camera_filter:
            camera_set = {cid for cid in camera_filter}
            cameras = [cam for cam in cameras if cam.id in camera_set]
        self.registry = CameraRegistry.from_configs(cameras)
        self.cameras = cameras

    async def run(self) -> None:
        stop_event = asyncio.Event()

        # Initialize post-processing concurrency limit from config
        postprocess_limit = getattr(self.runtime.general.clip, 'max_concurrent_postprocess', 1)
        StreamWorker.set_postprocess_limit(postprocess_limit)

        # Check if tracking is enabled in config
        tracking_enabled = getattr(self.runtime.general.clip, 'tracking_enabled', True)
        if tracking_enabled:
            LOGGER.info("Object tracking enabled for species identification")
        
        workers = [
            StreamWorker(
                camera=cam,
                runtime=self.runtime,
                detector=self.detector,
                notifier=self.notifier,
                storage=self.storage,
                tracking_enabled=tracking_enabled,
            )
            for cam in self.cameras
        ]
        
        worker_map = {w.camera.id: w for w in workers}
        
        # Track shared PTZ trackers so multiple cameras can feed into one
        shared_ptz_trackers = {}  # target_cam_id -> tracker
        
        # Initialize PTZ auto-tracking for cameras with it enabled
        for worker in workers:
            ptz_cfg = worker.camera.ptz_tracking
            
            # Handle self-tracking mode (camera tracks its own detections)
            if ptz_cfg.self_track:
                if worker.onvif_client and worker.onvif_profile_token:
                    # Check if there's already a tracker for this camera's PTZ
                    existing_tracker = shared_ptz_trackers.get(worker.camera.id)
                    if existing_tracker:
                        # Reuse existing tracker (another camera created it)
                        worker.ptz_tracker = existing_tracker
                        worker.ptz_drives_tracking = True
                        LOGGER.info(
                            "PTZ self-tracking enabled: %s shares tracker with primary",
                            worker.camera.id
                        )
                    else:
                        worker.ptz_tracker = create_ptz_tracker(
                            onvif_client=worker.onvif_client,
                            profile_token=worker.onvif_profile_token,
                            config={
                                'pan_scale': ptz_cfg.pan_scale,
                                'tilt_scale': ptz_cfg.tilt_scale,
                                'target_fill_pct': ptz_cfg.target_fill_pct,
                                'min_detection_area': getattr(ptz_cfg, 'min_detection_area', 0.005),
                                'smoothing': ptz_cfg.smoothing,
                                'update_interval': ptz_cfg.update_interval,
                                'pan_center_x': 0.5,  # Self-tracking always centers
                                'tilt_center_y': 0.5,
                                # Respect patrol config in self-track mode
                                'patrol_enabled': ptz_cfg.patrol_enabled,
                                'patrol_speed': ptz_cfg.patrol_speed,
                                'patrol_return_delay': ptz_cfg.patrol_return_delay,
                                'patrol_presets': ptz_cfg.patrol_presets,
                                'patrol_dwell_time': ptz_cfg.patrol_dwell_time,
                                'move_min_duration': getattr(ptz_cfg, 'move_min_duration', 0.6),
                                'cam1_fallback_delay': getattr(ptz_cfg, 'cam1_fallback_delay', 3.0),
                                'investigate_enabled': getattr(ptz_cfg, 'investigate_enabled', False),
                                'investigate_min_area': getattr(ptz_cfg, 'investigate_min_area', 0.0005),
                                'investigate_timeout': getattr(ptz_cfg, 'investigate_timeout', 4.0),
                                'investigate_cooldown': getattr(ptz_cfg, 'investigate_cooldown', 30.0),
                                'investigate_cooldown_radius': getattr(ptz_cfg, 'investigate_cooldown_radius', 0.10),
                            }
                        )
                        # Enable patrol and tracking based on config
                        worker.ptz_tracker.set_patrol_enabled(ptz_cfg.patrol_enabled)
                        worker.ptz_tracker.set_track_enabled(ptz_cfg.track_enabled)
                        worker.ptz_drives_tracking = True
                        shared_ptz_trackers[worker.camera.id] = worker.ptz_tracker
                        
                        mode_parts = []
                        if ptz_cfg.patrol_enabled:
                            if ptz_cfg.patrol_presets:
                                mode_parts.append(f"preset-patrol({len(ptz_cfg.patrol_presets)})")
                            else:
                                mode_parts.append("sweep-patrol")
                        if ptz_cfg.track_enabled:
                            mode_parts.append("self-track")
                        mode = "+".join(mode_parts) if mode_parts else "idle"
                        
                        LOGGER.info(
                            "PTZ self-tracking enabled (%s): %s controls its own PTZ",
                            mode, worker.camera.id
                        )
                else:
                    LOGGER.warning(
                        "Self-tracking enabled for %s but no ONVIF configured",
                        worker.camera.id
                    )
            
            # Handle cross-camera tracking (cam1 detects -> cam2 PTZ)
            elif ptz_cfg.enabled:
                # Find target camera for PTZ control. Refuse to silently default
                # to self -- that would attach this worker's tracker to its own
                # (potentially missing) ONVIF profile and either drive the
                # wrong camera or fail silently.
                target_id = ptz_cfg.target_camera_id
                if not target_id:
                    LOGGER.warning(
                        "PTZ tracking enabled for %s but no target_camera_id "
                        "configured; skipping (set target_camera_id or use "
                        "self_track: true)",
                        worker.camera.id
                    )
                    continue
                if target_id == worker.camera.id:
                    LOGGER.warning(
                        "PTZ target_camera_id for %s points at itself; use "
                        "self_track: true instead. Skipping.",
                        worker.camera.id
                    )
                    continue
                target_worker = worker_map.get(target_id)

                if target_worker and target_worker.onvif_client and target_worker.onvif_profile_token:
                    # If a tracker already exists for this PTZ target (e.g.,
                    # the target camera was processed first with self_track),
                    # reuse it instead of creating a competing instance with
                    # its own lock and state.
                    existing_tracker = shared_ptz_trackers.get(target_id)
                    if existing_tracker is not None:
                        worker.ptz_tracker = existing_tracker
                        worker.ptz_drives_tracking = True
                        LOGGER.info(
                            "PTZ auto-tracking: %s reusing shared tracker for %s PTZ",
                            worker.camera.id, target_id
                        )
                    else:
                        worker.ptz_tracker = create_ptz_tracker(
                            onvif_client=target_worker.onvif_client,
                            profile_token=target_worker.onvif_profile_token,
                            config={
                                'pan_scale': ptz_cfg.pan_scale,
                                'tilt_scale': ptz_cfg.tilt_scale,
                                'target_fill_pct': ptz_cfg.target_fill_pct,
                                'min_detection_area': getattr(ptz_cfg, 'min_detection_area', 0.005),
                                'smoothing': ptz_cfg.smoothing,
                                'update_interval': ptz_cfg.update_interval,
                                'pan_center_x': ptz_cfg.pan_center_x,
                                'tilt_center_y': ptz_cfg.tilt_center_y,
                                'patrol_enabled': ptz_cfg.patrol_enabled,
                                'patrol_speed': ptz_cfg.patrol_speed,
                                'patrol_return_delay': ptz_cfg.patrol_return_delay,
                                'patrol_presets': ptz_cfg.patrol_presets,
                                'patrol_dwell_time': ptz_cfg.patrol_dwell_time,
                                'move_min_duration': getattr(ptz_cfg, 'move_min_duration', 0.6),
                                'cam1_fallback_delay': getattr(ptz_cfg, 'cam1_fallback_delay', 3.0),
                                'investigate_enabled': getattr(ptz_cfg, 'investigate_enabled', False),
                                'investigate_min_area': getattr(ptz_cfg, 'investigate_min_area', 0.0005),
                                'investigate_timeout': getattr(ptz_cfg, 'investigate_timeout', 4.0),
                                'investigate_cooldown': getattr(ptz_cfg, 'investigate_cooldown', 30.0),
                                'investigate_cooldown_radius': getattr(ptz_cfg, 'investigate_cooldown_radius', 0.10),
                            }
                        )
                        # Enable patrol and tracking separately based on config
                        worker.ptz_tracker.set_patrol_enabled(ptz_cfg.patrol_enabled)
                        worker.ptz_tracker.set_track_enabled(ptz_cfg.track_enabled)
                        worker.ptz_drives_tracking = True

                        # Share this tracker so target camera can also use it for self-tracking
                        shared_ptz_trackers[target_id] = worker.ptz_tracker

                    if ptz_cfg.patrol_presets:
                        mode = f"preset-patrol({len(ptz_cfg.patrol_presets)})+track"
                    elif ptz_cfg.patrol_enabled:
                        mode = "sweep-patrol+track"
                    else:
                        mode = "track-only"
                    LOGGER.info(
                        "PTZ auto-tracking enabled (%s): %s detections -> %s PTZ",
                        mode, worker.camera.id, target_id
                    )
                else:
                    LOGGER.warning(
                        "PTZ tracking enabled for %s but target %s has no ONVIF",
                        worker.camera.id, target_id
                    )
        
        # Second pass: give target cameras access to the shared tracker
        # This enables:
        # 1. PTZ decision logs to be captured in target camera recordings
        # 2. Self-tracking (if self_track: true) where target camera detections can trigger PTZ
        LOGGER.info("Shared PTZ trackers available: %s", list(shared_ptz_trackers.keys()))
        for worker in workers:
            ptz_cfg = worker.camera.ptz_tracking
            LOGGER.debug(
                "PTZ setup check for %s: has_tracker=%s, self_track=%s, available_shared=%s",
                worker.camera.id, worker.ptz_tracker is not None, ptz_cfg.self_track,
                worker.camera.id in shared_ptz_trackers
            )
            if not worker.ptz_tracker:
                # Check if another camera created a tracker for this PTZ
                existing_tracker = shared_ptz_trackers.get(worker.camera.id)
                if existing_tracker:
                    worker.ptz_tracker = existing_tracker
                    if ptz_cfg.self_track:
                        # Target camera is configured for self-track; it may
                        # drive the shared tracker with its own detections.
                        worker.ptz_drives_tracking = True
                        LOGGER.info(
                            "PTZ self-tracking enabled: %s shares tracker (cam2 detections will also trigger tracking)",
                            worker.camera.id
                        )
                    else:
                        # Logging-only sharing: do NOT push this worker's
                        # detections (different coordinate space) into the
                        # tracker. Only the source camera's _process_frame
                        # will call ptz_tracker.update.
                        worker.ptz_drives_tracking = False
                        LOGGER.info(
                            "PTZ tracker shared with %s for logging (PTZ decisions will appear in recordings)",
                            worker.camera.id
                        )
                else:
                    LOGGER.debug(
                        "%s has no PTZ tracker and none shared for it",
                        worker.camera.id
                    )

        # Third pass: set up multi-camera detection sources for PTZ tracking
        # This allows cam2 (target) detections to take over tracking from cam1 (source)
        for worker in workers:
            ptz_cfg = worker.camera.ptz_tracking
            # Only set up multi-camera for cross-camera tracking (not self-track)
            # and only if multi_camera_tracking is enabled in config
            if (ptz_cfg.enabled
                and ptz_cfg.target_camera_id
                and ptz_cfg.multi_camera_tracking
                and worker.ptz_tracker):
                source_id = worker.camera.id  # e.g., cam1
                target_id = ptz_cfg.target_camera_id  # e.g., cam2
                target_worker = worker_map.get(target_id)

                if target_worker:
                    # Set up detection sources - both source and target cameras contribute
                    detection_sources = {
                        source_id: worker,
                        target_id: target_worker,
                    }

                    # Configure source worker for multi-camera tracking
                    worker._ptz_detection_sources = detection_sources
                    worker._ptz_source_camera_id = source_id
                    worker._ptz_target_camera_id = target_id

                    # Also configure target worker if it has the shared tracker
                    if target_worker.ptz_tracker is worker.ptz_tracker:
                        target_worker._ptz_detection_sources = detection_sources
                        target_worker._ptz_source_camera_id = source_id
                        target_worker._ptz_target_camera_id = target_id

                    LOGGER.info(
                        "Multi-camera PTZ tracking enabled: %s (source) + %s (target) -> PTZ "
                        "(target camera can take over for fine tracking)",
                        source_id, target_id
                    )

        # Start web server
        web_server = WebServer(
            worker_map, 
            storage_root=Path(self.runtime.general.storage_root),
            logs_root=Path(self.runtime.general.logs_root),
            port=8080,
            config_path=self.config_path,
            runtime=self.runtime,
        )
        
        await asyncio.gather(
            web_server.start(),
            *(worker.run(stop_event) for worker in workers)
        )
