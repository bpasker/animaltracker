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
from .ebird import EBirdClient, create_ebird_client
from .notification import NotificationContext, PushoverNotifier
from .storage import StorageManager
from .onvif_client import OnvifClient
from .tracker import ObjectTracker, create_tracker
from .ptz_tracker import PTZTracker, create_ptz_tracker
from .web import WebServer

LOGGER = logging.getLogger(__name__)


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
            f"hwaccel;cuda|"
            f"hwaccel_output_format;cuda"
        )
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    
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

    def update(self, detections: List[Detection], frame_ts: float, frame: np.ndarray) -> None:
        self.last_detection_ts = frame_ts
        
        # Update tracker if available
        if self.tracker:
            tracked_detections = self.tracker.update(detections, frame)
            # Use tracked detections for species updates
            for track_id, det in tracked_detections.items():
                # Get best species for this track so far
                best_species, best_conf, _ = self.tracker.get_track_species(track_id)
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
    # Initialized by PipelineOrchestrator based on config (default: 1 concurrent job)
    _postprocess_semaphore: threading.Semaphore = None
    _postprocess_limit: int = 1

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
        ebird_client: Optional[EBirdClient] = None,
        tracking_enabled: bool = True,
    ) -> None:
        self.camera = camera
        self.runtime = runtime
        self.detector = detector
        self.notifier = notifier
        self.storage = storage
        self.ebird_client = ebird_client
        self.tracking_enabled = tracking_enabled
        # Ensure buffer is at least 30s for manual clips
        clip_seconds = max(30.0, runtime.general.clip.pre_seconds + runtime.general.clip.post_seconds)
        self.clip_buffer = ClipBuffer(max_seconds=clip_seconds, fps=15)
        self.event_state: Optional[EventState] = None
        self.pending_detection_start_ts: Optional[float] = None
        self._snapshot_taken = False
        self.latest_frame: Optional[np.ndarray] = None
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
            # Offload connection to thread as it can block
            cap = await loop.run_in_executor(None, cv2.VideoCapture, rtsp_uri, capture_backend)
            
            if not cap.isOpened():
                if use_hwaccel:
                    # Fall back to software decoding if CUDA failed
                    LOGGER.warning("CUDA hardware decoding failed for %s, falling back to software", self.camera.id)
                    use_hwaccel = False
                    rtsp_uri = build_ffmpeg_uri(
                        self.camera.rtsp.uri,
                        self.camera.rtsp.transport,
                        hwaccel=False
                    )
                    cap = await loop.run_in_executor(None, cv2.VideoCapture, rtsp_uri, capture_backend)
                    if not cap.isOpened():
                        LOGGER.error("Unable to open RTSP stream for %s; retrying in 5s", self.camera.id)
                        await asyncio.sleep(5)
                        continue
                else:
                    LOGGER.error("Unable to open RTSP stream for %s; retrying in 5s", self.camera.id)
                    await asyncio.sleep(5)
                    continue

            LOGGER.info("Connected to stream for %s", self.camera.id)
            try:
                while not stop_event.is_set():
                    # Offload blocking OpenCV read to thread to keep web server responsive
                    ret, frame = await loop.run_in_executor(None, cap.read)
                    
                    if not ret:
                        LOGGER.warning("Stream lost for %s; reconnecting...", self.camera.id)
                        break
                    
                    self.latest_frame = frame
                    
                    if not self._snapshot_taken:
                        # Offload snapshot saving to thread
                        loop.run_in_executor(None, self.storage.save_snapshot, self.camera.id, frame.copy())
                        self._snapshot_taken = True

                    frame_ts = time.time()
                    self.clip_buffer.push(frame_ts, frame)
                    
                    # Capture every frame for active events (even if inference is skipped)
                    if self.event_state is not None:
                        self.event_state.frames.append((frame_ts, frame))

                        # Force-close events that exceed max duration (prevents memory leak)
                        max_duration = self.runtime.general.clip.max_event_seconds
                        event_elapsed = frame_ts - self.event_state.start_ts
                        if event_elapsed > max_duration:
                            LOGGER.warning(
                                "Event for %s exceeded max duration (%.1fs > %.1fs, %d frames) - force closing to prevent memory leak",
                                self.camera.id, event_elapsed, max_duration, len(self.event_state.frames)
                            )
                            await self._close_event(frame_ts)

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
                            self._process_frame(frame.copy(), frame_ts)
                        )
            finally:
                # Cancel any pending inference
                if inference_task and not inference_task.done():
                    inference_task.cancel()
                # Offload release to thread
                await loop.run_in_executor(None, cap.release)
            
            if not stop_event.is_set():
                await asyncio.sleep(1)  # Brief pause before reconnect

    async def _process_frame(self, frame: np.ndarray, ts: float) -> None:
        loop = asyncio.get_running_loop()        
        detections = await loop.run_in_executor(
            None, 
            lambda: self.detector.infer(
                frame, 
                conf_threshold=self.camera.thresholds.confidence,
                generic_confidence=self.camera.thresholds.generic_confidence
            )
        )
        filtered = self._filter_detections(detections)

        # Store for live view overlay
        frame_h, frame_w = frame.shape[:2]
        self.latest_detections = filtered
        self.latest_detection_ts = ts
        self.latest_frame_size = (frame_w, frame_h)

        # PTZ auto-tracking: always call update (handles patrol when no detections)
        if self.ptz_tracker:
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

                for cam_id, worker in self._ptz_detection_sources.items():
                    # Only use recent detections (within 1 second)
                    if ts - worker.latest_detection_ts < 1.0:
                        w, h = worker.latest_frame_size
                        if w > 0 and h > 0:
                            camera_detections[cam_id] = (
                                worker.latest_detections,
                                w,
                                h
                            )

                if camera_detections:
                    await loop.run_in_executor(
                        None,
                        self.ptz_tracker.update_multi_camera,
                        camera_detections,
                        self._ptz_source_camera_id,
                        self._ptz_target_camera_id
                    )
                else:
                    # No recent detections from any camera
                    await loop.run_in_executor(
                        None,
                        self.ptz_tracker.update,
                        [], frame_w, frame_h
                    )
            else:
                # Single-camera tracking mode
                await loop.run_in_executor(
                    None,
                    self.ptz_tracker.update,
                    filtered, frame_w, frame_h
                )
            # Periodically trim old decisions to prevent unbounded memory growth
            # Keep last 5 minutes of decisions (enough for any reasonable clip)
            cutoff = ts - 300  # 5 minutes
            self.ptz_tracker.trim_old_decisions(cutoff)

        if not filtered:
            self.pending_detection_start_ts = None
            await self._maybe_close_event(ts)
            return
        
        # Use all filtered detections to update state
        primary = filtered[0]
        if self.event_state is None:
            # Check for minimum duration
            if self.pending_detection_start_ts is None:
                self.pending_detection_start_ts = ts
                return # Wait for next frame
            
            if ts - self.pending_detection_start_ts < self.camera.thresholds.min_duration:
                return # Still waiting
            
            # Duration met, start event
            self.pending_detection_start_ts = None
            LOGGER.info("Started tracking %s on %s (%.2f)", primary.species, self.camera.id, primary.confidence)
            
            # Create tracker for this event if enabled
            event_tracker = None
            if self.tracking_enabled:
                event_tracker = create_tracker(enabled=True, frame_rate=15)
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
            
        self.event_state.update(filtered, ts, frame)

    def _normalize_species(self, species: str) -> str:
        """Normalize species name for comparison (lowercase, underscores)."""
        return species.lower().replace(' ', '_').replace('-', '_').strip()

    def _species_matches_exclude(self, detection_species: str, excludes: set) -> bool:
        """Check if a detection species matches any exclude pattern.
        
        Supports:
        - Exact match: "mammalia_rodentia_sciuridae" matches "mammalia_rodentia_sciuridae"
        - Prefix match: "mammalia_rodentia_sciuridae_sciurus" matches "mammalia_rodentia_sciuridae"
        - Common name match: "mammalia_rodentia_sciuridae" matches "squirrel"
        """
        from .species_names import get_common_name
        
        normalized = self._normalize_species(detection_species)
        
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
            
            # Check if exclude is a substring (for family/order level matching)
            # e.g., "sciuridae" in "mammalia_rodentia_sciuridae"
            if exclude_norm in normalized:
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
        
        # Get eBird filter mode if enabled
        ebird_mode = None
        if self.ebird_client and self.ebird_client.enabled:
            ebird_mode = self.runtime.general.ebird.filter_mode
        
        filtered: List[Detection] = []
        for det in detections:
            label = self._normalize_species(det.species)
            
            # Check includes (if specified, only allow listed species)
            if includes and not any(
                label == inc or label.startswith(inc + '_') or inc in label
                for inc in includes
            ):
                continue
            
            # Check excludes (skip if matches any exclude pattern)
            if all_excludes and self._species_matches_exclude(det.species, all_excludes):
                LOGGER.debug("Excluding detection: %s (matches exclude list)", det.species)
                continue
            
            # Apply eBird filtering for bird species
            if ebird_mode and self._is_bird_detection(det):
                is_present = self.ebird_client.is_species_present(det.taxonomy or det.species)
                
                if ebird_mode == "filter" and not is_present:
                    # Filter mode: skip species not recently seen in the region
                    LOGGER.debug("eBird filter: %s not present in region, skipping", det.species)
                    continue
                elif ebird_mode == "flag" and not is_present:
                    # Flag mode: mark but don't filter
                    LOGGER.info("eBird flag: %s not recently reported in region", det.species)
                # boost mode: could be used for sorting/prioritization later
            
            filtered.append(det)
        return filtered
    
    def _is_bird_detection(self, det: Detection) -> bool:
        """Check if a detection is a bird based on taxonomy or species name."""
        taxonomy = det.taxonomy or det.species
        taxonomy_lower = taxonomy.lower()
        
        # Check taxonomy for bird indicators
        bird_indicators = ['aves', 'bird', ';aves;', 'animalia;chordata;aves']
        for indicator in bird_indicators:
            if indicator in taxonomy_lower:
                return True
        
        # Check common bird species names
        species_lower = det.species.lower()
        common_bird_words = {
            'cardinal', 'robin', 'sparrow', 'finch', 'hawk', 'eagle', 'owl',
            'crow', 'raven', 'jay', 'woodpecker', 'hummingbird', 'duck', 'goose',
            'heron', 'crane', 'pelican', 'gull', 'tern', 'warbler', 'thrush',
            'wren', 'chickadee', 'nuthatch', 'titmouse', 'oriole', 'tanager',
            'grosbeak', 'bunting', 'dove', 'pigeon', 'falcon', 'kestrel'
        }
        for word in common_bird_words:
            if word in species_lower:
                return True
        
        return False

    async def _maybe_close_event(self, ts: float) -> None:
        if self.event_state is None:
            return
        idle = ts - self.event_state.last_detection_ts
        if idle < self.runtime.general.clip.post_seconds:
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
                LOGGER.info(
                    "No PTZ decisions in event window [%.1f - %.1f] for %s "
                    "(tracker has %d total decisions, track_enabled=%s, mode=%s)",
                    window_start, event_end, self.camera.id,
                    total_decisions,
                    self.ptz_tracker.is_track_enabled(),
                    self.ptz_tracker.get_mode()
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
                            no_animal_found = (not result.species_results or len(result.species_results) == 0) and result.tracks_detected == 0
                            if delete_if_no_animal and no_animal_found:
                                LOGGER.info(
                                    "FALSE POSITIVE CLEANUP: Post-processing found NO animal in clip for %s "
                                    "(species_results=%d, tracks=%d, raw_detections=%d) - deleting clip and skipping notification",
                                    camera_id, len(result.species_results) if result.species_results else 0,
                                    result.tracks_detected, result.raw_detections
                                )
                                # Delete the clip file and any associated files
                                try:
                                    if clip_path.exists():
                                        clip_path.unlink()
                                        LOGGER.debug("Deleted false positive clip: %s", clip_path)
                                    # Delete associated thumbnails
                                    thumb_pattern = clip_path.stem + "_thumb*.jpg"
                                    for thumb_file in clip_path.parent.glob(thumb_pattern):
                                        thumb_file.unlink()
                                        LOGGER.debug("Deleted thumbnail: %s", thumb_file)
                                    # Delete processing log
                                    log_file = clip_path.with_suffix('.log.json')
                                    if log_file.exists():
                                        log_file.unlink()
                                        LOGGER.debug("Deleted log file: %s", log_file)
                                except Exception as e:
                                    LOGGER.warning("Failed to clean up false positive clip files: %s", e)
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
        
        # Get eBird filter mode if enabled
        ebird_mode = None
        if self.ebird_client and self.ebird_client.enabled:
            ebird_mode = self.runtime.general.ebird.filter_mode
        
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
                    analysis_tracker.update(detections, frame)
                
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
                # Apply eBird filtering to tracked results
                best_species = None
                best_conf = 0.0
                
                for species, conf in tracked_species:
                    # Check eBird if enabled
                    if ebird_mode and self._is_bird_species(species):
                        is_present = self.ebird_client.is_species_present(species)
                        if is_present is False and ebird_mode == "filter":
                            LOGGER.debug("eBird filter: %s not in region, skipping", species)
                            continue
                        elif is_present is True and (best_species is None or conf > best_conf):
                            best_species = species
                            best_conf = conf
                            LOGGER.debug("eBird: %s confirmed in region", species)
                        elif is_present is None or is_present is False:
                            # Not a bird or not confirmed - use if no better option
                            if best_species is None or conf > best_conf:
                                best_species = species
                                best_conf = conf
                    else:
                        # Non-bird or no eBird filtering
                        if best_species is None or conf > best_conf:
                            best_species = species
                            best_conf = conf
                
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
                
                # Check eBird for bird species - boost if present in region
                ebird_boost = 0
                ebird_present = None
                if ebird_mode and self._is_bird_species(det.taxonomy or det.species):
                    ebird_present = self.ebird_client.is_species_present(det.taxonomy or det.species)
                    if ebird_present is True:
                        # Species confirmed in region - boost priority
                        ebird_boost = 3
                        LOGGER.debug("eBird: %s confirmed in region, boosting", species)
                    elif ebird_present is False:
                        # Species NOT seen in region - penalize
                        if ebird_mode == "filter":
                            # Skip entirely in filter mode
                            LOGGER.debug("eBird filter: %s not in region, skipping", species)
                            continue
                        else:
                            # Flag/boost mode: just penalize
                            ebird_boost = -2
                            LOGGER.debug("eBird: %s not reported in region, penalizing", species)
                
                species_scores[species] = {
                    'max_confidence': det.confidence,
                    'count': 1,
                    'specificity': specificity,
                    'taxonomy': det.taxonomy or species,
                    'ebird_boost': ebird_boost,
                    'ebird_present': ebird_present,
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
        
        # Pick the best: prioritize eBird presence, then specificity, then count, then confidence
        best_species = max(
            candidates.keys(),
            key=lambda s: (
                candidates[s].get('ebird_boost', 0),  # eBird confirmed species first
                candidates[s]['specificity'],
                candidates[s]['count'],
                candidates[s]['max_confidence']
            )
        )
        
        best_confidence = candidates[best_species]['max_confidence']
        ebird_info = candidates[best_species].get('ebird_present')
        ebird_status = ""
        if ebird_info is True:
            ebird_status = " [eBird: confirmed in region]"
        elif ebird_info is False:
            ebird_status = " [eBird: not reported in region]"
        
        LOGGER.info("Post-clip analysis: %d valid detections across %d frames -> %s (%.2f)%s", 
                   len(valid_detections), len(sample_indices), best_species, best_confidence, ebird_status)
        
        return best_species, best_confidence
    
    def _is_bird_species(self, taxonomy: str) -> bool:
        """Check if a taxonomy string represents a bird species."""
        taxonomy_lower = taxonomy.lower()
        
        # Check taxonomy for bird indicators
        bird_indicators = ['aves', ';aves;', 'animalia;chordata;aves']
        for indicator in bird_indicators:
            if indicator in taxonomy_lower:
                return True
        
        # Check common bird family/order names
        bird_terms = {
            'passeriformes', 'piciformes', 'strigiformes', 'falconiformes',
            'accipitriformes', 'anseriformes', 'columbiformes', 'apodiformes'
        }
        for term in bird_terms:
            if term in taxonomy_lower:
                return True
        
        return False


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
        
        # Create eBird client if configured
        ebird_cfg = runtime.general.ebird
        self.ebird_client = create_ebird_client(
            enabled=ebird_cfg.enabled,
            api_key_env=ebird_cfg.api_key_env,
            region=ebird_cfg.region,
            days_back=ebird_cfg.days_back,
            cache_hours=ebird_cfg.cache_hours,
        )
        if self.ebird_client and self.ebird_client.enabled:
            LOGGER.info(f"eBird integration enabled: region={ebird_cfg.region}, mode={ebird_cfg.filter_mode}")
        
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
                ebird_client=self.ebird_client,
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
                            }
                        )
                        # Enable patrol and tracking based on config
                        worker.ptz_tracker.set_patrol_enabled(ptz_cfg.patrol_enabled)
                        worker.ptz_tracker.set_track_enabled(ptz_cfg.track_enabled)
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
                # Find target camera for PTZ control
                target_id = ptz_cfg.target_camera_id or worker.camera.id
                target_worker = worker_map.get(target_id)
                
                if target_worker and target_worker.onvif_client and target_worker.onvif_profile_token:
                    worker.ptz_tracker = create_ptz_tracker(
                        onvif_client=target_worker.onvif_client,
                        profile_token=target_worker.onvif_profile_token,
                        config={
                            'pan_scale': ptz_cfg.pan_scale,
                            'tilt_scale': ptz_cfg.tilt_scale,
                            'target_fill_pct': ptz_cfg.target_fill_pct,
                            'smoothing': ptz_cfg.smoothing,
                            'update_interval': ptz_cfg.update_interval,
                            'pan_center_x': ptz_cfg.pan_center_x,
                            'tilt_center_y': ptz_cfg.tilt_center_y,
                            'patrol_enabled': ptz_cfg.patrol_enabled,
                            'patrol_speed': ptz_cfg.patrol_speed,
                            'patrol_return_delay': ptz_cfg.patrol_return_delay,
                            'patrol_presets': ptz_cfg.patrol_presets,
                            'patrol_dwell_time': ptz_cfg.patrol_dwell_time,
                        }
                    )
                    # Enable patrol and tracking separately based on config
                    worker.ptz_tracker.set_patrol_enabled(ptz_cfg.patrol_enabled)
                    worker.ptz_tracker.set_track_enabled(ptz_cfg.track_enabled)
                    
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
        for worker in workers:
            ptz_cfg = worker.camera.ptz_tracking
            if ptz_cfg.self_track and not worker.ptz_tracker:
                # Check if another camera created a tracker for this PTZ
                existing_tracker = shared_ptz_trackers.get(worker.camera.id)
                if existing_tracker:
                    worker.ptz_tracker = existing_tracker
                    LOGGER.info(
                        "PTZ self-tracking enabled: %s shares tracker (cam2 detections will also trigger tracking)",
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
