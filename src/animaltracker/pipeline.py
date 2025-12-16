"""Streaming pipeline orchestration."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .camera_registry import CameraRegistry
from .clip_buffer import ClipBuffer
from .config import CameraConfig, RuntimeConfig
from .detector import Detection, create_detector, DetectorProtocol
from .notification import NotificationContext, PushoverNotifier
from .storage import StorageManager
from .onvif_client import OnvifClient
from .web import WebServer

LOGGER = logging.getLogger(__name__)


def build_ffmpeg_uri(rtsp_uri: str, transport: str = "tcp") -> str:
    """Build RTSP URI with FFmpeg env options for OpenCV CAP_FFMPEG backend."""
    # FFmpeg uses RTSP_TRANSPORT env or query param; we set env before capture
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    return rtsp_uri


@dataclass
class EventState:
    camera: CameraConfig
    start_ts: float
    species: set[str]
    max_confidence: float
    last_detection_ts: float
    frames: List[tuple[float, np.ndarray]] = field(default_factory=list)

    def update(self, detections: List[Detection], frame_ts: float, frame: np.ndarray) -> None:
        self.last_detection_ts = frame_ts
        for det in detections:
            self.species.add(det.species)
            if det.confidence > self.max_confidence:
                self.max_confidence = det.confidence
        # Frames are now appended in the main loop to ensure full framerate
        # self.frames.append((frame_ts, frame))

    @property
    def species_label(self) -> str:
        return "+".join(sorted(self.species))

    @property
    def duration(self) -> float:
        return self.last_detection_ts - self.start_ts


class StreamWorker:
    def __init__(
        self,
        camera: CameraConfig,
        runtime: RuntimeConfig,
        detector: DetectorProtocol,
        notifier: PushoverNotifier,
        storage: StorageManager,
    ) -> None:
        self.camera = camera
        self.runtime = runtime
        self.detector = detector
        self.notifier = notifier
        self.storage = storage
        # Ensure buffer is at least 30s for manual clips
        clip_seconds = max(30.0, runtime.general.clip.pre_seconds + runtime.general.clip.post_seconds)
        self.clip_buffer = ClipBuffer(max_seconds=clip_seconds, fps=15)
        self.event_state: Optional[EventState] = None
        self.pending_detection_start_ts: Optional[float] = None
        self._snapshot_taken = False
        self.latest_frame: Optional[np.ndarray] = None
        
        # Initialize ONVIF client if configured
        self.onvif_client: Optional[OnvifClient] = None
        self.onvif_profile_token: Optional[str] = None
        self.peer_worker: Optional[StreamWorker] = None
        
        if camera.onvif.host:
            user, password = camera.onvif.credentials()
            if user and password:
                try:
                    self.onvif_client = OnvifClient(
                        host=camera.onvif.host,
                        port=camera.onvif.port,
                        username=user,
                        password=password
                    )
                    # Cache the first profile token for PTZ
                    profiles = self.onvif_client.get_profiles()
                    if profiles:
                        self.onvif_profile_token = profiles[0].metadata.get("token")
                except Exception as e:
                    LOGGER.warning(f"Failed to initialize ONVIF for {camera.id}: {e}")

    def track_target(self, x: float, y: float) -> None:
        """Move PTZ to target coordinates (0..1)."""
        if not self.onvif_client or not self.onvif_profile_token:
            return
            
        # Map 0..1 to -1..1 (or configured scale)
        # Center is 0.5 -> 0.0
        pan = (x - 0.5) * self.camera.tracking.pan_scale
        tilt = (y - 0.5) * self.camera.tracking.tilt_scale
        
        # Clamp to -1..1
        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))
        
        # Use AbsoluteMove if supported, otherwise this logic needs ContinuousMove with feedback loop
        # For now, we assume AbsoluteMove works for "Slew to Cue"
        try:
            # Run in thread to avoid blocking
            threading.Thread(
                target=self.onvif_client.ptz_move_absolute,
                args=(self.onvif_profile_token, pan, tilt, 0.0)
            ).start()
        except Exception as e:
            LOGGER.warning(f"Failed to track target on {self.camera.id}: {e}")

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
        rtsp_uri = build_ffmpeg_uri(self.camera.rtsp.uri, self.camera.rtsp.transport)
        
        # Frame skipping counter
        frame_count = 0
        skip_factor = self.camera.rtsp.frame_skip

        while not stop_event.is_set():
            loop = asyncio.get_running_loop()
            # Offload connection to thread as it can block
            cap = await loop.run_in_executor(None, cv2.VideoCapture, rtsp_uri, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
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
                    
                    # Skip inference on some frames to save CPU
                    frame_count += 1
                    if frame_count % skip_factor != 0:
                        continue

                    if self.camera.detect_enabled:
                        await self._process_frame(frame, frame_ts)
            finally:
                # Offload release to thread
                await loop.run_in_executor(None, cap.release)
            
            if not stop_event.is_set():
                await asyncio.sleep(1)  # Brief pause before reconnect

    async def _process_frame(self, frame: np.ndarray, ts: float) -> None:
        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(
            None, 
            self.detector.infer, 
            frame, 
            self.camera.thresholds.confidence
        )
        filtered = self._filter_detections(detections)
        
        # Tracking Logic
        if filtered and self.camera.tracking.enabled and self.peer_worker:
            # Find the most confident detection
            target = max(filtered, key=lambda d: d.confidence)
            # Calculate centroid (0..1)
            cx = (target.bbox[0] + target.bbox[2]) / 2
            cy = (target.bbox[1] + target.bbox[3]) / 2
            self.peer_worker.track_target(cx, cy)

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
            self.event_state = EventState(
                camera=self.camera,
                start_ts=ts,
                species={d.species for d in filtered},
                max_confidence=max(d.confidence for d in filtered),
                last_detection_ts=ts,
            )
            # Add pre-event frames from buffer, filtered by pre_seconds
            buffered = self.clip_buffer.dump()
            cutoff = ts - self.runtime.general.clip.pre_seconds
            self.event_state.frames.extend([f for f in buffered if f[0] >= cutoff])
            
        self.event_state.update(filtered, ts, frame)

    def _filter_detections(self, detections: List[Detection]) -> List[Detection]:
        includes = set(s.lower() for s in self.camera.include_species)
        excludes = set(s.lower() for s in self.camera.exclude_species)
        global_excludes = set(s.lower() for s in self.runtime.general.exclusion_list)
        filtered: List[Detection] = []
        for det in detections:
            label = det.species.lower()
            if includes and label not in includes:
                continue
            if label in excludes or label in global_excludes:
                continue
            filtered.append(det)
        return filtered

    async def _maybe_close_event(self, ts: float) -> None:
        if self.event_state is None:
            return
        idle = ts - self.event_state.last_detection_ts
        if idle < self.runtime.general.clip.post_seconds:
            return
        clip_path = self.storage.build_clip_path(
            self.camera.id,
            self.event_state.species_label,
            self.event_state.start_ts,
            self.runtime.general.clip.format,
        )
        
        # Offload clip writing and notification to thread
        loop = asyncio.get_running_loop()
        
        def finalize_event(frames, path, ctx, priority, sound):
            self.storage.write_clip(frames, path)
            self.notifier.send(ctx, priority=priority, sound=sound)
            LOGGER.info("Event for %s closed; clip at %s", ctx.camera_id, path)

        ctx = NotificationContext(
            species=self.event_state.species_label,
            confidence=self.event_state.max_confidence,
            camera_id=self.camera.id,
            camera_name=self.camera.name,
            clip_path=str(clip_path),
            event_started_at=self.event_state.start_ts,
            event_duration=self.event_state.duration,
        )
        
        loop.run_in_executor(
            None, 
            finalize_event, 
            self.event_state.frames, 
            clip_path, 
            ctx, 
            self.camera.notification.priority, 
            self.camera.notification.sound
        )
        
        self.event_state = None


class PipelineOrchestrator:
    def __init__(
        self,
        runtime: RuntimeConfig,
        model_path: str | None = None,
        engine: str = "cameratrapai",
        camera_filter: Optional[List[str]] = None,
    ) -> None:
        self.runtime = runtime
        self.detector = create_detector(engine, model_path=model_path)
        self.notifier = PushoverNotifier(
            runtime.general.notification.pushover_app_token_env,
            runtime.general.notification.pushover_user_key_env,
        )
        self.storage = StorageManager(
            storage_root=Path(self.runtime.general.storage_root),
            logs_root=Path(self.runtime.general.logs_root),
        )
        cameras = runtime.cameras
        if camera_filter:
            camera_set = {cid for cid in camera_filter}
            cameras = [cam for cam in cameras if cam.id in camera_set]
        self.registry = CameraRegistry.from_configs(cameras)
        self.cameras = cameras

    async def run(self) -> None:
        stop_event = asyncio.Event()
        workers = [
            StreamWorker(
                camera=cam,
                runtime=self.runtime,
                detector=self.detector,
                notifier=self.notifier,
                storage=self.storage,
            )
            for cam in self.cameras
        ]
        
        # Link workers for tracking
        worker_map = {w.camera.id: w for w in workers}
        for w in workers:
            if w.camera.tracking.target_camera_id:
                target = worker_map.get(w.camera.tracking.target_camera_id)
                if target:
                    w.peer_worker = target
                    LOGGER.info(f"Linked camera {w.camera.id} to target {target.camera.id}")
                else:
                    LOGGER.warning(f"Target camera {w.camera.tracking.target_camera_id} not found for {w.camera.id}")

        # Start web server
        web_server = WebServer(
            worker_map, 
            storage_root=Path(self.runtime.general.storage_root),
            logs_root=Path(self.runtime.general.logs_root),
            port=8080
        )
        
        await asyncio.gather(
            web_server.start(),
            *(worker.run(stop_event) for worker in workers)
        )
