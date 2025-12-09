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
from .detector import Detection, YoloDetector
from .notification import NotificationContext, PushoverNotifier
from .storage import StorageManager
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
    species: str
    max_confidence: float
    last_detection_ts: float
    frames: List[tuple[float, np.ndarray]] = field(default_factory=list)

    def update(self, detection: Detection, frame_ts: float, frame: np.ndarray) -> None:
        self.last_detection_ts = frame_ts
        if detection.confidence > self.max_confidence:
            self.max_confidence = detection.confidence
        self.frames.append((frame_ts, frame))

    @property
    def duration(self) -> float:
        return self.last_detection_ts - self.start_ts


class StreamWorker:
    def __init__(
        self,
        camera: CameraConfig,
        runtime: RuntimeConfig,
        detector: YoloDetector,
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
        self._snapshot_taken = False
        self.latest_frame: Optional[np.ndarray] = None

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
        if not filtered:
            await self._maybe_close_event(ts)
            return
        detection = filtered[0]
        if self.event_state is None:
            LOGGER.info("Started tracking %s on %s (%.2f)", detection.species, self.camera.id, detection.confidence)
            self.event_state = EventState(
                camera=self.camera,
                start_ts=ts,
                species=detection.species,
                max_confidence=detection.confidence,
                last_detection_ts=ts,
            )
            self.event_state.frames.extend(self.clip_buffer.dump())
        self.event_state.update(detection, ts, frame)

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
        if idle < 3.0:
            return
        clip_path = self.storage.build_clip_path(
            self.camera.id,
            self.event_state.species,
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
            species=self.event_state.species,
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
        model_path: str = "yolov8n.pt",
        camera_filter: Optional[List[str]] = None,
    ) -> None:
        self.runtime = runtime
        self.detector = YoloDetector(model_path=model_path)
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
        
        # Start web server
        worker_map = {w.camera.id: w for w in workers}
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
