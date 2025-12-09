"""Streaming pipeline orchestration."""
from __future__ import annotations

import asyncio
import logging
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

LOGGER = logging.getLogger(__name__)


def build_gstreamer_pipeline(rtsp_uri: str, transport: str = "tcp", latency_ms: int = 0) -> str:
    protocol_flag = "protocols=tcp" if transport.lower() == "tcp" else ""
    return " ".join([
        f"rtspsrc location={rtsp_uri} {protocol_flag} latency={latency_ms} drop-on-late=true",
        "! queue",
        "! application/x-rtp,media=video,encoding-name=H264",
        "! rtph264depay",
        "! h264parse",
        "! avdec_h264",
        "! videoconvert",
        "! appsink drop=1 sync=false",
    ])


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
        clip_seconds = runtime.general.clip.pre_seconds + runtime.general.clip.post_seconds
        self.clip_buffer = ClipBuffer(max_seconds=clip_seconds, fps=15)
        self.event_state: Optional[EventState] = None

    async def run(self, stop_event: asyncio.Event) -> None:
        LOGGER.info("Starting worker for %s", self.camera.id)
        pipeline = build_gstreamer_pipeline(
            self.camera.rtsp.uri,
            self.camera.rtsp.transport,
            self.camera.rtsp.latency_ms,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open RTSP stream for {self.camera.id}")
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0)
                ret, frame = cap.read()
                if not ret:
                    LOGGER.warning("Frame grab failed for %s; retrying", self.camera.id)
                    await asyncio.sleep(0.1)
                    continue
                frame_ts = time.time()
                self.clip_buffer.push(frame_ts, frame)
                await self._process_frame(frame, frame_ts)
        finally:
            cap.release()

    async def _process_frame(self, frame: np.ndarray, ts: float) -> None:
        detections = self.detector.infer(frame, conf_threshold=self.camera.thresholds.confidence)
        filtered = self._filter_detections(detections)
        if not filtered:
            self._maybe_close_event(ts)
            return
        detection = filtered[0]
        if self.event_state is None:
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

    def _maybe_close_event(self, ts: float) -> None:
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
        self.storage.write_clip(self.event_state.frames, clip_path)
        ctx = NotificationContext(
            species=self.event_state.species,
            confidence=self.event_state.max_confidence,
            camera_id=self.camera.id,
            camera_name=self.camera.name,
            clip_path=str(clip_path),
            event_started_at=self.event_state.start_ts,
            event_duration=self.event_state.duration,
        )
        self.notifier.send(
            ctx,
            priority=self.camera.notification.priority,
            sound=self.camera.notification.sound,
        )
        LOGGER.info("Event for %s closed; clip at %s", self.camera.id, clip_path)
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
        await asyncio.gather(*(worker.run(stop_event) for worker in workers))
