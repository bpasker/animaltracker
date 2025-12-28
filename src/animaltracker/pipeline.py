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
from .detector import Detection, BaseDetector, create_detector
from .notification import NotificationContext, PushoverNotifier
from .storage import StorageManager
from .onvif_client import OnvifClient
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


# Keep GStreamer pipelines for systems that have it (e.g., Jetson)
def build_gstreamer_pipeline(rtsp_uri: str, transport: str = "tcp", latency_ms: int = 0) -> str:
    """Build GStreamer pipeline string for NVDEC hardware decoding.
    
    Uses NVIDIA's nvv4l2decoder for hardware H264/H265 decoding on GTX/RTX GPUs.
    Falls back to nvdec if nvv4l2decoder not available.
    """
    protocols = "tcp" if transport == "tcp" else "udp"
    latency = max(latency_ms, 100)  # Minimum 100ms for stable streaming
    
    # GStreamer pipeline for NVDEC hardware decoding
    # Works with GTX 1080, RTX series, Jetson, etc.
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

    def update(self, detections: List[Detection], frame_ts: float, frame: np.ndarray) -> None:
        self.last_detection_ts = frame_ts
        for det in detections:
            self.species.add(det.species)
            if det.confidence > self.max_confidence:
                self.max_confidence = det.confidence
            
            # Track top N frames for each species (highest confidence)
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
        detector: BaseDetector,
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
        skip_factor = self.camera.rtsp.frame_skip
        
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
        
        # Offload clip writing, post-analysis, and notification to thread
        loop = asyncio.get_running_loop()
        post_analysis_enabled = self.runtime.general.clip.post_analysis
        post_analysis_frames = self.runtime.general.clip.post_analysis_frames
        
        def finalize_event(frames, camera_id, start_ts, clip_format, ctx_base, priority, sound, species_key_frames):
            """Finalize event with optional post-clip species analysis."""
            final_species = ctx_base['species']
            final_confidence = ctx_base['confidence']
            
            # Step 1: Run post-clip analysis if enabled
            if post_analysis_enabled:
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
            
            # Step 4: Save detection thumbnails for each species
            if species_key_frames:
                self.storage.save_detection_thumbnails(clip_path, species_key_frames)
            
            # Step 5: Send notification with refined info
            ctx = NotificationContext(
                species=final_species,
                confidence=final_confidence,
                camera_id=ctx_base['camera_id'],
                camera_name=ctx_base['camera_name'],
                clip_path=str(clip_path),
                event_started_at=ctx_base['event_started_at'],
                event_duration=ctx_base['event_duration'],
            )
            self.notifier.send(ctx, priority=priority, sound=sound)
            LOGGER.info("Event for %s closed; clip at %s (species: %s)", 
                       ctx.camera_id, clip_path, final_species)

        ctx_base = {
            'species': self.event_state.species_label,
            'confidence': self.event_state.max_confidence,
            'camera_id': self.camera.id,
            'camera_name': self.camera.name,
            'event_started_at': self.event_state.start_ts,
            'event_duration': self.event_state.duration,
        }
        
        # Copy species key frames before clearing event state
        species_key_frames = dict(self.event_state.species_key_frames)
        
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
            species_key_frames
        )
        
        self.event_state = None

    def _analyze_clip_frames(self, frames: List[tuple], num_samples: int = 5) -> tuple[str, float]:
        """Analyze frames from a clip to get the most specific species identification.
        
        Samples frames evenly throughout the clip, runs detection on each,
        and returns the most specific/confident identification.
        
        Args:
            frames: List of (timestamp, frame) tuples
            num_samples: Number of frames to analyze (default 5)
            
        Returns:
            (species_label, confidence) - best identification found
        """
        if not frames:
            return "", 0.0
        
        # Sample frames evenly throughout the clip
        total_frames = len(frames)
        if total_frames <= num_samples:
            sample_indices = list(range(total_frames))
        else:
            step = total_frames / num_samples
            sample_indices = [int(i * step) for i in range(num_samples)]
        
        # Collect all detections across sampled frames
        all_detections = []
        for idx in sample_indices:
            _, frame = frames[idx]
            try:
                # Use lower thresholds for post-analysis to catch more species
                detections = self.detector.infer(
                    frame, 
                    conf_threshold=0.3,
                    generic_confidence=0.5  # Lower generic threshold for post-analysis
                )
                all_detections.extend(detections)
            except Exception as e:
                LOGGER.warning("Post-clip analysis failed on frame %d: %s", idx, e)
                continue
        
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
        
        # Create detector from config settings
        detector_cfg = runtime.general.detector
        self.detector = create_detector(
            backend=detector_cfg.backend,
            model_path=model_path or detector_cfg.model_path,
            model_version=detector_cfg.speciesnet_version,
            country=detector_cfg.country,
            admin1_region=detector_cfg.admin1_region,
            latitude=detector_cfg.latitude,
            longitude=detector_cfg.longitude,
            generic_confidence=detector_cfg.generic_confidence,
        )
        LOGGER.info(f"Using {self.detector.backend_name} detector backend")
        
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
        
        worker_map = {w.camera.id: w for w in workers}

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
