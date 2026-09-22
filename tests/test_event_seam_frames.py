"""Every captured frame reaches the clip once.

The read loop pushes each frame into the rolling buffer, and when an event
has just opened it builds the clip writer and seeds it from that buffer, so
the frame it is holding is already queued. It then wrote that same frame
again as the first live frame: one duplicate at the seam of every clip, a
frame of stutter between the pre-roll and the live part. Found in the bug
hunt of 2026-09-19 (item 1.21).

The real ``run()`` loop is driven here with a fake camera and a recording
stand-in for the writer.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.clip_buffer import ClipBuffer
from animaltracker.pipeline import EventState, StreamWorker
from animaltracker.storage import StorageManager


class Camera:
    """cv2.VideoCapture stand-in: frames numbered by their pixel value."""

    def __init__(self, *args, **kwargs) -> None:
        self.n = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.n += 1
        if self.n > Camera.frames:
            time.sleep(0.01)
            return False, None            # "stream lost": the loop leaves and checks stop
        time.sleep(0.002)
        return True, np.full((8, 8, 3), self.n, dtype=np.uint8)

    def release(self) -> None:
        pass


Camera.frames = 12


class RecordingWriter:
    """StreamingClipWriter stand-in: remembers every frame it is offered."""
    instances: list = []

    def __init__(self, temp_path, fps=15, max_pending=300, stride=1) -> None:
        self.temp_path = Path(temp_path)
        self.fps, self.stride, self.capture_fps = fps, stride, fps
        self.seeded: list = []
        self.written: list = []
        self.frame_count = 0
        RecordingWriter.instances.append(self)

    def seed(self, frames):
        self.seeded = [int(f[0, 0, 0]) for f in frames]
        self.frame_count += len(self.seeded)
        return len(self.seeded)

    def write(self, frame):
        self.written.append(int(frame[0, 0, 0]))
        self.frame_count += 1

    def close(self):
        return None


def worker(tmp_path) -> StreamWorker:
    storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    camera = SimpleNamespace(
        id="cam1", name="Cam 1", exclude_species=[], detect_enabled=False,
        rtsp=SimpleNamespace(uri="rtsp://camera.invalid/stream", transport="tcp", hwaccel=False, frame_skip=1),
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    w = object.__new__(StreamWorker)
    w.camera = camera
    w.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=10, post_seconds=5, max_event_seconds=300, format="mp4",
                             post_analysis=False, post_analysis_frames=0, unified_post_processing=False),
        detector=SimpleNamespace(), exclusion_list=[], notification=SimpleNamespace(web_base_url=None),
    ))
    w.detector = object()
    w.storage = storage
    w.notifier = SimpleNamespace(send=lambda *a, **k: None)
    w.tracker = None
    w.ptz_tracker = None
    w.ptz_drives_tracking = False
    w.stream_connected = True
    w.clip_buffer = ClipBuffer(max_seconds=30, fps=15)
    w._record_stride = 1
    for name in ("perf_infer_count", "perf_frames_read", "perf_frames_dropped_busy"):
        setattr(w, name, 0)
    for name in ("perf_infer_time_total", "perf_infer_time_max", "perf_frame_age_total", "perf_frame_age_max"):
        setattr(w, name, 0.0)
    w._perf_window_start = time.time()
    w._perf_log_interval = 10.0
    w.perf_last_snapshot = {}
    w._snapshot_taken = True
    w._init_log_state()
    # An event that opened a moment ago and has no writer yet: the first
    # frame the loop reads is the one that builds and seeds it.
    w.event_state = EventState(
        camera=camera, start_ts=time.time(), species={"animal"}, max_confidence=0.9,
        last_detection_ts=time.time(), clip_writer=None,
    )
    return w


def test_the_frame_that_opens_the_writer_is_not_written_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod.cv2, "VideoCapture", Camera)
    monkeypatch.setattr(pipeline_mod, "StreamingClipWriter", RecordingWriter)
    RecordingWriter.instances = []
    w = worker(tmp_path)

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(w.run(stop))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (
                RecordingWriter.instances and len(RecordingWriter.instances[0].written) >= Camera.frames - 1):
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())

    assert len(RecordingWriter.instances) == 1
    rec = RecordingWriter.instances[0]
    got = rec.seeded + rec.written
    assert rec.seeded == [1], "the opening frame comes from the buffer, once"
    assert got == list(range(1, Camera.frames + 1)), got


def test_frames_before_the_event_are_the_seed_and_none_repeats(tmp_path, monkeypatch):
    """With a pre-roll in the buffer the seam is between it and the first
    live frame; every captured frame still appears exactly once."""
    monkeypatch.setattr(pipeline_mod.cv2, "VideoCapture", Camera)
    monkeypatch.setattr(pipeline_mod, "StreamingClipWriter", RecordingWriter)
    RecordingWriter.instances = []
    w = worker(tmp_path)
    now = time.time()
    for i in range(-3, 0):                                   # frames -3..-1 captured before the event
        w.clip_buffer.push(now - 1 + i * 0.05, np.full((8, 8, 3), 200 + i, dtype=np.uint8))

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(w.run(stop))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (
                RecordingWriter.instances and len(RecordingWriter.instances[0].written) >= Camera.frames - 1):
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())

    rec = RecordingWriter.instances[0]
    assert rec.seeded == [197, 198, 199, 1]
    assert rec.written == list(range(2, Camera.frames + 1))
    assert len(set(rec.seeded + rec.written)) == len(rec.seeded + rec.written)
