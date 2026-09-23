"""An event does not wait for the camera to come back before it is saved.

Background (bug hunt, 2026-09-19): events are closed from the frame path,
by ``_process_frame`` once idle for ``post_seconds`` and by the read loop
at the maximum length. Neither runs without frames. An event open when the
RTSP stream dropped stayed open for the whole outage: no clip and no alert
until the camera came back, and then a clip that cut from the animal
straight to whatever was there at reconnection.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.pipeline import EventState, StreamWorker
from animaltracker.storage import StorageManager

POST_SECONDS = 5


class FakeWriter:
    frame_count = 30

    def __init__(self, temp_avi: Path) -> None:
        self.temp_avi = temp_avi

    def close(self) -> Path:
        return self.temp_avi


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, ctx, **kwargs) -> None:
        self.sent.append(ctx)


class DeadCamera:
    """cv2.VideoCapture for a camera that does not answer."""

    opened = 0

    def __init__(self, *args, **kwargs) -> None:
        DeadCamera.opened += 1

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def worker_with_open_event(tmp_path: Path, idle_for: float):
    storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    now = time.time()
    start = now - idle_for - 10
    temp_avi = storage.logs_root / "event_temp" / f"cam1_{int(start)}_0a1b2c3d.temp.avi"
    temp_avi.parent.mkdir(parents=True, exist_ok=True)
    temp_avi.write_bytes(b"the recording so far")

    def transcode(avi: Path, output: Path) -> bool:
        avi.unlink()
        output.write_bytes(b"mp4")
        return True

    storage.transcode_avi_to_mp4 = transcode
    camera = SimpleNamespace(
        id="cam1", name="Cam 1", exclude_species=[], detect_enabled=False,
        rtsp=SimpleNamespace(uri="rtsp://camera.invalid/stream", transport="tcp", hwaccel=False, frame_skip=1),
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    notifier = FakeNotifier()
    w = object.__new__(StreamWorker)
    w._tracker_lock = threading.Lock()
    w.camera = camera
    w.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=5, post_seconds=POST_SECONDS, format="mp4", max_event_seconds=300,
                             post_analysis=True, post_analysis_frames=0, unified_post_processing=False),
        detector=SimpleNamespace(), exclusion_list=[], notification=SimpleNamespace(web_base_url=None),
    ))
    w.detector = object()
    w.storage = storage
    w.notifier = notifier
    w.tracker = None
    w.ptz_tracker = None
    w.ptz_drives_tracking = False
    w.stream_connected = True
    w._init_log_state()
    w.event_state = EventState(
        camera=camera, start_ts=start, species={"animal"}, max_confidence=0.9,
        last_detection_ts=now - idle_for, clip_writer=FakeWriter(temp_avi),
    )
    return w, storage, notifier, start


@pytest.fixture
def dead_camera(monkeypatch):
    DeadCamera.opened = 0
    monkeypatch.setattr(pipeline_mod.cv2, "VideoCapture", DeadCamera)
    real_sleep = asyncio.sleep

    async def short_sleep(delay, *args, **kwargs):
        await real_sleep(min(delay, 0.01))

    monkeypatch.setattr(pipeline_mod.asyncio, "sleep", short_sleep)
    StreamWorker._postprocess_semaphore = None
    yield
    StreamWorker._postprocess_semaphore = None


def run_until(worker, done, timeout: float = 10.0) -> None:
    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(worker.run(stop))
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline and not done():
                await asyncio.sleep(0.02)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())


def test_an_event_idle_for_post_seconds_is_closed_while_the_camera_is_away(tmp_path, dead_camera):
    worker, storage, notifier, start = worker_with_open_event(tmp_path, idle_for=POST_SECONDS + 2)
    clip = storage.build_clip_path("cam1", "animal", start)

    run_until(worker, lambda: len(notifier.sent) == 1)

    assert DeadCamera.opened >= 1 and worker.stream_connected is False
    assert worker.event_state is None
    assert clip.exists() and [ctx.clip_path for ctx in notifier.sent] == [str(clip)]


def test_a_stream_that_may_be_back_in_a_moment_keeps_its_event(tmp_path, dead_camera):
    worker, storage, notifier, start = worker_with_open_event(tmp_path, idle_for=1)
    attempts = DeadCamera.opened

    run_until(worker, lambda: DeadCamera.opened >= attempts + 5)

    assert worker.event_state is not None            # still inside post_seconds: the same event carries on
    assert notifier.sent == []


def test_with_no_event_open_there_is_nothing_to_close(tmp_path, dead_camera):
    worker, _, notifier, _ = worker_with_open_event(tmp_path, idle_for=60)
    worker.event_state = None

    run_until(worker, lambda: DeadCamera.opened >= 3)

    assert notifier.sent == []


class SlowTracker:
    """An ObjectTracker whose update can be held mid-way, logging each call."""

    active_track_count = 1

    def __init__(self) -> None:
        self.calls: list = []
        self.inside_update = threading.Event()
        self.finish_update = threading.Event()

    def update(self, detections, frame, frame_idx):
        self.calls.append("update-start")
        self.inside_update.set()
        self.finish_update.wait(5)
        self.calls.append("update-end")
        return detections

    def prune_stale_tracks(self) -> int:
        return 0

    def get_unique_species(self):
        self.calls.append("read")
        return [("animal", 0.9)]

    def get_all_species(self):
        return {}

    def get_track_species(self, tid):
        return None, 0.0, None

    def reset(self) -> None:
        self.calls.append("reset")


def test_a_forced_close_waits_for_the_tracker_update_in_flight(tmp_path):
    # The read loop force-closes an event at max_event_seconds while the
    # inference task may be inside tracker.update() on an executor thread.
    # ObjectTracker has no lock of its own, and the close's reads raised
    # "dictionary changed size during iteration", which restarted the camera.
    worker, storage, notifier, start = worker_with_open_event(tmp_path, idle_for=1)
    tracker = SlowTracker()
    worker.tracker = tracker
    worker.event_state.tracker = tracker

    update = threading.Thread(target=worker._tick_live_tracker, args=([], None, 1))
    update.start()
    try:
        assert tracker.inside_update.wait(5)
        closing = threading.Thread(target=lambda: asyncio.run(worker._maybe_close_event(time.time(), force=True)))
        closing.start()
        time.sleep(0.2)
        assert "read" not in tracker.calls          # held back while the update runs
    finally:
        tracker.finish_update.set()
        update.join(5)
    closing.join(5)

    assert tracker.calls[:2] == ["update-start", "update-end"]
    assert tracker.calls.index("read") < tracker.calls.index("reset")
