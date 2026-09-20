"""One camera's unexpected error does not take the other cameras down.

Background (bug hunt, 2026-09-19): ``StreamWorker.run`` guards its inference
task but not its read loop, and the orchestrator gathered the workers bare.
An unexpected exception in one camera's loop ended its ``run()``, the gather
raised, and the whole process exited: every other camera and the web UI with
it. systemd starts it again, but a cause that persists (a temp directory that
cannot be written whenever an event starts) makes that a crash loop with
nothing recording at all.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.pipeline import EventState, PipelineOrchestrator, StreamWorker
from animaltracker.storage import StorageManager


@pytest.fixture
def orchestrator(monkeypatch):
    orch = object.__new__(PipelineOrchestrator)
    monkeypatch.setattr(PipelineOrchestrator, "_WORKER_RESTART_DELAY", 0.01)
    monkeypatch.setattr(PipelineOrchestrator, "_WORKER_RESTART_DELAY_MAX", 0.04)
    return orch


class FakeWorker:
    """``run`` raises ``failures`` times, then runs until asked to stop."""

    def __init__(self, camera_id: str, failures: int = 0, error: BaseException | None = None) -> None:
        self.camera = SimpleNamespace(id=camera_id)
        self.failures = failures
        self.error = error or RuntimeError(f"{camera_id}: the temp directory cannot be written")
        self.starts = 0
        self.recoveries = 0
        self.running = False
        self.cancelled = False

    async def run(self, stop_event: asyncio.Event) -> None:
        self.starts += 1
        if self.starts <= self.failures:
            raise self.error
        self.running = True
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.running = False

    async def _recover_after_error(self) -> None:
        self.recoveries += 1


async def until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.005)
    return condition()


def test_a_camera_that_fails_is_restarted_and_the_others_never_notice(orchestrator, caplog):
    caplog.set_level(logging.WARNING, logger="animaltracker.pipeline")
    broken, healthy = FakeWorker("cam1", failures=2), FakeWorker("cam2")

    async def main() -> None:
        stop = asyncio.Event()
        tasks = asyncio.gather(*(orchestrator._run_worker_supervised(w, stop) for w in (broken, healthy)))
        assert await until(lambda: broken.running and healthy.running)
        assert healthy.starts == 1 and not healthy.cancelled      # it ran straight through
        stop.set()
        await asyncio.wait_for(tasks, timeout=5)

    asyncio.run(main())

    assert broken.starts == 3 and broken.recoveries == 2
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2 and all(r.exc_info for r in errors)  # with the traceback, each time
    assert "the other cameras are unaffected" in errors[0].getMessage()


def test_bare_gather_is_what_used_to_take_everything_down():
    """The old wiring, kept as a record of the failure this replaces."""
    broken, healthy = FakeWorker("cam1", failures=1), FakeWorker("cam2")

    async def main() -> None:
        stop = asyncio.Event()
        with pytest.raises(RuntimeError):
            await asyncio.gather(broken.run(stop), healthy.run(stop))
        stop.set()

    asyncio.run(main())

    assert broken.starts == 1                          # and nothing ever started it again


def test_the_pause_grows_while_a_camera_keeps_failing(orchestrator, monkeypatch):
    waits: list = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout=None):
        waits.append(timeout)
        return await real_wait_for(awaitable, timeout=0.001)

    monkeypatch.setattr(pipeline_mod.asyncio, "wait_for", wait_for)
    broken = FakeWorker("cam1", failures=5)

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(orchestrator._run_worker_supervised(broken, stop))
        assert await until(lambda: broken.running)
        stop.set()
        await task

    asyncio.run(main())

    assert waits == [0.01, 0.02, 0.04, 0.04, 0.04]


def test_a_stop_during_the_pause_ends_it_at_once(orchestrator, monkeypatch):
    monkeypatch.setattr(PipelineOrchestrator, "_WORKER_RESTART_DELAY", 30.0)
    broken = FakeWorker("cam1", failures=99)

    async def main() -> float:
        stop = asyncio.Event()
        task = asyncio.ensure_future(orchestrator._run_worker_supervised(broken, stop))
        assert await until(lambda: broken.recoveries == 1)
        started = time.monotonic()
        stop.set()
        await asyncio.wait_for(task, timeout=5)
        return time.monotonic() - started

    assert asyncio.run(main()) < 1.0
    assert broken.starts == 1


def test_cancellation_is_not_an_error_to_recover_from(orchestrator):
    worker = FakeWorker("cam1")

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(orchestrator._run_worker_supervised(worker, stop))
        assert await until(lambda: worker.running)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())

    assert worker.starts == 1 and worker.recoveries == 0


# --- what the worker leaves behind -------------------------------------------------------

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


def worker_with_open_event(tmp_path: Path):
    storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    start = time.time() - 20
    temp_avi = storage.logs_root / "event_temp" / f"cam1_{int(start)}_0a1b2c3d.temp.avi"
    temp_avi.parent.mkdir(parents=True, exist_ok=True)
    temp_avi.write_bytes(b"the recording so far")

    def transcode(avi: Path, output: Path) -> bool:
        avi.unlink()
        output.write_bytes(b"mp4")
        return True

    storage.transcode_avi_to_mp4 = transcode
    camera = SimpleNamespace(
        id="cam1", name="Cam 1", exclude_species=[],
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    w = object.__new__(StreamWorker)
    w.camera = camera
    w.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=5, post_seconds=5, format="mp4", post_analysis=True,
                             post_analysis_frames=0, unified_post_processing=False),
        detector=SimpleNamespace(), exclusion_list=[], notification=SimpleNamespace(web_base_url=None),
    ))
    w.detector = object()
    w.storage = storage
    w.notifier = FakeNotifier()
    w.tracker = None
    w.ptz_tracker = None
    w.ptz_drives_tracking = False
    w.stream_connected = True
    w.event_state = EventState(
        camera=camera, start_ts=start, species={"animal"}, max_confidence=0.9,
        last_detection_ts=time.time() - 1, clip_writer=FakeWriter(temp_avi),
    )
    return w, storage, start


def test_the_event_that_was_open_is_saved_and_alerts(tmp_path):
    worker, storage, start = worker_with_open_event(tmp_path)
    StreamWorker._postprocess_semaphore = None

    async def main() -> None:
        await worker._recover_after_error()
        assert await until(lambda: len(worker.notifier.sent) == 1)

    asyncio.run(main())

    assert worker.event_state is None and worker.stream_connected is False
    assert storage.build_clip_path("cam1", "animal", start).exists()


def test_an_event_that_cannot_even_be_closed_is_dropped_so_the_camera_can_go_on(tmp_path, monkeypatch, caplog):
    worker, _, _ = worker_with_open_event(tmp_path)

    async def broken_close(ts, force=False):
        raise OSError("No space left on device")

    monkeypatch.setattr(worker, "_maybe_close_event", broken_close)

    asyncio.run(worker._recover_after_error())

    assert worker.event_state is None
    assert "dropping it" in caplog.text


def test_with_no_event_open_there_is_nothing_to_do(tmp_path):
    worker, _, _ = worker_with_open_event(tmp_path)
    worker.event_state = None

    asyncio.run(worker._recover_after_error())

    assert worker.notifier.sent == []


def test_the_orchestrator_supervises_every_worker_it_gathers():
    import inspect

    source = inspect.getsource(PipelineOrchestrator.run)
    assert "_run_worker_supervised(worker, stop_event) for worker in workers" in source
    assert "worker.run(stop_event) for worker in workers" not in source


# --- the real run loop, dying mid-read ---------------------------------------------------

class ExplodingCamera:
    """cv2.VideoCapture whose first stream blows up inside the read loop."""

    opened = 0

    def __init__(self, *args, **kwargs) -> None:
        ExplodingCamera.opened += 1
        self.first = ExplodingCamera.opened == 1

    def isOpened(self) -> bool:
        return self.first                      # later attempts: the camera does not answer

    def read(self):
        raise RuntimeError("the decoder fell over")

    def release(self) -> None:
        pass


def test_the_real_run_loop_is_restarted_after_it_dies_mid_read(tmp_path, orchestrator, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="animaltracker.pipeline")
    ExplodingCamera.opened = 0
    monkeypatch.setattr(pipeline_mod.cv2, "VideoCapture", ExplodingCamera)
    real_sleep = asyncio.sleep

    async def short_sleep(delay, *args, **kwargs):
        await real_sleep(min(delay, 0.01))

    monkeypatch.setattr(pipeline_mod.asyncio, "sleep", short_sleep)
    worker, _, _ = worker_with_open_event(tmp_path)
    worker.event_state = None
    worker.camera.detect_enabled = False
    worker.camera.rtsp = SimpleNamespace(uri="rtsp://camera.invalid/stream", transport="tcp",
                                         hwaccel=False, frame_skip=1)
    worker._init_log_state()

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.ensure_future(orchestrator._run_worker_supervised(worker, stop))
        assert await until(lambda: ExplodingCamera.opened >= 4)     # it died once and kept trying
        assert not task.done()
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())

    assert "the decoder fell over" in caplog.text
    assert "Restarting the worker for cam1" in caplog.text
