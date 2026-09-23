"""An event's recording must survive a restart, a full queue and the CLI.

Background (bug hunt, 2026-09-19): while an event records, its frames go to
``logs_root/event_temp/<camera>_<epoch>_<tag>.temp.avi``, the only copy until
it is transcoded into the clip. Three things conspired to lose it:

* the transcode ran only once the job held a post-processing slot, and a
  SpeciesNet job on a long clip holds a slot for most of an hour;
* ``StorageManager.__post_init__`` deleted every ``*.temp.avi`` it found, so
  a deploy in that window deleted every queued event, clip and alert, with no
  MP4 for the recovery sweep to find;
* ``cleanup`` builds a StorageManager on the same config, so running it, even
  with ``--dry-run``, deleted the running service's open recordings.

Now the clip is saved before the job queues, building a StorageManager
touches nothing, and what a previous run left behind is turned into
``<epoch>_animal.mp4`` clips that the recovery sweep analyses.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker.analysis_recovery import (
    RECOVERED_CLIP_LABEL,
    UNCLASSIFIED_LABELS,
    find_unfinished_clips,
    is_unclassified_clip,
)
from animaltracker.pipeline import AnalysisWorkers, EventState, PipelineOrchestrator, StreamWorker
from animaltracker.storage import StorageManager

EVENT_TS = 1789000000  # 2026-09-10 in any zone a test host is likely to use


def _storage(tmp_path: Path) -> StorageManager:
    return StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")


def _temp(storage: StorageManager, name: str, payload: bytes = b"RIFF....AVI ") -> Path:
    path = storage.logs_root / "event_temp" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fake_transcode(calls: list):
    """Stand-in with the real contract on success: write the clip, drop the AVI."""
    def transcode(temp_avi: Path, output_path: Path) -> bool:
        calls.append((temp_avi.name, output_path))
        # The clip appears last: a test that waits for it on another thread
        # may look at everything else the moment it exists.
        temp_avi.unlink()
        output_path.write_bytes(b"mp4")
        return True
    return transcode


# --- building a StorageManager is harmless -------------------------------------

def test_constructing_a_storage_manager_leaves_open_recordings_alone(tmp_path):
    first = _storage(tmp_path)
    live = _temp(first, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")

    # What ``cleanup --dry-run`` does next to a running service.
    second = StorageManager(storage_root=first.storage_root, logs_root=first.logs_root)

    assert live.exists()
    assert second.list_orphan_event_temps() == [live]


def test_orphans_are_listed_oldest_event_first_and_a_missing_directory_is_empty(tmp_path):
    storage = _storage(tmp_path)
    assert storage.list_orphan_event_temps() == []

    late = _temp(storage, f"cam1_{EVENT_TS + 500}_aaaaaaaa.temp.avi")
    early = _temp(storage, f"otteson2_{EVENT_TS}_ffffffff.temp.avi")
    _temp(storage, "notes.txt")

    assert storage.list_orphan_event_temps() == [early, late]


# --- an orphan becomes the clip its event would have saved ----------------------

def test_an_orphan_is_saved_where_the_live_path_would_have_put_it(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")

    clip = storage.recover_orphan_event_temp(orphan, RECOVERED_CLIP_LABEL)

    assert clip == storage.build_clip_path("cam1", "animal", EVENT_TS)
    assert clip.exists() and not orphan.exists()
    assert clip.relative_to(storage.storage_root / "clips").parts[0] == "cam1"
    # ...and it is exactly what the recovery sweep looks for.
    assert RECOVERED_CLIP_LABEL in UNCLASSIFIED_LABELS
    assert is_unclassified_clip(clip)
    assert find_unfinished_clips(storage.storage_root / "clips", min_age_s=0) == [clip]


def test_a_camera_id_with_underscores_is_read_from_the_end(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    orphan = _temp(storage, f"back_yard_2_{EVENT_TS}_0a1b2c3d.temp.avi")

    clip = storage.recover_orphan_event_temp(orphan, "animal")

    assert clip == storage.build_clip_path("back_yard_2", "animal", EVENT_TS)


@pytest.mark.parametrize("saved_as", [f"{EVENT_TS}_animal.mp4", f"{EVENT_TS}_mammalia_carnivora_canidae.mp4"])
def test_an_orphan_whose_clip_was_already_saved_is_just_removed(tmp_path, monkeypatch, saved_as):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    saved = storage.build_clip_path("cam1", "animal", EVENT_TS).with_name(saved_as)
    saved.write_bytes(b"the real clip")
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")

    assert storage.recover_orphan_event_temp(orphan, "animal") is None

    assert calls == [] and not orphan.exists()
    assert saved.read_bytes() == b"the real clip"


def test_a_half_written_transcode_does_not_count_as_the_saved_clip(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)
    clip.with_suffix(".tmp.mp4").write_bytes(b"ffmpeg was killed here")
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")

    assert storage.recover_orphan_event_temp(orphan, "animal") == clip
    assert len(calls) == 1


def test_a_file_that_is_not_an_event_recording_is_removed(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    stray = _temp(storage, "leftover.temp.avi")

    assert storage.recover_orphan_event_temp(stray, "animal") is None
    assert calls == [] and not stray.exists()


def test_a_recording_that_cannot_be_read_yields_no_clip(tmp_path):
    storage = _storage(tmp_path)
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi", payload=b"")  # empty: the real transcode refuses it

    assert storage.recover_orphan_event_temp(orphan, "animal") is None
    assert not orphan.exists()
    assert list((storage.storage_root / "clips").rglob("*.mp4")) == []


def _failing_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    """Put an ffmpeg on PATH that fails the way a full disk makes it fail."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\necho 'No space left on device' >&2\nexit 1\n")
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")


def test_a_failed_transcode_keeps_the_recording(tmp_path, monkeypatch):
    # The AVI is the event's only copy until the MP4 exists. A transcode
    # that fails for a passing reason (ENOSPC, the NFS archive away) used to
    # delete it all the same.
    _failing_ffmpeg(tmp_path, monkeypatch)
    storage = _storage(tmp_path)
    recording = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    assert storage.transcode_avi_to_mp4(recording, clip) is False

    assert recording.exists()
    assert not clip.exists()
    assert not clip.with_suffix(".tmp.mp4").exists()


def test_an_orphan_that_fails_to_transcode_waits_for_the_next_startup(tmp_path, monkeypatch):
    _failing_ffmpeg(tmp_path, monkeypatch)
    storage = _storage(tmp_path)
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")

    assert storage.recover_orphan_event_temp(orphan, RECOVERED_CLIP_LABEL) is None

    assert orphan.exists()
    assert storage.list_orphan_event_temps() == [orphan]


def test_a_real_recording_is_transcoded_end_to_end(tmp_path):
    storage = _storage(tmp_path)
    orphan = storage.logs_root / "event_temp" / f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(orphan), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG test recordings")
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()

    clip = storage.recover_orphan_event_temp(orphan, RECOVERED_CLIP_LABEL)

    if clip is None:
        pytest.skip("no H.264 encoder here (neither ffmpeg nor OpenCV avc1)")
    assert not orphan.exists()
    cap = cv2.VideoCapture(str(clip))
    try:
        assert cap.isOpened() and cap.read()[0]
    finally:
        cap.release()


# --- the pipeline's startup pass -------------------------------------------------

class FakeSweeper:
    min_age_s = 0.0

    def __init__(self) -> None:
        self.sweeps_requested = 0

    def request_sweep(self) -> None:
        self.sweeps_requested += 1


def _orchestrator(storage: StorageManager) -> PipelineOrchestrator:
    orch = object.__new__(PipelineOrchestrator)
    orch.storage = storage
    orch._orphan_stop = threading.Event()
    return orch


def test_startup_saves_every_orphan_and_then_asks_for_a_sweep(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")
    _temp(storage, f"cam2_{EVENT_TS + 60}_1b2c3d4e.temp.avi")
    _temp(storage, "leftover.temp.avi")
    orch = _orchestrator(storage)
    sweeper = FakeSweeper()
    # Wait out the "let the clips age" pause at once.
    monkeypatch.setattr(orch._orphan_stop, "wait", lambda timeout=None: False)

    orch._recover_orphan_recordings(storage.list_orphan_event_temps(), sweeper)

    clips = sorted(p.name for p in (storage.storage_root / "clips").rglob("*.mp4"))
    assert clips == [f"{EVENT_TS}_animal.mp4", f"{EVENT_TS + 60}_animal.mp4"]
    assert storage.list_orphan_event_temps() == []
    assert sweeper.sweeps_requested == 1


def test_startup_with_nothing_worth_saving_asks_for_no_sweep(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    _temp(storage, "leftover.temp.avi")
    sweeper = FakeSweeper()

    _orchestrator(storage)._recover_orphan_recordings(storage.list_orphan_event_temps(), sweeper)

    assert sweeper.sweeps_requested == 0


def test_a_stopping_pipeline_leaves_the_remaining_orphans_for_next_time(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    orphan = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")
    orch = _orchestrator(storage)
    orch._orphan_stop.set()
    sweeper = FakeSweeper()

    orch._recover_orphan_recordings([orphan], sweeper)

    assert orphan.exists() and sweeper.sweeps_requested == 0


# --- a closing event saves its clip before it queues --------------------------

class FakeWriter:
    def __init__(self, temp_avi: Path) -> None:
        self.temp_avi = temp_avi
        self.frame_count = 42

    def close(self) -> Path:
        return self.temp_avi


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, ctx, **kwargs) -> None:
        self.sent.append(ctx)


def _closing_worker(storage: StorageManager, temp_avi: Path, notifier: FakeNotifier,
                    camera_id: str = "cam1") -> StreamWorker:
    camera = SimpleNamespace(
        id=camera_id, name=camera_id.title(), exclude_species=[],
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    clip_cfg = SimpleNamespace(
        pre_seconds=5, post_seconds=5, format="mp4",
        post_analysis=True, post_analysis_frames=0,
        unified_post_processing=False,  # the analysis itself is not under test
    )
    worker = object.__new__(StreamWorker)
    worker.camera = camera
    worker.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=clip_cfg, detector=SimpleNamespace(), exclusion_list=[],
        notification=SimpleNamespace(web_base_url=None),
    ))
    worker.detector = object()
    worker.storage = storage
    worker.notifier = notifier
    worker.tracker = None
    worker.ptz_tracker = None
    worker.ptz_drives_tracking = False
    worker.event_state = EventState(
        camera=camera, start_ts=float(EVENT_TS), species={"animal"},
        max_confidence=0.9, last_detection_ts=float(EVENT_TS + 12),
        clip_writer=FakeWriter(temp_avi),
    )
    return worker


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def one_busy_slot():
    """A single post-processing slot, held by "another clip" for the test."""
    StreamWorker.set_postprocess_limit(1)
    slot = StreamWorker._ensure_postprocess_semaphore()
    assert slot.acquire(timeout=1)
    released = []

    def release() -> None:
        if not released:
            released.append(True)
            slot.release()

    try:
        yield release
    finally:
        release()
        StreamWorker._postprocess_semaphore = None
        StreamWorker._postprocess_limit = 2


def test_the_clip_is_saved_while_every_post_processing_slot_is_busy(tmp_path, monkeypatch, one_busy_slot):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    temp_avi = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")
    notifier = FakeNotifier()
    worker = _closing_worker(storage, temp_avi, notifier)
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    async def close_event() -> None:
        await worker._maybe_close_event(float(EVENT_TS + 30), force=True)
        # finalize_event now runs in the default executor; give it the floor.
        loop = asyncio.get_running_loop()
        try:
            assert await loop.run_in_executor(None, _wait_for, clip.exists)

            # Saved, although the job has not got its slot yet: a restart now
            # costs the analysis (which the sweep redoes), not the event.
            assert [name for name, _ in calls] == [temp_avi.name]
            assert not temp_avi.exists()
            assert notifier.sent == []
            assert StreamWorker.analysis_registry.is_active(clip)
        finally:
            # Always, or a failed assertion leaves the job blocked and
            # asyncio.run() waiting minutes for its thread.
            one_busy_slot()
        assert await loop.run_in_executor(None, _wait_for, lambda: len(notifier.sent) == 1)

    asyncio.run(close_event())

    assert notifier.sent[0].clip_path == str(clip)
    assert _wait_for(lambda: not StreamWorker.analysis_registry.is_active(clip))
    assert worker.event_state is None


class SlowWriter(FakeWriter):
    """A writer with frames still queued: ``close()`` takes as long as the drain."""

    def __init__(self, temp_avi: Path) -> None:
        super().__init__(temp_avi)
        self.drained = threading.Event()
        self.closed_on: list = []

    def close(self) -> Path:
        self.closed_on.append(threading.current_thread())
        assert self.drained.wait(10), "the test never let the drain finish"
        return self.temp_avi


def test_closing_an_event_does_not_wait_for_its_writer_to_drain(tmp_path, monkeypatch):
    """The close runs inside the camera's inference task, and until that task
    ends the read loop drops every frame. Awaiting the drain there blinded the
    camera for as long as the encoder needed for its queue: some twenty
    seconds after every event where it runs at half the capture rate."""
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    temp_avi = _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi")
    notifier = FakeNotifier()
    worker = _closing_worker(storage, temp_avi, notifier)
    writer = SlowWriter(temp_avi)
    worker.event_state.clip_writer = writer
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    async def close_event() -> None:
        loop_thread = threading.current_thread()
        started = time.monotonic()
        try:
            await asyncio.wait_for(worker._maybe_close_event(float(EVENT_TS + 30), force=True), timeout=2.0)
            elapsed = time.monotonic() - started
        except BaseException:
            writer.drained.set()
            raise

        # Back at once, with the recording still draining: the inference
        # task is free and the camera can detect again.
        if not (elapsed < 1.0 and worker.event_state is None and not clip.exists()):
            writer.drained.set()
        assert elapsed < 1.0
        assert worker.event_state is None
        assert not clip.exists()

        loop = asyncio.get_running_loop()
        try:
            assert await loop.run_in_executor(None, _wait_for, lambda: bool(writer.closed_on))
            assert writer.closed_on[0] is not loop_thread        # drained on the job's own thread
            assert StreamWorker.analysis_registry.is_active(clip)  # and claimed while it drains
        finally:
            writer.drained.set()
        assert await loop.run_in_executor(None, _wait_for, lambda: len(notifier.sent) == 1)

    asyncio.run(close_event())

    assert clip.exists() and notifier.sent[0].clip_path == str(clip)


def test_a_writer_that_never_produced_a_file_yields_no_clip_and_no_alert(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    calls: list = []
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode(calls))
    notifier = FakeNotifier()
    worker = _closing_worker(storage, storage.logs_root / "event_temp" / "never.temp.avi", notifier)

    class FailedWriter(FakeWriter):
        def close(self):
            return None                                    # what StreamingClipWriter returns on failure

    worker.event_state.clip_writer = FailedWriter(Path("unused"))
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    async def close_event() -> None:
        await worker._maybe_close_event(float(EVENT_TS + 30), force=True)
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(
            None, _wait_for, lambda: not StreamWorker.analysis_registry.is_active(clip))

    asyncio.run(close_event())

    assert calls == [] and notifier.sent == [] and not clip.exists()


def test_an_event_with_no_recording_is_dropped_without_waiting_for_a_slot(tmp_path, one_busy_slot):
    storage = _storage(tmp_path)
    worker = _closing_worker(storage, storage.logs_root / "event_temp" / "gone.temp.avi", FakeNotifier())
    worker.event_state.clip_writer = None

    assert worker._save_event_clip(None, storage.build_clip_path("cam1", "animal", EVENT_TS)) is False


# --- a waiting analysis is a queue entry, not a thread ---------------------------

def test_events_that_close_while_every_slot_is_busy_do_not_use_up_the_shared_executor(
        tmp_path, monkeypatch, one_busy_slot):
    """Analysis jobs used to wait for their slot inside a thread of the event
    loop's default executor, the pool every camera read and inference call
    needs. Two threads stand in for it here: four queued events would have
    pinned both, the last two recordings would never even have been saved,
    and nothing else could have run."""
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))
    notifier = FakeNotifier()
    cameras = ["cam1", "cam2", "otteson1", "otteson2"]
    workers = [
        _closing_worker(storage, _temp(storage, f"{cam}_{EVENT_TS}_0a1b2c3d.temp.avi"), notifier, camera_id=cam)
        for cam in cameras
    ]
    clips = [storage.build_clip_path(cam, "animal", EVENT_TS) for cam in cameras]
    registry = StreamWorker.analysis_registry

    async def close_events() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=2))
        try:
            for worker in workers:
                await worker._maybe_close_event(float(EVENT_TS + 30), force=True)

            # Every clip is saved although no analysis can start...
            saved = await asyncio.wait_for(
                loop.run_in_executor(None, _wait_for, lambda: all(c.exists() for c in clips)), timeout=10)
            assert saved
            # ...and the shared pool is free again: this would never be scheduled otherwise.
            assert await asyncio.wait_for(loop.run_in_executor(None, lambda: "a camera read"), timeout=2) == "a camera read"
            assert all(registry.is_active(c) for c in clips) and registry.live_busy()
            assert notifier.sent == []
        finally:
            one_busy_slot()
        assert await loop.run_in_executor(None, _wait_for, lambda: len(notifier.sent) == 4)

    asyncio.run(close_events())

    assert sorted(ctx.camera_id for ctx in notifier.sent) == sorted(cameras)
    assert _wait_for(lambda: not registry.live_busy() and not any(registry.is_active(c) for c in clips))


def test_an_analysis_that_raises_still_releases_its_clip(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage, "transcode_avi_to_mp4", _fake_transcode([]))

    class ExplodingNotifier(FakeNotifier):
        def send(self, ctx, **kwargs) -> None:
            super().send(ctx, **kwargs)
            raise RuntimeError("pushover is down")

    notifier = ExplodingNotifier()
    worker = _closing_worker(storage, _temp(storage, f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi"), notifier)
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)
    registry = StreamWorker.analysis_registry

    async def close_event() -> None:
        await worker._maybe_close_event(float(EVENT_TS + 30), force=True)
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, _wait_for, lambda: len(notifier.sent) == 1)

    asyncio.run(close_event())

    assert _wait_for(lambda: not registry.is_active(clip) and not registry.live_busy())
    assert clip.exists()


def test_analysis_workers_run_every_job_survive_a_bad_one_and_never_block_exit():
    pool = AnalysisWorkers(name="test-analysis")
    pool.ensure(2)
    done: list = []
    gate = threading.Event()

    def slow() -> None:
        gate.wait(5)
        done.append("slow")

    def bad() -> None:
        raise ValueError("a job that blows up")

    assert pool.submit(slow) == 0
    assert pool.submit(slow) in (0, 1)
    pool.submit(bad)
    queued_behind = pool.submit(lambda: done.append("after the bad one"))
    assert queued_behind >= 1                      # both workers are busy: it waits, as a queue entry
    assert done == []

    gate.set()
    assert _wait_for(lambda: sorted(done) == ["after the bad one", "slow", "slow"])
    assert pool.waiting == 0

    pool.ensure(1)                                 # never shrinks, never doubles up
    pool.ensure(3)
    names = sorted(t.name for t in pool._threads)
    assert names == ["test-analysis-1", "test-analysis-2", "test-analysis-3"]
    assert all(t.daemon for t in pool._threads)    # a restart does not wait for an hour-long job
