"""A detector that fails is not a clip without an animal.

Background (bug hunt, 2026-09-19): ``_analyze_video`` caught every exception
the detector raised on a sampled frame, logged it and moved on, and
``process_clip`` reported success whatever had happened. A clip on which the
detector never ran came back as a successful "no animal", and the live path
acts on that by deleting the clip and skipping the alert. Choosing YOLO as
the post-processing detector on the settings page did it to every event
(``YoloDetector.infer`` had no ``return_filtered`` parameter and the
post-processor always passes one, so each frame raised ``TypeError``); a
CUDA out-of-memory error did it to whichever clip it hit.

On a PTZ camera the failure was then hidden for good: the pipeline parks the
event's PTZ decisions in the clip's sidecar, and with no analysis in it that
file still counted as "analysis finished", so the recovery sweep skipped the
clip and the archive showed "No frame" forever.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker.analysis_recovery import find_unfinished_clips, has_analysis_sidecar
from animaltracker.detector import (
    BaseDetector,
    Detection,
    MegaDetectorBackend,
    SpeciesNetDetector,
    YoloDetector,
)
from animaltracker.pipeline import EventState, StreamWorker
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings
from animaltracker.storage import StorageManager

EVENT_TS = 1789000000
DOG = "mammalia_carnivora_canidae"
WIDTH, HEIGHT, FRAMES = 320, 240, 20


class ScriptedDetector(BaseDetector):
    """``script(call_number)`` returns the detections, or an exception to raise."""

    def __init__(self, script) -> None:
        self.script = script
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        call = self.calls
        self.calls += 1
        outcome = self.script(call)
        if isinstance(outcome, Exception):
            raise outcome
        return (outcome, []) if return_filtered else outcome


def dog():
    return [Detection(species=DOG, confidence=0.9, bbox=[100.0, 100.0, 180.0, 170.0], taxonomy=DOG)]


def always_fails(call):
    return RuntimeError("CUDA out of memory")


def write_video(path: Path, frames: int = FRAMES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 9) % 255, dtype=np.uint8))
    writer.release()
    return path


def make_clip(root: Path, label: str = "animal") -> Path:
    return write_video(root / "clips" / "cam1" / "2026" / "09" / "10" / f"{EVENT_TS}_{label}.mp4")


def processor(root: Path, detector) -> ClipPostProcessor:
    return ClipPostProcessor(detector=detector, storage_root=root,
                             settings=build_processing_settings(None, {"sample_rate": 1}))


def ptz_only_sidecar(clip: Path, decisions) -> Path:
    """The file pipeline.py step 5.5 writes when there is no analysis to add to."""
    path = clip.with_suffix(".log.json")
    with open(path, "w") as f:
        json.dump({"clip": clip.name, "ptz_decisions": decisions}, f, indent=2, default=str)
    return path


# --- every backend speaks the post-processor's dialect ------------------------

@pytest.mark.parametrize("backend", [BaseDetector, YoloDetector, MegaDetectorBackend, SpeciesNetDetector])
def test_every_backend_takes_what_the_post_processor_passes(backend):
    params = inspect.signature(backend.infer).parameters
    assert {"frame", "conf_threshold", "generic_confidence", "return_filtered"} <= set(params)


def test_yolo_returns_the_filtered_list_when_asked():
    class NoBoxes:
        names = {16: "dog"}

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=None)]

    det = object.__new__(YoloDetector)
    det.model = NoBoxes()
    det.class_map = det.model.names
    det.animal_only = True
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    assert det.infer(frame, conf_threshold=0.3, generic_confidence=0.8, return_filtered=True) == ([], [])
    assert det.infer(frame, conf_threshold=0.3) == []


# --- the post-processor ------------------------------------------------------------

def test_a_detector_that_fails_on_every_frame_is_a_failed_job_that_touches_nothing(tmp_path):
    clip = make_clip(tmp_path)
    old_thumb = clip.with_name(f"{clip.stem}_thumb_animal_t0.jpg")
    old_thumb.write_bytes(b"\xff\xd8\xff the key frame from an earlier run")

    result = processor(tmp_path, ScriptedDetector(always_fails)).process_clip(clip)

    assert result.success is False
    assert result.inference_errors == FRAMES
    assert "20 of 20" in result.error and "CUDA out of memory" in result.error
    assert result.species_results == {} and result.new_path is None
    assert clip.exists()
    assert not clip.with_suffix(".log.json").exists()      # nothing claims it is finished
    assert old_thumb.read_bytes().startswith(b"\xff\xd8\xff the key frame")
    assert sorted(p.name for p in clip.parent.iterdir()) == [clip.name, old_thumb.name]
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == [clip]


def test_a_few_failed_frames_leave_a_result_that_says_so(tmp_path):
    clip = make_clip(tmp_path)
    flaky = ScriptedDetector(lambda call: RuntimeError("one-off") if call in (3, 11) else dog())

    result = processor(tmp_path, flaky).process_clip(clip)

    assert result.success is True and result.inference_errors == 2
    assert result.new_species == DOG
    log = json.loads(result.new_path.with_suffix(".log.json").read_text())
    assert log["video"]["inference_errors"] == 2 and log["video"]["frames_inferred"] == FRAMES
    assert "one-off" in log["video"]["first_inference_error"]


def test_half_the_frames_failing_is_already_too_many(tmp_path):
    clip = make_clip(tmp_path)
    every_other = ScriptedDetector(lambda call: RuntimeError("flaky") if call % 2 else dog())

    assert processor(tmp_path, every_other).process_clip(clip).success is False


def test_a_clean_run_reports_no_errors(tmp_path):
    result = processor(tmp_path, ScriptedDetector(lambda call: dog())).process_clip(make_clip(tmp_path))

    assert result.success is True and result.inference_errors == 0


# --- a sidecar with nothing but PTZ decisions ------------------------------------------

def test_a_ptz_only_sidecar_is_not_a_finished_analysis(tmp_path):
    clip = make_clip(tmp_path)
    assert has_analysis_sidecar(clip) is False                        # no sidecar at all

    ptz_only_sidecar(clip, [{"timestamp": EVENT_TS + 1.0, "event": "move", "pan": 0.2}] * 40)
    assert has_analysis_sidecar(clip) is False
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == [clip]

    clip.with_suffix(".log.json").write_text("{}")                     # unreadable as either: left alone
    assert has_analysis_sidecar(clip) is True


def test_an_analysis_sidecar_counts_with_or_without_ptz_decisions(tmp_path):
    clip = make_clip(tmp_path)
    proc = processor(tmp_path, ScriptedDetector(lambda call: []))

    proc._save_processing_log(clip, [], None, {"frames_to_analyze": 20})
    assert has_analysis_sidecar(clip) is True

    proc._save_processing_log(clip, [], None, {"frames_to_analyze": 20},
                              ptz_decisions=[{"timestamp": EVENT_TS + 1.0, "event": "move"}])
    assert has_analysis_sidecar(clip) is True
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == []


def test_the_analysis_that_finally_runs_keeps_the_ptz_decisions(tmp_path):
    clip = make_clip(tmp_path)
    decisions = [{"timestamp": EVENT_TS + 1.0, "event": "move", "pan": 0.2}]
    ptz_only_sidecar(clip, decisions)

    result = processor(tmp_path, ScriptedDetector(lambda call: dog())).process_clip(clip)

    assert result.new_path.name == f"{EVENT_TS}_{DOG}.mp4"
    log = json.loads(result.new_path.with_suffix(".log.json").read_text())
    assert log["ptz_decisions"] == decisions
    assert log["analysis_summary"]["frames_with_detections"] == FRAMES
    assert list(log)[-1] == "ptz_decisions"                  # where the pipeline appends them
    # The log under the old name described a file that is gone.
    assert not clip.with_suffix(".log.json").exists()
    assert has_analysis_sidecar(result.new_path)


def test_a_reanalysis_in_place_keeps_them_too(tmp_path):
    clip = make_clip(tmp_path)
    decisions = [{"timestamp": EVENT_TS + 1.0, "event": "move"}]
    ptz_only_sidecar(clip, decisions)
    proc = processor(tmp_path, ScriptedDetector(lambda call: []))      # nothing found: stays "animal"

    assert proc.process_clip(clip).new_path is None
    assert proc.process_clip(clip).new_path is None                    # and again, from a full sidecar

    assert json.loads(clip.with_suffix(".log.json").read_text())["ptz_decisions"] == decisions


# --- the live path: keep the clip, send the alert, let the sweep retry -----------

class FakeWriter:
    def __init__(self, temp_avi: Path) -> None:
        self.temp_avi = temp_avi
        self.frame_count = FRAMES

    def close(self) -> Path:
        return self.temp_avi


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, ctx, **kwargs) -> None:
        self.sent.append(ctx)


class FakePtzTracker:
    """Enough of PTZTracker for an event that closes with decisions on record."""

    _decision_log: list = []

    def get_decisions_in_window(self, start_ts, end_ts):
        return [{"timestamp": EVENT_TS + 2.0, "event": "move", "pan": 0.1}]

    def clear_lock(self) -> None:
        pass

    def is_track_enabled(self) -> bool:
        return True

    def get_mode(self) -> str:
        return "tracking"


def closing_worker(tmp_path: Path, monkeypatch, detector, *, ptz: bool = False):
    storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    temp_avi = storage.logs_root / "event_temp" / f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi"
    temp_avi.parent.mkdir(parents=True, exist_ok=True)
    temp_avi.write_bytes(b"the recording")

    def transcode(avi: Path, output: Path) -> bool:
        avi.unlink()
        write_video(output)
        return True

    monkeypatch.setattr(storage, "transcode_avi_to_mp4", transcode)
    camera = SimpleNamespace(
        id="cam1", name="Cam 1", exclude_species=[],
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    notifier = FakeNotifier()
    worker = object.__new__(StreamWorker)
    worker.camera = camera
    worker.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=5, post_seconds=5, format="mp4", post_analysis=True,
                             post_analysis_frames=0, unified_post_processing=True, sample_rate=1),
        detector=SimpleNamespace(), exclusion_list=[],
        notification=SimpleNamespace(web_base_url=None),
    ))
    worker.detector = object()
    worker.storage = storage
    worker.notifier = notifier
    worker.tracker = None
    worker.ptz_tracker = FakePtzTracker() if ptz else None
    worker.ptz_drives_tracking = False
    worker._get_postprocess_detector = lambda: detector
    worker.event_state = EventState(
        camera=camera, start_ts=float(EVENT_TS), species={"animal"},
        max_confidence=0.9, last_detection_ts=float(EVENT_TS + 12),
        clip_writer=FakeWriter(temp_avi),
    )
    return worker, storage, notifier


def close_and_wait(worker, done) -> None:
    async def run() -> None:
        await worker._maybe_close_event(float(EVENT_TS + 30), force=True)
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, wait_for, done)

    asyncio.run(run())


def wait_for(condition, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture(autouse=True)
def fresh_slots():
    StreamWorker._postprocess_semaphore = None
    StreamWorker._postprocess_limit = 2
    yield
    StreamWorker._postprocess_semaphore = None


def test_a_failed_analysis_keeps_the_clip_and_still_alerts(tmp_path, monkeypatch):
    worker, storage, notifier = closing_worker(tmp_path, monkeypatch, ScriptedDetector(always_fails))
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    close_and_wait(worker, lambda: len(notifier.sent) == 1)

    assert clip.exists()                                   # it used to be deleted as a false positive
    assert notifier.sent[0].species == "animal" and notifier.sent[0].clip_path == str(clip)
    assert find_unfinished_clips(storage.storage_root / "clips", min_age_s=0) == [clip]


def test_a_failed_analysis_on_a_ptz_camera_is_still_found_by_the_sweep(tmp_path, monkeypatch):
    worker, storage, notifier = closing_worker(tmp_path, monkeypatch, ScriptedDetector(always_fails), ptz=True)
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    close_and_wait(worker, lambda: len(notifier.sent) == 1)

    sidecar = json.loads(clip.with_suffix(".log.json").read_text())
    assert list(sidecar) == ["clip", "ptz_decisions"]      # all the pipeline had to record
    assert find_unfinished_clips(storage.storage_root / "clips", min_age_s=0) == [clip]


def test_errors_on_some_frames_and_nothing_found_is_no_licence_to_delete(tmp_path, monkeypatch):
    flaky = ScriptedDetector(lambda call: RuntimeError("one-off") if call == 4 else [])
    worker, storage, notifier = closing_worker(tmp_path, monkeypatch, flaky)
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    close_and_wait(worker, lambda: len(notifier.sent) == 1)

    assert clip.exists() and has_analysis_sidecar(clip)


def test_a_clean_analysis_that_finds_nothing_still_removes_the_false_positive(tmp_path, monkeypatch):
    nothing = ScriptedDetector(lambda call: [])
    worker, storage, notifier = closing_worker(tmp_path, monkeypatch, nothing)
    clip = storage.build_clip_path("cam1", "animal", EVENT_TS)

    close_and_wait(worker, lambda: nothing.calls == FRAMES and not clip.exists())

    assert wait_for(lambda: not StreamWorker.analysis_registry.is_active(clip))
    assert notifier.sent == []
    assert list(clip.parent.iterdir()) == []
