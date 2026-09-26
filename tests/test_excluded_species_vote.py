"""An excluded animal has no vote in a clip's species.

Background (2026-09-26): a camera's exclude_species was checked once, against
the name the post-processor had already given the clip, and that name came
from a vote in which the excluded animal took part like any other. On the
feeder cameras, which exclude squirrels, a squirrel on the feeder for forty
sampled frames outvoted a cardinal that visited for twelve: the clip was
named for the squirrel and deleted as excluded, cardinal and all, with no
alert. It also pooled its votes with its own class, so a deer and a squirrel
together outvoted a cardinal that either alone would have lost to.

Now the excluded animal, and the generic labels on its branch ("animal",
"mammal", "rodent" beside a squirrel), are left out of the vote. The clip is
named for an excluded animal, and so deleted, only when the rest of the clip
does not add up to ``min_detection_frames`` sampled frames.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker.detector import BaseDetector, Detection
from animaltracker.pipeline import EventState, PipelineOrchestrator, StreamWorker
from animaltracker.postprocess import (
    ClipPostProcessor,
    ProcessingSettings,
    build_processing_settings,
    excluded_species_for,
    process_all_clips,
)
from animaltracker.storage import StorageManager

EVENT_TS = 1789000000
WIDTH, HEIGHT, FRAMES = 320, 240, 60      # sampled every 2nd frame: 30 detector calls
SQUIRREL = "mammalia_rodentia_sciuridae"
CARDINAL = "bird_passeriformes_cardinalidae"
DEER = "mammalia_cetartiodactyla_cervidae"
NO_SQUIRRELS = ["mammalia rodentia sciuridae"]   # as cameras.yml spells it

FEEDER = [20.0, 150.0, 80.0, 200.0]
BRANCH = [240.0, 30.0, 280.0, 70.0]
LAWN = [120.0, 60.0, 200.0, 180.0]
FENCE = [250.0, 170.0, 300.0, 220.0]


class ScriptedDetector(BaseDetector):
    """``script(call)`` returns the detections for the call-th sampled frame of a clip."""

    def __init__(self, script) -> None:
        self.script = script
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        detections = self.script(self.calls % (FRAMES // 2))
        self.calls += 1
        return (detections, []) if return_filtered else detections


def animal(species, conf, box):
    return Detection(species=species, confidence=conf, bbox=list(box), taxonomy=species)


def visits(*animals):
    """A script from ``(species, box, calls)``: each animal is seen on its calls."""
    def script(call):
        return [animal(species, 0.85, box) for species, box, calls in animals if call in calls]
    return script


ALL_CLIP = range(30)
SQUIRREL_ON_THE_FEEDER = (SQUIRREL, FEEDER, ALL_CLIP)


def write_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 4) % 255, dtype=np.uint8))
    writer.release()
    return path


def clip_path(root: Path, camera: str = "cam1") -> Path:
    return root / "clips" / camera / "2026" / "09" / "26" / f"{EVENT_TS}_animal.mp4"


def analyse(tmp_path, script, exclude=NO_SQUIRRELS, rename=False, **overrides):
    clip = write_video(clip_path(tmp_path))
    settings = build_processing_settings(None, {"sample_rate": 2, **overrides}, exclude_species=exclude)
    return ClipPostProcessor(detector=ScriptedDetector(script), storage_root=tmp_path,
                             settings=settings).process_clip(clip, update_filename=rename)


def excluded_entries(result):
    return {e.species: e.reason for e in result.processing_log if e.event == "species_excluded"}


def key_frame_species(path) -> str:
    """The species a key frame is of: ``<clip>_thumb_<species>_t<n>.jpg``."""
    return Path(path).stem.split("_thumb_", 1)[1].rsplit("_t", 1)[0]


# --- what the post-processor names the clip -------------------------------------------

def test_the_cardinal_names_the_clip_the_squirrel_was_in(tmp_path):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))

    result = analyse(tmp_path, script)

    assert result.new_species == CARDINAL
    assert set(result.species_results) == {SQUIRREL, CARDINAL}   # the squirrel is still in the result
    assert result.detection_frames == 6                         # the cardinal's frames, not the squirrel's
    assert "no vote" in excluded_entries(result)[SQUIRREL]


def test_without_the_exclusion_the_squirrel_names_it_as_before(tmp_path):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))

    result = analyse(tmp_path, script, exclude=[])

    assert result.new_species == SQUIRREL
    assert result.detection_frames == 30
    assert excluded_entries(result) == {}


def test_the_squirrel_no_longer_pools_its_votes_with_the_deer(tmp_path):
    # Mammals 20 + 15 outvoted the cardinal's 30 at the class level, and the
    # deer then beat the squirrel: a clip of mostly cardinal was a deer.
    script = visits((DEER, LAWN, range(20)), (SQUIRREL, FEEDER, range(15)), (CARDINAL, BRANCH, ALL_CLIP))

    assert analyse(tmp_path / "before", script, exclude=[]).new_species == DEER
    assert analyse(tmp_path / "after", script).new_species == CARDINAL


def test_a_generic_label_beside_the_squirrel_counts_as_the_squirrel(tmp_path):
    # A separate "animal" track would otherwise name the squirrel's clip
    # "Animal", and it would alert.
    script = visits(SQUIRREL_ON_THE_FEEDER, ("animal", FENCE, range(20, 28)))

    result = analyse(tmp_path, script)

    assert set(result.species_results) == {SQUIRREL, "animal"}
    assert result.new_species == SQUIRREL                        # so the live path deletes it
    assert result.detection_frames == 30                        # the clip's evidence, as before
    assert "generic" in excluded_entries(result)["animal"]
    assert "nothing else" in excluded_entries(result)[SQUIRREL]


def test_one_stray_frame_beside_the_squirrel_is_not_a_bird(tmp_path):
    result = analyse(tmp_path, visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, {12})))

    assert result.new_species == SQUIRREL


@pytest.mark.parametrize("calls, named", [(range(12, 14), SQUIRREL), (range(12, 15), CARDINAL)])
def test_the_rest_of_the_clip_must_reach_min_detection_frames(tmp_path, calls, named):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, calls))

    result = analyse(tmp_path, script, min_detection_frames=3)

    assert CARDINAL in result.species_results
    assert result.new_species == named
    if named == SQUIRREL:
        assert "3 needed" in excluded_entries(result)[CARDINAL]


def test_without_tracking_the_per_frame_vote_leaves_the_squirrel_out_too(tmp_path):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))

    result = analyse(tmp_path, script, tracking_enabled=False)

    assert result.new_species == CARDINAL
    assert result.detection_frames == 6


def test_an_exclusion_with_nothing_excluded_in_the_clip_changes_nothing(tmp_path):
    script = visits((CARDINAL, BRANCH, range(10, 16)), ("animal", FENCE, range(20, 28)))

    with_it = analyse(tmp_path / "with", script)
    without = analyse(tmp_path / "without", script, exclude=[])

    assert (with_it.new_species, with_it.detection_frames) == (without.new_species, without.detection_frames)
    assert excluded_entries(with_it) == {}


def test_the_sidecar_records_what_was_left_out_and_why(tmp_path):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))

    result = analyse(tmp_path, script, rename=True)

    assert result.new_path.name == f"{EVENT_TS}_{CARDINAL}.mp4"
    log = json.loads(result.new_path.with_suffix(".log.json").read_text())
    assert log["settings"]["exclude_species"] == NO_SQUIRRELS
    assert log["video"]["species_set_aside"] == [SQUIRREL]
    assert any(e["event"] == "species_excluded" and e["species"] == SQUIRREL for e in log["log_entries"])
    # Both animals keep their key frame.
    assert sorted(key_frame_species(p) for p in result.thumbnails_saved) == [CARDINAL, SQUIRREL]


# --- the live path: alert for the cardinal, delete only an all-excluded clip ------------

class FakeWriter:
    frame_count = FRAMES

    def __init__(self, temp_avi: Path) -> None:
        self.temp_avi = temp_avi

    def close(self) -> Path:
        return self.temp_avi


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, ctx, **kwargs) -> None:
        self.sent.append(ctx)


def closing_worker(tmp_path: Path, detector, camera_excludes=(), global_excludes=()):
    storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    temp_avi = storage.logs_root / "event_temp" / f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi"
    temp_avi.parent.mkdir(parents=True, exist_ok=True)
    temp_avi.write_bytes(b"the recording")

    def transcode(avi: Path, output: Path) -> bool:
        avi.unlink()
        write_video(output)
        return True

    storage.transcode_avi_to_mp4 = transcode
    camera = SimpleNamespace(
        id="cam1", name="Cam 1", exclude_species=list(camera_excludes),
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    notifier = FakeNotifier()
    worker = object.__new__(StreamWorker)
    worker._tracker_lock = threading.Lock()
    worker.camera = camera
    worker.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=5, post_seconds=5, format="mp4", post_analysis=True,
                             post_analysis_frames=0, unified_post_processing=True, sample_rate=2,
                             delete_if_no_animal=True, min_detection_frames=2),
        detector=SimpleNamespace(), exclusion_list=list(global_excludes),
        notification=SimpleNamespace(web_base_url=None),
    ))
    worker.detector = object()
    worker.storage = storage
    worker.notifier = notifier
    worker.tracker = None
    worker.ptz_tracker = None
    worker.ptz_drives_tracking = False
    worker._get_postprocess_detector = lambda: detector
    worker.event_state = EventState(
        camera=camera, start_ts=float(EVENT_TS), species={"animal"},
        max_confidence=0.9, last_detection_ts=float(EVENT_TS + 12),
        clip_writer=FakeWriter(temp_avi),
    )
    return worker, storage, notifier


def wait_for(condition, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def close_and_wait(worker, done) -> None:
    async def run() -> None:
        await worker._maybe_close_event(float(EVENT_TS + 30), force=True)
        assert await asyncio.get_running_loop().run_in_executor(None, wait_for, done)

    asyncio.run(run())


@pytest.fixture(autouse=True)
def fresh_slots():
    StreamWorker._postprocess_semaphore = None
    StreamWorker._postprocess_limit = 2
    yield
    StreamWorker._postprocess_semaphore = None


@pytest.mark.parametrize("where", ["camera", "global"])
def test_the_cardinal_alerts_from_the_squirrel_s_clip(tmp_path, where):
    detector = ScriptedDetector(visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16))))
    lists = {"camera_excludes": NO_SQUIRRELS} if where == "camera" else {"global_excludes": NO_SQUIRRELS}
    worker, storage, notifier = closing_worker(tmp_path, detector, **lists)

    close_and_wait(worker, lambda: len(notifier.sent) == 1)

    alert = notifier.sent[0]
    assert alert.species == CARDINAL
    assert Path(alert.clip_path).name == f"{EVENT_TS}_{CARDINAL}.mp4" and Path(alert.clip_path).exists()
    assert key_frame_species(alert.thumbnail_path) == CARDINAL    # the cardinal's photo, not the squirrel's


def test_a_clip_of_nothing_but_excluded_animals_is_still_deleted(tmp_path):
    detector = ScriptedDetector(visits(SQUIRREL_ON_THE_FEEDER, ("animal", FENCE, range(20, 28))))
    worker, storage, notifier = closing_worker(tmp_path, detector, camera_excludes=NO_SQUIRRELS)
    day = storage.build_clip_path("cam1", "animal", EVENT_TS).parent

    close_and_wait(worker, lambda: detector.calls == FRAMES // 2 and not any(day.glob("*.mp4")))

    assert wait_for(lambda: not StreamWorker.analysis_registry.live_busy())
    assert notifier.sent == []
    assert list(day.iterdir()) == []


# --- the recovery sweep and the command line use the clip's own camera ------------------

class SweepWorker:
    def __init__(self, detector) -> None:
        self.detector = detector

    def _get_postprocess_detector(self):
        return self.detector


def recovering_orchestrator(tmp_path: Path) -> PipelineOrchestrator:
    orch = object.__new__(PipelineOrchestrator)
    orch.storage = StorageManager(storage_root=tmp_path, logs_root=tmp_path / "logs")
    orch.runtime = SimpleNamespace(
        general=SimpleNamespace(clip=SimpleNamespace(sample_rate=2, min_detection_frames=2),
                                exclusion_list=[]),
        cameras=[SimpleNamespace(id="cam1", exclude_species=NO_SQUIRRELS)],
    )
    return orch


@pytest.mark.parametrize("camera, named", [("cam1", CARDINAL), ("cam9", SQUIRREL)])
def test_the_recovery_sweep_names_a_clip_with_its_own_camera_s_exclusions(tmp_path, camera, named):
    # cam9 is retired: its clip borrows cam1's worker, never cam1's exclusions.
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))
    clip = write_video(clip_path(tmp_path, camera))
    orch = recovering_orchestrator(tmp_path)

    assert orch._recover_clip({"cam1": SweepWorker(ScriptedDetector(script))}, clip)

    assert clip.with_name(f"{EVENT_TS}_{named}.mp4").exists()


def test_reprocessing_everything_names_each_clip_with_its_own_camera_s_exclusions(tmp_path):
    script = visits(SQUIRREL_ON_THE_FEEDER, (CARDINAL, BRANCH, range(10, 16)))
    for camera in ("cam1", "cam2"):
        write_video(clip_path(tmp_path, camera))

    results = process_all_clips(
        tmp_path, ScriptedDetector(script),
        settings=build_processing_settings(None, {"sample_rate": 2}),
        exclusions_for=lambda camera: NO_SQUIRRELS if camera == "cam1" else [],
    )

    named = {r.original_path.parts[-5]: r.new_species for r in results}
    assert named == {"cam1": CARDINAL, "cam2": SQUIRREL}


# --- the settings ----------------------------------------------------------------------

def test_the_settings_carry_the_exclusions_and_the_bar():
    settings = build_processing_settings(SimpleNamespace(min_detection_frames=4), exclude_species=NO_SQUIRRELS)

    assert settings.exclude_species == NO_SQUIRRELS
    assert settings.min_detection_frames == 4
    assert ProcessingSettings.from_dict(settings.to_dict()) == settings
    defaults = build_processing_settings(None)
    assert defaults.exclude_species == [] and defaults.min_detection_frames == 2


@pytest.mark.parametrize("override", [{"exclude_species": "squirrel"}, {"exclude_species": [3]},
                                      {"min_detection_frames": 0}, {"min_detection_frames": True}])
def test_a_bad_override_is_refused_before_the_analysis(override):
    with pytest.raises(ValueError):
        build_processing_settings(None, override)


def test_a_camera_s_exclusions_are_its_own_list_and_the_global_one():
    runtime = SimpleNamespace(
        general=SimpleNamespace(exclusion_list=["mammalia carnivora felidae"]),
        cameras=[SimpleNamespace(id="cam1", exclude_species=NO_SQUIRRELS),
                 SimpleNamespace(id="cam2", exclude_species=[])],
    )

    assert excluded_species_for(runtime, "cam1") == NO_SQUIRRELS + ["mammalia carnivora felidae"]
    assert excluded_species_for(runtime, "cam2") == ["mammalia carnivora felidae"]
    assert excluded_species_for(runtime, "retired") == ["mammalia carnivora felidae"]
    assert excluded_species_for(None, "cam1") == []
