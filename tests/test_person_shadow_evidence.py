"""A person's shadow is not evidence that an animal was there.

Background (bug hunt, 2026-09-19): SpeciesNet sometimes reads a person as an
animal. The post-processor recognises those boxes (they sit where a person
box is, somewhere in the clip) and drops the tracks made of them. But the
live path's false-positive gate counted ``min_detection_frames`` against
every sampled frame that had a detection in the log, the dropped tracks'
frames included. Someone walking past (a dozen shadow frames) plus one stray
"squirrel" frame elsewhere gave thirteen "detection frames" against a minimum
of two: the clip was kept and alerted as a squirrel, which is exactly what
the minimum exists to reject.

Without tracking, a species that kept any real detection had only its count
corrected: it went out at the shadow's confidence, with crops of the person
as its key frames.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker.detector import NON_ANIMAL_REASON_PREFIX, BaseDetector, Detection
from animaltracker.pipeline import EventState, StreamWorker
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings
from animaltracker.storage import StorageManager

EVENT_TS = 1789000000
WIDTH, HEIGHT, FRAMES = 320, 240, 60
PERSON_BOX = [100.0, 40.0, 160.0, 200.0]
ELSEWHERE = [250.0, 180.0, 290.0, 210.0]
CROW = "bird_passeriformes_corvidae"
SQUIRREL = "mammalia_rodentia_sciuridae"


class ScriptedDetector(BaseDetector):
    """``script(call)`` returns ``(detections, filtered)`` for the call-th sampled frame."""

    def __init__(self, script) -> None:
        self.script = script
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        detections, filtered = self.script(self.calls)
        self.calls += 1
        return (detections, filtered) if return_filtered else detections


def animal(species, conf, box):
    return Detection(species=species, confidence=conf, bbox=list(box), taxonomy=species)


def a_person():
    return (Detection(species="person", confidence=0.95, bbox=list(PERSON_BOX), taxonomy="human"),
            f"{NON_ANIMAL_REASON_PREFIX}person")


def someone_walks_past(squirrel_frames):
    """A person, read as a crow for twelve sampled frames, and a squirrel on ``squirrel_frames``."""
    def script(call):
        detections, filtered = [], []
        if 5 <= call <= 16:
            detections.append(animal(CROW, 0.93, PERSON_BOX))
        else:
            filtered.append(a_person())
        if call in squirrel_frames:
            detections.append(animal(SQUIRREL, 0.55, ELSEWHERE))
        return detections, filtered
    return script


def write_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 4) % 255, dtype=np.uint8))
    writer.release()
    return path


def analyse(tmp_path, script, **overrides):
    clip = write_video(tmp_path / "clips" / "cam1" / "2026" / "09" / "10" / f"{EVENT_TS}_animal.mp4")
    settings = build_processing_settings(None, {"sample_rate": 2, **overrides})
    return ClipPostProcessor(detector=ScriptedDetector(script), storage_root=tmp_path,
                             settings=settings).process_clip(clip, update_filename=False)


# --- what the post-processor reports -------------------------------------------------

def test_the_frames_of_a_dropped_shadow_track_are_not_detection_frames(tmp_path):
    result = analyse(tmp_path, someone_walks_past(squirrel_frames={20}))

    assert (result.tracking_summary or {}).get("person_shadow_tracks") == 1
    assert list(result.species_results) == [SQUIRREL]
    assert result.detection_frames == 1              # the log holds thirteen frames with a detection
    logged = {e.frame_idx for e in result.processing_log if e.event in ("tracked", "accepted") and e.frame_idx >= 0}
    assert len(logged) == 13


def test_a_real_visitor_is_counted_frame_by_frame(tmp_path):
    result = analyse(tmp_path, someone_walks_past(squirrel_frames=set(range(18, 26))))

    assert list(result.species_results) == [SQUIRREL]
    # All eight, the first included, although ByteTrack only reports a track
    # from its second match: a visit of exactly the minimum length must pass.
    assert result.detection_frames == 8


def test_a_clip_with_only_the_person_has_none(tmp_path):
    result = analyse(tmp_path, someone_walks_past(squirrel_frames=set()))

    assert result.species_results == {} and result.detection_frames == 0


def test_without_tracking_a_species_is_rebuilt_from_its_real_detections(tmp_path):
    def script(call):                                  # the person is a crow; so is a real crow, twice
        detections, filtered = [], []
        if 5 <= call <= 16:
            detections.append(animal(CROW, 0.93, PERSON_BOX))
        else:
            filtered.append(a_person())
        if call in (20, 22):
            detections.append(animal(CROW, 0.61, ELSEWHERE))
        return detections, filtered

    result = analyse(tmp_path, script, tracking_enabled=False)

    crow = result.species_results[CROW]
    assert crow.count == 2 and result.detection_frames == 2
    assert crow.confidence == pytest.approx(0.61)      # it went out at the shadow's 0.93
    assert [kf[2] for kf in crow.key_frames] == [ELSEWHERE, ELSEWHERE]   # they were crops of the person
    assert all(kf[0] is not None and kf[0].shape[:2] == (HEIGHT, WIDTH) for kf in crow.key_frames)


def test_without_any_person_in_the_clip_nothing_is_rebuilt(tmp_path):
    result = analyse(tmp_path, lambda call: ([animal(CROW, 0.8, ELSEWHERE)] if call % 2 == 0 else [], []),
                     tracking_enabled=False)

    crow = result.species_results[CROW]
    assert crow.confidence == pytest.approx(0.8) and len(crow.key_frames) == 3
    assert result.detection_frames == crow.count == 15


# --- the live path's false-positive gate ------------------------------------------------

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


def closing_worker(tmp_path: Path, detector):
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
        id="cam1", name="Cam 1", exclude_species=[],
        notification=SimpleNamespace(priority=0, sound=None, destinations=None),
    )
    notifier = FakeNotifier()
    worker = object.__new__(StreamWorker)
    worker.camera = camera
    worker.runtime = SimpleNamespace(general=SimpleNamespace(
        clip=SimpleNamespace(pre_seconds=5, post_seconds=5, format="mp4", post_analysis=True,
                             post_analysis_frames=0, unified_post_processing=True, sample_rate=2,
                             delete_if_no_animal=True, min_detection_frames=2),
        detector=SimpleNamespace(), exclusion_list=[],
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


def test_someone_walking_past_and_one_stray_frame_is_a_false_positive(tmp_path):
    detector = ScriptedDetector(someone_walks_past(squirrel_frames={20}))
    worker, storage, notifier = closing_worker(tmp_path, detector)
    day = storage.build_clip_path("cam1", "animal", EVENT_TS).parent

    close_and_wait(worker, lambda: detector.calls == FRAMES // 2 and not any(day.glob("*.mp4")))

    assert wait_for(lambda: not StreamWorker.analysis_registry.live_busy())
    assert notifier.sent == []                        # it alerted as a squirrel
    assert list(day.iterdir()) == []


def test_a_squirrel_that_stayed_still_alerts_with_the_person_in_the_clip(tmp_path):
    detector = ScriptedDetector(someone_walks_past(squirrel_frames=set(range(18, 26))))
    worker, storage, notifier = closing_worker(tmp_path, detector)

    close_and_wait(worker, lambda: len(notifier.sent) == 1)

    assert notifier.sent[0].species == SQUIRREL
    assert Path(notifier.sent[0].clip_path).exists()
