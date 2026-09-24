"""The motion gate: only frames in which something moved need the live detector.

Background (2026-09-23): the one live MegaDetector every camera shares ran
75-92% busy at ~114 ms a check although the yards are still most of the
time. ``motion.MotionGate`` compares each frame with a small, slowly
updated background; ``thresholds.motion_gate`` decides whether its verdict
only gets counted ("observe", the default: every frame is still checked)
or acted on ("on"). Observing, it counts the frames it would have skipped
in which the detector found something that passed the filters, which is
what says whether switching it on is safe.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from animaltracker.detector import Detection
from animaltracker.motion import MotionGate
from animaltracker.pipeline import StreamWorker

W, H = 1280, 720


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def yard(seed: int = 1, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), 60, dtype=np.float32)
    base[200:260, 600:640] = 200                      # the feeder
    if noise:
        base += rng.normal(0, noise, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def with_animal(frame: np.ndarray, x: int, size: int = 80) -> np.ndarray:
    out = frame.copy()
    out[400:400 + size, x:x + size] = 170
    return out


def gate_with_clock():
    clock = Clock()
    return MotionGate("Otteson1", clock=clock), clock


def step(gate, clock, frame, dt=0.33, event_open=False):
    clock.t += dt
    reading = gate.measure(frame)
    reason = gate.wants_check(reading, event_open)
    if reason:
        gate.note_check()
    return reading, reason


def test_the_first_frame_is_always_checked():
    gate, clock = gate_with_clock()
    reading, reason = step(gate, clock, yard())
    assert reading.fresh and reason == "first"


def test_a_still_scene_is_skipped_between_heartbeats():
    gate, clock = gate_with_clock()
    step(gate, clock, yard())
    reasons = [step(gate, clock, yard())[1] for _ in range(45)]      # 15 s of a still yard
    assert reasons.count("heartbeat") == 2                          # at ~5 s and ~10 s
    assert reasons.count("") == 43


def test_night_noise_does_not_count_as_motion():
    gate, clock = gate_with_clock()
    step(gate, clock, yard(seed=0, noise=8.0))
    readings = [step(gate, clock, yard(seed=i, noise=8.0))[0] for i in range(1, 20)]
    assert max(r.changed_px for r in readings) < MotionGate.MIN_CHANGED_PX


def test_the_smallest_animal_the_filters_accept_is_motion():
    # min_detection_area is 0.5% of the frame: about 68x68 pixels at 1280x720.
    gate, clock = gate_with_clock()
    step(gate, clock, yard())
    step(gate, clock, yard())
    reading, reason = step(gate, clock, with_animal(yard(), 300, size=68))
    assert reason == "motion" and reading.changed_px >= MotionGate.MIN_CHANGED_PX


def test_every_frame_is_checked_while_an_event_is_open():
    gate, clock = gate_with_clock()
    step(gate, clock, yard())
    assert all(step(gate, clock, yard(), event_open=True)[1] == "event" for _ in range(10))


def test_an_animal_that_stops_fades_into_the_background_and_the_heartbeat_still_looks():
    gate, clock = gate_with_clock()
    step(gate, clock, yard())
    still = with_animal(yard(), 300)
    reasons = [step(gate, clock, still)[1] for _ in range(75)]       # 25 s standing still
    assert reasons[0] == "motion"
    assert "" in reasons[-10:]                                     # absorbed by then
    assert reasons.count("heartbeat") >= 3                         # still looked at every 5 s


def test_a_gap_starts_the_background_again():
    gate, clock = gate_with_clock()
    step(gate, clock, yard())
    reading, reason = step(gate, clock, with_animal(yard(), 300), dt=MotionGate.STALE_AFTER_S + 1)
    assert reading.fresh and reason == "first"


def test_the_minute_summary_counts_skips_and_misses():
    gate, clock = gate_with_clock()
    reading, _ = step(gate, clock, yard())
    gate.record(reading, "first", skipped=False, found=0)
    for i in range(10):
        reading, reason = step(gate, clock, yard())
        gate.record(reading, reason, skipped=False, found=1 if i == 3 and not reason else 0)
    clock.t += MotionGate.SUMMARY_EVERY_S
    reading, reason = step(gate, clock, yard())
    gate.record(reading, reason, skipped=False, found=0)

    s = gate.last_summary
    assert s["frames"] == 12
    assert s["missed"] == 1
    assert s["would_skip"] >= 9 and s["skipped"] == 0


# --- in the pipeline ---------------------------------------------------------

class Stop(Exception):
    pass


class StopsAfterTheGate(StreamWorker):
    """Ends _process_frame at its first step after the gate has been told
    what the detector found (it records the frame size for the live view)."""

    @property
    def latest_frame_size(self):
        return (0, 0)

    @latest_frame_size.setter
    def latest_frame_size(self, value):
        raise Stop


def worker(mode: str, found: int = 0):
    w = object.__new__(StopsAfterTheGate)
    w._tracker_lock = threading.Lock()
    w.camera = SimpleNamespace(
        id="Otteson1",
        thresholds=SimpleNamespace(blur_threshold=0.0, ptz_settle_time=0.0, motion_gate=mode,
                                   confidence=0.5, generic_confidence=0.9),
        inference_max_width=0, detect_enabled=True,
    )
    w.tracker = None
    w.event_state = None
    w.ptz_tracker = None
    w.onvif_client = None
    w.perf_infer_count = w.perf_infer_time_total = w.perf_infer_time_max = 0
    w.perf_frame_age_total = w.perf_frame_age_max = 0
    calls = []

    def infer(frame, **kw):
        calls.append(1)
        return [Detection(species="animal", confidence=0.9, bbox=[1, 1, 2, 2])] * found

    w.detector = SimpleNamespace(infer=infer)
    return w, calls


def feed(w, frames):
    async def main():
        for i, f in enumerate(frames):
            try:
                await w._process_frame(f, 1789000000.0 + i, i)
            except Stop:
                pass
    asyncio.run(main())


@pytest.fixture(autouse=True)
def plain_filters(monkeypatch):
    monkeypatch.setattr(StreamWorker, "_filter_detections", lambda self, d: d)
    monkeypatch.setattr(StreamWorker, "_filter_false_positives", lambda self, d, w, h: d)
    monkeypatch.setattr(StreamWorker, "_raw_detection_log_level", lambda self, now=None: 10)


def test_observing_checks_every_frame():
    w, calls = worker("observe")
    feed(w, [yard()] * 6)
    assert len(calls) == 6


def test_on_skips_the_still_frames():
    w, calls = worker("on")
    feed(w, [yard()] * 6)
    assert len(calls) == 1                                # the first frame only
    assert w.perf_frames_skipped_motion == 5


def test_observing_counts_a_detection_on_a_frame_it_would_have_skipped(caplog):
    caplog.set_level(logging.INFO, logger="animaltracker.motion")
    w, calls = worker("observe", found=1)
    feed(w, [yard()] * 3)
    gate = w._motion_gate_for_camera()
    assert gate._window["missed"] >= 1
    assert "would have skipped a frame with 1 detection" in caplog.text
