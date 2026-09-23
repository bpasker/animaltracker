"""The blur check runs on the executor, not on the event loop, and only after a PTZ move.

Background (bug hunt, 2026-09-19): every inferred frame of every camera with
``blur_threshold`` above 0 (the default is 50) had a Laplacian variance taken
over the whole frame, directly in the coroutine. Measured on this machine:
2.5 ms at 1080p, 4.9 ms at 2688x1512. That is time on the one loop that also
dispatches every camera's reads, serves every MJPEG frame and answers every
request.

The metric is deliberately unchanged: it is still the full frame, as the
configured thresholds were tuned against, not the downscaled copy inference
may use.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from animaltracker.pipeline import StreamWorker

WIDTH, HEIGHT = 160, 120


MOVED_AT = 1789000000.0


def worker(blur_threshold: float, ptz: bool = True, moved_at: float = MOVED_AT) -> StreamWorker:
    """A camera whose own head the PTZ tracker moved at ``moved_at``, or a fixed one."""
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(
        id="cam1",
        thresholds=SimpleNamespace(blur_threshold=blur_threshold, confidence=0.5, generic_confidence=0.9,
                                   ptz_settle_time=0.0),
        detect_enabled=True,
    )
    w.tracker = None
    w.event_state = None
    client = object()
    w.onvif_client = client if ptz else None
    w.ptz_tracker = SimpleNamespace(onvif_client=client, get_last_move_time=lambda: moved_at) if ptz else None
    return w


def flat_frame():
    """No edges at all: Laplacian variance 0, under any positive threshold."""
    return np.full((HEIGHT, WIDTH, 3), 127, dtype=np.uint8)


def run(coro_fn):
    seen = {}

    async def main():
        seen["loop_thread"] = threading.current_thread()
        await coro_fn()

    asyncio.run(main())
    return seen


def test_the_laplacian_is_taken_on_a_worker_thread(monkeypatch):
    w = worker(50.0)
    where = {}
    real = StreamWorker._compute_blur_score

    def watched(frame):
        where["thread"] = threading.current_thread()
        return real(frame)

    monkeypatch.setattr(StreamWorker, "_compute_blur_score", staticmethod(watched))

    seen = run(lambda: w._process_frame(flat_frame(), 1789000000.0, 0))

    assert "thread" in where, "the blur check never ran"
    assert where["thread"] is not seen["loop_thread"]


def test_a_blurry_frame_is_still_skipped(monkeypatch):
    w = worker(50.0)
    inferred = []
    w.detector = SimpleNamespace(infer=lambda *a, **k: inferred.append(1) or [])

    run(lambda: w._process_frame(flat_frame(), 1789000000.0, 0))

    assert inferred == []            # a flat frame scores 0: never reaches the detector


def test_the_check_is_skipped_entirely_when_it_is_switched_off(monkeypatch):
    w = worker(0.0)
    calls = []
    monkeypatch.setattr(StreamWorker, "_compute_blur_score",
                        staticmethod(lambda frame: calls.append(1) or 0.0))
    w.detector = SimpleNamespace(infer=lambda *a, **k: [])
    w.camera.inference_max_width = 0
    w.latest_detections = []
    w.latest_detection_ts = 0.0
    w.ptz_tracker = None

    try:
        run(lambda: w._process_frame(flat_frame(), 1789000000.0, 0))
    except Exception:
        pass                          # the rest of the frame path is not under test

    assert calls == []


def test_the_metric_is_the_full_frame_laplacian():
    import cv2

    frame = np.random.default_rng(0).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    expected = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    assert StreamWorker._compute_blur_score(frame) == pytest.approx(expected)
    assert StreamWorker._compute_blur_score(flat_frame()) == pytest.approx(0.0)


def test_a_blurry_frame_is_counted_for_the_monitor():
    # The monitor's detector card says why a camera checks nothing: at night
    # Otteson1's clean infrared frames scored ~24 against the default 50,
    # so every one was dropped here and the camera was blind until dawn.
    w = worker(50.0)
    w.detector = SimpleNamespace(infer=lambda *a, **k: [])

    run(lambda: w._process_frame(flat_frame(), 1789000000.0, 0))

    assert w.perf_frames_skipped_blur == 1


class ReachedTheDetector(Exception):
    pass


def reaches_detector(w, ts):
    """Run one frame; True if it got past the blur filter to inference."""
    def infer(*a, **k):
        raise ReachedTheDetector

    w.detector = SimpleNamespace(infer=infer)
    w.camera.thresholds.inference_max_width = 0
    try:
        run(lambda: w._process_frame(flat_frame(), ts, 0))
    except ReachedTheDetector:
        return True
    return False


def test_a_fixed_camera_is_never_blur_filtered():
    # 2026-09-22: Otteson1's night infrared picture scored ~24 against the
    # default 50 and it never recorded at night. The filter is for frames a
    # PTZ move smeared; a fixed camera's dim frame goes to the detector.
    w = worker(50.0, ptz=False)
    assert reaches_detector(w, MOVED_AT)
    assert w.perf_frames_skipped_blur == 0


def test_a_ptz_camera_long_after_its_last_move_is_not_blur_filtered():
    w = worker(50.0, moved_at=MOVED_AT - 60)
    assert reaches_detector(w, MOVED_AT)
    assert w.perf_frames_skipped_blur == 0


def test_a_ptz_camera_just_after_a_move_is_blur_filtered():
    w = worker(50.0, moved_at=MOVED_AT - 1)
    assert not reaches_detector(w, MOVED_AT)
    assert w.perf_frames_skipped_blur == 1
