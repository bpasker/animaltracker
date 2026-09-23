"""The monitor's detector card: how busy each model is and what each camera gets checked.

Background (2026-09-22): the monitor's GPU gauge showed NVML utilisation,
the share of a short sample in which a kernel ran. Polled every two seconds
it swung 0-70% between readings, and it could not say whether detection
keeps up: every camera takes turns on one live model, which can be booked
while the chip reads 25%. The card now shows the share of the last minute
each model held its lock (``detector.ModelBusyMeter``), and per camera how
many recorded frames were checked and why not the rest. Building it showed
Otteson1 checking nothing after dark: the blur filter rejected every
infrared frame.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from animaltracker.detector import BaseDetector, ModelBusyMeter
from animaltracker.web import WebServer


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_the_meter_reports_each_jobs_share_and_their_union():
    clock = Clock()
    meter = ModelBusyMeter(clock=clock)
    clock.t += 60.0
    # 30 s of live checks, 12 s of analysis, 6 s of it at the same time.
    for i in range(30):
        meter.record("live", 1000.0 + 2 * i, 1000.0 + 2 * i + 1)
    for i in range(12):
        meter.record("analysis", 1000.0 + 2 * i + 0.5, 1000.0 + 2 * i + 1.5)

    s = meter.summary()

    assert s["live_pct"] == pytest.approx(50.0)
    assert s["analysis_pct"] == pytest.approx(20.0)
    assert s["busy_pct"] == pytest.approx(60.0)      # union: overlap counted once
    assert s["live_ms_avg"] == pytest.approx(1000.0)
    assert s["live_runs"] == 30 and s["analysis_runs"] == 12


def test_the_meter_forgets_what_is_older_than_its_window():
    clock = Clock()
    meter = ModelBusyMeter(clock=clock)
    meter.record("live", 1000.0, 1030.0)
    clock.t = 1100.0
    meter.record("live", 1099.0, 1100.0)

    s = meter.summary()

    assert s["live_pct"] == pytest.approx(100 / 60, abs=0.1)
    assert s["live_runs"] == 1


def test_a_forward_pass_is_recorded_under_its_detectors_usage(monkeypatch):
    from animaltracker import detector as detector_mod

    clock = Clock()
    meter = ModelBusyMeter(clock=clock)
    monkeypatch.setattr(detector_mod, "MODEL_BUSY", meter)

    class Stub(BaseDetector):
        backend_name = "stub"

        def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
            with self.model_lock:
                clock.t += 0.2
            return []

    live, analysis = Stub(), Stub()
    live.usage, analysis.usage = "live", "analysis"
    clock.t += 10.0
    live.infer(None)
    analysis.infer(None)

    s = meter.summary()
    assert (s["live_runs"], s["analysis_runs"]) == (1, 1)
    assert s["live_ms_avg"] == pytest.approx(200.0)


# --- /api/monitor ------------------------------------------------------------

class Req:
    query: dict = {}
    match_info: dict = {}


def worker(perf, detect=True, blur=50.0):
    cam = SimpleNamespace(id="Otteson1", name="Otteson1", location="Yard", detect_enabled=detect,
                          thresholds=SimpleNamespace(blur_threshold=blur))
    return SimpleNamespace(
        camera=cam, latest_frame=object(), event_state=None, tracking_enabled=False,
        clip_buffer=SimpleNamespace(frame_count=1, max_frames=2, duration=1.0, max_seconds=2),
        get_perf_stats=lambda: dict(perf),
    )


def monitor(tmp_path, workers):
    server = WebServer(workers, tmp_path, tmp_path / "logs", port=0)
    resp = asyncio.run(server.handle_get_monitor_data(Req()))
    return json.loads(resp.body.decode())


def test_a_camera_whose_frames_are_all_too_blurry_says_so(tmp_path):
    perf = {"window_sec": 10.0, "capture_fps": 24.0, "inferred_fps": 0.0, "frames_dropped_busy": 10,
            "skipped_blur_fps": 4.7, "skipped_settle_fps": 0.0, "frame_age_avg_ms": 0.0, "at": time.time()}

    cam = monitor(tmp_path, {"Otteson1": worker(perf)})["cameras"][0]

    assert cam["detect_enabled"] is True and cam["blur_threshold"] == 50.0
    assert cam["detection"]["checked_fps"] == 0.0
    assert cam["detection"]["blurry_fps"] == 4.7
    assert cam["detection"]["busy_fps"] == 1.0


def test_a_stale_perf_window_is_not_reported_as_current(tmp_path):
    perf = {"window_sec": 10.0, "capture_fps": 24.0, "inferred_fps": 4.7, "at": time.time() - 120}

    cam = monitor(tmp_path, {"Otteson1": worker(perf)})["cameras"][0]

    assert cam["detection"] is None


def test_the_payload_carries_the_detector_capacity(tmp_path):
    body = monitor(tmp_path, {})

    cap = body["capacity"]
    assert cap["available"] is True
    for key in ("live_pct", "analysis_pct", "busy_pct", "window_s"):
        assert key in cap
