"""The journal was ~90% periodic [PERF] and per-frame [REALTIME] lines, and an
RTSP outage logged an ERROR every 5 s. These pin the throttles that fix it."""
import logging
from types import SimpleNamespace

import pytest

import animaltracker.pipeline as pipeline
from animaltracker.pipeline import StreamWorker, _format_duration


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(pipeline.time, "time", c)
    return c


def make_worker():
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(id="cam1")
    w._init_log_state()
    return w


def records(caplog, level=None):
    out = [r for r in caplog.records if r.name == "animaltracker.pipeline"]
    if level is not None:
        out = [r for r in out if r.levelno == level]
    return out


def test_format_duration():
    assert _format_duration(45) == "45s"
    assert _format_duration(185) == "3m 05s"
    assert _format_duration(8040) == "2h 14m"


def test_rtsp_open_failures_log_error_once_then_a_minute_roll_up(caplog, clock):
    caplog.set_level(logging.DEBUG, logger="animaltracker.pipeline")
    w = make_worker()

    w._log_rtsp_open_failure()
    assert [r.levelno for r in records(caplog)] == [logging.ERROR]
    assert "Unable to open RTSP stream for cam1" in records(caplog)[0].getMessage()

    for _ in range(11):  # 55 s of 5 s retries: nothing above DEBUG
        clock.t += 5
        w._log_rtsp_open_failure()
    assert [r.levelno for r in records(caplog) if r.levelno > logging.DEBUG] == [logging.ERROR]
    assert len(records(caplog, logging.DEBUG)) == 11

    clock.t += 5  # 60 s since the first failure -> one WARNING roll-up
    w._log_rtsp_open_failure()
    warnings = records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "13 attempts over 1m 00s" in warnings[0].getMessage()

    clock.t += 30
    w._log_rtsp_connected()
    infos = records(caplog, logging.INFO)
    assert len(infos) == 1
    assert infos[0].getMessage() == "Connected to stream for cam1 after 13 failed attempts over 1m 30s"
    assert w._rtsp_fail_count == 0

    caplog.clear()
    w._log_rtsp_connected()
    assert [r.getMessage() for r in records(caplog)] == ["Connected to stream for cam1"]


def test_raw_detections_are_info_once_per_burst(clock):
    w = make_worker()
    assert w._raw_detection_log_level() == logging.INFO
    clock.t += 1
    assert w._raw_detection_log_level() == logging.DEBUG
    clock.t += 9  # still inside the burst (gap measured from the last frame)
    assert w._raw_detection_log_level() == logging.DEBUG
    clock.t += 11  # quiet for more than the 10 s gap -> new burst
    assert w._raw_detection_log_level() == logging.INFO


def test_tracker_update_failure_is_warned_once_with_traceback(caplog):
    caplog.set_level(logging.DEBUG, logger="animaltracker.pipeline")
    w = make_worker()
    err = RuntimeError("tracker exploded")
    w._note_tracker_update_failure(err)
    w._note_tracker_update_failure(err)
    w._note_tracker_update_failure(err)
    levels = [r.levelno for r in records(caplog)]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    first = records(caplog)[0]
    assert first.exc_info and first.exc_info[1] is err
    assert "tracker exploded" in first.getMessage()


def _perf_worker(clock):
    w = make_worker()
    w.detector = SimpleNamespace(pop_perf_stats=lambda: {})
    w.perf_infer_count = 0
    w.perf_infer_time_total = 0.0
    w.perf_infer_time_max = 0.0
    w.perf_frame_age_total = 0.0
    w.perf_frame_age_max = 0.0
    w.perf_frames_read = 0
    w.perf_frames_dropped_busy = 0
    w._perf_window_start = clock.t
    w._perf_log_interval = 10.0
    w.perf_last_snapshot = {}
    w._perf_summary_start = clock.t
    return w


def test_perf_window_is_debug_and_the_minute_roll_up_is_info(caplog, clock):
    caplog.set_level(logging.DEBUG, logger="animaltracker.pipeline")
    w = _perf_worker(clock)
    for window in range(6):
        clock.t += 10
        w.perf_frames_read = 200
        w.perf_infer_count = 5
        w.perf_frames_dropped_busy = 5
        w.perf_frame_age_total = 0.75  # 150 ms average
        w.perf_frame_age_max = 0.2 + 0.01 * window
        w._maybe_log_perf_stats()
        assert w.perf_last_snapshot["capture_fps"] == 20.0, "10 s snapshot still feeds /api/cameras"
        if window < 5:
            assert records(caplog, logging.INFO) == []
    debug_lines = [r.getMessage() for r in records(caplog, logging.DEBUG)]
    assert len(debug_lines) == 6
    assert debug_lines[0].startswith("[PERF] cam1: capture=20.0fps infer=0.5fps drop=5 (50.0%)")
    assert debug_lines[0].endswith("| window=10s")

    info_lines = [r.getMessage() for r in records(caplog, logging.INFO)]
    assert info_lines == [
        "[PERF] cam1: capture=20.0fps infer=0.5fps drop=30 (50.0%) "
        "frame_age avg=150ms max=250ms | window=60s"
    ]
    assert w._perf_summary_infer_count == 0, "roll-up counters reset after logging"


def test_perf_roll_up_weights_stage_timings_by_call_count(caplog, clock):
    caplog.set_level(logging.INFO, logger="animaltracker.pipeline")
    w = _perf_worker(clock)
    w._accumulate_perf_summary(
        infer_count=5, capture_count=200, dropped=5,
        stage_stats={'count': 5, 'prep_avg_ms': 10, 'prep_max_ms': 12,
                     'infer_avg_ms': 100, 'infer_max_ms': 110, 'post_avg_ms': 1, 'post_max_ms': 2},
    )
    w._accumulate_perf_summary(
        infer_count=15, capture_count=200, dropped=5,
        stage_stats={'count': 15, 'prep_avg_ms': 30, 'prep_max_ms': 40,
                     'infer_avg_ms': 100, 'infer_max_ms': 130, 'post_avg_ms': 1, 'post_max_ms': 2},
    )
    w._log_perf_summary(clock.t + 60, 60.0)
    (line,) = [r.getMessage() for r in records(caplog, logging.INFO)]
    # prep: (10*5 + 30*15) / 20 = 25 ms; max is the max of maxima.
    assert "prep=25(40) infer=100(130) post=1(2)" in line
    assert "capture=6.7fps infer=0.3fps drop=10 (33.3%)" in line
