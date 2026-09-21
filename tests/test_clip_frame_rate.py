"""A clip is stamped with the rate its frames were captured at.

The event writer was created with a hard-coded ``fps=15`` whatever the camera
delivered. cam1 captures 20 fps and every frame is written, so its clips
played 1.33x slow; Otteson1 captures 24 fps, 1.6x. Every time measured in
clip seconds was stretched to match: the player, the clip page, the sidecar's
``duration_seconds`` and key-frame times, and ByteTrack's lost-track window
in post-processing. Otteson2 (30 fps) looked about right in daylight only by
accident, because its encoder dropped about half the frames.

Now the rate is measured from the pre-roll buffer's capture times, and a
camera whose encoder falls behind records every 2nd frame from its next event
on, stamped at half the rate, so a truthful stamp never plays a clip fast.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker import storage as storage_mod
from animaltracker.clip_buffer import ClipBuffer
from animaltracker.pipeline import StreamWorker
from animaltracker.storage import StorageManager, StreamingClipWriter, measured_frame_rate


def times(rate: float, seconds: float, start: float = 1_789_000_000.0) -> list:
    n = int(rate * seconds)
    return [start + i / rate for i in range(n)]


# --- measuring the rate --------------------------------------------------------------------

@pytest.mark.parametrize("rate", [20, 24, 30, 15])
def test_a_steady_stream_measures_as_its_rate(rate):
    assert measured_frame_rate(times(rate, 10)) == float(rate)


def test_a_stall_in_the_stream_is_not_counted_as_slow_frames():
    before = times(20, 5)
    after = times(20, 5, start=before[-1] + 3.0)      # the camera went quiet for 3 s
    assert measured_frame_rate(before + after) == 20.0


def test_read_jitter_is_snapped_to_the_cameras_whole_rate():
    rng = np.random.default_rng(7)
    ts = np.cumsum(rng.normal(1 / 20, 0.004, 400)) + 1_789_000_000.0
    assert measured_frame_rate(list(ts)) == 20.0


def test_a_rate_between_whole_numbers_is_kept():
    assert measured_frame_rate(times(12.5, 10)) == 12.5


@pytest.mark.parametrize("ts", [
    [],
    [1.0],
    times(20, 0.4),                       # under a second of frames
    [5.0] * 50,                           # no time passed at all
])
def test_too_little_to_measure_says_so(ts):
    assert measured_frame_rate(ts) is None


# --- the writer's stamp and stride -----------------------------------------------------------

def frame(h=48, w=64, value=0):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_the_file_is_stamped_at_the_rate_it_was_given(tmp_path):
    writer = StreamingClipWriter(tmp_path / "a.temp.avi", fps=20)
    writer.seed(frame(value=i) for i in range(10))
    for i in range(10):
        writer.write(frame(value=100 + i))
    path = writer.close()

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0, abs=0.01)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    finally:
        cap.release()


def test_a_stride_keeps_every_nth_frame_and_stamps_the_rate_that_leaves(tmp_path):
    writer = StreamingClipWriter(tmp_path / "b.temp.avi", fps=30, stride=2)
    seeded = writer.seed(frame() for _ in range(5))       # offered 0-4: keeps 0, 2, 4
    for _ in range(5):                                    # offered 5-9: keeps 6, 8
        writer.write(frame())
    path = writer.close()

    assert seeded == 3 and writer.live_frames == 2 and writer.frame_count == 5
    assert writer.fps == 15.0 and writer.capture_fps == 30.0
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(15.0, abs=0.01)
    finally:
        cap.release()


def test_without_a_stride_nothing_changes(tmp_path):
    writer = StreamingClipWriter(tmp_path / "c.temp.avi", fps=15)
    assert writer.seed(frame() for _ in range(4)) == 4
    writer.write(frame())
    writer.close()
    assert writer.frame_count == 5 and writer.fps == 15.0 and writer.live_frames == 1


def test_the_transcoded_clip_keeps_the_rate(tmp_path):
    st = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    writer = StreamingClipWriter(tmp_path / "d.temp.avi", fps=24)
    writer.seed(frame(value=i * 5) for i in range(48))
    temp = writer.close()
    out = tmp_path / "storage" / "clips" / "cam1" / "1789000000_animal.mp4"
    out.parent.mkdir(parents=True)

    if not st.transcode_avi_to_mp4(temp, out):
        pytest.skip("no H.264 encoder in this OpenCV build")
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(24.0, abs=0.05)
    finally:
        cap.release()


# --- the worker ------------------------------------------------------------------------------

class RecordingVideoWriter:
    """cv2.VideoWriter stand-in that remembers the rate it was opened at."""
    opened_at: list = []

    def __init__(self, path, fourcc, fps, size):
        type(self).opened_at.append(fps)
        Path(path).write_bytes(b"x")

    def isOpened(self):
        return True

    def write(self, frame):
        pass

    def release(self):
        pass


@pytest.fixture
def recording_writer(monkeypatch):
    cls = type("Recording", (RecordingVideoWriter,), {"opened_at": []})
    monkeypatch.setattr(storage_mod.cv2, "VideoWriter", cls)
    return cls


def worker(tmp_path, *, perf=None, stride=1) -> StreamWorker:
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(id="cam1")
    w.runtime = SimpleNamespace(general=SimpleNamespace(clip=SimpleNamespace(pre_seconds=10)))
    w.storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    w.clip_buffer = ClipBuffer(max_seconds=30, fps=15)
    w.perf_last_snapshot = perf
    w._record_stride = stride
    return w


def fill(w: StreamWorker, rate: float, seconds: float, end: float) -> None:
    for ts in times(rate, seconds, start=end - seconds):
        w.clip_buffer.push(ts, frame(8, 8))


def test_an_event_on_a_20_fps_camera_is_stamped_20(tmp_path, recording_writer):
    w = worker(tmp_path)
    start = time.time()
    fill(w, 20, 12, end=start)

    writer = w._open_event_writer(start)
    writer.close()

    assert writer.fps == 20.0 and recording_writer.opened_at == [20.0]
    # the pre-roll is still the last pre_seconds before the event
    assert writer.frame_count == pytest.approx(200, abs=2)


def test_the_learned_stride_halves_what_is_written_and_the_stamp(tmp_path, recording_writer):
    w = worker(tmp_path, stride=2)
    start = time.time()
    fill(w, 30, 12, end=start)

    writer = w._open_event_writer(start)
    writer.close()

    assert recording_writer.opened_at == [15.0]
    assert writer.frame_count == pytest.approx(150, abs=2)


def test_a_buffer_too_short_to_measure_falls_back_to_the_perf_window(tmp_path, recording_writer):
    w = worker(tmp_path, perf={"capture_fps": 24.0})
    start = time.time()
    fill(w, 24, 0.3, end=start)                           # just connected

    w._open_event_writer(start).close()

    assert recording_writer.opened_at == [24.0]


def test_with_nothing_to_go_on_it_is_the_old_15(tmp_path, recording_writer):
    w = worker(tmp_path, perf=None)

    writer = w._open_event_writer(time.time())
    writer.close()

    assert writer.fps == 15.0


# --- learning the stride ------------------------------------------------------------------

def closed(live, dropped, stride=1, capture=30.0):
    return SimpleNamespace(live_frames=live, dropped_frames=dropped, stride=stride,
                           capture_fps=capture, fps=capture / stride)


def test_an_event_that_lost_half_its_frames_halves_the_next(tmp_path, caplog):
    w = worker(tmp_path)
    with caplog.at_level(logging.WARNING):
        w._learn_record_stride(closed(live=320, dropped=163))

    assert w._record_stride == 2
    assert "kept only 49% of live frames at 30.0 fps; recording every 2nd frame (15.0 fps)" in caplog.text


def test_a_few_dropped_frames_change_nothing(tmp_path):
    w = worker(tmp_path)
    w._learn_record_stride(closed(live=600, dropped=3))     # a night event on Otteson2
    assert w._record_stride == 1


def test_an_event_too_short_to_judge_changes_nothing(tmp_path):
    w = worker(tmp_path)
    w._learn_record_stride(closed(live=20, dropped=15))
    assert w._record_stride == 1


def test_it_keeps_stepping_while_the_encoder_still_falls_behind_and_stops_at_4(tmp_path):
    w = worker(tmp_path)
    for _ in range(5):
        w._learn_record_stride(closed(live=300, dropped=100, stride=w._record_stride))
    assert w._record_stride == 4


def test_a_late_report_from_an_older_stride_is_ignored(tmp_path):
    """Two events closing out of order must not raise the stride twice."""
    w = worker(tmp_path, stride=2)
    w._learn_record_stride(closed(live=300, dropped=150, stride=1))
    assert w._record_stride == 2


def test_a_writer_without_counters_is_left_alone(tmp_path):
    w = worker(tmp_path)
    w._learn_record_stride(SimpleNamespace(frame_count=30))
    assert w._record_stride == 1
