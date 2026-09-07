"""Tests for StreamingClipWriter's off-loop encoding.

Pins the fix for BUGS.md item 3: seeding the pre-roll and writing live
frames must return immediately (the encode happens on the writer thread),
close() must drain everything before returning, and a slow encoder must
shed live frames rather than block the caller.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from animaltracker import storage
from animaltracker.storage import StreamingClipWriter


class FakeVideoWriter:
    """Stand-in for cv2.VideoWriter: one byte per frame, optional delay."""

    delay = 0.0
    opened = True
    fail_on_frame = None  # 0-based index of a write() that raises
    instances: list = []

    def __init__(self, path, fourcc, fps, size):
        self.path = Path(path)
        self.size = size
        self.writes = 0
        self.released = False
        self.thread = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")
        type(self).instances.append(self)

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.thread = threading.current_thread()
        idx = self.writes
        self.writes += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on_frame is not None and idx == self.fail_on_frame:
            raise RuntimeError("simulated encoder failure")
        with self.path.open("ab") as fh:
            fh.write(b"x")

    def release(self):
        self.released = True


@pytest.fixture
def fake_writer(monkeypatch):
    """Patch cv2.VideoWriter with a fresh FakeVideoWriter subclass per test."""
    def _install(**attrs):
        cls = type("PatchedFakeVideoWriter", (FakeVideoWriter,), {"instances": [], **attrs})
        monkeypatch.setattr(storage.cv2, "VideoWriter", cls)
        return cls
    return _install


def frames(n, h=8, w=8):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_seed_and_write_return_before_anything_is_encoded(fake_writer, tmp_path):
    fake = fake_writer(delay=0.005)
    writer = StreamingClipWriter(tmp_path / "ev.avi", fps=15)

    started = time.perf_counter()
    assert writer.seed(frames(100)) == 100
    for frame in frames(20):
        writer.write(frame)
    elapsed = time.perf_counter() - started
    # 120 encodes at 5ms each is >= 0.6s if done inline; queueing is microseconds.
    assert elapsed < 0.1

    out = writer.close()
    assert out == tmp_path / "ev.avi"
    assert writer.frame_count == 120
    assert out.stat().st_size == 120
    assert writer.dropped_frames == 0
    inst = fake.instances[0]
    assert inst.released
    assert inst.thread is not threading.main_thread()


def test_slow_encoder_sheds_live_frames_but_never_seed_frames(fake_writer, tmp_path):
    fake_writer(delay=0.02)
    writer = StreamingClipWriter(tmp_path / "ev.avi", fps=15, max_pending=10)

    assert writer.seed(frames(50)) == 50  # exempt from max_pending
    for frame in frames(30):  # backlog is ~50 >= 10 for every one of these
        writer.write(frame)
    writer.close()

    assert writer.dropped_frames == 30
    assert writer.frame_count == 50


def test_encoder_that_cannot_open_yields_no_clip(fake_writer, tmp_path):
    fake_writer(opened=False)
    path = tmp_path / "ev.avi"
    writer = StreamingClipWriter(path, fps=15)
    for frame in frames(5):
        writer.write(frame)
    assert writer.close() is None
    assert writer.frame_count == 0
    assert not path.exists()


def test_write_error_is_survived_and_earlier_frames_are_kept(fake_writer, tmp_path):
    fake_writer(fail_on_frame=2)
    writer = StreamingClipWriter(tmp_path / "ev.avi", fps=15)
    for frame in frames(5):
        writer.write(frame)
    out = writer.close()
    assert out is not None
    assert writer.frame_count == 4
    assert writer.write_errors == 1


def test_frames_of_a_different_size_are_skipped(fake_writer, tmp_path):
    fake_writer()
    writer = StreamingClipWriter(tmp_path / "ev.avi", fps=15)
    writer.write(frames(1, 8, 8)[0])
    writer.write(frames(1, 16, 16)[0])
    writer.write(frames(1, 8, 8)[0])
    writer.close()
    assert writer.frame_count == 2


def test_close_with_nothing_written_returns_none_and_late_writes_are_ignored(fake_writer, tmp_path):
    fake = fake_writer()
    path = tmp_path / "ev.avi"
    writer = StreamingClipWriter(path, fps=15)
    assert writer.close() is None
    assert not path.exists()
    assert fake.instances == []  # encoder never opened
    writer.write(frames(1)[0])  # no-op after close: must not raise or count
    assert writer.frame_count == 0
