"""Saving the last 30 seconds by hand: written before it is reported, at the
rate it was captured, without deleting anything to make room.

Bug hunt items 3.29 and 3.15, plus the manual-clip half of the 15 fps stamp:

- The command palette's toast read ``res.filename`` from a plain-text
  "Clip saved: <name>" response, so it never showed the name.
- The endpoint answered before the file existed: the write ran on a bare
  thread, so "Clip saved" and its "View" link came seconds early, and a
  failed write (full disk, no encoder) was visible only in the log.
- ``write_clip`` stamped 15 fps like the old event clips: cam1's 30 s of
  20 fps played as 40 s.
- On a full disk ``ensure_space_for_clip`` deleted the oldest clips, video
  only, with no ``min_days`` floor. Its glob was one level short of the
  archive layout so it found none, but it must never be fixed into working.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from animaltracker.pipeline import StreamWorker
from animaltracker.storage import StorageManager
from animaltracker.web import WebServer

SRC = Path(__file__).resolve().parents[1] / "src" / "animaltracker"


def worker(tmp_path, rate=20.0, seconds=12.0, perf=None):
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(id="cam1")
    w.storage = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    w.perf_last_snapshot = perf
    now = time.time()
    n = int(rate * seconds)
    frames = [(now - seconds + i / rate, np.zeros((8, 8, 3), np.uint8)) for i in range(n)]
    w.clip_buffer = SimpleNamespace(dump=lambda: list(frames))
    return w


class RecordingWrite:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, frames, path, fps=15):
        self.calls.append((len(frames), path, fps))
        if self.ok:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mp4")
        return self.ok


# --- the worker --------------------------------------------------------------------------------

def test_a_manual_clip_is_stamped_at_the_rate_its_frames_arrived(tmp_path):
    w = worker(tmp_path, rate=20.0)
    w.storage.write_clip = RecordingWrite()

    name = w.save_manual_clip()

    assert name and re.match(r"manual_cam1_\d+\.mp4$", name)
    (count, path, fps) = w.storage.write_clip.calls[0]
    assert fps == 20.0
    assert path == w.storage.storage_root / "clips" / name


def test_only_the_last_thirty_seconds_are_written(tmp_path):
    w = worker(tmp_path, rate=10.0, seconds=45.0)
    w.storage.write_clip = RecordingWrite()

    w.save_manual_clip()

    count = w.storage.write_clip.calls[0][0]
    assert 295 <= count <= 301


def test_a_write_that_fails_is_no_clip(tmp_path):
    w = worker(tmp_path)
    w.storage.write_clip = RecordingWrite(ok=False)

    assert w.save_manual_clip() is None


def test_an_empty_buffer_is_no_clip(tmp_path):
    w = worker(tmp_path, seconds=0.0)
    w.storage.write_clip = RecordingWrite()

    assert w.save_manual_clip() is None and w.storage.write_clip.calls == []


def test_too_few_frames_to_measure_falls_back_to_the_perf_window(tmp_path):
    w = worker(tmp_path, rate=24.0, seconds=0.3, perf={"capture_fps": 24.0})
    w.storage.write_clip = RecordingWrite()

    w.save_manual_clip()

    assert w.storage.write_clip.calls[0][2] == 24.0


# --- the endpoint ------------------------------------------------------------------------------

def make_server(tmp_path, worker_obj):
    return WebServer({"cam1": worker_obj} if worker_obj else {}, tmp_path / "storage", tmp_path / "logs", port=19900)


def call(server, camera_id):
    request = SimpleNamespace(match_info={"camera_id": camera_id})
    return asyncio.run(server.handle_save_clip(request))


def test_the_response_is_json_with_the_filename_once_the_clip_exists(tmp_path):
    w = worker(tmp_path)
    w.storage.write_clip = RecordingWrite()
    server = make_server(tmp_path, w)

    response = call(server, "cam1")

    assert response.status == 200 and response.content_type == "application/json"
    import json
    body = json.loads(response.text)
    assert body["filename"] == body["path"] and body["filename"].startswith("manual_cam1_")
    assert (w.storage.storage_root / "clips" / body["filename"]).exists(), "answered before the file existed"


def test_the_archive_lists_the_new_clip_at_once(tmp_path):
    """The scan is cached for a few seconds; the toast's "View" link must
    find the clip that was just written, so the save invalidates it."""
    w = worker(tmp_path)
    w.storage.write_clip = RecordingWrite()
    server = make_server(tmp_path, w)
    assert server._scan_recordings_cached() == []          # cached: empty

    import json
    body = json.loads(call(server, "cam1").text)

    assert [c["filename"] for c in server._scan_recordings_cached()] == [body["filename"]]


def test_the_write_runs_on_the_executor_not_the_loop(tmp_path):
    import threading
    w = worker(tmp_path)
    seen = []

    def write(frames, path, fps=15):
        seen.append(threading.current_thread() is threading.main_thread())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        return True

    w.storage.write_clip = write
    call(make_server(tmp_path, w), "cam1")

    assert seen == [False]


def test_a_failed_write_is_a_500_with_a_reason(tmp_path):
    import json
    w = worker(tmp_path)
    w.storage.write_clip = RecordingWrite(ok=False)

    response = call(make_server(tmp_path, w), "cam1")

    assert response.status == 500
    assert "write failed" in json.loads(response.text)["error"]


def test_an_unknown_camera_is_a_404(tmp_path):
    assert call(make_server(tmp_path, None), "nope").status == 404


# --- the page reads what the server sends -----------------------------------------------------

def test_both_callers_take_the_filename_from_the_json_body():
    app = (SRC / "static" / "app.js").read_text()
    live = (SRC / "static" / "views" / "live.js").read_text()
    assert "var filename = res && typeof res === 'object' ? res.filename : null;" in app
    # View searches for the saved file, and is offered only when there is one,
    # as on the Live card.
    assert "action: filename ? { label: 'View'" in app
    assert "router.go('/recordings', { q: filename });" in app
    assert "var filename = res && typeof res === 'object' ? (res.filename || null) : null;" in live


# --- space is checked, never made -------------------------------------------------------------

def test_the_space_check_deletes_nothing(tmp_path, monkeypatch):
    st = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    day = st.storage_root / "clips" / "cam1" / "2026" / "01" / "01"
    day.mkdir(parents=True)
    old = day / "1767225600_bird.mp4"
    old.write_bytes(b"x" * 1024)
    monkeypatch.setattr(st, "has_sufficient_space", lambda required: False)

    assert st.ensure_space_for_clip(10_000_000) is False
    assert old.exists()


def test_the_write_is_refused_not_forced_when_the_disk_is_full(tmp_path, monkeypatch):
    st = StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")
    monkeypatch.setattr(st, "has_sufficient_space", lambda required: False)
    out = st.storage_root / "clips" / "manual_cam1_1.mp4"

    ok = st.write_clip([(0.0, np.zeros((8, 8, 3), np.uint8))] * 5, out, fps=20)

    assert ok is False and not out.exists()


def test_nothing_in_storage_prunes_outside_retention():
    src = (SRC / "storage.py").read_text()
    assert "get_clips_sorted_by_age" not in src
    body = src[src.index("def ensure_space_for_clip"):]
    body = body[:body.index("\n    def ") if "\n    def " in body else len(body)]
    assert "unlink" not in body
