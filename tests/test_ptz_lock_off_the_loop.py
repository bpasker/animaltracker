"""Nothing the event loop calls may wait for the PTZ tracker's lock.

Background (bug hunt, 2026-09-19): ``PTZTracker._lock`` is held for the whole
of an update, ONVIF requests included, so it stays taken for as long as the
camera takes to answer: tens to hundreds of milliseconds normally, up to the
5 s SOAP timeout when its ONVIF service stalls. The pipeline took that lock
on the event loop: ``trim_old_decisions`` on every processed frame, and
``get_decisions_in_window`` and ``clear_lock`` whenever an event closed. cam1
and cam2 share one tracker, so while cam2's update was mid-request, cam1's
frame handler blocked the loop, and with it every camera's reads, every
stream and the whole web UI, for the rest of that request. The Live page's
patrol and track toggles did the same, and made ONVIF requests of their own
on the loop.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from animaltracker.detector import Detection
from animaltracker.ptz_tracker import PTZMode, create_ptz_tracker
from animaltracker.web import WebServer

W, H = 1920, 1080
PROMPT = 0.25  # seconds; the blocked request below lasts until the test ends it


def det(cx, cy, w, h, conf=0.9, track_id=None):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    return Detection(species="animal", confidence=conf,
                     bbox=[x1, y1, x1 + w * W, y1 + h * H], track_id=track_id)


class StalledOnvif:
    """An ONVIF client whose next move hangs until the test lets it go."""

    def __init__(self) -> None:
        self.calls: list = []
        self.in_request = threading.Event()
        self.answer = threading.Event()
        self.answer.set()

    def stall_next_move(self) -> None:
        self.in_request.clear()
        self.answer.clear()

    def ptz_get_position(self, token):
        return {"pan": 0.0, "tilt": 0.0, "zoom": 0.2, "available": True}

    def ptz_move(self, token, p, t, z):
        self.calls.append("move")
        self.in_request.set()
        assert self.answer.wait(10), "the test never let the camera answer"

    def ptz_stop(self, token):
        self.calls.append("stop")

    def get_presets(self, *a, **k):
        return []


def make_tracker():
    onvif = StalledOnvif()
    tracker = create_ptz_tracker(onvif, "Profile_1", {"update_interval": 0.0, "tracking_step_duration": 0.0})
    tracker.set_track_enabled(True)
    tracker._mode = PTZMode.TRACKING
    return tracker, onvif


@pytest.fixture
def mid_request():
    """A tracker whose update is stuck inside an ONVIF request, holding the lock."""
    tracker, onvif = make_tracker()
    tracker.update([det(0.5, 0.5, 0.2, 0.2, track_id=3)], W, H)       # locks on to track 3
    assert tracker._locked_track_id == 3
    onvif.stall_next_move()
    update = threading.Thread(
        target=tracker.update, args=([det(0.85, 0.5, 0.2, 0.2, track_id=3)], W, H), daemon=True)
    update.start()
    assert onvif.in_request.wait(5), "the update never reached the camera"
    assert tracker._lock.locked()
    try:
        yield tracker, onvif, update
    finally:
        onvif.answer.set()
        update.join(5)


def timed(call):
    started = time.monotonic()
    result = call()
    return result, time.monotonic() - started


# --- what the pipeline calls from the event loop -----------------------------------

def test_the_per_frame_trim_does_not_wait_for_the_camera(mid_request):
    tracker, _, _ = mid_request

    removed, took = timed(lambda: tracker.trim_old_decisions(time.time() - 300))

    assert took < PROMPT and removed == 0


def test_reading_the_decisions_for_a_closing_event_does_not_wait_either(mid_request):
    tracker, _, _ = mid_request
    now = time.time()

    decisions, took = timed(lambda: tracker.get_decisions_in_window(now - 60, now + 60))
    assert took < PROMPT
    assert decisions and all("event" in d for d in decisions)

    (count, oldest, newest), took = timed(tracker.decision_log_span)
    assert took < PROMPT and count == len(tracker.get_decision_log()) and oldest <= newest


def test_clearing_the_lock_never_blocks_and_the_next_update_starts_clean(mid_request):
    tracker, onvif, update = mid_request

    _, took = timed(tracker.clear_lock)

    assert took < PROMPT
    assert tracker._lock_reset_pending is True
    assert tracker._locked_track_id == 3            # the update in flight keeps the state it started with

    onvif.answer.set()
    update.join(5)
    assert tracker._locked_track_id == 3            # ...to its end, as when clear_lock() waited for it

    tracker.update([], W, H)                        # the next one resets before it looks at anything
    assert tracker._lock_reset_pending is False
    assert tracker._locked_track_id is None


def test_an_uncontended_clear_lock_resets_at_once():
    tracker, _ = make_tracker()
    tracker.update([det(0.5, 0.5, 0.2, 0.2, track_id=3)], W, H)
    assert tracker._locked_track_id == 3

    tracker.clear_lock()

    assert tracker._locked_track_id is None and tracker._lock_reset_pending is False
    assert not tracker._lock.locked()


def test_the_multi_camera_update_honours_a_pending_reset_too():
    tracker, _ = make_tracker()
    tracker.update([det(0.5, 0.5, 0.2, 0.2, track_id=3)], W, H)
    tracker._lock_reset_pending = True

    tracker.update_multi_camera({"cam1": ([], W, H)}, "cam1", "cam2")

    assert tracker._lock_reset_pending is False and tracker._locked_track_id is None


def test_the_log_survives_being_written_trimmed_and_read_at_once():
    tracker, _ = make_tracker()
    errors: list = []
    stop = threading.Event()

    def write() -> None:
        try:
            for i in range(4000):
                tracker._log_decision("move", {"i": i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    while not stop.is_set():
        try:
            tracker.trim_old_decisions(time.time() - 300)
            tracker.get_decisions_in_window(0, time.time() + 1)
            tracker.decision_log_span()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            break
    writer.join(5)

    assert errors == []
    log = tracker.get_decision_log()
    assert 0 < len(log) <= tracker._decision_log_max_entries
    indexes = [d["details"]["i"] for d in log if d["event"] == "move" and "i" in d["details"]]
    assert indexes == sorted(indexes) and indexes[-1] == 3999      # nothing lost from the tail, order kept


# --- the Live page's toggles -----------------------------------------------------------

class JsonRequest:
    def __init__(self, body, camera_id="cam2") -> None:
        self._body = body
        self.match_info = {"camera_id": camera_id}

    async def json(self):
        return self._body


class RecordingTracker:
    def __init__(self) -> None:
        self.threads: dict = {}
        self._mode = PTZMode.IDLE
        self.patrol = False
        self.track = False

    def set_patrol_enabled(self, enabled) -> None:
        self.threads["patrol"] = threading.current_thread()
        self.patrol = bool(enabled)

    def set_track_enabled(self, enabled) -> None:
        self.threads["track"] = threading.current_thread()
        self.track = bool(enabled)

    def is_patrol_enabled(self) -> bool:
        return self.patrol

    def is_track_enabled(self) -> bool:
        return self.track


def test_the_patrol_and_track_toggles_touch_the_tracker_off_the_event_loop(tmp_path):
    tracker = RecordingTracker()
    worker = SimpleNamespace(ptz_tracker=tracker, onvif_client=object(), onvif_profile_token="P1")
    server = WebServer({"cam2": worker}, tmp_path, tmp_path / "logs", port=0)
    seen: dict = {}

    async def toggle() -> None:
        seen["loop"] = threading.current_thread()
        seen["patrol"] = await server.handle_ptz_patrol(JsonRequest({"enabled": True}))
        seen["track"] = await server.handle_ptz_track(JsonRequest({"enabled": True}))

    asyncio.run(toggle())

    assert seen["patrol"].status == 200 and seen["track"].status == 200
    assert json.loads(seen["track"].body)["track_enabled"] is True
    assert tracker.patrol and tracker.track
    assert tracker.threads["patrol"] is not seen["loop"]
    assert tracker.threads["track"] is not seen["loop"]
