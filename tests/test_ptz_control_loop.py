"""Regression tests for the PTZ control-loop refinements.

Each test pins a behaviour observed in the 2026-05-09 incident reviews:
zoom-in never firing, the spatial lock rejecting the subject after the
camera's own slew, LOCK_SKIP refusing re-acquisition mid-track, Stop spam on
hold, an abandoned step-stop after one ONVIF error, and duplicate moves for
one re-read detection.
"""
import time

from animaltracker.detector import Detection
from animaltracker.ptz_tracker import PTZMode, create_ptz_tracker


class FakeOnvif:
    def __init__(self, fail_stops: int = 0):
        self.calls = []
        self._fail_stops = fail_stops

    def ptz_get_position(self, token):
        return {"pan": 0.0, "tilt": 0.0, "zoom": 0.2, "available": True}

    def ptz_move(self, token, p, t, z):
        self.calls.append(("move", round(p, 3), round(t, 3), round(z, 3)))

    def ptz_stop(self, token):
        if self._fail_stops > 0:
            self._fail_stops -= 1
            raise RuntimeError("simulated ONVIF timeout")
        self.calls.append(("stop",))

    def get_presets(self, *a, **k):
        return []

    def moves(self):
        return [c for c in self.calls if c[0] == "move"]

    def stops(self):
        return [c for c in self.calls if c[0] == "stop"]


W, H = 1920, 1080


def det(cx, cy, w, h, conf=0.9, track_id=None):
    """Detection centred at normalized (cx, cy) with normalized size (w, h)."""
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    return Detection(species="animal", confidence=conf,
                     bbox=[x1, y1, x1 + w * W, y1 + h * H], track_id=track_id)


def tracker(**cfg):
    cfg.setdefault("update_interval", 0.0)
    cfg.setdefault("tracking_step_duration", 0.0)   # no timer threads
    onvif = FakeOnvif(fail_stops=cfg.pop("fail_stops", 0))
    t = create_ptz_tracker(onvif, "Profile_1", cfg)
    t.set_track_enabled(True)
    t._mode = PTZMode.TRACKING
    return t, onvif


# --------------------------------------------------------------------------
# Zoom-in was structurally dead
# --------------------------------------------------------------------------

def test_deadzone_zooms_in_on_centred_small_subject() -> None:
    t, onvif = tracker(target_fill_pct=0.6)
    t._current_capture_ts = time.time()
    moved = t._do_tracking_from_target([det(0.5, 0.5, 0.20, 0.15)], W, H, "cam2")
    assert moved
    assert onvif.moves() == [("move", 0.0, 0.0, 0.3)], "expected a zoom-only pulse"
    assert onvif.stops() == []
    assert t._decision_log[-1].event == "deadzone_zoom"


def test_deadzone_parks_when_fill_is_on_target() -> None:
    t, onvif = tracker(target_fill_pct=0.6)
    t._current_capture_ts = time.time()
    moved = t._do_tracking_from_target([det(0.5, 0.5, 0.60, 0.45)], W, H, "cam2")
    assert not moved
    assert onvif.moves() == []
    assert onvif.stops() == [("stop",)]


def test_deadzone_zooms_out_on_overfilled_subject() -> None:
    t, onvif = tracker(target_fill_pct=0.6)
    t._current_capture_ts = time.time()
    t._do_tracking_from_target([det(0.5, 0.5, 0.95, 0.60)], W, H, "cam2")
    assert onvif.moves() and onvif.moves()[0][3] < 0


def test_zoom_in_allowed_while_nearly_centred() -> None:
    """Offset 0.10 is past the deadzone but inside the new 0.15 gate."""
    t, onvif = tracker(target_fill_pct=0.6, low_fill_threshold=0.0)  # no cap noise
    t._current_capture_ts = time.time()
    t._do_tracking_from_target([det(0.60, 0.5, 0.20, 0.15)], W, H, "cam2")
    (m,) = onvif.moves()
    assert m[1] > 0 and m[3] > 0, f"expected pan and zoom-in together, got {m}"


def test_zoom_in_suppressed_while_far_off_centre() -> None:
    t, onvif = tracker(target_fill_pct=0.6, low_fill_threshold=0.0)
    t._current_capture_ts = time.time()
    t._do_tracking_from_target([det(0.80, 0.5, 0.20, 0.15)], W, H, "cam2")
    (m,) = onvif.moves()
    assert m[1] > 0 and m[3] == 0.0


# --------------------------------------------------------------------------
# Spatial lock must survive the camera's own slew
# --------------------------------------------------------------------------

def _lock_at(t, cx, cy, source="cam2"):
    t._ptz_camera_id = "cam2"
    chosen = t._select_best_detection([det(cx, cy, 0.1, 0.1, track_id=5)], W, H, source_camera=source)
    assert chosen is not None and t._locked_track_id == 5
    return chosen


def test_spatial_lock_continues_after_own_move_displaces_subject() -> None:
    """Incident: anchor (0.70, 0.84) -> after a good slew the animal is at
    (0.47, 0.50), dist 0.41. The tight radius produced LOCK_HOLD x3."""
    t, _ = tracker()
    _lock_at(t, 0.70, 0.84)
    t._last_move_time = time.time() + 0.01           # we moved since that sighting
    chosen = t._select_best_detection([det(0.47, 0.50, 0.1, 0.1, track_id=None)], W, H,
                                      source_camera="cam2")
    assert chosen is not None, "subject displaced by our own slew must keep the lock"
    assert t._consecutive_lock_misses == 0 or chosen is not None


def test_spatial_lock_stays_tight_without_own_move() -> None:
    t, _ = tracker()
    _lock_at(t, 0.70, 0.84)
    t._last_move_time = 0.0                           # no move since the sighting
    chosen = t._select_best_detection([det(0.47, 0.50, 0.1, 0.1, track_id=None)], W, H,
                                      source_camera="cam2")
    assert chosen is None, "a 0.41 jump with no camera motion is a different object"


def test_spatial_lock_stays_tight_for_static_wide_camera_anchor() -> None:
    """cam2 moving does not displace anything in cam1's frame."""
    t, _ = tracker()
    _lock_at(t, 0.70, 0.84, source="cam1")
    t._last_move_time = time.time() + 0.01
    chosen = t._select_best_detection([det(0.47, 0.50, 0.1, 0.1, track_id=None)], W, H,
                                      source_camera="cam1")
    assert chosen is None


# --------------------------------------------------------------------------
# Re-acquisition threshold depends on whether we are already tracking
# --------------------------------------------------------------------------

def test_untracked_lock_accepted_at_65pct_while_tracking() -> None:
    t, _ = tracker()
    chosen = t._select_best_detection([det(0.5, 0.5, 0.1, 0.1, conf=0.65)], W, H, "cam2")
    assert chosen is not None


def test_untracked_lock_refused_at_65pct_from_patrol() -> None:
    t, _ = tracker()
    t._mode = PTZMode.PATROL
    chosen = t._select_best_detection([det(0.5, 0.5, 0.1, 0.1, conf=0.65)], W, H, "cam2")
    assert chosen is None, "cold lock on an unconfirmed 65% flash is how the leaf hang happened"


# --------------------------------------------------------------------------
# Hold path, cross-camera handoff, step-stop retry, duplicate re-reads
# --------------------------------------------------------------------------

def test_hold_path_issues_stop_once() -> None:
    t, onvif = tracker()
    _lock_at(t, 0.70, 0.84)
    t._last_move_time = 0.0
    far = [det(0.10, 0.10, 0.1, 0.1, track_id=None)]
    for _ in range(3):
        t._current_capture_ts = time.time()
        t._do_tracking_from_target(far, W, H, "cam2")
    assert len(onvif.stops()) == 1


def test_cross_camera_handoff_resets_smoothed_offset() -> None:
    t, _ = tracker()
    _lock_at(t, 0.2, 0.2, source="cam1")
    t._target_pan, t._target_tilt = 0.8, -0.8          # residual from cam1 frame
    t._select_best_detection([det(0.5, 0.5, 0.1, 0.1, track_id=9)], W, H, source_camera="cam2")
    assert (t._target_pan, t._target_tilt) == (0.0, 0.0)


def test_step_stop_retries_after_onvif_failure() -> None:
    t, onvif = tracker(tracking_step_duration=0.05, fail_stops=1)
    t._current_capture_ts = time.time()
    t._do_tracking_from_target([det(0.80, 0.5, 0.20, 0.15)], W, H, "cam2")
    assert onvif.moves()
    time.sleep(0.4)                                   # step timer + one retry
    with t._lock:
        pass
    assert onvif.stops() == [("stop",)], "the retried Stop must land"
    assert t._holding_position


def test_identical_bbox_from_other_worker_is_not_a_second_move() -> None:
    t, onvif = tracker(low_fill_threshold=0.0)
    d = [det(0.80, 0.5, 0.20, 0.15)]
    t._current_capture_ts = 1000.000
    assert t._do_tracking_from_target(d, W, H, "cam2")
    t._current_capture_ts = 1000.180                  # other worker's capture_ts
    assert not t._do_tracking_from_target(d, W, H, "cam2")
    assert len(onvif.moves()) == 1


def test_fresh_bbox_shortly_after_a_move_still_moves() -> None:
    t, onvif = tracker(low_fill_threshold=0.0)
    t._current_capture_ts = 1000.000
    assert t._do_tracking_from_target([det(0.80, 0.5, 0.20, 0.15)], W, H, "cam2")
    t._current_capture_ts = 1000.180
    assert t._do_tracking_from_target([det(0.78, 0.5, 0.20, 0.15)], W, H, "cam2")
    assert len(onvif.moves()) == 2
