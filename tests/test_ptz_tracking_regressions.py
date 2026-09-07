"""Regression tests for the PTZ tracking-loss fixes.

Each test pins one of the failures diagnosed from the 2026-05-09 coyote clip
(tmp/ptz_review_1778332418), where the controller lost a continuously visible
subject 3 times in 55s and never converged on centre.
"""
import time

from animaltracker.detector import Detection
from animaltracker.pipeline import StreamWorker
from animaltracker.ptz_tracker import PTZMode, PTZTracker, create_ptz_tracker


# --------------------------------------------------------------------------
# Fix 1: the patrol-oriented min-area gate must step aside while tracking
# --------------------------------------------------------------------------

class _FakeThresholds:
    min_detection_area = 0.005
    tracking_min_detection_area = 0.0005


class _FakeCamera:
    id = "cam1"
    thresholds = _FakeThresholds()


class _FakePTZTracker:
    def __init__(self, mode):
        self._mode = mode

    def is_track_enabled(self):
        return True

    def get_mode(self):
        return self._mode


def _worker(mode=None):
    w = object.__new__(StreamWorker)
    w.camera = _FakeCamera()
    w.ptz_tracker = _FakePTZTracker(mode) if mode else None
    w.ptz_drives_tracking = mode is not None
    return w


# cam1's sub-stream is 896x512; a coyote at range is ~60x30 px = 1800 px^2,
# below the 0.005 patrol gate (2294 px^2) but well above the tracking gate.
CAM1_W, CAM1_H = 896, 512
COYOTE_AT_RANGE = [Detection(species="animal", confidence=0.83,
                             bbox=[400.0, 240.0, 460.0, 270.0])]


def test_small_tracked_subject_survives_while_tracking() -> None:
    assert _worker("tracking")._filter_false_positives(
        COYOTE_AT_RANGE, CAM1_W, CAM1_H
    ) == COYOTE_AT_RANGE


def test_small_subject_survives_while_investigating() -> None:
    assert _worker("investigate")._filter_false_positives(
        COYOTE_AT_RANGE, CAM1_W, CAM1_H
    ) == COYOTE_AT_RANGE


def test_small_detection_still_filtered_during_patrol() -> None:
    assert _worker("patrol")._filter_false_positives(
        COYOTE_AT_RANGE, CAM1_W, CAM1_H
    ) == []


def test_small_detection_still_filtered_without_a_tracker() -> None:
    assert _worker(None)._filter_false_positives(
        COYOTE_AT_RANGE, CAM1_W, CAM1_H
    ) == []


def test_leaf_like_aspect_ratio_filtered_even_while_tracking() -> None:
    """Relaxing the area gate must not relax the branch/leaf shape gate."""
    branch = [Detection(species="animal", confidence=0.8,
                        bbox=[100.0, 200.0, 700.0, 250.0])]  # 12:1
    assert _worker("tracking")._filter_false_positives(branch, CAM1_W, CAM1_H) == []


# --------------------------------------------------------------------------
# Fix 2: the low-fill cap must not crush the axis the animal moves along
# --------------------------------------------------------------------------

def _tracker(**cfg):
    cfg.setdefault("update_interval", 0.0)
    cfg.setdefault("tracking_step_duration", 0.0)  # no timer threads in tests
    return create_ptz_tracker(_FakeOnvif(), "Profile_1", cfg)


class _FakeOnvif:
    def __init__(self):
        self.calls = []

    def ptz_get_position(self, token):
        return {"pan": 0.0, "tilt": 0.0, "zoom": 0.2, "available": True}

    def ptz_move(self, token, p, t, z):
        self.calls.append(("move", p, t, z))

    def ptz_stop(self, token):
        self.calls.append(("stop",))

    def get_presets(self, *a, **k):
        return []


def test_low_fill_cap_scales_on_offset_magnitude_not_per_axis() -> None:
    """Offset (-0.074, +0.479): pan must not be throttled 4x by its own axis.

    Observed in the incident: desired pan -0.111 was cut to -0.028 because
    |offset_x| alone drove the pan cap, even though the overall offset was
    far past low_fill_cap_full_offset.
    """
    t = _tracker(low_fill_velocity_cap=0.30, low_fill_cap_full_offset=0.40)
    pan, tilt, capped = t._apply_low_fill_cap(
        -0.111, 0.659, current_fill=0.188, offset_x=-0.074, offset_y=0.479,
    )
    assert capped
    # |offset| = 0.485 >= 0.40, so the full cap applies to BOTH axes.
    assert abs(pan) == 0.111, "pan was under the cap and must pass through"
    assert abs(tilt) == 0.30, "tilt should clamp to the full cap, not a fraction"


def test_low_fill_cap_still_tapers_near_centre() -> None:
    """The anti-overshoot taper must survive: small offset -> small cap."""
    t = _tracker(low_fill_velocity_cap=0.30, low_fill_cap_full_offset=0.40)
    pan, tilt, capped = t._apply_low_fill_cap(
        0.5, 0.5, current_fill=0.10, offset_x=0.04, offset_y=0.03,
    )
    assert capped
    expected = 0.30 * (0.05 / 0.40)  # |offset| = 0.05
    assert abs(pan - expected) < 1e-9
    assert abs(tilt - expected) < 1e-9


def test_low_fill_cap_inactive_for_large_subjects() -> None:
    t = _tracker(low_fill_velocity_cap=0.30)
    pan, tilt, capped = t._apply_low_fill_cap(
        0.8, -0.7, current_fill=0.46, offset_x=0.3, offset_y=-0.3,
    )
    assert not capped and (pan, tilt) == (0.8, -0.7)


# --------------------------------------------------------------------------
# Fix 4: sightings that don't drive a move still prove the target is there
# --------------------------------------------------------------------------

def _det(x1, y1, x2, y2, conf=0.9):
    return Detection(species="animal", confidence=conf, bbox=[x1, y1, x2, y2])


def test_rate_limited_sighting_still_refreshes_the_patrol_timer() -> None:
    """A detection dropped by update_interval must not count as 'lost'."""
    t = _tracker(update_interval=10.0, patrol_return_delay=5.0)
    t.set_track_enabled(True)
    t._mode = PTZMode.TRACKING
    t._last_detection_time = time.time() - 100.0  # stale "who drove" clock

    t.update([_det(900, 500, 1100, 700)], 1920, 1080)
    assert t._last_target_seen_time > 0.0

    t.update([], 1920, 1080)
    assert t._mode == PTZMode.TRACKING, "one empty tick must not abandon the subject"


def test_settle_blanked_camera_does_not_trigger_patrol_return() -> None:
    """cam1 keeps seeing the animal while cam2 is blanked by the settle gate.

    Before the fix _last_detection_time only advanced when a sighting drove a
    move, so blanking cam2 after every move ran the 2s patrol timer out on a
    target cam1 could still see.
    """
    t = _tracker(patrol_return_delay=5.0, update_interval=0.0)
    t.set_track_enabled(True)
    t._mode = PTZMode.TRACKING
    t._last_detection_time = time.time() - 100.0
    t._last_detection_source = "cam2"

    # cam1 sees it; cam2 is blanked (settle gate published an empty list).
    for _ in range(3):
        t.update_multi_camera(
            {"cam1": ([_det(400, 240, 460, 270)], 896, 512), "cam2": ([], 1920, 1080)},
            "cam1", "cam2", frame_capture_ts=time.time(),
        )
    assert t._mode == PTZMode.TRACKING

    # A single tick where neither camera reports must not abandon it either.
    t.update_multi_camera({}, "cam1", "cam2", frame_capture_ts=time.time())
    assert t._mode == PTZMode.TRACKING


def test_target_seen_clock_is_separate_from_handoff_clock() -> None:
    """cam1_fallback_delay must keep expiring; the two clocks must not merge."""
    t = _tracker(patrol_return_delay=5.0, cam1_fallback_delay=3.0)
    t.set_track_enabled(True)
    t._mode = PTZMode.TRACKING
    drove_at = time.time() - 10.0
    t._last_detection_time = drove_at
    t._last_detection_source = "cam2"

    t.update_multi_camera(
        {"cam1": ([_det(400, 240, 460, 270)], 896, 512)},
        "cam1", "cam2", frame_capture_ts=time.time(),
    )
    # Sighting clock advanced, but the handoff clock did not -- otherwise the
    # cam1 suppression window would never expire.
    assert t._last_target_seen_time > drove_at
    assert t._last_detection_time != t._last_target_seen_time or \
        t._last_detection_source == "cam1"


def test_genuinely_lost_target_still_returns_to_patrol() -> None:
    t = _tracker(patrol_return_delay=0.2, update_interval=0.0)
    t.set_track_enabled(True)
    t.set_patrol_enabled(True)
    t._mode = PTZMode.TRACKING
    t.update([_det(900, 500, 1100, 700)], 1920, 1080)

    time.sleep(0.35)
    t.update([], 1920, 1080)
    assert t._mode == PTZMode.PATROL, "a real loss must still fall back to patrol"
