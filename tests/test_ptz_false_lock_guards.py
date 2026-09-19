"""The two guards against locking the PTZ on to something that is not an animal.

Background (bug hunt, 2026-09-19), both reproduced against the real tracker:

* The strict confidence for a first lock on a detection ByteTrack has not
  confirmed (0.75, against 0.60 while re-acquiring) was chosen by testing
  ``_mode``. Every update path switches the mode to TRACKING *before* it
  selects a target, so the mode never said "cold": from patrol, an
  unconfirmed 65% flash switched to TRACKING, locked and moved the camera.
  The unit test for the strict bar passed only because it set ``_mode`` by
  hand.
* The static-target watchdog released a lock that had not moved for 45 s and
  then fell straight through to "no lock: pick the best detection", which is
  the same stationary blob. It was re-locked in the same call with a fresh
  motion anchor, every 45 s, for ever, and the camera never went back to
  patrol: the hang the watchdog was written for.
"""
from __future__ import annotations

import time

from animaltracker.detector import Detection
from animaltracker.ptz_tracker import PTZMode, create_ptz_tracker

W, H = 1920, 1080


class FakeOnvif:
    def __init__(self) -> None:
        self.calls: list = []

    def ptz_get_position(self, token):
        return {"pan": 0.0, "tilt": 0.0, "zoom": 0.2, "available": True}

    def ptz_move(self, token, p, t, z):
        self.calls.append(("move", round(p, 3), round(t, 3), round(z, 3)))

    def ptz_stop(self, token):
        self.calls.append(("stop",))

    def ptz_goto_preset(self, *a, **k):
        self.calls.append(("goto",))

    def get_presets(self, *a, **k):
        return []

    def moves(self):
        return [c for c in self.calls if c[0] == "move"]


def det(cx, cy, w=0.2, h=0.2, conf=0.9, track_id=None):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    return Detection(species="animal", confidence=conf,
                     bbox=[x1, y1, x1 + w * W, y1 + h * H], track_id=track_id)


def patrolling(**cfg):
    cfg.setdefault("update_interval", 0.0)
    cfg.setdefault("tracking_step_duration", 0.0)
    onvif = FakeOnvif()
    tracker = create_ptz_tracker(onvif, "Profile_1", cfg)
    tracker.set_patrol_enabled(True)
    tracker.set_track_enabled(True)
    tracker.update([], W, H)                         # a patrol tick
    assert tracker.get_mode() == PTZMode.PATROL.value
    return tracker, onvif


def tracking_moves(onvif: FakeOnvif, patrol_speed: float) -> list:
    return [m for m in onvif.moves() if abs(abs(m[1]) - patrol_speed) > 1e-6 or m[2] or m[3]]


# --- the strict bar for a first lock -------------------------------------------------

def test_an_unconfirmed_flash_seen_from_patrol_does_not_lock_or_move_the_camera():
    tracker, onvif = patrolling()

    for _ in range(3):                               # and not on the second look either
        tracker.update([det(0.8, 0.5, conf=0.65)], W, H)

    assert tracker._locked_bbox_center is None and tracker._locked_track_id is None
    assert tracking_moves(onvif, tracker.patrol_speed) == []


def test_a_confirmed_track_locks_from_patrol_whatever_its_confidence():
    tracker, onvif = patrolling()

    tracker.update([det(0.8, 0.5, conf=0.5, track_id=3)], W, H)

    assert tracker._locked_track_id == 3
    assert tracking_moves(onvif, tracker.patrol_speed)


def test_a_strong_unconfirmed_detection_still_locks_from_patrol():
    tracker, _ = patrolling()

    tracker.update([det(0.8, 0.5, conf=0.80)], W, H)

    assert tracker._locked_bbox_center is not None


def test_once_something_has_locked_the_lower_bar_re_acquires_it():
    tracker, _ = patrolling()
    tracker.update([det(0.8, 0.5, track_id=3)], W, H)             # the episode's first lock
    tracker._reset_lock_state_locked()                              # lost after its misses

    tracker.update([det(0.3, 0.3, conf=0.65)], W, H)                # ByteTrack ids do not survive the slew

    assert tracker._locked_bbox_center is not None


def test_the_next_episode_starts_strict_again():
    tracker, _ = patrolling(patrol_return_delay=0.05)
    tracker.update([det(0.8, 0.5, track_id=3)], W, H)
    time.sleep(0.1)
    tracker.update([], W, H)
    tracker.update([], W, H)
    assert tracker.get_mode() == PTZMode.PATROL.value

    tracker.update([det(0.3, 0.3, conf=0.65)], W, H)

    assert tracker._locked_bbox_center is None


# --- the static-target watchdog ----------------------------------------------------

def stuck_on(tracker, where, track_id=7):
    tracker.update([det(*where, track_id=track_id)], W, H)
    assert tracker._locked_track_id == track_id
    tracker._lock_motion_anchor_time -= tracker._lock_static_release_sec + 1     # it has not moved since


def test_a_released_static_target_is_not_locked_again_and_patrol_resumes():
    tracker, onvif = patrolling(patrol_return_delay=0.05)
    blob = (0.7, 0.65)
    stuck_on(tracker, blob)

    tracker.update([det(*blob, track_id=7)], W, H)                  # the watchdog fires

    assert tracker._locked_track_id is None, "released and re-locked in the same call"
    assert [d["event"] for d in tracker.get_decision_log()].count("static_target_released") == 1

    time.sleep(0.1)
    for _ in range(3):                                              # still detected, every tick
        tracker.update([det(*blob, track_id=7)], W, H)

    assert tracker._locked_track_id is None
    assert tracker.get_mode() == PTZMode.PATROL.value               # it no longer counts as a sighting


def test_an_animal_elsewhere_in_the_frame_is_tracked_as_usual():
    tracker, _ = patrolling()
    stuck_on(tracker, (0.7, 0.65))
    tracker.update([det(0.7, 0.65, track_id=7)], W, H)

    tracker.update([det(0.7, 0.65, track_id=7), det(0.2, 0.3, track_id=9)], W, H)

    assert tracker._locked_track_id == 9


def test_an_animal_that_was_only_resting_is_lockable_once_it_moves_off_the_spot():
    tracker, _ = patrolling()
    stuck_on(tracker, (0.7, 0.65))
    tracker.update([det(0.7, 0.65, track_id=7)], W, H)

    tracker.update([det(0.58, 0.65, track_id=7)], W, H)             # more than the reject radius away

    assert tracker._locked_track_id == 7


def test_a_locked_animal_walking_across_the_spot_keeps_its_lock():
    tracker, _ = patrolling()
    stuck_on(tracker, (0.7, 0.65))
    tracker.update([det(0.7, 0.65, track_id=7)], W, H)
    tracker.update([det(0.2, 0.3, track_id=9)], W, H)
    assert tracker._locked_track_id == 9

    tracker.update([det(0.71, 0.66, track_id=9)], W, H)             # right over the released blob

    assert tracker._locked_track_id == 9


def test_the_spot_is_forgotten_after_a_while_and_belongs_to_one_camera():
    tracker, _ = patrolling()
    stuck_on(tracker, (0.7, 0.65))
    tracker.update([det(0.7, 0.65, track_id=7)], W, H)
    now = time.time()
    blob = [det(0.7, 0.65, track_id=7)]

    assert tracker._drop_static_rejects(blob, W, H, "single", now) == []
    assert tracker._drop_static_rejects(blob, W, H, "cam2", now) == blob             # another camera's frame
    assert tracker._drop_static_rejects(blob, W, H, "single", now + tracker._static_reject_sec + 1) == blob
    assert tracker._static_rejects == []
