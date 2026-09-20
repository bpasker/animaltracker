"""A continuous-sweep patrol starts sweeping again after a tracking episode.

Background (bug hunt, 2026-09-19): ``_do_patrol`` only re-issues its
ContinuousMove when the sweep velocity changes, and remembers the last one it
sent. Every way into PATROL forgot that value except the return from a lost
target. Tracking had moved the camera and stopped it, but the tracker still
believed the sweep from before the episode was running: it issued nothing, and
the camera sat still until the 90 s direction reversal. Patrol by presets was
not affected, because that path sends the camera to a preset.
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


def det(cx, cy, track_id):
    x1, y1 = (cx - 0.1) * W, (cy - 0.1) * H
    return Detection(species="animal", confidence=0.9,
                     bbox=[x1, y1, x1 + 0.2 * W, y1 + 0.2 * H], track_id=track_id)


def sweeping(**cfg):
    cfg.setdefault("update_interval", 0.0)
    cfg.setdefault("patrol_return_delay", 0.05)
    onvif = FakeOnvif()
    tracker = create_ptz_tracker(onvif, "Profile_1", cfg)
    tracker.set_patrol_enabled(True)
    tracker.set_track_enabled(True)
    tracker.update([], W, H)
    sweep = [c for c in onvif.calls if c[0] == "move"]
    assert len(sweep) == 1 and sweep[0][2:] == (0.0, 0.0)          # pan only: the sweep
    return tracker, onvif, sweep[0]


def track_then_lose(tracker) -> None:
    tracker.update([det(0.85, 0.5, track_id=3)], W, H)
    assert tracker.get_mode() == PTZMode.TRACKING.value
    time.sleep(0.1)
    tracker.update([], W, H)
    tracker.update([], W, H)
    assert tracker.get_mode() == PTZMode.PATROL.value


def test_the_sweep_is_issued_again_once_the_target_is_lost():
    tracker, onvif, sweep = sweeping(tracking_step_duration=0.0)
    track_then_lose(tracker)

    for _ in range(3):
        tracker.update([], W, H)

    assert onvif.calls[-1] == sweep                                  # it is what the camera is doing now
    assert onvif.calls.count(sweep) == 2                             # issued once more, not on every tick


def test_with_the_step_timer_stopping_the_camera_it_resumes_as_well():
    tracker, onvif, sweep = sweeping(tracking_step_duration=0.05)
    tracker.update([det(0.85, 0.5, track_id=3)], W, H)
    time.sleep(0.15)                                                 # the step timer has fired its Stop
    assert ("stop",) in onvif.calls
    tracker.update([], W, H)
    time.sleep(0.1)
    for _ in range(3):
        tracker.update([], W, H)

    assert tracker.get_mode() == PTZMode.PATROL.value
    assert onvif.calls[-1] == sweep


def test_a_stop_from_the_step_timer_does_not_strand_a_patrol_that_was_switched_on_meanwhile():
    tracker, onvif, sweep = sweeping(tracking_step_duration=0.0)
    tracker._patrol_velocity = sweep[1:]                             # the tracker believes it is sweeping

    tracker._stop_tracking_step_locked(time.time(), time.time(), time.time())

    assert tracker._patrol_velocity is None                          # the camera stopped: sweep again
