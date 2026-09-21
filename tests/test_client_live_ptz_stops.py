"""Nothing on the Live page can leave the PTZ head slewing, checked in the source.

live.js says at the top what it exists for: "a lost PTZ stop leaves a camera
slewing until the server's 10s dead-man fires. Every path that can end motion
... routes through stopAllMotion()". The repository has no JavaScript test
runner, so these assertions keep that invariant from drifting; the behaviour
itself was checked in a browser against scripts/dev_web.py.

Background (bug hunt, 2026-09-19), both confirmed there against the previous
commit: a pulse (click-to-centre, frame-a-box, a tap) is a move plus a stop on
a timer, and unmount cancelled that timer without sending the stop, so leaving
the page within the pulse window left the head slewing (old code: "move" and
no "stop"). And the stage's own guard tested for a card state of 'offline' or
'no-route', but after a drop the state is 'reconnecting' and 'no-route' is
assigned nowhere, so clicking the frozen frame still drove the head with no
picture to aim by (old code, card reading "Reconnecting - attempt 2": a move
went out).
"""
from __future__ import annotations

import re
from pathlib import Path

LIVE = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "views" / "live.js").read_text()


def body_of(name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, LIVE, re.S).group(0)


def test_a_pulse_that_is_cut_short_still_sends_its_stop():
    flush = body_of("flushPulse")
    assert "cancel(card.pulseTimer)" in flush and "sendStop(card)" in flush

    both = body_of("stopAllMotion")
    assert "stopJog()" in both and "flushPulse(card)" in both


def test_every_way_out_of_the_page_routes_through_it():
    for event in ("'blur'", "'pagehide'"):
        assert re.search(r"on\(window, %s, function \(\) \{ stopAllMotion\(\); \}\)" % event, LIVE)
    assert "if (ev.key === 'Escape') stopAllMotion();" in LIVE
    assert "stopAllMotion();" in body_of("syncVisibility")

    unmount = re.search(r"unmount: function \(\) \{.*?\n  \}", LIVE, re.S).group(0)
    assert "stopAllMotion();" in unmount
    # ...and never the bare cancel that dropped the stop on the floor.
    assert "cancel(card.pulseTimer)" not in unmount


def test_a_pulse_obeys_the_same_lock_out_as_a_held_jog():
    assert "if (card.controlDown) return;" in body_of("pulse")
    assert "if (card.controlDown) return;" in body_of("startJog")


def test_the_stage_obeys_the_flag_that_hides_the_controls():
    stage = body_of("wireStage")
    assert "if (card.controlDown) return;" in stage
    # 'no-route' is a CSS class, never a card.state: testing for it guarded nothing.
    assert "card.state === 'no-route'" not in LIVE
    assert "card.state === 'offline'" not in stage
