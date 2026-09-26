"""Full screen on the Live page works on an iPhone, checked in the source.

iPhone Safari has no Fullscreen API for anything but a <video>, so the Live
card's full-screen button found no requestFullscreen and posted an info toast
("This browser will not put the video full screen") on every tap; three taps
stacked three toasts over the page and the picture stayed where it was. The
stage is an <img> under the veil, the hatch and the age pill that keep a
frozen frame from reading as live, which a native video player would drop, so
full screen is now the stage pinned over the whole window (.is-full), which
the Fullscreen API then lifts onto the screen where it exists.

The repository has no JavaScript test runner; the behaviour was checked in a
browser against scripts/dev_web.py with requestFullscreen deleted (portrait,
landscape and simulated safe areas, the desktop focus layout, the API granted
and refused), and these assertions keep it from drifting.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
LIVE = (STATIC / "views" / "live.js").read_text()
CSS = (STATIC / "app.css").read_text()


def body_of(name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, LIVE, re.S).group(0)


def css_rule(selector: str) -> str:
    return re.search(re.escape(selector) + r"\{(.*?)\}", CSS, re.S).group(1)


def test_no_api_pins_the_stage_rather_than_saying_no():
    toggle = body_of("toggleFullscreen")
    assert "will not put the video full screen" not in LIVE
    assert "toast" not in toggle
    # Pinned before the API is even looked for, so a missing API (iPhone) and
    # a refused one (an iframe without allowfullscreen) both leave it pinned.
    assert toggle.index("setFull(card);") < toggle.index("requestFullscreen")
    assert "if (!req) return;" in toggle


def test_the_pinned_stage_has_a_way_out():
    # The head and its button are under the pinned stage; the exit rides on
    # the picture, and the stage must not capture its press as "centre here".
    card = body_of("buildCard")
    assert "card.exitBtn.addEventListener('pointerdown', function (ev) { ev.stopPropagation(); });" in card
    assert "card.exitBtn)" in card                       # inside .frame__tr

    stage = body_of("wireStage")
    assert "if (S.full === card && !S.fullApi) setFull(null);" in stage

    change = body_of("onFullscreenChange")
    assert "else if (!el && S.fullApi)" in change and "setFull(null);" in change


def test_a_tap_on_the_letterbox_does_not_aim_the_head():
    # stagePoint clamps to the picture, so a tap on a bar read as its edge.
    stage = body_of("wireStage")
    guard = stage.index("if (S.full === card && (ev.clientX < p.rect.left")
    assert guard < stage.index("card.dragStart = {")


def test_only_the_full_stage_holds_a_socket():
    assert "if (S.full) return S.full === card;" in body_of("cardWantsStream")
    assert "applyLayout();" in body_of("setFull")
    # ...so a pin the layout stops placing (a phone-width window widened into
    # desktop focus, which shows only the primary) must go, or nothing streams.
    layout = body_of("applyLayout")
    unpin = layout.index("if (S.full && order.indexOf(S.full) < 0) markFull(null);")
    assert unpin < layout.index("cardWantsStream(card)")


def test_the_picture_and_its_overlays_share_one_box():
    # The veil and hatch are what stop a frozen frame reading as live; they,
    # the frame and the crosshair layer all take the fitted box.
    shared = re.search(r"((?:\.cam__stage\.is-full[^{,]*,\s*)+\.cam__stage\.is-full[^{]*)\{(.*?)\}", CSS, re.S)
    selectors = shared.group(1)
    for part in (".frame", ".cam__veil", ".ptzstage__layer", "::after"):
        assert ".cam__stage.is-full" + (" " if not part.startswith("::") else "") + part in selectors
    assert "width:var(--full-w" in shared.group(2) and "height:var(--full-h" in shared.group(2)


def test_the_desktop_focus_stage_does_not_collapse():
    # .cam--primary .cam__stage centres itself with align-self, which engines
    # now apply to a fixed box too: pinned, it shrank to 0px tall.
    rule = css_rule(".cam__stage.is-full")
    assert "position:fixed" in rule and "align-self:stretch" in rule and "max-width:none" in rule
