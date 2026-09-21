"""The archive checks its session before anything late touches it.

``unmount`` sets the view's session to null, so any callback that can run
afterwards must ask first: reading ``S.anything`` there throws instead of
returning quietly. Verified in a browser against scripts/dev_web.py at phone
width — open Filters, then leave the route: unmount closes the sheet and the
close animation outlives the session. Before this, that raised
"Cannot read properties of null (reading 'sheets')"; after it, nothing.

Background (bug hunt, 2026-09-19): the day sheet guarded its onClose and the
filter sheet did not, a bulk reanalysis could run for minutes and then call a
timer helper that dereferences the session, and the list, calendar, month and
day loaders all painted straight into it.
"""
from __future__ import annotations

import re
from pathlib import Path

RECORDINGS = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
              / "views" / "recordings.js").read_text()


def body_of(name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, RECORDINGS, re.S).group(0)


def test_there_is_one_session_check_and_it_covers_the_unmount_window():
    alive = body_of("alive")
    assert "return !!S && !S.unmounting;" in alive


def test_the_timer_helper_refuses_to_schedule_for_a_gone_screen():
    later = body_of("later")
    assert later.index("if (!S) return null;") < later.index("window.setTimeout")
    assert "if (!S) return;" in later          # ...and the callback checks again when it fires


def test_every_loader_that_paints_asks_first():
    for name in ("loadGrid", "loadMore", "refreshGrid", "loadMonth"):
        assert "if (!alive()) return;" in body_of(name), name


def test_both_sheets_survive_being_closed_by_unmount():
    closers = re.findall(r"onClose: function \(\) \{(.*?)\n    \}", RECORDINGS, re.S)
    assert len(closers) == 2
    for body in closers:
        assert "if (!S" in body


def test_a_bulk_reanalysis_that_outlives_the_screen_stops_painting_but_keeps_going():
    step = RECORDINGS[RECORDINGS.index("  function step(i) {"):]
    step = step[:step.index("\n  }\n")]
    assert "if (alive()) later(function () { refreshGrid(); }, 1500);" in step
    # The clips the operator asked for are still queued: the chain itself is
    # not cancelled, only the repaint is skipped.
    assert "step(i + 1);" in step and "if (!alive()) return;" not in step


def test_no_api_handler_paints_without_asking():
    for handler in re.finditer(r"\.(?:then|catch)\(function \((?:data|err|res)\) \{\n(\s*)(.+)", RECORDINGS):
        first = handler.group(2).strip()
        assert first.startswith(("if (!alive()", "if (!S", "/*")), first
