"""A fourth toast never fires an undo deadline early.

``core/toast.js`` shows at most three toasts. When a fourth arrived the loop
dismissed ``live[0]`` with ``fireExpire`` set, which ran the oldest undo
toast's ``onExpire`` on the spot: delete four clips quickly and the first
was gone with its Undo still promised (bug hunt item 3.30). The comment
above the loop said the opposite of what the loop did.

Now a toast holding a deadline keeps its place; the oldest one without one
makes room, and if every visible toast is waiting on the user the strip
grows. Verified in a browser against scripts/dev_web.py: three undo toasts
plus a plain fourth fired nothing and kept all three Undos; four plain
toasts evicted the first.
"""
from __future__ import annotations

import re
from pathlib import Path

TOAST = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "core" / "toast.js").read_text()


def eviction_loop() -> str:
    start = TOAST.index("while (live.length > MAX_VISIBLE) {")
    return TOAST[start:TOAST.index("\n  }\n", start)]


def test_the_loop_never_dismisses_with_fire_expire():
    loop = eviction_loop()
    assert "dismiss(evict, false);" in loop
    assert "dismiss(oldest, true)" not in TOAST


def test_a_toast_holding_a_deadline_is_never_the_one_evicted():
    loop = eviction_loop()
    assert "if (!live[k].onExpire) { evict = live[k]; break; }" in loop


def test_the_newest_toast_is_never_evicted():
    assert "for (var k = 0; k < live.length - 1; k++)" in eviction_loop()


def test_when_every_toast_holds_a_deadline_the_strip_grows():
    assert "if (!evict) break;" in eviction_loop()


def test_on_expire_fires_only_from_the_timer_and_from_flush():
    """dismiss(..., true) is reached when a toast's own timer runs out and
    from toast.flush() on pagehide, so a delete is not lost with the tab.
    Nothing else, the eviction loop included, passes true."""
    calls = re.findall(r"dismiss\(([^,]+),\s*(true|false|!!fireExpire)\)", TOAST)
    fire_true = [c[0] for c in calls if c[1] == "true"]
    assert fire_true == ["entry", "pending[i]"], fire_true
