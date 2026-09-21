"""Leaving the settings page with unsaved edits, checked in the source.

The behaviour was checked in a browser against scripts/dev_web.py: an in-app
Back press with one unsaved change restores the URL and opens the dialog;
"Stay here" keeps the draft, "Discard and leave" goes, and "Save and leave"
posts the config and then goes. There is no JavaScript test runner, so these
assertions keep the pieces from drifting apart.

Background (bug hunt, 2026-09-19): the guard covered page unloads
(beforeunload) and in-app anchors (a capture-phase click handler), but Back
and Forward reach neither. A Back press or a trackpad swipe discarded the
whole draft in silence. And the dialog's "Save and leave" called save() and
ignored it, so it saved and stayed: the click looked ignored.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
SETTINGS = (STATIC / "views" / "settings.js").read_text()
ROUTER = (STATIC / "core" / "router.js").read_text()


def body_of(source: str, name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, source, re.S).group(0)


def test_the_router_can_be_vetoed_and_puts_the_url_back():
    start = re.search(r"start: function \(mountRoot\) \{.*?\n    \},", ROUTER, re.S).group(0)
    assert "navGuard(" in start
    # popstate cannot be cancelled, so the guard's "no" is honoured by pushing
    # the previous URL back before resolve() would have mounted anything.
    assert "window.history.pushState(null, '', from);" in start
    assert start.index("pushState(null, '', from)") < start.index("resolve();")
    assert "guard: function (fn)" in ROUTER


def test_a_guard_never_outlives_the_view_that_registered_it():
    resolve = re.search(r"function resolve\(\) \{.*?\n  \}\n", ROUTER, re.S).group(0)
    unmount_block = resolve[resolve.index("if (mounted) {"):]
    assert unmount_block.index("navGuard = null;") < unmount_block.index("mounted.view.unmount()")


def test_settings_guards_back_with_the_same_dialog_as_a_link_click():
    guards = body_of(SETTINGS, "installGuards")
    assert "router.guard(function (to)" in guards
    assert guards.count("confirmLeave(") == 2          # the click handler and the guard
    assert "track(router.guard(" in guards             # dropped when the view unmounts
    # The dialog itself is built once, in the shared function.
    assert SETTINGS.count("'Save and leave'") == 1


def test_save_and_leave_leaves_only_once_the_write_succeeded():
    leave = body_of(SETTINGS, "confirmLeave")
    # The session check is null-safe: unmount sets S to null, and this
    # deliberately navigates away while the save is in flight.
    assert re.search(r"save\(\)\.then\(function \(saved\) \{\s*if \(saved && S && !S\.destroyed\) proceed\(\);", leave)

    save = body_of(SETTINGS, "save")
    assert "return api.saveConfig(" in save            # the caller can wait for it
    # already saving / no draft, validation, payload: nothing was written
    assert save.count("return Promise.resolve(false)") == 3
    assert "return Promise.resolve(true)" in save             # nothing to save: safe to leave
    assert "return true;" in save and save.count("return false;") >= 2
