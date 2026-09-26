"""Every toast in the web UI follows one house style, checked in the source.

A review of all 96 toast call sites (2026-09-25) found them drifting apart:
the trash-can `danger` kind reporting failed fetches and a missing preset
name, the same "needs a name" slip raised as an error in one dialog, danger
in another and info in a third, headlines with and without a closing full
stop, "analysed" beside "Reanalyze", "recording" on one page and "clip"
everywhere else, failures phrased four different ways, a bulk reanalysis
announced as "queued" when the server had already finished it, and error
toasts about a poll that stayed up, stale, after the poll recovered. The
style they now share is written at the top of core/toast.js; these tests
hold the call sites to it. The repository has no JavaScript test runner;
the behaviour was checked in a browser against scripts/dev_web.py.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
SOURCES = {p.relative_to(STATIC).as_posix(): p.read_text() for p in
           [STATIC / "app.js", *sorted((STATIC / "views").glob("*.js"))]}
TOAST = (STATIC / "core" / "toast.js").read_text()

CALL = re.compile(r"\b(toast(?:\.(?:success|info|danger|error|progress))?|shout)\(")


def first_argument(src: str, start: int) -> str:
    """The source text of a call's first argument: up to the top-level comma
    or the closing parenthesis, skipping over strings and nested brackets."""
    depth, i, quote = 0, start, None
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return src[start:i]
            depth -= 1
        elif c == "," and depth == 0:
            return src[start:i]
        i += 1
    raise AssertionError("unterminated call")


def calls():
    """(file, line, callee, title source, the call's text up to 400 chars)."""
    for name, src in SOURCES.items():
        for m in CALL.finditer(src):
            if src[max(0, m.start() - 9):m.start()] == "function ":
                continue
            arg_start = m.end()
            if m.group(1) == "shout":        # shout(key, message, err, opts)
                arg_start += len(first_argument(src, arg_start)) + 1
            title = first_argument(src, arg_start).strip()
            line = src.count("\n", 0, m.start()) + 1
            yield name, line, m.group(1), title, src[m.start():m.start() + 400]


def literals(title: str) -> list[str]:
    return re.findall(r"'((?:[^'\\]|\\.)*)'", title)


def test_the_review_found_every_call_site():
    assert len(list(calls())) >= 90


def test_a_headline_never_ends_in_a_full_stop():
    stops = [(f, n, t) for f, n, _, t, _ in calls()
             if literals(t) and re.search(r"\.'\s*$", t) and not t.endswith("…'")]
    assert stops == []


def test_danger_is_only_a_deletion_with_an_undo():
    # Its icon is a trash can; a failed fetch or a missing name wore it.
    for name, line, callee, _, text in calls():
        if callee == "toast.danger":
            raise AssertionError(f"{name}:{line} uses toast.danger without an Undo")
        if callee == "toast" and "kind: 'danger'" in text:
            assert "undo: {" in text, f"{name}:{line}"


def test_one_spelling_and_one_noun():
    for name, line, _, title, _ in calls():
        for lit in literals(title):
            # The verb: "analysis" is the noun on both sides of the Atlantic.
            assert not re.search(r"analys(e|ed|es|ing)\b", lit, re.I), \
                f"{name}:{line} {lit!r}: the UI says analyze"
            assert "recording" not in lit.lower(), f"{name}:{line} {lit!r}: an item is a clip"


def test_failures_say_could_not():
    """An error headline is "Could not ..." unless it reports a state
    (disconnected, critical, not back) rather than an action that failed."""
    states = ("Disconnected", " is critical", "has not come back")
    for name, line, callee, title, _ in calls():
        if callee not in ("toast.error", "shout"):
            continue
        lits = literals(title)
        if not lits or title.startswith(("NOT_SAVED", "where")):
            continue
        head = lits[0]
        if any(s in "".join(lits) for s in states):
            continue
        assert head.startswith("Could not"), f"{name}:{line} {head!r}"


def test_the_house_style_is_written_down():
    for rule in ("THE HOUSE STYLE", "danger    a deletion, and only with an Undo",
                 "no closing full stop", "re-arms that one"):
        assert rule in TOAST


def test_an_identical_plain_toast_re_arms_instead_of_stacking():
    body = TOAST[TOAST.index("export function toast("):]
    guard = body.index("var plain = !o.undo && !o.action && !o.retry && kind !== 'progress';")
    assert body.index("rearm(twin, timeout);") > guard
    assert body.index("return twin.handle;") < body.index("var el = h('div.toast'")


def test_poll_toasts_come_down_when_the_poll_recovers_and_when_the_page_goes():
    monitor = SOURCES["views/monitor.js"]
    assert monitor.count("closePollToast('monitor');") == 2       # recovery + teardown
    assert monitor.count("closePollToast('logs');") == 2
    assert "Monitor reconnected" not in monitor                   # the stale error stayed beside it
    live = SOURCES["views/live.js"]
    for key in ("'server'", "'cameras'", "'monitor'"):
        assert f"hush({key});" in live
    assert "Object.keys(shouted).forEach(hush);" in live


def test_background_polls_leave_an_outage_to_the_disconnected_toast():
    assert "if (api.isDisconnected(err)) return;" in SOURCES["views/monitor.js"]
    assert "if (api.isDisconnected(err)) return;" in SOURCES["views/recordings.js"]
    assert "!api.isDisconnected(err)" in SOURCES["views/settings.js"]


def test_no_progress_toast_is_left_spinning_by_a_page_that_went_away():
    detail = SOURCES["views/detail.js"]
    unmount = detail[detail.index("unmount: function"):]
    assert "S.job.toast.close();" in unmount
    settings = SOURCES["views/settings.js"]
    wait = settings[settings.index("function waitForServer("):settings.index("YAML PREVIEW")]
    assert "S.abort.signal" not in wait and "if (!S || S.destroyed) return;" not in wait


def test_a_deliberate_restart_is_not_reported_as_a_disconnection():
    app = SOURCES["app.js"]
    assert "store.select(['connected', 'restarting']" in app
    assert "var down = !s.connected && !s.restarting;" in app
    assert "store.set({ restarting: true });" in SOURCES["views/settings.js"]


def test_bulk_reanalysis_says_what_the_server_did():
    rec = SOURCES["views/recordings.js"]
    assert "' queued for SpeciesNet'" not in rec and "'The rest were queued." not in rec
    assert "' reanalyzed');" in rec


def test_copy_works_over_plain_http_everywhere():
    # The app is served over http on the LAN, where navigator.clipboard does
    # not exist; the clip page's own copy had no fallback and always failed.
    dom = (STATIC / "core" / "dom.js").read_text()
    assert "export function copyText(text)" in dom and "execCommand('copy')" in dom
    for name, src in SOURCES.items():
        assert "navigator.clipboard" not in src, name


def test_flush_fires_deadlines_and_leaves_every_other_toast_up():
    # Recordings flushes on unmount to commit a pending delete; flush closed
    # every toast, "Disconnected" and a restart's progress included.
    flush = TOAST[TOAST.index("toast.flush = function"):]
    assert "live.filter(function (entry) { return !!entry.onExpire; })" in flush[:300]


def test_a_retry_that_needs_its_page_goes_down_with_the_page():
    # Left up, such a Retry ran against a session that was gone and threw.
    detail = SOURCES["views/detail.js"]
    for title in ("Could not play this clip", "Could not load this clip",
                  "Could not read the processing log"):
        at = detail.index(f"toast.error('{title}'")
        assert detail[:at].rstrip().endswith("closeWithPage("), title
    rec = SOURCES["views/recordings.js"]
    report = rec[rec.index("function reportError("):]
    assert "if (S) track(function () { t.close(); });" in report[:500]
    # ...but the delete's Retry needs no screen and the clips are still on disk.
    commit = rec[rec.index("function commitDelete("):rec.index("function reanalyzeSelection(")]
    assert "reportError(" not in commit and "commitDelete(batch, restoreTotal); }" in commit
    assert "if (S.saveToast) S.saveToast.close();" in SOURCES["views/settings.js"]
    assert "if (S) reg(function () { et.close(); });" in SOURCES["views/live.js"]
