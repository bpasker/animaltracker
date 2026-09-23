"""The router asks a view's guard on every way out of it, and only then.

Background (bug hunt, 2026-09-22): the settings page's unsaved-edits guard
was asked on link clicks (its own click listener) and on Back/Forward (the
router's popstate). ``router.navigate`` / ``go`` never asked it, so the
``g l`` shortcut, the command palette, the search box and a toast's "View"
left at once and threw the draft away. And popstate asked on every event,
including the one a fragment link (the shell's skip link, ``#main``) fires
with the URL unchanged: "Discard and leave" then silenced the guard and
navigated to the page it was already on, so the discarded value stayed in
the form, uncounted, and went out with the next Save.

These drive the real module under Node with a stub window and history.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROUTER = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "core" / "router.js"

HARNESS = r"""
const listeners = {};
const loc = new URL('http://app.test/app/settings');
const history = [];
globalThis.window = {
  get location() { return { pathname: loc.pathname, search: loc.search, href: loc.href }; },
  history: {
    pushState(state, title, url) { const u = new URL(url, loc.href); loc.pathname = u.pathname; loc.search = u.search; loc.hash = u.hash; history.push(u.pathname + u.search); },
    replaceState(state, title, url) { const u = new URL(url, loc.href); loc.pathname = u.pathname; loc.search = u.search; },
    back() {},
  },
  addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
  console,
};
function popTo(url) {           // what the browser does: move the URL, then fire popstate
  const u = new URL(url, loc.href); loc.pathname = u.pathname; loc.search = u.search;
  for (const fn of listeners.popstate || []) fn();
}
const { router } = await import(process.argv[2]);
const log = [];
const view = name => ({
  mount() { log.push('mount ' + name); },
  unmount() { log.push('unmount ' + name); },
  update() { log.push('update ' + name); },
});
router.register('/settings', view('settings')).register('/live', view('live'));
router.start({ firstChild: null });

let asked = [];
let answer = false;
router.guard((to, from) => { asked.push(to); return answer; });

const steps = {};
router.setQuery({ section: 'cam1' });                     // a section: same view, update()
steps.section = { asked: asked.slice(), url: loc.pathname + loc.search };

popTo('/app/settings?section=cam1');                      // skip link: popstate, URL unchanged
steps.fragment = { asked: asked.slice() };

router.go('/live');                                       // a shortcut / palette / search
steps.go = { asked: asked.slice(), url: loc.pathname + loc.search };

popTo('/app/live');                                       // Back to another view
steps.back = { asked: asked.slice(), url: loc.pathname + loc.search };

answer = true;
router.go('/live');
steps.allowed = { url: loc.pathname + loc.search, log: log.slice() };
console.log(JSON.stringify(steps));
"""


@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path_factory.mktemp("router") / "harness.mjs"
    script.write_text(HARNESS)
    out = subprocess.run([node, str(script), ROUTER.as_uri()], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_query_change_the_view_handles_itself_does_not_ask(steps):
    assert steps["section"] == {"asked": [], "url": "/app/settings?section=cam1"}


def test_a_fragment_links_popstate_does_not_ask(steps):
    assert steps["fragment"]["asked"] == []


def test_navigate_asks_before_it_leaves_and_honours_no(steps):
    assert steps["go"] == {"asked": ["/app/live"], "url": "/app/settings?section=cam1"}


def test_back_asks_and_puts_the_url_back(steps):
    assert steps["back"]["asked"] == ["/app/live", "/app/live"]
    assert steps["back"]["url"] == "/app/settings?section=cam1"


def test_a_yes_leaves(steps):
    assert steps["allowed"]["url"] == "/app/live"
    assert steps["allowed"]["log"][-2:] == ["unmount settings", "mount live"]
