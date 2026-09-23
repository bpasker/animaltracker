"""A delete in its undo window follows the clip if it is renamed meanwhile.

Background (bug hunt, 2026-09-22): the 6 s undo window is long enough for an
"Analyzing…" clip to finish and be renamed to its species. A refresh in that
window moved the card's pending-delete mark to the new name, but the delete
still held the old one: the server answered "File not found", the client
counted that as deleted, and the clip stayed on disk under its new name with
its card hidden until a reload.

Checked in a browser against scripts/dev_web.py: select a clip, Delete,
rename it on disk, refresh inside the window; the renamed file is removed.
There is no JavaScript test runner for the views, so these assertions keep
the pieces from drifting apart.
"""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
          / "views" / "recordings.js").read_text()


def body_of(name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, SOURCE, re.S).group(0)


def test_a_rename_updates_every_pending_delete():
    follow = body_of("followRename")
    assert "S.deleteBatches.forEach(" in follow
    assert "batch.paths[i] = newPath" in follow


def test_the_delete_sends_the_names_current_when_the_window_closes():
    select = body_of("deleteSelection")
    assert "S.deleteBatches.add(batch)" in select
    assert "commitDelete(batch, restoreTotal)" in select
    commit = body_of("commitDelete")
    assert commit.index("var paths = batch.paths.slice();") < commit.index("api.bulkDelete(paths")
    # Finished either way: the batch no longer follows renames.
    assert commit.count("S.deleteBatches.delete(batch)") == 2
