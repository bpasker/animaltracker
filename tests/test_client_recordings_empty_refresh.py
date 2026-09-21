"""A refresh that comes back empty clears the archive instead of lying.

``refreshGrid`` re-fetches the newest page and reconciles it with what is on
screen: clips the server no longer lists are pruned.  The prune was gated on
``incoming.length``, so a page with no clips at all skipped it -- delete the
last clip of a filtered view (or of the whole archive) and the cards stayed,
clickable, until a hard reload.  The count lied too: ``data.total || S.total``
discards a real zero and keeps the previous number.

An empty page is only uninformative when the server says there is more behind
it; with ``has_more`` false it is a complete answer and everything on screen is
stale.  That is the distinction this file pins.

Found in the bug hunt of 2026-09-19.
"""
from __future__ import annotations

import re
from pathlib import Path

RECORDINGS = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
              / "views" / "recordings.js").read_text()


def body_of(name: str) -> str:
    return re.search(r"function %s\(.*?\n\}\n" % name, RECORDINGS, re.S).group(0)


REFRESH = body_of("refreshGrid")


# --------------------------------------------------------------------------


def test_an_empty_but_complete_page_still_prunes():
    """The gate admits an empty page when nothing more is coming."""
    assert "(incoming.length || complete)" in REFRESH
    # ...and `complete` is has_more inverted, computed before the gate.
    assert REFRESH.index("var complete = !data.has_more;") < REFRESH.index("(incoming.length || complete)")


def test_an_empty_page_that_claims_more_is_left_alone():
    """Without `complete` the gate still needs clips to define a window."""
    gate = re.search(r"if \(S\.filters\.sort === 'newest' && \(([^)]*)\)\)", REFRESH).group(1)
    assert gate == "incoming.length || complete"


def test_the_prune_drops_everything_unseen_when_the_page_is_complete():
    filt = REFRESH[REFRESH.index("S.clips = S.clips.filter"):]
    filt = filt[:filt.index("});")]
    assert "if (seen[clipKey(c)]) return true;" in filt
    assert "if (complete) return false;" in filt
    # An incomplete page only vouches for its own window: older clips stay.
    assert "return typeof c.epoch === 'number' && c.epoch < oldest;" in filt


def test_only_the_newest_sort_prunes():
    """Any other order makes the first page an arbitrary slice, not a window."""
    assert "S.filters.sort === 'newest' && (" in REFRESH


def test_a_total_of_zero_is_believed():
    """`|| S.total` would keep the stale count on the last delete."""
    assert "S.total = typeof data.total === 'number' ? data.total : S.total;" in REFRESH
    assert "S.total = data.total || S.total;" not in REFRESH


def test_load_more_believes_a_zero_total_too():
    assert "S.total = typeof data.total === 'number' ? data.total : S.total;" in body_of("loadMore")


def test_the_archive_wide_count_is_believed_at_zero_too():
    """The empty state quotes it ("the archive holds N clips in total"), so a
    stale one told the operator to clear filters that were hiding nothing."""
    assert "data.archive_total || S.archiveTotal" not in REFRESH
    assert "typeof data.archive_total === 'number'" in REFRESH


def test_no_loader_keeps_a_stale_count():
    """The whole file: nothing falls back to the previous count on a zero."""
    assert "data.total || S.total" not in RECORDINGS
    assert "data.archive_total || S.archiveTotal" not in RECORDINGS


def test_a_prune_to_nothing_still_repaints_and_resets_the_offset():
    assert "if (fresh.length || pruned) {" in REFRESH
    assert "S.offset = S.clips.length;" in REFRESH
    assert REFRESH.index("var pruned = before - S.clips.length;") < REFRESH.index("renderGrid();")


def test_a_changed_archive_recounts_the_filter_chips():
    """The chips come from the unfiltered archive, which only loadGrid loads:
    a deleted species kept its chip, a new one had none until a reload."""
    changed = REFRESH[REFRESH.index("if (fresh.length || pruned) {"):]
    changed = changed[:changed.index("\n      }")]
    assert "loadUniverse();" in changed
