"""The contract between the archive and the clip page, checked in the source.

The repository has no JavaScript test runner; the behaviour itself was checked
in a browser against scripts/dev_web.py. These assertions keep the three ends
of the contract from drifting apart again.

Background (bug hunt, 2026-09-19): the clip page was built to carry the
archive's filters along, so that Back returns to the filtered archive and
previous/next walk the set being reviewed. But the archive built every clip
link with no query at all, and the clip page kept ``camera`` where the archive
writes ``cameras`` and had no ``location``. Back always landed on the
unfiltered archive, previous/next walked every clip, and a clip older than the
newest 500 had both arrows disabled.
"""
from __future__ import annotations

import re
from pathlib import Path

VIEWS = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "views"
RECORDINGS = (VIEWS / "recordings.js").read_text()
DETAIL = (VIEWS / "detail.js").read_text()


def keys_written_by(function_name: str, source: str) -> set:
    body = re.search(r"function %s\(.*?\n}\n" % function_name, source, re.S).group(0)
    return set(re.findall(r"\bout\.([a-z]+) =", body))


def test_every_clip_link_carries_the_archive_s_place():
    assert "return router.href('/clips/' + api.encodePath(path), archiveQuery());" in RECORDINGS
    assert "router.go('/clips/' + api.encodePath(clip.path), archiveQuery());" in RECORDINGS
    assert "router.go('/clips/' + api.encodePath(clip.path), {})" not in RECORDINGS


def test_the_clip_page_keeps_every_key_the_archive_writes():
    written = keys_written_by("archiveQuery", RECORDINGS)
    assert written == {"cameras", "location", "species", "from", "to", "q", "sort",
                       "view", "year", "month", "date"}
    kept = set(re.findall(r"'([a-z]+)'", re.search(r"var FILTER_KEYS = \[(.*?)\];", DETAIL, re.S).group(1)))
    assert written <= kept


def test_the_neighbour_lookup_speaks_the_list_api_s_dialect():
    body = re.search(r"function neighborsQuery\(.*?\n}\n", DETAIL, re.S).group(0)
    assert "query.cameras || query.camera" in body and "q.camera = cameras" in body
    for key in ("location", "species", "from", "to", "q", "sort"):
        assert f"'{key}'" in body
