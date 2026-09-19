"""Every endpoint that takes a clip path keeps it inside the clips directory.

Background (bug hunt, 2026-09-19): only the delete endpoint resolved the path
and required it to stay under ``clips/``. Reanalyse and the processing-log
endpoint tested for the substring ``'..'``, and clip detail tested nothing.
Joining an absolute path onto a directory yields the absolute path, so

* ``POST /recordings/reprocess {"path": "/home/user/video.mp4"}`` ran
  SpeciesNet on any file on the host, renamed it in place and wrote key
  frames and a log beside it;
* ``GET /recordings/log//any/where/x.mp4`` returned the whole of
  ``/any/where/x.log.json``;
* ``GET /api/clip/../x.mp4`` reported the size and mtime of files anywhere.

The substring test also refused a legitimate clip whose name merely contains
two dots. All of them now share ``WebServer._confined_clip_path``.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from animaltracker.analysis_recovery import ClipAnalysisRegistry
from animaltracker.web import WebServer

CLIP = "cam1/2026/09/10/1789000000_animal.mp4"


class JsonRequest:
    def __init__(self, body=None, match_info=None):
        self._body = body
        self.match_info = match_info or {}
        self.query = {}

    async def json(self):
        return self._body


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "storage"
    (root / "clips" / "cam1" / "2026" / "09" / "10").mkdir(parents=True)
    (root / "clips" / CLIP).write_bytes(b"a clip")
    return WebServer({}, root, tmp_path / "logs", port=0, analysis_registry=ClipAnalysisRegistry())


@pytest.fixture
def outside(tmp_path):
    """A video with a log beside it, next to the storage root, not in it."""
    clip = tmp_path / "outside.mp4"
    clip.write_bytes(b"somebody else's video")
    clip.with_suffix(".log.json").write_text(json.dumps({"secret": "not yours"}))
    return clip


def escapes(outside):
    return [
        str(outside),                          # absolute: the join yields it unchanged
        "../../outside.mp4",                   # climbs out of clips/ and storage/
        "cam1/../../../outside.mp4",
        "cam1/2026/../../../../outside.mp4",
    ]


# --- the helper ------------------------------------------------------------------

def test_paths_inside_the_archive_come_back_normalised(server):
    clips = server.storage_root / "clips"

    assert server._confined_clip_path(CLIP) == clips / CLIP
    assert server._confined_clip_path("cam1/../cam1/2026/09/10/1789000000_animal.mp4") == clips / CLIP
    assert server._confined_clip_path("cam1/2026/09/10/a..b.mp4") == clips / "cam1/2026/09/10/a..b.mp4"
    assert server._confined_clip_path("cam1/2026/09/10/not_there_yet.mp4") is not None


def test_paths_that_leave_the_archive_are_refused(server, outside):
    for rel_path in escapes(outside) + ["/", "..", "", None, 7, "cam1/\x00.mp4"]:
        assert server._confined_clip_path(rel_path) is None, rel_path


def test_a_symlink_that_points_out_of_the_archive_is_refused(server, outside):
    link = server.storage_root / "clips" / "cam1" / "link.mp4"
    os.symlink(outside, link)

    assert server._confined_clip_path("cam1/link.mp4") is None


def test_a_symlinked_storage_root_keeps_the_spelling_the_registry_keys_on(tmp_path):
    real = tmp_path / "nfs_mount"
    (real / "clips" / "cam1").mkdir(parents=True)
    (real / "clips" / "cam1" / "1789000000_animal.mp4").write_bytes(b"x")
    os.symlink(real, tmp_path / "storage")
    server = WebServer({}, tmp_path / "storage", tmp_path / "logs", port=0)

    found = server._confined_clip_path("cam1/1789000000_animal.mp4")

    # The pipeline registers clips as <storage_root>/clips/..., unresolved.
    assert found == tmp_path / "storage" / "clips" / "cam1" / "1789000000_animal.mp4"
    assert found.is_file()


# --- reanalyse ---------------------------------------------------------------------

def test_reanalyse_refuses_a_file_outside_the_archive_and_leaves_it_alone(server, outside):
    for rel_path in escapes(outside):
        resp = asyncio.run(server.handle_reprocess(JsonRequest({"path": rel_path})))

        assert resp.status == 403, rel_path
    assert outside.read_bytes() == b"somebody else's video"
    assert sorted(p.name for p in outside.parent.iterdir() if p.is_file()) == ["outside.log.json", "outside.mp4"]
    assert server.reprocessing_jobs == {}


def test_reanalyse_no_longer_refuses_a_name_that_contains_two_dots(server):
    resp = asyncio.run(server.handle_reprocess(JsonRequest({"path": "cam1/2026/09/10/a..b.mp4"})))

    assert resp.status == 404          # not there, but not "Invalid path" either


# --- the processing log ------------------------------------------------------------

def test_the_log_endpoint_reads_no_log_outside_the_archive(server, outside):
    for rel_path in escapes(outside):
        resp = asyncio.run(server.handle_get_processing_log(JsonRequest(match_info={"path": rel_path})))

        assert resp.status == 403, rel_path
        assert b"not yours" not in (resp.body or b"")


def test_the_log_endpoint_still_serves_a_clip_s_own_log(server):
    clip = server.storage_root / "clips" / CLIP
    request = JsonRequest(match_info={"path": CLIP})

    resp = asyncio.run(server.handle_get_processing_log(request))
    assert resp.status == 200 and json.loads(resp.body)["exists"] is False

    clip.with_suffix(".log.json").write_text(json.dumps({"clip": clip.name, "log_entries": []}))
    resp = asyncio.run(server.handle_get_processing_log(request))
    assert json.loads(resp.body) == {"exists": True, "data": {"clip": clip.name, "log_entries": []}}

    resp = asyncio.run(server.handle_get_processing_log(JsonRequest(match_info={"path": "cam1/nope.mp4"})))
    assert resp.status == 404


# --- clip detail ---------------------------------------------------------------------

def test_clip_detail_says_nothing_about_files_outside_the_archive(server, outside):
    for rel_path in escapes(outside):
        assert server._get_clip_detail(rel_path) is None, rel_path
        assert server._renamed_clip(rel_path) is None, rel_path

        resp = asyncio.run(server.handle_clip_api(JsonRequest(match_info={"path": rel_path})))
        assert resp.status == 404
        assert json.loads(resp.body) == {"error": "Clip not found"}


def test_clip_detail_still_describes_a_real_clip(server):
    detail = server._get_clip_detail(CLIP)

    assert detail["path"] == CLIP and detail["size"] == len(b"a clip")
