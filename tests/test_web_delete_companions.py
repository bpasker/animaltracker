"""Deleting a clip removes its key frames and its processing log too.

Background (bug hunt, 2026-09-19): the delete endpoint unlinked the video and
nothing else. The archive lists videos only, so the clip's
``<stem>_thumb_*.jpg`` key frames and ``<stem>.log.json`` stayed on the NAS
for good: invisible in the UI and, with no retention running, never pruned.
The pipeline's own false-positive cleanup removes all three.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from animaltracker.web import WebServer

DAY = "cam1/2026/09/10"
STEM = "1789000000_mammalia_carnivora_canidae"


@pytest.fixture
def server(tmp_path):
    (tmp_path / "clips" / DAY).mkdir(parents=True)
    return WebServer({}, tmp_path, tmp_path / "logs", port=0)


def make_clip(server, stem: str = STEM, thumbs=("canidae_t0", "canidae_t1"), listed=None) -> Path:
    day = server.storage_root / "clips" / DAY
    clip = day / f"{stem}.mp4"
    clip.write_bytes(b"video")
    names = [f"{stem}_thumb_{t}.jpg" for t in thumbs]
    for name in names:
        (day / name).write_bytes(b"\\xff\\xd8\\xff")
    log = {"clip": clip.name, "thumbnails": [{"file": n} for n in (names if listed is None else listed)]}
    clip.with_suffix(".log.json").write_text(json.dumps(log))
    return clip


def remaining(server) -> list:
    return sorted(p.name for p in (server.storage_root / "clips" / DAY).iterdir())


def test_the_key_frames_and_the_log_go_with_the_clip(server):
    make_clip(server)

    ok, message = server._delete_file(f"{DAY}/{STEM}.mp4")

    assert ok and message == f"Deleted {DAY}/{STEM}.mp4"
    assert remaining(server) == []


def test_a_neighbouring_clip_keeps_everything(server):
    make_clip(server)
    other = "1789000000_mammalia_carnivora_canidae+person"          # same epoch, a longer stem
    make_clip(server, stem=other, thumbs=("person_t0",))
    third = "1789000321_animal"
    make_clip(server, stem=third, thumbs=("animal_t0",))

    assert server._delete_file(f"{DAY}/{STEM}.mp4")[0]

    assert remaining(server) == sorted([
        f"{other}.mp4", f"{other}.log.json", f"{other}_thumb_person_t0.jpg",
        f"{third}.mp4", f"{third}.log.json", f"{third}_thumb_animal_t0.jpg",
    ])


def test_key_frames_a_stale_listing_hides_are_found_through_the_log(server, monkeypatch):
    make_clip(server)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(()))   # the NFS listing shows nothing

    assert server._delete_file(f"{DAY}/{STEM}.mp4")[0]

    assert remaining(server) == []


def test_the_log_cannot_point_the_delete_at_someone_else_s_files(server):
    day = server.storage_root / "clips" / DAY
    (day / "1789000999_animal_thumb_animal_t0.jpg").write_bytes(b"another clip's key frame")
    outside = server.storage_root / "keep.jpg"
    outside.write_bytes(b"not a key frame at all")
    make_clip(server, thumbs=(), listed=[
        "1789000999_animal_thumb_animal_t0.jpg",      # another clip's
        "../../../../keep.jpg",                        # not a plain name
        f"{STEM}_thumb_canidae_t0.txt",                # not a thumbnail
    ])

    assert server._delete_file(f"{DAY}/{STEM}.mp4")[0]

    assert remaining(server) == ["1789000999_animal_thumb_animal_t0.jpg"]
    assert outside.exists()


def test_a_clip_with_no_companions_and_a_file_that_is_not_a_clip_behave_as_before(server):
    day = server.storage_root / "clips" / DAY
    (day / "manual_cam1_1789000000.mp4").write_bytes(b"video")
    (day / f"{STEM}_thumb_canidae_t0.jpg").write_bytes(b"jpg")
    (day / f"{STEM}.log.json").write_text("{}")

    assert server._delete_file(f"{DAY}/manual_cam1_1789000000.mp4")[0]
    assert server._delete_file(f"{DAY}/{STEM}_thumb_canidae_t0.jpg")[0]      # deleting one key frame by name

    assert remaining(server) == [f"{STEM}.log.json"]


# --- a clip that is being analysed --------------------------------------------------
#
# Background (bug hunt, 2026-09-19): nothing stopped a clip from being deleted from
# under its analysis. The post-processor then wrote key frames and a log for a
# file that was gone, failed its rename, and on the live path sent an alert whose
# link was dead.

import asyncio

from animaltracker.analysis_recovery import ClipAnalysisRegistry


class QueryRequest:
    def __init__(self, path: str) -> None:
        self.query = {"path": path}


class JsonRequest:
    def __init__(self, body) -> None:
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def analysing(tmp_path):
    (tmp_path / "clips" / DAY).mkdir(parents=True)
    registry = ClipAnalysisRegistry()
    server = WebServer({}, tmp_path, tmp_path / "logs", port=0, analysis_registry=registry)
    clip = make_clip(server)
    registry.begin(clip, "event")
    return server, registry, clip


def test_a_clip_under_analysis_is_not_deleted(analysing):
    server, registry, clip = analysing
    before = remaining(server)

    resp = asyncio.run(server.handle_delete_recording(QueryRequest(f"{DAY}/{STEM}.mp4")))

    assert resp.status == 409 and "being analysed" in resp.text
    assert remaining(server) == before

    registry.end(clip)
    resp = asyncio.run(server.handle_delete_recording(QueryRequest(f"{DAY}/{STEM}.mp4")))
    assert resp.status == 200 and remaining(server) == []


def test_a_reanalysis_from_the_clip_page_protects_it_too(server):
    make_clip(server)
    rel = f"{DAY}/{STEM}.mp4"
    server.reprocessing_jobs[rel] = {"started": "now"}

    assert server._delete_file(rel) == (False, WebServer.DELETE_REFUSED_ANALYSING)


def test_a_bulk_delete_removes_the_rest_and_says_which_it_left(analysing):
    import json as _json

    server, _registry, _clip = analysing
    other = "1789000321_animal"
    make_clip(server, stem=other, thumbs=("animal_t0",))

    resp = asyncio.run(server.handle_bulk_delete(JsonRequest(
        {"paths": [f"{DAY}/{STEM}.mp4", f"{DAY}/{other}.mp4"]})))

    body = _json.loads(resp.body)
    assert body["deleted_count"] == 1 and body["total_requested"] == 2
    assert [r["success"] for r in body["results"]] == [False, True]
    assert "being analysed" in body["results"][0]["message"]
    assert f"{STEM}.mp4" in remaining(server) and f"{other}.mp4" not in remaining(server)
