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
