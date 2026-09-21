"""Retention also removes key frames and logs whose clip is already gone.

A delete that removed only the video left the rest behind: the web delete did
until e236509, and a rename used to leave the old name's log. On production
that came to 954 events, 4,370 files and 0.82 GB, most of it from January,
and nothing ever found them because the archive and retention both start
from the videos.

``find_leftovers`` groups files by directory and event epoch, never by name,
because the name after the epoch changes: the post-processor writes a clip's
key frames and log under the name it is about to take and renames the video
last. The cases below that must NOT be swept are the point of this file as
much as the ones that must.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import time
from pathlib import Path

import pytest

from animaltracker.cli import cmd_cleanup
from animaltracker.storage import StorageManager

DAY = 86400
NOW = time.time()


def storage(tmp_path) -> StorageManager:
    return StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")


def day_dir(st: StorageManager, camera: str, epoch: int) -> Path:
    when = time.localtime(epoch)
    d = (st.storage_root / "clips" / camera
         / f"{when.tm_year:04d}" / f"{when.tm_mon:02d}" / f"{when.tm_mday:02d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def event(st: StorageManager, camera: str, age_days: float, label: str = "animal", *,
          video: bool = True, thumbs: int = 2, sidecar: bool = True) -> tuple:
    """An event's files as the pipeline writes them; returns (epoch, [paths])."""
    epoch = int(NOW - age_days * DAY)
    d = day_dir(st, camera, epoch)
    stem = f"{epoch}_{label}"
    paths = []
    if video:
        clip = d / f"{stem}.mp4"
        clip.write_bytes(b"x" * 4096)
        paths.append(clip)
    for i in range(thumbs):
        thumb = d / f"{stem}_thumb_{label}_t{i}.jpg"
        thumb.write_bytes(b"\xff\xd8\xff" * 100)
        paths.append(thumb)
    if sidecar:
        log = d / f"{stem}.log.json"
        log.write_text(json.dumps({"clip": f"{stem}.mp4"}))
        paths.append(log)
    return epoch, paths


def on_disk(st: StorageManager) -> set:
    return {p.name for p in (st.storage_root / "clips").rglob("*") if p.is_file()}


# --- what goes --------------------------------------------------------------------------

def test_key_frames_and_a_log_with_no_clip_go_once_past_the_limit(tmp_path):
    st = storage(tmp_path)
    _, files = event(st, "cam2", 240, video=False, thumbs=3)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert not any(p.exists() for p in files)
    assert report.leftovers_removed == 1 and report.leftover_files == 4
    assert report.leftover_bytes > 0
    assert report.deleted == [], "no clip was involved, and none is reported"


def test_a_log_left_under_an_old_name_goes_too(tmp_path):
    """The two the production prune missed: <epoch>_animal.log.json beside
    nothing, because the clip had been renamed and then pruned."""
    st = storage(tmp_path)
    _, files = event(st, "cam2", 270, label="animal", video=False, thumbs=0)

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert on_disk(st) == set()


def test_they_are_aged_like_clips_and_kept_until_then(tmp_path):
    st = storage(tmp_path)
    _, young = event(st, "Otteson2", 9, video=False)       # a deleted false positive, last week

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert all(p.exists() for p in young)
    assert report.leftovers_seen == 1 and report.leftovers_removed == 0


def test_the_floor_protects_them_whatever_max_days_says(tmp_path):
    st = storage(tmp_path)
    _, files = event(st, "cam1", 3, video=False)

    st.prune_clips(max_days=1, min_days=7, dry_run=False)

    assert all(p.exists() for p in files)


# --- what must never go -------------------------------------------------------------------

def test_a_clip_being_renamed_holds_its_new_named_files(tmp_path):
    """Mid-analysis: key frames and the log are written under the name the clip
    is about to take, and the video still has its old one. Same epoch, so the
    video holds the whole group."""
    st = storage(tmp_path)
    epoch, _ = event(st, "cam1", 30, label="animal", thumbs=0, sidecar=False)
    d = day_dir(st, "cam1", epoch)
    (d / f"{epoch}_deer_thumb_deer_t0.jpg").write_bytes(b"\xff")
    (d / f"{epoch}_deer.log.json").write_text("{}")

    assert st.find_leftovers() == []


def test_a_half_written_video_holds_its_group(tmp_path):
    st = storage(tmp_path)
    epoch, _ = event(st, "cam1", 200, video=False)
    (day_dir(st, "cam1", epoch) / f"{epoch}_animal.tmp.mp4").write_bytes(b"x")

    assert st.find_leftovers() == []


def test_an_event_still_recording_holds_its_group(tmp_path):
    """Its video is in event_temp, not beside the key frames, until the
    transcode; a crashed run's recording is recovered into the clip at the
    next startup, whatever its age."""
    st = storage(tmp_path)
    epoch, files = event(st, "Otteson2", 200, video=False)
    temp = st.logs_root / "event_temp"
    temp.mkdir(parents=True, exist_ok=True)
    (temp / f"Otteson2_{epoch}_0a1b2c3d.temp.avi").write_bytes(b"x")

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert all(p.exists() for p in files)
    assert report.leftovers_seen == 0


def test_a_recording_for_another_camera_does_not_hold_it(tmp_path):
    st = storage(tmp_path)
    epoch, files = event(st, "Otteson2", 200, video=False)
    temp = st.logs_root / "event_temp"
    temp.mkdir(parents=True, exist_ok=True)
    (temp / f"cam1_{epoch}_0a1b2c3d.temp.avi").write_bytes(b"x")

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert not any(p.exists() for p in files)


def test_a_file_the_pipeline_does_not_write_holds_its_group(tmp_path):
    st = storage(tmp_path)
    epoch, files = event(st, "cam1", 200, video=False)
    notes = day_dir(st, "cam1", epoch) / f"{epoch}_notes.txt"
    notes.write_text("kept by hand")

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert notes.exists() and all(p.exists() for p in files)


def test_files_outside_an_event_are_never_considered(tmp_path):
    st = storage(tmp_path)
    root = st.storage_root / "clips"
    root.mkdir(parents=True, exist_ok=True)
    stray_root = root / "1766000000_thumb_root.jpg"          # the clips root is manual clips'
    stray_root.write_bytes(b"x")
    d = day_dir(st, "cam1", int(NOW - 300 * DAY))
    unnamed = d / "thumb_without_epoch.jpg"
    unnamed.write_bytes(b"x")
    for p in (stray_root, unnamed):
        os.utime(p, (NOW - 300 * DAY, NOW - 300 * DAY))

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert stray_root.exists() and unnamed.exists()


def test_a_live_clips_files_are_not_leftovers(tmp_path):
    st = storage(tmp_path)
    event(st, "cam1", 30)

    assert st.find_leftovers() == []


# --- together with clips, and the dry run ----------------------------------------------

def test_a_pruned_clips_old_name_log_is_a_leftover_on_the_next_pass(tmp_path):
    """Leftovers are listed before anything goes, so the dry run and the real
    run agree; what a clip's removal strands is picked up the pass after."""
    st = storage(tmp_path)
    epoch, clip_files = event(st, "cam1", 200, label="deer")
    stray = day_dir(st, "cam1", epoch) / f"{epoch}_animal.log.json"
    stray.write_text("{}")

    first = st.prune_clips(max_days=120, min_days=7, dry_run=False)
    assert not any(p.exists() for p in clip_files) and stray.exists()
    assert first.leftovers_removed == 0

    second = st.prune_clips(max_days=120, min_days=7, dry_run=False)
    assert not stray.exists() and second.leftovers_removed == 1


def test_the_dry_run_touches_nothing_and_matches_the_real_run(tmp_path):
    st = storage(tmp_path)
    event(st, "cam1", 300)                                  # a clip past the limit
    event(st, "cam2", 250, video=False, thumbs=3)           # two leftovers past it
    event(st, "cam2", 140, label="bird", video=False, thumbs=1)
    event(st, "Otteson2", 9, video=False)                   # a young one
    before = on_disk(st)

    preview = st.prune_clips(max_days=120, min_days=7, dry_run=True)
    assert on_disk(st) == before

    real = st.prune_clips(max_days=120, min_days=7, dry_run=False)
    for field in ("leftovers_seen", "leftovers_removed", "leftover_files", "leftover_bytes",
                  "files_removed", "freed_bytes"):
        assert getattr(preview, field) == getattr(real, field), field
    assert preview.leftovers_removed == 2 and preview.leftover_files == 4 + 2


def test_a_dry_runs_ceiling_estimate_counts_the_leftovers(tmp_path, monkeypatch):
    st = storage(tmp_path)
    event(st, "cam1", 200, video=False, thumbs=3)
    seen = []

    def usage(assume_freed=0):
        seen.append(assume_freed)
        return 50.0

    monkeypatch.setattr(st, "_utilization_pct", usage)
    report = st.prune_clips(max_days=120, min_days=7, max_utilization_pct=80, dry_run=True)

    assert seen and seen[-1] == report.leftover_bytes > 0


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores the mode")
def test_a_leftover_that_cannot_be_removed_is_a_failure(tmp_path):
    st = storage(tmp_path)
    epoch, files = event(st, "cam2", 200, video=False)
    d = day_dir(st, "cam2", epoch)
    d.chmod(0o555)
    try:
        report = st.prune_clips(max_days=120, min_days=7, dry_run=False)
    finally:
        d.chmod(0o755)

    assert set(report.failed) == set(files)


# --- the command ---------------------------------------------------------------------------

def write_config(tmp_path) -> Path:
    cfg = tmp_path / "cameras.yml"
    cfg.write_text(textwrap.dedent(f"""
        general:
          storage_root: {tmp_path / 'storage'}
          logs_root: {tmp_path / 'logs'}
          notification:
            pushover_app_token_env: PUSHOVER_APP_TOKEN
            pushover_user_key_env: PUSHOVER_USER_KEY
          retention:
            max_days: 120
            min_days: 7
        cameras: []
    """))
    return cfg


def test_the_dry_run_says_what_it_would_sweep(tmp_path, caplog):
    st = storage(tmp_path)
    event(st, "cam2", 250, video=False, thumbs=3)
    event(st, "Otteson2", 9, video=False)
    before = on_disk(st)

    with caplog.at_level(logging.INFO):
        code = cmd_cleanup(argparse.Namespace(config=str(write_config(tmp_path)), dry_run=True))

    text = caplog.text
    assert code == 0
    assert on_disk(st) == before
    assert "would also remove key frames and logs of 1 event(s) whose clip is already gone: 4 file(s)" in text
    assert "1 more event(s) with key frames but no clip, kept until they pass 120 days" in text
    assert "Nothing was removed. Run without --dry-run to apply." in text


def test_the_real_run_removes_them_and_says_so(tmp_path, caplog):
    st = storage(tmp_path)
    _, files = event(st, "cam2", 250, video=False, thumbs=3)

    with caplog.at_level(logging.INFO):
        code = cmd_cleanup(argparse.Namespace(config=str(write_config(tmp_path)), dry_run=False))

    assert code == 0
    assert not any(p.exists() for p in files)
    assert "also removed key frames and logs of 1 event(s)" in caplog.text
