"""Retention: what it removes, what it must never remove, and the dry run.

Background (bug hunt, 2026-09-19): ``cleanup`` globbed ``clips/*/*/*/*``, one
level short of ``clips/<camera>/<YYYY>/<MM>/<DD>/<file>``, so every match was
a directory and ``is_file()`` was false for all of them. Retention had
deleted nothing since the first commit: ``max_days`` was not enforced, and
``min_days`` and ``max_utilization_pct`` were read by nothing at all.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from animaltracker.storage import StorageManager

DAY = 86400
NOW = time.time()


def storage(tmp_path) -> StorageManager:
    return StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")


def add_clip(st: StorageManager, camera: str, age_days: float, species: str = "animal",
             *, thumbs: int = 2, sidecar: bool = True, size: int = 4096,
             mtime_age_days: float = None) -> Path:
    """A clip as the pipeline writes it: <epoch>_<species>.mp4 plus its files."""
    started = NOW - age_days * DAY
    when = time.localtime(started)
    day = (st.storage_root / "clips" / camera
           / f"{when.tm_year:04d}" / f"{when.tm_mon:02d}" / f"{when.tm_mday:02d}")
    day.mkdir(parents=True, exist_ok=True)
    clip = day / f"{int(started)}_{species}.mp4"
    clip.write_bytes(b"x" * size)
    names = []
    for i in range(thumbs):
        thumb = day / f"{clip.stem}_thumb_{species}_t{i}.jpg"
        thumb.write_bytes(b"\xff\xd8\xff")
        names.append(thumb.name)
    if sidecar:
        clip.with_suffix(".log.json").write_text(json.dumps(
            {"clip": clip.name, "thumbnails": [{"file": n} for n in names]}))
    stamp = NOW - (mtime_age_days if mtime_age_days is not None else age_days) * DAY
    os.utime(clip, (stamp, stamp))
    return clip


def names(st: StorageManager) -> set:
    return {p.name for p in (st.storage_root / "clips").rglob("*") if p.is_file()}


# --- the glob that never matched anything ---------------------------------------

def test_clips_are_found_wherever_the_pipeline_puts_them(tmp_path):
    st = storage(tmp_path)
    nested = add_clip(st, "cam1", 200)
    manual = st.storage_root / "clips" / "manual_cam1_1789000000.mp4"
    manual.write_bytes(b"x")

    found = set(st.find_clips())

    assert nested in found and manual in found


def test_a_half_written_transcode_is_not_a_clip(tmp_path):
    st = storage(tmp_path)
    add_clip(st, "cam1", 200)
    (st.storage_root / "clips" / "cam1" / "1789000000_animal.tmp.mp4").write_bytes(b"x")

    assert all(not p.name.endswith(".tmp.mp4") for p in st.find_clips())


# --- age ---------------------------------------------------------------------------

def test_clips_past_the_limit_go_and_the_rest_stay(tmp_path):
    st = storage(tmp_path)
    old = add_clip(st, "cam1", 200)
    recent = add_clip(st, "cam1", 30)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert report.deleted == [old]
    assert not old.exists() and recent.exists()
    assert report.examined == 2 and report.kept == 1


def test_a_clip_goes_with_its_key_frames_and_its_log(tmp_path):
    st = storage(tmp_path)
    add_clip(st, "cam1", 200, thumbs=3)
    add_clip(st, "cam1", 10, species="deer", thumbs=1)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert report.files_removed == 5                 # clip + 3 key frames + log
    assert names(st) == {
        p.name for p in (st.storage_root / "clips").rglob("*deer*") if p.is_file()}


def test_age_comes_from_the_event_not_the_file(tmp_path):
    """A reanalysis rewrites the sidecar and leaves the video's mtime alone;
    a long event's transcode finishes minutes after it began."""
    st = storage(tmp_path)
    # Recorded 200 days ago, but its file was touched yesterday.
    touched = add_clip(st, "cam1", 200, mtime_age_days=1)
    # Recorded yesterday, but with an ancient mtime.
    fresh = add_clip(st, "cam2", 1, species="deer", mtime_age_days=300)

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert not touched.exists(), "an old event is old however new its file is"
    assert fresh.exists(), "a new event is new however old its file is"


def test_a_name_without_an_epoch_falls_back_to_the_file(tmp_path):
    st = storage(tmp_path)
    manual = st.storage_root / "clips" / "manual_cam1_1789000000.mp4"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_bytes(b"x")
    stamp = NOW - 200 * DAY
    os.utime(manual, (stamp, stamp))

    assert st.prune_clips(max_days=120, min_days=7, dry_run=False).deleted == [manual]


# --- the floor ----------------------------------------------------------------------

def test_nothing_younger_than_min_days_is_ever_removed(tmp_path):
    st = storage(tmp_path)
    young = add_clip(st, "cam1", 3)

    # Even with a limit that would otherwise take it, and a full disk.
    report = st.prune_clips(max_days=1, min_days=7, max_utilization_pct=1, dry_run=False)

    assert young.exists() and report.deleted == [] and report.protected == 1


def test_the_floor_wins_when_it_is_past_the_limit(tmp_path):
    st = storage(tmp_path)
    clip = add_clip(st, "cam1", 20)

    assert st.prune_clips(max_days=10, min_days=30, dry_run=False).deleted == []
    assert clip.exists()


# --- the disk ceiling ----------------------------------------------------------------

def test_a_disk_over_its_ceiling_takes_the_oldest_first(tmp_path, monkeypatch):
    st = storage(tmp_path)
    oldest = add_clip(st, "cam1", 100)
    middle = add_clip(st, "cam1", 90, species="deer")
    newest = add_clip(st, "cam1", 80, species="bird")
    # Well inside max_days, but the disk is full. Usage falls as clips
    # actually go, which is what the real pass re-measures between deletes.
    def usage(assume_freed=0, **kw):
        left = len([p for p in (st.storage_root / "clips").rglob("*.mp4") if p.is_file()])
        return 95.0 if left > 1 else 50.0

    monkeypatch.setattr(st, "_utilization_pct", usage)

    report = st.prune_clips(max_days=365, min_days=7, max_utilization_pct=80, dry_run=False)

    assert not oldest.exists() and not middle.exists()
    assert newest.exists()
    assert report.by_camera == {"cam1": 2}


def test_it_says_so_when_it_cannot_get_under_the_ceiling(tmp_path, monkeypatch):
    st = storage(tmp_path)
    add_clip(st, "cam1", 3)                       # protected by the floor
    monkeypatch.setattr(st, "_utilization_pct", lambda assume_freed=0, **kw: 99.0)

    report = st.prune_clips(max_days=365, min_days=7, max_utilization_pct=80, dry_run=False)

    assert report.deleted == [] and report.still_over_ceiling is True


def test_a_volume_slow_to_report_freed_space_is_not_emptied(tmp_path, monkeypatch):
    # Bug hunt, 2026-09-22: the real run re-read the disk after every delete,
    # so on a volume that reports frees late (an NFS server keeping
    # snapshots) usage never seemed to fall and it deleted down to the
    # min_days floor, far past what its dry run had listed.
    from collections import namedtuple

    from animaltracker import storage as storage_mod

    Usage = namedtuple("Usage", "total used free")
    st = storage(tmp_path)
    clips = [add_clip(st, "cam1", age, size=10_000) for age in (100, 90, 80, 70, 60)]
    # 83% of 100 kB used, and the reading never changes: one 10 kB clip
    # brings it under the 80% ceiling.
    monkeypatch.setattr(storage_mod.shutil, "disk_usage", lambda path: Usage(100_000, 83_000, 17_000))

    preview = st.prune_clips(max_days=365, min_days=7, max_utilization_pct=80, dry_run=True)
    report = st.prune_clips(max_days=365, min_days=7, max_utilization_pct=80, dry_run=False)

    assert len(preview.deleted) == 1 and report.deleted == preview.deleted
    assert not clips[0].exists() and all(c.exists() for c in clips[1:])
    assert report.still_over_ceiling is False


# --- the dry run ----------------------------------------------------------------------

def test_a_dry_run_removes_nothing_and_reports_everything(tmp_path):
    st = storage(tmp_path)
    old_one = add_clip(st, "cam1", 300, thumbs=2)
    old_two = add_clip(st, "Otteson2", 200, species="deer", thumbs=1)
    add_clip(st, "cam1", 5)
    before = names(st)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=True)

    assert names(st) == before, "a dry run must not touch a single file"
    assert set(report.deleted) == {old_one, old_two}
    assert report.files_removed == 2 + 3 + 2       # 2 clips, 3 key frames, 2 logs
    assert report.freed_bytes > 0
    assert report.by_camera == {"cam1": 1, "Otteson2": 1}
    assert report.protected == 1
    assert report.dry_run is True
    assert time.strftime("%Y", time.localtime(report.oldest)) <= time.strftime("%Y", time.localtime(report.newest))


def test_the_dry_run_and_the_real_run_agree(tmp_path):
    st = storage(tmp_path)
    for age in (300, 200, 150, 30):
        add_clip(st, "cam1", age, species=f"s{int(age)}")

    preview = st.prune_clips(max_days=120, min_days=7, dry_run=True)
    real = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert preview.deleted == real.deleted
    assert preview.files_removed == real.files_removed


# --- half-finished work is not a clip -------------------------------------------------
#
# The production archive held three *.temp.avi files, 3.8 GB, with no MP4 beside
# them: events whose transcode never completed, from before the pipeline learned
# to recover them. Pruning them as "old clips" would throw away footage that was
# never saved in the first place.

def interrupted(st: StorageManager, camera: str, age_days: float, name: str) -> Path:
    started = NOW - age_days * DAY
    when = time.localtime(started)
    day = (st.storage_root / "clips" / camera
           / f"{when.tm_year:04d}" / f"{when.tm_mon:02d}" / f"{when.tm_mday:02d}")
    day.mkdir(parents=True, exist_ok=True)
    path = day / name
    path.write_bytes(b"x" * 8192)
    os.utime(path, (started, started))
    return path


def test_a_recent_interrupted_recording_is_kept_and_reported(tmp_path):
    """It may still be worth turning into a clip, so it is not swept away."""
    st = storage(tmp_path)
    stranded = interrupted(st, "cam2", 3, "1789000000_animal.temp.avi")
    old = add_clip(st, "cam1", 300)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert report.deleted == [old]
    assert stranded.exists()
    assert report.interrupted_seen == 1 and report.interrupted_removed == 0
    assert st.find_interrupted_recordings() == [stranded]


def test_one_past_the_limit_goes_with_the_clips(tmp_path):
    st = storage(tmp_path)
    stranded = interrupted(st, "cam2", 300, "1766760676_animal.temp.avi")

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert not stranded.exists()
    assert report.interrupted_removed == 1
    assert st.find_interrupted_recordings() == []


def test_it_is_never_counted_as_a_clip(tmp_path):
    st = storage(tmp_path)
    interrupted(st, "cam2", 300, "1766760676_animal.temp.avi")
    add_clip(st, "cam1", 300)

    report = st.prune_clips(max_days=120, min_days=7, dry_run=True)

    assert report.examined == 1            # one clip, not two
    assert report.interrupted_seen == 1


def test_a_half_written_transcode_is_junk_once_it_is_stale(tmp_path):
    st = storage(tmp_path)
    half = interrupted(st, "cam2", 300, "1766760999_animal.tmp.mp4")

    st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert not half.exists()


def test_an_intermediate_is_aged_by_when_it_was_written_not_by_its_name(tmp_path):
    """A transcode running now may be finishing a months-old event — the
    recovery sweep does exactly that — and must not be deleted mid-write."""
    st = storage(tmp_path)
    in_flight = interrupted(st, "cam2", 300, "1766760676_animal.temp.avi")
    os.utime(in_flight, (NOW, NOW))          # being written right now

    report = st.prune_clips(max_days=120, min_days=7, dry_run=False)

    assert in_flight.exists()
    assert report.interrupted_removed == 0 and report.protected == 1
