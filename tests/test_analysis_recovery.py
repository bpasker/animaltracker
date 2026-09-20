"""The recovery of clips whose post-processing a restart interrupted.

A finished analysis always writes the ``.log.json`` sidecar, so a clip that
still carries the real-time detector's ``animal`` label and has no sidecar
is exactly one whose job was cut off. The sweeper finishes those, one at a
time, newest first, only while no live event needs the post-processor, and
never touches a clip another job holds.
"""
import os
import threading
import time
from pathlib import Path

import pytest

from animaltracker.analysis_recovery import (
    ClipAnalysisRegistry,
    RecoverySweeper,
    find_unfinished_clips,
    is_unclassified_clip,
    sidecar_path,
)

NOW = 1_789_400_000.0


def make_clip(clips_dir, camera, epoch, label, *, age_s=3600.0, sidecar=False):
    d = clips_dir / camera / "2026" / "09" / "14"
    d.mkdir(parents=True, exist_ok=True)
    clip = d / f"{int(epoch)}_{label}.mp4"
    clip.write_bytes(b"\x00")
    os.utime(clip, (NOW - age_s, NOW - age_s))
    if sidecar:
        sidecar_path(clip).write_text("{}")
    return clip


# --------------------------------------------------------------- selection

def test_only_generic_labels_count_as_unclassified():
    assert is_unclassified_clip(Path("1789384540_animal.mp4"))
    assert is_unclassified_clip(Path("1789384540_animal+animal.mp4"))
    assert not is_unclassified_clip(Path("1789384540_mammalia_carnivora_canidae.mp4"))
    assert not is_unclassified_clip(Path("1789384540_person.mp4"))
    assert not is_unclassified_clip(Path("1789384540_animal+bird.mp4"))
    assert not is_unclassified_clip(Path("manual_cam1_1789384540.mp4"))
    assert not is_unclassified_clip(Path("1789384540.mp4"))


def test_sidecar_path_sits_beside_the_clip():
    clip = Path("/x/cam1/2026/09/14/1789384540_animal.mp4")
    assert sidecar_path(clip) == Path("/x/cam1/2026/09/14/1789384540_animal.log.json")


def test_unfinished_means_unclassified_name_and_no_sidecar(tmp_path):
    clips = tmp_path / "clips"
    cut_off = make_clip(clips, "Otteson2", 1789384540, "animal")
    make_clip(clips, "cam1", 1789384541, "animal", sidecar=True)             # finished, no animal found
    make_clip(clips, "cam1", 1789384542, "mammalia_carnivora_canidae")       # classified (pre-sidecar era)
    make_clip(clips, "cam1", 1789384543, "person")                           # never post-processed by design

    assert find_unfinished_clips(clips, now=NOW) == [cut_off]


def test_manual_clips_in_the_root_are_not_candidates(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    manual = clips / "1789384540_animal.mp4"
    manual.write_bytes(b"\x00")
    os.utime(manual, (NOW - 3600, NOW - 3600))
    (clips / "notes.txt").write_text("x")

    assert find_unfinished_clips(clips, now=NOW) == []


def test_a_clip_written_moments_ago_is_left_to_its_live_event(tmp_path):
    clips = tmp_path / "clips"
    young = make_clip(clips, "cam1", 1789384540, "animal", age_s=30.0)
    old = make_clip(clips, "cam1", 1789384000, "animal", age_s=600.0)

    assert find_unfinished_clips(clips, now=NOW) == [old]
    assert find_unfinished_clips(clips, now=NOW, min_age_s=0) == [young, old]   # newest first


def test_newest_clip_comes_first(tmp_path):
    clips = tmp_path / "clips"
    oldest = make_clip(clips, "cam2", 1779667652, "animal", age_s=9_000_000)
    newest = make_clip(clips, "Otteson2", 1789384540, "animal", age_s=1_000)
    middle = make_clip(clips, "cam1", 1788000000, "animal", age_s=500_000)

    assert find_unfinished_clips(clips, now=NOW) == [newest, middle, oldest]


def test_missing_clips_dir_is_empty(tmp_path):
    assert find_unfinished_clips(tmp_path / "nope") == []


# ---------------------------------------------------------------- registry

def test_registry_claims_a_path_once(tmp_path):
    reg = ClipAnalysisRegistry()
    clip = tmp_path / "1_animal.mp4"

    assert reg.begin(clip, "event")
    assert not reg.begin(clip, "recovery")
    assert reg.is_active(clip)
    assert reg.source_of(clip) == "event"
    assert [e["source"] for e in reg.active()] == ["event"]

    reg.end(clip)
    assert not reg.is_active(clip)
    assert reg.active() == []
    reg.end(clip)  # idempotent


def test_registry_matches_equivalent_spellings_of_one_path(tmp_path):
    reg = ClipAnalysisRegistry()
    clip = tmp_path / "cam1" / "1_animal.mp4"
    assert reg.begin(clip, "event")
    assert reg.is_active(tmp_path / "cam1" / "." / "1_animal.mp4")
    assert reg.is_active(str(clip))


def test_live_count_never_goes_negative():
    reg = ClipAnalysisRegistry()
    assert not reg.live_busy()
    reg.live_begin()
    reg.live_begin()
    assert reg.live_count == 2
    reg.live_end()
    reg.live_end()
    reg.live_end()
    assert reg.live_count == 0
    assert not reg.live_busy()


# ----------------------------------------------------------------- sweeper

def sweeper(clips, registry, process, **kw):
    kw.setdefault("min_age_s", 0)
    kw.setdefault("live_wait_s", 0.01)
    return RecoverySweeper(clips, registry, process, **kw)


def test_sweep_processes_every_unfinished_clip_newest_first(tmp_path):
    clips = tmp_path / "clips"
    old = make_clip(clips, "cam2", 1779667652, "animal", age_s=9_000_000)
    new = make_clip(clips, "Otteson2", 1789384540, "animal", age_s=1_000)
    make_clip(clips, "cam1", 1789384541, "animal", sidecar=True)
    seen = []

    def process(path):
        seen.append(path)
        sidecar_path(path).write_text("{}")   # what a finished job leaves behind
        return True

    sw = sweeper(clips, ClipAnalysisRegistry(), process)
    assert sw.sweep() == 2
    assert seen == [new, old]
    status = sw.status()
    assert status["processed"] == 2 and status["failed"] == 0 and status["current"] is None
    assert status["last_candidates"] == 2

    # Nothing left: the sidecars are there now.
    assert sw.sweep() == 0
    assert seen == [new, old]


def test_sweep_skips_a_clip_another_job_holds(tmp_path):
    clips = tmp_path / "clips"
    held = make_clip(clips, "cam1", 1789384540, "animal")
    free = make_clip(clips, "cam1", 1789384000, "animal")
    reg = ClipAnalysisRegistry()
    reg.begin(held, "event")
    seen = []

    sw = sweeper(clips, reg, lambda p: seen.append(p) or True)
    assert sw.sweep() == 1
    assert seen == [free]


def test_a_failed_clip_is_tried_once_per_process(tmp_path):
    clips = tmp_path / "clips"
    bad = make_clip(clips, "cam1", 1789384540, "animal")
    calls = []

    sw = sweeper(clips, ClipAnalysisRegistry(), lambda p: calls.append(p) and False)
    assert sw.sweep() == 0
    assert sw.sweep() == 0
    assert calls == [bad]
    assert sw.status()["failed"] == 1


def test_an_exception_in_one_clip_does_not_end_the_sweep(tmp_path):
    clips = tmp_path / "clips"
    boom = make_clip(clips, "cam1", 1789384540, "animal", age_s=10)
    fine = make_clip(clips, "cam1", 1789384000, "animal", age_s=20)
    seen = []

    def process(path):
        seen.append(path)
        if path == boom:
            raise RuntimeError("cannot open video")
        return True

    sw = sweeper(clips, ClipAnalysisRegistry(), process)
    assert sw.sweep() == 1
    assert seen == [boom, fine]
    assert sw.status()["failed"] == 1


def test_sweep_does_nothing_while_disabled(tmp_path):
    clips = tmp_path / "clips"
    make_clip(clips, "cam1", 1789384540, "animal")
    flag = {"on": False}
    seen = []

    sw = sweeper(clips, ClipAnalysisRegistry(), lambda p: seen.append(p) or True,
                 enabled=lambda: flag["on"])
    assert sw.sweep() == 0
    assert seen == []
    assert sw.status()["enabled"] is False

    flag["on"] = True
    assert sw.sweep() == 1
    assert sw.status()["enabled"] is True


def test_switching_recovery_off_stops_a_sweep_that_is_under_way(tmp_path):
    """The switch was read once per sweep, and a backlog can run for hours."""
    clips = tmp_path / "clips"
    for i in range(4):
        make_clip(clips, "cam1", 1789384540 + i, "animal", age_s=1_000 + i)
    switch = {"on": True}
    seen = []

    def process(path):
        seen.append(path)
        sidecar_path(path).write_text("{}")
        if len(seen) == 1:
            switch["on"] = False                 # someone turns it off during the first clip
        return True

    sw = sweeper(clips, ClipAnalysisRegistry(), process, enabled=lambda: switch["on"])

    assert sw.sweep() == 1
    assert len(seen) == 1
    switch["on"] = True
    assert sw.sweep() == 3                       # and the rest wait for it to come back on


def test_a_clip_analysed_by_someone_else_meanwhile_is_not_analysed_again(tmp_path):
    clips = tmp_path / "clips"
    newest = make_clip(clips, "cam1", 1789384549, "animal", age_s=1_000)
    reanalysed = make_clip(clips, "cam1", 1789384540, "animal", age_s=2_000)
    renamed = make_clip(clips, "cam1", 1789384530, "animal", age_s=3_000)
    seen = []

    def process(path):
        seen.append(path)
        sidecar_path(path).write_text("{}")
        if path == newest:
            # While this one ran, the clip page reanalysed the next (it kept
            # its name) and a second reanalysis renamed the one after.
            sidecar_path(reanalysed).write_text('{"clip": "theirs", "settings": {}}')
            renamed.rename(renamed.with_name("1789384530_mammalia_carnivora_canidae.mp4"))
        return True

    sw = sweeper(clips, ClipAnalysisRegistry(), process)

    assert sw.sweep() == 1
    assert seen == [newest]
    assert sidecar_path(reanalysed).read_text() == '{"clip": "theirs", "settings": {}}'
    assert sw.status()["failed"] == 0            # a clip that went away is not a failure


def test_sweep_waits_for_live_post_processing_to_finish(tmp_path):
    clips = tmp_path / "clips"
    clip = make_clip(clips, "cam1", 1789384540, "animal")
    reg = ClipAnalysisRegistry()
    reg.live_begin()
    order = []

    def process(path):
        order.append(("recovery", reg.live_count))
        return True

    sw = sweeper(clips, reg, process)

    def release():
        time.sleep(0.05)
        order.append(("live done", None))
        reg.live_end()

    t = threading.Thread(target=release)
    t.start()
    assert sw.sweep() == 1
    t.join()
    assert order == [("live done", None), ("recovery", 0)]


def test_stop_interrupts_a_wait_for_live_events(tmp_path):
    clips = tmp_path / "clips"
    make_clip(clips, "cam1", 1789384540, "animal")
    reg = ClipAnalysisRegistry()
    reg.live_begin()   # never released
    seen = []

    sw = sweeper(clips, reg, lambda p: seen.append(p) or True)
    threading.Timer(0.05, sw.stop).start()
    assert sw.sweep() == 0
    assert seen == []
    assert sw.stopped


def test_thread_runs_a_sweep_after_the_initial_delay(tmp_path):
    clips = tmp_path / "clips"
    make_clip(clips, "cam1", 1789384540, "animal")
    done = threading.Event()

    def process(path):
        sidecar_path(path).write_text("{}")
        done.set()
        return True

    sw = sweeper(clips, ClipAnalysisRegistry(), process, initial_delay_s=0.01, interval_s=60)
    sw.start()
    try:
        assert done.wait(5), "the sweeper thread never ran"
    finally:
        sw.stop()
        sw.join(5)
    assert not sw.is_alive()
    assert sw.status()["running"] is False
