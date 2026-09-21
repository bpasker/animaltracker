"""The clip is renamed last, after everything this run produced is on disk.

Background (bug hunt, 2026-09-19): ``process_clip`` renamed the clip to its
species and only then wrote the key frames and the sidecar. The rename is what
tells the recovery sweep a clip is finished (it looks only at clips that still
carry the generic label), so a kill in that window — every deploy is a
restart, and since c096327 a restart abandons a running analysis — left a clip
named for its species with no key frames and no log, and nothing would ever
come back to it. It read "No frame" for good.

Killed the other way round the clip still has its generic name and no sidecar
of its own, which is exactly what the sweep picks up and redoes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from animaltracker.analysis_recovery import find_unfinished_clips, has_analysis_sidecar
from animaltracker.detector import BaseDetector, Detection
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings

WIDTH, HEIGHT, FRAMES = 320, 240, 30
EPOCH = 1789000000
DOG = "mammalia_carnivora_canidae"


class AlwaysADog(BaseDetector):
    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        dets = [Detection(species=DOG, confidence=0.9, bbox=[100.0, 100.0, 180.0, 170.0], taxonomy=DOG)]
        return (dets, []) if return_filtered else dets


def make_clip(root: Path, label: str = "animal") -> Path:
    path = root / "clips" / "cam1" / "2026" / "09" / "10" / f"{EPOCH}_{label}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()
    return path


def processor(root: Path) -> ClipPostProcessor:
    return ClipPostProcessor(detector=AlwaysADog(), storage_root=root,
                             settings=build_processing_settings(None, {"sample_rate": 3}))


def names(clip: Path) -> list:
    return sorted(p.name for p in clip.parent.iterdir())


def test_the_key_frames_and_the_log_are_already_there_when_the_rename_runs(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)
    target = clip.with_name(f"{EPOCH}_{DOG}.mp4")
    seen = {}
    real = ClipPostProcessor._rename_clip

    def watched(self, clip_path, new_species):
        seen["at_rename"] = sorted(p.name for p in clip_path.parent.iterdir())
        return real(self, clip_path, new_species)

    monkeypatch.setattr(ClipPostProcessor, "_rename_clip", watched)

    result = processor(tmp_path).process_clip(clip)

    assert result.new_path == target
    assert f"{target.stem}.log.json" in seen["at_rename"]
    assert any(n.startswith(f"{target.stem}_thumb_") for n in seen["at_rename"])
    assert f"{target.stem}.mp4" not in seen["at_rename"]      # the clip itself moves last


def test_a_kill_while_the_log_is_being_written_leaves_a_clip_the_sweep_redoes(tmp_path, monkeypatch):
    """The window the bug lived in. With the rename first, the clip already
    carried its species by this point and the sweep would never look at it
    again; with the rename last, it is still generic and still recoverable."""
    clip = make_clip(tmp_path)

    def killed(self, *a, **kw):
        raise KeyboardInterrupt("the service was restarted here")

    monkeypatch.setattr(ClipPostProcessor, "_save_processing_log", killed)

    with pytest.raises(KeyboardInterrupt):
        processor(tmp_path).process_clip(clip)

    assert clip.exists()                                       # still under its generic name
    assert not has_analysis_sidecar(clip)                      # nothing claims it is finished
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == [clip]


def test_a_kill_before_the_rename_leaves_a_clip_the_sweep_redoes(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)

    def killed(self, clip_path, new_species):
        raise KeyboardInterrupt("the service was restarted here")

    monkeypatch.setattr(ClipPostProcessor, "_rename_clip", killed)

    with pytest.raises(KeyboardInterrupt):
        processor(tmp_path).process_clip(clip)

    assert clip.exists() and not has_analysis_sidecar(clip)
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == [clip]


def test_a_rename_that_cannot_happen_leaves_the_clip_for_the_sweep(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)

    def refuses(self, clip_path, new_species):
        return None                                            # e.g. the target appeared meanwhile

    monkeypatch.setattr(ClipPostProcessor, "_rename_clip", refuses)

    result = processor(tmp_path).process_clip(clip)

    assert result.success and result.new_path is None
    assert clip.exists() and not has_analysis_sidecar(clip)
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == [clip]


def test_a_finished_analysis_leaves_the_clip_its_frames_and_its_log_together(tmp_path):
    clip = make_clip(tmp_path)
    target = clip.with_name(f"{EPOCH}_{DOG}.mp4")

    result = processor(tmp_path).process_clip(clip)

    assert result.new_path == target and target.exists() and not clip.exists()
    assert has_analysis_sidecar(target)
    assert find_unfinished_clips(tmp_path / "clips", min_age_s=0) == []
    log = json.loads(target.with_suffix(".log.json").read_text())
    assert log["clip"] == target.name
    assert [t["file"] for t in log["thumbnails"]] == [
        n for n in names(target) if n.startswith(f"{target.stem}_thumb_")]
    # Nothing is left under the old name.
    assert not any(n.startswith(f"{EPOCH}_animal") for n in names(target))


def test_a_stale_key_frame_from_an_earlier_run_never_replaces_a_fresh_one(tmp_path):
    clip = make_clip(tmp_path)
    target = clip.with_name(f"{EPOCH}_{DOG}.mp4")
    stale = clip.with_name(f"{EPOCH}_animal_thumb_{DOG}_t0.jpg")
    stale.write_bytes(b"\xff\xd8\xff stale")

    result = processor(tmp_path).process_clip(clip)

    assert result.new_path == target
    fresh = target.with_name(f"{target.stem}_thumb_{DOG}_t0.jpg")
    assert fresh.exists() and not fresh.read_bytes().endswith(b"stale")
    assert not stale.exists()


def test_a_clip_that_keeps_its_name_still_gets_its_log(tmp_path):
    clip = make_clip(tmp_path, label=DOG)      # already named for what it is

    result = processor(tmp_path).process_clip(clip)

    assert result.new_path is None and clip.exists()
    assert has_analysis_sidecar(clip)


def test_the_sample_frame_fallback_reads_the_file_where_it_actually_is(tmp_path):
    """Its thumbnails are named for the clip, but the bytes come from the file:
    while the outputs are being written those are two different paths."""
    clip = make_clip(tmp_path)
    target = clip.with_name(f"{EPOCH}_{DOG}.mp4")
    proc = processor(tmp_path)

    saved = proc._extract_sample_frames(target, num_samples=2, source=clip)

    assert len(saved) == 2
    assert all(p.name.startswith(f"{target.stem}_thumb_") for p in saved)
    assert all(p.exists() and p.stat().st_size > 0 for p in saved)


# --- only a clip the pipeline named is renamed ---------------------------------------
#
# Background (bug hunt, 2026-09-19): the rename kept everything before the first
# underscore as the timestamp. A manual clip is "manual_<camera>_<epoch>.mp4", so
# that was the word "manual" and the clip became "manual_<species>.mp4": camera and
# time both gone. The archive reads the second underscore-separated part as the
# camera, so it then showed the species' first word there; a second manual clip of
# the same species could not be renamed at all; and the clip page follows a rename
# by the epoch, which no longer existed.

@pytest.mark.parametrize("name", [
    "manual_cam1_1789000000.mp4",       # what save_manual_clip writes
    "notes.mp4",
    f"{EPOCH}.mp4",                     # an epoch with no label
])
def test_a_clip_the_pipeline_did_not_name_keeps_its_name(tmp_path, name):
    clip = make_clip(tmp_path)
    kept = clip.with_name(name)
    clip.rename(kept)

    result = processor(tmp_path).process_clip(kept)

    assert result.success and result.new_path is None
    assert kept.exists()
    assert result.new_species == DOG                  # still identified...
    assert has_analysis_sidecar(kept)                 # ...and recorded under its own name
    assert any(n.startswith(f"{kept.stem}_thumb_") for n in names(kept))


def test_two_manual_clips_of_the_same_species_both_keep_their_own_identity(tmp_path):
    first = make_clip(tmp_path)
    second = make_clip(tmp_path, label="second")
    a = first.with_name("manual_cam1_1789000000.mp4")
    b = second.with_name("manual_cam2_1789000500.mp4")
    first.rename(a)
    second.rename(b)
    proc = processor(tmp_path)

    assert proc.process_clip(a).new_path is None
    assert proc.process_clip(b).new_path is None

    assert a.exists() and b.exists()
    assert has_analysis_sidecar(a) and has_analysis_sidecar(b)
