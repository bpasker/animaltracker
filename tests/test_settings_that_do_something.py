"""Two settings-page controls that were read and then ignored.

Background (bug hunt, 2026-09-19): ``clip.post_analysis`` ("Re-run the
post-processing detector on saved clips") and ``clip.post_analysis_frames``
("Frames to analyse, 0 picks about one frame per second") were both read into
locals in ``_finalize_event_body`` that nothing used. The switch changed
nothing — only ``unified_post_processing`` was obeyed — and a long clip had no
bound on how many frames it would work through, which is what makes a job take
the best part of an hour.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.detector import BaseDetector, Detection
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings

WIDTH, HEIGHT, FRAMES = 160, 120, 90
DOG = "mammalia_carnivora_canidae"


class CountingDetector(BaseDetector):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        self.calls += 1
        dets = [Detection(species=DOG, confidence=0.9, bbox=[20.0, 20.0, 60.0, 60.0], taxonomy=DOG)]
        return (dets, []) if return_filtered else dets


def make_clip(root: Path) -> Path:
    path = root / "clips" / "cam1" / "2026" / "09" / "10" / "1789000000_animal.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return path


# --- "Frames to analyse" --------------------------------------------------------

@pytest.mark.parametrize("cap, sampled", [
    (0, 30),     # no ceiling: every third frame of 90
    (10, 10),    # every ninth frame of 90
    (7, 7),      # every thirteenth
    (3, 3),
    (500, 30),   # a ceiling above the clip changes nothing
])
def test_the_ceiling_bounds_what_a_clip_costs(tmp_path, cap, sampled):
    detector = CountingDetector()
    settings = build_processing_settings(
        SimpleNamespace(sample_rate=3, post_analysis_frames=cap))

    result = ClipPostProcessor(detector=detector, storage_root=tmp_path, settings=settings).process_clip(
        make_clip(tmp_path), update_filename=False, regenerate_thumbnails=False)

    assert detector.calls == sampled
    assert result.frames_analyzed == sampled
    assert result.success
    if cap:
        assert detector.calls <= cap       # a ceiling is never exceeded


def test_the_ceiling_comes_from_the_configured_field_and_is_recorded():
    settings = build_processing_settings(SimpleNamespace(post_analysis_frames=42))

    assert settings.max_frames == 42
    assert settings.to_dict()["max_frames"] == 42
    assert build_processing_settings(None).max_frames == 0          # no config: no ceiling


def test_a_reanalysis_can_override_the_ceiling():
    assert build_processing_settings(SimpleNamespace(post_analysis_frames=42),
                                     {"max_frames": 5}).max_frames == 5


# --- "Post-analysis" ------------------------------------------------------------

def test_the_switch_decides_whether_a_closing_event_is_analysed_at_all():
    source = inspect.getsource(pipeline_mod.StreamWorker._maybe_close_event)
    gate = source[source.index("use_unified_processor = "):]
    assert "self.runtime.general.clip.post_analysis" in gate.split("\n\n")[0]
    # ...and the local that was read and dropped on the floor is gone.
    assert "post_analysis_enabled" not in source


def test_the_recovery_sweep_obeys_the_same_switch():
    source = inspect.getsource(pipeline_mod.PipelineOrchestrator.run)
    enabled = source[source.index("enabled=lambda:"):source.index("enabled=lambda:") + 300]
    assert "post_analysis" in enabled and "recover_unfinished_clips" in enabled
