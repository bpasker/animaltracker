"""The sample rate a reanalysis is given, and the frame count it reports.

Background (bug hunt, 2026-09-19): ``_analyze_video`` treats a sample rate of
0 as "choose for me", but ``process_clip`` then worked out ``frames_analyzed``
by dividing by the configured rate. The clip page's dialog lets a 0 through,
so such a reanalysis ran to the end, most of an hour under load, and then
died on a division by zero with nothing written. Without tracking, a rate of
1 also means "choose for me", and the figure overstated what had been looked
at; the live path's false-positive gates rely on it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker.analysis_recovery import ClipAnalysisRegistry
from animaltracker.detector import BaseDetector, Detection
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings
from animaltracker.web import WebServer

WIDTH, HEIGHT, FRAMES = 320, 240, 45
DOG = "mammalia_carnivora_canidae"


class CountingDetector(BaseDetector):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        self.calls += 1
        dets = [Detection(species=DOG, confidence=0.9, bbox=[100.0, 100.0, 180.0, 170.0], taxonomy=DOG)]
        return (dets, []) if return_filtered else dets


def make_clip(root: Path) -> Path:
    path = root / "clips" / "cam1" / "2026" / "09" / "10" / "1789000000_animal.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 5) % 255, dtype=np.uint8))
    writer.release()
    return path


def analyse(tmp_path, **overrides):
    detector = CountingDetector()
    settings = build_processing_settings(None, overrides)
    result = ClipPostProcessor(detector=detector, storage_root=tmp_path, settings=settings).process_clip(
        make_clip(tmp_path), update_filename=False, regenerate_thumbnails=False)
    return result, detector


def test_a_rate_of_zero_means_automatic_all_the_way_through(tmp_path):
    result, detector = analyse(tmp_path, sample_rate=0)

    assert result.success and result.new_species == DOG
    assert detector.calls == 15                       # every third of 45 frames
    assert result.frames_analyzed == 15


@pytest.mark.parametrize("overrides, sampled", [
    ({"sample_rate": 5}, 9),
    ({"sample_rate": 1}, 45),
    ({"sample_rate": 1, "tracking_enabled": False}, 3),     # without tracking, 1 is "about one a second"
])
def test_frames_analyzed_is_what_the_detector_was_shown(tmp_path, overrides, sampled):
    result, detector = analyse(tmp_path, **overrides)

    assert detector.calls == sampled
    assert result.frames_analyzed == sampled


@pytest.mark.parametrize("overrides, names", [
    ({"sample_rate": -1}, "sample_rate"),
    ({"sample_rate": 2.5}, "sample_rate"),
    ({"sample_rate": "fast"}, "sample_rate"),
    ({"sample_rate": True}, "sample_rate"),
    ({"confidence_threshold": 30}, "confidence_threshold"),   # a percentage, not a fraction
    ({"generic_confidence": None}, "generic_confidence"),
])
def test_values_an_analysis_could_only_fail_on_are_refused_up_front(overrides, names):
    with pytest.raises(ValueError, match=names):
        build_processing_settings(None, overrides)


def test_the_configuration_s_own_values_pass():
    clip_cfg = SimpleNamespace(sample_rate=3, post_analysis_confidence=0.3, post_analysis_generic_confidence=0.5)

    settings = build_processing_settings(clip_cfg)

    assert (settings.sample_rate, settings.confidence_threshold, settings.generic_confidence) == (3, 0.3, 0.5)


def test_the_clip_page_gets_a_400_at_once_not_a_500_an_hour_later(tmp_path):
    class JsonRequest:
        def __init__(self, body) -> None:
            self._body = body

        async def json(self):
            return self._body

    clip = make_clip(tmp_path)
    registry = ClipAnalysisRegistry()
    server = WebServer({}, tmp_path, tmp_path / "logs", port=0, analysis_registry=registry)
    rel = str(clip.relative_to(tmp_path / "clips"))

    resp = asyncio.run(server.handle_reprocess(JsonRequest({"path": rel, "settings": {"sample_rate": -3}})))

    assert resp.status == 400 and "sample_rate" in resp.text
    assert not registry.is_active(clip) and server.reprocessing_jobs == {}
