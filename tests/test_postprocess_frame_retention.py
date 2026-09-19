"""Post-processing must not hold the clip in memory while it analyses it.

Background (bug hunt, 2026-09-19): ``_analyze_video`` appended a full copy of
every sampled frame that had a detection to a list nothing ever read. A
2688x1512 frame is 12 MB, so a six-minute Otteson clip with the animal in
view throughout held about 33 GB for the length of the job, times
``max_concurrent_postprocess``, on a host whose memory plateaus near 40 GB.
An OOM kill leaves the clip without a sidecar, so the recovery sweep would
pick the same clip up 90 s after every start. The tracker already keeps the
best frames of each track, which is all the key frames need.
"""
from __future__ import annotations

import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest

from animaltracker.detector import BaseDetector, Detection
from animaltracker.postprocess import ClipPostProcessor, build_processing_settings

WIDTH, HEIGHT = 640, 480
FRAME_BYTES = WIDTH * HEIGHT * 3
SQUIRREL = "mammalia_rodentia_sciuridae"


class AnimalInEveryFrame(BaseDetector):
    """A detector that finds the same squirrel in every frame it is shown."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, frame, conf_threshold=0.5, generic_confidence=None, return_filtered=False):
        self.calls += 1
        detections = [Detection(species=SQUIRREL, confidence=0.9,
                                bbox=[100.0, 100.0, 180.0, 170.0], taxonomy=SQUIRREL)]
        return (detections, []) if return_filtered else detections


def _make_clip(path: Path, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v test clips")
    for i in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return path


def _peak_bytes(root: Path, frames: int) -> tuple:
    clip = _make_clip(root / "clips" / "cam1" / "2026" / "09" / "19" / f"17890{frames:05d}_animal.mp4", frames)
    detector = AnimalInEveryFrame()
    processor = ClipPostProcessor(
        detector=detector, storage_root=root,
        settings=build_processing_settings(None, {"sample_rate": 1}),
    )
    tracemalloc.start()
    try:
        result = processor.process_clip(clip, update_filename=False, regenerate_thumbnails=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, detector.calls, result


def test_memory_does_not_grow_with_the_number_of_sampled_frames(tmp_path):
    short_peak, short_calls, _ = _peak_bytes(tmp_path, 40)
    long_peak, long_calls, result = _peak_bytes(tmp_path, 160)

    # The analysis really did look at every frame and found the animal.
    assert (short_calls, long_calls) == (40, 160)
    assert result.success and SQUIRREL in result.species_results

    # 120 more sampled frames with a detection cost 110 MB when each was
    # copied into a list. A handful of frames (decode buffer, the track's
    # best frames) is all a job needs, however long the clip.
    assert long_peak - short_peak < 10 * FRAME_BYTES
    assert long_peak < 30 * FRAME_BYTES


def test_the_species_builders_take_the_tracker_alone():
    """The list fed two builder parameters that nothing read; both are gone."""
    import inspect

    for name in ("_build_tracked_species_results", "_build_tracked_species_results_with_log"):
        params = list(inspect.signature(getattr(ClipPostProcessor, name)).parameters)
        assert params == ["self", "tracker"], name
