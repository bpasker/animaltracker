"""One post-processing detector for the whole process, not one per camera.

Background (bug hunt, 2026-09-19): ``StreamWorker._get_postprocess_detector``
cached the SpeciesNet it built on the worker, so every camera that closed an
event loaded a copy of its own. With three cameras that was three models on
an 8 GB GPU that real-time detection already fills, and in host memory, while
at most ``max_concurrent_postprocess`` jobs ever run. Each camera's first
clip after a restart also paid its own load, which delayed that alert and
reopened the window in which a model load breaks a real-time forward pass on
another thread (BUGS.md item 5).
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.pipeline import StreamWorker


@pytest.fixture(autouse=True)
def no_detector_yet():
    StreamWorker._shared_postprocess_detector = None
    yield
    StreamWorker._shared_postprocess_detector = None


def worker(camera_id: str) -> StreamWorker:
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(id=camera_id)
    w.runtime = SimpleNamespace(general=SimpleNamespace(detector=SimpleNamespace(postprocess_backend="speciesnet")))
    return w


def count_builds(monkeypatch, load_seconds: float = 0.0) -> list:
    built: list = []

    def build(detector_cfg):
        time.sleep(load_seconds)            # a model load takes a while
        detector = SimpleNamespace(backend_name="speciesnet", number=len(built) + 1)
        built.append(detector)
        return detector

    monkeypatch.setattr(pipeline_mod, "create_postprocess_detector", build)
    return built


def test_every_camera_gets_the_same_instance_and_it_is_built_once(monkeypatch):
    built = count_builds(monkeypatch)
    cameras = [worker("cam1"), worker("cam2"), worker("otteson2")]

    detectors = [w._get_postprocess_detector() for w in cameras]
    detectors += [w._get_postprocess_detector() for w in cameras]

    assert len(built) == 1
    assert all(d is built[0] for d in detectors)


def test_cameras_that_close_their_first_events_together_still_load_it_once(monkeypatch):
    built = count_builds(monkeypatch, load_seconds=0.05)
    cameras = [worker(f"cam{i}") for i in range(8)]
    got: list = []
    start = threading.Barrier(len(cameras))

    def first_event(w: StreamWorker) -> None:
        start.wait(5)
        got.append(w._get_postprocess_detector())

    threads = [threading.Thread(target=first_event, args=(w,)) for w in cameras]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(built) == 1 and len(got) == len(cameras)
    assert all(d is built[0] for d in got)


def test_a_load_that_fails_is_tried_again_by_the_next_job(monkeypatch):
    attempts: list = []

    def build(detector_cfg):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return SimpleNamespace(backend_name="speciesnet")

    monkeypatch.setattr(pipeline_mod, "create_postprocess_detector", build)
    w = worker("cam1")

    with pytest.raises(RuntimeError):
        w._get_postprocess_detector()
    assert StreamWorker._shared_postprocess_detector is None     # nothing half-built is kept

    assert w._get_postprocess_detector().backend_name == "speciesnet"
    assert len(attempts) == 2
