"""One detector instance serves every camera, and the workers call it from
executor threads at once. The YOLOv5 head inside MegaDetector caches its
anchor grid per input shape, so two concurrent forward passes with
differently shaped frames (a 4:3 camera beside a 16:9 one) corrupted each
other: ~1,400 "The size of tensor a (96) must match the size of tensor b
(120)" errors on 2026-09-12. Every backend now serialises its forward pass
on BaseDetector.model_lock; these tests pin that without a GPU or model."""
import threading
import time

import numpy as np

from animaltracker.detector import BaseDetector, MegaDetectorBackend, YoloDetector


class Overlap:
    """Counts how many forward passes are in flight at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)

    def leave(self):
        with self.lock:
            self.active -= 1


class FakeSNDetector:
    """Stands in for speciesnet's SpeciesNetDetector: a shape-dependent cache
    that breaks, like the real anchor grid, when a call sees another call's
    shape."""

    def __init__(self, overlap):
        self.overlap = overlap
        self.cached_shape = None

    def preprocess(self, pil_img):
        return pil_img.shape

    def predict(self, key, preprocessed):
        self.overlap.enter()
        try:
            self.cached_shape = preprocessed
            time.sleep(0.005)
            if self.cached_shape != preprocessed:
                raise RuntimeError("The size of tensor a must match the size of tensor b")
            return {"detections": [{"conf": 0.9, "category": "1", "bbox": [0.1, 0.1, 0.2, 0.2]}]}
        finally:
            self.overlap.leave()


class FakePIL:
    @staticmethod
    def fromarray(arr):
        return arr


def megadetector(overlap):
    det = object.__new__(MegaDetectorBackend)
    det._detector = FakeSNDetector(overlap)
    det._PILImage = FakePIL
    det.model_version = "test"
    for name in ("_stage_t_prep", "_stage_t_infer", "_stage_t_post",
                 "_stage_t_prep_max", "_stage_t_infer_max", "_stage_t_post_max"):
        setattr(det, name, 0.0)
    det._stage_count = 0
    return det


def hammer(det, shapes, rounds=15):
    errors = []
    results = []

    def worker(shape):
        frame = np.zeros(shape, dtype=np.uint8)
        for _ in range(rounds):
            try:
                results.append(det.infer(frame, conf_threshold=0.5))
            except Exception as err:  # noqa: BLE001 - the test reports it
                errors.append(err)

    threads = [threading.Thread(target=worker, args=(shape,)) for shape in shapes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, results


def test_megadetector_forward_passes_never_overlap_across_frame_shapes():
    overlap = Overlap()
    det = megadetector(overlap)
    errors, results = hammer(det, [(960, 1280, 3), (768, 1280, 3), (384, 640, 3)])

    assert errors == []
    assert overlap.peak == 1, "two forward passes ran at once"
    assert overlap.calls == 45 and len(results) == 45
    assert all(len(r) == 1 and r[0].species == "animal" for r in results)


def test_megadetector_without_the_lock_would_have_collided():
    """The fake reproduces the failure mode, so the test above means something."""

    class NoLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    overlap = Overlap()
    det = megadetector(overlap)
    det.__dict__["_model_lock"] = NoLock()
    errors, _ = hammer(det, [(960, 1280, 3), (768, 1280, 3)], rounds=30)
    assert errors, "the fake no longer models the grid-cache race"


class FakeYolo:
    def __init__(self, overlap):
        self.overlap = overlap
        self.names = {0: "person"}

    def predict(self, **kwargs):
        self.overlap.enter()
        try:
            time.sleep(0.003)
            return []
        finally:
            self.overlap.leave()


def test_yolo_predict_is_serialised_too():
    overlap = Overlap()
    det = object.__new__(YoloDetector)
    det.model = FakeYolo(overlap)
    det.class_map = det.model.names
    det.animal_only = True
    errors, results = hammer(det, [(480, 640, 3), (720, 1280, 3)], rounds=10)
    assert errors == [] and len(results) == 20
    assert overlap.peak == 1


def test_model_lock_is_per_instance_and_created_once():
    a = object.__new__(MegaDetectorBackend)
    b = object.__new__(MegaDetectorBackend)
    assert a.model_lock is a.model_lock
    assert a.model_lock is not b.model_lock
    assert isinstance(BaseDetector._lock_guard, type(threading.Lock()))
