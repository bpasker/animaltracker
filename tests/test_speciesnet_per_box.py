"""SpeciesNet judges each MegaDetector box on its own crop, not the frame's top box.

Background (cam1, 2026-09-07): a dog crossed the gravel bed for 22 seconds
with a MegaDetector box in 141 sampled frames. The kids' bike on the porch is
a permanent "vehicle 0.94" box, the library ensemble labels a frame by its
highest-scoring box, and its crop classifier only ever saw the bike. 254 of
262 frames were logged ``non_animal:vehicle``; the 8 kept were the frames
where the dog happened to score 0.95.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from animaltracker.detector import NON_ANIMAL_REASON_PREFIX, SpeciesNetDetector
from animaltracker.postprocess import ClipPostProcessor, ProcessingLogEntry, ProcessingSettings

# The library's own label strings (speciesnet.constants.Classification).
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
DOG = "8f4c1e5a-0000-4000-8000-000000000001;mammalia;carnivora;canidae;canis;familiaris;domestic dog"
DEER = "8f4c1e5a-0000-4000-8000-000000000002;mammalia;cetartiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"

W, H = 1600, 1200
FRAME = np.zeros((H, W, 3), dtype=np.uint8)

# MegaDetector boxes as the library reports them: normalized [x, y, w, h].
BIKE = {"category": "3", "label": "vehicle", "conf": 0.94, "bbox": [0.25, 0.59, 0.18, 0.18]}
TRUCK = {"category": "3", "label": "vehicle", "conf": 0.50, "bbox": [0.30, 0.52, 0.12, 0.07]}
PERSON = {"category": "2", "label": "human", "conf": 0.95, "bbox": [0.33, 0.20, 0.34, 0.80]}
DOG_BOX = {"category": "1", "label": "animal", "conf": 0.90, "bbox": [0.67, 0.65, 0.04, 0.13]}
DEER_BOX = {"category": "1", "label": "animal", "conf": 0.60, "bbox": [0.10, 0.40, 0.10, 0.12]}
FAINT_BOX = {"category": "1", "label": "animal", "conf": 0.10, "bbox": [0.80, 0.80, 0.05, 0.05]}


def _key(bbox):
    return (round(bbox[0], 4), round(bbox[1], 4))


def _pixels(box):
    x, y, w, h = box["bbox"]
    return [x * W, y * H, (x + w) * W, (y + h) * H]


class FakeBBox:
    def __init__(self, xmin, ymin, width, height):
        self.xmin, self.ymin, self.width, self.height = xmin, ymin, width, height


class FakePIL:
    @staticmethod
    def fromarray(arr):
        return arr


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def preprocess(self, img):
        return img

    def predict(self, key, img):
        return {"detections": list(self.dets)}


class FakeClassifier:
    """Answers according to which box it was asked to crop to."""

    def __init__(self, answers, whole_frame=(BLANK, 0.97)):
        self.answers = answers        # {(xmin, ymin): (label, score)}
        self.whole_frame = whole_frame
        self.calls = []               # one list of crop keys per predict/batch call

    def preprocess(self, img, bboxes=None, resize=True):
        return ("crop", (round(bboxes[0].xmin, 4), round(bboxes[0].ymin, 4))) if bboxes else ("frame", None)

    def batch_predict(self, keys, imgs):
        self.calls.append([crop for _, crop in imgs])
        results = []
        for key, (_, crop) in zip(keys, imgs):
            label, score = self.whole_frame if crop is None else self.answers[crop]
            results.append({
                "filepath": key,
                "classifications": {"classes": [label, BLANK], "scores": [score, round(1 - score, 3)]},
            })
        return results

    def predict(self, key, img):
        return self.batch_predict([key], [img])[0]


class FakeEnsemble:
    """The top classifier class wins; records what each combine() was shown."""

    def __init__(self):
        self.calls = []               # one {key: detections} per combine() call

    def combine(self, filepaths, classifier_results, detector_results, geolocation_results, partial_predictions):
        self.calls.append({k: detector_results[k]["detections"] for k in filepaths})
        preds = []
        for k in filepaths:
            cls = classifier_results[k]["classifications"]
            preds.append({
                "prediction": cls["classes"][0],
                "prediction_score": cls["scores"][0],
                "prediction_source": "classifier",
                "detections": detector_results[k]["detections"],
            })
        return preds


def _detector(dets, answers=None, whole_frame=(BLANK, 0.97)) -> SpeciesNetDetector:
    det = object.__new__(SpeciesNetDetector)
    det.country = "USA"
    det.admin1_region = "MN"
    det.generic_confidence = 0.8
    det._PILImage = FakePIL
    det._BBox = FakeBBox
    det._geolocation = {"country": "USA", "admin1_region": "MN"}
    det._label_human = HUMAN
    det._label_vehicle = VEHICLE
    det._sn_detector = FakeDetector(dets)
    det._sn_classifier = FakeClassifier(answers or {}, whole_frame)
    det._sn_ensemble = FakeEnsemble()
    return det


def _infer(det):
    return det.infer(FRAME, conf_threshold=0.3, generic_confidence=0.8, return_filtered=True)


# --- the bug -------------------------------------------------------------------

def test_the_dog_beside_the_bike_is_judged_on_its_own_box():
    det = _detector([BIKE, DOG_BOX], {_key(DOG_BOX["bbox"]): (DOG, 0.79)})

    detections, filtered = _infer(det)

    assert [(d.species, round(d.confidence, 2)) for d in detections] == [("mammalia_carnivora_canidae", 0.79)]
    assert detections[0].bbox == pytest.approx(_pixels(DOG_BOX))
    # The bike is still reported, with its own box, for the person-shadow filter.
    assert [(d.species, r) for d, r in filtered] == [("vehicle", f"{NON_ANIMAL_REASON_PREFIX}vehicle")]
    assert filtered[0][0].bbox == pytest.approx(_pixels(BIKE))
    # Only the dog was cropped, in a single batch; the bike needed no classifier.
    assert det._sn_classifier.calls == [[_key(DOG_BOX["bbox"])]]


def test_every_animal_box_gets_its_own_verdict():
    det = _detector(
        [DOG_BOX, DEER_BOX],
        {_key(DOG_BOX["bbox"]): (DOG, 0.80), _key(DEER_BOX["bbox"]): (DEER, 0.85)},
    )

    detections, filtered = _infer(det)

    assert filtered == []
    assert sorted(d.species for d in detections) == ["mammalia_carnivora_canidae", "mammalia_cetartiodactyla_cervidae"]
    by_species = {d.species: d for d in detections}
    assert by_species["mammalia_carnivora_canidae"].bbox == pytest.approx(_pixels(DOG_BOX))
    assert by_species["mammalia_cetartiodactyla_cervidae"].bbox == pytest.approx(_pixels(DEER_BOX))
    # One batched classifier call, and the ensemble saw each box alone.
    assert det._sn_classifier.calls == [[_key(DOG_BOX["bbox"]), _key(DEER_BOX["bbox"])]]
    (call,) = det._sn_ensemble.calls
    assert sorted(len(dets) for dets in call.values()) == [1, 1]
    assert {dets[0]["conf"] for dets in call.values()} == {0.90, 0.60}


def test_verdicts_keep_the_detector_order():
    det = _detector([BIKE, DOG_BOX, DEER_BOX], {
        _key(DOG_BOX["bbox"]): (DOG, 0.80), _key(DEER_BOX["bbox"]): (DEER, 0.85),
    })

    detections, filtered = _infer(det)

    assert [d.species for d in detections] == ["mammalia_carnivora_canidae", "mammalia_cetartiodactyla_cervidae"]
    assert [d.species for d, _ in filtered] == ["vehicle"]


# --- people and vehicles -------------------------------------------------------

def test_a_confident_person_needs_no_classifier():
    det = _detector([PERSON])

    detections, filtered = _infer(det)

    assert detections == []
    assert [(d.species, r, round(d.confidence, 2)) for d, r in filtered] == [
        ("person", f"{NON_ANIMAL_REASON_PREFIX}person", 0.95)
    ]
    assert filtered[0][0].bbox == pytest.approx(_pixels(PERSON))
    assert det._sn_classifier.calls == []
    assert det._sn_ensemble.calls == []


def test_a_mid_confidence_vehicle_box_is_shown_to_the_classifier():
    det = _detector([TRUCK], {_key(TRUCK["bbox"]): (VEHICLE, 0.90)})

    detections, filtered = _infer(det)

    assert detections == []
    assert [(d.species, r) for d, r in filtered] == [("vehicle", f"{NON_ANIMAL_REASON_PREFIX}vehicle")]
    assert det._sn_classifier.calls == [[_key(TRUCK["bbox"])]]


def test_the_classifier_may_overrule_a_mid_confidence_detector_class():
    # The ensemble lets a confident classifier win over a "human 0.4"/"vehicle 0.5" box.
    det = _detector([TRUCK], {_key(TRUCK["bbox"]): (DEER, 0.85)})

    detections, filtered = _infer(det)

    assert filtered == []
    assert [d.species for d in detections] == ["mammalia_cetartiodactyla_cervidae"]
    assert detections[0].bbox == pytest.approx(_pixels(TRUCK))


# --- the whole-frame path ------------------------------------------------------

def test_an_empty_frame_takes_the_whole_frame_path():
    det = _detector([])

    detections, filtered = _infer(det)

    assert detections == []
    assert [(d.species, r) for d, r in filtered] == [("blank", "no_animal_detected")]
    assert det._sn_classifier.calls == [[None]]           # the full frame, no crop
    assert det._sn_ensemble.calls == [{"inmem": []}]      # combine() saw the frame's (empty) box list


def test_faint_boxes_alone_take_the_whole_frame_path():
    det = _detector([FAINT_BOX], {_key(FAINT_BOX["bbox"]): (BLANK, 0.90)})

    detections, filtered = _infer(det)

    assert detections == []
    assert [(d.species, r) for d, r in filtered] == [("blank", "no_animal_detected")]
    # As the library does it: crop to the top box, and one frame-level combine().
    assert det._sn_classifier.calls == [[_key(FAINT_BOX["bbox"])]]
    assert list(det._sn_ensemble.calls[0]) == ["inmem"]
    assert det._sn_ensemble.calls[0]["inmem"] == [FAINT_BOX]


def test_faint_boxes_do_not_get_their_own_verdict_next_to_a_real_one():
    det = _detector([DOG_BOX, FAINT_BOX], {_key(DOG_BOX["bbox"]): (DOG, 0.8)})

    detections, _ = _infer(det)

    assert [d.species for d in detections] == ["mammalia_carnivora_canidae"]
    assert det._sn_classifier.calls == [[_key(DOG_BOX["bbox"])]]


# --- limits and helpers --------------------------------------------------------

def test_crops_per_frame_are_capped():
    boxes = [
        {"category": "1", "label": "animal", "conf": 0.5, "bbox": [0.05 * i, 0.1, 0.04, 0.04]}
        for i in range(SpeciesNetDetector.PER_BOX_MAX_BOXES + 4)
    ]
    det = _detector(boxes, {_key(b["bbox"]): (DOG, 0.8) for b in boxes})

    detections, _ = _infer(det)

    assert len(detections) == SpeciesNetDetector.PER_BOX_MAX_BOXES
    assert len(det._sn_classifier.calls[0]) == SpeciesNetDetector.PER_BOX_MAX_BOXES


def test_box_kind_reads_the_label_or_the_category():
    class Label(str):
        value = "human"

    assert SpeciesNetDetector._box_kind({"label": Label("Detection.HUMAN"), "category": "1"}) == "human"
    assert SpeciesNetDetector._box_kind({"label": "vehicle"}) == "vehicle"
    assert SpeciesNetDetector._box_kind({"category": "3"}) == "vehicle"
    assert SpeciesNetDetector._box_kind({"category": "2"}) == "human"
    assert SpeciesNetDetector._box_kind({"category": "1"}) == "animal"
    assert SpeciesNetDetector._box_kind({}) == "animal"


# --- the sidecar log -----------------------------------------------------------

def test_log_summary_counts_frames_not_entries(tmp_path):
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings()
    proc.detector = object()
    entries = [
        # frame 10: the bike, the dog on its track, and a second track
        ProcessingLogEntry(frame_idx=10, event="detector_filtered", species="vehicle", confidence=0.94,
                           reason=f"{NON_ANIMAL_REASON_PREFIX}vehicle", bbox=[1, 1, 2, 2]),
        ProcessingLogEntry(frame_idx=10, event="tracked", species="canidae", confidence=0.8, track_id=1),
        ProcessingLogEntry(frame_idx=10, event="tracked", species="canidae", confidence=0.7, track_id=2),
        # frame 12: the bike again and an animal box that classified as nothing
        ProcessingLogEntry(frame_idx=12, event="detector_filtered", species="vehicle", confidence=0.94,
                           reason=f"{NON_ANIMAL_REASON_PREFIX}vehicle", bbox=[1, 1, 2, 2]),
        ProcessingLogEntry(frame_idx=12, event="detector_filtered", species="blank", confidence=0.0,
                           reason="no_animal_detected"),
        ProcessingLogEntry(frame_idx=12, event="detector_filtered", species="blank", confidence=0.0,
                           reason="no_animal_detected"),
    ]
    clip = tmp_path / "1788816926_animal.mp4"

    proc._save_processing_log(clip, entries, None, {"frames_to_analyze": 4})

    summary = json.loads((tmp_path / "1788816926_animal.log.json").read_text())["analysis_summary"]
    assert summary["frames_with_detections"] == 1
    assert summary["frames_with_non_animal"] == 2
    assert summary["frames_filtered_other"] == 2
    assert summary["frames_with_no_animal"] == 1
    assert summary["detection_rate_pct"] == 25.0
