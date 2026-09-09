"""The person-shadow filter: an animal label sitting on a person IS the person.

Background (cam1, 2026-09-08): someone sat on the front step for ten minutes.
SpeciesNet said "person" in 331 of 420 sampled frames and "myocastoridae" in a
handful, all with the same box. The regional blocklist threw the persons away
as exotic primates, the handful won the vote, and a "Rodent" alert went out.
Four clips, four alerts, one person.
"""
from __future__ import annotations

from animaltracker.detector import NON_ANIMAL_REASON_PREFIX, SpeciesNetDetector
from animaltracker.postprocess import (
    ClipPostProcessor,
    ProcessingLogEntry,
    ProcessingSettings,
    SpeciesResult,
    _bbox_iou,
)
from animaltracker.tracker import ObjectTracker, TrackInfo

HUMAN_TAXONOMY = "990ae9dd-1234-4567-89ab-0123456789ab;mammalia;primates;hominidae;homo;sapiens;human"
PERSON_BOX = [526.0, 236.0, 1064.0, 1200.0]          # the seated person
PERSON_BOX_SHIFTED = [540.0, 250.0, 1080.0, 1210.0]  # the same person a moment later
DOG_BOX = [1400.0, 900.0, 1700.0, 1180.0]            # a dog on a leash, well apart


def _detector() -> SpeciesNetDetector:
    det = object.__new__(SpeciesNetDetector)
    det.country = "USA"
    return det


def _processor(**overrides) -> ClipPostProcessor:
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings(**overrides)
    return proc


def _tracker(*tracks: TrackInfo) -> ObjectTracker:
    tracker = object.__new__(ObjectTracker)
    tracker.tracks = {t.track_id: t for t in tracks}
    return tracker


def _track(track_id: int, votes) -> TrackInfo:
    """votes: iterable of (frame_idx, species, bbox)."""
    info = TrackInfo(track_id=track_id)
    for frame_idx, species, bbox in votes:
        info.add_classification(species, 0.9, None, bbox, frame_idx)
    info.first_seen_frame = min(v[0] for v in votes)
    return info


def _person_boxes(frames, box=PERSON_BOX):
    return {f: [box] for f in frames}


# --- detector -----------------------------------------------------------------

def test_the_regional_blocklist_would_call_a_human_exotic():
    # This was the bug: humans are primates and primates are blocklisted for
    # North America, so every "person" frame was discarded as impossible. The
    # non-animal check therefore has to run before the blocklist.
    assert _detector()._is_exotic_species("person", HUMAN_TAXONOMY)


def test_non_animal_labels_are_recognised():
    assert SpeciesNetDetector._non_animal_label("person") == "person"
    assert SpeciesNetDetector._non_animal_label("Human") == "person"
    assert SpeciesNetDetector._non_animal_label("vehicle") == "vehicle"
    assert SpeciesNetDetector._non_animal_label("bird_passeriformes_corvidae") is None
    assert SpeciesNetDetector._non_animal_label("animal") is None
    assert SpeciesNetDetector._non_animal_label("") is None


def test_reason_prefix_names_the_label():
    assert f"{NON_ANIMAL_REASON_PREFIX}person" == "non_animal:person"


def test_box_follows_the_class_the_label_names():
    boxes = [
        {"category": "2", "label": "human", "conf": 0.95, "bbox": [0.1, 0.1, 0.2, 0.5]},
        {"category": "1", "label": "animal", "conf": 0.70, "bbox": [0.6, 0.6, 0.2, 0.2]},
    ]
    assert SpeciesNetDetector._pick_detection_box(boxes, "mammalia_carnivora_canidae")["conf"] == 0.70
    assert SpeciesNetDetector._pick_detection_box(boxes, "person")["conf"] == 0.95
    # No box of the wanted class: the most confident box of any class.
    assert SpeciesNetDetector._pick_detection_box(boxes[:1], "bird")["conf"] == 0.95


# --- tracker-level filter ------------------------------------------------------

def test_a_track_sitting_on_the_person_is_dropped():
    person = _person_boxes(range(0, 400, 2))
    shadow = _track(8, [(220, "mammalia_rodentia_rodent", PERSON_BOX_SHIFTED),
                        (270, "mammalia_rodentia_myocastoridae", PERSON_BOX)])
    tracker = _tracker(shadow)

    removed, log = _processor()._drop_person_shadow_tracks(tracker, person)

    assert removed == 1
    assert tracker.tracks == {}
    assert log[0].event == "track_filtered"
    assert "person shadow" in log[0].reason
    assert log[0].track_id == 8


def test_a_dog_next_to_its_walker_survives():
    person = _person_boxes(range(0, 400, 2))
    dog = _track(4, [(f, "mammalia_carnivora_canidae", DOG_BOX) for f in range(192, 420, 12)])
    tracker = _tracker(dog)

    removed, _ = _processor()._drop_person_shadow_tracks(tracker, person)

    assert removed == 0
    assert 4 in tracker.tracks


def test_only_the_shadow_track_goes_when_both_are_present():
    person = _person_boxes(range(0, 400, 2))
    dog = _track(4, [(f, "mammalia_carnivora_canidae", DOG_BOX) for f in range(192, 420, 12)])
    shadow = _track(1, [(94, "animal", PERSON_BOX)])
    tracker = _tracker(dog, shadow)

    removed, _ = _processor()._drop_person_shadow_tracks(tracker, person)

    assert removed == 1
    assert set(tracker.tracks) == {4}


def test_a_person_seen_only_far_away_in_time_does_not_count():
    # The person left at frame 100; an animal on the same spot at frame 400 is real.
    person = _person_boxes(range(0, 100, 2))
    later = _track(2, [(400, "mammalia_carnivora_felidae", PERSON_BOX)])
    tracker = _tracker(later)

    removed, _ = _processor()._drop_person_shadow_tracks(tracker, person)

    assert removed == 0


def test_min_fraction_is_honoured():
    person = _person_boxes(range(0, 400, 2))
    mostly_dog = _track(3, [(f, "mammalia_carnivora_canidae", DOG_BOX) for f in range(0, 300, 10)]
                          + [(310, "mammalia_carnivora_canidae", PERSON_BOX)])
    tracker = _tracker(mostly_dog)

    removed, _ = _processor()._drop_person_shadow_tracks(tracker, person)

    assert removed == 0


def test_filter_can_be_switched_off_by_settings():
    proc = _processor(person_shadow_enabled=False)
    assert proc.settings.person_shadow_enabled is False
    # The processor consults the flag before calling the filter; the filter
    # itself is unconditional, which keeps it testable on its own.
    person = _person_boxes(range(0, 400, 2))
    tracker = _tracker(_track(8, [(220, "mammalia_rodentia_myocastoridae", PERSON_BOX)]))
    removed, _ = proc._drop_person_shadow_tracks(tracker, person)
    assert removed == 1


# --- per-frame fallback --------------------------------------------------------

def test_per_frame_votes_are_recounted_without_shadows():
    person = _person_boxes(range(0, 400, 2))
    results = {
        "bird_passeriformes_corvidae": SpeciesResult(
            species="bird_passeriformes_corvidae", confidence=0.9, count=2,
            specificity=3, taxonomy=None, key_frames=[]),
        "mammalia_carnivora_canidae": SpeciesResult(
            species="mammalia_carnivora_canidae", confidence=0.9, count=3,
            specificity=3, taxonomy=None, key_frames=[]),
    }
    log = [
        ProcessingLogEntry(frame_idx=156, event="accepted", species="bird_passeriformes_corvidae",
                           confidence=0.9, bbox=PERSON_BOX),
        ProcessingLogEntry(frame_idx=158, event="accepted", species="bird_passeriformes_corvidae",
                           confidence=0.9, bbox=PERSON_BOX_SHIFTED),
        ProcessingLogEntry(frame_idx=200, event="accepted", species="mammalia_carnivora_canidae",
                           confidence=0.9, bbox=DOG_BOX),
        ProcessingLogEntry(frame_idx=210, event="accepted", species="mammalia_carnivora_canidae",
                           confidence=0.9, bbox=DOG_BOX),
        ProcessingLogEntry(frame_idx=212, event="accepted", species="mammalia_carnivora_canidae",
                           confidence=0.9, bbox=PERSON_BOX),
    ]

    dropped = _processor()._drop_person_shadow_species(results, log, person)

    assert dropped == ["bird_passeriformes_corvidae"]
    assert set(results) == {"mammalia_carnivora_canidae"}
    assert results["mammalia_carnivora_canidae"].count == 2


def test_settings_round_trip_carries_the_shadow_knobs():
    s = ProcessingSettings(person_shadow_iou=0.7, person_shadow_window=30,
                           person_shadow_min_fraction=0.8)
    again = ProcessingSettings.from_dict(s.to_dict())
    assert (again.person_shadow_enabled, again.person_shadow_iou,
            again.person_shadow_window, again.person_shadow_min_fraction) == (True, 0.7, 30, 0.8)


def test_settings_defaults_survive_an_old_dict():
    again = ProcessingSettings.from_dict({"sample_rate": 2})
    assert again.person_shadow_enabled is True
    assert again.person_shadow_iou == 0.6


def test_bbox_iou():
    assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(_bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert _bbox_iou(None, [0, 0, 1, 1]) == 0.0
    assert _bbox_iou([0, 0, 1], [0, 0, 1, 1]) == 0.0
