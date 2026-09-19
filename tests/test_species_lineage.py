"""A label's specificity is its taxonomy depth, and a species is chosen by
walking down the taxonomy with the votes.

Background (bug hunt, 2026-09-19): three places scored how specific a label
is from keyword lists of their own. A family a list happened to name scored
3, an unlisted family in a listed order scored 2 and an unlisted family in an
unlisted order scored 4, and the species was taken from the highest score
before any vote was counted. So one "corvidae" frame at 0.35 renamed a track
of forty "cardinalidae" frames at 0.9, one turkey frame renamed a dog, and the
hierarchical merge, which absorbs the less specific of two tracks, folded a
skunk into a cat and a twenty-detection cow into a four-detection deer.
Across tracks the vote count was always 1 (it was read from a list already
de-duplicated by species), so a clip went to whichever track had the single
most confident frame: a two-detection cat over a dog seen 76 times.
"""
from __future__ import annotations

import pytest

from animaltracker.postprocess import ClipPostProcessor, ProcessingSettings, SpeciesResult
from animaltracker.species_names import pick_species_by_lineage, species_lineage, species_rank
from animaltracker.tracker import ObjectTracker, TrackInfo

CARDINAL = "bird_passeriformes_cardinalidae"
CROW = "bird_passeriformes_corvidae"
TURKEY = "bird_galliformes_phasianidae"
DUCK = "bird_anseriformes_anatidae"
DOG = "mammalia_carnivora_canidae"
CAT = "mammalia_carnivora_felidae"
SKUNK = "mammalia_carnivora_mephitidae"
COW = "mammalia_cetartiodactyla_bovidae"
DEER = "mammalia_cetartiodactyla_cervidae"
SNAKE = "reptile_squamata_colubridae"
CARNIVORE = "mammalia_carnivora_carnivorous mammal"
RODENT = "mammalia_rodentia_rodent"
MAMMAL = "mammalia_mammal"

FAMILIES = [CARDINAL, CROW, TURKEY, DUCK, DOG, CAT, SKUNK, COW, DEER, SNAKE]

BOX = [500.0, 700.0, 600.0, 800.0]
FAR_BOX = [1300.0, 100.0, 1400.0, 200.0]


def _track(track_id: int, votes, bbox=BOX) -> TrackInfo:
    """votes: iterable of (frame_idx, species, confidence)."""
    votes = list(votes)
    info = TrackInfo(track_id=track_id)
    for frame_idx, species, confidence in votes:
        info.add_classification(species, confidence, None, bbox, frame_idx)
    info.first_seen_frame = min(v[0] for v in votes)
    return info


def _tracker(*tracks: TrackInfo) -> ObjectTracker:
    tracker = object.__new__(ObjectTracker)
    tracker.tracks = {t.track_id: t for t in tracks}
    return tracker


def _processor() -> ClipPostProcessor:
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings()
    proc.invalid_terms = set()
    return proc


# --- the lineage a label names ---------------------------------------------

@pytest.mark.parametrize("label, lineage", [
    ("animal", ()),
    ("unknown", ()),
    ("", ()),
    ("bird", ("bird",)),
    ("mammal", ("mammalia",)),
    (MAMMAL, ("mammalia",)),
    ("reptilia_reptile", ("reptile",)),
    (RODENT, ("mammalia", "rodentia")),
    (CARNIVORE, ("mammalia", "carnivora")),
    ("bird_passeriformes_passeriformes order", ("bird", "passeriformes")),
    ("mammalia_rodentia", ("mammalia", "rodentia")),          # rollup with no common name
    (DOG, ("mammalia", "carnivora", "canidae")),
    (CARDINAL, ("bird", "passeriformes", "cardinalidae")),
    ("Mammalia_Carnivora_Canidae", ("mammalia", "carnivora", "canidae")),
    ("dog", ("dog",)),                                         # YOLO: a branch of its own
])
def test_lineage_is_read_from_the_label(label, lineage):
    assert species_lineage(label) == lineage


def test_a_rollup_is_an_ancestor_of_its_families():
    for ancestor in ("animal", MAMMAL, CARNIVORE):
        assert species_lineage(DOG)[:len(species_lineage(ancestor))] == species_lineage(ancestor)
    # ...and a rodent rollup is not an ancestor of a dog.
    assert species_lineage(DOG)[:2] != species_lineage(RODENT)


# --- one scale, used everywhere ---------------------------------------------

@pytest.mark.parametrize("label, rank", [
    ("animal", 0), ("bird", 1), (MAMMAL, 1), ("reptilia_reptile", 1),
    (RODENT, 2), (CARNIVORE, 2), ("canidae", 3),
] + [(family, 3) for family in FAMILIES])
def test_every_label_at_one_level_gets_one_rank_on_all_three_scales(label, rank):
    assert species_rank(label) == rank
    assert TrackInfo(track_id=0)._calculate_specificity(label) == rank
    assert object.__new__(ObjectTracker)._get_species_hierarchy(label)[1] == rank
    assert _processor()._calculate_specificity(label) == rank


@pytest.mark.parametrize("label, category", [
    ("animal", "animal"), (DOG, "mammal"), (MAMMAL, "mammal"), (CROW, "bird"), ("bird", "bird"),
    (SNAKE, "reptile"), ("reptilia_reptile", "reptile"), ("dog", "animal"),
])
def test_hierarchy_category_comes_from_the_class(label, category):
    assert object.__new__(ObjectTracker)._get_species_hierarchy(label)[0] == category


# --- a track's species ----------------------------------------------------------

@pytest.mark.parametrize("majority, stray", [
    (CARDINAL, CROW),     # unlisted family vs listed family, same order
    (DOG, TURKEY),        # listed family vs unlisted family in an unlisted order
    (COW, DEER),          # unlisted vs listed, same order
    (SKUNK, CAT),
])
def test_one_stray_frame_does_not_rename_a_track(majority, stray):
    votes = [(f, majority, 0.9) for f in range(0, 120, 3)] + [(120, stray, 0.35)]
    assert _track(1, votes).get_best_species()[0] == majority
    # Not even when the stray is the most confident frame of the lot.
    votes[-1] = (120, stray, 0.99)
    assert _track(1, votes).get_best_species() == (majority, 0.9, None)


def test_a_specific_label_still_beats_its_own_generic_ancestors():
    votes = ([(f, "animal", 0.9) for f in range(0, 90, 3)]
             + [(f, MAMMAL, 0.8) for f in range(90, 120, 3)]
             + [(120, CARNIVORE, 0.7), (123, DEER, 0.4), (126, DEER, 0.45)])
    # Deer contradicts the one carnivore rollup; two votes to one settle it.
    assert _track(1, votes).get_best_species()[0] == DEER

    votes = [(f, "animal", 0.9) for f in range(0, 150, 3)] + [(150, DOG, 0.4)]
    assert _track(2, votes).get_best_species()[0] == DOG


def test_a_generic_majority_beats_a_stray_from_another_class():
    votes = [(f, "bird", 0.9) for f in range(0, 120, 3)] + [(120, CAT, 0.95)]
    assert _track(1, votes).get_best_species()[0] == "bird"


def test_contradicting_labels_part_at_the_level_where_they_differ():
    # 5 cat + 4 dog are nine carnivores, against six deer: a carnivore, and
    # among the carnivores, the cat.
    picked = pick_species_by_lineage({CAT: (5, 0.6), DOG: (4, 0.9), DEER: (6, 0.99)})
    assert picked == CAT


def test_ties_fall_to_confidence_and_nothing_gives_nothing():
    assert pick_species_by_lineage({CAT: (3, 0.6), DOG: (3, 0.9)}) == DOG
    assert pick_species_by_lineage({}) == ""
    assert pick_species_by_lineage({"animal": (4, 0.5)}) == "animal"


# --- the hierarchical merge -------------------------------------------------------

@pytest.mark.parametrize("first, second", [(SKUNK, CAT), (COW, DEER)])
def test_hierarchical_merge_leaves_two_families_apart(first, second):
    tracker = _tracker(
        _track(1, [(f, first, 0.9) for f in range(0, 60, 3)]),
        _track(2, [(f, second, 0.9) for f in range(100, 112, 3)], bbox=FAR_BOX),
    )

    assert tracker.merge_hierarchical_tracks(max_frame_gap=120) == 0
    assert sorted(tracker.tracks) == [1, 2]


def test_hierarchical_merge_still_absorbs_a_generic_neighbour():
    tracker = _tracker(
        _track(1, [(f, DOG, 0.9) for f in range(0, 60, 3)]),
        _track(2, [(f, "animal", 0.9) for f in range(70, 82, 3)]),
        _track(3, [(f, MAMMAL, 0.9) for f in range(90, 99, 3)]),
    )

    assert tracker.merge_hierarchical_tracks(max_frame_gap=120) == 2
    assert list(tracker.tracks) == [1]
    assert tracker.tracks[1].get_best_species()[0] == DOG


# --- the clip's species -----------------------------------------------------------

def test_a_species_brings_every_detection_of_its_tracks():
    tracker = _tracker(
        _track(1, [(f, DOG, 0.9) for f in range(0, 120, 3)]),                  # 40
        _track(2, [(f, DOG, 0.93) for f in range(200, 308, 3)]),                # 36
        _track(3, [(400, CAT, 0.95), (403, CAT, 0.9)], bbox=FAR_BOX),
    )
    proc = _processor()

    assert proc._tracked_species_votes(tracker) == {DOG: (76, 0.93), CAT: (2, 0.95)}

    results, _log, _summary = proc._build_tracked_species_results_with_log(tracker)
    assert {s: (r.count, r.confidence) for s, r in results.items()} == {DOG: (76, 0.93), CAT: (2, 0.95)}
    assert proc._build_tracked_species_results(tracker)[DOG].count == 76
    # The two-detection cat has the most confident frame and still loses.
    assert proc._select_best_species(results) == (DOG, 0.93)


def _result(species: str, count: int, confidence: float) -> SpeciesResult:
    return SpeciesResult(species=species, confidence=confidence, count=count,
                         specificity=species_rank(species), taxonomy=None, key_frames=[])


def test_the_clip_goes_to_the_animal_that_was_seen_not_the_loudest_frame():
    proc = _processor()
    # A deer for thirty frames, and a stove lid read as a duck in four.
    results = {DEER: _result(DEER, 30, 0.80), DUCK: _result(DUCK, 4, 0.97)}
    assert proc._select_best_species(results) == (DEER, 0.80)

    # An order rollup no longer ties with a family on word count.
    results = {RODENT: _result(RODENT, 40, 0.9), CROW: _result(CROW, 2, 0.95)}
    assert proc._select_best_species(results) == (RODENT, 0.9)

    # A generic track next to the animal's own track changes nothing.
    results = {"animal": _result("animal", 50, 0.95), DOG: _result(DOG, 6, 0.5)}
    assert proc._select_best_species(results) == (DOG, 0.5)
    assert proc._select_best_species({}) == ("", 0.0)
