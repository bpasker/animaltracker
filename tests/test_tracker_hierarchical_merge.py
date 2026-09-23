"""The hierarchical merge only folds a track into one it is an ancestor of.

Background (bug hunt, 2026-09-22): ``merge_hierarchical_tracks`` absorbed any
less specific track of the same class within 120 frames, anywhere in the
frame, because its only species test was "both mammals". A "rodent" in one
corner and a dog elsewhere two seconds later became one track, the vote
settled it by count, and the dog was gone from the clip. The pass has no
spatial test, so the species test is the only guard it has: the generic
label must lie on the specific label's own branch.
"""
from __future__ import annotations

from animaltracker.tracker import ObjectTracker, TrackInfo

DOG = "mammalia_carnivora_canidae"


def track(track_id: int, species: str, frames, box) -> TrackInfo:
    info = TrackInfo(track_id=track_id, first_seen_frame=min(frames))
    for f in frames:
        info.add_classification(species, 0.8, None, list(box), f)
    return info


def tracker_with(*tracks: TrackInfo) -> ObjectTracker:
    tracker = ObjectTracker()
    tracker.tracks = {t.track_id: t for t in tracks}
    return tracker


def test_a_different_animal_of_the_same_class_is_not_absorbed():
    rodent = track(1, "mammalia_rodentia_rodent", range(0, 90, 3), (10, 10, 60, 60))
    dog = track(2, DOG, (150, 153, 156), (400, 300, 600, 450))
    tracker = tracker_with(rodent, dog)

    assert tracker.merge_hierarchical_tracks(max_frame_gap=120, min_specific_detections=2) == 0
    assert sorted(t.get_best_species()[0] for t in tracker.tracks.values()) == [
        DOG, "mammalia_rodentia_rodent"]


def test_the_specific_tracks_own_ancestors_are_still_absorbed():
    dog = track(1, DOG, range(0, 30, 3), (100, 100, 200, 200))
    generic = [
        track(2, "animal", (40, 43), (110, 100, 210, 200)),
        track(3, "mammalia_mammal", (60, 63), (120, 100, 220, 200)),
        track(4, "mammalia_carnivora_carnivorous mammal", (80, 83), (130, 100, 230, 200)),
    ]
    tracker = tracker_with(dog, *generic)

    assert tracker.merge_hierarchical_tracks(max_frame_gap=120, min_specific_detections=2) == 3
    assert list(tracker.tracks) == [1]
    assert tracker.tracks[1].get_best_species()[0] == DOG
