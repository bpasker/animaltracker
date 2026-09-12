"""ByteTrack's matching threshold is a cost, and one frame is not a species.

Background (cam1, 2026-09-09, clip 1788996074): a golden retriever on a leash
walked past the door twice. SpeciesNet read it as "felidae" in three of the
64 sampled frames it accepted. ObjectTracker handed supervision's ByteTrack a
minimum_matching_threshold of 0.1 believing it was an IoU floor; it is the
maximum matching *cost* (1 - IoU x confidence), so no detection ever continued
a track, every track was a single frame, the merge passes did the stitching,
and two of the cat frames survived them as their own "Cat" tracks with 0.0 s
thumbnails on the clip page.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from animaltracker.detector import Detection
from animaltracker.postprocess import ClipPostProcessor, ProcessingSettings
from animaltracker.tracker import ObjectTracker, TrackInfo

FIXTURE = Path(__file__).parent / "fixtures" / "clip_1788996074_accepted.json"
CANIDAE = "mammalia_carnivora_canidae"
FELIDAE = "mammalia_carnivora_felidae"
OLD_THRESHOLD = 0.1

DOG_BOX = [500.0, 700.0, 580.0, 760.0]
DOG_BOX_LATER = [480.0, 710.0, 560.0, 770.0]
FAR_BOX = [1300.0, 100.0, 1380.0, 160.0]


def _clip() -> dict:
    return json.loads(FIXTURE.read_text())


def _replay(**tracker_kwargs) -> ObjectTracker:
    """Feed the clip's accepted detections to a fresh tracker, sampled frame by sampled frame."""
    clip = _clip()
    video = clip["video"]
    by_frame = defaultdict(list)
    for entry in clip["log_entries"]:
        by_frame[entry["frame_idx"]].append(entry)
    tracker = ObjectTracker(
        frame_rate=video["effective_fps"],
        lost_track_buffer=clip["settings"]["lost_track_buffer"],
        **tracker_kwargs,
    )
    for frame_idx in range(0, video["total_frames"], video["actual_sample_rate"]):
        detections = [
            Detection(species=e["species"], confidence=e["confidence"], bbox=list(e["bbox"]))
            for e in by_frame.get(frame_idx, [])
        ]
        tracker.update(detections, None, frame_idx=frame_idx)
    return tracker


def _processor(**overrides) -> ClipPostProcessor:
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings.from_dict({**_clip()["settings"], **overrides})
    return proc


def _votes(tracker: ObjectTracker) -> Counter:
    return Counter(c.species for t in tracker.tracks.values() for c in t.classifications)


def _species(tracker: ObjectTracker) -> list:
    return sorted(s for s, _ in tracker.get_unique_species())


def _track(track_id: int, votes) -> TrackInfo:
    """votes: iterable of (frame_idx, species, bbox)."""
    info = TrackInfo(track_id=track_id)
    for frame_idx, species, bbox in votes:
        info.add_classification(species, 0.9, None, bbox, frame_idx)
    info.first_seen_frame = min(v[0] for v in votes)
    return info


def _tracker(*tracks: TrackInfo) -> ObjectTracker:
    tracker = object.__new__(ObjectTracker)
    tracker.tracks = {t.track_id: t for t in tracks}
    return tracker


# --- the threshold ---------------------------------------------------------------

def test_the_old_threshold_made_every_track_a_single_frame():
    # Characterises the bug so the number means something: 64 accepted
    # detections became 25 one-frame tracks, and the other 39 votes were
    # never recorded because ByteTrack only reports a track from its second
    # consecutive match.
    tracker = _replay(minimum_matching_threshold=OLD_THRESHOLD)

    assert all(len(t.classifications) == 1 for t in tracker.tracks.values())
    assert len(tracker.tracks) == 25


def test_the_default_threshold_keeps_the_dog_in_one_track_per_visit():
    tracker = _replay()

    accepted = len(_clip()["log_entries"])
    recorded = sum(len(t.classifications) for t in tracker.tracks.values())
    assert recorded >= 0.8 * accepted
    # Two visits, plus at most a parallel duplicate for the overlap merge.
    assert len(tracker.tracks) <= 4


# --- the clip end to end -----------------------------------------------------------

def test_a_dog_read_as_a_cat_for_one_frame_is_a_minority_vote():
    tracker = _replay()

    _processor()._merge_tracks(tracker)

    assert _species(tracker) == [CANIDAE]
    assert len(tracker.tracks) == 2  # the dog came by twice
    # The cat frames are still on record inside the dog's tracks, outvoted.
    assert _votes(tracker)[FELIDAE] == 3


def test_the_closing_passes_catch_the_cat_frames_even_when_the_tracker_fragments():
    # Belt and braces: with the old threshold the two recorded felidae frames
    # were tracks of their own. The late gap-fill and weak-track passes fold
    # them into the dog's tracks whatever the tracker did. The spatial pass
    # is held to overlap only (no reach), or it stitches the one-frame
    # fragments itself and the closing passes are never exercised.
    tracker = _replay(minimum_matching_threshold=OLD_THRESHOLD)

    log = _processor(spatial_merge_reach=0)._merge_tracks(tracker)

    assert _species(tracker) == [CANIDAE]
    assert len(tracker.tracks) == 2
    assert _votes(tracker)[FELIDAE] == 2
    assert any(e.event in ("gap_fill_merge", "weak_track_merge") for e in log)


# --- the weak-track pass -----------------------------------------------------------

def test_a_one_frame_cat_between_dog_frames_joins_the_dog():
    dog = _track(1, [(f, CANIDAE, DOG_BOX) for f in (186, 194, 198)])
    cat = _track(2, [(190, FELIDAE, DOG_BOX)])
    tracker = _tracker(dog, cat)

    assert tracker.merge_weak_tracks(min_detections=2) == 1
    assert set(tracker.tracks) == {1}
    assert tracker.tracks[1].get_best_species()[0] == CANIDAE
    assert [c.species for c in tracker.tracks[1].classifications].count(FELIDAE) == 1


def test_the_host_has_to_span_the_fragment():
    dog = _track(1, [(f, CANIDAE, DOG_BOX) for f in (200, 204, 208)])
    cat_before = _track(2, [(190, FELIDAE, DOG_BOX)])
    tracker = _tracker(dog, cat_before)

    assert tracker.merge_weak_tracks(min_detections=2) == 0
    assert set(tracker.tracks) == {1, 2}


def test_a_fragment_elsewhere_in_the_frame_is_left_alone():
    dog = _track(1, [(f, CANIDAE, DOG_BOX) for f in (186, 190, 194)])
    squirrel = _track(2, [(188, "mammalia_rodentia_sciuridae", FAR_BOX)])
    tracker = _tracker(dog, squirrel)

    assert tracker.merge_weak_tracks(min_detections=2) == 0
    assert set(tracker.tracks) == {1, 2}


def test_a_bird_is_not_folded_into_a_mammal():
    dog = _track(1, [(f, CANIDAE, DOG_BOX) for f in (186, 190, 194)])
    bird = _track(2, [(188, "bird_passeriformes_corvidae", DOG_BOX)])
    tracker = _tracker(dog, bird)

    assert tracker.merge_weak_tracks(min_detections=2) == 0


def test_two_fragments_do_not_absorb_each_other():
    tracker = _tracker(_track(1, [(186, CANIDAE, DOG_BOX)]),
                       _track(2, [(190, FELIDAE, DOG_BOX)]))

    assert tracker.merge_weak_tracks(min_detections=2) == 0
    assert set(tracker.tracks) == {1, 2}


# --- the late gap-fill -------------------------------------------------------------

def _dog_with_a_gap() -> TrackInfo:
    return _track(1, [(f, CANIDAE, DOG_BOX) for f in (100, 104, 108, 200, 204, 208)])


def test_a_real_visitor_inside_the_gap_keeps_its_track():
    visitor = _track(2, [(f, FELIDAE, DOG_BOX_LATER) for f in (140, 150, 160)])
    tracker = _tracker(_dog_with_a_gap(), visitor)

    assert tracker.merge_gap_filling_tracks(max_detections=1) == 0
    assert set(tracker.tracks) == {1, 2}


def test_a_one_frame_blip_inside_the_gap_is_absorbed():
    blip = _track(2, [(150, FELIDAE, DOG_BOX_LATER)])
    tracker = _tracker(_dog_with_a_gap(), blip)

    assert tracker.merge_gap_filling_tracks(max_detections=1) == 1
    assert set(tracker.tracks) == {1}
