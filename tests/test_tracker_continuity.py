"""A track that starts where the last one ended is the same animal, IoU or not.

Background (cam1, 2026-09-11, clip 1789166124): a dog on a leash walked away
from the door beside its owner. ByteTrack held it as one track while it was
close, then missed it for six sampled frames as it shrank into the distance.
Its constant-velocity prediction kept sliding left while the dog had turned
right, so when the detector found the dog again it became a second track,
and SpeciesNet, now looking at a 60-pixel crop, called that track a squirrel
twice and a deer once. The spatial merge compared the first track's last box
with the second track's first box: the centres were a third of a body length
apart, but the boxes had shrunk so much that their IoU was 0.34, under the
configured 0.6, and the clip page showed a dog and a squirrel.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from animaltracker.detector import Detection
from animaltracker.postprocess import ClipPostProcessor, ProcessingSettings
from animaltracker.tracker import ObjectTracker, TrackInfo

FIXTURE = Path(__file__).parent / "fixtures" / "clip_1789166124_accepted.json"
CANIDAE = "mammalia_carnivora_canidae"
SCIURIDAE = "mammalia_rodentia_sciuridae"
CERVIDAE = "mammalia_artiodactyla_cervidae"

# A 100 px box and the boxes an animal could occupy a moment later.
BOX = [500.0, 700.0, 600.0, 800.0]
BOX_SHIFTED_HALF = [550.0, 700.0, 650.0, 800.0]      # centre 0.5 body lengths on, IoU 0.33
BOX_SHIFTED_ONE = [600.0, 700.0, 700.0, 800.0]       # exactly one body length on, IoU 0
BOX_FAR = [900.0, 700.0, 1000.0, 800.0]              # four body lengths on
BOX_SHRUNK = [530.0, 720.0, 570.0, 760.0]            # same centre, 40 px: IoU 0.16


def _clip() -> dict:
    return json.loads(FIXTURE.read_text())


def _replay() -> ObjectTracker:
    clip = _clip()
    video = clip["video"]
    by_frame = defaultdict(list)
    for entry in clip["log_entries"]:
        by_frame[entry["frame_idx"]].append(entry)
    tracker = ObjectTracker(
        frame_rate=video["effective_fps"],
        lost_track_buffer=clip["settings"]["lost_track_buffer"],
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


# --- the clip ------------------------------------------------------------------------

def test_the_tracker_splits_the_dog_when_it_walks_away():
    # Characterises the fragmentation the merge has to repair: the dog is one
    # track on the porch and another, read as a squirrel, in the distance.
    tracker = _replay()

    assert len(tracker.tracks) == 2
    assert _species(tracker) == [CANIDAE, SCIURIDAE]


def test_the_dog_that_walked_away_is_one_dog():
    tracker = _replay()

    log = _processor()._merge_tracks(tracker)

    assert _species(tracker) == [CANIDAE]
    assert len(tracker.tracks) == 1
    assert any(e.event == "spatial_merge" for e in log)
    # The far frames are still on record inside the dog's track, outvoted.
    assert _votes(tracker)[SCIURIDAE] == 2
    assert _votes(tracker)[CERVIDAE] == 1


def test_overlap_alone_left_the_squirrel_standing():
    # The configured IoU floor of 0.6 is what the clip was processed with;
    # the fragments overlapped by 0.34, so without the reach test they stay
    # apart and the squirrel keeps its thumbnail.
    tracker = _replay()

    _processor(spatial_merge_reach=0)._merge_tracks(tracker)

    assert _species(tracker) == [CANIDAE, SCIURIDAE]


# --- the reach test ----------------------------------------------------------------

def test_a_track_starting_within_one_body_length_continues_the_last():
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    later = _track(2, [(118, SCIURIDAE, BOX_SHIFTED_ONE), (122, SCIURIDAE, BOX_SHIFTED_ONE)])
    tracker = _tracker(dog, later)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=1.0) == 1
    assert set(tracker.tracks) == {1}
    assert tracker.tracks[1].last_seen_frame == 122


def test_a_shrinking_box_with_the_same_centre_is_the_same_animal():
    # Walking straight away from the camera: no move, little overlap.
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    smaller = _track(2, [(116, CERVIDAE, BOX_SHRUNK), (120, CERVIDAE, BOX_SHRUNK)])
    tracker = _tracker(dog, smaller)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=1.0) == 1
    assert set(tracker.tracks) == {1}


def test_reach_is_measured_in_body_lengths():
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    later = _track(2, [(118, SCIURIDAE, BOX_SHIFTED_ONE), (122, SCIURIDAE, BOX_SHIFTED_ONE)])
    tracker = _tracker(dog, later)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=0.5) == 0
    assert set(tracker.tracks) == {1, 2}


def test_a_track_starting_far_away_keeps_its_own_identity():
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    other = _track(2, [(118, SCIURIDAE, BOX_FAR), (122, SCIURIDAE, BOX_FAR)])
    tracker = _tracker(dog, other)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=1.0) == 0
    assert set(tracker.tracks) == {1, 2}


def test_the_frame_gap_still_bounds_the_reach():
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    later = _track(2, [(160, SCIURIDAE, BOX_SHIFTED_HALF), (164, SCIURIDAE, BOX_SHIFTED_HALF)])
    tracker = _tracker(dog, later)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=1.0) == 0


def test_reach_zero_leaves_overlap_as_the_only_test():
    dog = _track(1, [(100, CANIDAE, BOX), (104, CANIDAE, BOX)])
    later = _track(2, [(118, SCIURIDAE, BOX_SHIFTED_HALF), (122, SCIURIDAE, BOX_SHIFTED_HALF)])
    tracker = _tracker(dog, later)

    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.6, max_frame_gap=30, reach=0) == 0
    assert tracker.merge_spatially_adjacent_tracks(iou_threshold=0.3, max_frame_gap=30, reach=0) == 1


# --- the setting -------------------------------------------------------------------

def test_the_reach_setting_round_trips_and_defaults():
    settings = ProcessingSettings(spatial_merge_reach=0.75)

    assert ProcessingSettings.from_dict(settings.to_dict()).spatial_merge_reach == 0.75
    # Sidecars written before the setting existed replay with the default.
    assert ProcessingSettings.from_dict({"spatial_merge_iou": 0.6}).spatial_merge_reach == 1.0
