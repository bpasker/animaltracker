"""Being out of sight for a moment does not make the next animal the same one.

Background (bug hunt, 2026-09-19): ``merge_gap_filling_tracks`` absorbed any
smaller track that sat inside a time gap of a larger one. Its docstring said
so ("doesn't require IoU match, just that the smaller track is temporally
sandwiched") and the post-processor ran it unbounded, among the spatial
passes, although it had no spatial test at all. A second animal that visits
while the first is out of sight is sandwiched too: a cardinal in the far
corner disappeared into a dog with a three second gap, and the clip reported
one animal. The list of gaps was also built once per larger track, so two
fragments that overlapped each other in time could both be merged into the
same, by then filled, gap. And ``merge_spatially_adjacent_tracks`` skipped
tracks that overlap in time only when the gap was negative: tracks that share
exactly one frame, two animals side by side, were joined.
"""
from __future__ import annotations

from animaltracker.postprocess import ClipPostProcessor, ProcessingSettings
from animaltracker.tracker import ObjectTracker, TrackInfo

DOG = "mammalia_carnivora_canidae"
CAT = "mammalia_carnivora_felidae"
CARDINAL = "bird_passeriformes_cardinalidae"
DEER = "mammalia_cetartiodactyla_cervidae"

DOG_BOX = [500.0, 700.0, 600.0, 780.0]          # 100 px long
NEARBY = [560.0, 705.0, 660.0, 785.0]           # 0.6 body lengths along
FAR_CORNER = [1500.0, 80.0, 1540.0, 120.0]      # some eleven body lengths away


def track(track_id: int, species: str, frames, box) -> TrackInfo:
    info = TrackInfo(track_id=track_id)
    for f in frames:
        info.add_classification(species, 0.9, None, box, f)
    info.first_seen_frame = min(frames)
    return info


def tracker_of(*tracks: TrackInfo) -> ObjectTracker:
    t = object.__new__(ObjectTracker)
    t.tracks = {info.track_id: info for info in tracks}
    return t


def dog_with_a_gap() -> TrackInfo:
    return track(1, DOG, list(range(100, 130, 3)) + list(range(200, 230, 3)), DOG_BOX)


# --- the gap-fill ---------------------------------------------------------------------

def test_a_second_animal_elsewhere_in_the_frame_keeps_its_track():
    cardinal = track(2, CARDINAL, range(140, 173, 3), FAR_CORNER)          # 11 detections, in the gap
    t = tracker_of(dog_with_a_gap(), cardinal)

    assert t.merge_gap_filling_tracks() == 0
    assert {i: info.get_best_species()[0] for i, info in t.tracks.items()} == {1: DOG, 2: CARDINAL}


def test_a_fragment_on_the_animal_s_own_path_is_still_the_animal_whatever_it_was_read_as():
    misread = track(2, CAT, (140, 143, 146), NEARBY)
    t = tracker_of(dog_with_a_gap(), misread)

    assert t.merge_gap_filling_tracks() == 1
    assert list(t.tracks) == [1]
    assert t.tracks[1].get_best_species()[0] == DOG                        # three cat frames, outvoted


def test_a_longer_wait_allows_more_movement_but_only_so_much():
    def moved(body_lengths: float, frames, resumes_at: int):
        dx = body_lengths * 100.0
        box = [DOG_BOX[0] + dx, DOG_BOX[1], DOG_BOX[2] + dx, DOG_BOX[3]]
        return tracker_of(
            track(1, DOG, list(range(100, 130, 3)) + list(range(resumes_at, resumes_at + 30, 3)), DOG_BOX),
            track(2, DOG, frames, box),
        )

    # 2.5 body lengths away, a dozen frames after the last sighting and a
    # dozen before the next: nothing moves that far that fast...
    assert moved(2.5, (140, 143), resumes_at=157).merge_gap_filling_tracks(reach=1.0, reach_frames=30) == 0
    # ...but ninety frames on it is three times the reach, which is plausible...
    assert moved(2.5, (217, 220), resumes_at=400).merge_gap_filling_tracks(reach=1.0, reach_frames=30) == 1
    # ...and however long the wait, not from the other side of the frame.
    assert moved(9.0, (250, 253), resumes_at=400).merge_gap_filling_tracks(reach=1.0, reach_frames=30) == 0


def test_with_the_distance_test_off_only_overlap_counts():
    beside = tracker_of(dog_with_a_gap(), track(2, DOG, (140, 143), NEARBY))
    assert beside.merge_gap_filling_tracks(reach=0) == 0                   # IoU 0.23 < 0.3

    on_top = tracker_of(dog_with_a_gap(), track(2, DOG, (140, 143), [510.0, 700.0, 610.0, 780.0]))
    assert on_top.merge_gap_filling_tracks(reach=0) == 1


def test_two_fragments_that_overlap_each_other_cannot_both_fill_the_same_gap():
    t = tracker_of(
        track(1, DOG, list(range(100, 130, 3)) + list(range(300, 330, 3)), DOG_BOX),
        track(2, DOG, range(150, 201, 3), NEARBY),                          # fills 150-198
        track(3, DEER, range(180, 221, 3), NEARBY),                         # concurrent with track 2
    )

    assert t.merge_gap_filling_tracks() == 1

    assert sorted(t.tracks) == [1, 3]
    frames = [c.frame_idx for c in t.tracks[1].classifications]
    assert len(frames) == len(set(frames)), "one animal has one box per frame"


def test_tracks_without_boxes_are_merged_on_timing_alone_as_before():
    t = tracker_of(
        track(1, DOG, (100, 104, 108, 200, 204, 208), None),
        track(2, DOG, (150,), None),
    )

    assert t.merge_gap_filling_tracks() == 1


def test_the_post_processor_s_merge_keeps_both_animals():
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings()
    t = tracker_of(dog_with_a_gap(), track(2, CARDINAL, range(140, 173, 3), FAR_CORNER))

    proc._merge_tracks(t)

    assert sorted(info.get_best_species()[0] for info in t.tracks.values()) == [CARDINAL, DOG]


# --- the adjacency pass -----------------------------------------------------------------

def test_two_animals_that_share_a_frame_are_not_one_that_moved():
    doe = track(1, DEER, range(80, 105, 3), DOG_BOX)                         # last seen at frame 104
    fawn = track(2, DEER, range(104, 130, 3), NEARBY)                        # first seen at frame 104
    t = tracker_of(doe, fawn)

    assert t.merge_spatially_adjacent_tracks(iou_threshold=0.3, max_frame_gap=30, reach=1.0) == 0
    assert sorted(t.tracks) == [1, 2]


def test_a_track_that_picks_up_a_frame_later_is_still_stitched():
    first = track(1, DEER, range(80, 105, 3), DOG_BOX)
    second = track(2, DEER, range(107, 130, 3), NEARBY)
    t = tracker_of(first, second)

    assert t.merge_spatially_adjacent_tracks(iou_threshold=0.3, max_frame_gap=30, reach=1.0) == 1
