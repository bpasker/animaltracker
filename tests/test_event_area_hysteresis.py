"""The minimum-area gate steps aside for the subject of an open event.

Pinned on the Otteson2 squirrel clip of 2026-09-12 13:11 (1789236706): the
squirrel walked in with a 0.61% box, which started the event, then sat down
at 0.45-0.49% of the 2688x1512 frame, under the 0.5% ``min_detection_area``
gate. Every realtime detection for the next 10 s was discarded, the event
closed on ``post_seconds`` with the squirrel in full view, and the clip
ended with it still sitting there. The boxes below are what the production
MegaDetector produced on that saved clip.
"""
import numpy as np

from animaltracker.detector import Detection
from animaltracker.pipeline import EventState, StreamWorker


class _FakeThresholds:
    min_detection_area = 0.005
    tracking_min_detection_area = 0.0005


class _FakeCamera:
    id = "Otteson2"
    thresholds = _FakeThresholds()


def _worker(event=None):
    w = object.__new__(StreamWorker)
    w.camera = _FakeCamera()
    w.ptz_tracker = None
    w.ptz_drives_tracking = False
    w.event_state = event
    return w


def _event(*bboxes):
    ev = EventState(camera=_FakeCamera(), start_ts=0.0, species=set(),
                    max_confidence=0.0, last_detection_ts=0.0)
    ev.last_accepted_bboxes = [list(map(float, b)) for b in bboxes]
    return ev


def _det(bbox, conf=0.8):
    return Detection(species="animal", confidence=conf, bbox=list(map(float, bbox)))


W, H = 2688, 1512
FRAME = np.zeros((8, 8, 3), dtype=np.uint8)

# Frame 297: walking, 212x117 px = 0.61% of the frame, passes the gate on its own.
WALKING = [2083, 976, 2295, 1093]
# Frame 309 onwards: sitting, 168x119 px = 0.49%, under the 0.5% gate.
SITTING = [2084, 979, 2252, 1098]


def test_sitting_subject_is_kept_while_its_event_is_open():
    kept = _worker(_event(WALKING))._filter_false_positives([_det(SITTING)], W, H)
    assert [d.bbox for d in kept] == [_det(SITTING).bbox]


def test_entry_gate_is_unchanged_without_an_event():
    assert _worker(None)._filter_false_positives([_det(SITTING)], W, H) == []


def test_small_box_elsewhere_in_the_frame_is_still_dropped():
    leaf = [200, 200, 368, 319]  # the sitting squirrel's size, other side of the yard
    assert _worker(_event(WALKING))._filter_false_positives([_det(leaf)], W, H) == []


def test_box_under_the_relaxed_floor_is_dropped_even_on_the_subject():
    speck = [2150, 1020, 2180, 1050]  # 900 px^2, under 0.05% of the frame
    assert _worker(_event(WALKING))._filter_false_positives([_det(speck)], W, H) == []


def test_leaf_shape_on_the_subject_is_still_dropped():
    twig = [2000, 1030, 2400, 1060]  # 13:1 and overlapping the last box
    assert _worker(_event(WALKING))._filter_false_positives([_det(twig)], W, H) == []


def test_continuity_reaches_one_body_length_and_no_further():
    last = [[100.0, 100.0, 200.0, 150.0]]  # 100 wide, 50 tall
    inside = [280.0, 180.0, 300.0, 200.0]   # centre (290, 190)
    outside = [300.0, 180.0, 320.0, 200.0]  # centre (310, 190)
    assert StreamWorker._continues_event_subject(inside, last)
    assert not StreamWorker._continues_event_subject(outside, last)


def test_event_update_records_the_accepted_boxes():
    ev = _event()
    ev.update([_det(WALKING)], 1.0, FRAME, frame_idx=297)
    assert ev.last_accepted_bboxes == [_det(WALKING).bbox]
    ev.update([], 2.0, FRAME, frame_idx=298)  # must not forget where the subject was
    assert ev.last_accepted_bboxes == [_det(WALKING).bbox]


def test_squirrel_clip_replay_keeps_the_event_alive_through_the_sit():
    """Replays the production detector's boxes for the live part of the clip.

    Each accepted frame becomes the reference for the next, so the chain has
    to survive a frame with no box (348) and a run under the 0.5 confidence
    gate (405-432) that the detector never reports.
    """
    live = [  # (frame, confidence, bbox) from the realtime MegaDetector
        (300, 0.80, [2085, 978, 2172, 1090]),
        (303, 0.50, [2084, 978, 2172, 1091]),
        (309, 0.70, [2084, 979, 2252, 1097]),
        (327, 0.58, [2083, 979, 2252, 1097]),
        (345, 0.67, [2084, 981, 2252, 1096]),
        (351, 0.67, [2083, 983, 2251, 1097]),
        (360, 0.68, [2084, 982, 2251, 1095]),
        (402, 0.81, [2083, 982, 2251, 1098]),
        (435, 0.52, [2084, 984, 2251, 1093]),
        (456, 0.60, [2084, 984, 2251, 1094]),
    ]
    ev = _event()
    ev.update([_det(WALKING, 0.78)], 297 / 30, FRAME, frame_idx=297)
    worker = _worker(ev)
    for frame_idx, conf, bbox in live:
        kept = worker._filter_false_positives([_det(bbox, conf)], W, H)
        assert kept, f"frame {frame_idx} was dropped"
        ev.update(kept, frame_idx / 30, FRAME, frame_idx=frame_idx)
    assert ev.last_detection_ts == 456 / 30
