"""The live tracker forgets tracks that nothing can continue any more.

Background (bug hunt, 2026-09-19): the live ``ObjectTracker`` runs for the
life of the process and was only ever cleared when an event closed. An event
needs ``min_frames`` detections over ``min_duration`` seconds, but ByteTrack
reports a track from its second match, so a bird crossing the frame or a leaf
that flickers for a second left a track without ever starting an event. Each
keeps up to two copies of a full frame, 24 MB on the 2688x1512 cameras, and
they piled up until that camera's next event, hours later on a quiet one.
Then they supplied that event's label and key frames.
"""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from animaltracker.detector import Detection
from animaltracker.pipeline import StreamWorker
from animaltracker.tracker import ObjectTracker

DOG = "mammalia_carnivora_canidae"
BIRD = "bird"
BOX = [500.0, 500.0, 620.0, 600.0]
ELSEWHERE = [1300.0, 100.0, 1400.0, 180.0]


def det(box, species=DOG, conf=0.9):
    return Detection(species=species, confidence=conf, bbox=list(box))


def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def live_tracker() -> ObjectTracker:
    return ObjectTracker(frame_rate=15)             # as StreamWorker builds it


def see(tracker: ObjectTracker, box, times: int, species=DOG) -> int:
    """Show the tracker the same object ``times`` times; returns its track id."""
    track_id = None
    for _ in range(times):
        tracked = tracker.update([det(box, species)], frame())
        if tracked:
            track_id = next(iter(tracked))
    assert track_id is not None, "ByteTrack reports a track from its second match"
    return track_id


def idle(tracker: ObjectTracker, updates: int) -> None:
    for _ in range(updates):
        tracker.update([], frame())


def test_a_track_byte_track_has_given_up_on_is_forgotten():
    tracker = live_tracker()
    limit = tracker.tracker.max_time_lost
    flicker = see(tracker, BOX, times=3)

    idle(tracker, limit)
    assert tracker.prune_stale_tracks() == 0          # ByteTrack could still continue it
    assert flicker in tracker.tracks

    idle(tracker, 2)
    assert tracker.prune_stale_tracks() == 1
    assert tracker.tracks == {} and tracker._last_update_seen == {}


def test_forgetting_a_track_lets_go_of_its_frames():
    tracker = live_tracker()
    see(tracker, BOX, times=3)
    info = next(iter(tracker.tracks.values()))
    kept = weakref.ref(info.best_frame)
    del info

    idle(tracker, tracker.tracker.max_time_lost + 2)
    tracker.prune_stale_tracks()
    gc.collect()

    assert kept() is None


def test_a_track_that_is_still_being_seen_stays_whatever_else_goes():
    tracker = live_tracker()
    limit = tracker.tracker.max_time_lost
    gone = see(tracker, ELSEWHERE, times=3, species=BIRD)
    for _ in range(limit + 5):                        # the dog is there throughout; the bird never returns
        tracker.update([det(BOX)], frame())
    dog = next(t for t in tracker.tracks if t != gone)

    assert tracker.prune_stale_tracks() == 1

    assert list(tracker.tracks) == [dog]
    assert tracker.tracks[dog].get_best_species()[0] == DOG


def test_the_limit_can_be_given_and_a_reset_starts_the_count_again():
    tracker = live_tracker()
    see(tracker, BOX, times=3)
    idle(tracker, 5)

    assert tracker.prune_stale_tracks(max_updates_unseen=10) == 0
    assert tracker.prune_stale_tracks(max_updates_unseen=4) == 1

    tracker.reset()
    assert tracker._last_update_seen == {}
    see(tracker, BOX, times=3)
    assert tracker.prune_stale_tracks() == 0


# --- the pipeline: between events, not during one -----------------------------------

def worker_with(tracker: ObjectTracker, event_open: bool) -> StreamWorker:
    w = object.__new__(StreamWorker)
    w.camera = SimpleNamespace(id="cam1")
    w.tracker = tracker
    w.event_state = SimpleNamespace() if event_open else None
    return w


def test_between_events_every_tick_clears_out_what_went_stale():
    tracker = live_tracker()
    worker = worker_with(tracker, event_open=False)
    for _ in range(3):
        tracked = worker._tick_live_tracker([det(BOX)], frame(), 0)
    assert list(tracked) and len(tracker.tracks) == 1   # the update's result is handed back

    for i in range(tracker.tracker.max_time_lost + 2):
        assert worker._tick_live_tracker([], frame(), i) == {}

    assert tracker.tracks == {}


def test_during_an_event_an_animal_that_left_still_counts_towards_its_label():
    tracker = live_tracker()
    worker = worker_with(tracker, event_open=True)
    for _ in range(3):
        worker._tick_live_tracker([det(BOX)], frame(), 0)

    for i in range(tracker.tracker.max_time_lost + 20):
        worker._tick_live_tracker([], frame(), i)

    assert [s for s, _ in tracker.get_unique_species()] == [DOG]


def test_the_post_processor_never_prunes():
    """There, every track is part of the clip's result however early it ended."""
    import inspect

    from animaltracker import postprocess

    assert "prune_stale_tracks" not in inspect.getsource(postprocess)
