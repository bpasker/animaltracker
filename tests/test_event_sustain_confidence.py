"""A weaker box keeps an open event going where its animal was just seen.

Pinned on jessDahuaBack, 2026-10-01 22:30-22:50: a cottontail sat on the
lawn under IR for twenty minutes and came out as a dozen clips. The live
MegaDetector scored it 0.5-0.65 on a few frames and 0.2-0.45 on most, so
every run of 10 s under the camera's 0.5 ``confidence`` closed the event
on ``post_seconds`` and the next 0.5 frame opened another.

``REPLAY`` is what the production MegaDetector returned (at 0.15 and up, on
the live path's 1280-wide frames, every 8th source frame) for 20-50 s of
clip 1790912064: the rabbit at about (1770, 1190), and a lawn feature at
(1721, 777) that scores up to 0.45 all night and must not count.
"""
import numpy as np

from animaltracker.detector import Detection
from animaltracker.pipeline import SUSTAIN_ANCHOR_S, EventState, StreamWorker


class _Thresholds:
    confidence = 0.5
    sustain_confidence = 0.2
    min_detection_area = 0.001
    tracking_min_detection_area = 0.0005


class _Camera:
    id = "jessDahuaBack"
    thresholds = _Thresholds()
    include_species = []
    exclude_species = []


class _General:
    exclusion_list = []


class _Runtime:
    general = _General()


class _Detector:
    single_threshold = True


def _worker(event=None, detector=None):
    w = object.__new__(StreamWorker)
    w.camera = _Camera()
    w.runtime = _Runtime()
    w.detector = detector or _Detector()
    w.ptz_tracker = None
    w.ptz_drives_tracking = False
    w.event_state = event
    return w


def _event(ts=0.0):
    return EventState(camera=_Camera(), start_ts=ts, species=set(),
                      max_confidence=0.0, last_detection_ts=ts)


def _det(bbox, conf):
    return Detection(species="animal", confidence=conf, bbox=list(map(float, bbox)))


W, H = 2688, 1520
FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
RABBIT = [1734, 1148, 1812, 1230]
LAWN = [1457, 688, 1986, 867]

REPLAY = [  # (seconds into the clip, [(confidence, pixel box), ...])
    (20.0, [(0.58, [1734, 1148, 1812, 1230]), (0.2, [1457, 688, 1986, 867])]),
    (20.3, [(0.6, [1732, 1148, 1814, 1230])]),
    (20.5, [(0.57, [1734, 1148, 1812, 1230]), (0.18, [1457, 688, 1986, 867])]),
    (20.8, [(0.57, [1734, 1146, 1812, 1230])]),
    (21.1, [(0.55, [1732, 1148, 1814, 1230])]),
    (21.3, [(0.64, [1734, 1148, 1812, 1230]), (0.16, [1457, 688, 1988, 867])]),
    (21.6, [(0.65, [1734, 1148, 1812, 1230]), (0.3, [1457, 688, 1988, 867])]),
    (21.9, [(0.62, [1734, 1148, 1814, 1230]), (0.22, [1457, 688, 1988, 867])]),
    (22.1, [(0.6, [1734, 1148, 1831, 1230]), (0.29, [1457, 690, 1986, 867])]),
    (22.4, [(0.27, [1734, 1146, 1854, 1241]), (0.23, [1457, 690, 1986, 867])]),
    (22.7, [(0.34, [1457, 690, 1986, 867])]),
    (22.9, []),
    (23.2, [(0.39, [1728, 1148, 1833, 1236])]),
    (23.5, [(0.25, [1726, 1148, 1854, 1243]), (0.2, [1455, 690, 1986, 867])]),
    (23.7, [(0.33, [1728, 1148, 1835, 1239])]),
    (24.0, [(0.32, [1728, 1148, 1833, 1236]), (0.24, [1455, 690, 1986, 867])]),
    (24.3, [(0.39, [1726, 1148, 1833, 1236])]),
    (24.5, [(0.22, [1728, 1148, 1833, 1236])]),
    (24.8, [(0.19, [1728, 1148, 1835, 1239])]),
    (25.1, [(0.27, [1728, 1148, 1831, 1236])]),
    (25.3, [(0.16, [1730, 1148, 1835, 1239])]),
    (25.6, [(0.36, [1728, 1148, 1845, 1236])]),
    (25.9, [(0.26, [1728, 1148, 1848, 1239])]),
    (26.1, [(0.24, [1728, 1148, 1843, 1239])]),
    (26.4, [(0.23, [1728, 1148, 1835, 1239])]),
    (26.7, [(0.2, [1728, 1148, 1839, 1241])]),
    (26.9, [(0.18, [1726, 1148, 1839, 1243])]),
    (27.2, [(0.16, [1726, 1148, 1847, 1245])]),
    (27.5, []),
    (27.7, [(0.31, [1724, 1148, 1856, 1243])]),
    (28.0, [(0.24, [1724, 1148, 1858, 1245])]),
    (28.3, [(0.28, [1722, 1148, 1850, 1236]), (0.17, [1457, 690, 1988, 867])]),
    (28.5, [(0.22, [1724, 1148, 1845, 1236])]),
    (28.8, [(0.16, [1724, 1148, 1848, 1234])]),
    (29.1, [(0.23, [1726, 1148, 1856, 1234])]),
    (29.3, [(0.34, [1722, 1148, 1854, 1234])]),
    (29.6, [(0.17, [1724, 1148, 1854, 1234])]),
    (29.9, [(0.4, [1457, 688, 1984, 867]), (0.19, [1722, 1148, 1854, 1239])]),
    (30.1, [(0.26, [1457, 690, 1986, 867]), (0.21, [1726, 1148, 1850, 1232])]),
    (30.4, [(0.55, [1726, 1146, 1843, 1230]), (0.17, [1457, 690, 1986, 867])]),
    (30.7, [(0.27, [1457, 688, 1986, 867]), (0.23, [1724, 1146, 1866, 1226])]),
    (30.9, [(0.27, [1457, 688, 1986, 867]), (0.26, [1724, 1146, 1860, 1226])]),
    (31.2, [(0.2, [1721, 1146, 1852, 1226]), (0.17, [1457, 688, 1988, 867])]),
    (31.5, [(0.22, [1719, 1146, 1858, 1226])]),
    (31.7, [(0.22, [1719, 1146, 1860, 1226])]),
    (32.0, [(0.18, [1721, 1146, 1862, 1226])]),
    (32.3, [(0.3, [1726, 1146, 1833, 1226]), (0.15, [1457, 688, 1986, 867])]),
    (32.5, [(0.25, [1726, 1146, 1835, 1226])]),
    (32.8, [(0.32, [1724, 1146, 1837, 1224])]),
    (33.1, [(0.34, [1719, 1146, 1831, 1226])]),
    (33.3, [(0.15, [1715, 1148, 1858, 1226])]),
    (33.6, []),
    (33.9, []),
    (34.1, []),
    (34.4, []),
    (34.7, []),
    (34.9, []),
    (35.2, []),
    (35.5, [(0.15, [1457, 690, 1986, 867])]),
    (35.7, [(0.2, [1457, 690, 1986, 867])]),
    (36.0, [(0.17, [1457, 690, 1988, 867])]),
    (36.3, []),
    (36.5, []),
    (36.8, []),
    (37.1, []),
    (37.3, []),
    (37.6, [(0.22, [1455, 688, 1988, 867]), (0.17, [1719, 1148, 1852, 1239])]),
    (37.9, [(0.18, [1724, 1148, 1852, 1239])]),
    (38.1, []),
    (38.4, [(0.17, [1457, 688, 1988, 867])]),
    (38.7, [(0.32, [1457, 688, 1986, 867]), (0.16, [1726, 1148, 1850, 1247])]),
    (38.9, [(0.25, [1457, 688, 1988, 867])]),
    (39.2, [(0.19, [1728, 1148, 1854, 1243])]),
    (39.5, [(0.23, [1728, 1148, 1858, 1243])]),
    (39.7, [(0.24, [1728, 1148, 1845, 1243])]),
    (40.0, []),
    (40.3, []),
    (40.5, [(0.29, [1728, 1148, 1858, 1239])]),
    (40.8, [(0.16, [1728, 1148, 1856, 1241])]),
    (41.1, [(0.16, [1728, 1148, 1854, 1241])]),
    (41.3, [(0.45, [1457, 688, 1986, 867])]),
    (41.6, [(0.41, [1457, 688, 1986, 867])]),
    (41.9, [(0.15, [1457, 690, 1986, 867])]),
    (42.1, [(0.15, [1457, 690, 1986, 867])]),
    (42.4, [(0.21, [1726, 1148, 1856, 1234]), (0.19, [1457, 690, 1986, 867])]),
    (42.7, [(0.26, [1457, 690, 1986, 867])]),
    (42.9, [(0.25, [1457, 690, 1988, 867])]),
    (43.2, []),
    (43.5, []),
    (43.7, [(0.34, [1726, 1148, 1856, 1239])]),
    (44.0, [(0.17, [1728, 1148, 1852, 1239]), (0.17, [1457, 688, 1986, 867])]),
    (44.3, []),
    (44.5, [(0.17, [1457, 688, 1988, 867])]),
    (44.8, [(0.26, [1726, 1146, 1866, 1226])]),
    (45.1, [(0.17, [1722, 1146, 1869, 1226])]),
    (45.3, [(0.38, [1457, 690, 1986, 867])]),
    (45.6, [(0.19, [1457, 690, 1986, 867])]),
    (45.9, [(0.36, [1724, 1148, 1845, 1232])]),
    (46.1, [(0.28, [1724, 1148, 1833, 1232])]),
    (46.4, [(0.28, [1726, 1148, 1833, 1234])]),
    (46.7, [(0.42, [1728, 1146, 1820, 1230])]),
    (46.9, [(0.38, [1724, 1148, 1816, 1232])]),
    (47.2, [(0.27, [1726, 1148, 1829, 1232])]),
    (47.5, [(0.44, [1726, 1148, 1826, 1234])]),
    (47.7, [(0.43, [1726, 1148, 1829, 1234])]),
    (48.0, [(0.19, [1728, 1148, 1827, 1236])]),
    (48.3, [(0.4, [1726, 1148, 1831, 1241])]),
    (48.5, [(0.46, [1728, 1148, 1833, 1236])]),
    (48.8, [(0.47, [1728, 1148, 1831, 1236]), (0.19, [1457, 690, 1986, 867])]),
    (49.1, [(0.45, [1730, 1148, 1831, 1236]), (0.24, [1457, 690, 1986, 867])]),
    (49.3, [(0.39, [1728, 1148, 1835, 1236]), (0.22, [1457, 688, 1986, 867])]),
    (49.6, [(0.39, [1730, 1148, 1829, 1236]), (0.24, [1457, 688, 1986, 867])]),
    (49.9, [(0.21, [1457, 690, 1986, 867])]),
]


def _replay(sustain: float) -> float:
    """Run REPLAY through the live path's decisions; the longest idle stretch."""
    _Thresholds.sustain_confidence = sustain
    try:
        ev = _event(REPLAY[0][0])
        worker = _worker(ev)
        longest = 0.0
        for ts, boxes in REPLAY:
            dets = [_det(b, c) for c, b in boxes if c >= worker._live_inference_confidence()]
            strong = [d for d in dets if d.confidence >= 0.5]
            weak = [d for d in dets if d.confidence < 0.5]
            longest = max(longest, ts - ev.last_detection_ts)
            if strong:
                ev.update(worker._filter_false_positives(strong, W, H), ts, FRAME)
            elif worker._sustaining_detections(weak, W, H, ts):
                ev.sustain(ts)
        return longest
    finally:
        _Thresholds.sustain_confidence = 0.2


def test_replay_splits_the_rabbit_without_sustain():
    assert _replay(0.5) >= 10.0  # post_seconds: the event closed here


def test_replay_keeps_one_event_with_sustain():
    assert _replay(0.2) < 10.0


def test_lawn_feature_never_sustains():
    ev = _event()
    ev.update([_det(RABBIT, 0.6)], 0.0, FRAME)
    assert _worker(ev)._sustaining_detections([_det(LAWN, 0.45)], W, H, 1.0) == []


def test_weak_rabbit_box_sustains():
    ev = _event()
    ev.update([_det(RABBIT, 0.6)], 0.0, FRAME)
    kept = _worker(ev)._sustaining_detections([_det([1728, 1148, 1833, 1236], 0.27)], W, H, 5.0)
    assert len(kept) == 1


def test_anchor_expires_and_weak_boxes_do_not_renew_it():
    ev = _event()
    ev.update([_det(RABBIT, 0.6)], 0.0, FRAME)
    worker = _worker(ev)
    weak = [_det(RABBIT, 0.3)]
    assert worker._sustaining_detections(weak, W, H, SUSTAIN_ANCHOR_S - 1)
    ev.sustain(SUSTAIN_ANCHOR_S - 1)
    assert worker._sustaining_detections(weak, W, H, SUSTAIN_ANCHOR_S + 1) == []


def test_every_recent_subject_anchors():
    """Two rabbits a yard apart take turns at reaching the threshold."""
    left = [783, 784, 901, 846]
    ev = _event()
    ev.update([_det(left, 0.7)], 0.0, FRAME)
    ev.update([_det(RABBIT, 0.6)], 3.0, FRAME)
    assert _worker(ev)._sustaining_detections([_det(left, 0.3)], W, H, 6.0)


def test_sustain_adds_nothing_but_time():
    ev = _event()
    ev.update([_det(RABBIT, 0.6)], 0.0, FRAME)
    before = (set(ev.species), ev.max_confidence, list(ev.subject_boxes), dict(ev.species_key_frames))
    ev.sustain(4.0)
    assert ev.last_detection_ts == 4.0 and ev.sustained_frames == 1
    assert (set(ev.species), ev.max_confidence, list(ev.subject_boxes), dict(ev.species_key_frames)) == before


def test_inference_threshold_drops_only_while_an_event_is_open():
    assert _worker(None)._live_inference_confidence() == 0.5
    assert _worker(_event())._live_inference_confidence() == 0.2


def test_two_threshold_detector_keeps_its_own_bars():
    class _SpeciesNet:
        single_threshold = False
    assert _worker(_event(), _SpeciesNet())._live_inference_confidence() == 0.5


def test_sustain_at_or_above_confidence_is_off():
    _Thresholds.sustain_confidence = 0.7
    try:
        assert _worker(_event())._live_inference_confidence() == 0.5
    finally:
        _Thresholds.sustain_confidence = 0.2
