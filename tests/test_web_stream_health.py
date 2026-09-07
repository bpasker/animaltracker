"""Characterization tests for the web UI's stream-health rendering.

These tests pin the CURRENT behaviour of:

  * ``WebServer._render_stream_down_placeholder`` (web.py ~5493-5521)
  * ``WebServer._render_frame_jpeg``              (web.py ~5297-5397)
  * ``WebServer.handle_snapshot``                 (web.py ~5251-5296)

They are a safety net for a rewrite: anything asserted here is behaviour the
rewrite must reproduce, quirks included.  Nothing under ``src/`` is modified.

Testing approach
----------------
``_render_frame_jpeg`` and ``_render_stream_down_placeholder`` are plain
methods (not closures inside the handlers), so they are called directly.
The banner's age wording and the bounding-box coordinates only ever exist as
*pixels* in the encoded JPEG, so where the exact string/coordinate matters the
tests install a thin recording proxy over the ``cv2`` name inside
``animaltracker.web`` that records ``putText``/``rectangle`` arguments and
delegates to the real OpenCV call.  The real production code path still runs
end to end; the proxy only observes it.  Pixel-level assertions on the decoded
JPEG back up the recorded calls where it is cheap to do so.

``STALE_AFTER_SEC`` is a local constant inside the handlers, so the threshold
is pinned through ``handle_snapshot`` with a fake clock and a fake worker.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import pytest

from animaltracker import web as web_mod
from animaltracker.detector import Detection
from animaltracker.web import WebServer


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class FakeWorker:
    """Minimal stand-in for StreamWorker exposing only what the handlers read."""

    def __init__(self, frame=None, frame_ts=0.0, connected=True, detections=None):
        self.latest_frame = frame
        self.latest_frame_ts = frame_ts
        self.stream_connected = connected
        self.latest_detections = detections if detections is not None else []


class _CvRecorder:
    """Proxy over the real cv2 module that records draw calls."""

    def __init__(self):
        self.texts = []      # strings passed to putText, in call order
        self.rectangles = []  # (pt1, pt2, color, thickness) in call order

    def putText(self, img, text, org, *args, **kwargs):  # noqa: N802 (cv2 name)
        self.texts.append(text)
        return cv2.putText(img, text, org, *args, **kwargs)

    def rectangle(self, img, pt1, pt2, color, thickness=1, *args, **kwargs):
        self.rectangles.append((tuple(pt1), tuple(pt2), tuple(color), thickness))
        return cv2.rectangle(img, pt1, pt2, color, thickness, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(cv2, name)


@pytest.fixture
def recorder(monkeypatch):
    rec = _CvRecorder()
    monkeypatch.setattr(web_mod, 'cv2', rec)
    return rec


_PORT = [8730]


@pytest.fixture
def server(tmp_path):
    """A WebServer with no cameras; enough for the pure rendering methods."""
    _PORT[0] += 1
    return WebServer(
        workers={},
        storage_root=tmp_path / 'storage',
        logs_root=tmp_path / 'logs',
        port=_PORT[0],
        config_path=tmp_path / 'cameras.yml',
        runtime=None,
    )


def decode(jpeg_bytes):
    """Decode JPEG bytes back to a BGR ndarray (None if undecodable)."""
    return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def green_mask(img):
    """Mask of pixels that read as the overlay green (0,255,0) after JPEG."""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    return (g > 150) & (r < 120) & (b < 120)


def gray_frame(width, height, value=200):
    return np.full((height, width, 3), value, dtype=np.uint8)


def banner_text(server, age_seconds, width=1280, height=720, recorder=None):
    """Render a stale frame and return the age line the banner drew."""
    recorder.texts.clear()
    recorder.rectangles.clear()
    server._render_frame_jpeg(gray_frame(width, height), [], True, age_seconds)
    # putText order for a stale frame: "STREAM DOWN" then the age line.
    assert recorder.texts[0] == "STREAM DOWN"
    return recorder.texts[1]


# --------------------------------------------------------------------------
# _render_stream_down_placeholder
# --------------------------------------------------------------------------

def test_placeholder_returns_decodable_jpeg_of_fixed_640x360(server):
    payload = server._render_stream_down_placeholder('cam1')

    assert isinstance(payload, bytes) and payload
    assert payload[:2] == b'\xff\xd8'  # JPEG SOI marker
    img = decode(payload)
    assert img is not None
    assert img.shape == (360, 640, 3)


def test_placeholder_draws_stream_down_banner_and_camera_id(server, recorder):
    server._render_stream_down_placeholder('cam2')

    assert recorder.texts == ["STREAM DOWN", "camera: cam2"]
    # Solid red banner box, then a white 2px outline around the same box.
    assert recorder.rectangles[0][2] == (0, 0, 220)
    assert recorder.rectangles[0][3] == -1
    assert recorder.rectangles[1][2] == (255, 255, 255)
    assert recorder.rectangles[1][3] == 2


def test_placeholder_background_is_black_outside_the_banner(server):
    img = decode(server._render_stream_down_placeholder('cam1'))
    # Corners are well outside the centred banner box.
    for y, x in ((2, 2), (2, 637), (357, 2), (357, 637)):
        assert int(img[y, x].max()) < 20


def test_placeholder_ignores_the_age_argument(server, recorder):
    # QUIRK: _render_stream_down_placeholder accepts `age` but never renders it,
    # so a camera down for 3 hours looks identical to one down for 3 seconds
    # — asserted as-is to detect rewrite drift.
    a = server._render_stream_down_placeholder('cam1', age=float('inf'))
    b = server._render_stream_down_placeholder('cam1', age=3.0)
    c = server._render_stream_down_placeholder('cam1', age=99999.0)
    assert a == b == c
    assert not any('ago' in t for t in recorder.texts)


def test_placeholder_with_empty_camera_id(server, recorder):
    payload = server._render_stream_down_placeholder('')
    assert decode(payload).shape == (360, 640, 3)
    assert recorder.texts[1] == "camera: "


def test_placeholder_with_unicode_camera_id(server, recorder):
    # cv2's Hershey fonts are ASCII-only; non-ASCII is substituted, not fatal.
    payload = server._render_stream_down_placeholder('caméra-ünïcode-å')
    assert decode(payload).shape == (360, 640, 3)
    assert recorder.texts[1] == "camera: caméra-ünïcode-å"


def test_placeholder_with_quotes_and_apostrophes_in_camera_id(server, recorder):
    payload = server._render_stream_down_placeholder("brandon's \"cam\"")
    assert decode(payload).shape == (360, 640, 3)
    assert recorder.texts[1] == "camera: brandon's \"cam\""


def test_placeholder_with_overlong_camera_id_still_encodes(server, recorder):
    # QUIRK: the banner box is sized from the text with no clamping, so a very
    # long camera id produces a negative x0 and the text is clipped off the
    # left edge rather than wrapped or truncated — asserted as-is to detect
    # rewrite drift.
    payload = server._render_stream_down_placeholder('c' * 400)
    assert decode(payload).shape == (360, 640, 3)
    x0 = recorder.rectangles[0][0][0]
    assert x0 < 0


# --------------------------------------------------------------------------
# stale banner: the relative-age wording ladder
# --------------------------------------------------------------------------

@pytest.mark.parametrize("age,expected", [
    (float('inf'), "no frames received"),
    (0.0, "last frame 0s ago"),
    (0.9, "last frame 0s ago"),
    (1.0, "last frame 1s ago"),
    (5.1, "last frame 5s ago"),
    (59.0, "last frame 59s ago"),
    (59.999, "last frame 59s ago"),
    (60.0, "last frame 1m ago"),
    (61.0, "last frame 1m ago"),
    (119.0, "last frame 1m ago"),
    (120.0, "last frame 2m ago"),
    (3599.0, "last frame 59m ago"),
    (3599.999, "last frame 59m ago"),
    (3600.0, "last frame 1h ago"),
    (7200.0, "last frame 2h ago"),
    (86399.0, "last frame 23h ago"),
    (86400.0, "last frame 24h ago"),
    (86400.001, "no frames received"),
    (172800.0, "no frames received"),
])
def test_stale_banner_age_wording_ladder(server, recorder, age, expected):
    # The ladder boundaries are inclusive-low: 60 flips to minutes, 3600 to
    # hours, and only *strictly* above 86400 does it collapse back to
    # "no frames received" (so 86400.0 exactly still reads "24h ago").
    assert banner_text(server, age, recorder=recorder) == expected


def test_stale_banner_negative_age_from_clock_skew(server, recorder):
    # QUIRK: a backwards clock jump yields a negative age, which falls into the
    # "< 60" seconds branch and renders a negative second count rather than
    # being clamped to 0 — asserted as-is to detect rewrite drift.
    assert banner_text(server, -1.5, recorder=recorder) == "last frame -1s ago"
    assert banner_text(server, -0.5, recorder=recorder) == "last frame 0s ago"


def test_stale_banner_negative_infinity_age_raises(server, recorder):
    # QUIRK: only +inf is special-cased. A -inf age falls into the "< 60"
    # branch and int(float('-inf')) raises OverflowError, so rendering the
    # frame blows up instead of degrading to a banner — asserted as-is to
    # detect rewrite drift.
    with pytest.raises(OverflowError):
        banner_text(server, float('-inf'), recorder=recorder)


def test_stale_banner_title_is_stream_down(server, recorder):
    server._render_frame_jpeg(gray_frame(1280, 720), [], True, 12.0)
    assert recorder.texts[0] == "STREAM DOWN"


def test_stale_frame_is_dimmed_to_45_percent(server):
    out = decode(server._render_frame_jpeg(gray_frame(1280, 720, 200), [], True, 12.0)[1].tobytes())
    # Corner pixel is untouched by the centred banner: 200 * 0.45 == 90.
    assert abs(int(out[3, 3, 1]) - 90) <= 3


def test_live_frame_is_not_dimmed(server):
    out = decode(server._render_frame_jpeg(gray_frame(1280, 720, 200), [], False, 0.2)[1].tobytes())
    assert abs(int(out[3, 3, 1]) - 200) <= 3


def test_stale_banner_uses_red_fill_with_white_outline(server, recorder):
    server._render_frame_jpeg(gray_frame(1280, 720), [], True, 12.0)
    assert recorder.rectangles[0][2] == (0, 0, 220)
    assert recorder.rectangles[0][3] == -1
    assert recorder.rectangles[1][2] == (255, 255, 255)
    assert recorder.rectangles[1][3] == 2


# --------------------------------------------------------------------------
# detection boxes: suppressed when stale
# --------------------------------------------------------------------------

def _one_detection():
    return [Detection(species='deer', confidence=0.87, bbox=[600.0, 300.0, 700.0, 400.0])]


def test_boxes_are_suppressed_when_the_stream_is_stale(server, recorder):
    # Deliberate guarantee: boxes must never be drawn over a frozen frame.
    server._render_frame_jpeg(gray_frame(1280, 720), _one_detection(), True, 42.0)

    assert not any('87%' in t for t in recorder.texts)
    assert recorder.texts == ["STREAM DOWN", "last frame 42s ago"]
    # Only the two banner rectangles were drawn — no detection box.
    assert len(recorder.rectangles) == 2


def test_no_green_box_pixels_on_a_stale_frame(server):
    ok, buf = server._render_frame_jpeg(gray_frame(1280, 720), _one_detection(), True, 42.0)
    assert ok
    assert green_mask(decode(buf.tobytes())).sum() == 0


def test_boxes_are_drawn_when_the_stream_is_live(server, recorder):
    server._render_frame_jpeg(gray_frame(1280, 720), _one_detection(), False, 0.1)

    assert recorder.rectangles[0][:3] == ((600, 300), (700, 400), (0, 255, 0))
    assert recorder.rectangles[0][3] == 2  # 2px outline
    assert recorder.texts and recorder.texts[0].endswith(' 87%')


def test_many_stale_detections_still_draw_nothing(server, recorder):
    dets = [Detection(species='deer', confidence=0.5, bbox=[i, i, i + 50, i + 50])
            for i in range(0, 500, 25)]
    server._render_frame_jpeg(gray_frame(1280, 720), dets, True, 10.0)
    assert len(recorder.rectangles) == 2  # banner only


# --------------------------------------------------------------------------
# detection boxes: original-frame pixel coordinates, drawn before the downscale
# --------------------------------------------------------------------------

def test_boxes_use_original_pixel_coords_and_are_drawn_before_downscale(server, recorder):
    ok, buf = server._render_frame_jpeg(gray_frame(1280, 720), _one_detection(), False, 0.1)
    assert ok
    out = decode(buf.tobytes())

    # The rectangle call used the *original* 1280x720 coordinates ...
    assert recorder.rectangles[0][0] == (600, 300)
    assert recorder.rectangles[0][1] == (700, 400)
    # ... and the resize to 640px wide happened afterwards.
    assert out.shape == (360, 640, 3)

    # So in the encoded image the box lands at exactly half the coordinates.
    ys, xs = np.nonzero(green_mask(out))
    assert xs.size > 0
    assert abs(int(xs.min()) - 300) <= 2
    assert abs(int(xs.max()) - 350) <= 2
    assert abs(int(ys.max()) - 200) <= 2


def test_label_is_placed_above_the_box_using_the_common_name(server, recorder):
    server._render_frame_jpeg(gray_frame(1280, 720), _one_detection(), False, 0.1)
    label = recorder.texts[0]
    assert label.endswith(' 87%')
    assert label != ' 87%'  # a common name was resolved and prefixed


def test_label_for_a_box_at_the_top_edge_is_pushed_down_into_frame(server, recorder):
    # label_y = max(y1 - 10, text_height + 10): a box at y1=0 gets its label
    # drawn *inside* the frame rather than off the top.
    det = [Detection(species='deer', confidence=0.5, bbox=[10.0, 0.0, 110.0, 100.0])]
    server._render_frame_jpeg(gray_frame(1280, 720), det, False, 0.1)
    # label background rect is the 2nd rectangle (after the box outline)
    label_rect_top = recorder.rectangles[1][0][1]
    assert label_rect_top >= 0


def test_detection_with_empty_bbox_is_skipped(server, recorder):
    det = [Detection(species='deer', confidence=0.5, bbox=[])]
    ok, buf = server._render_frame_jpeg(gray_frame(1280, 720), det, False, 0.1)
    assert ok
    assert recorder.rectangles == []
    assert recorder.texts == []


def test_negative_and_out_of_frame_bbox_does_not_raise(server):
    det = [
        Detection(species='deer', confidence=0.5, bbox=[-50.0, -50.0, 20.0, 20.0]),
        Detection(species='deer', confidence=0.5, bbox=[5000.0, 5000.0, 6000.0, 6000.0]),
    ]
    ok, buf = server._render_frame_jpeg(gray_frame(1280, 720), det, False, 0.1)
    assert ok
    assert decode(buf.tobytes()) is not None


def test_render_frame_jpeg_mutates_the_caller_s_array_when_live(server):
    # QUIRK: overlays are drawn in place on the array handed in; only the
    # callers' own .copy() keeps the worker's latest_frame clean — asserted
    # as-is to detect rewrite drift.
    img = gray_frame(1280, 720)
    before = img.copy()
    server._render_frame_jpeg(img, _one_detection(), False, 0.1)
    assert not np.array_equal(img, before)


def test_render_frame_jpeg_leaves_the_caller_s_array_alone_when_stale(server):
    # The stale path reassigns `img` via addWeighted, so the dimming and the
    # banner land on a new array and the input survives untouched.
    img = gray_frame(1280, 720)
    before = img.copy()
    server._render_frame_jpeg(img, _one_detection(), True, 42.0)
    assert np.array_equal(img, before)


# --------------------------------------------------------------------------
# the 640px downscale rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("w,h,expected", [
    (1280, 720, (360, 640, 3)),
    (641, 480, (479, 640, 3)),   # int(480 * 640/641) == 479
    (640, 480, (480, 640, 3)),   # exactly 640: untouched
    (639, 480, (480, 639, 3)),   # below 640: never upscaled
    (320, 240, (240, 320, 3)),
])
def test_downscale_only_applies_above_640px_wide(server, w, h, expected):
    ok, buf = server._render_frame_jpeg(gray_frame(w, h), [], False, 0.1)
    assert ok
    assert decode(buf.tobytes()).shape == expected


def test_tiny_one_pixel_frame_still_encodes(server):
    ok, buf = server._render_frame_jpeg(gray_frame(1, 1), [], False, 0.1)
    assert ok
    assert decode(buf.tobytes()).shape == (1, 1, 3)


def test_tiny_frame_stale_banner_overflows_but_still_encodes(server):
    # QUIRK: on a 1x1 frame the banner box is far larger than the image; the
    # draw is clipped by OpenCV and the result is a 1x1 JPEG with no legible
    # warning at all — asserted as-is to detect rewrite drift.
    ok, buf = server._render_frame_jpeg(gray_frame(1, 1), [], True, 42.0)
    assert ok
    assert decode(buf.tobytes()).shape == (1, 1, 3)


# --------------------------------------------------------------------------
# STALE_AFTER_SEC = 5.0, through handle_snapshot
# --------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, camera_id):
        self.match_info = {'camera_id': camera_id}


class FakeClock:
    """Stands in for the `_time` module inside animaltracker.web."""

    def __init__(self, now):
        self._now = now

    def time(self):
        return self._now


def snapshot(server, camera_id='cam1'):
    return asyncio.run(server.handle_snapshot(FakeRequest(camera_id)))


@pytest.fixture
def frozen_clock(monkeypatch):
    clock = FakeClock(1_000_000.0)
    monkeypatch.setattr(web_mod, '_time', clock)
    return clock


@pytest.mark.parametrize("age,status", [
    (0.0, 'ok'),
    (4.9, 'ok'),
    (5.0, 'ok'),      # threshold is `age > 5.0`, so exactly 5.0 is still live
    (5.000001, 'down'),
    (6.0, 'down'),
    (100.0, 'down'),
])
def test_stale_after_sec_threshold_is_strictly_greater_than_5(server, frozen_clock, age, status):
    server.workers['cam1'] = FakeWorker(
        frame=gray_frame(1280, 720),
        frame_ts=frozen_clock.time() - age,
        connected=True,
    )
    resp = snapshot(server)
    assert resp.headers['X-Stream-Status'] == status


def test_disconnected_stream_is_stale_even_with_a_fresh_frame(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(
        frame=gray_frame(1280, 720),
        frame_ts=frozen_clock.time(),  # age 0
        connected=False,
    )
    assert snapshot(server).headers['X-Stream-Status'] == 'down'


def test_zero_frame_timestamp_means_infinite_age(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(frame=gray_frame(1280, 720), frame_ts=0.0, connected=True)
    resp = snapshot(server)
    assert resp.headers['X-Stream-Status'] == 'down'
    assert resp.headers['X-Frame-Age-Seconds'] == 'inf'


def test_none_frame_timestamp_is_coerced_to_zero(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(frame=gray_frame(1280, 720), frame_ts=None, connected=True)
    resp = snapshot(server)
    assert resp.headers['X-Frame-Age-Seconds'] == 'inf'


def test_snapshot_age_header_is_one_decimal_place(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(
        frame=gray_frame(1280, 720),
        frame_ts=frozen_clock.time() - 12.3456,
        connected=True,
    )
    assert snapshot(server).headers['X-Frame-Age-Seconds'] == '12.3'


def test_snapshot_is_never_cached(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(frame=gray_frame(1280, 720), frame_ts=frozen_clock.time())
    assert snapshot(server).headers['Cache-Control'] == 'no-store'


def test_snapshot_returns_decodable_jpeg(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(frame=gray_frame(1280, 720), frame_ts=frozen_clock.time())
    resp = snapshot(server)
    assert resp.content_type == 'image/jpeg'
    assert decode(resp.body).shape == (360, 640, 3)


def test_snapshot_suppresses_boxes_once_the_frame_goes_stale(server, frozen_clock):
    fresh = FakeWorker(
        frame=gray_frame(1280, 720),
        frame_ts=frozen_clock.time() - 1.0,
        detections=_one_detection(),
    )
    server.workers['cam1'] = fresh
    assert green_mask(decode(snapshot(server).body)).sum() > 0

    fresh.latest_frame_ts = frozen_clock.time() - 6.0
    assert green_mask(decode(snapshot(server).body)).sum() == 0


def test_snapshot_does_not_mutate_the_worker_s_latest_frame(server, frozen_clock):
    frame = gray_frame(1280, 720)
    server.workers['cam1'] = FakeWorker(
        frame=frame, frame_ts=frozen_clock.time(), detections=_one_detection(),
    )
    snapshot(server)
    assert np.array_equal(frame, gray_frame(1280, 720))


def test_snapshot_treats_none_detections_as_empty(server, frozen_clock):
    worker = FakeWorker(frame=gray_frame(1280, 720), frame_ts=frozen_clock.time())
    worker.latest_detections = None
    server.workers['cam1'] = worker
    assert green_mask(decode(snapshot(server).body)).sum() == 0


# --------------------------------------------------------------------------
# handle_snapshot: no frame at all, and unknown cameras
# --------------------------------------------------------------------------

def test_snapshot_with_no_frame_returns_the_placeholder(server, frozen_clock):
    server.workers['cam1'] = FakeWorker(frame=None, frame_ts=0.0, connected=False)
    resp = snapshot(server)
    assert resp.headers['X-Stream-Status'] == 'down'
    assert resp.content_type == 'image/jpeg'
    assert decode(resp.body).shape == (360, 640, 3)
    assert resp.body == server._render_stream_down_placeholder('cam1')


def test_snapshot_age_header_is_hardcoded_inf_when_there_is_no_frame(server, frozen_clock):
    # QUIRK: the no-frame branch always reports X-Frame-Age-Seconds: inf, even
    # when latest_frame_ts is recent (a worker that had a timestamp but whose
    # frame was cleared) — asserted as-is to detect rewrite drift.
    server.workers['cam1'] = FakeWorker(
        frame=None, frame_ts=frozen_clock.time() - 2.0, connected=True,
    )
    resp = snapshot(server)
    assert resp.headers['X-Frame-Age-Seconds'] == 'inf'
    assert resp.headers['X-Stream-Status'] == 'down'


def test_snapshot_for_an_unknown_camera_is_404(server):
    resp = snapshot(server, 'nope')
    assert resp.status == 404
    assert resp.text == "Camera not found"


def test_snapshot_for_an_empty_camera_id_is_404(server):
    resp = snapshot(server, '')
    assert resp.status == 404
