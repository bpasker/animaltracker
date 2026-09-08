"""Tests for the JSON API the front-end is built against.

These assert *intended* behaviour (unlike the `test_web_*` characterization
modules): the contract a rewritten client depends on. Handlers are driven
directly with a minimal fake request rather than through an HTTP client, so the
suite needs no aiohttp test plugin.
"""

import asyncio
import json
import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from animaltracker.web import WebServer


class FakeRequest:
    """Just enough of aiohttp's Request for these handlers."""

    def __init__(self, query=None, match_info=None):
        self.query = query or {}
        self.match_info = match_info or {}


def call(handler, **kw):
    """Run an async handler and decode its JSON response."""
    resp = asyncio.run(handler(FakeRequest(**kw)))
    return resp.status, json.loads(resp.body.decode())


@pytest.fixture
def server(tmp_path):
    return WebServer({}, tmp_path, tmp_path / "logs", port=0)


def make_clip(clips_dir, camera, epoch, species, size=1024):
    when = time.localtime(epoch)
    d = clips_dir / camera / f"{when.tm_year:04d}" / f"{when.tm_mon:02d}" / f"{when.tm_mday:02d}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{int(epoch)}_{species}.mp4"
    f.write_bytes(b"x" * size)
    os.utime(f, (epoch, epoch))
    return f


@pytest.fixture
def archive(server, tmp_path):
    """Three cameras' worth of clips at known times and species."""
    clips = tmp_path / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    base = time.mktime((2026, 9, 7, 12, 0, 0, 0, 0, -1))
    make_clip(clips, "cam1", base, "mammalia_artiodactyla_cervidae_odocoileus_virginianus")
    make_clip(clips, "cam1", base - 3600, "bird_passeriformes_cardinalidae")
    make_clip(clips, "cam2", base - 86400, "mammalia_carnivora_procyonidae_procyon_lotor", size=4096)
    server._invalidate_scan_cache()
    return clips


# --------------------------------------------------------------------------
# /api/recordings
# --------------------------------------------------------------------------

def test_list_returns_all_clips_newest_first(server, archive):
    status, body = call(server.handle_recordings_api)
    assert status == 200
    assert body["total"] == 3
    assert body["archive_total"] == 3
    epochs = [c["epoch"] for c in body["clips"]]
    assert epochs == sorted(epochs, reverse=True), "default order is newest first"


def test_clip_shape_is_json_safe_and_complete(server, archive):
    _, body = call(server.handle_recordings_api)
    clip = body["clips"][0]
    for key in ("path", "filename", "camera", "species", "raw_species",
                "date", "time", "epoch", "size", "size_mb", "thumbnails"):
        assert key in clip, f"missing {key}"
    assert isinstance(clip["time"], str), "datetime must be serialised"
    # ISO-8601 with an offset, so the client never has to guess a timezone.
    assert clip["time"][4] == "-" and ("+" in clip["time"][10:] or "-" in clip["time"][10:])
    json.dumps(clip)  # must round-trip


def test_filter_by_camera(server, archive):
    _, body = call(server.handle_recordings_api, query={"camera": "cam2"})
    assert body["total"] == 1
    assert {c["camera"] for c in body["clips"]} == {"cam2"}


def test_filter_by_multiple_cameras(server, archive):
    _, body = call(server.handle_recordings_api, query={"camera": "cam1,cam2"})
    assert body["total"] == 3


def test_filter_by_species_display_name(server, archive):
    _, body = call(server.handle_recordings_api, query={"species": "Raccoon"})
    assert body["total"] == 1
    assert body["clips"][0]["species"] == "Raccoon"


def test_free_text_search_matches_species_and_camera(server, archive):
    _, deer = call(server.handle_recordings_api, query={"q": "deer"})
    assert deer["total"] == 1

    _, cam = call(server.handle_recordings_api, query={"q": "cam2"})
    assert cam["total"] == 1

    _, none = call(server.handle_recordings_api, query={"q": "zebra"})
    assert none["total"] == 0


def test_search_is_case_insensitive(server, archive):
    _, lower = call(server.handle_recordings_api, query={"q": "raccoon"})
    _, upper = call(server.handle_recordings_api, query={"q": "RACCOON"})
    assert lower["total"] == upper["total"] == 1


def test_date_range_filters_inclusively(server, archive):
    _, body = call(server.handle_recordings_api,
                   query={"from": "2026-09-07", "to": "2026-09-07"})
    assert body["total"] == 2, "both same-day clips, not the previous day's"


def test_sort_orders(server, archive):
    _, oldest = call(server.handle_recordings_api, query={"sort": "oldest"})
    assert oldest["clips"][0]["epoch"] < oldest["clips"][-1]["epoch"]

    _, by_species = call(server.handle_recordings_api, query={"sort": "species"})
    names = [c["species"] for c in by_species["clips"]]
    assert names == sorted(names, key=str.lower)

    _, largest = call(server.handle_recordings_api, query={"sort": "largest"})
    assert largest["clips"][0]["size"] == 4096


def test_pagination_reports_has_more_and_slices(server, archive):
    _, first = call(server.handle_recordings_api, query={"limit": "2"})
    assert len(first["clips"]) == 2
    assert first["has_more"] is True

    _, last = call(server.handle_recordings_api, query={"limit": "2", "offset": "2"})
    assert len(last["clips"]) == 1
    assert last["has_more"] is False

    assert first["clips"][0]["path"] != last["clips"][0]["path"]


def test_facets_count_the_whole_match_not_the_page(server, archive):
    """Counts must not change while paging, or the filter UI flickers."""
    _, full = call(server.handle_recordings_api)
    _, paged = call(server.handle_recordings_api, query={"limit": "1"})
    assert full["facets"] == paged["facets"]
    assert {f["value"]: f["count"] for f in full["facets"]["cameras"]} == {"cam1": 2, "cam2": 1}


def test_facets_reflect_the_active_filter(server, archive):
    _, body = call(server.handle_recordings_api, query={"camera": "cam1"})
    assert {f["value"] for f in body["facets"]["cameras"]} == {"cam1"}


def test_bad_pagination_is_rejected(server, archive):
    status, body = call(server.handle_recordings_api, query={"limit": "abc"})
    assert status == 400
    assert "error" in body


def test_limit_is_clamped(server, archive):
    _, body = call(server.handle_recordings_api, query={"limit": "99999"})
    assert body["limit"] <= 500, "an unbounded limit would let one request walk the archive"


def test_empty_archive_returns_an_empty_page_not_an_error(server, tmp_path):
    (tmp_path / "clips").mkdir(parents=True, exist_ok=True)
    status, body = call(server.handle_recordings_api)
    assert status == 200
    assert body["clips"] == [] and body["total"] == 0
    assert body["facets"]["cameras"] == []


# --------------------------------------------------------------------------
# /api/clip/{path}
# --------------------------------------------------------------------------

def test_clip_detail_returns_serialisable_payload(server, archive):
    _, listing = call(server.handle_recordings_api)
    path = listing["clips"][0]["path"]

    status, body = call(server.handle_clip_api, match_info={"path": path})
    assert status == 200
    assert body["path"] == path
    assert body["url"] == f"/clips/{path}"
    assert isinstance(body["time"], str)
    assert body["reprocessing"] is False
    json.dumps(body)


def test_missing_clip_is_404(server, archive):
    status, body = call(server.handle_clip_api, match_info={"path": "cam9/nope.mp4"})
    assert status == 404
    assert "error" in body


def test_clip_detail_reports_an_in_flight_reprocess(server, archive):
    _, listing = call(server.handle_recordings_api)
    path = listing["clips"][0]["path"]
    server.reprocessing_jobs[path] = {"started": time.time(), "clip_name": "x"}

    _, body = call(server.handle_clip_api, match_info={"path": path})
    assert body["reprocessing"] is True, "the client polls this to show progress"


# --------------------------------------------------------------------------
# /api/cameras
# --------------------------------------------------------------------------

class FakeCam:
    def __init__(self, cid, name):
        self.id, self.name, self.location = cid, name, "North Fence"
        self.ptz_tracking = None


class FakeWorker:
    def __init__(self, frame=object(), age=0.0, connected=True, ptz=False):
        self.camera = FakeCam("cam1", "Backyard Wide")
        self.latest_frame = frame
        self.latest_frame_ts = (time.time() - age) if frame is not None else 0.0
        self.stream_connected = connected
        self.onvif_client = object() if ptz else None
        self.onvif_profile_token = "Profile_1" if ptz else None
        self.ptz_tracker = object() if ptz else None


@pytest.mark.parametrize(
    "worker,expected",
    [
        (FakeWorker(age=0.5), "live"),
        (FakeWorker(age=30.0), "stale"),
        (FakeWorker(connected=False), "stale"),
        (FakeWorker(frame=None), "offline"),
    ],
)
def test_camera_state_reflects_stream_health(tmp_path, worker, expected):
    srv = WebServer({"cam1": worker}, tmp_path, tmp_path / "logs", port=0)
    _, body = call(srv.handle_cameras_api)
    assert body["cameras"][0]["state"] == expected


def test_camera_payload_exposes_ptz_capability(tmp_path):
    srv = WebServer({"cam1": FakeWorker(ptz=True)}, tmp_path, tmp_path / "logs", port=0)
    _, body = call(srv.handle_cameras_api)
    cam = body["cameras"][0]
    assert cam["has_ptz"] is True and cam["has_tracker"] is True
    assert cam["stream_url"] == "/stream/cam1"


def test_no_cameras_is_an_empty_list_not_an_error(server):
    status, body = call(server.handle_cameras_api)
    assert status == 200
    assert body["cameras"] == []
    assert "timezone" in body
