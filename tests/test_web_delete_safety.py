"""Tests for the destructive paths in web.py: the delete containment guard and the
PTZ dead-man watchdog.

Both were flagged as high-risk-and-uncovered while building the characterization
suite: `_delete_file` is the only path-traversal guard in the app, and a lost PTZ
'stop' leaves physical hardware moving. Unlike the `test_web_*` characterization
modules, these assert *intended* behaviour, not merely current behaviour.
"""

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from animaltracker import web as web_mod
from animaltracker.web import WebServer


@pytest.fixture
def server(tmp_path):
    """A WebServer rooted at tmp_path with an empty clips tree."""
    return WebServer({}, tmp_path, tmp_path / "logs", port=0)


@pytest.fixture
def clips(server, tmp_path):
    d = tmp_path / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# _delete_file containment
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel_path",
    [
        "../secret.txt",
        "../../etc/passwd",
        "cam1/../../secret.txt",
        "/etc/passwd",
        "/",
        "..",
        "cam1/../..",
    ],
)
def test_delete_rejects_paths_outside_clips(server, clips, tmp_path, rel_path):
    """Anything resolving outside clips/ is refused with the exact 403 sentinel."""
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")

    ok, msg = server._delete_file(rel_path)

    assert ok is False
    assert msg == "Invalid path", "handle_delete_recording maps this string to 403"
    assert outside.exists(), "guard must not delete anything outside clips/"


def test_delete_allows_legitimate_name_containing_double_dot(server, clips):
    """A filename that merely *contains* '..' is legitimate and must be deletable.

    The previous substring check (`'..' in rel_path`) rejected these outright.
    """
    (clips / "cam1").mkdir()
    target = clips / "cam1" / "a..b.mp4"
    target.write_bytes(b"x")

    ok, msg = server._delete_file("cam1/a..b.mp4")

    assert ok is True
    assert not target.exists()


def test_delete_removes_file_within_nested_archive_layout(server, clips):
    target = clips / "cam1" / "2026" / "09" / "07" / "1788792747_bird.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    ok, _ = server._delete_file("cam1/2026/09/07/1788792747_bird.mp4")

    assert ok is True
    assert not target.exists()


def test_delete_missing_file_reports_not_found(server, clips):
    ok, msg = server._delete_file("cam1/nope.mp4")
    assert ok is False
    assert msg == "File not found", "handle_delete_recording maps this string to 404"


def test_delete_refuses_a_directory(server, clips):
    (clips / "cam1").mkdir()
    ok, msg = server._delete_file("cam1")
    assert ok is False
    assert msg == "File not found"
    assert (clips / "cam1").is_dir()


def test_delete_does_not_follow_symlink_out_of_clips(server, clips, tmp_path):
    """A symlink inside clips/ pointing outside must not become a delete primitive."""
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    link = clips / "escape.mp4"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    ok, msg = server._delete_file("escape.mp4")

    assert ok is False
    assert msg == "Invalid path"
    assert outside.exists(), "target of the symlink must survive"


def test_delete_tolerates_symlinked_storage_root(tmp_path):
    """Resolving both sides means a symlinked storage root still deletes normally.

    The SSD mount in production makes this a realistic layout.
    """
    real = tmp_path / "real_storage"
    (real / "clips" / "cam1").mkdir(parents=True)
    (real / "clips" / "cam1" / "clip.mp4").write_bytes(b"x")

    link = tmp_path / "storage"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    srv = WebServer({}, link, tmp_path / "logs", port=0)

    ok, _ = srv._delete_file("cam1/clip.mp4")

    assert ok is True
    assert not (real / "clips" / "cam1" / "clip.mp4").exists()


# --------------------------------------------------------------------------
# PTZ dead-man watchdog
# --------------------------------------------------------------------------

class FakeOnvif:
    def __init__(self):
        self.moves = []
        self.stops = []

    def ptz_move(self, token, pan, tilt, zoom):
        self.moves.append((token, pan, tilt, zoom))

    def ptz_stop(self, token):
        self.stops.append(token)


class FakeWorker:
    def __init__(self):
        self.onvif_client = FakeOnvif()
        self.onvif_profile_token = "Profile_1"


@pytest.fixture
def ptz(tmp_path):
    worker = FakeWorker()
    srv = WebServer({"cam1": worker}, tmp_path, tmp_path / "logs", port=0)
    return srv, worker


def test_deadman_stops_camera_when_no_stop_arrives(ptz, monkeypatch):
    """A move with no follow-up stop must be stopped by the server."""
    async def _t():
        srv, worker = ptz
        monkeypatch.setattr(web_mod, "PTZ_DEADMAN_SECONDS", 0.05)

        srv._arm_ptz_deadman("cam1", worker)
        assert worker.onvif_client.stops == [], "must not stop immediately"

        await asyncio.sleep(0.15)

        assert worker.onvif_client.stops == ["Profile_1"]
        assert "cam1" not in srv._ptz_deadman, "watchdog should clean up after firing"

    asyncio.run(_t())


def test_explicit_stop_cancels_the_watchdog(ptz, monkeypatch):
    """The normal press-and-release path must not produce a second, later stop."""
    async def _t():
        srv, worker = ptz
        monkeypatch.setattr(web_mod, "PTZ_DEADMAN_SECONDS", 0.05)

        srv._arm_ptz_deadman("cam1", worker)
        srv._cancel_ptz_deadman("cam1")

        await asyncio.sleep(0.15)

        assert worker.onvif_client.stops == [], "cancelled watchdog must not fire"
        assert "cam1" not in srv._ptz_deadman

    asyncio.run(_t())


def test_rearming_extends_rather_than_stacking(ptz, monkeypatch):
    """Successive moves re-arm one timer instead of queueing several stops."""
    async def _t():
        srv, worker = ptz
        monkeypatch.setattr(web_mod, "PTZ_DEADMAN_SECONDS", 0.12)

        srv._arm_ptz_deadman("cam1", worker)
        await asyncio.sleep(0.06)
        srv._arm_ptz_deadman("cam1", worker)  # supersedes the first
        await asyncio.sleep(0.06)

        assert worker.onvif_client.stops == [], "first timer should have been cancelled"

        await asyncio.sleep(0.12)
        assert worker.onvif_client.stops == ["Profile_1"], "exactly one stop, from the latest arm"

    asyncio.run(_t())


def test_watchdogs_are_independent_per_camera(tmp_path, monkeypatch):
    async def _t():
        w1, w2 = FakeWorker(), FakeWorker()
        srv = WebServer({"cam1": w1, "cam2": w2}, tmp_path, tmp_path / "logs", port=0)
        monkeypatch.setattr(web_mod, "PTZ_DEADMAN_SECONDS", 0.05)

        srv._arm_ptz_deadman("cam1", w1)
        srv._arm_ptz_deadman("cam2", w2)
        srv._cancel_ptz_deadman("cam2")

        await asyncio.sleep(0.15)

        assert w1.onvif_client.stops == ["Profile_1"], "cam1 was never stopped, so it fires"
        assert w2.onvif_client.stops == [], "cam2 was stopped explicitly"

    asyncio.run(_t())


def test_cancel_is_safe_when_nothing_is_armed(ptz):
    srv, worker = ptz
    srv._cancel_ptz_deadman("cam1")  # must not raise
    srv._cancel_ptz_deadman("unknown-camera")
    assert worker.onvif_client.stops == []


def test_deadman_survives_a_failing_stop_call(ptz, monkeypatch):
    """An ONVIF error in the watchdog must be logged, not left as a dangling task."""
    async def _t():
        srv, worker = ptz
        monkeypatch.setattr(web_mod, "PTZ_DEADMAN_SECONDS", 0.05)

        def boom(token):
            raise RuntimeError("camera unreachable")

        worker.onvif_client.ptz_stop = boom

        srv._arm_ptz_deadman("cam1", worker)
        await asyncio.sleep(0.15)

        assert "cam1" not in srv._ptz_deadman, "entry must be cleaned up even on failure"

    asyncio.run(_t())


