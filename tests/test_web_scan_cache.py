"""Tests for the TTL cache in front of the archive scan.

Every archive read walks the whole clips tree and stats each file, and a single
page load does it twice. `_scan_recordings_cached` fronts that with a short TTL.
The uncached `_scan_recordings` primitive must keep its characterized behaviour,
so these tests also pin that it is *not* cached.
"""

import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from animaltracker import web as web_mod
from animaltracker.web import WebServer


def _clip(clips_dir, camera, name, when=None):
    """Create a clip in the camera/YYYY/MM/DD layout the scanner expects."""
    d = clips_dir / camera / "2026" / "09" / "07"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_bytes(b"x")
    if when is not None:
        import os
        os.utime(f, (when, when))
    return f


@pytest.fixture
def server(tmp_path):
    return WebServer({}, tmp_path, tmp_path / "logs", port=0)


@pytest.fixture
def clips(server, tmp_path):
    d = tmp_path / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_cached_scan_returns_same_result_within_ttl(server, clips):
    _clip(clips, "cam1", "1788792747_bird.mp4")

    first = server._scan_recordings_cached()
    _clip(clips, "cam1", "1788792748_animal.mp4")  # appears on disk after the first read
    second = server._scan_recordings_cached()

    assert len(first) == 1
    assert second is first, "within the TTL the cached list is reused verbatim"


def test_uncached_primitive_is_never_cached(server, clips):
    """_scan_recordings is the characterized primitive and must always hit the disk."""
    _clip(clips, "cam1", "1788792747_bird.mp4")
    assert len(server._scan_recordings()) == 1

    _clip(clips, "cam1", "1788792748_animal.mp4")
    assert len(server._scan_recordings()) == 2, "primitive must not be cached"


def test_invalidation_forces_a_rescan(server, clips):
    _clip(clips, "cam1", "1788792747_bird.mp4")
    assert len(server._scan_recordings_cached()) == 1

    _clip(clips, "cam1", "1788792748_animal.mp4")
    server._invalidate_scan_cache()

    assert len(server._scan_recordings_cached()) == 2


def test_cache_expires_after_ttl(server, clips, monkeypatch):
    monkeypatch.setattr(web_mod, "RECORDINGS_CACHE_TTL", 0.05)
    _clip(clips, "cam1", "1788792747_bird.mp4")
    assert len(server._scan_recordings_cached()) == 1

    _clip(clips, "cam1", "1788792748_animal.mp4")
    time.sleep(0.08)

    assert len(server._scan_recordings_cached()) == 2, "stale entry must expire"


def test_delete_invalidates_the_cache(server, clips):
    """A delete this process performs must be visible immediately, not after the TTL."""
    _clip(clips, "cam1", "1788792747_bird.mp4")
    _clip(clips, "cam1", "1788792748_animal.mp4")
    assert len(server._scan_recordings_cached()) == 2

    ok, _ = server._delete_file("cam1/2026/09/07/1788792747_bird.mp4")
    assert ok is True

    assert len(server._scan_recordings_cached()) == 1, "delete must not leave a stale list"


def test_cache_starts_cold(server, clips):
    assert server._scan_cache is None
    _clip(clips, "cam1", "1788792747_bird.mp4")
    assert len(server._scan_recordings_cached()) == 1
    assert server._scan_cache is not None


def test_cached_and_uncached_agree_on_a_cold_cache(server, clips):
    for i in range(5):
        _clip(clips, "cam1", f"178879274{i}_bird.mp4")

    server._invalidate_scan_cache()

    assert server._scan_recordings_cached() == server._scan_recordings()


def test_empty_archive_is_cached_without_error(server, clips):
    assert server._scan_recordings_cached() == []
    assert server._scan_recordings_cached() == []
