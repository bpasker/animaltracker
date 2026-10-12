"""The Monitor shows the disk events record to, and what is waiting on it.

Every event streams into ``logs_root/event_temp`` and reaches the archive only
when it closes. On 2026-10-09 the archive filled, the recordings it refused
stayed in ``event_temp`` and filled the local disk, and every clip for three
days stopped where its file ran out of room, while the Monitor's only disk
gauge, the archive's, read 84%.
"""
from __future__ import annotations

import itertools
import pytest

from animaltracker import web
from animaltracker.storage import StorageManager
from animaltracker.web import WebServer

_PORTS = itertools.count(19800)
EVENT_TS = 1789000000


@pytest.fixture
def storage(tmp_path):
    return StorageManager(storage_root=tmp_path / "storage", logs_root=tmp_path / "logs")


def _server(storage) -> WebServer:
    return WebServer({}, storage.storage_root, storage.logs_root, port=next(_PORTS), storage=storage)


def _system(server: WebServer) -> dict:
    return server._monitor_host_stats()[0]


def test_one_disk_for_both_shows_no_second_gauge(storage):
    assert _system(_server(storage))["recording_disk"] is None


def test_a_separate_recording_disk_is_reported_with_its_free_space(storage, monkeypatch):
    monkeypatch.setattr(web, "_same_filesystem", lambda a, b: False)   # the archive on NFS

    disk = _system(_server(storage))["recording_disk"]

    assert set(disk) == {"percent", "used_gb", "total_gb", "free_gb"}
    assert 0 <= disk["percent"] <= 100 and disk["free_gb"] <= disk["total_gb"]


def test_recordings_not_saved_yet_are_counted(storage):
    tmp = storage.logs_root / "event_temp"
    tmp.mkdir(parents=True)
    kept = tmp / f"cam1_{EVENT_TS}_0a1b2c3d.temp.avi"
    kept.write_bytes(b"x" * 1024)
    live = tmp / f"cam1_{EVENT_TS + 60}_1b2c3d4e.temp.avi"   # an open event: not counted
    live.write_bytes(b"x" * 1024)
    storage.keep_recording(kept)

    assert _system(_server(storage))["unsaved_recordings"] == {"count": 1, "gb": 0.0}


def test_without_a_pipeline_nothing_is_waiting(storage):
    server = WebServer({}, storage.storage_root, storage.logs_root, port=next(_PORTS))

    assert _system(server)["unsaved_recordings"] == {"count": 0, "gb": 0}
