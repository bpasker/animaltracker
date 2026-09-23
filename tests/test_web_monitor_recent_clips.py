"""The monitor's recent-clips panel lists the newest clips in the archive.

It used to glob ``clips/<camera>/*.mp4``, one directory above where the
pipeline actually writes (``clips/<camera>/<date>/``), so on the production
layout it matched nothing and the panel was permanently empty.  It also put
the whole ``(display, raw)`` tuple from ``_parse_species_from_filename`` into
'species', which would have rendered as "Deer,deer" had anything ever matched.

Found in the bug hunt of 2026-09-19.  Fixed by reusing the archive's own scan.
"""
from __future__ import annotations

import itertools
import os

import pytest

from animaltracker.web import WebServer

_PORTS = itertools.count(19700)


@pytest.fixture
def server(tmp_path):
    return WebServer({}, tmp_path / "storage", tmp_path / "logs", port=next(_PORTS))


def write_clip(server, camera, date, name, epoch):
    path = server.storage_root / "clips" / camera / date / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0123456789")
    os.utime(path, (epoch, epoch))
    return path


def recent(server):
    return server._monitor_host_stats()[3]


# --------------------------------------------------------------------------


def test_the_panel_finds_clips_in_the_dated_directories_the_pipeline_writes(server):
    write_clip(server, "cam1", "2026-09-19", "1789000000_bird.mp4", 1789000000)

    got = recent(server)

    assert [c["path"] for c in got] == ["cam1/2026-09-19/1789000000_bird.mp4"]
    assert got[0]["camera"] == "cam1"


def test_species_is_the_display_name_alone_not_the_parse_tuple(server):
    write_clip(server, "cam1", "2026-09-19", "1789000000_mammalia_artiodactyla_cervidae.mp4",
               1789000000)

    got = recent(server)

    assert got[0]["species"] == "Deer"


def test_it_shows_the_five_newest_clips_newest_first(server):
    for i in range(8):
        write_clip(server, "cam1", "2026-09-19", f"178900000{i}_bird.mp4", 1789000000 + i)

    got = recent(server)

    assert len(got) == 5
    assert [c["path"].split("_")[0].split("/")[-1] for c in got] == [
        "1789000007", "1789000006", "1789000005", "1789000004", "1789000003",
    ]


def test_the_time_is_a_formatted_clock_string(server):
    write_clip(server, "cam1", "2026-09-19", "1789000000_bird.mp4", 1789000000)

    stamp = recent(server)[0]["time"]

    assert len(stamp) == 8 and stamp.count(":") == 2


def test_an_empty_archive_gives_an_empty_panel_not_an_error(server):
    (server.storage_root / "clips").mkdir(parents=True, exist_ok=True)

    assert recent(server) == []


# --- one figure failing does not blank the others ----------------------------------------------

def test_a_worker_without_a_detector_name_still_leaves_the_host_figures_and_the_clips(server):
    """Every figure is its own attempt. One AttributeError used to be caught
    by the handler and answered with all of them blank, which is what hid
    the empty panel in the dev harness (its stand-in worker had no
    detector) and would hide a real archive behind a stand-in too."""
    from types import SimpleNamespace
    write_clip(server, "cam1", "2026-09-19", "1789000000_bird.mp4", 1789000000)
    server.workers = {"cam1": SimpleNamespace()}          # no .detector, no .runtime

    system, gpu, detector_info, recent = server._monitor_host_stats()

    assert detector_info == {"backend": "unknown", "country": None}
    assert system["disk_total_gb"] > 0, "the host figures were blanked with it"
    assert [c["species"] for c in recent] == ["Bird"]


def test_a_failing_archive_listing_leaves_the_host_figures(server, monkeypatch):
    def boom():
        raise OSError("Stale file handle")
    monkeypatch.setattr(server, "_scan_recordings_cached", boom)
    (server.storage_root / "clips").mkdir(parents=True, exist_ok=True)

    system, gpu, detector_info, recent = server._monitor_host_stats()

    assert recent == [] and system["disk_total_gb"] > 0


def test_polling_the_monitor_does_not_rescan_the_archive_every_few_seconds(server, monkeypatch):
    # Bug hunt, 2026-09-22: the page polls every 2 s and the scan cache
    # lives 3 s, so an open monitor walked the whole NFS archive about every
    # 4 s. The panel now keeps its clips for RECENT_CLIPS_TTL.
    from animaltracker import web as web_mod

    write_clip(server, "cam1", "2026-09-19", "1789000000_bird.mp4", 1789000000)
    scans = []
    real = server._scan_recordings
    monkeypatch.setattr(server, "_scan_recordings", lambda: scans.append(1) or real())
    clock = [1000.0]
    monkeypatch.setattr(web_mod._time, "monotonic", lambda: clock[0])

    for _ in range(10):                 # twenty seconds of polling
        recent(server)
        clock[0] += 2.0
    assert len(scans) == 1

    clock[0] += web_mod.RECENT_CLIPS_TTL
    recent(server)
    assert len(scans) == 2


def test_a_delete_through_the_server_shows_at_the_next_poll(server):
    clip = write_clip(server, "cam1", "2026-09-19", "1789000000_bird.mp4", 1789000000)
    assert recent(server)
    clip.unlink()
    server._invalidate_scan_cache()
    assert recent(server) == []
