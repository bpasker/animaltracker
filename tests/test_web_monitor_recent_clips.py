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
