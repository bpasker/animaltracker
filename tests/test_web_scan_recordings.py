"""Characterization tests for WebServer._scan_recordings / _get_thumbnails_for_clip.

These pin the CURRENT behaviour of archive scanning and the clip/thumbnail
filename contract (web.py ~1748-1856) so that a rewrite which changes
behaviour fails loudly.  Quirks are asserted as-is and flagged with
`# QUIRK:` comments.
"""

import itertools
import os
import shutil

import pytest

from animaltracker import web as web_mod
from animaltracker.web import WebServer


# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------

_PORTS = itertools.count(19000)


@pytest.fixture
def make_server(tmp_path):
    """Build a real WebServer rooted at a fresh tmp dir.

    __init__ mkdirs storage_root/'clips', so the clips dir exists afterwards.
    """
    counter = itertools.count()

    def _make():
        root = tmp_path / f"storage{next(counter)}"
        return WebServer(
            {},
            root,
            tmp_path / "logs",
            port=next(_PORTS),
        )

    return _make


@pytest.fixture
def server(make_server):
    return make_server()


@pytest.fixture
def clips_dir(server):
    return server.storage_root / "clips"


def write_file(path, data=b"0123456789"):
    """Create a real (tiny) file, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def set_mtime(path, ts):
    os.utime(path, (ts, ts))
    return path


def by_filename(clips):
    return {c["filename"]: c for c in clips}


# Fixed epoch seconds used across tests (all well clear of DST edges).
T_2024_05_17 = 1715900000.0   # 2024-05-16 20:53:20 UTC
T_2023_01_02 = 1672660000.0
T_OLDER = 1600000000.0


# --------------------------------------------------------------------------
# empty / missing directory
# --------------------------------------------------------------------------

def test_scan_returns_empty_list_when_clips_dir_missing(server):
    """The clips dir is created by __init__; if it later disappears we get []."""
    shutil.rmtree(server.storage_root / "clips")
    assert not (server.storage_root / "clips").exists()

    assert server._scan_recordings() == []


def test_scan_returns_empty_list_for_empty_clips_dir(server, clips_dir):
    assert clips_dir.is_dir()
    assert server._scan_recordings() == []


def test_scan_ignores_empty_camera_directories(server, clips_dir):
    (clips_dir / "cam1" / "2024" / "05" / "17").mkdir(parents=True)

    assert server._scan_recordings() == []


def test_scan_ignores_non_mp4_files(server, clips_dir):
    write_file(clips_dir / "notes.txt")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.jpg")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mkv")

    assert server._scan_recordings() == []


# --------------------------------------------------------------------------
# automated clips: camera/year/month/day layout
# --------------------------------------------------------------------------

def test_automated_clip_fields(server, clips_dir):
    clip = write_file(
        clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mp4",
        b"abcd",
    )
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert set(result) == {
        "path",
        "camera",
        "date",
        "filename",
        "time",
        "size",
        "species",
        "raw_species",
        "thumbnails",
    }
    assert result["path"] == os.path.join("cam1", "2024", "05", "17", "1715900000_deer.mp4")
    assert result["camera"] == "cam1"
    assert result["filename"] == "1715900000_deer.mp4"
    assert result["size"] == 4
    assert result["species"] == "Deer"
    assert result["raw_species"] == "deer"
    assert result["thumbnails"] == []


def test_automated_clip_path_is_relative_to_clips_dir(server, clips_dir):
    write_file(clips_dir / "cam2" / "2024" / "05" / "17" / "1715900000_bird.mp4")

    (result,) = server._scan_recordings()

    assert not os.path.isabs(result["path"])
    assert (clips_dir / result["path"]).exists()


def test_camera_is_top_level_directory_name_at_any_depth(server, clips_dir):
    """rglob walks arbitrarily deep; 'camera' is always the top-level dir name."""
    write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "extra" / "1715900001_deer.mp4")

    cameras = {c["camera"] for c in server._scan_recordings()}

    assert cameras == {"cam1"}


def test_clip_with_no_underscore_in_name_is_unknown_species(server, clips_dir):
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000.mp4")

    (result,) = server._scan_recordings()

    assert result["species"] == "Unknown"
    assert result["raw_species"] == "unknown"


def test_clip_species_with_apostrophe_and_unicode_survives(server, clips_dir):
    clip = write_file(
        clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_o'brien’s éléphant.mp4"
    )
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert result["filename"] == "1715900000_o'brien’s éléphant.mp4"
    assert result["raw_species"] == "o'brien’s éléphant"
    # QUIRK: for an unmapped name the display species keeps only the LAST
    # word (spaces are normalised to underscores and the last meaningful
    # segment wins), so "o'brien’s éléphant" displays as just "Éléphant".
    assert result["species"] == "Éléphant"


# --------------------------------------------------------------------------
# time / date come from st_mtime
# --------------------------------------------------------------------------

def test_clip_time_comes_from_st_mtime(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mp4")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert result["time"].timestamp() == pytest.approx(T_2024_05_17)
    assert result["time"].tzinfo is not None
    assert result["time"].tzinfo is web_mod.CENTRAL_TZ


def test_changing_mtime_changes_reported_time_and_date(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mp4")

    set_mtime(clip, T_2024_05_17)
    first = server._scan_recordings()[0]

    set_mtime(clip, T_OLDER)
    second = server._scan_recordings()[0]

    assert first["time"] != second["time"]
    assert second["time"].timestamp() == pytest.approx(T_OLDER)
    assert second["date"] == second["time"].strftime("%Y-%m-%d")


def test_date_ignores_year_month_day_directories(server, clips_dir):
    # QUIRK: the y/m/d directory layout is decorative — 'date' is derived from
    # st_mtime, not from the path. A clip filed under 1999/01/01 but touched
    # today reports today. Asserted as-is to detect rewrite drift.
    clip = write_file(clips_dir / "cam1" / "1999" / "01" / "01" / "1715900000_deer.mp4")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    from datetime import datetime

    expected = datetime.fromtimestamp(T_2024_05_17, tz=web_mod.CENTRAL_TZ).strftime("%Y-%m-%d")
    assert result["date"] == expected
    assert not result["date"].startswith("1999")


def test_date_ignores_timestamp_embedded_in_filename(server, clips_dir):
    # QUIRK: the leading epoch in the filename is never parsed for the date.
    clip = write_file(clips_dir / "cam1" / "1600000000_deer.mp4")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert result["time"].timestamp() == pytest.approx(T_2024_05_17)


# --------------------------------------------------------------------------
# sort order
# --------------------------------------------------------------------------

def test_clips_sorted_by_time_descending(server, clips_dir):
    old = write_file(clips_dir / "cam1" / "1600000000_deer.mp4")
    mid = write_file(clips_dir / "cam1" / "1672660000_bird.mp4")
    new = write_file(clips_dir / "cam2" / "1715900000_raccoon.mp4")
    set_mtime(old, T_OLDER)
    set_mtime(mid, T_2023_01_02)
    set_mtime(new, T_2024_05_17)

    names = [c["filename"] for c in server._scan_recordings()]

    assert names == [
        "1715900000_raccoon.mp4",
        "1672660000_bird.mp4",
        "1600000000_deer.mp4",
    ]


def test_manual_clip_wins_ties_because_sort_is_stable(server, clips_dir):
    # QUIRK: on an exact mtime tie the manual (root-level) clip always sorts
    # first, because manual clips are appended before automated ones and
    # list.sort is stable. Asserted as-is to detect rewrite drift.
    manual = write_file(clips_dir / "manual_cam1_1715900000.mp4")
    auto = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    set_mtime(manual, T_2024_05_17)
    set_mtime(auto, T_2024_05_17)

    names = [c["filename"] for c in server._scan_recordings()]

    assert names == ["manual_cam1_1715900000.mp4", "1715900000_deer.mp4"]


def test_manual_and_automated_clips_share_one_sorted_list(server, clips_dir):
    manual = write_file(clips_dir / "manual_cam1_1600000000.mp4")
    auto = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    set_mtime(manual, T_OLDER)
    set_mtime(auto, T_2024_05_17)

    names = [c["filename"] for c in server._scan_recordings()]

    assert names == ["1715900000_deer.mp4", "manual_cam1_1600000000.mp4"]


# --------------------------------------------------------------------------
# manual clips: bare .mp4 files sitting directly in clips/
# --------------------------------------------------------------------------

def test_manual_clip_fields(server, clips_dir):
    clip = write_file(clips_dir / "manual_cam1_1715900000.mp4", b"xyz")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert result["path"] == "manual_cam1_1715900000.mp4"
    assert result["camera"] == "cam1"
    assert result["date"] == "Manual"       # literal, not a date string
    assert result["filename"] == "manual_cam1_1715900000.mp4"
    assert result["species"] == "Manual clip"
    assert result["raw_species"] == "manual"
    assert result["size"] == 3
    assert result["thumbnails"] == []
    assert result["time"].timestamp() == pytest.approx(T_2024_05_17)


def test_manual_clip_camera_is_second_underscore_segment(server, clips_dir):
    write_file(clips_dir / "snapshot_cam42_extra_1715900000.mp4")

    (result,) = server._scan_recordings()

    assert result["camera"] == "cam42"


def test_manual_clip_camera_keeps_extension_when_only_two_segments(server, clips_dir):
    # QUIRK: camera is filename.split('_')[1] with no extension stripping, so a
    # two-segment name yields a "camera" that still ends in .mp4.
    write_file(clips_dir / "manual_cam1.mp4")

    (result,) = server._scan_recordings()

    assert result["camera"] == "cam1.mp4"


def test_manual_clip_without_underscore_is_unknown_camera(server, clips_dir):
    write_file(clips_dir / "clip.mp4")

    (result,) = server._scan_recordings()

    assert result["camera"] == "unknown"


def test_manual_clip_trailing_underscore_yields_extension_as_camera(server, clips_dir):
    # QUIRK: "_cam1.mp4" splits to ['', 'cam1.mp4'] so camera is 'cam1.mp4';
    # but "cam1_.mp4" splits to ['cam1', '.mp4'] giving a camera of '.mp4'.
    write_file(clips_dir / "cam1_.mp4")

    (result,) = server._scan_recordings()

    assert result["camera"] == ".mp4"


def test_manual_clip_species_is_never_parsed_from_filename(server, clips_dir):
    # QUIRK: even a fully species-tagged name in the root is labelled
    # 'Manual clip' / 'manual' — the manual path never calls the species parser.
    write_file(clips_dir / "1715900000_bird_passeriformes_cardinalidae.mp4")

    (result,) = server._scan_recordings()

    assert result["species"] == "Manual clip"
    assert result["raw_species"] == "manual"
    # ...and the camera is the first species token, not a camera id.
    assert result["camera"] == "bird"


def test_manual_clips_never_get_thumbnails(server, clips_dir):
    # QUIRK: matching thumbnails exist on disk but the manual path hardcodes [].
    clip = write_file(clips_dir / "manual_cam1_1715900000.mp4")
    write_file(clips_dir / "manual_cam1_1715900000_thumb_deer.jpg")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert result["thumbnails"] == []
    # ...even though the helper would have found it.
    assert len(server._get_thumbnails_for_clip(clip)) == 1


def test_root_glob_is_not_recursive(server, clips_dir):
    """A bare .mp4 inside a camera dir is an automated clip, not a manual one."""
    write_file(clips_dir / "cam1" / "manual_cam1_1715900000.mp4")

    (result,) = server._scan_recordings()

    assert result["date"] != "Manual"
    assert result["camera"] == "cam1"
    assert result["species"] != "Manual clip"


# --------------------------------------------------------------------------
# _get_thumbnails_for_clip
# --------------------------------------------------------------------------

def test_no_thumbnails_returns_empty_list(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")

    assert server._get_thumbnails_for_clip(clip) == []


def test_thumbnail_basic_form_fields(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer_thumb_cervidae.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert set(thumb) == {"path", "species", "url"}       # no track_index key
    assert thumb["path"] == os.path.join(
        "cam1", "2024", "05", "17", "1715900000_deer_thumb_cervidae.jpg"
    )
    assert thumb["species"] == "Deer"
    assert thumb["url"] == "/clips/" + thumb["path"]


def test_thumbnail_url_uses_os_path_separator(server, clips_dir):
    # QUIRK: the URL is built from str(Path) so it inherits the OS separator
    # rather than always using '/'.
    clip = write_file(clips_dir / "cam1" / "2024" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "2024" / "1715900000_deer_thumb_deer.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert thumb["url"] == "/clips/" + str(os.path.join("cam1", "2024", "1715900000_deer_thumb_deer.jpg"))


def test_thumbnail_track_index_form(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_bird.mp4")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae_t0.jpg")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae_t1.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [t["track_index"] for t in thumbs] == [0, 1]
    assert [t["species"] for t in thumbs] == ["Crow/Jay", "Crow/Jay"]


def test_thumbnail_track_index_sorts_numerically_not_lexically(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_bird.mp4")
    for idx in (10, 2, 1):
        write_file(clips_dir / "cam1" / f"1715900000_bird_thumb_corvidae_t{idx}.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [t["track_index"] for t in thumbs] == [1, 2, 10]


def test_thumbnail_legacy_trailing_numeric_form_has_no_track_index(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_bird.mp4")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_bird_1.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert "track_index" not in thumb
    # the legacy "_1" is stripped before the species lookup
    assert thumb["species"] == "Bird"


def test_thumbnail_legacy_multi_digit_suffix_stripped(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_cervidae_12.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert "track_index" not in thumb
    assert thumb["species"] == "Deer"


def test_untracked_thumbnails_sort_after_tracked_ones(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_bird.mp4")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae.jpg")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae_t3.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [t.get("track_index") for t in thumbs] == [3, None]


def test_track_index_above_sentinel_sorts_after_untracked(server, clips_dir):
    # QUIRK: untracked thumbnails use a literal 999 sort sentinel, so a track
    # index of 1000 or more sorts *after* the untracked ones.
    clip = write_file(clips_dir / "cam1" / "1715900000_bird.mp4")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae.jpg")
    write_file(clips_dir / "cam1" / "1715900000_bird_thumb_corvidae_t1000.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [t.get("track_index") for t in thumbs] == [None, 1000]


def test_thumbnails_with_equal_track_index_sort_by_path(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_x.mp4")
    write_file(clips_dir / "cam1" / "1715900000_x_thumb_zebra_t0.jpg")
    write_file(clips_dir / "cam1" / "1715900000_x_thumb_alpaca_t0.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [os.path.basename(t["path"]) for t in thumbs] == [
        "1715900000_x_thumb_alpaca_t0.jpg",
        "1715900000_x_thumb_zebra_t0.jpg",
    ]


def test_untracked_thumbnails_sort_by_path(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_x.mp4")
    write_file(clips_dir / "cam1" / "1715900000_x_thumb_zebra.jpg")
    write_file(clips_dir / "cam1" / "1715900000_x_thumb_alpaca.jpg")

    thumbs = server._get_thumbnails_for_clip(clip)

    assert [os.path.basename(t["path"]) for t in thumbs] == [
        "1715900000_x_thumb_alpaca.jpg",
        "1715900000_x_thumb_zebra.jpg",
    ]


def test_thumbnail_uses_last_thumb_marker_when_repeated(server, clips_dir):
    # QUIRK: species is taken from the LAST "_thumb_" split, so a clip whose own
    # name contains "_thumb_" has its species read from the tail segment.
    clip = write_file(clips_dir / "cam1" / "1715900000_thumb_bogus.mp4")
    write_file(clips_dir / "cam1" / "1715900000_thumb_bogus_thumb_cervidae.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert thumb["species"] == "Deer"


def test_thumbnail_with_empty_species_segment_is_unknown(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    assert thumb["species"] == "Unknown"
    assert "track_index" not in thumb


def test_thumbnail_species_with_apostrophe_and_unicode(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_o'brien éléphant.jpg")

    (thumb,) = server._get_thumbnails_for_clip(clip)

    # QUIRK: same last-segment-only fallback as for clip species names.
    assert thumb["species"] == "Éléphant"
    assert thumb["path"].endswith("1715900000_deer_thumb_o'brien éléphant.jpg")


def test_thumbnails_do_not_leak_between_clips(server, clips_dir):
    clip_a = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    clip_b = write_file(clips_dir / "cam1" / "1715900001_bird.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_cervidae.jpg")
    write_file(clips_dir / "cam1" / "1715900001_bird_thumb_corvidae.jpg")

    assert [t["species"] for t in server._get_thumbnails_for_clip(clip_a)] == ["Deer"]
    assert [t["species"] for t in server._get_thumbnails_for_clip(clip_b)] == ["Crow/Jay"]


def test_thumbnails_must_live_beside_the_clip(server, clips_dir):
    """Only clip_path.parent is globbed — a thumb one dir up is not found."""
    clip = write_file(clips_dir / "cam1" / "2024" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_cervidae.jpg")

    assert server._get_thumbnails_for_clip(clip) == []


def test_only_jpg_thumbnails_are_matched(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "1715900000_deer.mp4")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_cervidae.png")
    write_file(clips_dir / "cam1" / "1715900000_deer_thumb_cervidae.jpeg")

    assert server._get_thumbnails_for_clip(clip) == []


def test_thumbnails_are_attached_to_automated_clips_by_scan(server, clips_dir):
    clip = write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_bird.mp4")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_bird_thumb_corvidae_t1.jpg")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_bird_thumb_corvidae_t0.jpg")
    set_mtime(clip, T_2024_05_17)

    (result,) = server._scan_recordings()

    assert [t["track_index"] for t in result["thumbnails"]] == [0, 1]
    assert all(t["url"].startswith("/clips/") for t in result["thumbnails"])


# --------------------------------------------------------------------------
# whole-tree smoke test
# --------------------------------------------------------------------------

def test_full_tree_scan(server, clips_dir):
    manual = write_file(clips_dir / "manual_cam2_1600000000.mp4")
    a = write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer.mp4")
    b = write_file(clips_dir / "cam2" / "2023" / "01" / "02" / "1672660000_bird_passeriformes_cardinalidae.mp4")
    write_file(clips_dir / "cam1" / "2024" / "05" / "17" / "1715900000_deer_thumb_cervidae_t0.jpg")
    set_mtime(manual, T_OLDER)
    set_mtime(a, T_2024_05_17)
    set_mtime(b, T_2023_01_02)

    clips = server._scan_recordings()

    assert [c["camera"] for c in clips] == ["cam1", "cam2", "cam2"]
    found = by_filename(clips)
    assert found["1672660000_bird_passeriformes_cardinalidae.mp4"]["species"] == "Cardinal"
    assert found["1672660000_bird_passeriformes_cardinalidae.mp4"]["raw_species"] == (
        "bird_passeriformes_cardinalidae"
    )
    assert found["manual_cam2_1600000000.mp4"]["date"] == "Manual"
    assert len(found["1715900000_deer.mp4"]["thumbnails"]) == 1
