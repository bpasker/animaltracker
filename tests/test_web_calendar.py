"""Characterization tests for the recordings calendar aggregation in web.py.

These tests pin the CURRENT behaviour of:
  - WebServer._build_calendar_data  (web.py ~1915-1994)
  - WebServer._get_clips_for_date   (web.py ~2000-2078)

They are a safety net for a rewrite: anything marked "# QUIRK:" is odd/buggy
today, but is asserted exactly as-is so a rewrite that changes it fails loudly.
"""

import os
from datetime import datetime

import pytest

from animaltracker import web as web_mod
from animaltracker.web import WebServer


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

_PORT = [18800]


def _next_port():
    _PORT[0] += 1
    return _PORT[0]


@pytest.fixture
def server(tmp_path):
    """A WebServer with no cameras, rooted at pytest tmp dirs."""
    storage = tmp_path / "storage"
    logs = tmp_path / "logs"
    storage.mkdir()
    logs.mkdir()
    return WebServer({}, storage, logs, port=_next_port())


def clip(time_str, camera="cam1", species="Unknown", **extra):
    """Build a clip dict shaped like _scan_recordings output.

    time_str: "YYYY-MM-DD HH:MM:SS"
    """
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=web_mod.CENTRAL_TZ
    )
    data = {
        "path": extra.pop("path", f"{camera}/{time_str}.mp4"),
        "camera": camera,
        "date": dt.strftime("%Y-%m-%d"),
        "filename": extra.pop("filename", f"{time_str}.mp4"),
        "time": dt,
        "size": extra.pop("size", 1024 * 1024),
        "species": species,
        "raw_species": extra.pop("raw_species", "unknown"),
        "thumbnails": extra.pop("thumbnails", []),
    }
    data.update(extra)
    return data


# --------------------------------------------------------------------------
# _build_calendar_data — empty / trivial archives
# --------------------------------------------------------------------------

def test_calendar_empty_archive_returns_empty_years_and_filters(server):
    assert server._build_calendar_data([]) == {
        "years": {},
        "filters": {"cameras": [], "species": []},
    }


def test_calendar_single_clip_full_day_shape(server):
    data = server._build_calendar_data(
        [clip("2024-03-05 07:42:11", camera="cam2", species="Whitetail Deer")]
    )

    assert data["years"]["2024"]["total"] == 1
    assert data["years"]["2024"]["months"]["3"]["total"] == 1
    day = data["years"]["2024"]["months"]["3"]["days"]["5"]
    assert day == {
        "count": 1,
        "species": ["Whitetail Deer"],
        "cameras": ["cam2"],
        "first_clip_time": "07:42",
        "last_clip_time": "07:42",
    }
    # The internal 'clips' bucket accumulated per day is dropped from the output.
    assert "clips" not in day
    assert data["filters"] == {"cameras": ["cam2"], "species": ["Whitetail Deer"]}


def test_calendar_keys_are_strings_not_ints(server):
    data = server._build_calendar_data([clip("2024-03-05 07:42:11")])
    assert list(data["years"].keys()) == ["2024"]
    assert list(data["years"]["2024"]["months"].keys()) == ["3"]
    assert list(data["years"]["2024"]["months"]["3"]["days"].keys()) == ["5"]
    # No zero padding: March is "3", not "03"; the 5th is "5", not "05".
    assert "03" not in data["years"]["2024"]["months"]


# --------------------------------------------------------------------------
# _build_calendar_data — descending insertion order
# --------------------------------------------------------------------------

def test_calendar_years_months_days_in_descending_insertion_order(server):
    clips = [
        clip("2023-01-02 01:00:00"),
        clip("2024-12-31 23:00:00"),
        clip("2024-01-01 00:00:00"),
        clip("2024-12-01 12:00:00"),
        clip("2024-12-15 12:00:00"),
    ]
    data = server._build_calendar_data(clips)

    assert list(data["years"].keys()) == ["2024", "2023"]
    assert list(data["years"]["2024"]["months"].keys()) == ["12", "1"]
    assert list(data["years"]["2024"]["months"]["12"]["days"].keys()) == [
        "31",
        "15",
        "1",
    ]


def test_calendar_month_ordering_is_numeric_not_lexicographic(server):
    """Months sort as ints before stringification, so 10/9/2 not "10"/"2"/"9"."""
    clips = [
        clip("2024-02-01 10:00:00"),
        clip("2024-09-01 10:00:00"),
        clip("2024-10-01 10:00:00"),
    ]
    data = server._build_calendar_data(clips)
    assert list(data["years"]["2024"]["months"].keys()) == ["10", "9", "2"]


def test_calendar_day_ordering_is_numeric_not_lexicographic(server):
    clips = [
        clip("2024-05-02 10:00:00"),
        clip("2024-05-09 10:00:00"),
        clip("2024-05-21 10:00:00"),
    ]
    data = server._build_calendar_data(clips)
    assert list(data["years"]["2024"]["months"]["5"]["days"].keys()) == [
        "21",
        "9",
        "2",
    ]


def test_calendar_input_order_does_not_change_output_order(server):
    """Output order comes from sorted(), not from the order clips arrive."""
    forward = [
        clip("2024-01-01 10:00:00"),
        clip("2024-06-15 10:00:00"),
        clip("2025-02-02 10:00:00"),
    ]
    data_a = server._build_calendar_data(forward)
    data_b = server._build_calendar_data(list(reversed(forward)))
    assert list(data_a["years"].keys()) == list(data_b["years"].keys()) == [
        "2025",
        "2024",
    ]
    assert list(data_a["years"]["2024"]["months"].keys()) == ["6", "1"]
    assert list(data_b["years"]["2024"]["months"].keys()) == ["6", "1"]


# --------------------------------------------------------------------------
# _build_calendar_data — counts, boundaries
# --------------------------------------------------------------------------

def test_calendar_totals_roll_up_across_month_and_year_boundary(server):
    clips = [
        clip("2023-12-31 23:59:59"),
        clip("2024-01-01 00:00:00"),
        clip("2024-01-01 00:00:01"),
        clip("2024-02-01 00:00:00"),
    ]
    data = server._build_calendar_data(clips)

    assert data["years"]["2023"]["total"] == 1
    assert data["years"]["2024"]["total"] == 3
    assert data["years"]["2024"]["months"]["1"]["total"] == 2
    assert data["years"]["2024"]["months"]["2"]["total"] == 1
    assert data["years"]["2024"]["months"]["1"]["days"]["1"]["count"] == 2
    assert data["years"]["2023"]["months"]["12"]["days"]["31"]["count"] == 1


def test_calendar_leap_day_is_its_own_bucket(server):
    data = server._build_calendar_data([clip("2024-02-29 12:00:00")])
    assert list(data["years"]["2024"]["months"]["2"]["days"].keys()) == ["29"]


def test_calendar_day_number_collides_across_months(server):
    """Day 1 of Jan and day 1 of Feb stay separate (keyed under their month)."""
    clips = [clip("2024-01-01 05:00:00"), clip("2024-02-01 06:00:00")]
    data = server._build_calendar_data(clips)
    months = data["years"]["2024"]["months"]
    assert months["1"]["days"]["1"]["first_clip_time"] == "05:00"
    assert months["2"]["days"]["1"]["first_clip_time"] == "06:00"


# --------------------------------------------------------------------------
# _build_calendar_data — first/last clip time
# --------------------------------------------------------------------------

def test_calendar_first_and_last_clip_time_span_the_day(server):
    clips = [
        clip("2024-04-10 13:05:00"),
        clip("2024-04-10 06:30:00"),
        clip("2024-04-10 23:59:00"),
    ]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["first_clip_time"] == "06:30"
    assert day["last_clip_time"] == "23:59"


def test_calendar_clip_times_are_minute_resolution_only(server):
    # QUIRK: first/last clip times are '%H:%M' strings compared lexicographically,
    # so seconds are discarded entirely — asserted as-is to detect rewrite drift.
    clips = [clip("2024-04-10 08:00:59"), clip("2024-04-10 08:00:01")]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["first_clip_time"] == "08:00"
    assert day["last_clip_time"] == "08:00"
    assert day["count"] == 2


def test_calendar_midnight_clip_sets_first_time_to_0000(server):
    clips = [clip("2024-04-10 15:00:00"), clip("2024-04-10 00:00:00")]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["first_clip_time"] == "00:00"
    assert day["last_clip_time"] == "15:00"


# --------------------------------------------------------------------------
# _build_calendar_data — species fan-out
# --------------------------------------------------------------------------

def test_calendar_multi_species_display_fans_out_on_comma_space(server):
    clips = [clip("2024-04-10 09:00:00", species="Cardinal, Blue Jay, Squirrel")]
    data = server._build_calendar_data(clips)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["Blue Jay", "Cardinal", "Squirrel"]
    assert day["count"] == 1  # one clip, three species buckets
    assert data["filters"]["species"] == ["Blue Jay", "Cardinal", "Squirrel"]


def test_calendar_species_split_requires_comma_space_exactly(server):
    # QUIRK: the split is on ', ' (comma + space) only. A display string using a
    # bare comma stays a single species bucket — asserted as-is to detect drift.
    clips = [clip("2024-04-10 09:00:00", species="Cardinal,Blue Jay")]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["Cardinal,Blue Jay"]


def test_calendar_species_fragments_are_not_stripped(server):
    # QUIRK: splitting "A ,B" on ', ' yields "A ," and "B" — no whitespace/comma
    # cleanup is done on the fragments. Asserted as-is to detect rewrite drift.
    clips = [clip("2024-04-10 09:00:00", species="Deer ,Fox")]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["Deer ,Fox"]


def test_calendar_unknown_species_is_excluded_from_buckets(server):
    clips = [clip("2024-04-10 09:00:00", species="Unknown")]
    data = server._build_calendar_data(clips)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["count"] == 1
    assert day["species"] == []
    assert data["filters"]["species"] == []
    # The camera still shows up even with no species.
    assert day["cameras"] == ["cam1"]


def test_calendar_unknown_is_dropped_from_a_mixed_species_string(server):
    clips = [clip("2024-04-10 09:00:00", species="Deer, Unknown")]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["Deer"]


def test_calendar_unknown_exclusion_is_case_sensitive(server):
    # QUIRK: only the exact string 'Unknown' is filtered; 'unknown' / 'UNKNOWN'
    # survive as real species buckets — asserted as-is to detect rewrite drift.
    clips = [
        clip("2024-04-10 09:00:00", species="unknown"),
        clip("2024-04-10 09:05:00", species="UNKNOWN", camera="cam2"),
    ]
    data = server._build_calendar_data(clips)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["UNKNOWN", "unknown"]
    assert data["filters"]["species"] == ["UNKNOWN", "unknown"]


def test_calendar_empty_species_string_produces_no_bucket(server):
    clips = [clip("2024-04-10 09:00:00", species="")]
    data = server._build_calendar_data(clips)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == []
    assert data["filters"]["species"] == []


def test_calendar_missing_species_key_defaults_to_unknown(server):
    c = clip("2024-04-10 09:00:00")
    del c["species"]
    data = server._build_calendar_data([c])
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["count"] == 1
    assert day["species"] == []


def test_calendar_species_are_sorted_ascending_unlike_the_date_keys(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Zebra"),
        clip("2024-04-10 09:01:00", species="Aardvark"),
        clip("2024-04-10 09:02:00", species="mole"),
    ]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    # QUIRK: plain sorted() so uppercase sorts before lowercase ("mole" last).
    assert day["species"] == ["Aardvark", "Zebra", "mole"]


def test_calendar_species_with_apostrophes_and_unicode_survive_intact(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Cooper's Hawk"),
        clip("2024-04-10 09:01:00", species='He said "hi"', camera="cam2"),
        clip("2024-04-10 09:02:00", species="Coyote – Canyón \U0001f43a"),
    ]
    data = server._build_calendar_data(clips)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == [
        "Cooper's Hawk",
        "Coyote – Canyón \U0001f43a",
        'He said "hi"',
    ]
    assert data["filters"]["species"] == day["species"]


def test_calendar_species_deduplicated_per_day_but_counted_per_clip(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Deer"),
        clip("2024-04-10 10:00:00", species="Deer"),
    ]
    day = server._build_calendar_data(clips)["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["species"] == ["Deer"]
    assert day["count"] == 2


# --------------------------------------------------------------------------
# _build_calendar_data — cameras
# --------------------------------------------------------------------------

def test_calendar_cameras_dedup_and_sort_per_day_and_globally(server):
    clips = [
        clip("2024-04-10 09:00:00", camera="cam2"),
        clip("2024-04-10 09:01:00", camera="cam1"),
        clip("2024-04-10 09:02:00", camera="cam2"),
        clip("2024-04-11 09:02:00", camera="cam3"),
    ]
    data = server._build_calendar_data(clips)
    days = data["years"]["2024"]["months"]["4"]["days"]
    assert days["10"]["cameras"] == ["cam1", "cam2"]
    assert days["11"]["cameras"] == ["cam3"]
    assert data["filters"]["cameras"] == ["cam1", "cam2", "cam3"]


def test_calendar_camera_named_unknown_is_not_filtered(server):
    # Only species get the 'Unknown' filter; a camera literally named 'unknown'
    # (what _scan_recordings assigns to unparsable manual clips) is kept.
    clips = [clip("2024-04-10 09:00:00", camera="unknown")]
    data = server._build_calendar_data(clips)
    assert data["filters"]["cameras"] == ["unknown"]


# --------------------------------------------------------------------------
# _build_calendar_data — manual clips
# --------------------------------------------------------------------------

def test_calendar_manual_clips_leak_into_days_and_species_buckets(server):
    """Manual clips carry date='Manual' and species='Manual clip'.

    QUIRK: _build_calendar_data ignores clip['date'] entirely and uses
    clip['time'], so manual clips land on a normal calendar day; and because
    'Manual clip' != 'Unknown' it becomes a first-class species bucket and a
    selectable filter option — asserted as-is to detect rewrite drift.
    """
    manual = clip("2024-04-10 09:00:00", camera="cam1", species="Manual clip")
    manual["date"] = "Manual"
    manual["raw_species"] = "manual"

    data = server._build_calendar_data([manual, clip("2024-04-10 10:00:00", species="Deer")])
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["count"] == 2
    assert day["species"] == ["Deer", "Manual clip"]
    assert data["filters"]["species"] == ["Deer", "Manual clip"]


def test_calendar_manual_clip_from_real_scan_leaks_into_calendar(server, tmp_path):
    """End-to-end through _scan_recordings: a root-level mp4 is a manual clip."""
    clips_dir = server.storage_root / "clips"
    f = clips_dir / "manual_cam7_20240410.mp4"
    f.write_bytes(b"x" * 2048)
    ts = datetime(2024, 4, 10, 9, 0, 0, tzinfo=web_mod.CENTRAL_TZ).timestamp()
    os.utime(f, (ts, ts))

    scanned = server._scan_recordings()
    assert len(scanned) == 1
    assert scanned[0]["date"] == "Manual"
    assert scanned[0]["species"] == "Manual clip"
    # Camera comes from the *second* underscore-separated token of the filename.
    assert scanned[0]["camera"] == "cam7"

    data = server._build_calendar_data(scanned)
    day = data["years"]["2024"]["months"]["4"]["days"]["10"]
    assert day["count"] == 1
    assert day["species"] == ["Manual clip"]
    assert day["cameras"] == ["cam7"]
    assert data["filters"] == {"cameras": ["cam7"], "species": ["Manual clip"]}


# --------------------------------------------------------------------------
# _get_clips_for_date — basic shape
# --------------------------------------------------------------------------

def test_get_clips_for_date_empty_input(server):
    result = server._get_clips_for_date([], "2024-04-10")
    assert result == {
        "date": "2024-04-10",
        "clips": [],
        "summary": {
            "total": 0,
            "by_species": {},
            "by_camera": {},
            "by_hour": {},
            "peak_hour": None,
        },
    }


def test_get_clips_for_date_no_match_still_echoes_requested_date(server):
    result = server._get_clips_for_date(
        [clip("2024-04-11 09:00:00")], "2024-04-10"
    )
    assert result["date"] == "2024-04-10"
    assert result["clips"] == []
    assert result["summary"]["peak_hour"] is None


def test_get_clips_for_date_unvalidated_date_string_is_echoed(server):
    # QUIRK: _get_clips_for_date does no date validation of its own (the HTTP
    # handler does it); garbage simply matches nothing and is echoed back.
    result = server._get_clips_for_date([clip("2024-04-10 09:00:00")], "not-a-date")
    assert result["date"] == "not-a-date"
    assert result["clips"] == []


def test_get_clips_for_date_single_clip_full_shape(server):
    c = clip(
        "2024-04-10 14:05:09",
        camera="cam2",
        species="Whitetail Deer",
        raw_species="mammalia_cervidae_odocoileus_virginianus",
        size=3_145_728,
        path="cam2/2024/04/10/1712775909_deer.mp4",
        filename="1712775909_deer.mp4",
        thumbnails=[
            {"path": "cam2/a_thumb_deer.jpg", "species": "Deer", "url": "/clips/x"}
        ],
    )
    result = server._get_clips_for_date([c], "2024-04-10")
    assert result["summary"]["total"] == 1
    got = result["clips"][0]
    assert got == {
        "path": "cam2/2024/04/10/1712775909_deer.mp4",
        "camera": "cam2",
        "time": "14:05:09",
        "time_display": "02:05 PM",
        "hour": 14,
        "species": "Whitetail Deer",
        "raw_species": "mammalia_cervidae_odocoileus_virginianus",
        "species_icon": got["species_icon"],
        "size_mb": 3.0,
        "filename": "1712775909_deer.mp4",
        "thumbnails": [{"url": "/clips/cam2/a_thumb_deer.jpg", "species": "Deer"}],
    }
    # 'date' and 'size' from the scan record are not carried through.
    assert "date" not in got and "size" not in got


def test_get_clips_for_date_thumbnail_url_is_rebuilt_ignoring_scanned_url(server):
    # QUIRK: _scan_recordings already stores a 'url' on each thumbnail, but this
    # method ignores it and re-derives "/clips/{path}" — asserted as-is.
    c = clip(
        "2024-04-10 09:00:00",
        thumbnails=[{"path": "cam1/t.jpg", "species": "Fox", "url": "/WRONG"}],
    )
    got = server._get_clips_for_date([c], "2024-04-10")["clips"][0]
    assert got["thumbnails"] == [{"url": "/clips/cam1/t.jpg", "species": "Fox"}]


def test_get_clips_for_date_missing_thumbnails_and_raw_species_defaults(server):
    c = clip("2024-04-10 09:00:00")
    del c["thumbnails"]
    del c["raw_species"]
    del c["species"]
    got = server._get_clips_for_date([c], "2024-04-10")["clips"][0]
    assert got["thumbnails"] == []
    assert got["raw_species"] == "unknown"
    assert got["species"] == "Unknown"
    assert got["species_icon"] == "❓"


def test_get_clips_for_date_size_mb_rounds_to_two_places(server):
    c = clip("2024-04-10 09:00:00", size=1)
    assert server._get_clips_for_date([c], "2024-04-10")["clips"][0]["size_mb"] == 0.0
    c = clip("2024-04-10 09:00:00", size=1_572_864)
    assert server._get_clips_for_date([c], "2024-04-10")["clips"][0]["size_mb"] == 1.5


def test_get_clips_for_date_time_display_uses_12_hour_with_padding(server):
    clips = [
        clip("2024-04-10 00:07:00"),
        clip("2024-04-10 12:00:00"),
        clip("2024-04-10 09:30:00"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10")
    displays = [c["time_display"] for c in result["clips"]]
    # QUIRK: %I zero-pads, so midnight renders "12:07 AM" and 9am "09:30 AM".
    assert displays == ["12:07 AM", "09:30 AM", "12:00 PM"]


# --------------------------------------------------------------------------
# _get_clips_for_date — ordering
# --------------------------------------------------------------------------

def test_get_clips_for_date_sorted_ascending_by_time_of_day(server):
    clips = [
        clip("2024-04-10 23:00:00", path="c"),
        clip("2024-04-10 01:00:00", path="a"),
        clip("2024-04-10 12:00:00", path="b"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10")
    assert [c["path"] for c in result["clips"]] == ["a", "b", "c"]


def test_get_clips_for_date_sort_is_stable_for_identical_times(server):
    clips = [
        clip("2024-04-10 08:00:00", path="second", camera="cam2"),
        clip("2024-04-10 08:00:00", path="first", camera="cam1"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10")
    # Stable sort keeps input order for equal '%H:%M:%S' keys.
    assert [c["path"] for c in result["clips"]] == ["second", "first"]


# --------------------------------------------------------------------------
# _get_clips_for_date — date boundaries
# --------------------------------------------------------------------------

def test_get_clips_for_date_boundary_seconds_included_and_excluded(server):
    clips = [
        clip("2024-04-09 23:59:59", path="prev"),
        clip("2024-04-10 00:00:00", path="start"),
        clip("2024-04-10 23:59:59", path="end"),
        clip("2024-04-11 00:00:00", path="next"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10")
    assert [c["path"] for c in result["clips"]] == ["start", "end"]


def test_get_clips_for_date_requires_zero_padded_date(server):
    # QUIRK: matching is a raw string compare against '%Y-%m-%d', so "2024-4-10"
    # never matches — asserted as-is to detect rewrite drift.
    clips = [clip("2024-04-10 09:00:00")]
    assert server._get_clips_for_date(clips, "2024-4-10")["clips"] == []
    assert len(server._get_clips_for_date(clips, "2024-04-10")["clips"]) == 1


def test_get_clips_for_date_spans_year_boundary_correctly(server):
    clips = [
        clip("2023-12-31 23:00:00", path="nye"),
        clip("2024-01-01 00:30:00", path="nyd"),
    ]
    assert [c["path"] for c in server._get_clips_for_date(clips, "2023-12-31")["clips"]] == ["nye"]
    assert [c["path"] for c in server._get_clips_for_date(clips, "2024-01-01")["clips"]] == ["nyd"]


# --------------------------------------------------------------------------
# _get_clips_for_date — camera filter
# --------------------------------------------------------------------------

def test_get_clips_for_date_camera_filter_is_exact_match(server):
    clips = [
        clip("2024-04-10 09:00:00", camera="cam1"),
        clip("2024-04-10 10:00:00", camera="cam2"),
        clip("2024-04-10 11:00:00", camera="cam10"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10", camera="cam1")
    assert result["summary"]["total"] == 1
    assert result["clips"][0]["camera"] == "cam1"
    assert result["summary"]["by_camera"] == {"cam1": 1}


def test_get_clips_for_date_camera_filter_is_case_sensitive(server):
    clips = [clip("2024-04-10 09:00:00", camera="cam1")]
    # QUIRK: camera filter is case-sensitive while the species filter is not.
    assert server._get_clips_for_date(clips, "2024-04-10", camera="CAM1")["clips"] == []


def test_get_clips_for_date_empty_string_camera_disables_the_filter(server):
    # QUIRK: falsy filter values are treated as "no filter", so camera=''
    # returns everything rather than nothing.
    clips = [clip("2024-04-10 09:00:00", camera="cam1")]
    assert len(server._get_clips_for_date(clips, "2024-04-10", camera="")["clips"]) == 1
    assert len(server._get_clips_for_date(clips, "2024-04-10", camera=None)["clips"]) == 1


def test_get_clips_for_date_unknown_camera_returns_empty_result(server):
    clips = [clip("2024-04-10 09:00:00", camera="cam1")]
    result = server._get_clips_for_date(clips, "2024-04-10", camera="nope")
    assert result["clips"] == []
    assert result["summary"] == {
        "total": 0,
        "by_species": {},
        "by_camera": {},
        "by_hour": {},
        "peak_hour": None,
    }


# --------------------------------------------------------------------------
# _get_clips_for_date — species filter
# --------------------------------------------------------------------------

def test_get_clips_for_date_species_filter_is_case_insensitive_substring(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Whitetail Deer", path="deer"),
        clip("2024-04-10 10:00:00", species="Red Fox", path="fox"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10", species="dEeR")
    assert [c["path"] for c in result["clips"]] == ["deer"]


def test_get_clips_for_date_species_filter_matches_substrings_of_other_species(server):
    # QUIRK: the species filter is an unanchored substring test, so filtering by
    # "Deer" also returns "Reindeer" and "Deer Mouse" — asserted as-is.
    clips = [
        clip("2024-04-10 09:00:00", species="Reindeer", path="reindeer"),
        clip("2024-04-10 10:00:00", species="Deer Mouse", path="mouse"),
        clip("2024-04-10 11:00:00", species="Whitetail Deer", path="deer"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10", species="Deer")
    assert [c["path"] for c in result["clips"]] == ["reindeer", "mouse", "deer"]


def test_get_clips_for_date_species_filter_matches_multi_species_clips(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Cardinal, Blue Jay", path="both"),
        clip("2024-04-10 10:00:00", species="Squirrel", path="sq"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10", species="Blue Jay")
    assert [c["path"] for c in result["clips"]] == ["both"]


def test_get_clips_for_date_species_filter_unknown_matches_unknown_clips(server):
    # QUIRK: unlike the calendar, 'Unknown' is a perfectly usable species filter
    # here and matches the default species value.
    clips = [
        clip("2024-04-10 09:00:00", species="Unknown", path="u"),
        clip("2024-04-10 10:00:00", species="Deer", path="d"),
    ]
    result = server._get_clips_for_date(clips, "2024-04-10", species="unknown")
    assert [c["path"] for c in result["clips"]] == ["u"]


def test_get_clips_for_date_species_filter_manual_clip(server):
    manual = clip("2024-04-10 09:00:00", species="Manual clip", path="m")
    manual["date"] = "Manual"
    clips = [manual, clip("2024-04-10 10:00:00", species="Deer", path="d")]
    result = server._get_clips_for_date(clips, "2024-04-10", species="manual")
    assert [c["path"] for c in result["clips"]] == ["m"]
    assert result["summary"]["by_species"] == {"Manual clip": 1}


def test_get_clips_for_date_empty_string_species_disables_the_filter(server):
    clips = [clip("2024-04-10 09:00:00", species="Deer")]
    assert len(server._get_clips_for_date(clips, "2024-04-10", species="")["clips"]) == 1


def test_get_clips_for_date_camera_and_species_filters_combine(server):
    clips = [
        clip("2024-04-10 09:00:00", camera="cam1", species="Deer", path="a"),
        clip("2024-04-10 10:00:00", camera="cam2", species="Deer", path="b"),
        clip("2024-04-10 11:00:00", camera="cam1", species="Fox", path="c"),
    ]
    result = server._get_clips_for_date(
        clips, "2024-04-10", camera="cam1", species="Deer"
    )
    assert [c["path"] for c in result["clips"]] == ["a"]


def test_get_clips_for_date_species_filter_with_apostrophe(server):
    clips = [clip("2024-04-10 09:00:00", species="Cooper's Hawk", path="hawk")]
    result = server._get_clips_for_date(clips, "2024-04-10", species="cooper's")
    assert [c["path"] for c in result["clips"]] == ["hawk"]


def test_get_clips_for_date_species_filter_with_unicode(server):
    clips = [clip("2024-04-10 09:00:00", species="Canyón Wren", path="wren")]
    result = server._get_clips_for_date(clips, "2024-04-10", species="canyón")
    assert [c["path"] for c in result["clips"]] == ["wren"]
    # A de-accented query does not match.
    assert server._get_clips_for_date(clips, "2024-04-10", species="canyon")["clips"] == []


# --------------------------------------------------------------------------
# _get_clips_for_date — summary aggregation
# --------------------------------------------------------------------------

def test_get_clips_for_date_by_species_fans_out_on_comma_space(server):
    clips = [
        clip("2024-04-10 09:00:00", species="Cardinal, Blue Jay"),
        clip("2024-04-10 10:00:00", species="Cardinal"),
    ]
    summary = server._get_clips_for_date(clips, "2024-04-10")["summary"]
    assert summary["total"] == 2
    assert summary["by_species"] == {"Cardinal": 2, "Blue Jay": 1}
    # Species counts can exceed the clip total because one clip fans out.
    assert sum(summary["by_species"].values()) == 3


def test_get_clips_for_date_by_species_counts_unknown_unlike_the_calendar(server):
    # QUIRK: _build_calendar_data drops 'Unknown' from its species buckets, but
    # _get_clips_for_date's by_species happily counts it. Asserted as-is.
    clips = [clip("2024-04-10 09:00:00", species="Unknown")]
    summary = server._get_clips_for_date(clips, "2024-04-10")["summary"]
    assert summary["by_species"] == {"Unknown": 1}


def test_get_clips_for_date_by_species_skips_only_empty_fragments(server):
    clips = [clip("2024-04-10 09:00:00", species="")]
    summary = server._get_clips_for_date(clips, "2024-04-10")["summary"]
    assert summary["by_species"] == {}
    assert summary["total"] == 1


def test_get_clips_for_date_by_camera_and_by_hour_counts(server):
    clips = [
        clip("2024-04-10 09:10:00", camera="cam1"),
        clip("2024-04-10 09:50:00", camera="cam2"),
        clip("2024-04-10 17:00:00", camera="cam1"),
    ]
    summary = server._get_clips_for_date(clips, "2024-04-10")["summary"]
    assert summary["by_camera"] == {"cam1": 2, "cam2": 1}
    assert summary["by_hour"] == {9: 2, 17: 1}
    assert summary["peak_hour"] == 9


def test_get_clips_for_date_by_hour_keys_are_ints_not_strings(server):
    summary = server._get_clips_for_date(
        [clip("2024-04-10 09:00:00")], "2024-04-10"
    )["summary"]
    assert list(summary["by_hour"].keys()) == [9]
    assert isinstance(summary["peak_hour"], int)


def test_get_clips_for_date_peak_hour_tie_picks_first_hour_encountered(server):
    # QUIRK: on a tie, max() returns the first key in the defaultdict's insertion
    # order, i.e. the hour of the first matching clip in the INPUT list — not the
    # earliest hour of the day. Asserted as-is to detect rewrite drift.
    clips = [
        clip("2024-04-10 20:00:00"),
        clip("2024-04-10 06:00:00"),
    ]
    assert server._get_clips_for_date(clips, "2024-04-10")["summary"]["peak_hour"] == 20

    reordered = list(reversed(clips))
    assert server._get_clips_for_date(reordered, "2024-04-10")["summary"]["peak_hour"] == 6


def test_get_clips_for_date_summary_counts_only_filtered_clips(server):
    clips = [
        clip("2024-04-10 09:00:00", camera="cam1", species="Deer"),
        clip("2024-04-10 10:00:00", camera="cam2", species="Fox"),
        clip("2024-04-11 10:00:00", camera="cam1", species="Deer"),
    ]
    summary = server._get_clips_for_date(clips, "2024-04-10", camera="cam1")["summary"]
    assert summary == {
        "total": 1,
        "by_species": {"Deer": 1},
        "by_camera": {"cam1": 1},
        "by_hour": {9: 1},
        "peak_hour": 9,
    }


def test_get_clips_for_date_summary_containers_are_plain_dicts(server):
    summary = server._get_clips_for_date(
        [clip("2024-04-10 09:00:00", species="Deer")], "2024-04-10"
    )["summary"]
    for key in ("by_species", "by_camera", "by_hour"):
        assert type(summary[key]) is dict
        # A missing key raises rather than defaulting to 0.
        with pytest.raises(KeyError):
            summary[key]["nope"]


# --------------------------------------------------------------------------
# Cross-check: calendar day counts agree with the day endpoint
# --------------------------------------------------------------------------

def test_calendar_day_count_matches_get_clips_for_date_total(server):
    clips = [
        clip("2024-04-10 09:00:00", camera="cam1", species="Deer"),
        clip("2024-04-10 10:00:00", camera="cam2", species="Cardinal, Blue Jay"),
        clip("2024-04-10 11:00:00", camera="cam1", species="Unknown"),
        clip("2024-04-11 11:00:00", camera="cam1", species="Fox"),
    ]
    calendar = server._build_calendar_data(clips)
    day_meta = calendar["years"]["2024"]["months"]["4"]["days"]["10"]
    day_api = server._get_clips_for_date(clips, "2024-04-10")

    assert day_meta["count"] == day_api["summary"]["total"] == 3
    # QUIRK: the two surfaces disagree on species — the calendar hides 'Unknown'
    # while the day summary counts it. Asserted as-is to detect rewrite drift.
    assert day_meta["species"] == ["Blue Jay", "Cardinal", "Deer"]
    assert sorted(day_api["summary"]["by_species"]) == [
        "Blue Jay",
        "Cardinal",
        "Deer",
        "Unknown",
    ]


def test_calendar_and_day_endpoint_over_a_realistic_multi_day_archive(server):
    clips = [
        clip("2024-05-01 06:15:00", camera="cam1", species="Whitetail Deer"),
        clip("2024-05-01 06:20:00", camera="cam2", species="Whitetail Deer"),
        clip("2024-05-01 19:45:00", camera="cam1", species="Raccoon"),
        clip("2024-04-30 23:58:00", camera="cam1", species="Unknown"),
        clip("2023-11-02 12:00:00", camera="cam1", species="Cardinal, Squirrel"),
    ]
    data = server._build_calendar_data(clips)

    assert list(data["years"].keys()) == ["2024", "2023"]
    assert data["years"]["2024"]["total"] == 4
    assert list(data["years"]["2024"]["months"].keys()) == ["5", "4"]
    may1 = data["years"]["2024"]["months"]["5"]["days"]["1"]
    assert may1["count"] == 3
    assert may1["cameras"] == ["cam1", "cam2"]
    assert may1["species"] == ["Raccoon", "Whitetail Deer"]
    assert may1["first_clip_time"] == "06:15"
    assert may1["last_clip_time"] == "19:45"
    assert data["filters"] == {
        "cameras": ["cam1", "cam2"],
        "species": ["Cardinal", "Raccoon", "Squirrel", "Whitetail Deer"],
    }

    day = server._get_clips_for_date(clips, "2024-05-01")
    assert day["summary"]["by_hour"] == {6: 2, 19: 1}
    assert day["summary"]["peak_hour"] == 6
    assert [c["time"] for c in day["clips"]] == ["06:15:00", "06:20:00", "19:45:00"]
