"""Characterization tests for WebServer._get_clip_detail (web.py ~6952-7080).

These pin CURRENT behaviour of clip-detail assembly and thumbnail->track
matching so that a rewrite of web.py fails loudly if it drifts.  Anything
that looks like a bug is asserted AS-IS and flagged with a `# QUIRK:` note.

Area covered:
  * sidecar (`<clip>.log.json`) parsing: tracking_summary.tracks sorted by
    first_frame, frame -> second conversion using the sidecar's video fps
  * enrichment of thumbnails with start_time/end_time/duration/track_id/
    confidence
  * new-format matching by track_index (`_t0`, `_t1`) and the legacy
    species-name fallback
  * absent sidecar (previously raised UnboundLocalError), malformed sidecar
  * missing clip -> None
  * the shape of the returned dict (species vs raw_species, default fps 15.0)
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import web as web_mod
from animaltracker.web import WebServer


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

_PORTS = itertools.count(18400)


@pytest.fixture
def make_server(tmp_path):
    """Build a WebServer with no cameras rooted at a fresh tmp_path."""

    def _make(runtime=None, subdir: str = "srv"):
        root = tmp_path / subdir
        server = WebServer(
            {},
            root / "storage",
            root / "logs",
            port=next(_PORTS),
            config_path=root / "config" / "cameras.yml",
            runtime=runtime,
        )
        return server

    return _make


@pytest.fixture
def server(make_server):
    return make_server()


def clips_dir(server: WebServer) -> Path:
    return server.storage_root / "clips"


def write_clip(server: WebServer, rel_path: str, size: int = 1234) -> Path:
    """Create a fake .mp4 at clips/<rel_path>; returns the absolute path."""
    path = clips_dir(server) / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


def write_thumb(clip_path: Path, suffix: str) -> Path:
    """Create `<clip stem>_thumb_<suffix>.jpg` next to the clip."""
    thumb = clip_path.with_name(f"{clip_path.stem}_thumb_{suffix}.jpg")
    thumb.write_bytes(b"\xff\xd8\xff")
    return thumb


def write_sidecar(clip_path: Path, payload) -> Path:
    """Write the `<clip>.log.json` sidecar. `payload` may be raw text."""
    log_path = clip_path.with_suffix(".log.json")
    if isinstance(payload, str):
        log_path.write_text(payload)
    else:
        log_path.write_text(json.dumps(payload))
    return log_path


def track(track_id, species, first_frame, last_frame, confidence=0.9):
    return {
        "track_id": track_id,
        "best_species": species,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "best_confidence": confidence,
    }


def sidecar_payload(tracks, fps=None):
    payload = {"tracking_summary": {"tracks": tracks}}
    if fps is not None:
        payload["video"] = {"fps": fps}
    return payload


def by_path(detail: dict) -> dict:
    """Index the returned thumbnails by their basename for easy assertions."""
    return {Path(t["path"]).name: t for t in detail["thumbnails"]}


CARDINAL_CLIP = "cam1/1766587074_bird_passeriformes_cardinalidae.mp4"


# --------------------------------------------------------------------------
# missing / non-file clips
# --------------------------------------------------------------------------


def test_missing_clip_returns_none(server):
    assert server._get_clip_detail("cam1/does_not_exist.mp4") is None


def test_missing_camera_directory_returns_none(server):
    assert server._get_clip_detail("nosuchcam/1766587074_bird.mp4") is None


def test_directory_masquerading_as_clip_returns_none(server):
    (clips_dir(server) / "cam1" / "notaclip.mp4").mkdir(parents=True)
    assert server._get_clip_detail("cam1/notaclip.mp4") is None


def test_empty_rel_path_returns_none(server):
    # clips_dir / '' is the clips directory itself, which is not a file.
    assert server._get_clip_detail("") is None


# --------------------------------------------------------------------------
# returned dict shape
# --------------------------------------------------------------------------


def test_returned_keys_are_exactly_this_set(server):
    write_clip(server, CARDINAL_CLIP)
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert set(detail) == {
        "path",
        "filename",
        "camera",
        "species",
        "raw_species",
        "time",
        "size",
        "size_mb",
        "thumbnails",
        "fps",
        "global_settings",
    }


def test_species_is_display_string_and_raw_species_is_taxonomy(server):
    write_clip(server, CARDINAL_CLIP)
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["species"] == "Cardinal"
    assert detail["raw_species"] == "bird_passeriformes_cardinalidae"


def test_path_filename_camera_and_size_fields(server):
    clip = write_clip(server, CARDINAL_CLIP, size=2048)
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["path"] == CARDINAL_CLIP
    assert detail["filename"] == "1766587074_bird_passeriformes_cardinalidae.mp4"
    assert detail["camera"] == "cam1"
    assert detail["size"] == 2048
    assert detail["size_mb"] == 2048 / (1024 * 1024)
    assert detail["time"] == datetime.fromtimestamp(
        clip.stat().st_mtime, tz=web_mod.CENTRAL_TZ
    )


def test_camera_is_unknown_for_clip_at_clips_root(server):
    write_clip(server, "1766587074_bird.mp4")
    detail = server._get_clip_detail("1766587074_bird.mp4")
    # QUIRK: camera is derived purely from the first path segment, so a clip
    # sitting directly in clips/ reports the literal string 'unknown'
    # (never None) — asserted as-is to detect rewrite drift.
    assert detail["camera"] == "unknown"


def test_camera_is_first_segment_of_a_nested_path(server):
    rel = "cam2/2026/09/1766587074_deer.mp4"
    write_clip(server, rel)
    detail = server._get_clip_detail(rel)
    # QUIRK: only parts[0] is used, so date-sharded subdirectories are
    # silently ignored — asserted as-is to detect rewrite drift.
    assert detail["camera"] == "cam2"


def test_unparseable_filename_yields_unknown_species(server):
    write_clip(server, "cam1/noundersscore.mp4")
    detail = server._get_clip_detail("cam1/noundersscore.mp4")
    assert detail["species"] == "Unknown"
    assert detail["raw_species"] == "unknown"


def test_species_with_apostrophe_survives_round_trip(server):
    rel = "cam1/1766587074_o'possum.mp4"
    write_clip(server, rel)
    detail = server._get_clip_detail(rel)
    assert detail["species"] == "O'Possum"
    assert detail["raw_species"] == "o'possum"


def test_unicode_species_survives_round_trip(server):
    rel = "cam1/1766587074_ünïcørn.mp4"
    write_clip(server, rel)
    detail = server._get_clip_detail(rel)
    assert detail["species"] == "Ünïcørn"
    assert detail["raw_species"] == "ünïcørn"


# --------------------------------------------------------------------------
# fps defaulting
# --------------------------------------------------------------------------


def test_default_fps_is_15_without_sidecar(server):
    write_clip(server, CARDINAL_CLIP)
    assert server._get_clip_detail(CARDINAL_CLIP)["fps"] == 15.0


def test_sidecar_fps_overrides_the_default(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_sidecar(clip, sidecar_payload([], fps=30.0))
    assert server._get_clip_detail(CARDINAL_CLIP)["fps"] == 30.0


def test_zero_fps_in_sidecar_falls_back_to_15(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=0))
    detail = server._get_clip_detail(CARDINAL_CLIP)
    # QUIRK: fps is read with a truthiness test, so a genuine 0 (or 0.0)
    # silently becomes 15.0 instead of being reported/handled as invalid —
    # asserted as-is to detect rewrite drift.
    assert detail["fps"] == 15.0


def test_missing_video_block_falls_back_to_15(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)]))
    assert server._get_clip_detail(CARDINAL_CLIP)["fps"] == 15.0


def test_non_numeric_fps_leaks_into_the_result_and_kills_enrichment(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, sidecar_payload([track(7, "cardinalidae", 0, 30)], fps="abc"))
    detail = server._get_clip_detail(CARDINAL_CLIP)
    # QUIRK: video_fps is assigned before it is validated, so a string fps is
    # returned verbatim as 'fps' while the frame->second division raises and
    # is swallowed, leaving every thumbnail unenriched — asserted as-is to
    # detect rewrite drift.
    assert detail["fps"] == "abc"
    assert "start_time" not in detail["thumbnails"][0]


# --------------------------------------------------------------------------
# no sidecar / malformed sidecar
# --------------------------------------------------------------------------


def test_no_sidecar_returns_cleanly_with_unenriched_thumbnails(server):
    """Regression: this used to raise UnboundLocalError."""
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_thumb(clip, "corvidae_t1")

    detail = server._get_clip_detail(CARDINAL_CLIP)

    assert detail is not None
    assert len(detail["thumbnails"]) == 2
    for thumb in detail["thumbnails"]:
        assert set(thumb) == {"path", "species", "url", "track_index"}
        assert "start_time" not in thumb
        assert "track_id" not in thumb


def test_malformed_sidecar_json_is_swallowed(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, "{not: valid json,,,")

    detail = server._get_clip_detail(CARDINAL_CLIP)

    assert detail["fps"] == 15.0
    assert "start_time" not in detail["thumbnails"][0]


def test_empty_sidecar_file_is_swallowed(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, "")
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["fps"] == 15.0
    assert "start_time" not in detail["thumbnails"][0]


def test_sidecar_holding_a_json_list_is_swallowed(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, [1, 2, 3])
    detail = server._get_clip_detail(CARDINAL_CLIP)
    # QUIRK: a structurally valid but wrongly-shaped sidecar raises inside the
    # broad `except Exception` and is only logged at warning level; the caller
    # cannot tell it apart from "no sidecar" — asserted as-is.
    assert detail["fps"] == 15.0
    assert "start_time" not in detail["thumbnails"][0]


def test_sidecar_with_empty_tracks_list_leaves_thumbnails_unenriched(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, sidecar_payload([], fps=24.0))
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["fps"] == 24.0
    assert "start_time" not in detail["thumbnails"][0]


def test_sidecar_without_tracking_summary_leaves_thumbnails_unenriched(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, {"video": {"fps": 20.0}})
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["fps"] == 20.0
    assert "start_time" not in detail["thumbnails"][0]


def test_sidecar_lookup_replaces_the_clip_extension(server):
    """The sidecar is `<stem>.log.json`, NOT `<filename>.log.json`."""
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    # Wrong name: keeps the .mp4 in place.
    clip.with_name(clip.name + ".log.json").write_text(
        json.dumps(sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=30.0))
    )
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["fps"] == 15.0
    assert "start_time" not in detail["thumbnails"][0]

    # Right name: extension swapped.
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=30.0))
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["fps"] == 30.0
    assert detail["thumbnails"][0]["start_time"] == 0.0


# --------------------------------------------------------------------------
# new-format matching by track_index
# --------------------------------------------------------------------------


def test_track_index_matching_enriches_each_thumbnail(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_thumb(clip, "corvidae_t1")
    write_sidecar(
        clip,
        sidecar_payload(
            [
                track(11, "cardinalidae", 0, 30, confidence=0.81),
                track(22, "corvidae", 60, 150, confidence=0.42),
            ],
            fps=30.0,
        ),
    )

    thumbs = by_path(server._get_clip_detail(CARDINAL_CLIP))

    t0 = thumbs["1766587074_bird_passeriformes_cardinalidae_thumb_cardinalidae_t0.jpg"]
    assert t0["track_index"] == 0
    assert t0["track_id"] == 11
    assert t0["species"] == "Cardinal"
    assert t0["start_time"] == 0.0
    assert t0["end_time"] == 1.0
    assert t0["duration"] == 1.0
    assert t0["confidence"] == 0.81

    t1 = thumbs["1766587074_bird_passeriformes_cardinalidae_thumb_corvidae_t1.jpg"]
    assert t1["track_index"] == 1
    assert t1["track_id"] == 22
    assert t1["start_time"] == 2.0
    assert t1["end_time"] == 5.0
    assert t1["duration"] == 3.0
    assert t1["confidence"] == 0.42


def test_tracks_are_sorted_by_first_frame_not_by_sidecar_order(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "corvidae_t0")
    write_thumb(clip, "cardinalidae_t1")
    write_sidecar(
        clip,
        sidecar_payload(
            [
                # Deliberately out of order: the late track is listed first.
                track(99, "cardinalidae", 300, 330),
                track(11, "corvidae", 15, 45),
            ],
            fps=15.0,
        ),
    )

    thumbs = by_path(server._get_clip_detail(CARDINAL_CLIP))
    stem = "1766587074_bird_passeriformes_cardinalidae"

    # index 0 goes to the EARLIEST first_frame, regardless of sidecar order.
    assert thumbs[f"{stem}_thumb_corvidae_t0.jpg"]["track_id"] == 11
    assert thumbs[f"{stem}_thumb_corvidae_t0.jpg"]["start_time"] == 1.0
    assert thumbs[f"{stem}_thumb_cardinalidae_t1.jpg"]["track_id"] == 99
    assert thumbs[f"{stem}_thumb_cardinalidae_t1.jpg"]["start_time"] == 20.0


def test_track_index_wins_over_species_even_when_species_disagrees(server):
    """The thumbnail's own species label is ignored once _tN matches."""
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "corvidae_t0")  # says "Crow/Jay"
    write_sidecar(
        clip, sidecar_payload([track(5, "cardinalidae", 90, 120)], fps=30.0)
    )

    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    # QUIRK: enrichment is positional, so a thumbnail labelled Crow/Jay is
    # given the timing and track_id of a Cardinal track with no consistency
    # check, and its displayed species stays 'Crow/Jay' — asserted as-is.
    assert thumb["species"] == "Crow/Jay"
    assert thumb["track_id"] == 5
    assert thumb["start_time"] == 3.0


def test_missing_first_and_last_frame_default_to_zero(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(
        clip,
        {
            "video": {"fps": 30.0},
            "tracking_summary": {"tracks": [{"best_species": "cardinalidae"}]},
        },
    )

    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert thumb["start_time"] == 0.0
    assert thumb["end_time"] == 0.0
    assert thumb["duration"] == 0.0
    # QUIRK: a track with no track_id/best_confidence enriches the thumbnail
    # with track_id None and confidence 0 rather than being skipped —
    # asserted as-is to detect rewrite drift.
    assert thumb["track_id"] is None
    assert thumb["confidence"] == 0


def test_single_frame_track_has_zero_duration(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 45, 45)], fps=15.0))
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert thumb["start_time"] == 3.0
    assert thumb["end_time"] == 3.0
    assert thumb["duration"] == 0.0


def test_last_frame_before_first_frame_yields_negative_duration(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 60, 30)], fps=30.0))
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    # QUIRK: durations are never clamped, so an inverted frame range produces
    # a negative duration — asserted as-is to detect rewrite drift.
    assert thumb["duration"] == -1.0


def test_track_index_beyond_the_track_list_falls_back_to_species(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t7")
    write_sidecar(clip, sidecar_payload([track(3, "cardinalidae", 30, 60)], fps=30.0))

    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    # An out-of-range index silently drops through to the species fallback,
    # which happens to match — so the thumbnail IS enriched.
    assert thumb["track_index"] == 7
    assert thumb["track_id"] == 3
    assert thumb["start_time"] == 1.0


def test_track_index_beyond_list_with_no_species_match_stays_unenriched(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "corvidae_t7")
    write_sidecar(clip, sidecar_payload([track(3, "cardinalidae", 30, 60)], fps=30.0))
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert "start_time" not in thumb
    assert "track_id" not in thumb


def test_track_with_empty_species_is_skipped_but_still_consumes_an_index(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_thumb(clip, "corvidae_t1")
    write_sidecar(
        clip,
        sidecar_payload(
            [
                track(1, "", 0, 30),  # no best_species -> not indexed
                track(2, "corvidae", 60, 90),
            ],
            fps=30.0,
        ),
    )

    thumbs = by_path(server._get_clip_detail(CARDINAL_CLIP))
    stem = "1766587074_bird_passeriformes_cardinalidae"

    # QUIRK: tracks with a falsy best_species are skipped, yet enumerate()
    # still advances the index — so index 0 is absent while index 1 exists.
    # Asserted as-is to detect rewrite drift.
    t0 = thumbs[f"{stem}_thumb_cardinalidae_t0.jpg"]
    assert "start_time" not in t0
    t1 = thumbs[f"{stem}_thumb_corvidae_t1.jpg"]
    assert t1["track_id"] == 2
    assert t1["start_time"] == 2.0


# --------------------------------------------------------------------------
# legacy species-name fallback
# --------------------------------------------------------------------------


def test_legacy_thumbnail_matches_by_species_name(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae")  # no _tN suffix
    write_sidecar(
        clip,
        sidecar_payload(
            [track(42, "bird_passeriformes_cardinalidae", 45, 105, 0.77)], fps=15.0
        ),
    )

    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert "track_index" not in thumb
    assert thumb["species"] == "Cardinal"
    assert thumb["track_id"] == 42
    assert thumb["start_time"] == 3.0
    assert thumb["end_time"] == 7.0
    assert thumb["duration"] == 4.0
    assert thumb["confidence"] == 0.77


def test_legacy_numeric_suffix_is_stripped_before_species_lookup(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_2")
    write_sidecar(clip, sidecar_payload([track(9, "cardinalidae", 15, 30)], fps=15.0))

    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    # The trailing `_2` is a legacy per-thumbnail counter, not a track index.
    assert "track_index" not in thumb
    assert thumb["species"] == "Cardinal"
    assert thumb["track_id"] == 9


def test_species_fallback_keeps_only_the_first_track_per_species(server):
    """The headline quirk of the legacy path."""
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae")
    write_thumb(clip, "cardinalidae_1")
    write_sidecar(
        clip,
        sidecar_payload(
            [
                track(100, "cardinalidae", 0, 30, confidence=0.30),
                track(200, "cardinalidae", 300, 450, confidence=0.99),
            ],
            fps=30.0,
        ),
    )

    thumbs = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"]
    assert len(thumbs) == 2

    # QUIRK: tracks_by_species keeps only the FIRST track for a given common
    # name, so BOTH legacy thumbnails are stamped with track 100's timing and
    # the second, longer/more-confident track (200) is unreachable. The source
    # comment claims the kept track is "highest confidence usually" but it is
    # really just the earliest first_frame after sorting — here the kept track
    # has the LOWER confidence. Asserted as-is to detect rewrite drift.
    for thumb in thumbs:
        assert thumb["track_id"] == 100
        assert thumb["start_time"] == 0.0
        assert thumb["end_time"] == 1.0
        assert thumb["confidence"] == 0.30


def test_species_fallback_is_keyed_on_common_name_not_raw_taxonomy(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae")
    write_sidecar(
        clip,
        sidecar_payload(
            [track(4, "bird_passeriformes_cardinalidae_cardinalis", 30, 60)], fps=30.0
        ),
    )
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    # Different raw strings collapse to the same common name and still match.
    assert thumb["species"] == "Cardinal"
    assert thumb["track_id"] == 4


def test_legacy_thumbnail_with_unmatched_species_stays_unenriched(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "sciuridae")
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=30.0))
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert thumb["species"] == "Squirrel"
    assert "start_time" not in thumb


def test_empty_species_suffix_thumbnail_is_labelled_unknown_and_unenriched(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "")  # ..._thumb_.jpg
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=30.0))
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    assert thumb["species"] == "Unknown"
    assert "start_time" not in thumb


def test_mixed_new_and_legacy_thumbnails_in_one_clip(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    write_thumb(clip, "sciuridae")  # legacy, matches by species
    write_sidecar(
        clip,
        sidecar_payload(
            [
                track(10, "cardinalidae", 0, 15),
                track(20, "sciuridae", 150, 300),
            ],
            fps=15.0,
        ),
    )

    thumbs = by_path(server._get_clip_detail(CARDINAL_CLIP))
    stem = "1766587074_bird_passeriformes_cardinalidae"

    assert thumbs[f"{stem}_thumb_cardinalidae_t0.jpg"]["track_id"] == 10
    legacy = thumbs[f"{stem}_thumb_sciuridae.jpg"]
    assert legacy["track_id"] == 20
    assert legacy["start_time"] == 10.0
    assert legacy["end_time"] == 20.0

    # Indexed thumbnails sort before legacy ones (sentinel index 999).
    order = [Path(t["path"]).name for t in server._get_clip_detail(CARDINAL_CLIP)["thumbnails"]]
    assert order == [
        f"{stem}_thumb_cardinalidae_t0.jpg",
        f"{stem}_thumb_sciuridae.jpg",
    ]


def test_thumbnails_of_other_clips_in_the_same_directory_are_not_picked_up(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    other = write_clip(server, "cam1/1766599999_deer.mp4")
    write_thumb(other, "cervidae_t0")

    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert len(detail["thumbnails"]) == 1
    assert "cardinalidae_t0" in detail["thumbnails"][0]["path"]


def test_thumbnail_path_and_url_are_relative_to_the_clips_dir(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_thumb(clip, "cardinalidae_t0")
    thumb = server._get_clip_detail(CARDINAL_CLIP)["thumbnails"][0]
    stem = "1766587074_bird_passeriformes_cardinalidae"
    assert thumb["path"] == f"cam1/{stem}_thumb_cardinalidae_t0.jpg"
    assert thumb["url"] == f"/clips/cam1/{stem}_thumb_cardinalidae_t0.jpg"


def test_clip_with_no_thumbnails_returns_an_empty_list(server):
    clip = write_clip(server, CARDINAL_CLIP)
    write_sidecar(clip, sidecar_payload([track(1, "cardinalidae", 0, 30)], fps=30.0))
    assert server._get_clip_detail(CARDINAL_CLIP)["thumbnails"] == []


# --------------------------------------------------------------------------
# global_settings
# --------------------------------------------------------------------------


EXPECTED_DEFAULT_SETTINGS = {
    "sample_rate": 3,
    "confidence_threshold": 0.3,
    "generic_confidence": 0.5,
    "tracking_enabled": True,
    "track_merge_gap": 120,
    "spatial_merge_enabled": True,
    "spatial_merge_iou": 0.3,
    "hierarchical_merge_enabled": True,
    "single_animal_mode": False,
    "thumbnail_cropped": True,
}


def test_global_settings_defaults_without_a_runtime(server):
    write_clip(server, CARDINAL_CLIP)
    detail = server._get_clip_detail(CARDINAL_CLIP)
    assert detail["global_settings"] == EXPECTED_DEFAULT_SETTINGS


def test_global_settings_read_from_runtime_clip_config(make_server):
    runtime = SimpleNamespace(
        general=SimpleNamespace(
            timezone=None,
            clip=SimpleNamespace(
                sample_rate=5,
                post_analysis_confidence=0.55,
                post_analysis_generic_confidence=0.65,
                tracking_enabled=False,
                track_merge_gap=90,
                spatial_merge_enabled=False,
                spatial_merge_iou=0.75,
                hierarchical_merge_enabled=False,
                single_animal_mode=True,
                thumbnail_cropped=False,
            ),
        )
    )
    server = make_server(runtime=runtime, subdir="with-runtime")
    write_clip(server, CARDINAL_CLIP)

    assert server._get_clip_detail(CARDINAL_CLIP)["global_settings"] == {
        "sample_rate": 5,
        "confidence_threshold": 0.55,
        "generic_confidence": 0.65,
        "tracking_enabled": False,
        "track_merge_gap": 90,
        "spatial_merge_enabled": False,
        "spatial_merge_iou": 0.75,
        "hierarchical_merge_enabled": False,
        "single_animal_mode": True,
        "thumbnail_cropped": False,
    }


def test_global_settings_fill_in_defaults_for_missing_clip_attributes(make_server):
    runtime = SimpleNamespace(
        general=SimpleNamespace(
            timezone=None, clip=SimpleNamespace(sample_rate=9)
        )
    )
    server = make_server(runtime=runtime, subdir="partial-runtime")
    write_clip(server, CARDINAL_CLIP)

    settings = server._get_clip_detail(CARDINAL_CLIP)["global_settings"]
    assert settings["sample_rate"] == 9
    assert {k: v for k, v in settings.items() if k != "sample_rate"} == {
        k: v for k, v in EXPECTED_DEFAULT_SETTINGS.items() if k != "sample_rate"
    }


def test_runtime_present_but_clip_config_none_uses_defaults(make_server):
    runtime = SimpleNamespace(general=SimpleNamespace(timezone=None, clip=None))
    server = make_server(runtime=runtime, subdir="null-clip-cfg")
    write_clip(server, CARDINAL_CLIP)
    assert server._get_clip_detail(CARDINAL_CLIP)["global_settings"] == (
        EXPECTED_DEFAULT_SETTINGS
    )
