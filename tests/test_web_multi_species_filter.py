"""A clip labelled with several species must survive its own filter chip.

The archive shows one clip's species as a single string and offers that same
string as a filter facet, so the string has to round-trip: ask for it back and
you get the clips it came from.  Joined with ", ", it did not.  The species
filter splits its query value on commas (``species=Deer,Bird`` is two species),
so the chip "Bird / Mammal" -- then "Bird, Mammal" -- split back into "Bird"
and "Mammal", selecting every single-species Bird clip and every Mammal clip
and excluding the multi-species clips the chip was built from.

Found in the bug hunt of 2026-09-19.  Fixed by joining on
``SPECIES_DISPLAY_SEPARATOR``.
"""
from __future__ import annotations

import itertools

import pytest

from animaltracker.web import SPECIES_DISPLAY_SEPARATOR, WebServer

_PORTS = itertools.count(19600)


@pytest.fixture
def server(tmp_path):
    return WebServer({}, tmp_path / "storage", tmp_path / "logs", port=next(_PORTS))


def parse(server, name):
    return server._parse_species_from_filename(name)[0]


# --------------------------------------------------------------------------
# the separator itself
# --------------------------------------------------------------------------


def test_the_separator_cannot_collide_with_the_filter_delimiter():
    """The one invariant: whatever joins species, it is not a comma."""
    assert "," not in SPECIES_DISPLAY_SEPARATOR


def test_a_multi_species_display_name_holds_no_comma(server):
    assert parse(server, "1788800356_bird+mammalia.mp4") == "Bird / Mammal"
    assert parse(server, "1788800356_bird+mammalia+reptilia.mp4") == "Bird / Mammal / Reptile"


# --------------------------------------------------------------------------
# the round-trip through the real filter
# --------------------------------------------------------------------------


def clip(species, camera="cam1", filename="x.mp4"):
    return {
        "path": f"{camera}/{filename}",
        "camera": camera,
        "date": "2026-09-19",
        "filename": filename,
        "species": species,
        "raw_species": "bird",
    }


def test_a_multi_species_chip_selects_exactly_the_clips_it_came_from(server):
    """The bug end to end: the name the parser gives a clip, fed back as the
    chip the facet list offers, must select that clip and only that clip."""
    both = clip(parse(server, "1788800356_bird+mammalia.mp4"), filename="both.mp4")
    bird = clip(parse(server, "1788800356_bird.mp4"), filename="bird.mp4")
    mammal = clip(parse(server, "1788800356_mammalia.mp4"), filename="mammal.mp4")
    clips = [both, bird, mammal]

    # The facet's value is the clip's species string verbatim (see
    # handle_recordings_api), so this is exactly what clicking the chip sends.
    got = server._filter_clips(clips, {"species": both["species"]})

    assert [c["filename"] for c in got] == ["both.mp4"]


def test_the_old_separator_would_have_selected_the_wrong_clips(server):
    """Why the fix matters, spelled out against the former behaviour."""
    clips = [
        clip("Bird, Mammal", filename="both.mp4"),
        clip("Bird", filename="bird.mp4"),
        clip("Mammal", filename="mammal.mp4"),
    ]

    got = server._filter_clips(clips, {"species": "Bird, Mammal"})

    # Every clip except the one the chip was made from -- and the trailing
    # space in " Mammal" is not trimmed, so "Mammal" alone misses too.
    assert [c["filename"] for c in got] == ["bird.mp4"]


def test_several_multi_species_chips_still_combine(server):
    """Commas keep their job: selecting more than one chip at a time."""
    clips = [
        clip("Bird / Mammal", filename="a.mp4"),
        clip("Deer / Squirrel", filename="b.mp4"),
        clip("Bird", filename="c.mp4"),
    ]

    got = server._filter_clips(clips, {"species": "Bird / Mammal,Deer / Squirrel"})

    assert [c["filename"] for c in got] == ["a.mp4", "b.mp4"]


def test_free_text_search_still_finds_one_species_inside_the_joined_name(server):
    """The joined name is one token to the filter but still searchable text."""
    clips = [clip("Bird / Mammal", filename="a.mp4"), clip("Reptile", filename="b.mp4")]

    assert [c["filename"] for c in server._filter_clips(clips, {"q": "mammal"})] == ["a.mp4"]
