"""Characterization tests for species name parsing, display names and icons.

These tests pin the CURRENT behaviour of:

  * ``WebServer._parse_species_from_filename``   (web.py ~1856)
  * ``WebServer._extract_species_from_filename`` (web.py ~9880)
  * ``get_common_name`` / ``get_species_icon``   (species_names.py)

They are a safety net for a rewrite of ``web.py``: several assertions below
capture behaviour that is arguably WRONG. Those are marked with ``# QUIRK:``
comments. Do not "fix" them here -- if a rewrite changes them, that is exactly
the signal these tests exist to produce.
"""

import pytest

from animaltracker.species_names import (
    PARTIAL_PATTERNS,
    SPECIES_MAP,
    get_common_name,
    get_species_icon,
)
from animaltracker.web import WebServer


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A WebServer with no cameras -- enough for pure filename parsing."""
    root = tmp_path_factory.mktemp("web_species_naming")
    return WebServer(
        workers={},
        storage_root=root / "storage",
        logs_root=root / "logs",
        port=18901,
    )


@pytest.fixture
def parse(server):
    return server._parse_species_from_filename


@pytest.fixture
def extract(server):
    return server._extract_species_from_filename


UUID_A = "a1b2c3d4-1234-5678-9abc-def012345678"
UUID_UPPER = "A1B2C3D4-1234-5678-9ABC-DEF012345678"


# ==========================================================================
# _parse_species_from_filename -- happy path, real archive filenames
# ==========================================================================


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "1788800356_mammalia_lagomorpha_leporidae_sylvilagus_floridanus.mp4",
            ("Eastern Cottontail", "mammalia_lagomorpha_leporidae_sylvilagus_floridanus"),
        ),
        ("1788792747_bird.mp4", ("Bird", "bird")),
        (
            "1766587074_bird_passeriformes_cardinalidae.mp4",
            ("Cardinal", "bird_passeriformes_cardinalidae"),
        ),
        (
            "1766589347_mammalia_carnivora_canidae.mp4",
            ("Dog/Canid", "mammalia_carnivora_canidae"),
        ),
        ("1778509786_reptilia_reptile.mp4", ("Reptile", "reptilia_reptile")),
        (
            "1788800356_mammalia_carnivora_felidae_felis_catus.mp4",
            ("Domestic Cat", "mammalia_carnivora_felidae_felis_catus"),
        ),
    ],
)
def test_parse_real_archive_filenames(parse, filename, expected):
    assert parse(filename) == expected


def test_parse_returns_display_and_raw_tuple(parse):
    """The contract is a 2-tuple: (display name, raw species for icon lookup)."""
    result = parse("1766587074_bird_passeriformes_cardinalidae.mp4")
    assert isinstance(result, tuple)
    assert len(result) == 2
    display, raw = result
    assert display == "Cardinal"
    # raw is fed straight back into the icon lookup
    assert get_species_icon(raw) == "\U0001f426"


def test_parse_keeps_apostrophes_in_common_names(parse):
    display, raw = parse(
        "1788800356_bird_accipitriformes_accipitridae_accipiter_cooperii.mp4"
    )
    assert display == "Cooper's Hawk"
    assert raw == "bird_accipitriformes_accipitridae_accipiter_cooperii"


# ==========================================================================
# _parse_species_from_filename -- timestamp prefix stripping
# ==========================================================================


def test_parse_strips_only_up_to_the_first_underscore(parse):
    """Everything before the FIRST '_' is treated as the timestamp."""
    assert parse("1788800356_mammalia_carnivora.mp4")[1] == "mammalia_carnivora"


def test_parse_no_underscore_is_unknown(parse):
    assert parse("noextensionorunderscore") == ("Unknown", "unknown")
    assert parse("1788800356.mp4") == ("Unknown", "unknown")


def test_parse_empty_filename_is_unknown(parse):
    assert parse("") == ("Unknown", "unknown")


def test_parse_empty_species_part_is_unknown(parse):
    assert parse("1788800356_.mp4") == ("Unknown", "unknown")


def test_parse_empty_timestamp_prefix_still_parses(parse):
    # QUIRK: the prefix is never validated as a timestamp -- a leading '_' with
    # nothing before it is accepted -- asserted as-is to detect rewrite drift.
    assert parse("_bird.mp4") == ("Bird", "bird")


def test_parse_non_timestamp_prefix_eats_only_first_field(parse):
    # QUIRK: 'clip_cam2_<ts>.mp4' (the ptz-review naming) is parsed as species
    # 'cam2_1778327229' and displayed as the bare timestamp -- asserted as-is
    # to detect rewrite drift.
    assert parse("clip_cam2_1778327229.mp4") == ("1778327229", "cam2_1778327229")


def test_parse_strips_only_the_last_extension(parse):
    # QUIRK: rsplit('.', 1) leaves '.tar' glued to the species -- asserted
    # as-is to detect rewrite drift.
    assert parse("1788800356_gecko_lizard.tar.gz") == ("Lizard.Tar", "gecko_lizard.tar")


def test_parse_tolerates_missing_extension(parse):
    assert parse("1788800356_bird") == ("Bird", "bird")


def test_parse_extension_case_insensitive(parse):
    assert parse("1788792747_bird.MP4") == ("Bird", "bird")


# ==========================================================================
# _parse_species_from_filename -- UUID stripping
# ==========================================================================


def test_parse_strips_leading_uuid_and_trailing_semicolons(parse):
    assert parse(f"1788800356_{UUID_A};mammalia;carnivora;canidae.mp4") == (
        "Dog/Canid",
        "mammalia_carnivora_canidae",
    )


def test_parse_strips_uppercase_uuid(parse):
    assert parse(f"1788800356_{UUID_UPPER};bird.mp4") == ("Bird", "bird")


def test_parse_uuid_only_species_is_unknown(parse):
    assert parse(f"1788800356_{UUID_A}.mp4") == ("Unknown", "unknown")


def test_parse_strips_uuid_embedded_between_underscores(parse):
    # QUIRK: only the UUID text is removed, so the surrounding underscores
    # collapse into a double underscore in the raw name -- asserted as-is to
    # detect rewrite drift.
    assert parse(f"1788800356_bird_{UUID_A}_x.mp4") == ("Bird", "bird__x")


def test_parse_uuid_lookalike_is_not_stripped(parse):
    # QUIRK: a non-hex 'uuid-not-a-uuid' survives and, because ';' segments are
    # re-joined with '_', it wins the display name over the real species
    # -- asserted as-is to detect rewrite drift.
    assert parse("1788800356_uuid-not-a-uuid;bird.mp4") == (
        "Uuid",
        "uuid-not-a-uuid_bird",
    )


# ==========================================================================
# _parse_species_from_filename -- '+' splitting (multi-species)
# ==========================================================================


def test_parse_plus_splits_multiple_species(parse):
    assert parse("1788800356_bird+mammalia.mp4") == ("Bird, Mammal", "bird")


def test_parse_raw_is_the_first_species_only(parse):
    """The icon is driven by the FIRST raw species, not the whole display."""
    display, raw = parse(
        "1788800356_bird_accipitriformes_accipitridae_accipiter_cooperii+mammalia.mp4"
    )
    assert display == "Cooper's Hawk, Mammal"
    assert raw == "bird_accipitriformes_accipitridae_accipiter_cooperii"


def test_parse_empty_plus_segments_are_skipped(parse):
    assert parse("1788800356_bird++mammalia.mp4") == ("Bird, Mammal", "bird")


def test_parse_bare_plus_is_unknown(parse):
    assert parse("1788800356_+.mp4") == ("Unknown", "unknown")


def test_parse_related_ranks_are_both_displayed(parse):
    """Family and species level of the same animal are two distinct names."""
    assert parse(
        "1788800356_mammalia_carnivora_felidae"
        "+mammalia_carnivora_felidae_felis_catus.mp4"
    ) == ("Cat, Domestic Cat", "mammalia_carnivora_felidae")


# ==========================================================================
# _parse_species_from_filename -- ';' splitting
# ==========================================================================


def test_parse_semicolon_segments_are_joined_with_underscore(parse):
    # QUIRK: within one '+' part, ';' segments are joined with '_' and looked up
    # as a SINGLE name, so 'bird;mammalia' resolves via the 'bird' prefix and
    # 'mammalia' is silently lost -- asserted as-is to detect rewrite drift.
    assert parse("1788800356_bird;mammalia.mp4") == ("Bird", "bird_mammalia")


def test_parse_semicolon_blank_segments_are_dropped(parse):
    assert parse("1788800356_bird;;;mammalia.mp4") == ("Bird", "bird_mammalia")


def test_parse_semicolon_segments_are_stripped_of_whitespace(parse):
    assert parse("1788800356_  bird  ;mammalia.mp4") == ("Bird", "bird_mammalia")


def test_parse_uuid_between_semicolon_segments(parse):
    # QUIRK: with the UUID removed the remaining ranks are joined as
    # 'mammalia_canidae', which has no map entry, so the 'mammalia' prefix wins
    # and the animal is reported as a generic Mammal -- asserted as-is to
    # detect rewrite drift.
    assert parse(f"1788800356_mammalia;{UUID_UPPER};canidae.mp4") == (
        "Mammal",
        "mammalia_canidae",
    )


# ==========================================================================
# _parse_species_from_filename -- placeholder dropping
# ==========================================================================


@pytest.mark.parametrize(
    "species_part", ["no cv result", "unknown", "blank", "empty"]
)
def test_parse_drops_placeholder_tokens(parse, species_part):
    # QUIRK: 'blank' has a real SPECIES_MAP entry ('Empty Frame') but is dropped
    # here before lookup -- asserted as-is to detect rewrite drift.
    assert parse(f"1788792411_{species_part}.mp4") == ("Unknown", "unknown")


def test_parse_placeholder_dropping_is_case_insensitive(parse):
    assert parse("1788800356_NO CV RESULT;bird.mp4") == ("Bird", "bird")


@pytest.mark.parametrize(
    "filename",
    [
        "1788800356_no cv result;bird.mp4",
        "1788800356_bird;no cv result.mp4",
        "1788800356_blank+bird.mp4",
        "1788800356_bird+blank.mp4",
    ],
)
def test_parse_placeholders_dropped_around_a_real_species(parse, filename):
    assert parse(filename) == ("Bird", "bird")


def test_parse_all_placeholders_is_unknown(parse):
    assert parse("1788800356_no cv result+unknown+blank.mp4") == ("Unknown", "unknown")


def test_parse_underscored_no_cv_result_is_not_dropped(parse):
    # QUIRK: the placeholder list only contains the spaced 'no cv result', so
    # the underscored variant emitted elsewhere in the codebase survives and is
    # displayed as 'Result' -- asserted as-is to detect rewrite drift.
    assert parse("1788800356_no_cv_result.mp4") == ("Result", "no_cv_result")


def test_parse_manual_clip_is_not_dropped(parse):
    # QUIRK: 'manual clip' is filtered by the stats endpoint but not here, so it
    # displays as 'Clip' -- asserted as-is to detect rewrite drift.
    assert parse("1788800356_manual clip.mp4") == ("Clip", "manual clip")


# ==========================================================================
# _parse_species_from_filename -- dedupe and the max-3 join
# ==========================================================================


def test_parse_dedupes_identical_species(parse):
    assert parse("1788800356_bird+bird.mp4") == ("Bird", "bird")


def test_parse_dedupe_is_case_insensitive_but_raw_keeps_case(parse):
    # QUIRK: the display name is deduped case-insensitively while the raw
    # species preserves the original casing of the FIRST occurrence
    # -- asserted as-is to detect rewrite drift.
    assert parse("1788800356_Bird+bird.mp4") == ("Bird", "Bird")


def test_parse_dedupes_across_different_raw_names(parse):
    """Two different raw names that map to the same common name collapse."""
    assert parse("1788800356_bird+bird_x_y_z.mp4") == ("Bird", "bird")


def test_parse_joins_with_comma_space(parse):
    assert parse("1788800356_bird+bird_passeriformes.mp4") == (
        "Bird, Songbird",
        "bird",
    )


def test_parse_limits_display_to_three_species(parse):
    display, raw = parse(
        "1788800356_bird+mammalia+reptilia+mammalia_carnivora_felidae.mp4"
    )
    assert display == "Bird, Mammal, Reptile"
    assert display.count(",") == 2
    # QUIRK: the 4th species ('Cat') is silently dropped with no ellipsis or
    # '+N more' marker -- asserted as-is to detect rewrite drift.
    assert "Cat" not in display
    assert raw == "bird"


def test_parse_dedupe_happens_before_the_three_limit(parse):
    assert parse("1788800356_bird+mammalia+reptilia+bird.mp4") == (
        "Bird, Mammal, Reptile",
        "bird",
    )


# ==========================================================================
# _parse_species_from_filename -- unicode
# ==========================================================================


def test_parse_handles_unicode_species_text(parse):
    assert parse("1788800356_café_naïve.mp4") == (
        "Naïve",
        "café_naïve",
    )


def test_parse_handles_emoji_in_filename(parse):
    assert parse("1788800356_\U0001f98c_deer.mp4") == ("Deer", "\U0001f98c_deer")


# ==========================================================================
# _extract_species_from_filename
# ==========================================================================


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "1788800356_mammalia_lagomorpha_leporidae_sylvilagus_floridanus.mp4",
            ["mammalia lagomorpha leporidae sylvilagus floridanus"],
        ),
        ("1788792747_bird.mp4", ["bird"]),
        (
            "1766587074_bird_passeriformes_cardinalidae.mp4",
            ["bird passeriformes cardinalidae"],
        ),
    ],
)
def test_extract_returns_lowercased_space_separated_names(extract, filename, expected):
    assert extract(filename) == expected


def test_extract_returns_empty_list_without_underscore(extract):
    assert extract("noextensionorunderscore") == []
    assert extract("1788800356.mp4") == []
    assert extract("") == []


def test_extract_drops_placeholder_only_names(extract):
    assert extract("1788792411_blank.mp4") == []
    assert extract("1788800356_unknown.mp4") == []
    assert extract("1788800356_no cv result.mp4") == []


def test_extract_returns_all_plus_separated_species(extract):
    assert extract("1788800356_bird+mammalia.mp4") == ["bird", "mammalia"]


def test_extract_dedupes_repeats(extract):
    assert extract("1788800356_bird+mammalia+reptilia+bird.mp4") == [
        "bird",
        "mammalia",
        "reptilia",
    ]


def test_extract_has_no_three_item_limit(extract):
    # QUIRK: unlike _parse_species_from_filename this one is unbounded
    # -- asserted as-is to detect rewrite drift.
    assert extract(
        "1788800356_bird+mammalia+reptilia+mammalia_carnivora_felidae.mp4"
    ) == [
        "bird",
        "mammalia",
        "reptilia",
        "mammalia carnivora felidae",
    ]


def test_extract_takes_the_last_semicolon_segment(extract):
    # QUIRK: this walks the ';' segments in REVERSE and breaks after the first
    # kept one, so only the most specific segment survives -- the exact
    # opposite of _parse_species_from_filename, which joins them all
    # -- asserted as-is to detect rewrite drift.
    assert extract("1788800356_bird;mammalia.mp4") == ["mammalia"]


def test_extract_disagrees_with_parse_on_semicolon_names(extract, parse):
    filename = f"1788800356_mammalia;{UUID_UPPER};canidae.mp4"
    # QUIRK: the two filename parsers give different answers for the same clip
    # -- asserted as-is to detect rewrite drift.
    assert extract(filename) == ["canidae"]
    assert parse(filename) == ("Mammal", "mammalia_canidae")


def test_extract_breaks_after_first_kept_segment_only(extract):
    """The 'break' fires even for a segment that was skipped... it does not.

    Only the placeholder ``continue`` skips ahead; the first real segment from
    the right ends the loop for that '+' part.
    """
    assert extract("1788800356_bird;no cv result.mp4") == ["bird"]


def test_extract_strips_uuid(extract):
    assert extract(f"1788800356_{UUID_A};mammalia;carnivora;canidae.mp4") == ["canidae"]
    assert extract(f"1788800356_{UUID_A}.mp4") == []


def test_extract_lowercases(extract):
    assert extract("1788800356_BIRD.mp4") == ["bird"]


def test_extract_keeps_underscored_no_cv_result(extract):
    # QUIRK: '_' -> ' ' happens AFTER the placeholder check, so 'no_cv_result'
    # is emitted as the string 'no cv result' the filter was meant to drop
    # -- asserted as-is to detect rewrite drift.
    assert extract("1788800356_no_cv_result.mp4") == ["no cv result"]


# ==========================================================================
# get_common_name
# ==========================================================================


@pytest.mark.parametrize("empty", ["", None])
def test_get_common_name_empty_is_unknown(empty):
    assert get_common_name(empty) == "Unknown"


@pytest.mark.parametrize(
    "species, expected",
    [
        ("bird", "Bird"),
        ("mammalia", "Mammal"),
        ("reptilia", "Reptile"),
        ("animal", "Animal"),
        ("unknown", "Unknown"),
        ("blank", "Empty Frame"),
        ("bird_passeriformes_cardinalidae", "Cardinal"),
        (
            "bird_passeriformes_cardinalidae_cardinalis_cardinalis",
            "Northern Cardinal",
        ),
        ("mammalia_lagomorpha_leporidae_sylvilagus_floridanus", "Eastern Cottontail"),
        ("reptilia_squamata_colubridae", "Snake (Colubrid)"),
        ("mammalia_primates_hylobatidae", "Gibbon (likely misidentified)"),
    ],
)
def test_get_common_name_exact_matches(species, expected):
    assert get_common_name(species) == expected


@pytest.mark.parametrize(
    "species",
    [
        "BIRD_PASSERIFORMES_CARDINALIDAE",
        "bird-passeriformes-cardinalidae",
        "bird passeriformes cardinalidae",
        "Bird Passeriformes-CARDINALIDAE",
    ],
)
def test_get_common_name_normalizes_case_spaces_and_hyphens(species):
    assert get_common_name(species) == "Cardinal"


def test_get_common_name_prefix_fallback_trims_from_the_right():
    """Unknown trailing ranks fall back to the longest known prefix."""
    assert (
        get_common_name("bird_passeriformes_cardinalidae_cardinalis_cardinalis_extra")
        == "Northern Cardinal"
    )
    assert get_common_name("bird_x_y_z") == "Bird"
    assert get_common_name("mammalia_artiodactyla_cervidae_odocoileus_hemionus") == "Deer"


def test_get_common_name_prefix_beats_partial_pattern():
    # QUIRK: the prefix walk runs before PARTIAL_PATTERNS, so 'mammalia' wins
    # over the 'lagomorpha' pattern and a rabbit is displayed as a generic
    # 'Mammal' -- asserted as-is to detect rewrite drift.
    assert get_common_name("mammalia_lagomorpha") == "Mammal"
    assert get_common_name("mammalia_artiodactyla_something") == "Mammal"
    assert get_common_name("animal_bird") == "Animal"


def test_get_common_name_partial_pattern_when_no_prefix_matches():
    assert get_common_name("zzz_qqq_cardinalidae_x") == "Cardinal"
    assert get_common_name("xx_leporidae") == "Rabbit"
    assert get_common_name("some_unknownclass_carnivora") == "Carnivore"


def test_get_common_name_partial_pattern_matches_substrings_anywhere():
    # QUIRK: PARTIAL_PATTERNS are unanchored regexes, so any embedded substring
    # matches -- asserted as-is to detect rewrite drift.
    assert get_common_name("xfelidaex") == "Cat"


def test_get_common_name_falls_back_to_last_meaningful_part():
    assert get_common_name("totally_made_up_thing") == "Thing"
    assert get_common_name("aves_something") == "Something"
    assert get_common_name("x") == "X"


def test_get_common_name_last_part_fallback_skips_generic_words():
    """'bird'/'mammalia'/... are skipped when walking parts right-to-left."""
    # 'zzz_bird': no map prefix, no partial pattern, 'bird' is skipped as
    # generic, so the leading nonsense token is what gets displayed.
    assert get_common_name("zzz_bird") == "Zzz"
    assert get_common_name("zzz_mammalia") == "Zzz"


def test_get_common_name_last_resort_titlecases_whole_string():
    # QUIRK: an all-empty-parts name reaches the final fallback and returns
    # whitespace -- asserted as-is to detect rewrite drift.
    assert get_common_name("___") == "   "
    assert get_common_name("_") == " "


def test_get_common_name_unmapped_symbols_pass_through():
    assert get_common_name("%%%") == "%%%"


def test_get_common_name_empty_string_token_is_not_special_cased():
    # 'empty' is filtered by the callers, never by get_common_name itself.
    assert get_common_name("empty") == "Empty"
    assert "empty" not in SPECIES_MAP


def test_get_common_name_is_total_over_the_map():
    """Every key in SPECIES_MAP round-trips to its own value."""
    for key, value in SPECIES_MAP.items():
        assert get_common_name(key) == value


def test_partial_patterns_are_reachable_regexes():
    """Each partial pattern still resolves when no prefix match exists."""
    for pattern, name in PARTIAL_PATTERNS:
        probe = f"zzzclass_{pattern}"
        assert get_common_name(probe) == name


# ==========================================================================
# get_species_icon
# ==========================================================================


@pytest.mark.parametrize("empty", ["", None])
def test_get_species_icon_empty_is_question_mark(empty):
    assert get_species_icon(empty) == "❓"


@pytest.mark.parametrize(
    "species, icon",
    [
        ("bird", "\U0001f426"),
        ("bird_passeriformes_cardinalidae", "\U0001f426"),
        ("aves_something", "\U0001f426"),
        ("mammalia_carnivora_felidae_felis_catus", "\U0001f431"),
        ("mammalia_carnivora_canidae", "\U0001f415"),
        ("mammalia_artiodactyla_cervidae", "\U0001f98c"),
        ("mammalia_carnivora_ursidae", "\U0001f43b"),
        ("mammalia_carnivora_procyonidae_procyon_lotor", "\U0001f99d"),
        ("mammalia_rodentia_sciuridae", "\U0001f43f️"),
        ("mammalia_lagomorpha_leporidae_sylvilagus_floridanus", "\U0001f430"),
        ("mammalia_didelphimorphia_didelphidae", "\U0001f400"),
        ("mammalia_carnivora_mephitidae", "\U0001f9a8"),
        ("mammalia_artiodactyla_bovidae", "\U0001f404"),
        ("mammalia_rodentia", "\U0001f42d"),
        ("reptilia_squamata_colubridae", "\U0001f40d"),
        ("reptilia_testudines", "\U0001f422"),
        ("reptilia_squamata_gekkonidae", "\U0001f98e"),
        ("amphibia_anura", "\U0001f438"),
        ("mammalia_primates", "\U0001f9d1"),
        ("mammalia", "\U0001f43e"),
        ("unknown", "❓"),
        ("animal", "❓"),
    ],
)
def test_get_species_icon_mappings(species, icon):
    assert get_species_icon(species) == icon


def test_get_species_icon_normalizes_case_spaces_and_hyphens():
    assert get_species_icon("MAMMALIA-CARNIVORA FELIDAE") == "\U0001f431"


def test_get_species_icon_bird_check_wins_over_family_checks():
    # QUIRK: the bird branch runs first, so a hypothetical bird name containing
    # a mammal family token still gets a bird -- asserted as-is to detect
    # rewrite drift.
    assert get_species_icon("bird_felidae") == "\U0001f426"


def test_get_species_icon_reptilia_prefix_needed_for_lizard_default():
    """Reptile default is a lizard only under the reptilia/squamata branch."""
    assert get_species_icon("reptilia_reptile") == "\U0001f98e"
    # QUIRK: 'reptile' alone does not start with 'reptilia' and contains no
    # 'squamata', so it falls through to the generic paw
    # -- asserted as-is to detect rewrite drift.
    assert get_species_icon("reptile") == "\U0001f43e"


def test_get_species_icon_defaults_to_paw_for_anything_unrecognized():
    assert get_species_icon("totally_made_up_thing") == "\U0001f43e"
    assert get_species_icon("vehicle") == "\U0001f43e"
    assert get_species_icon("%%%") == "\U0001f43e"


def test_get_species_icon_blank_is_a_paw_not_a_question_mark():
    # QUIRK: 'blank' maps to the 'Empty Frame' display name but gets the
    # generic animal paw icon -- asserted as-is to detect rewrite drift.
    assert get_common_name("blank") == "Empty Frame"
    assert get_species_icon("blank") == "\U0001f43e"
    assert get_species_icon("empty") == "\U0001f43e"


def test_get_species_icon_substring_matches_are_greedy():
    # QUIRK: 'cat' is matched as a substring, so unrelated words containing it
    # get a cat icon -- asserted as-is to detect rewrite drift.
    assert get_species_icon("cattle") == "\U0001f431"
    assert get_species_icon("mammalia_artiodactyla_bovidae_bos_taurus") == "\U0001f404"


def test_icon_and_display_name_can_disagree():
    """The rabbit-as-Mammal case: display says Mammal, icon says rabbit."""
    # QUIRK: get_common_name and get_species_icon use independent resolution
    # orders -- asserted as-is to detect rewrite drift.
    assert get_common_name("mammalia_lagomorpha") == "Mammal"
    assert get_species_icon("mammalia_lagomorpha") == "\U0001f430"


# ==========================================================================
# end-to-end: filename -> display name + icon
# ==========================================================================


@pytest.mark.parametrize(
    "filename, display, icon",
    [
        (
            "1788800356_mammalia_lagomorpha_leporidae_sylvilagus_floridanus.mp4",
            "Eastern Cottontail",
            "\U0001f430",
        ),
        ("1788792747_bird.mp4", "Bird", "\U0001f426"),
        ("1788792411_blank.mp4", "Unknown", "❓"),
        ("1766587074_bird_passeriformes_cardinalidae.mp4", "Cardinal", "\U0001f426"),
        ("1766589347_mammalia_carnivora_canidae.mp4", "Dog/Canid", "\U0001f415"),
        ("1778509786_reptilia_reptile.mp4", "Reptile", "\U0001f98e"),
        (
            "1788800356_bird+mammalia_carnivora_felidae.mp4",
            "Bird, Cat",
            "\U0001f426",
        ),
    ],
)
def test_filename_to_display_and_icon(parse, filename, display, icon):
    got_display, raw = parse(filename)
    assert got_display == display
    assert get_species_icon(raw) == icon
