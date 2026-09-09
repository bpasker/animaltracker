"""Journal entries written before stderr lines carried a syslog priority are
all priority 6; the log page falls back to the level name in the text."""
from animaltracker.web import _LEGACY_LEVEL_RE


def test_legacy_level_regex_accepts_both_prefix_styles():
    assert _LEGACY_LEVEL_RE.match("ERROR:animaltracker.pipeline:Unable to open").group(1) == "ERROR"
    assert _LEGACY_LEVEL_RE.match("WARNING animaltracker.pipeline: Stream lost").group(1) == "WARNING"
    assert _LEGACY_LEVEL_RE.match("CRITICAL animaltracker.web: down").group(1) == "CRITICAL"
    assert _LEGACY_LEVEL_RE.match("INFO animaltracker.pipeline: [PERF] cam1") is None
    assert _LEGACY_LEVEL_RE.match("[tls @ 0x1] IO error: Connection timed out") is None
    assert _LEGACY_LEVEL_RE.match("ERRORS were found") is None
