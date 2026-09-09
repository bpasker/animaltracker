"""/api/logs: the level/logger prefix is parsed off every line, type filters
run on the message body, and the type or level filter is pushed down into
journalctl's own --grep so a sparse type covers the whole time range.

Handlers are driven directly with a minimal fake request; journalctl is a
fake subprocess that returns a fixed mix of new-format, old-format, systemd
and FFmpeg lines and ignores the pushdown (the Python filters must be right
on their own).
"""
import asyncio
import json
import re
from datetime import datetime

import pytest

from animaltracker import web
from animaltracker.web import (
    WebServer, _classify_log_line, _journal_grep_pattern, _matches_log_filter,
)


class FakeRequest:
    def __init__(self, query=None):
        self.query = query or {}
        self.match_info = {}


def call(handler, **query):
    resp = asyncio.run(handler(FakeRequest(query)))
    return resp.status, json.loads(resp.body.decode())


JOURNAL = [  # (PRIORITY, MESSAGE), oldest first
    (6, "INFO animaltracker.pipeline: [PERF] cam1: capture=20.0fps infer=3.6fps drop=1 (1.0%) | window=60s"),
    (6, "INFO:animaltracker.tracker:Spatial merge: Track 3 (x) + Track 4 (y)"),
    (6, "ERROR:animaltracker.pipeline:Unable to open RTSP stream for cam1; retrying in 5s"),
    (4, "WARNING animaltracker.pipeline: Stream lost for cam1; reconnecting..."),
    (6, "[tls @ 0x1] IO error: Connection timed out"),
    (6, "INFO animaltracker.cli: Loading secrets from /opt/speciesnet/animaltracker/config/secrets.env"),
    (6, "INFO animaltracker.pipeline: Started tracking animal on cam2"),
]


def fake_journal(monkeypatch, calls, fail_with_grep=False):
    """journalctl stand-in: records argv, returns JOURNAL as `-o json` lines."""

    async def fake_exec(*argv, **kw):
        calls.append(list(argv))

        class Proc:
            returncode = 0

            async def communicate(self):
                if fail_with_grep and '--grep' in argv:
                    self.returncode = 1
                    return b"", b"Compiled without pattern matching support"
                base = 1_757_000_000_000_000
                out = [json.dumps({
                    "__REALTIME_TIMESTAMP": str(base + i * 1_000_000),
                    "PRIORITY": str(prio),
                    "MESSAGE": msg,
                }) for i, (prio, msg) in enumerate(JOURNAL)]
                return ("\n".join(out) + "\n").encode(), b""

        return Proc()

    monkeypatch.setattr(web.asyncio, "create_subprocess_exec", fake_exec)


@pytest.fixture
def server(tmp_path):
    srv = WebServer({}, tmp_path, tmp_path / "logs", port=0)
    srv.workers = {"cam1": None, "cam2": None}  # only the ids are read
    return srv


def test_classify_new_and_old_prefixes_and_raw_lines():
    assert _classify_log_line("INFO animaltracker.pipeline: [PERF] cam1: x", 6) == (
        "info", "animaltracker.pipeline", "[PERF] cam1: x")
    assert _classify_log_line("ERROR:animaltracker.pipeline:Unable to open", 6) == (
        "error", "animaltracker.pipeline", "Unable to open")
    assert _classify_log_line("WARNING ptz.decisions: [MOVE] x", 4) == ("warning", "ptz.decisions", "[MOVE] x")
    assert _classify_log_line("CRITICAL animaltracker.web: down", 2) == ("error", "animaltracker.web", "down")
    # No prefix: the journal priority decides.
    tls = "[tls @ 0x1] IO error: Connection timed out"
    assert _classify_log_line(tls, 6) == ("info", "", tls)
    assert _classify_log_line("animaltracker.service: Main process exited", 3)[0] == "error"
    # Timestamped app-log file line.
    assert _classify_log_line("[2026-09-09 06:19:31,477] WARNING animaltracker.smoke: warn line") == (
        "warning", "animaltracker.smoke", "warn line")
    # File line without a prefix keeps the old keyword heuristic.
    assert _classify_log_line("Traceback (most recent call last):")[0] == "info"
    assert _classify_log_line("RuntimeError: boom")[0] == "error"


def test_tracking_filter_ignores_the_logger_name_and_paths():
    perf = "[PERF] cam1: capture=20.0fps infer=3.6fps drop=1 (1.0%)"
    assert not _matches_log_filter(perf, "tracking", "animaltracker.pipeline", "info")
    assert not _matches_log_filter(
        "Loading secrets from /opt/speciesnet/animaltracker/config/secrets.env",
        "tracking", "animaltracker.cli", "info")
    assert _matches_log_filter("Spatial merge: Track 3 + Track 4", "tracking", "animaltracker.tracker", "info")
    assert _matches_log_filter("Started tracking animal on cam2", "tracking", "animaltracker.pipeline", "info")
    assert _matches_log_filter("Saved track thumbnail: /x/animaltracker/storage/a.jpg",
                               "tracking", "animaltracker.postprocess", "info")
    assert _matches_log_filter("Object tracking enabled (ByteTrack, lost_buffer=120)",
                               "tracking", "animaltracker.tracker", "info")
    # Logger membership alone is enough.
    assert _matches_log_filter("something unrelated", "tracking", "animaltracker.tracker", "info")


def test_errors_filter_uses_the_level_not_only_words():
    assert _matches_log_filter("Unable to open RTSP stream for cam1", "errors", "animaltracker.pipeline", "error")
    assert not _matches_log_filter("Connected to stream for cam1", "errors", "animaltracker.pipeline", "info")
    assert _matches_log_filter("[tls @ 0x1] IO error: Connection timed out", "errors", "", "info")


def test_grep_pushdown_is_a_superset_of_the_python_filter():
    # Python's re supports the possessive quantifier since 3.11, so the PCRE
    # journalctl gets can be exercised here too.
    pat = re.compile(_journal_grep_pattern("tracking", "all"), re.IGNORECASE)
    for _prio, line in JOURNAL:
        level, logger, body = _classify_log_line(line, 6)
        if _matches_log_filter(body, "tracking", logger, level):
            assert pat.search(line), line
    assert not pat.search("INFO animaltracker.pipeline: [PERF] cam1: capture=20.0fps")
    assert not pat.search("INFO:animaltracker.pipeline:[PERF] cam1: capture=20.0fps")
    assert not pat.search("INFO animaltracker.cli: Loading secrets from /opt/speciesnet/animaltracker/config/x")
    assert pat.search("INFO animaltracker.tracker: anything from the tracker logger")

    assert _journal_grep_pattern("all", "all") is None
    assert _journal_grep_pattern("no-http", "all") is None
    err = re.compile(_journal_grep_pattern("all", "error"), re.IGNORECASE)
    assert err.search("ERROR:animaltracker.pipeline:Unable to open")
    assert err.search("ERROR animaltracker.pipeline: Unable to open")
    assert err.search("animaltracker.service: Main process exited, code=killed")
    assert not err.search("WARNING animaltracker.pipeline: Stream lost")
    errs = re.compile(_journal_grep_pattern("errors", "all"), re.IGNORECASE)
    assert errs.search("ERROR animaltracker.pipeline: Unable to open RTSP stream")
    assert errs.search("[tls @ 0x1] IO error: Connection timed out")
    assert not errs.search("INFO animaltracker.pipeline: Connected to stream for cam1")


def test_tracking_type_returns_only_tracking_lines_with_logger_and_body(monkeypatch, server):
    calls = []
    fake_journal(monkeypatch, calls)
    status, body = call(server.handle_get_logs, type="tracking", limit="50", minutes="60")
    assert status == 200
    assert body["source"] == "journalctl"
    assert [e["message"] for e in body["logs"]] == [  # newest first
        "Started tracking animal on cam2",
        "Spatial merge: Track 3 (x) + Track 4 (y)",
    ]
    assert [e["logger"] for e in body["logs"]] == ["animaltracker.pipeline", "animaltracker.tracker"]
    assert [e["camera"] for e in body["logs"]] == ["cam2", ""]
    assert body["skipped"] == len(JOURNAL) - 2
    argv = calls[0]
    assert "--grep" in argv and "--case-sensitive=false" in argv
    assert "-p" not in argv
    assert argv[argv.index("-n") + 1] == "1000"
    assert argv[argv.index("--since") + 1] == "60 minutes ago"


def test_error_level_includes_pre_prefix_entries_and_strips_prefix(monkeypatch, server):
    calls = []
    fake_journal(monkeypatch, calls)
    _, body = call(server.handle_get_logs, level="error", limit="50")
    assert [(e["level"], e["message"]) for e in body["logs"]] == [
        ("error", "Unable to open RTSP stream for cam1; retrying in 5s"),
    ]
    assert "-p" not in calls[0] and "--grep" in calls[0]
    _, body = call(server.handle_get_logs, level="warning", limit="50")
    assert [e["level"] for e in body["logs"]] == ["warning", "error"]


def test_all_types_keep_every_line_and_level_comes_from_priority(monkeypatch, server):
    calls = []
    fake_journal(monkeypatch, calls)
    _, body = call(server.handle_get_logs, limit="50")
    assert len(body["logs"]) == len(JOURNAL)
    tls = [e for e in body["logs"] if e["message"].startswith("[tls")][0]
    assert (tls["level"], tls["logger"]) == ("info", "")
    assert "--grep" not in calls[0]
    assert argv_n(calls[0]) == "500"


def argv_n(argv):
    return argv[argv.index("-n") + 1]


def test_grep_fallback_when_journalctl_lacks_pattern_support(monkeypatch, server):
    calls = []
    fake_journal(monkeypatch, calls, fail_with_grep=True)
    _, body = call(server.handle_get_logs, type="tracking", limit="50")
    assert len(calls) == 2
    assert "--grep" in calls[0] and "--grep" not in calls[1]
    assert argv_n(calls[1]) == "2000"
    assert body["source"] == "journalctl"
    assert [e["message"] for e in body["logs"]] == [
        "Started tracking animal on cam2",
        "Spatial merge: Track 3 (x) + Track 4 (y)",
    ]


def test_file_log_fallback_parses_the_prefix(monkeypatch, tmp_path):
    async def no_journal(*a, **k):
        raise FileNotFoundError("journalctl")

    monkeypatch.setattr(web.asyncio, "create_subprocess_exec", no_journal)
    logs = tmp_path / "logs"
    logs.mkdir()
    stamp = datetime.now(tz=web.CENTRAL_TZ).strftime('%Y-%m-%d %H:%M:%S') + ',000'
    (logs / "animaltracker.log").write_text(
        f"[{stamp}] INFO animaltracker.pipeline: [PERF] cam1: capture=20.0fps\n"
        f"[{stamp}] WARNING animaltracker.tracker: Merging Track 1 into Track 2\n",
        encoding="utf-8",
    )
    srv = WebServer({}, tmp_path, logs, port=0)
    _, body = call(srv.handle_get_logs, type="tracking", limit="50")
    assert body["source"] == "logfile"
    assert [(e["level"], e["logger"], e["message"]) for e in body["logs"]] == [
        ("warning", "animaltracker.tracker", "Merging Track 1 into Track 2"),
    ]
