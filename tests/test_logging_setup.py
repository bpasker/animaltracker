"""configure_logging() must win over library basicConfig calls, and under
systemd every line must carry a syslog priority so journalctl -p works."""
import logging
import logging.handlers
import os
import sys

import pytest

from animaltracker import logging_setup as ls


@pytest.fixture
def clean_root():
    """Root logger whose level/mode are restored and whose handlers added by
    the code under test are removed and closed afterwards. pytest's own
    capture handlers are (re)attached per test phase, so they are neither
    saved nor restored here."""
    root = logging.getLogger()
    saved_level = root.level
    saved_mode = ls._journal_mode
    yield root
    for h in list(root.handlers):
        if type(h) in (logging.StreamHandler, logging.handlers.RotatingFileHandler):
            root.removeHandler(h)
            h.close()
    root.setLevel(saved_level)
    ls._journal_mode = saved_mode


def _bare_root(root):
    """Drop every handler, as a fresh process has none. pytest attaches its
    capture handler at the start of each phase, so this must run inside the
    test body, not in a fixture."""
    for h in list(root.handlers):
        root.removeHandler(h)


def test_syslog_priority_mapping():
    assert ls.syslog_priority(logging.CRITICAL) == 2
    assert ls.syslog_priority(logging.ERROR) == 3
    assert ls.syslog_priority(logging.WARNING) == 4
    assert ls.syslog_priority(logging.INFO) == 6
    assert ls.syslog_priority(logging.DEBUG) == 7
    assert ls.syslog_priority(logging.WARNING + 5) == 4  # custom levels round down


def test_journal_formatter_prefixes_every_line_including_tracebacks():
    fmt = ls.JournalPrefixFormatter(ls.JOURNAL_FORMAT)
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "animaltracker.x", logging.ERROR, __file__, 1, "failed: %s", ("clip",), sys.exc_info()
        )
    lines = fmt.format(record).splitlines()
    assert lines[0] == "<3>ERROR animaltracker.x: failed: clip"
    assert len(lines) > 2, "traceback expected"
    assert all(line.startswith("<3>") for line in lines)
    assert any("ValueError: boom" in line for line in lines)

    info = logging.LogRecord("animaltracker.y", logging.INFO, __file__, 1, "ok", (), None)
    assert fmt.format(info) == "<6>INFO animaltracker.y: ok"


def test_configure_replaces_the_handler_a_library_installed(clean_root):
    _bare_root(clean_root)
    # What onvif-zeep does at import time.
    logging.basicConfig(level=logging.INFO)
    assert clean_root.handlers[-1].formatter._fmt == logging.BASIC_FORMAT

    assert ls.configure_logging(journal=False) is False
    assert len(clean_root.handlers) == 1
    assert clean_root.handlers[0].formatter._fmt == ls.FILE_FORMAT
    assert clean_root.level == logging.INFO

    assert ls.configure_logging(journal=True) is True
    assert len(clean_root.handlers) == 1
    assert isinstance(clean_root.handlers[0].formatter, ls.JournalPrefixFormatter)


def test_stderr_is_journal_detection(monkeypatch):
    monkeypatch.delenv(ls.JOURNAL_ENV_OVERRIDE, raising=False)
    monkeypatch.delenv("JOURNAL_STREAM", raising=False)
    assert ls.stderr_is_journal() is False

    monkeypatch.setenv("JOURNAL_STREAM", "garbage")
    assert ls.stderr_is_journal() is False

    st = os.fstat(2)
    monkeypatch.setenv("JOURNAL_STREAM", f"{st.st_dev + 1}:{st.st_ino}")
    assert ls.stderr_is_journal() is False, "must compare against the real stderr"
    monkeypatch.setenv("JOURNAL_STREAM", f"{st.st_dev}:{st.st_ino}")
    assert ls.stderr_is_journal() is True

    monkeypatch.setenv(ls.JOURNAL_ENV_OVERRIDE, "0")
    assert ls.stderr_is_journal() is False
    monkeypatch.delenv("JOURNAL_STREAM")
    monkeypatch.setenv(ls.JOURNAL_ENV_OVERRIDE, "1")
    assert ls.stderr_is_journal() is True


def test_attach_file_log_only_when_stderr_is_not_the_journal(clean_root, tmp_path):
    ls.configure_logging(journal=True)
    assert ls.attach_file_log(tmp_path / "journal-logs") is None
    assert not (tmp_path / "journal-logs").exists()

    ls.configure_logging(journal=False)
    path = ls.attach_file_log(tmp_path / "logs")
    assert path == tmp_path / "logs" / ls.APP_LOG_NAME
    logging.getLogger("animaltracker.test").warning("hello %s", "file")
    for h in clean_root.handlers:
        h.flush()
    text = path.read_text(encoding="utf-8")
    assert "WARNING animaltracker.test: hello file" in text
    assert text.startswith("["), "timestamped format expected in the file"
