"""Process-wide logging configuration.

Two problems this module exists to solve:

1. ``onvif-zeep`` calls ``logging.basicConfig(level=INFO)`` when it is
   imported.  A later plain ``basicConfig`` (the CLI's) is then a silent
   no-op, and the process runs with Python's default ``LEVEL:name:message``
   format and no timestamp.  ``configure_logging()`` passes ``force=True``
   so the intended configuration always wins.

2. Under systemd everything the process writes to stderr lands in the
   journal at priority 6 (informational), so ``journalctl -p err`` and the
   web UI's level filter cannot see the app's ERROR/WARNING lines.  journald
   honours a ``<N>`` syslog-level prefix on each line (``SyslogLevelPrefix=``
   defaults to on), so when stderr is the journal every line is prefixed
   with the record's priority and the timestamp -- which the journal adds
   itself -- is dropped.

Elsewhere (a terminal, launchd, Windows) lines are timestamped, and
``attach_file_log()`` adds a rotating ``logs/animaltracker.log`` so the web
UI's log page has something to read without journalctl.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

#: Format used when stderr is the systemd journal (it timestamps for us).
JOURNAL_FORMAT = "%(levelname)s %(name)s: %(message)s"
#: Format used everywhere else (terminal, redirected file, launchd).
FILE_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

APP_LOG_NAME = "animaltracker.log"
APP_LOG_MAX_BYTES = 10 * 1024 * 1024
APP_LOG_BACKUPS = 5

#: Environment override for the journal detection: "1"/"0".  Useful when
#: stderr is piped into something that forwards to the journal, or to get
#: timestamped output from a unit for a one-off debug run.
JOURNAL_ENV_OVERRIDE = "ANIMALTRACKER_LOG_JOURNAL"

_journal_mode: Optional[bool] = None


def syslog_priority(levelno: int) -> int:
    """Map a ``logging`` level to a syslog priority (sd-daemon(3))."""
    if levelno >= logging.CRITICAL:
        return 2
    if levelno >= logging.ERROR:
        return 3
    if levelno >= logging.WARNING:
        return 4
    if levelno >= logging.INFO:
        return 6
    return 7


class JournalPrefixFormatter(logging.Formatter):
    """Prefix every output line with ``<priority>`` for journald.

    journald splits a stderr stream on newlines and files each line as its
    own entry, so multi-line records (tracebacks) need the prefix on every
    line; otherwise only the first line would carry the level and the rest
    of the traceback would be logged as informational.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        prefix = f"<{syslog_priority(record.levelno)}>"
        return prefix + text.replace("\n", "\n" + prefix)


def stderr_is_journal() -> bool:
    """True when systemd connected this process's stderr to the journal.

    systemd exports ``JOURNAL_STREAM=<dev>:<inode>`` of the journal socket;
    it is compared with the actual stderr so the variable leaking into a
    child whose stderr was redirected does not fool us.  The
    ``ANIMALTRACKER_LOG_JOURNAL`` variable overrides the detection.
    """
    override = os.environ.get(JOURNAL_ENV_OVERRIDE)
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    value = os.environ.get("JOURNAL_STREAM")
    if not value:
        return False
    try:
        dev, inode = value.split(":", 1)
        st = os.fstat(2)
        return int(dev) == st.st_dev and int(inode) == st.st_ino
    except (ValueError, OSError):
        return False


def configure_logging(level: int = logging.INFO, *, journal: Optional[bool] = None) -> bool:
    """Configure the root logger for this process.

    Safe to call after third-party imports: ``force=True`` replaces any
    handler a library installed at import time.  ``journal`` forces the
    journald style on or off; by default it is auto-detected.  Returns the
    mode in effect (True when writing journald-style lines).
    """
    global _journal_mode
    if journal is None:
        journal = stderr_is_journal()
    handler = logging.StreamHandler()  # stderr
    if journal:
        handler.setFormatter(JournalPrefixFormatter(JOURNAL_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(FILE_FORMAT))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    _journal_mode = journal
    return journal


def attach_file_log(logs_root: Path, level: int = logging.NOTSET) -> Optional[Path]:
    """Also write the app log to ``logs_root/animaltracker.log`` (rotating).

    Skipped when stderr is the journal: there the journal *is* the log
    store and a second copy would only double the disk use.  Returns the
    path being written, or None when skipped.
    """
    journal = _journal_mode if _journal_mode is not None else stderr_is_journal()
    if journal:
        return None
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / APP_LOG_NAME
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=APP_LOG_MAX_BYTES, backupCount=APP_LOG_BACKUPS, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return path
