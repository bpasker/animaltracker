"""Three ways the edges gave up instead of answering (bug hunt 2.21, 3.14, 3.9).

- ``discover`` called ``camera.onvif.credentials()`` on every camera; every
  production camera lacks an ``onvif:`` block, so it died on the first with
  an AttributeError, and a camera that did not answer aborted the rest.
  ``ptz-test`` and ``zoom-calibrate`` had the same traceback for a camera
  without ONVIF.
- Two cameras with one id validated; everything keyed by id (the pipeline's
  workers, the settings page's save) then silently kept one.
- The logs API turned a non-integer ``minutes`` or ``limit`` into a 500, and
  ``limit=0`` returned everything the fetch cap allowed (``logs[-0:]``).
"""
from __future__ import annotations

import argparse
import logging
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import cli as cli_mod
from animaltracker.cli import cmd_discover, cmd_ptz_test, cmd_zoom_calibrate
from animaltracker.config import load_runtime_config
from animaltracker.web import _query_int


def write_config(tmp_path, cameras: str) -> Path:
    cfg = tmp_path / "cameras.yml"
    cfg.write_text(textwrap.dedent(f"""
        general:
          storage_root: {tmp_path / 'storage'}
          logs_root: {tmp_path / 'logs'}
          notification:
            pushover_app_token_env: PUSHOVER_APP_TOKEN
            pushover_user_key_env: PUSHOVER_USER_KEY
        cameras:
    """) + textwrap.indent(textwrap.dedent(cameras), "  "))
    return cfg


PLAIN = """
- id: cam1
  name: Front
  rtsp:
    uri: rtsp://cam1.invalid/stream
- id: cam2
  name: Back
  rtsp:
    uri: rtsp://cam2.invalid/stream
"""

WITH_ONVIF = PLAIN + """
- id: ptz
  name: Zoom
  rtsp:
    uri: rtsp://ptz.invalid/stream
  onvif:
    host: ptz.invalid
    port: 80
    username_env: PTZ_USER
    password_env: PTZ_PASS
"""


class Unreachable:
    """OnvifClient stand-in for a camera that accepts nothing."""
    built = []

    def __init__(self, host, port, username, password):
        Unreachable.built.append(host)

    def get_status(self):
        raise ConnectionError("no route to host")


# --- discover -------------------------------------------------------------------------------

def test_discover_skips_cameras_without_onvif_and_finishes(tmp_path, caplog):
    cfg = write_config(tmp_path, PLAIN)

    with caplog.at_level(logging.INFO):
        code = cmd_discover(argparse.Namespace(config=str(cfg), inspect=False, presets=False))

    assert code == 0
    assert "Camera cam1 has no onvif: block; skipped" in caplog.text
    assert "Camera cam2 has no onvif: block; skipped" in caplog.text


def test_discover_reports_a_camera_that_fails_and_goes_on(tmp_path, caplog, monkeypatch):
    cfg = write_config(tmp_path, WITH_ONVIF)
    monkeypatch.setenv("PTZ_USER", "u")
    monkeypatch.setenv("PTZ_PASS", "p")
    monkeypatch.setattr(cli_mod, "OnvifClient", Unreachable)
    Unreachable.built = []

    with caplog.at_level(logging.INFO):
        code = cmd_discover(argparse.Namespace(config=str(cfg), inspect=False, presets=False))

    assert code == 1, "a camera that failed is a failed command"
    assert Unreachable.built == ["ptz.invalid"]
    assert "Camera ptz (ptz.invalid:80): no route to host" in caplog.text
    assert "Camera cam2 has no onvif: block; skipped" in caplog.text


@pytest.mark.parametrize("command, args", [
    (cmd_ptz_test, {"camera": "cam1", "find_working": False}),
    (cmd_zoom_calibrate, {"wide_camera": "cam1", "zoom_camera": "cam2"}),
])
def test_the_ptz_commands_say_why_instead_of_a_traceback(tmp_path, caplog, command, args):
    cfg = write_config(tmp_path, PLAIN)

    with caplog.at_level(logging.ERROR):
        code = command(argparse.Namespace(config=str(cfg), **args))

    assert code == 1
    assert "has no onvif: block" in caplog.text


# --- duplicate camera ids -------------------------------------------------------------------

def test_two_cameras_with_one_id_are_refused(tmp_path):
    cfg = write_config(tmp_path, """
        - id: cam1
          name: Front
          rtsp:
            uri: rtsp://a.invalid/stream
        - id: cam1
          name: Also front
          rtsp:
            uri: rtsp://b.invalid/stream
    """)

    with pytest.raises(ValueError, match="camera id 'cam1' is used more than once"):
        load_runtime_config(cfg)


def test_distinct_ids_still_load(tmp_path):
    assert [c.id for c in load_runtime_config(write_config(tmp_path, PLAIN)).cameras] == ["cam1", "cam2"]


# --- the logs API's integers ----------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (None, 200),          # absent
    ("", 200),            # blank
    ("abc", 200),         # not a number: the default, not a 500
    ("0", 1),             # limit=0 used to mean "everything"
    ("-5", 1),
    ("50", 50),
    (" 50 ", 50),
    ("99999", 2000),      # the cap
])
def test_query_int_is_bounded_and_forgiving(raw, expected):
    assert _query_int(raw, default=200, low=1, high=2000) == expected


def test_the_logs_handler_reads_both_through_it():
    src = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "web.py").read_text()
    handler = src[src.index("async def handle_get_logs"):]
    handler = handler[:handler.index("\n    async def ")]
    assert "minutes = _query_int(request.query.get('minutes')" in handler
    assert "limit = _query_int(request.query.get('limit')" in handler
    assert not re.search(r"(?<!_query_)int\(request\.query\.get\(", handler)
