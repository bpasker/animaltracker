"""Recalling a PTZ preset from the Live page.

Background (bug hunt, 2026-09-19): the app's ``api.ptz.gotoPreset`` posted
``{"token": ...}`` while ``handle_ptz_goto_preset`` read ``preset_token``, the
name the retired server-rendered page had used. Every recall answered
400 "Missing preset_token" and the page toasted "Could not recall that
preset".
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker.web import WebServer

API_JS = Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static" / "core" / "api.js"


class JsonRequest:
    def __init__(self, body, camera_id="cam2") -> None:
        self._body = body
        self.match_info = {"camera_id": camera_id}

    async def json(self):
        return self._body


class FakeOnvif:
    def __init__(self) -> None:
        self.recalled: list = []

    def ptz_goto_preset(self, profile_token, preset_token, speed):
        self.recalled.append((profile_token, preset_token))


@pytest.fixture
def server(tmp_path):
    onvif = FakeOnvif()
    worker = SimpleNamespace(onvif_client=onvif, onvif_profile_token="Profile_1")
    return WebServer({"cam2": worker}, tmp_path, tmp_path / "logs", port=0), onvif


@pytest.mark.parametrize("body", [{"preset_token": "3"}, {"token": "3"}])
def test_the_camera_goes_to_the_preset_under_either_field_name(server, body):
    web_server, onvif = server

    resp = asyncio.run(web_server.handle_ptz_goto_preset(JsonRequest(body)))

    assert resp.status == 200
    assert onvif.recalled == [("Profile_1", "3")]


def test_no_token_at_all_is_still_a_bad_request(server):
    web_server, onvif = server

    resp = asyncio.run(web_server.handle_ptz_goto_preset(JsonRequest({})))

    assert resp.status == 400 and onvif.recalled == []


def test_the_app_posts_the_field_the_server_documents():
    source = API_JS.read_text()
    body = re.search(r"gotoPreset: function \(id, token, opts\) \{.*?\n    \},", source, re.S).group(0)

    assert "body: { preset_token: token }" in body
