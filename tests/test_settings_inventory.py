"""The settings page and the config store describe the same set of fields.

CLAUDE.md states the rule: GENERAL_FIELDS / CAMERA_FIELDS in configstore.py
list the managed keys and whether the pipeline reads each one live; the client
inventory (GENERAL_SECTIONS / CAMERA_GROUPS in settings.js) must carry the
matching restart flag, and adding a field means touching both lists. Nothing
checked it, and the two had already drifted: a control the server did not
manage sat on the page doing nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

from animaltracker.configstore import CAMERA_FIELDS, GENERAL_FIELDS

SETTINGS = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "static"
            / "views" / "settings.js").read_text()

# Blocks the server manages whole, whose sub-keys the page lists one by one.
BLOCK_FIELDS = {"onvif", "ptz_tracking"}
# Client-only: folded into `onvif: {...}` or `onvif: null` before the save.
CLIENT_ONLY = {"onvif.enabled"}


BOUNDS = {
    "GENERAL_SECTIONS": ("var GENERAL_SECTIONS = [", "var CAMERA_GROUPS = ["),
    "CAMERA_GROUPS": ("var CAMERA_GROUPS = [", "var GENERAL_SPECS = "),
}


def inventory(marker: str) -> dict:
    """{key: needs_restart} for one client inventory.

    Scanned from one ``key:`` to the next rather than by matching braces: a
    field's own spec can contain them (a validation pattern like /^[A-Z]{3}$/),
    which is what hid two fields from an earlier version of this test.
    """
    head, tail = BOUNDS[marker]
    start = SETTINGS.index(head)
    block = SETTINGS[start:SETTINGS.index(tail, start)]
    hits = list(re.finditer(r"key: '([^']+)'", block))
    out = {}
    for i, hit in enumerate(hits):
        entry = block[hit.start():hits[i + 1].start() if i + 1 < len(hits) else len(block)]
        out[hit.group(1)] = "restart: true" in entry
    return out


def managed(fields: dict) -> dict:
    """{key: needs_restart} for the server's list; live=True means no restart."""
    return {key: not live for key, live in fields.items() if key not in BLOCK_FIELDS}


def test_the_page_offers_exactly_what_the_store_manages():
    client = inventory("GENERAL_SECTIONS")
    server = managed(GENERAL_FIELDS)

    assert set(client) - set(server) == set(), "on the page, not managed by the store"
    assert set(server) - set(client) == set(), "managed by the store, not on the page"


def test_every_camera_field_is_on_the_page_under_the_block_it_belongs_to():
    client = set(inventory("CAMERA_GROUPS")) - CLIENT_ONLY
    server = set(managed(CAMERA_FIELDS))
    blocks = tuple(b + "." for b in BLOCK_FIELDS)

    assert server - client == set(), "managed by the store, not on the page"
    stray = {k for k in client - server if not k.startswith(blocks)}
    assert stray == set(), "on the page, neither managed nor part of a managed block"


def test_the_restart_flags_agree():
    for marker, fields in (("GENERAL_SECTIONS", GENERAL_FIELDS), ("CAMERA_GROUPS", CAMERA_FIELDS)):
        client, server = inventory(marker), managed(fields)
        disagree = {k: (client[k], server[k]) for k in set(client) & set(server) if client[k] != server[k]}
        assert disagree == {}, f"{marker}: client restart flag vs server (live=False)"
