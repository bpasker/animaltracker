"""Repeat alerts: a camera that has just alerted for an animal holds back
further alerts for it for a while, and every clip is still recorded.

Background (2026-09-25): a rabbit that sat on the Jess Dahua lawn made a
five-minute clip and a Pushover alert to two people every five minutes; the
week before, 30 of the 120 alerts were squirrels at Otteson1, half of them
within an hour of the one before. The exclusion lists could silence them only
by deleting the clips. ``notification.cooldown_minutes`` (every animal) and
``notification.species_cooldowns`` (an animal of its own) hold back the
alert alone, per camera and animal, timed by the events' start times and
remembered across a restart.
"""
from __future__ import annotations

import copy
import json
import logging
import time

import pytest
import requests
import yaml
from pydantic import ValidationError

from animaltracker import configstore, notification
from animaltracker.config import NotificationSettings, SpeciesCooldown
from animaltracker.notification import AlertTimes, NotificationContext, PushoverNotifier, cooldown_for
from animaltracker.species_names import species_matches

SQUIRREL = "mammalia_rodentia_sciuridae"
RABBIT = "mammalia_lagomorpha_leporidae"
DOG = "mammalia_carnivora_canidae"
DEER = "mammalia_artiodactyla_cervidae"
RACCOON = "mammalia_carnivora_procyonidae"
CARDINAL = "bird_passeriformes_cardinalidae"
MINUTE = 60.0
# Now, not a fixed date: the file drops times older than the longest cooldown.
T0 = float(int(time.time()))


# --------------------------------------------------------------------------
# Which animal a name in the settings means
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label,name", [
    (SQUIRREL, "squirrel"),            # the alert's own name for it
    (SQUIRREL, "Squirrel"),
    (RABBIT, "rabbit"),
    (DOG, "dog"),                      # one word of "Dog/Canid"
    (RACCOON, "raccoon"),
    (SQUIRREL, "sciuridae"),           # a taxonomy level
    (SQUIRREL, "rodentia"),
    (SQUIRREL, "mammalia_rodentia"),   # the start of the label
    (SQUIRREL, "mammal"),              # the class, under either spelling
    (CARDINAL, "bird"),
    ("mammalia artiodactyla cervidae", "deer"),   # the spaced form recent detections use
])
def test_a_name_matches_the_animal_it_means(label, name):
    assert species_matches(label, name)


@pytest.mark.parametrize("label,name", [
    (SQUIRREL, "rabbit"),
    (RABBIT, "squirrel"),
    (DEER, "dog"),
    ("bobcat", "cat"),                 # whole words only
    (SQUIRREL, ""),
    ("", "squirrel"),
])
def test_a_name_does_not_match_other_animals(label, name):
    assert not species_matches(label, name)


def test_an_animal_of_its_own_takes_its_wait_and_the_rest_the_default():
    s = settings(cooldown_minutes=10, species_cooldowns=[{"species": "Squirrel", "minutes": 120}])
    assert cooldown_for(s, SQUIRREL) == notification.Cooldown("squirrel", 120.0)
    assert cooldown_for(s, DEER) == notification.Cooldown("deer", 10.0)
    assert cooldown_for(s, DOG) == notification.Cooldown("dog canid", 10.0)


def test_the_first_entry_that_matches_applies():
    s = settings(species_cooldowns=[{"species": "rodentia", "minutes": 30}, {"species": "squirrel", "minutes": 120}])
    assert cooldown_for(s, SQUIRREL) == notification.Cooldown("rodentia", 30.0)


# --------------------------------------------------------------------------
# The settings
# --------------------------------------------------------------------------

def settings(**extra):
    return NotificationSettings(pushover_app_token_env="PUSHOVER_APP_TOKEN",
                                pushover_user_key_env="PUSHOVER_USER_KEY", **extra)


def test_the_default_alerts_for_every_clip():
    s = settings()
    assert s.cooldown_minutes == 0 and s.species_cooldowns == []


@pytest.mark.parametrize("extra,where", [
    ({"cooldown_minutes": -1}, ("cooldown_minutes",)),
    ({"cooldown_minutes": 7 * 24 * 60 + 1}, ("cooldown_minutes",)),
    ({"species_cooldowns": [{"species": "squirrel", "minutes": -5}]}, ("species_cooldowns", 0, "minutes")),
    ({"species_cooldowns": [{"species": "  ", "minutes": 5}]}, ("species_cooldowns", 0, "species")),
    ({"species_cooldowns": [{"species": "squirrel", "minutes": 5},
                            {"species": "Squirrel", "minutes": 9}]}, ("species_cooldowns",)),
])
def test_bad_cooldowns_are_refused_with_their_path(extra, where):
    with pytest.raises(ValidationError) as exc:
        settings(**extra)
    assert [tuple(e["loc"]) for e in exc.value.errors()] == [where]


# --------------------------------------------------------------------------
# The notifier holds repeats back
# --------------------------------------------------------------------------

class FakeResponse:
    def raise_for_status(self):
        pass


@pytest.fixture
def posts(monkeypatch):
    sent = []

    def fake_post(url, data=None, files=None, timeout=None):
        sent.append({"user": data["user"], "title": data["title"]})
        return FakeResponse()

    monkeypatch.setattr(notification.requests, "post", fake_post)
    return sent


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "user-key")
    return monkeypatch


def clip(species, at, camera_id="cam1"):
    return NotificationContext(
        species=species, confidence=0.9, camera_id=camera_id, camera_name="Yard",
        clip_path="/srv/clips/cam1/2026/09/25/1.mp4", event_started_at=at, event_duration=30.0,
    )


def test_a_second_alert_for_the_same_animal_waits_for_the_cooldown(posts, caplog):
    n = PushoverNotifier(settings(species_cooldowns=[{"species": "rabbit", "minutes": 30}]))
    with caplog.at_level(logging.INFO):
        assert n.send(clip(RABBIT, T0)) == 1
        assert n.send(clip(RABBIT, T0 + 5 * MINUTE)) == 0
        assert n.send(clip(RABBIT, T0 + 29 * MINUTE)) == 0
    assert "Holding back the Rabbit alert from cam1: its last rabbit alert was 5 min from this one" in caplog.text
    assert "The clip is kept." in caplog.text
    assert n.send(clip(RABBIT, T0 + 31 * MINUTE)) == 1
    assert len(posts) == 2


def test_the_cooldown_runs_from_the_last_alert_sent_not_the_last_clip(posts):
    """"How often I get them": a squirrel that never leaves still alerts
    once per cooldown, rather than never again."""
    n = PushoverNotifier(settings(cooldown_minutes=30))
    sent = [n.send(clip(SQUIRREL, T0 + m * MINUTE)) for m in range(0, 95, 5)]
    assert [i * 5 for i, s in enumerate(sent) if s] == [0, 30, 60, 90]


def test_each_camera_and_each_animal_keeps_its_own_clock(posts):
    n = PushoverNotifier(settings(cooldown_minutes=60))
    assert n.send(clip(SQUIRREL, T0)) == 1
    assert n.send(clip(SQUIRREL, T0 + MINUTE, camera_id="cam2")) == 1
    assert n.send(clip(DEER, T0 + MINUTE)) == 1
    assert n.send(clip(SQUIRREL, T0 + 2 * MINUTE)) == 0


def test_zero_alerts_for_every_clip(posts):
    n = PushoverNotifier(settings(species_cooldowns=[{"species": "squirrel", "minutes": 0}], cooldown_minutes=60))
    assert [n.send(clip(SQUIRREL, T0 + m * MINUTE)) for m in range(3)] == [1, 1, 1]


def test_an_older_clip_analysed_late_counts_too(posts):
    """Two analysis slots can finish out of order; the event times decide."""
    n = PushoverNotifier(settings(cooldown_minutes=30))
    assert n.send(clip(SQUIRREL, T0)) == 1
    assert n.send(clip(SQUIRREL, T0 - 10 * MINUTE)) == 0
    assert n.send(clip(SQUIRREL, T0 - 45 * MINUTE)) == 1, "a visit of its own, 45 minutes earlier"
    assert n.alert_times.last("cam1", "squirrel") == T0, "the latest alert still sets the clock"


def test_an_alert_that_reached_nobody_does_not_start_the_cooldown(monkeypatch):
    calls = []

    def failing_post(url, data=None, files=None, timeout=None):
        calls.append(data["user"])
        if len(calls) == 1:
            raise requests.ConnectionError("pushover is down")
        return FakeResponse()

    monkeypatch.setattr(notification.requests, "post", failing_post)
    n = PushoverNotifier(settings(cooldown_minutes=60))
    assert n.send(clip(SQUIRREL, T0)) == 0
    assert n.alert_times.last("cam1", "squirrel") is None
    assert n.send(clip(SQUIRREL, T0 + MINUTE)) == 1


def test_a_camera_that_alerts_nobody_does_not_start_the_cooldown(posts):
    n = PushoverNotifier(settings(cooldown_minutes=60))
    assert n.send(clip(SQUIRREL, T0), destinations=[]) == 0
    assert n.alert_times.last("cam1", "squirrel") is None


def test_cooldown_edits_apply_without_rebuilding_the_notifier(posts):
    """The pipeline hands the notifier the live settings object, which the
    settings page updates in place (configstore.hot_apply)."""
    s = settings()
    n = PushoverNotifier(s)
    assert n.send(clip(SQUIRREL, T0)) == 1
    assert n.send(clip(SQUIRREL, T0 + MINUTE)) == 1
    s.species_cooldowns = [SpeciesCooldown(species="squirrel", minutes=45)]
    assert n.send(clip(SQUIRREL, T0 + 2 * MINUTE)) == 0, "the alert a minute ago counts already"


def test_the_times_survive_a_restart(posts, tmp_path):
    path = tmp_path / "logs" / "alert_times.json"
    s = settings(cooldown_minutes=60)
    assert PushoverNotifier(s, alert_times_path=path).send(clip(RABBIT, T0)) == 1
    on_disk = json.loads(path.read_text())
    assert on_disk == {"cameras": {"cam1": {"rabbit": T0}}}

    again = PushoverNotifier(s, alert_times_path=path)
    assert again.send(clip(RABBIT, T0 + 10 * MINUTE)) == 0
    assert len(posts) == 1


def test_a_damaged_file_costs_a_repeat_alert_never_the_alert(posts, tmp_path, caplog):
    path = tmp_path / "alert_times.json"
    path.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        n = PushoverNotifier(settings(cooldown_minutes=60), alert_times_path=path)
    assert "Could not read the alert times" in caplog.text
    assert n.send(clip(RABBIT, T0)) == 1
    assert json.loads(path.read_text())["cameras"]["cam1"]["rabbit"] == T0


def test_times_older_than_the_longest_cooldown_are_dropped_from_the_file(tmp_path):
    path = tmp_path / "alert_times.json"
    times = AlertTimes(path)
    times.claim("cam1", "deer", 0, 1_000.0)            # decades ago
    times.claim("cam1", "squirrel", 0, 4_000_000_000.0)
    assert json.loads(path.read_text()) == {"cameras": {"cam1": {"squirrel": 4_000_000_000.0}}}


# --------------------------------------------------------------------------
# The settings page saves them and they apply live
# --------------------------------------------------------------------------

BASE = {
    "general": {
        "storage_root": "/tmp/at/storage",
        "logs_root": "/tmp/at/logs",
        "notification": {"pushover_app_token_env": "PUSHOVER_APP_TOKEN",
                         "pushover_user_key_env": "PUSHOVER_USER_KEY"},
    },
    "cameras": [{"id": "cam1", "name": "door", "rtsp": {"uri": "rtsp://10.0.0.2/stream"}}],
}


class FakeWorker:
    def __init__(self, camera):
        self.camera = camera


def test_saved_cooldowns_are_written_and_apply_live(tmp_path):
    cfg_path = tmp_path / "config" / "cameras.yml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(BASE, sort_keys=False), encoding="utf-8")
    rt = configstore.validate(configstore.load_raw(cfg_path))
    workers = {cam.id: FakeWorker(cam) for cam in rt.cameras}
    running_settings = rt.general.notification

    result = configstore.save(cfg_path, {"general": {"notification": {
        "cooldown_minutes": 15,
        "species_cooldowns": [{"species": "squirrel", "minutes": 120}, {"species": "rabbit", "minutes": 60}],
    }}}, rt, workers)

    written = configstore.load_raw(cfg_path)["general"]["notification"]
    assert written["cooldown_minutes"] == 15
    assert written["species_cooldowns"] == [{"species": "squirrel", "minutes": 120},
                                            {"species": "rabbit", "minutes": 60}]
    assert rt.general.notification is running_settings, "the notifier's object is updated in place"
    assert running_settings.cooldown_minutes == 15
    assert [(c.species, c.minutes) for c in running_settings.species_cooldowns] == [("squirrel", 120), ("rabbit", 60)]
    assert {"notification.cooldown_minutes", "notification.species_cooldowns"} <= set(result["applied_live"])
    assert configstore.pending_restart(result["config"], rt, workers) == []


def test_a_file_without_cooldowns_is_not_padded_with_them():
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {
        "general": {"notification": {"cooldown_minutes": 0, "species_cooldowns": []}},
    })
    assert new == BASE and changes["general"] == []
