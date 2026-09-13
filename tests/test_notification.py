"""PushoverNotifier routes each alert to named destinations.

A destination is an environment variable holding a Pushover user or group
key (optionally with its own application token). A camera passes the ids
it wants: None means every destination (or the fallback variable while
none are defined), [] means none. Nothing here touches the network: the
HTTP post is recorded by a fake."""
import logging

import pytest

from animaltracker import notification
from animaltracker.config import NotificationSettings
from animaltracker.notification import NotificationContext, PushoverNotifier


class FakeResponse:
    def raise_for_status(self):
        pass


@pytest.fixture
def posts(monkeypatch):
    sent = []

    def fake_post(url, data=None, files=None, timeout=None):
        sent.append({"user": data["user"], "token": data["token"], "title": data["title"], "files": files})
        return FakeResponse()

    monkeypatch.setattr(notification.requests, "post", fake_post)
    return sent


@pytest.fixture
def env(monkeypatch):
    for name in ("PUSHOVER_APP_TOKEN", "PUSHOVER_USER_KEY", "KEY_BRANDON", "KEY_JANE", "TOKEN_JANE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "legacy-1, legacy-2")
    monkeypatch.setenv("KEY_BRANDON", "kb")
    monkeypatch.setenv("KEY_JANE", "kj")
    return monkeypatch


def ctx(camera_id="cam1"):
    return NotificationContext(
        species="mammalia_carnivora_canidae", confidence=0.9, camera_id=camera_id, camera_name="Yard",
        clip_path="/srv/clips/cam1/2026/09/12/1.mp4", event_started_at=0.0, event_duration=3.0,
    )


LEGACY = dict(pushover_app_token_env="PUSHOVER_APP_TOKEN", pushover_user_key_env="PUSHOVER_USER_KEY")
MULTI = dict(pushover_app_token_env="PUSHOVER_APP_TOKEN", destinations=[
    {"id": "brandon", "name": "Brandon", "user_key_env": "KEY_BRANDON"},
    {"id": "jane", "user_key_env": "KEY_JANE"},
])


def users(posts):
    return [p["user"] for p in posts]


def test_legacy_variable_still_reaches_every_comma_separated_key(posts, env):
    n = PushoverNotifier(NotificationSettings(**LEGACY)).send(ctx())
    assert n == 2 and users(posts) == ["legacy-1", "legacy-2"]
    assert {p["token"] for p in posts} == {"app-token"}


def test_no_list_means_every_destination(posts, env):
    n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx(), destinations=None)
    assert n == 2 and users(posts) == ["kb", "kj"]
    assert posts[0]["title"].endswith(" detected @ Yard")


def test_a_camera_can_pick_a_subset_in_its_own_order(posts, env):
    n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx(), destinations=["jane"])
    assert n == 1 and users(posts) == ["kj"]


def test_an_empty_list_sends_nothing(posts, env, caplog):
    with caplog.at_level(logging.INFO):
        n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx(), destinations=[])
    assert n == 0 and posts == []
    assert "No Pushover destination for camera cam1" in caplog.text


def test_unknown_ids_are_skipped_with_a_warning_not_an_error(posts, env, caplog):
    with caplog.at_level(logging.WARNING):
        n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx("cam2"), destinations=["ghost", "jane", "jane"])
    assert n == 1 and users(posts) == ["kj"]
    assert "cam2 names Pushover destination(s) ghost" in caplog.text


def test_selecting_ids_while_no_destinations_exist_sends_nothing(posts, env):
    """The legacy variable is only the *default*; an explicit pick of ids
    that are not configured must not silently fall back to it."""
    n = PushoverNotifier(NotificationSettings(**LEGACY)).send(ctx(), destinations=["brandon"])
    assert n == 0 and posts == []


def test_a_destination_with_an_unset_variable_does_not_block_the_others(posts, env, caplog):
    env.delenv("KEY_JANE")
    with caplog.at_level(logging.ERROR):
        n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx())
    assert n == 1 and users(posts) == ["kb"]
    assert "reads KEY_JANE, which is not set" in caplog.text


def test_two_destinations_sharing_a_key_get_one_alert(posts, env):
    env.setenv("KEY_JANE", "kb")
    n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx())
    assert n == 1 and users(posts) == ["kb"]


def test_a_destination_can_send_with_its_own_app_token(posts, env):
    env.setenv("TOKEN_JANE", "jane-app")
    settings = NotificationSettings(pushover_app_token_env="PUSHOVER_APP_TOKEN", destinations=[
        {"id": "brandon", "user_key_env": "KEY_BRANDON"},
        {"id": "jane", "user_key_env": "KEY_JANE", "app_token_env": "TOKEN_JANE"},
    ])
    n = PushoverNotifier(settings).send(ctx())
    assert n == 2
    assert [(p["user"], p["token"]) for p in posts] == [("kb", "app-token"), ("kj", "jane-app")]


def test_a_missing_app_token_skips_only_the_destinations_that_need_it(posts, env, caplog):
    env.delenv("PUSHOVER_APP_TOKEN")
    env.setenv("TOKEN_JANE", "jane-app")
    settings = NotificationSettings(pushover_app_token_env="PUSHOVER_APP_TOKEN", destinations=[
        {"id": "brandon", "user_key_env": "KEY_BRANDON"},
        {"id": "jane", "user_key_env": "KEY_JANE", "app_token_env": "TOKEN_JANE"},
    ])
    with caplog.at_level(logging.ERROR):
        n = PushoverNotifier(settings).send(ctx())
    assert n == 1 and users(posts) == ["kj"]
    assert "needs the app token in PUSHOVER_APP_TOKEN, which is not set" in caplog.text


def test_settings_edits_apply_without_rebuilding_the_notifier(posts, env):
    """The pipeline hands the notifier the live settings object, and the
    settings page assigns new destinations onto it."""
    settings = NotificationSettings(**LEGACY)
    notifier = PushoverNotifier(settings)
    assert notifier.send(ctx()) == 2
    posts.clear()
    settings.destinations = NotificationSettings(**MULTI).destinations
    assert notifier.send(ctx(), destinations=["brandon"]) == 1 and users(posts) == ["kb"]


def test_the_thumbnail_is_read_once_and_attached_to_every_post(posts, env, tmp_path):
    thumb = tmp_path / "t.jpg"
    thumb.write_bytes(b"\xff\xd8\xff")
    c = ctx()
    c.thumbnail_path = str(thumb)
    PushoverNotifier(NotificationSettings(**MULTI)).send(c)
    assert len(posts) == 2
    assert posts[0]["files"] is posts[1]["files"]
    assert posts[0]["files"]["attachment"][0] == "t.jpg"


def test_a_failed_post_is_logged_and_the_rest_still_go_out(env, monkeypatch, caplog):
    calls = []

    class Boom(FakeResponse):
        def raise_for_status(self):
            raise notification.requests.RequestException("500")

    def flaky(url, data=None, files=None, timeout=None):
        calls.append(data["user"])
        return Boom() if data["user"] == "kb" else FakeResponse()

    monkeypatch.setattr(notification.requests, "post", flaky)
    with caplog.at_level(logging.ERROR):
        n = PushoverNotifier(NotificationSettings(**MULTI)).send(ctx())
    assert calls == ["kb", "kj"] and n == 1
    assert "Failed to send Pushover alert to 'Brandon'" in caplog.text
