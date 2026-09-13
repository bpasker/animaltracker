"""The notification models: named Pushover destinations and a camera's pick
of them. These rules are what both the pipeline at startup and the settings
page on save enforce."""
import pytest
from pydantic import ValidationError

from animaltracker.config import CameraNotificationSettings, NotificationSettings, RuntimeConfig


def errors(exc):
    return [(tuple(e["loc"]), e["msg"]) for e in exc.value.errors()]


def test_the_fallback_variable_alone_is_one_implicit_destination():
    s = NotificationSettings(pushover_app_token_env="T", pushover_user_key_env=" PUSHOVER_USER_KEY ")
    assert s.pushover_user_key_env == "PUSHOVER_USER_KEY"
    assert s.destinations == []
    assert [(d.id, d.user_key_env) for d in s.resolved_destinations()] == [("default", "PUSHOVER_USER_KEY")]


def test_configured_destinations_replace_the_fallback_and_carry_labels():
    s = NotificationSettings(pushover_app_token_env="T", destinations=[
        {"id": " brandon ", "name": " Brandon ", "user_key_env": " KEY_B "},
        {"id": "jane", "name": "  ", "user_key_env": "KEY_J", "app_token_env": " "},
    ])
    assert s.pushover_user_key_env is None
    assert [d.id for d in s.resolved_destinations()] == ["brandon", "jane"]
    assert s.destinations[0].label == "Brandon" and s.destinations[0].user_key_env == "KEY_B"
    assert s.destinations[1].name is None and s.destinations[1].label == "jane"
    assert s.destinations[1].app_token_env is None


@pytest.mark.parametrize("bad, where, what", [
    ({}, (), "set pushover_user_key_env or define at least one destination"),
    ({"pushover_user_key_env": "   "}, (), "set pushover_user_key_env or define at least one destination"),
    ({"destinations": [{"id": "a b", "user_key_env": "X"}]}, ("destinations", 0, "id"), "letters, digits"),
    ({"destinations": [{"id": "a", "user_key_env": "1X"}]}, ("destinations", 0, "user_key_env"), "environment variable name"),
    ({"destinations": [{"id": "a", "user_key_env": "X", "app_token_env": "no-dash"}]}, ("destinations", 0, "app_token_env"), "environment variable name"),
    ({"destinations": [{"id": "a", "user_key_env": "X"}, {"id": "a", "user_key_env": "Y"}]}, ("destinations",), "appears twice"),
])
def test_bad_notification_blocks_are_refused_with_a_useful_path(bad, where, what):
    with pytest.raises(ValidationError) as info:
        NotificationSettings(pushover_app_token_env="T", **bad)
    locs = errors(info)
    assert any(loc == where and what in msg for loc, msg in locs), locs


def test_a_camera_pick_is_trimmed_and_absent_means_everyone():
    assert CameraNotificationSettings().destinations is None
    assert CameraNotificationSettings(destinations=[]).destinations == []
    assert CameraNotificationSettings(destinations=[" jane ", "brandon"]).destinations == ["jane", "brandon"]
    with pytest.raises(ValidationError):
        CameraNotificationSettings(destinations=["a", "a"])
    with pytest.raises(ValidationError):
        CameraNotificationSettings(destinations=["", "a"])


def test_a_whole_config_loads_with_destinations_and_a_camera_subset():
    cfg = RuntimeConfig.model_validate({
        "general": {
            "storage_root": "/s", "logs_root": "/l",
            "notification": {"pushover_app_token_env": "T", "destinations": [
                {"id": "brandon", "user_key_env": "KEY_B"},
                {"id": "jane", "user_key_env": "KEY_J"},
            ]},
        },
        "cameras": [
            {"id": "cam1", "name": "Door", "rtsp": {"uri": "rtsp://x"}, "notification": {"destinations": ["jane"]}},
            {"id": "cam2", "name": "Yard", "rtsp": {"uri": "rtsp://y"}},
        ],
    })
    assert cfg.cameras[0].notification.destinations == ["jane"]
    assert cfg.cameras[1].notification.destinations is None
    assert [d.id for d in cfg.general.notification.destinations] == ["brandon", "jane"]
