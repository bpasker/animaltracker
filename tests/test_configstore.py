"""Tests for the settings editor's config transaction (configstore.py) and
the /api/config handlers built on it.

These pin *intended* behaviour: a save never writes a file that will not
load, never drops keys it does not manage, always leaves a backup, and
tells live-applied fields apart from restart-only ones.
"""

import asyncio
import copy
import json
import os
import pathlib
import stat
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from animaltracker import configstore  # noqa: E402
from animaltracker.config import RuntimeConfig  # noqa: E402
from animaltracker.web import WebServer  # noqa: E402


BASE = {
    "general": {
        "storage_root": "/tmp/at/storage",
        "logs_root": "/tmp/at/logs",
        "metrics_port": 9500,
        "log_level": "debug",                       # not managed: must survive
        "clip": {"pre_seconds": 10, "post_seconds": 10, "post_analysis_frames": 120},
        "detector": {"realtime_backend": "megadetector", "postprocess_backend": "speciesnet",
                     "backend": "speciesnet", "country": "USA", "admin1_region": "MN"},
        "ebird": {"enabled": False, "region": "US-MN"},   # not managed: must survive
        "exclusion_list": [],
        "notification": {"pushover_app_token_env": "PUSHOVER_APP_TOKEN",
                         "pushover_user_key_env": "PUSHOVER_USER_KEY",
                         "web_base_url": "http://192.168.1.195:8080/"},
        "retention": {"min_days": 7, "max_days": 120, "max_utilization_pct": 80},
    },
    "cameras": [
        {
            "id": "cam1", "name": "door", "location": "Back Yard", "detect_enabled": True,
            "rtsp": {"uri": "rtsps://192.168.1.1:7441/abc?enableSrtp", "transport": "tcp",
                     "hwaccel": True, "latency_ms": 200, "frame_skip": 3},
            "thresholds": {"confidence": 0.45, "min_frames": 5, "min_duration": 1.0,
                           "min_detection_area": 0.001, "generic_confidence": 0.4},
            "ptz_tracking": {"enabled": False, "self_track": False},
            "inference_max_width": 1280,
            "include_species": [], "exclude_species": ["mammalia rodentia sciuridae"],
            "notification": {"priority": 0, "sound": "pushover"},
        }
    ],
}


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "config" / "cameras.yml"
    p.parent.mkdir(parents=True)
    p.write_text("# hand comment\n" + yaml.safe_dump(BASE, sort_keys=False), encoding="utf-8")
    return p


class FakeWorker:
    def __init__(self, camera):
        self.camera = camera
        self.latest_frame = object()
        import time
        self.latest_frame_ts = time.time()
        self.stream_connected = True
        self.onvif_client = None
        self.onvif_profile_token = None
        self.ptz_tracker = None


def running(cfg_path):
    """A runtime + workers built from the file, the way the pipeline does."""
    rt = configstore.validate(configstore.load_raw(cfg_path))
    workers = {cam.id: FakeWorker(cam) for cam in rt.cameras}
    return rt, workers


# --------------------------------------------------------------------------
# merge_payload
# --------------------------------------------------------------------------

def test_merge_preserves_unmanaged_keys_and_reports_changes():
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {
        "general": {"clip": {"pre_seconds": 15}, "detector": {"country": "USA"}},
        "cameras": [{"id": "cam1", "thresholds": {"confidence": 0.6}}],
    })
    assert new["general"]["log_level"] == "debug"
    assert new["general"]["ebird"] == {"enabled": False, "region": "US-MN"}
    assert new["general"]["clip"]["post_analysis_frames"] == 120
    assert new["general"]["clip"]["pre_seconds"] == 15
    assert new["cameras"][0]["thresholds"]["min_detection_area"] == 0.001, "untouched sibling survives"
    assert new["cameras"][0]["thresholds"]["confidence"] == 0.6
    assert changes["general"] == ["clip.pre_seconds"]           # country was unchanged
    assert changes["cameras"] == {"cam1": ["thresholds.confidence"]}
    assert changes["added"] == [] and changes["removed"] == []


def test_merge_adds_and_removes_cameras_by_id():
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {
        "cameras": [
            {"id": "cam1"},
            {"id": "cam2", "name": "Creek", "rtsp": {"uri": "rtsp://x/y", "transport": "udp"},
             "onvif": {"host": "10.0.0.2", "port": 8000, "username_env": "CAM2_ONVIF_USER",
                       "password_env": "CAM2_ONVIF_PASS"}},
        ],
    })
    assert [c["id"] for c in new["cameras"]] == ["cam1", "cam2"]
    assert changes["added"] == ["cam2"]
    assert list(new["cameras"][1].keys())[:3] == ["id", "name", "rtsp"], "new block reads like the sample"

    new2, changes2 = configstore.merge_payload(new, {"cameras": [{"id": "cam2"}]})
    assert [c["id"] for c in new2["cameras"]] == ["cam2"]
    assert changes2["removed"] == ["cam1"]


def test_merge_refuses_empty_camera_set_duplicates_and_bad_ids():
    with pytest.raises(configstore.ConfigError):
        configstore.merge_payload(copy.deepcopy(BASE), {"cameras": []})
    with pytest.raises(configstore.ConfigError):
        configstore.merge_payload(copy.deepcopy(BASE), {"cameras": [{"id": "cam1"}, {"id": "cam1"}]})
    with pytest.raises(configstore.ConfigError) as info:
        configstore.merge_payload(copy.deepcopy(BASE), {"cameras": [{"id": "cam1"}, {"id": "bad id/"}]})
    assert "bad id/" in str(info.value)


def test_merge_none_removes_onvif_block_and_rounds_floats():
    base = copy.deepcopy(BASE)
    base["cameras"][0]["onvif"] = {"host": "10.0.0.1", "port": 80, "username_env": "U", "password_env": "P"}
    new, changes = configstore.merge_payload(base, {
        "cameras": [{"id": "cam1", "onvif": None, "thresholds": {"confidence": 0.30000000000000004}}],
    })
    assert "onvif" not in new["cameras"][0]
    assert new["cameras"][0]["thresholds"]["confidence"] == 0.3
    assert changes["cameras"]["cam1"] == ["onvif", "thresholds.confidence"]


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def test_validate_reports_field_paths_by_camera_id():
    raw = copy.deepcopy(BASE)
    raw["cameras"][0]["rtsp"]["hwaccel"] = "nvdec"       # the old UI wrote this; it never loaded
    raw["cameras"][0]["thresholds"]["confidence"] = 7
    with pytest.raises(configstore.ConfigError) as info:
        configstore.validate(raw)
    paths = {p["path"] for p in info.value.problems}
    assert "cameras.cam1.rtsp.hwaccel" in paths
    assert "cameras.cam1.thresholds.confidence" in paths


# --------------------------------------------------------------------------
# write_config
# --------------------------------------------------------------------------

def test_write_is_atomic_backed_up_and_keeps_mode(cfg_path):
    os.chmod(cfg_path, 0o640)
    raw = configstore.load_raw(cfg_path)
    raw["general"]["clip"]["pre_seconds"] = 12
    backup = configstore.write_config(cfg_path, raw)

    assert backup is not None and backup.parent == cfg_path.parent / "backups"
    assert "# hand comment" in backup.read_text(), "the backup is the file as it was"
    assert not cfg_path.with_name("cameras.yml.tmp").exists()
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o640
    reloaded = configstore.load_raw(cfg_path)
    assert reloaded["general"]["clip"]["pre_seconds"] == 12
    assert reloaded["general"]["ebird"]["region"] == "US-MN"
    assert cfg_path.read_text().startswith("# cameras.yml")


def test_backups_are_rotated(cfg_path):
    for i in range(5):
        raw = configstore.load_raw(cfg_path)
        raw["general"]["clip"]["pre_seconds"] = i
        configstore.write_config(cfg_path, raw, keep=3)
    backups = sorted((cfg_path.parent / "backups").glob("cameras.yml.*"))
    assert len(backups) == 3


# --------------------------------------------------------------------------
# hot_apply / pending_restart
# --------------------------------------------------------------------------

def test_live_fields_apply_and_restart_fields_wait(cfg_path):
    rt, workers = running(cfg_path)
    result = configstore.save(cfg_path, {
        "general": {"clip": {"pre_seconds": 20, "max_concurrent_postprocess": 4},
                    "detector": {"realtime_backend": "yolo"}},
        "cameras": [{"id": "cam1", "name": "front door",
                     "thresholds": {"confidence": 0.7},
                     "rtsp": {"uri": "rtsp://new/stream"}}],
    }, rt, workers)

    assert rt.general.clip.pre_seconds == 20
    assert workers["cam1"].camera.name == "front door"
    assert workers["cam1"].camera.thresholds.confidence == 0.7
    assert rt.general.clip.max_concurrent_postprocess == 2, "restart-only: running copy untouched"
    assert rt.general.detector.realtime_backend == "megadetector"
    assert workers["cam1"].camera.rtsp.uri.startswith("rtsps://192.168.1.1"), "restart-only"
    assert set(result["applied_live"]) == {"clip.pre_seconds", "cam1.name", "cam1.thresholds.confidence"}

    reasons = configstore.pending_restart(result["config"], rt, workers)
    joined = " ".join(reasons)
    assert "stream settings changed" in joined
    assert "real-time detector" in joined
    assert "post-processing concurrency" in joined
    assert "Camera" in joined and "cam1" in joined

    file_cfg = configstore.validate(configstore.load_raw(cfg_path))
    assert file_cfg.general.clip.max_concurrent_postprocess == 4
    assert file_cfg.cameras[0].rtsp.uri == "rtsp://new/stream"


def test_added_and_removed_cameras_are_pending_until_restart(cfg_path):
    rt, workers = running(cfg_path)
    result = configstore.save(cfg_path, {
        "cameras": [{"id": "cam1"},
                    {"id": "cam2", "name": "Creek", "rtsp": {"uri": "rtsp://x/y"}}],
    }, rt, workers)
    reasons = configstore.pending_restart(result["config"], rt, workers)
    assert any("cam2" in r and "not running" in r for r in reasons)

    result = configstore.save(cfg_path, {"cameras": [{"id": "cam2"}]}, rt, workers)
    reasons = configstore.pending_restart(result["config"], rt, workers)
    assert any("cam1" in r and "removed" in r for r in reasons)


def test_nothing_pending_right_after_startup(cfg_path):
    rt, workers = running(cfg_path)
    cfg = configstore.validate(configstore.load_raw(cfg_path))
    assert configstore.pending_restart(cfg, rt, workers) == []


def test_invalid_payload_leaves_file_untouched(cfg_path):
    rt, workers = running(cfg_path)
    before = cfg_path.read_text()
    with pytest.raises(configstore.ConfigError) as info:
        configstore.save(cfg_path, {"cameras": [{"id": "cam1", "thresholds": {"confidence": 3}}]}, rt, workers)
    assert info.value.problems[0]["path"] == "cameras.cam1.thresholds.confidence"
    assert cfg_path.read_text() == before
    assert not (cfg_path.parent / "backups").exists()
    assert workers["cam1"].camera.thresholds.confidence == 0.45


def test_unchanged_payload_does_not_rewrite(cfg_path):
    rt, workers = running(cfg_path)
    before = cfg_path.read_text()
    result = configstore.save(cfg_path, {"cameras": [{"id": "cam1", "name": "door"}]}, rt, workers)
    assert result["written"] is False and result["backup"] is None
    assert cfg_path.read_text() == before


def test_ptz_state_sync_only_for_changed_keys(cfg_path):
    rt, workers = running(cfg_path)
    result = configstore.save(cfg_path, {
        "cameras": [{"id": "cam1", "ptz_tracking": {"enabled": True, "patrol_return_delay": 9.0,
                                                    "target_fill_pct": 0.5}}],
    }, rt, workers)
    assert result["ptz_state"] == {"cam1": {"patrol_return_delay": 9.0}}


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------

def test_describe_reads_file_and_annotates_runtime(cfg_path, monkeypatch):
    rt, workers = running(cfg_path)
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "x")
    monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    payload = configstore.describe(cfg_path, rt, workers, lambda cid: {"running": True, "state": "live"})
    cam = payload["cameras"][0]
    assert cam["id"] == "cam1" and cam["runtime"]["state"] == "live"
    assert cam["thresholds"]["blur_threshold"] == 50.0, "schema defaults are filled in"
    assert cam["rtsp"]["hwaccel"] is True
    assert payload["general"]["clip"]["max_event_seconds"] == 300.0
    assert payload["env"] == {"PUSHOVER_APP_TOKEN": True, "PUSHOVER_USER_KEY": False}
    assert payload["restart"] == {"required": False, "reasons": [], "supported": False, "unit": None}
    assert "thresholds" in payload["defaults"]["camera"]
    assert "uri" not in payload["defaults"]["camera"]["rtsp"]


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, body=None, query=None):
        self._body = body
        self.query = query or {}
        self.match_info = {}

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def call(handler, body=None):
    resp = asyncio.run(handler(FakeRequest(body)))
    return resp.status, json.loads(resp.body.decode())


def make_server(cfg_path, tmp_path):
    rt, workers = running(cfg_path)
    srv = WebServer(workers, tmp_path / "storage", tmp_path / "logs", port=0,
                    config_path=cfg_path, runtime=rt)
    return srv


def test_get_config_handler(cfg_path, tmp_path, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    srv = make_server(cfg_path, tmp_path)
    status, body = call(srv.handle_get_config)
    assert status == 200
    assert body["cameras"][0]["runtime"]["running"] is True
    assert body["cameras"][0]["runtime"]["state"] == "live"
    assert body["config_path"] == str(cfg_path)


def test_save_handler_round_trip_and_validation(cfg_path, tmp_path, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    srv = make_server(cfg_path, tmp_path)

    status, body = call(srv.handle_save_config, {"cameras": [{"id": "cam1", "thresholds": {"confidence": 5}}]})
    assert status == 400
    assert body["problems"][0]["path"] == "cameras.cam1.thresholds.confidence"

    status, body = call(srv.handle_save_config, {
        "general": {"retention": {"max_days": 90}},
        "cameras": [{"id": "cam1"}, {"id": "cam2", "name": "Creek", "rtsp": {"uri": "rtsp://x/y"}}],
    })
    assert status == 200 and body["status"] == "ok"
    assert body["written"] is True and body["backup"]
    assert body["changes"]["added"] == ["cam2"]
    assert body["restart"]["required"] is True
    assert any("cam2" in r for r in body["restart"]["reasons"])
    assert srv.runtime.general.retention.max_days == 90

    status, body = call(srv.handle_get_config)
    assert [c["id"] for c in body["cameras"]] == ["cam1", "cam2"]
    assert body["cameras"][1]["runtime"] == {"running": False}

    status, body = call(srv.handle_save_config, ValueError("bad json"))
    assert status == 400


def test_restart_handler_refuses_outside_systemd(cfg_path, tmp_path, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    srv = make_server(cfg_path, tmp_path)
    status, body = call(srv.handle_restart)
    assert status == 501 and body["supported"] is False


def test_probe_handler_reports_missing_env_without_touching_network(cfg_path, tmp_path, monkeypatch):
    monkeypatch.delenv("CAMX_ONVIF_USER", raising=False)
    monkeypatch.delenv("CAMX_ONVIF_PASS", raising=False)
    srv = make_server(cfg_path, tmp_path)
    status, body = call(srv.handle_probe_camera, {
        "onvif": {"host": "10.0.0.9", "port": 80, "username_env": "CAMX_ONVIF_USER",
                  "password_env": "CAMX_ONVIF_PASS"},
        "env": ["CAMX_ONVIF_USER"],
    })
    assert status == 200
    assert body["onvif"]["ok"] is False and "not set" in body["onvif"]["error"]
    assert body["env"] == {"CAMX_ONVIF_USER": False}


def test_probe_rtsp_rejects_empty_uri():
    out = configstore.probe_rtsp("", "tcp")
    assert out["ok"] is False


def test_camera_id_pattern_matches_existing_style():
    import re
    for good in ("cam1", "front-door", "Barn_2"):
        assert re.match(configstore.CAMERA_ID_PATTERN, good)
    for bad in ("", "a b", "x/y", "../etc"):
        assert not re.match(configstore.CAMERA_ID_PATTERN, bad)


# --------------------------------------------------------------------------
# defaults stay implicit
# --------------------------------------------------------------------------

def test_saving_every_managed_key_does_not_pad_the_file_with_defaults():
    """The UI sends every managed key; a hand-kept file must not grow forty
    default lines per camera, and untouched defaults are not 'changes'."""
    from animaltracker.config import CameraConfig, GeneralSettings
    cam = configstore._dump(CameraConfig.model_validate(BASE["cameras"][0]))
    cam["onvif"] = None
    gen = configstore._dump(GeneralSettings.model_validate(BASE["general"]))
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {"general": gen, "cameras": [cam]})
    assert changes["general"] == []
    assert changes["cameras"] == {}
    assert new["cameras"][0]["ptz_tracking"] == {"enabled": False, "self_track": False}
    assert "blur_threshold" not in new["cameras"][0]["thresholds"]
    assert "spatial_merge_reach" not in new["general"]["clip"]
    assert new == BASE


def test_non_default_values_for_absent_keys_are_written():
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {
        "cameras": [{"id": "cam1", "thresholds": {"blur_threshold": 75.0},
                     "ptz_tracking": {"target_camera_id": None, "smoothing": 0.5}}],
    })
    assert new["cameras"][0]["thresholds"]["blur_threshold"] == 75.0
    assert new["cameras"][0]["ptz_tracking"] == {"enabled": False, "self_track": False, "smoothing": 0.5}
    assert changes["cameras"]["cam1"] == ["thresholds.blur_threshold", "ptz_tracking"]


def test_explicit_null_inside_a_block_survives_for_present_keys():
    base = copy.deepcopy(BASE)
    base["cameras"][0]["ptz_tracking"]["zoom_fov_calibration_path"] = "config/zoom.json"
    new, _ = configstore.merge_payload(base, {
        "cameras": [{"id": "cam1", "ptz_tracking": {"zoom_fov_calibration_path": None}}],
    })
    assert "zoom_fov_calibration_path" in new["cameras"][0]["ptz_tracking"]
    assert new["cameras"][0]["ptz_tracking"]["zoom_fov_calibration_path"] is None
    cfg = configstore.validate(new)
    assert cfg.cameras[0].ptz_tracking.zoom_fov_calibration_path is None


# --------------------------------------------------------------------------
# Pushover destinations
# --------------------------------------------------------------------------

DESTINATIONS = [
    {"id": "brandon", "name": "Brandon", "user_key_env": "PUSHOVER_USER_KEY", "app_token_env": None},
    {"id": "jane", "name": "", "user_key_env": "PUSHOVER_USER_KEY_JANE", "app_token_env": "PUSHOVER_APP_TOKEN_JANE"},
]


def test_destinations_and_camera_picks_round_trip_and_apply_live(cfg_path, monkeypatch):
    rt, workers = running(cfg_path)
    result = configstore.save(cfg_path, {
        "general": {"notification": {"destinations": DESTINATIONS, "pushover_user_key_env": None}},
        "cameras": [{"id": "cam1", "notification": {"destinations": ["jane"]}}],
    }, rt, workers)

    written = configstore.load_raw(cfg_path)["general"]["notification"]
    assert written["destinations"] == [
        {"id": "brandon", "name": "Brandon", "user_key_env": "PUSHOVER_USER_KEY"},
        {"id": "jane", "user_key_env": "PUSHOVER_USER_KEY_JANE", "app_token_env": "PUSHOVER_APP_TOKEN_JANE"},
    ], "entries are written whole, blank names and tokens dropped"
    assert "pushover_user_key_env" not in written, "a cleared fallback leaves the file"
    assert configstore.load_raw(cfg_path)["cameras"][0]["notification"]["destinations"] == ["jane"]

    assert [d.id for d in rt.general.notification.destinations] == ["brandon", "jane"], "applied live"
    assert rt.general.notification.pushover_user_key_env is None
    assert workers["cam1"].camera.notification.destinations == ["jane"]
    assert {"notification.destinations", "notification.pushover_user_key_env",
            "cam1.notification.destinations"} <= set(result["applied_live"])
    assert configstore.pending_restart(result["config"], rt, workers) == []

    monkeypatch.setenv("PUSHOVER_USER_KEY_JANE", "k")
    monkeypatch.delenv("PUSHOVER_APP_TOKEN_JANE", raising=False)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    described = configstore.describe(cfg_path, rt, workers)
    assert described["env"]["PUSHOVER_USER_KEY_JANE"] is True
    assert described["env"]["PUSHOVER_APP_TOKEN_JANE"] is False
    assert described["general"]["notification"]["destinations"][1]["name"] is None
    assert described["cameras"][0]["notification"]["destinations"] == ["jane"]

    # Back to "everyone": None removes the key, [] is kept as an explicit none.
    configstore.save(cfg_path, {"cameras": [{"id": "cam1", "notification": {"destinations": None}}]}, rt, workers)
    assert "destinations" not in configstore.load_raw(cfg_path)["cameras"][0]["notification"]
    assert workers["cam1"].camera.notification.destinations is None
    configstore.save(cfg_path, {"cameras": [{"id": "cam1", "notification": {"destinations": []}}]}, rt, workers)
    assert configstore.load_raw(cfg_path)["cameras"][0]["notification"]["destinations"] == []
    assert workers["cam1"].camera.notification.destinations == []


def test_a_camera_naming_an_unknown_destination_is_refused_on_its_own_field(cfg_path):
    rt, workers = running(cfg_path)
    before = cfg_path.read_text()
    with pytest.raises(configstore.ConfigError) as info:
        configstore.save(cfg_path, {"cameras": [{"id": "cam1", "notification": {"destinations": ["ghost"]}}]}, rt, workers)
    assert info.value.problems == [{
        "path": "cameras.cam1.notification.destinations",
        "message": "Unknown destination 'ghost'. No destinations are defined under General → Notifications.",
    }]
    assert cfg_path.read_text() == before

    configstore.save(cfg_path, {"general": {"notification": {"destinations": DESTINATIONS[:1]}}}, rt, workers)
    with pytest.raises(configstore.ConfigError) as info2:
        configstore.save(cfg_path, {"cameras": [{"id": "cam1", "notification": {"destinations": ["brandon", "ghost"]}}]}, rt, workers)
    assert "Define it under General" in info2.value.problems[0]["message"]


def test_bad_destination_entries_are_refused_with_their_index_in_the_path(cfg_path):
    rt, workers = running(cfg_path)
    with pytest.raises(configstore.ConfigError) as dup:
        configstore.save(cfg_path, {"general": {"notification": {"destinations": [
            {"id": "a", "user_key_env": "X"}, {"id": "a", "user_key_env": "Y"}]}}}, rt, workers)
    assert dup.value.problems[0]["path"] == "general.notification.destinations"
    with pytest.raises(configstore.ConfigError) as bad:
        configstore.save(cfg_path, {"general": {"notification": {"destinations": [
            {"id": "ok", "user_key_env": "X"}, {"id": "bad id", "user_key_env": "1Y"}]}}}, rt, workers)
    paths = {p["path"] for p in bad.value.problems}
    assert paths == {"general.notification.destinations.1.id", "general.notification.destinations.1.user_key_env"}
    with pytest.raises(configstore.ConfigError):
        configstore.merge_payload(copy.deepcopy(BASE), {"general": {"notification": {"destinations": "brandon"}}})


def test_clearing_the_fallback_without_destinations_is_refused(cfg_path):
    rt, workers = running(cfg_path)
    with pytest.raises(configstore.ConfigError) as info:
        configstore.save(cfg_path, {"general": {"notification": {"destinations": [], "pushover_user_key_env": None}}}, rt, workers)
    assert "at least one destination" in info.value.problems[0]["message"]
    assert configstore.load_raw(cfg_path)["general"]["notification"]["pushover_user_key_env"] == "PUSHOVER_USER_KEY"


def test_pushover_variable_names_apply_live_now(cfg_path):
    """The notifier reads the settings object on every send, so renaming a
    variable no longer waits for a restart (the variable itself still has
    to be in the environment, which the env tag shows)."""
    rt, workers = running(cfg_path)
    result = configstore.save(cfg_path, {"general": {"notification": {"pushover_app_token_env": "PO_TOKEN"}}}, rt, workers)
    assert rt.general.notification.pushover_app_token_env == "PO_TOKEN"
    assert "notification.pushover_app_token_env" in result["applied_live"]
    assert configstore.pending_restart(result["config"], rt, workers) == []


def test_an_empty_destination_list_is_not_written_for_a_file_that_lacks_it():
    new, changes = configstore.merge_payload(copy.deepcopy(BASE), {
        "general": {"notification": {"destinations": [], "pushover_user_key_env": "PUSHOVER_USER_KEY"}},
        "cameras": [{"id": "cam1", "notification": {"destinations": None}}],
    })
    assert new == BASE and changes["general"] == [] and changes["cameras"] == {}
