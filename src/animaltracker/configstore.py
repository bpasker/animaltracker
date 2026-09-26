"""Read, validate and rewrite ``config/cameras.yml`` for the settings page.

The settings screen in the client-side app (``/app/settings``) is the only
place an operator edits the configuration without a shell, so this module
is written to make that safe:

* **The file is the source of truth.** ``describe()`` reads and validates
  the file through the same pydantic models the CLI loads at startup, so
  the UI edits exactly what the next start will see, with every default
  filled in. Runtime state (is this camera running, is its ONVIF client up)
  is *annotated* onto that picture, never substituted for it.
* **Unmanaged keys survive.** A save deep-merges the managed keys into the
  parsed file rather than dumping the models, so hand-added blocks such as
  ``ebird:`` or ``log_level:`` pass through untouched. Comments do not:
  PyYAML cannot round-trip them, which is why every write keeps a backup.
* **Nothing invalid reaches disk.** The merged document is validated
  before the write. A payload that would not load at the next restart is
  refused with the field paths pydantic complained about.
* **Writes are atomic and backed up.** ``write_config()`` copies the old
  file to ``config/backups/`` (last :data:`BACKUP_KEEP` kept), writes a
  temp file next to the target and ``os.replace``s it, preserving the
  original owner and mode so a root-run service does not lock the
  operator out of their own file.
* **Live and restart-only fields are told apart.** Fields the pipeline
  reads on use are pushed into the running process; the rest are reported
  as pending until a restart. "Pending" is computed by diffing the file
  against the running config, not remembered from earlier saves, so it is
  right even after a hand edit.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from .config import ENV_NAME_PATTERN, CameraConfig, RuntimeConfig

LOGGER = logging.getLogger(__name__)

BACKUP_DIR_NAME = "backups"
BACKUP_KEEP = 20
CAMERA_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"
SECRETS_FILE_NAME = "secrets.env"      # beside cameras.yml; systemd's EnvironmentFile
SECRET_VALUE_MAX_LEN = 512
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
# Variables only this application reads: they may be set before the saved
# configuration names them, so a key can be pasted while adding a destination.
_FREELY_SETTABLE_RE = re.compile(r"^PUSHOVER_[A-Z0-9_]+$")
_ENV_PLAIN_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@+=,%-]*$")

_MISSING = object()
_WRITE_LOCK = threading.Lock()


class ConfigError(Exception):
    """A payload or file the settings page must not accept.

    ``problems`` is a list of ``{"path": ..., "message": ...}`` dicts the UI
    can attach to individual fields.
    """

    def __init__(self, message: str, problems: Optional[List[Dict[str, str]]] = None) -> None:
        super().__init__(message)
        self.problems = problems or []


# ---------------------------------------------------------------------------
# Field inventory
#
# Dotted key -> True when the running process reads the value on use (it can
# be applied live), False when it is consumed once at startup (a restart is
# needed). Anything not listed is not managed by the UI and passes through
# the file untouched.
# ---------------------------------------------------------------------------

GENERAL_FIELDS: Dict[str, bool] = {
    "storage_root": False,
    "logs_root": False,
    "metrics_port": False,
    "timezone": False,
    "detector.realtime_backend": False,
    "detector.postprocess_backend": False,
    "detector.model_path": False,
    "detector.speciesnet_version": False,
    "detector.country": False,
    "detector.admin1_region": False,
    "detector.latitude": False,
    "detector.longitude": False,
    "clip.pre_seconds": True,
    "clip.post_seconds": True,
    "clip.max_event_seconds": True,
    "clip.max_concurrent_postprocess": False,
    "clip.thumbnail_cropped": True,
    "clip.post_analysis": True,
    "clip.unified_post_processing": True,
    "clip.recover_unfinished_clips": True,   # the sweeper reads it before every pass
    "clip.post_analysis_frames": True,
    "clip.post_analysis_confidence": True,
    "clip.post_analysis_generic_confidence": True,
    "clip.delete_if_no_animal": True,
    "clip.min_detection_frames": True,
    "clip.min_reptile_detection_frames": True,
    "clip.sample_rate": True,
    "clip.tracking_enabled": True,
    "clip.track_merge_gap": True,
    "clip.spatial_merge_enabled": True,
    "clip.spatial_merge_iou": True,
    "clip.spatial_merge_reach": True,
    "clip.hierarchical_merge_enabled": True,
    "clip.single_animal_mode": True,
    "retention.min_days": True,
    "retention.max_days": True,
    "retention.max_utilization_pct": False,
    "notification.pushover_app_token_env": True,
    "notification.pushover_user_key_env": True,
    "notification.destinations": True,    # the whole list; the notifier reads it on every send
    "notification.web_base_url": True,
    "notification.cooldown_minutes": True,
    "notification.species_cooldowns": True,   # the whole list; read on every send too
    "exclusion_list": True,
}

CAMERA_FIELDS: Dict[str, bool] = {
    "name": True,
    "location": True,
    "detect_enabled": True,
    "inference_max_width": True,
    "rtsp.uri": False,
    "rtsp.transport": False,
    "rtsp.latency_ms": False,
    "rtsp.frame_skip": False,
    "rtsp.hwaccel": False,
    "onvif": False,           # the whole block; None removes it
    "thresholds.confidence": True,
    "thresholds.generic_confidence": True,
    "thresholds.min_frames": True,
    "thresholds.min_duration": True,
    "thresholds.min_detection_area": True,
    "thresholds.tracking_min_detection_area": True,
    "thresholds.blur_threshold": True,
    "thresholds.ptz_settle_time": True,
    "thresholds.motion_gate": True,
    "ptz_tracking": False,    # the whole block; the tracker is built at startup
    "include_species": True,
    "exclude_species": True,
    "notification.priority": True,
    "notification.sound": True,
    "notification.destinations": True,    # ids from general.notification.destinations; None = all
}

# Preferred key order for a camera block written from scratch, so a new
# camera reads like the sample file rather than like JSON.
_CAMERA_KEY_ORDER = [
    "id", "name", "location", "detect_enabled", "rtsp", "onvif", "thresholds",
    "ptz_tracking", "inference_max_width", "include_species", "exclude_species",
    "notification",
]

# What a restart-only general field is called in the pending-restart list.
_RESTART_GENERAL_WORDS = {
    "storage_root": "the storage path",
    "logs_root": "the logs path",
    "metrics_port": "the metrics port",
    "timezone": "the timezone",
    "detector.realtime_backend": "the real-time detector",
    "detector.postprocess_backend": "the post-processing detector",
    "detector.model_path": "the YOLO model path",
    "detector.speciesnet_version": "the SpeciesNet version",
    "detector.country": "the location priors",
    "detector.admin1_region": "the location priors",
    "detector.latitude": "the location priors",
    "detector.longitude": "the location priors",
    "clip.max_concurrent_postprocess": "the post-processing concurrency",
    "retention.max_utilization_pct": "the disk usage ceiling",
}

_RESTART_CAMERA_BLOCKS = (
    ("rtsp", "stream settings"),
    ("onvif", "ONVIF settings"),
    ("ptz_tracking", "PTZ tracking settings"),
)

# ptz_state.json overrides these at startup (the Live page writes it when the
# operator toggles patrol/track). A settings save that changes one of them
# must update the state file too, or the next restart quietly reverts it.
PTZ_STATE_KEYS = ("patrol_enabled", "track_enabled", "patrol_return_delay", "patrol_presets")


# ---------------------------------------------------------------------------
# Dotted-path helpers over plain dicts and pydantic models
# ---------------------------------------------------------------------------

def _get(obj: Any, dotted: str, default: Any = _MISSING) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _has(d: dict, dotted: str) -> bool:
    return _get(d, dotted) is not _MISSING


def _set(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _delete(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.get(part)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)


def _assign(model: Any, dotted: str, value: Any) -> None:
    """setattr through a dotted path on nested pydantic models."""
    parts = dotted.split(".")
    cur = model
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _dump(value: Any) -> Any:
    """A comparable plain-data view of a value or pydantic model."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        return value.dict()
    return value


def _clean(value: Any) -> Any:
    """Tidy JSON-decoded scalars before they land in YAML.

    Floats are rounded so ``0.30000000000000004`` never reaches the file;
    strings are stripped; nested containers are cleaned recursively.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _deep_merge(target: dict, incoming: dict, defaults: Optional[dict] = None) -> dict:
    """Merge ``incoming`` into ``target`` in place.

    A sub-key the file does not have is only written when its value differs
    from the schema default (``defaults``), so a hand-kept block such as
    ``ptz_tracking: {enabled: false}`` is not padded out with forty default
    lines on the first save. A ``None`` for a key the file does have is
    written as an explicit ``null`` rather than dropped: dropping would
    resurrect the default, which for ``zoom_fov_calibration_path`` is a
    real path the operator just cleared on purpose.
    """
    defaults = defaults or {}
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            sub_defaults = defaults.get(key) if isinstance(defaults.get(key), dict) else None
            _deep_merge(target[key], value, sub_defaults)
        elif key not in target and key in defaults and defaults[key] == value:
            continue
        else:
            target[key] = value
    return target


def _merge_managed(target: dict, payload: dict, fields: Dict[str, bool],
                   defaults: Optional[dict] = None) -> List[str]:
    """Copy the managed keys present in ``payload`` into ``target``.

    Returns the dotted keys whose value actually changed. A ``None`` removes
    the key (used for ``onvif`` and nullable strings), a dict deep-merges so
    sub-keys the UI does not know about survive. A key the file does not
    have is written only when the value differs from the schema default in
    ``defaults`` — the UI always sends every managed key, and a file that
    relies on defaults should stay that way.
    """
    changed: List[str] = []
    defaults = defaults or {}
    for key in fields:
        if not _has(payload, key):
            continue
        new = _clean(_get(payload, key))
        old = _get(target, key)
        default = _get(defaults, key)
        if new is None:
            if old is not _MISSING and old is not None:
                _delete(target, key)
                changed.append(key)
            continue
        if isinstance(new, dict):
            base = copy.deepcopy(old) if isinstance(old, dict) else {}
            sub_defaults = default if isinstance(default, dict) else None
            merged = _deep_merge(base, new, sub_defaults)
            if old is _MISSING and (not merged or merged == sub_defaults):
                continue
            if old is _MISSING or merged != old:
                _set(target, key, merged)
                changed.append(key)
            continue
        if old is _MISSING and default is not _MISSING and default == new:
            continue
        if old is _MISSING or old != new:
            _set(target, key, new)
            changed.append(key)
    return changed


_DEFAULTS_CACHE: Dict[str, dict] = {}


def general_defaults() -> dict:
    """The schema defaults of the general block, as plain data."""
    if "general" not in _DEFAULTS_CACHE:
        from .config import GeneralSettings
        probe = GeneralSettings(
            storage_root="", logs_root="",
            # A placeholder recipient: the model refuses a block with neither
            # a fallback variable nor a destination. Both keys are dropped
            # below, so they are never mistaken for defaults.
            notification={"pushover_app_token_env": "X", "pushover_user_key_env": "X"},
        )
        dumped = _dump(probe)
        for key in ("storage_root", "logs_root"):
            dumped.pop(key, None)
        dumped["notification"].pop("pushover_app_token_env", None)
        dumped["notification"].pop("pushover_user_key_env", None)
        _DEFAULTS_CACHE["general"] = dumped
    return copy.deepcopy(_DEFAULTS_CACHE["general"])


# ---------------------------------------------------------------------------
# Reading and validating
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> dict:
    """The parsed YAML document, or ``{}`` for an empty file."""
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} is not a YAML mapping")
    return data


def _problem_path(loc: Tuple[Any, ...], raw: dict) -> str:
    """Turn a pydantic error location into a path the UI can point at.

    ``('cameras', 1, 'rtsp', 'uri')`` becomes ``cameras.cam2.rtsp.uri`` so a
    problem lands on the camera by id rather than by list position.
    """
    parts: List[str] = []
    cams = raw.get("cameras") if isinstance(raw, dict) else None
    for i, item in enumerate(loc):
        if i == 1 and parts == ["cameras"] and isinstance(item, int):
            cam = cams[item] if isinstance(cams, list) and item < len(cams) else None
            cid = cam.get("id") if isinstance(cam, dict) else None
            parts.append(str(cid) if cid else str(item))
        else:
            parts.append(str(item))
    return ".".join(parts)


def validate(raw: dict) -> RuntimeConfig:
    """Validate a document the way ``load_runtime_config`` will at startup."""
    try:
        from pydantic import ValidationError
    except Exception:  # pragma: no cover - pydantic is a hard dependency
        ValidationError = Exception  # type: ignore[assignment]
    try:
        if hasattr(RuntimeConfig, "model_validate"):
            return RuntimeConfig.model_validate(raw)
        return RuntimeConfig.parse_obj(raw)  # pragma: no cover - pydantic v1
    except ValidationError as err:  # type: ignore[misc]
        problems = []
        for e in err.errors():
            problems.append({
                "path": _problem_path(tuple(e.get("loc", ())), raw),
                "message": str(e.get("msg", "invalid value")),
            })
        raise ConfigError("The configuration failed validation.", problems) from err
    except ValueError as err:
        raise ConfigError(str(err)) from err


# ---------------------------------------------------------------------------
# Merging a save payload
# ---------------------------------------------------------------------------

def _validate_new_id(cid: Any) -> str:
    import re
    if not isinstance(cid, str) or not cid.strip():
        raise ConfigError("Every camera needs an id.", [{"path": "cameras", "message": "A camera has no id."}])
    cid = cid.strip()
    if not re.match(CAMERA_ID_PATTERN, cid):
        raise ConfigError(
            f"'{cid}' is not a usable camera id.",
            [{"path": f"cameras.{cid}.id",
              "message": "Use letters, digits, '-' or '_' (up to 32 characters); the id names clip folders and URLs."}],
        )
    return cid


def _normalize_destinations(general_in: dict) -> None:
    """Tidy ``notification.destinations`` of a payload in place.

    Entries are written whole (known keys only, in the sample file's order,
    blank names dropped) so a saved list reads the same however the client
    assembled it. Values are not validated here: pydantic reports a bad id
    or variable name with the entry's index in the path.
    """
    notif = general_in.get("notification")
    if not isinstance(notif, dict) or "destinations" not in notif:
        return
    raw = notif["destinations"]
    if raw is None:
        notif["destinations"] = []
        return
    if not isinstance(raw, list):
        raise ConfigError("'notification.destinations' must be a list.",
                          [{"path": "general.notification.destinations", "message": "Expected a list of destinations."}])
    out: List[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError("Every destination must be an object.",
                              [{"path": f"general.notification.destinations.{i}", "message": "Expected an object with id, name and user_key_env."}])
        item: dict = {"id": str(entry.get("id") if entry.get("id") is not None else "").strip()}
        name = str(entry.get("name") if entry.get("name") is not None else "").strip()
        if name:
            item["name"] = name
        item["user_key_env"] = str(entry.get("user_key_env") if entry.get("user_key_env") is not None else "").strip()
        token_env = str(entry.get("app_token_env") if entry.get("app_token_env") is not None else "").strip()
        if token_env:
            item["app_token_env"] = token_env
        out.append(item)
    notif["destinations"] = out


def check_destination_refs(cfg: RuntimeConfig) -> List[Dict[str, str]]:
    """Problems for cameras naming destinations the general block lacks.

    The pipeline only warns about these (a routing typo must not stop
    detection or recording), so the settings page refuses them here, with
    the camera's field as the path, before anything reaches disk.
    """
    known = [dest.id for dest in cfg.general.notification.destinations]
    problems: List[Dict[str, str]] = []
    for cam in cfg.cameras:
        wanted = cam.notification.destinations
        if not wanted:
            continue
        missing = [w for w in wanted if w not in known]
        if not missing:
            continue
        names = ", ".join(f"'{m}'" for m in missing)
        tail = ("Define it under General → Notifications first." if known
                else "No destinations are defined under General → Notifications.")
        problems.append({
            "path": f"cameras.{cam.id}.notification.destinations",
            "message": f"Unknown destination {names}. {tail}",
        })
    return problems


def merge_payload(raw: dict, payload: dict) -> Tuple[dict, Dict[str, Any]]:
    """Apply a settings payload to a parsed document.

    ``payload`` is ``{"general": {...}, "cameras": [{...}, ...]}``; either half
    may be absent. Cameras are matched by id: known ids are updated, unknown
    ids appended, ids missing from the list removed. Returns the new document
    and a change summary::

        {"general": [dotted keys], "cameras": {id: [dotted keys]},
         "added": [ids], "removed": [ids]}
    """
    if not isinstance(payload, dict):
        raise ConfigError("The settings payload must be a JSON object.")
    new = copy.deepcopy(raw)
    changes: Dict[str, Any] = {"general": [], "cameras": {}, "added": [], "removed": []}

    general_in = payload.get("general")
    if general_in is not None:
        if not isinstance(general_in, dict):
            raise ConfigError("'general' must be an object.")
        general = new.get("general")
        if not isinstance(general, dict):
            general = {}
            new["general"] = general
        general_in = copy.deepcopy(general_in)
        _normalize_destinations(general_in)
        changes["general"] = _merge_managed(general, general_in, GENERAL_FIELDS, general_defaults())

    cameras_in = payload.get("cameras")
    if cameras_in is not None:
        if not isinstance(cameras_in, list):
            raise ConfigError("'cameras' must be a list.")
        existing_list = new.get("cameras") or []
        cam_defaults = camera_schema_defaults()
        existing: Dict[str, dict] = {}
        for cam in existing_list:
            if isinstance(cam, dict) and cam.get("id") is not None:
                existing[str(cam["id"])] = cam
        out: List[dict] = []
        seen: List[str] = []
        for cam_in in cameras_in:
            if not isinstance(cam_in, dict):
                raise ConfigError("Every camera must be an object.")
            cid = cam_in.get("id")
            if isinstance(cid, str):
                cid = cid.strip()
            if not cid:
                raise ConfigError("Every camera needs an id.", [{"path": "cameras", "message": "A camera has no id."}])
            cid = str(cid)
            if cid in seen:
                raise ConfigError(f"Camera id '{cid}' appears twice.",
                                  [{"path": f"cameras.{cid}.id", "message": "Duplicate camera id."}])
            seen.append(cid)
            if cid in existing:
                target = existing[cid]
                changed = _merge_managed(target, cam_in, CAMERA_FIELDS, cam_defaults)
                if changed:
                    changes["cameras"][cid] = changed
                out.append(target)
            else:
                _validate_new_id(cid)
                fresh: dict = {"id": cid}
                _merge_managed(fresh, cam_in, CAMERA_FIELDS, cam_defaults)
                ordered = {k: fresh[k] for k in _CAMERA_KEY_ORDER if k in fresh}
                for k, v in fresh.items():
                    ordered.setdefault(k, v)
                changes["added"].append(cid)
                out.append(ordered)
        removed = [cid for cid in existing if cid not in seen]
        if not out and not payload.get("allow_empty"):
            raise ConfigError(
                "Refusing to remove every camera. Add the replacement first, then remove the old one.",
                [{"path": "cameras", "message": "At least one camera must remain."}],
            )
        changes["removed"] = removed
        new["cameras"] = out

    return new, changes


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _header(path: Path) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# {path.name} — written by the Animal Tracker settings page on {stamp}.\n"
        f"# Comments are not preserved across saves; the previous version is in\n"
        f"# {BACKUP_DIR_NAME}/ next to this file.\n"
    )


def _rotate_backups(backup_dir: Path, name: str, keep: int) -> None:
    files = sorted(p for p in backup_dir.glob(name + ".*") if p.is_file())
    for old in files[:-keep] if keep > 0 else files:
        try:
            old.unlink()
        except OSError as err:
            LOGGER.warning("Could not prune old config backup %s: %s", old, err)


def backup_config(path: Path, keep: int = BACKUP_KEEP) -> Optional[Path]:
    """Copy ``path`` into ``backups/`` beside it; returns the copy's path."""
    if not path.exists():
        return None
    backup_dir = path.parent / BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = backup_dir / f"{path.name}.{stamp}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = backup_dir / f"{path.name}.{stamp}-{n}"
    shutil.copy2(path, candidate)
    _rotate_backups(backup_dir, path.name, keep)
    return candidate


def dump_yaml(raw: dict) -> str:
    return yaml.safe_dump(raw, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_text_atomically(path: Path, text: str, keep: int = BACKUP_KEEP,
                          create_mode: int = 0o644) -> Optional[Path]:
    """Back up ``path`` (when it exists), then replace it with ``text``.

    The temp file inherits the target's mode and owner (best effort: chown
    needs root, which the production service has) so a file the operator
    owns stays theirs after a root-run save. A file created from nothing
    gets ``create_mode`` (the secrets file asks for 0600).
    """
    backup = backup_config(path, keep=keep)
    tmp = path.with_name(path.name + ".tmp")
    try:
        st = path.stat()
    except OSError:
        st = None
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, create_mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    if st is not None:
        try:
            os.chmod(tmp, stat.S_IMODE(st.st_mode))
        except OSError:
            pass
        try:
            os.chown(tmp, st.st_uid, st.st_gid)
        except (OSError, AttributeError):
            pass
    os.replace(tmp, path)
    return backup


def write_config(path: Path, raw: dict, keep: int = BACKUP_KEEP) -> Optional[Path]:
    """Back up, then atomically replace ``path`` with ``raw`` as YAML."""
    return write_text_atomically(path, _header(path) + dump_yaml(raw), keep=keep)


# ---------------------------------------------------------------------------
# Live apply and pending-restart detection
# ---------------------------------------------------------------------------

def hot_apply(runtime: Any, workers: Dict[str, Any], cfg: RuntimeConfig,
              changes: Dict[str, Any] = None) -> List[str]:
    """Bring the running process in line with the file, for every live key.

    Values are taken from the *validated* config so the running models
    receive typed values (an int for ``min_frames``, a list for species).
    Restart-only keys are deliberately left alone: the pending-restart diff
    depends on the running copy still holding the old value.

    Every live key that differs is applied, not just the ones this save
    changed. Those are not the same set: a key edited in the file by hand,
    or by another writer, is already in the file the payload was merged into,
    so it never appears in the diff. It used to stay unapplied for ever —
    pressing Save could not fix it either, because the payload then matched
    the file and nothing counted as a change — while the page went on showing
    the file's value as though it were in force.

    ``changes`` is accepted and ignored, so old callers keep working.
    """
    applied: List[str] = []
    if runtime is not None and getattr(runtime, "general", None) is not None:
        for key in _live_general_drift(cfg, runtime):
            try:
                _assign(runtime.general, key, _get(cfg.general, key))
                applied.append(key)
            except Exception as err:  # noqa: BLE001 - one bad field must not block the rest
                LOGGER.warning("Could not apply general.%s live: %s", key, err)
    for cam in cfg.cameras:
        cid = cam.id
        worker = workers.get(cid)
        running = getattr(worker, "camera", None) if worker is not None else None
        if running is None:
            continue
        new_cam = cam
        for key in _live_camera_drift(new_cam, running):
            try:
                _assign(running, key, _get(new_cam, key))
                applied.append(f"{cid}.{key}")
            except Exception as err:  # noqa: BLE001
                LOGGER.warning("Could not apply %s.%s live: %s", cid, key, err)
    return applied


def _live_general_drift(cfg: RuntimeConfig, runtime: Any) -> List[str]:
    """Live general keys whose file value differs from the running one."""
    general = getattr(runtime, "general", None) if runtime is not None else None
    if general is None:
        return []
    out = []
    for key, live in GENERAL_FIELDS.items():
        if not live:
            continue
        if _dump(_get(cfg.general, key, None)) != _dump(_get(general, key, None)):
            out.append(key)
    return out


def _live_camera_drift(cam: Any, running: Any) -> List[str]:
    """Live camera keys whose file value differs from the running one."""
    out = []
    for key, live in CAMERA_FIELDS.items():
        if not live:
            continue
        if _dump(_get(cam, key, None)) != _dump(_get(running, key, None)):
            out.append(key)
    return out


def unapplied_live_changes(cfg: RuntimeConfig, runtime: Any, workers: Dict[str, Any]) -> List[str]:
    """Live settings the file has changed that the running process has not taken.

    A restart is not what these need — they apply without one — so they are
    reported apart from the restart reasons. Saving from the settings page
    now takes them (``hot_apply``), and so does a restart.
    """
    reasons: List[str] = []
    for key in _live_general_drift(cfg, runtime):
        reasons.append(f"general.{key}")
    for cam in cfg.cameras:
        worker = workers.get(cam.id)
        running = getattr(worker, "camera", None) if worker is not None else None
        if running is None:
            continue
        for key in _live_camera_drift(cam, running):
            reasons.append(f"{cam.id}.{key}")
    return reasons


def _camera_label(cam: Any) -> str:
    name = getattr(cam, "name", None)
    cid = getattr(cam, "id", "?")
    return f"'{name}' ({cid})" if name and name != cid else f"'{cid}'"


def pending_restart(cfg: RuntimeConfig, runtime: Any, workers: Dict[str, Any]) -> List[str]:
    """Why the running process differs from the file, in operator words."""
    reasons: List[str] = []
    file_ids = [cam.id for cam in cfg.cameras]
    for cam in cfg.cameras:
        worker = workers.get(cam.id)
        if worker is None:
            reasons.append(f"Camera {_camera_label(cam)} is configured but not running.")
            continue
        running = getattr(worker, "camera", None)
        if running is None:
            continue
        for block, words in _RESTART_CAMERA_BLOCKS:
            if _dump(getattr(cam, block, None)) != _dump(getattr(running, block, None)):
                reasons.append(f"Camera {_camera_label(cam)}: {words} changed.")
    for wid in workers:
        if wid not in file_ids:
            reasons.append(f"Camera '{wid}' is still running but has been removed from the configuration.")
    general = getattr(runtime, "general", None) if runtime is not None else None
    if general is not None:
        seen_words = set()
        for key, words in _RESTART_GENERAL_WORDS.items():
            if _get(cfg.general, key, None) != _get(general, key, None):
                if words not in seen_words:
                    seen_words.add(words)
                    reasons.append(f"General: {words} changed.")
    return reasons


def ptz_state_updates(old_cfg: Optional[RuntimeConfig], new_cfg: RuntimeConfig,
                      changed_cameras: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """The ptz_state.json entries a save must refresh, per camera.

    Only keys whose *file* value changed are returned, so a Live-page toggle
    the operator made earlier is not undone by an unrelated PTZ edit.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for cid in changed_cameras:
        try:
            new_cam = new_cfg.camera_by_id(cid)
        except KeyError:
            continue
        old_cam = None
        if old_cfg is not None:
            try:
                old_cam = old_cfg.camera_by_id(cid)
            except KeyError:
                old_cam = None
        new_ptz = getattr(new_cam, "ptz_tracking", None)
        old_ptz = getattr(old_cam, "ptz_tracking", None) if old_cam is not None else None
        if new_ptz is None:
            continue
        patch = {}
        for key in PTZ_STATE_KEYS:
            new_val = getattr(new_ptz, key, None)
            old_val = getattr(old_ptz, key, None) if old_ptz is not None else _MISSING
            if old_val is _MISSING or old_val != new_val:
                patch[key] = new_val
        if patch:
            out[cid] = patch
    return out


# ---------------------------------------------------------------------------
# The save transaction
# ---------------------------------------------------------------------------

def save(path: Path, payload: dict, runtime: Any, workers: Dict[str, Any]) -> Dict[str, Any]:
    """Merge, validate, back up, write, then apply what can be applied live."""
    with _WRITE_LOCK:
        raw = load_raw(path)
        try:
            old_cfg: Optional[RuntimeConfig] = validate(raw)
        except ConfigError:
            old_cfg = None
        new_raw, changes = merge_payload(raw, payload)
        cfg = validate(new_raw)
        dangling = check_destination_refs(cfg)
        if dangling:
            raise ConfigError("Some cameras name Pushover destinations that do not exist.", dangling)
        touched = bool(changes["general"] or changes["cameras"] or changes["added"] or changes["removed"])
        backup = write_config(path, new_raw) if touched else None
        # Always, not only when this payload changed something: a value
        # edited in the file by hand is the case that needs reconciling.
        applied = hot_apply(runtime, workers, cfg)
        ptz_sync = ptz_state_updates(old_cfg, cfg, list(changes["cameras"].keys()) + list(changes["added"]))
        return {
            "changes": changes,
            "applied_live": applied,
            "backup": str(backup) if backup else None,
            "written": touched,
            "config": cfg,
            "ptz_state": ptz_sync,
        }


# ---------------------------------------------------------------------------
# Describing the configuration for the UI
# ---------------------------------------------------------------------------

def env_references(cfg: RuntimeConfig) -> List[Dict[str, Any]]:
    """Every environment variable the configuration names.

    Each entry is ``{"name", "used_by": [...], "live": bool}``: ``live`` when
    the running process reads the variable on use (the Pushover notifier
    reads the environment on every send), false when it is read once at
    startup (ONVIF credentials, taken when a camera worker connects).
    """
    refs: Dict[str, Dict[str, Any]] = {}

    def add(name: Optional[str], used_by: str, live: bool) -> None:
        if not name:
            return
        entry = refs.setdefault(name, {"name": name, "used_by": [], "live": True})
        entry["used_by"].append(used_by)
        entry["live"] = bool(entry["live"] and live)

    notif = getattr(cfg.general, "notification", None)
    if notif is not None:
        add(getattr(notif, "pushover_app_token_env", None), "Pushover app token", True)
        add(getattr(notif, "pushover_user_key_env", None), "Pushover fallback user key", True)
        for dest in (getattr(notif, "destinations", None) or []):
            add(dest.user_key_env, f"user key of destination '{dest.label}'", True)
            add(dest.app_token_env, f"app token of destination '{dest.label}'", True)
    for cam in cfg.cameras:
        if cam.onvif is not None:
            add(getattr(cam.onvif, "username_env", None), f"ONVIF user of camera '{cam.id}'", False)
            add(getattr(cam.onvif, "password_env", None), f"ONVIF password of camera '{cam.id}'", False)
    return list(refs.values())


def _env_names(cfg: RuntimeConfig) -> List[str]:
    return [ref["name"] for ref in env_references(cfg)]


def env_presence(names: Iterable[str]) -> Dict[str, bool]:
    """Whether each named variable is set and non-empty. Values never leave."""
    return {str(n): bool(os.environ.get(str(n))) for n in names if n}


# ---------------------------------------------------------------------------
# Secrets: write-only values for the variables the configuration names
# ---------------------------------------------------------------------------

def secrets_path(config_path: Path) -> Path:
    """The env file beside the config, the one systemd loads as EnvironmentFile."""
    return Path(config_path).parent / SECRETS_FILE_NAME


def _env_literal(value: str) -> str:
    """``value`` as an env-file line carries it: bare when plain, otherwise
    double-quoted with backslash escapes, which systemd and a shell
    ``source`` both read."""
    if _ENV_PLAIN_VALUE_RE.match(value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def set_secret(config_path: Path, name: str, value: Optional[str], keep: int = BACKUP_KEEP) -> Dict[str, Any]:
    """Write ``NAME=value`` into the env file beside the config and into this
    process's environment; ``None`` or an empty value removes it.

    Only a variable the saved configuration names can be set, plus any
    ``PUSHOVER_…`` variable (nothing but this application's notifier reads
    those, so a key can be pasted while the destination that will use it is
    still unsaved): the page has no login, and the process environment is
    shared with everything the service runs. The value is never logged and
    never returned. The file is
    backed up and replaced atomically like ``cameras.yml``; a file created
    here is mode 0600, an existing one keeps its mode and owner. The first
    existing ``NAME=`` line is replaced in place (later duplicates are
    dropped), otherwise the line is appended; comments and other lines are
    kept as they are.
    """
    name = (name or "").strip()
    if not re.match(ENV_NAME_PATTERN, name):
        raise ConfigError("That is not an environment variable name.",
                          [{"path": "secrets.name", "message": "Use letters, digits and underscores."}])
    if value is not None:
        value = str(value)
        if "\n" in value or "\r" in value:
            raise ConfigError("A secret cannot contain line breaks.",
                              [{"path": "secrets.value", "message": "Line breaks are not allowed."}])
        value = value.strip()
        if len(value) > SECRET_VALUE_MAX_LEN:
            raise ConfigError(f"A secret cannot be longer than {SECRET_VALUE_MAX_LEN} characters.",
                              [{"path": "secrets.value", "message": f"At most {SECRET_VALUE_MAX_LEN} characters."}])
        if not value:
            value = None

    cfg = validate(load_raw(Path(config_path)))
    allowed = {ref["name"]: ref for ref in env_references(cfg)}
    ref = allowed.get(name)
    if ref is None and _FREELY_SETTABLE_RE.match(name):
        ref = {"name": name, "used_by": ["not named by the saved configuration yet"], "live": True}
    if ref is None:
        raise ConfigError(
            f"{name} is not a variable the configuration names.",
            [{"path": "secrets.name",
              "message": "PUSHOVER_… variables can be set at any time; others once the saved configuration names them."}],
        )
    path = secrets_path(config_path)
    with _WRITE_LOCK:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
        out: List[str] = []
        done = False
        for line in lines:
            match = _ENV_LINE_RE.match(line)
            if match and match.group(1) == name and not line.lstrip().startswith("#"):
                if done:
                    continue
                done = True
                if value is not None:
                    out.append(f"{name}={_env_literal(value)}\n")
                continue
            out.append(line)
        if value is not None and not done:
            if out and not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append(f"{name}={_env_literal(value)}\n")
        backup = write_text_atomically(path, "".join(out), keep=keep, create_mode=0o600)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    LOGGER.info("Secret %s %s in %s (%s)", name, "set" if value is not None else "removed",
                path, "; ".join(ref["used_by"]))
    return {
        "name": name,
        "set": value is not None,
        "backup": str(backup) if backup else None,
        "file": str(path),
        "live": bool(ref["live"]),
        "used_by": list(ref["used_by"]),
    }


def camera_defaults() -> Dict[str, Any]:
    """The schema defaults for a camera, for the client's "Add camera" form."""
    probe = CameraConfig(id="new", name="new", rtsp={"uri": "rtsp://example/stream"})
    dumped = _dump(probe)
    for key in ("id", "name"):
        dumped.pop(key, None)
    dumped["rtsp"].pop("uri", None)
    dumped["onvif"] = {"host": "", "port": 80, "profile": None, "username_env": "", "password_env": ""}
    return dumped


def camera_schema_defaults() -> dict:
    """Camera defaults for merging: what an absent key means to pydantic."""
    if "camera" not in _DEFAULTS_CACHE:
        probe = CameraConfig(id="x", name="x", rtsp={"uri": "rtsp://x"})
        dumped = _dump(probe)
        for key in ("id", "name", "onvif"):
            dumped.pop(key, None)
        dumped["rtsp"].pop("uri", None)
        _DEFAULTS_CACHE["camera"] = dumped
    return copy.deepcopy(_DEFAULTS_CACHE["camera"])


def describe(path: Path, runtime: Any, workers: Dict[str, Any],
             annotate: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
             recent_detections: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
    """The GET payload: the validated file, plus runtime annotations."""
    raw = load_raw(path)
    cfg = validate(raw)
    cameras = []
    for cam in cfg.cameras:
        entry = _dump(cam)
        entry["runtime"] = (annotate(cam.id) if annotate else None) or {"running": cam.id in workers}
        entry["recent_detections"] = (recent_detections or {}).get(cam.id, {})
        cameras.append(entry)
    reasons = pending_restart(cfg, runtime, workers)
    unapplied = unapplied_live_changes(cfg, runtime, workers)
    support = restart_support()
    spath = secrets_path(path)
    return {
        "config_path": str(path),
        "backup_dir": str(path.parent / BACKUP_DIR_NAME),
        "general": _dump(cfg.general),
        "cameras": cameras,
        "env": env_presence(_env_names(cfg)),
        "secrets": {"file": str(spath), "exists": spath.exists(), "variables": env_references(cfg)},
        "defaults": {"camera": camera_defaults()},
        "fields": {"general": GENERAL_FIELDS, "camera": CAMERA_FIELDS},
        "unapplied": {
            "count": len(unapplied),
            "keys": unapplied,
        },
        "restart": {
            "required": bool(reasons),
            "reasons": reasons,
            "supported": support["supported"],
            "unit": support["unit"],
        },
    }


# ---------------------------------------------------------------------------
# Connection probes
# ---------------------------------------------------------------------------

def _run_with_timeout(fn: Callable[[], Dict[str, Any]], timeout: float, what: str) -> Dict[str, Any]:
    """Run ``fn`` on a daemon thread and give up after ``timeout`` seconds.

    The thread is abandoned rather than joined on timeout: FFmpeg and zeep
    calls cannot be interrupted, and both carry their own socket timeouts
    so the thread ends on its own shortly after.
    """
    import concurrent.futures
    started = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"probe-{what}")
    future = executor.submit(fn)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        result = {"ok": False, "error": f"No answer within {timeout:.0f} s."}
    except Exception as err:  # noqa: BLE001 - the probe reports, it never raises
        result = {"ok": False, "error": f"{type(err).__name__}: {err}"}
    finally:
        executor.shutdown(wait=False)
    result["elapsed_ms"] = int((time.time() - started) * 1000)
    return result


def probe_rtsp(uri: str, transport: str = "tcp", timeout: float = 10.0) -> Dict[str, Any]:
    """Open the stream with software decoding and read one frame."""
    uri = (uri or "").strip()
    if not uri:
        return {"ok": False, "error": "No stream URI given.", "elapsed_ms": 0}
    transport = transport if transport in ("tcp", "udp") else "tcp"

    def work() -> Dict[str, Any]:
        import cv2
        from .pipeline import _CAPTURE_OPEN_LOCK, build_ffmpeg_uri
        with _CAPTURE_OPEN_LOCK:
            resolved = build_ffmpeg_uri(uri, transport, hwaccel=False)
            cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                return {"ok": False, "error": "Could not open the stream (host unreachable, wrong path, or wrong credentials)."}
            ok, frame = cap.read()
            if not ok or frame is None:
                return {"ok": False, "error": "The stream opened but delivered no frame."}
            height, width = frame.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            return {"ok": True, "width": int(width), "height": int(height), "fps": round(fps, 1)}
        finally:
            cap.release()

    return _run_with_timeout(work, timeout, "rtsp")


def probe_onvif(host: str, port: int, username_env: str, password_env: str,
                timeout: float = 12.0) -> Dict[str, Any]:
    """Connect to the ONVIF service and list its media profiles."""
    host = (host or "").strip()
    env = env_presence([username_env, password_env])
    if not host:
        return {"ok": False, "error": "No ONVIF host given.", "env": env, "elapsed_ms": 0}
    user = os.environ.get(username_env or "", "")
    password = os.environ.get(password_env or "", "")
    if not user or not password:
        missing = [n for n in (username_env, password_env) if not os.environ.get(n or "")]
        return {
            "ok": False, "env": env, "elapsed_ms": 0,
            "error": ", ".join(missing or ["credentials"]) +
                     " not set in the process environment. Add it to config/secrets.env and restart.",
        }
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 80

    def work() -> Dict[str, Any]:
        import socket
        try:
            with socket.create_connection((host, port), timeout=3.0):
                pass
        except OSError as err:
            return {"ok": False, "error": f"TCP connect to {host}:{port} failed: {err}"}
        from .onvif_client import OnvifClient
        client = OnvifClient(host=host, port=port, username=user, password=password)
        device = {}
        try:
            device = client.get_status()
        except Exception as err:  # noqa: BLE001 - device info is a nicety
            device = {"error": str(err)}
        profiles = client.get_profiles()
        return {
            "ok": True,
            "device": device,
            "profiles": [
                {"token": p.metadata.get("token"), "uri": p.uri, "snapshot_uri": p.snapshot_uri}
                for p in profiles
            ],
        }

    result = _run_with_timeout(work, timeout, "onvif")
    result["env"] = env
    return result


# ---------------------------------------------------------------------------
# Restarting the service
# ---------------------------------------------------------------------------

def detect_unit() -> Optional[str]:
    """The systemd unit this process runs under, if any."""
    override = os.environ.get("ANIMALTRACKER_UNIT")
    if override:
        return override
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        seg = line.rsplit("/", 1)[-1].strip()
        if seg.endswith(".service"):
            return seg
    return None


def restart_support() -> Dict[str, Any]:
    under_systemd = bool(os.environ.get("INVOCATION_ID"))
    unit = detect_unit() if under_systemd else None
    return {"supported": bool(under_systemd and unit), "unit": unit}


def restart_now(unit: str) -> None:
    """Ask systemd to restart ``unit``; fall back to a failure exit.

    ``systemctl restart`` from inside the unit works: the job is queued
    before systemd sends SIGTERM to this cgroup. If that is refused (no
    permission), exit with status 3 so a ``Restart=on-failure`` or
    ``Restart=always`` policy brings the service back; with any other
    policy the fallback would leave it down, so it is logged and skipped.
    """
    try:
        subprocess.run(["systemctl", "restart", unit], check=True, timeout=30,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    except Exception as err:  # noqa: BLE001
        LOGGER.error("systemctl restart %s failed: %s", unit, err)
    policy = ""
    try:
        policy = subprocess.run(
            ["systemctl", "show", "-p", "Restart", "--value", unit],
            timeout=10, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except Exception as err:  # noqa: BLE001
        LOGGER.error("Could not read the restart policy of %s: %s", unit, err)
    if policy in ("always", "on-failure"):
        LOGGER.warning("Exiting with status 3 so systemd (%s, Restart=%s) restarts the service", unit, policy)
        logging.shutdown()
        os._exit(3)
    LOGGER.error("Cannot restart %s from inside the process (Restart=%s); restart it by hand.", unit, policy or "?")
