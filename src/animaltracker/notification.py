"""Notification delivery (Pushover)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

from .config import MAX_COOLDOWN_MINUTES, NotificationSettings, PushoverDestination
from .species_names import get_common_name, species_matches, species_words

LOGGER = logging.getLogger(__name__)
PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"

# Where the pipeline keeps AlertTimes, under general.logs_root.
ALERT_TIMES_FILE = "alert_times.json"


@dataclass
class NotificationContext:
    species: str
    confidence: float
    camera_id: str
    camera_name: str
    clip_path: str
    event_started_at: float
    event_duration: float
    thumbnail_path: Optional[str] = None
    storage_root: Optional[str] = None
    web_base_url: Optional[str] = None


@dataclass(frozen=True)
class Cooldown:
    """The repeat rule a clip's species falls under.

    ``animal`` keys the camera's timer: the species_cooldowns entry that
    matched, or else the common name, as lowercase words. A settings entry
    named "squirrel" and the common name "Squirrel" share one timer.
    """
    animal: str
    minutes: float


def cooldown_for(settings: NotificationSettings, species: str) -> Cooldown:
    """The first species_cooldowns entry that matches ``species``, else cooldown_minutes."""
    for entry in getattr(settings, "species_cooldowns", None) or []:
        if species_matches(species, entry.species):
            return Cooldown(" ".join(species_words(entry.species)), float(entry.minutes))
    return Cooldown(" ".join(species_words(get_common_name(species))),
                    float(getattr(settings, "cooldown_minutes", 0.0) or 0.0))


@dataclass(frozen=True)
class Claim:
    """What AlertTimes.claim decided, and what it needs to take it back."""
    allowed: bool
    camera_id: str
    animal: str
    last: Optional[float]      # the camera's time for this animal before the claim
    stamped: Optional[float]   # what the claim wrote (None when held back)


class AlertTimes:
    """When each camera last alerted for each animal, kept on disk.

    A camera that has just alerted for an animal holds further alerts for it
    back until that animal's cooldown has passed. The times are the events'
    start times, not the moments the analyses finished, so a backlog of
    analyses does not bunch alerts up; and they live in a small JSON file, so
    a restart does not re-alert for the squirrel of ten minutes ago. Only the
    alert is held back: the clip is recorded, analysed and kept as usual.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._times: Dict[str, Dict[str, float]] = self._load()

    def _load(self) -> Dict[str, Dict[str, float]]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            out: Dict[str, Dict[str, float]] = {}
            for camera_id, animals in (raw.get("cameras") or {}).items():
                out[str(camera_id)] = {str(a): float(t) for a, t in animals.items()}
            return out
        except Exception as err:  # noqa: BLE001 - a bad file costs one repeat alert, never the alert itself
            LOGGER.warning("Could not read the alert times in %s (%s); starting afresh", self.path, err)
            return {}

    def _save_locked(self) -> None:
        if self.path is None:
            return
        horizon = time.time() - MAX_COOLDOWN_MINUTES * 60
        cameras = {}
        for camera_id, animals in self._times.items():
            kept = {a: t for a, t in animals.items() if t >= horizon}
            if kept:
                cameras[camera_id] = kept
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"cameras": cameras}, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as err:
            LOGGER.warning("Could not save the alert times to %s: %s", self.path, err)

    def last(self, camera_id: str, animal: str) -> Optional[float]:
        with self._lock:
            return self._times.get(camera_id, {}).get(animal)

    def claim(self, camera_id: str, animal: str, minutes: float, at: float) -> Claim:
        """Take the alert for ``animal`` on ``camera_id`` for an event that started ``at``.

        Allowed unless the camera alerted for this animal less than
        ``minutes`` before or after ``at`` (an analysis can finish out of
        order). An allowed claim writes its time at once, before the alert
        is sent, so two analyses finishing together cannot both go out;
        ``release`` takes it back when the alert reached nobody.
        """
        with self._lock:
            animals = self._times.setdefault(camera_id, {})
            last = animals.get(animal)
            if last is not None and minutes > 0 and abs(at - last) < minutes * 60:
                return Claim(False, camera_id, animal, last, None)
            stamped = at if last is None else max(at, last)
            animals[animal] = stamped
            self._save_locked()
            return Claim(True, camera_id, animal, last, stamped)

    def release(self, claim: Claim) -> None:
        """Undo an allowed claim, unless a later one has replaced it."""
        if not claim.allowed:
            return
        with self._lock:
            animals = self._times.get(claim.camera_id, {})
            if animals.get(claim.animal) != claim.stamped:
                return
            if claim.last is None:
                animals.pop(claim.animal, None)
            else:
                animals[claim.animal] = claim.last
            self._save_locked()


class PushoverNotifier:
    """Pushover sender that routes each alert to named destinations.

    ``settings`` is the running ``NotificationSettings``; it is read on every
    send, so the settings page can change destinations and variable names
    without a restart. A destination names an environment variable holding a
    Pushover user or group key (a comma-separated list sends to each key).

    A camera passes the destination ids it wants: ``None`` means every
    destination (or the fallback ``pushover_user_key_env`` while none are
    defined), ``[]`` means none. Ids that are not configured are logged and
    skipped rather than raised: a routing typo must never stop detection.

    Repeat alerts are held back per camera and animal (``cooldown_minutes``,
    ``species_cooldowns``, read on every send like the destinations); the
    times are kept in ``alert_times_path`` when one is given.
    """

    def __init__(self, settings: NotificationSettings, alert_times_path: Optional[Path] = None) -> None:
        self.settings = settings
        self.alert_times = AlertTimes(alert_times_path)

    def recipients_for(self, destinations: Optional[Sequence[str]], camera_id: str = "?") -> List[PushoverDestination]:
        """The destinations an alert from ``camera_id`` goes to."""
        if destinations is None:
            return self.settings.resolved_destinations()
        known = {dest.id: dest for dest in self.settings.destinations}
        picked: List[PushoverDestination] = []
        unknown: List[str] = []
        for wanted in destinations:
            dest = known.get(wanted)
            if dest is None:
                unknown.append(str(wanted))
            elif all(p.id != dest.id for p in picked):
                picked.append(dest)
        if unknown:
            LOGGER.warning(
                "Camera %s names Pushover destination(s) %s that are not configured; ignoring them",
                camera_id, ", ".join(unknown),
            )
        return picked

    def _app_token(self, dest: PushoverDestination) -> Optional[str]:
        """The application token a destination sends with; None (and logged) when unset."""
        env = dest.app_token_env or self.settings.pushover_app_token_env or ""
        token = os.environ.get(env) if env else None
        if not token:
            LOGGER.error(
                "Pushover destination '%s' needs the app token in %s, which is not set in the environment; skipping it",
                dest.label, env or "<unset>",
            )
        return token

    @staticmethod
    def _user_keys(dest: PushoverDestination) -> List[str]:
        """The key(s) a destination's variable holds; empty (and logged) when unset."""
        raw = os.environ.get(dest.user_key_env or "", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            LOGGER.error(
                "Pushover destination '%s' reads %s, which is not set in the environment; skipping it",
                dest.label, dest.user_key_env,
            )
        return keys

    @staticmethod
    def _attachment(ctx: NotificationContext) -> Optional[dict]:
        """The thumbnail as a multipart file, read once for every recipient."""
        if not ctx.thumbnail_path:
            return None
        try:
            from pathlib import Path
            thumb_path = Path(ctx.thumbnail_path)
            if thumb_path.exists():
                with open(thumb_path, "rb") as img_file:
                    LOGGER.debug("Attaching thumbnail: %s", thumb_path)
                    return {"attachment": (thumb_path.name, img_file.read(), "image/jpeg")}
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Failed to attach thumbnail %s: %s", ctx.thumbnail_path, e)
        return None

    def send(self, ctx: NotificationContext, priority: int = 0, sound: Optional[str] = None,
             destinations: Optional[Sequence[str]] = None) -> int:
        """Send the alert; returns how many user keys it was delivered to.

        Nothing is sent while the camera's cooldown for this animal runs.
        """
        common_name = get_common_name(ctx.species)
        recipients = self.recipients_for(destinations, ctx.camera_id)
        if not recipients:
            LOGGER.info("No Pushover destination for camera %s; alert for %s not sent", ctx.camera_id, common_name)
            return 0

        rule = cooldown_for(self.settings, ctx.species)
        claim = self.alert_times.claim(ctx.camera_id, rule.animal, rule.minutes, ctx.event_started_at)
        if not claim.allowed:
            LOGGER.info(
                "Holding back the %s alert from %s: its last %s alert was %.0f min from this one "
                "and the cooldown is %g min. The clip is kept.",
                common_name, ctx.camera_id, rule.animal,
                abs(ctx.event_started_at - claim.last) / 60, rule.minutes,
            )
            return 0

        sent = 0
        try:
            data = {
                "title": f"{common_name} detected @ {ctx.camera_name}",
                "message": self._format_message(ctx),
                "priority": priority,
                "sound": sound or "pushover",
            }
            # Add clickable URL to video if web_base_url is configured
            clip_url = self._build_clip_url(ctx)
            if clip_url:
                data["url"] = clip_url
                data["url_title"] = "View Recording"
            files = self._attachment(ctx)

            LOGGER.info("Dispatching Pushover alert for %s (%s) to %s",
                        common_name, ctx.camera_id, ", ".join(dest.label for dest in recipients))
            seen_keys: set[str] = set()
            for dest in recipients:
                token = self._app_token(dest)
                if not token:
                    continue
                for user_key in self._user_keys(dest):
                    if user_key in seen_keys:
                        continue  # two destinations sharing a key still get one alert
                    seen_keys.add(user_key)
                    try:
                        response = requests.post(PUSHOVER_ENDPOINT, data=dict(data, token=token, user=user_key),
                                                 files=files, timeout=15)
                        response.raise_for_status()
                        sent += 1
                        LOGGER.debug("Pushover alert sent to '%s' (user key ending in ...%s)", dest.label, user_key[-4:])
                    except requests.RequestException as exc:  # noqa: BLE001
                        LOGGER.exception("Failed to send Pushover alert to '%s' (user key ending in ...%s): %s",
                                         dest.label, user_key[-4:], exc)
        finally:
            if not sent:
                # Reached nobody: the next clip of this animal may try again.
                self.alert_times.release(claim)
        return sent

    def _build_clip_url(self, ctx: NotificationContext) -> Optional[str]:
        """Build a clickable URL to the clip if web_base_url is configured."""
        if not ctx.web_base_url or not ctx.storage_root or not ctx.clip_path:
            return None
        
        try:
            from pathlib import Path
            clip_path = Path(ctx.clip_path)
            storage_root = Path(ctx.storage_root)
            clips_dir = storage_root / "clips"
            
            # Get relative path from clips directory
            if clips_dir in clip_path.parents or clip_path.parent == clips_dir:
                rel_path = clip_path.relative_to(clips_dir)
            else:
                # Fallback: try to find "clips" in the path
                parts = clip_path.parts
                if "clips" in parts:
                    clips_idx = parts.index("clips")
                    rel_path = Path(*parts[clips_idx + 1:])
                else:
                    rel_path = clip_path.name
            
            # Build URL: base_url/clips/relative_path
            base_url = ctx.web_base_url.rstrip("/")
            return f"{base_url}/clips/{rel_path}"
        except Exception as e:
            LOGGER.warning("Failed to build clip URL: %s", e)
            return None

    @staticmethod
    def _format_message(ctx: NotificationContext) -> str:
        common_name = get_common_name(ctx.species)
        return (
            f"Species: {common_name}\n"
            f"Confidence: {ctx.confidence:.2f}\n"
            f"Camera: {ctx.camera_name} ({ctx.camera_id})\n"
            f"Duration: {ctx.event_duration:.1f}s"
        )
