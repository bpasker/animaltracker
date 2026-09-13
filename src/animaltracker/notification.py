"""Notification delivery (Pushover)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import requests

from .config import NotificationSettings, PushoverDestination
from .species_names import get_common_name

LOGGER = logging.getLogger(__name__)
PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"


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
    """

    def __init__(self, settings: NotificationSettings) -> None:
        self.settings = settings

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
        """Send the alert; returns how many user keys it was delivered to."""
        common_name = get_common_name(ctx.species)
        recipients = self.recipients_for(destinations, ctx.camera_id)
        if not recipients:
            LOGGER.info("No Pushover destination for camera %s; alert for %s not sent", ctx.camera_id, common_name)
            return 0

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
        sent = 0
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
