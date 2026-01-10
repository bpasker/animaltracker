"""Notification delivery (Pushover)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

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
    """Pushover notification sender supporting multiple user keys.
    
    User keys can be specified as a comma-separated list in the environment variable.
    For example: PUSHOVER_USER_KEY=user1key,user2key,user3key
    
    Each user key receives an independent notification.
    """
    
    def __init__(self, app_token_env: str, user_key_env: str) -> None:
        self.app_token_env = app_token_env
        self.user_key_env = user_key_env

    @property
    def _app_token(self) -> str:
        token = os.environ.get(self.app_token_env)
        if not token:
            raise RuntimeError(f"Missing env var {self.app_token_env} for Pushover token")
        return token

    @property
    def _user_keys(self) -> list[str]:
        """Return list of user keys (supports comma-separated values)."""
        keys_str = os.environ.get(self.user_key_env)
        if not keys_str:
            raise RuntimeError(f"Missing env var {self.user_key_env} for Pushover user key")
        # Split by comma and strip whitespace from each key
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(f"No valid user keys found in {self.user_key_env}")
        return keys

    def send(self, ctx: NotificationContext, priority: int = 0, sound: Optional[str] = None) -> None:
        message = self._format_message(ctx)
        common_name = get_common_name(ctx.species)
        user_keys = self._user_keys
        LOGGER.info("Dispatching Pushover alert for %s (%s) to %d recipient(s)", 
                    common_name, ctx.camera_id, len(user_keys))
        
        for user_key in user_keys:
            files = None
            data = {
                "token": self._app_token,
                "user": user_key,
                "title": f"{common_name} detected @ {ctx.camera_name}",
                "message": message,
                "priority": priority,
                "sound": sound or "pushover",
            }
            
            # Add clickable URL to video if web_base_url is configured
            clip_url = self._build_clip_url(ctx)
            if clip_url:
                data["url"] = clip_url
                data["url_title"] = "View Recording"
            
            # Attach thumbnail image if available
            if ctx.thumbnail_path:
                try:
                    from pathlib import Path
                    thumb_path = Path(ctx.thumbnail_path)
                    if thumb_path.exists():
                        # Read image file and attach as multipart form data
                        with open(thumb_path, "rb") as img_file:
                            files = {"attachment": (thumb_path.name, img_file.read(), "image/jpeg")}
                        LOGGER.debug("Attaching thumbnail: %s", thumb_path)
                except Exception as e:
                    LOGGER.warning("Failed to attach thumbnail %s: %s", ctx.thumbnail_path, e)
            
            try:
                response = requests.post(PUSHOVER_ENDPOINT, data=data, files=files, timeout=15)
                response.raise_for_status()
                LOGGER.debug("Pushover alert sent successfully to user key ending in ...%s", user_key[-4:])
            except requests.RequestException as exc:  # noqa: BLE001
                LOGGER.exception("Failed to send Pushover alert to user key ending in ...%s: %s", 
                                user_key[-4:], exc)

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
