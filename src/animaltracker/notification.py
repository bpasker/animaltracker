"""Notification delivery (Pushover)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

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


class PushoverNotifier:
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
    def _user_key(self) -> str:
        key = os.environ.get(self.user_key_env)
        if not key:
            raise RuntimeError(f"Missing env var {self.user_key_env} for Pushover user key")
        return key

    def send(self, ctx: NotificationContext, priority: int = 0, sound: Optional[str] = None) -> None:
        message = self._format_message(ctx)
        LOGGER.info("Dispatching Pushover alert for %s (%s)", ctx.species, ctx.camera_id)
        files = None
        data = {
            "token": self._app_token,
            "user": self._user_key,
            "title": f"{ctx.species.title()} detected @ {ctx.camera_name}",
            "message": message,
            "priority": priority,
            "sound": sound or "pushover",
        }
        try:
            response = requests.post(PUSHOVER_ENDPOINT, data=data, files=files, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # noqa: BLE001
            LOGGER.exception("Failed to send Pushover alert: %s", exc)

    @staticmethod
    def _format_message(ctx: NotificationContext) -> str:
        return (
            f"Species: {ctx.species}\n"
            f"Confidence: {ctx.confidence:.2f}\n"
            f"Camera: {ctx.camera_name} ({ctx.camera_id})\n"
            f"Duration: {ctx.event_duration:.1f}s\n"
            f"Clip: {ctx.clip_path}"
        )
