"""Camera registry and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from .config import CameraConfig


@dataclass
class CameraState:
    config: CameraConfig
    last_frame_ts: float | None = None
    last_notification_ts: float | None = None
    healthy: bool = True


@dataclass
class CameraRegistry:
    cameras: Dict[str, CameraState] = field(default_factory=dict)

    @classmethod
    def from_configs(cls, configs: Iterable[CameraConfig]) -> "CameraRegistry":
        states = {cfg.id: CameraState(config=cfg) for cfg in configs}
        return cls(cameras=states)

    def list_ids(self) -> list[str]:
        return list(self.cameras.keys())

    def update_health(self, camera_id: str, healthy: bool) -> None:
        if camera_id in self.cameras:
            self.cameras[camera_id].healthy = healthy

    def record_frame(self, camera_id: str, ts: float) -> None:
        if camera_id in self.cameras:
            self.cameras[camera_id].last_frame_ts = ts

    def record_notification(self, camera_id: str, ts: float) -> None:
        if camera_id in self.cameras:
            self.cameras[camera_id].last_notification_ts = ts
