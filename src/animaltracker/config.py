"""Configuration loading for Animal Tracker."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os
import yaml
from pydantic import BaseModel, Field, validator


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class ClipSettings(BaseModel):
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=5.0, ge=0)
    format: str = "mp4"
    codec: str = "h264"
    # Post-clip analysis settings
    post_analysis: bool = Field(default=True, description="Run species re-analysis on saved clips for better identification")
    post_analysis_frames: int = Field(default=60, ge=1, le=120, description="Number of frames to analyze from each clip")


class DetectorSettings(BaseModel):
    """Configuration for the detection backend."""
    backend: str = Field(default="yolo", pattern="^(yolo|speciesnet)$")
    # YOLO settings
    model_path: str = "yolov8n.pt"
    # SpeciesNet settings
    speciesnet_version: str = "v4.0.2a"  # v4.0.2a (crop) or v4.0.2b (full-image)
    country: Optional[str] = None  # ISO 3166-1 alpha-3 (e.g., "USA")
    admin1_region: Optional[str] = None  # State code for US (e.g., "CA")


class RetentionSettings(BaseModel):
    min_days: int = Field(default=7, ge=1)
    max_days: int = Field(default=30, ge=1)
    max_utilization_pct: int = Field(default=80, ge=1, le=99)


class NotificationSettings(BaseModel):
    pushover_app_token_env: str
    pushover_user_key_env: str


class ONVIFSettings(BaseModel):
    host: str
    port: int = 80
    profile: Optional[str] = None
    username_env: str
    password_env: str

    def credentials(self) -> tuple[str, str]:
        return (
            os.environ.get(self.username_env, ""),
            os.environ.get(self.password_env, ""),
        )


class RTSPSettings(BaseModel):
    uri: str
    frame_skip: int = Field(default=1, ge=1, description="Process every Nth frame (1=all, 2=half, etc)")
    transport: str = Field(default="tcp", pattern="^(tcp|udp)$")
    latency_ms: int = Field(default=0, ge=0)


class ThresholdSettings(BaseModel):
    confidence: float = Field(default=0.5, ge=0, le=1)
    min_frames: int = Field(default=3, ge=1)
    min_duration: float = Field(default=2.0, ge=0)


class CameraNotificationSettings(BaseModel):
    priority: int = 0
    sound: Optional[str] = None


class CameraConfig(BaseModel):
    id: str
    name: str
    location: Optional[str] = None
    rtsp: RTSPSettings
    onvif: ONVIFSettings
    thresholds: ThresholdSettings = ThresholdSettings()
    detect_enabled: bool = True
    include_species: List[str] = Field(default_factory=list)
    exclude_species: List[str] = Field(default_factory=list)
    notification: CameraNotificationSettings = CameraNotificationSettings()

    @validator("id")
    def _validate_id(cls, value: str) -> str:
        safe = value.strip()
        if not safe:
            raise ValueError("Camera id cannot be empty")
        return safe


class GeneralSettings(BaseModel):
    storage_root: str
    logs_root: str
    metrics_port: int = 9500
    clip: ClipSettings = ClipSettings()
    detector: DetectorSettings = DetectorSettings()
    exclusion_list: List[str] = Field(default_factory=list)
    notification: NotificationSettings
    retention: RetentionSettings = RetentionSettings()


class RuntimeConfig(BaseModel):
    general: GeneralSettings
    cameras: List[CameraConfig]

    def camera_by_id(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(f"Camera '{camera_id}' not found")


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load YAML configuration into strongly typed settings."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = _load_yaml(cfg_path)
    return RuntimeConfig.parse_obj(raw)
