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
    # Thumbnail settings
    thumbnail_cropped: bool = Field(default=True, description="Crop thumbnails to detection area (True=zoomed, False=full frame with bbox)")
    # Post-clip analysis settings
    post_analysis: bool = Field(default=True, description="Run species re-analysis on saved clips for better identification")
    # Note: post_analysis_frames is auto-calculated as 1 frame per second of clip duration
    post_analysis_confidence: float = Field(default=0.3, ge=0, le=1, description="Confidence threshold for post-analysis (lower catches more)")
    post_analysis_generic_confidence: float = Field(default=0.5, ge=0, le=1, description="Generic category threshold for post-analysis")
    sample_rate: int = Field(default=3, ge=1, le=30, description="Analyze every Nth frame (lower = more thorough)")
    tracking_enabled: bool = Field(default=True, description="Enable object tracking to identify same animal across frames")
    track_merge_gap: int = Field(default=120, ge=10, le=500, description="Max frame gap to merge same-species tracks")
    spatial_merge_enabled: bool = Field(default=True, description="Merge tracks in same location (ignores species misclassifications)")
    spatial_merge_iou: float = Field(default=0.3, ge=0.1, le=0.9, description="Min bounding box overlap to merge (0.3 = 30%)")
    hierarchical_merge_enabled: bool = Field(default=True, description="Merge 'animal' tracks into specific species tracks")
    single_animal_mode: bool = Field(default=False, description="Force merge ALL non-overlapping tracks into one")
    # Unified post-processing (new approach - analyze saved video file instead of in-memory frames)
    unified_post_processing: bool = Field(default=True, description="Use unified post-processor for consistent results (recommended)")
    post_analysis_frames: int = Field(default=0, ge=0, description="Number of frames for post-analysis (0=auto-calculate)")


class DetectorSettings(BaseModel):
    """Configuration for the detection backend."""
    backend: str = Field(default="yolo", pattern="^(yolo|speciesnet)$")
    # YOLO settings
    model_path: str = "yolov8n.pt"
    # SpeciesNet settings
    speciesnet_version: str = "v4.0.2a"  # v4.0.2a (crop) or v4.0.2b (full-image)
    country: Optional[str] = None  # ISO 3166-1 alpha-3 (e.g., "USA")
    admin1_region: Optional[str] = None  # State code for US (e.g., "TX")
    # Geospatial priors for improved species accuracy
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Camera latitude for species range filtering")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Camera longitude for species range filtering")
    # Tiered confidence thresholds (SpeciesNet only)
    # Generic categories (animal, bird, mammalia) require higher confidence
    # Specific species (cardinal, blue_jay) use the camera's normal threshold
    generic_confidence: Optional[float] = Field(default=0.9, ge=0, le=1, description="Higher threshold for generic categories like 'animal', 'bird'")


class EBirdSettings(BaseModel):
    """Configuration for eBird seasonal species filtering."""
    enabled: bool = Field(default=False, description="Enable eBird seasonal filtering for birds")
    api_key_env: str = Field(default="EBIRD_API_KEY", description="Environment variable containing eBird API key")
    region: str = Field(default="US-MN", description="eBird region code (e.g., US-MN, US, CA-ON)")
    days_back: int = Field(default=14, ge=1, le=30, description="Days of recent observations to consider")
    filter_mode: str = Field(default="flag", pattern="^(flag|filter|boost)$", description="flag=mark unlikely, filter=remove unlikely, boost=increase confidence for present species")
    cache_hours: int = Field(default=24, ge=1, le=168, description="Hours to cache eBird data")


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
    hwaccel: bool = Field(default=False, description="Use NVDEC GPU hardware decoding (requires GStreamer)")


class ThresholdSettings(BaseModel):
    confidence: float = Field(default=0.5, ge=0, le=1)
    generic_confidence: float = Field(default=0.9, ge=0, le=1, description="Higher threshold for generic categories like 'animal', 'bird'")
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
    ebird: EBirdSettings = EBirdSettings()
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
