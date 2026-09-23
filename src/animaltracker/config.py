"""Configuration loading for Animal Tracker."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os
import re
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, validator

# A destination id names the recipient in each camera's list, the way a
# camera id names clip folders: short, permanent, safe in YAML and URLs.
DESTINATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"
ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class ClipSettings(BaseModel):
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=5.0, ge=0)
    max_event_seconds: float = Field(default=300.0, ge=30, description="Maximum event duration in seconds (prevents memory leak from endless events)")
    max_concurrent_postprocess: int = Field(default=2, ge=1, le=8, description="Maximum concurrent post-processing jobs (prevents RAM explosion)")
    format: str = "mp4"
    codec: str = "h264"
    # Thumbnail settings
    thumbnail_cropped: bool = Field(default=True, description="Crop thumbnails to detection area (True=zoomed, False=full frame with bbox)")
    # Post-clip analysis settings
    post_analysis: bool = Field(default=True, description="Run species re-analysis on saved clips for better identification")
    # Note: post_analysis_frames is auto-calculated as 1 frame per second of clip duration
    post_analysis_confidence: float = Field(default=0.3, ge=0, le=1, description="Confidence threshold for post-analysis (lower catches more)")
    post_analysis_generic_confidence: float = Field(default=0.5, ge=0, le=1, description="Generic category threshold for post-analysis")
    # False positive cleanup: delete clips where post-processing finds no animal
    delete_if_no_animal: bool = Field(default=True, description="Delete clip and skip notification if post-processing finds no animal (reduces false positives)")
    min_detection_frames: int = Field(default=2, ge=1, le=100, description="Minimum number of sampled frames with a detection for a clip to be considered a real animal event. Single-frame SpeciesNet hits in an otherwise blank clip are treated as false positives and the clip is deleted (when delete_if_no_animal=True).")
    min_reptile_detection_frames: int = Field(default=8, ge=1, le=100, description="Minimum sampled detection frames required to keep class-only reptile/amphibian clips. Helps reject static pipe/hose false positives.")
    sample_rate: int = Field(default=3, ge=1, le=30, description="Analyze every Nth frame (lower = more thorough)")
    tracking_enabled: bool = Field(default=True, description="Enable object tracking to identify same animal across frames")
    track_merge_gap: int = Field(default=120, ge=10, le=500, description="Max frame gap to merge same-species tracks")
    spatial_merge_enabled: bool = Field(default=True, description="Merge tracks in same location (ignores species misclassifications)")
    spatial_merge_iou: float = Field(default=0.3, ge=0.1, le=0.9, description="Min bounding box overlap to merge (0.3 = 30%)")
    spatial_merge_reach: float = Field(default=1.0, ge=0, le=5, description="Also merge a track that starts within this many body lengths (longest box side) of where the previous one ended, whatever the overlap; 0 = overlap only")
    hierarchical_merge_enabled: bool = Field(default=True, description="Merge 'animal' tracks into specific species tracks")
    single_animal_mode: bool = Field(default=False, description="Force merge ALL non-overlapping tracks into one")
    # Unified post-processing (new approach - analyze saved video file instead of in-memory frames)
    unified_post_processing: bool = Field(default=True, description="Use unified post-processor for consistent results (recommended)")
    recover_unfinished_clips: bool = Field(default=True, description="Shortly after startup and every 30 minutes, finish analysing clips whose post-processing was interrupted (a restart mid-job leaves a clip unclassified with no key frames); runs newest first while no live event needs the post-processor")
    post_analysis_frames: int = Field(default=0, ge=0, description="Number of frames for post-analysis (0=auto-calculate)")


class DetectorSettings(BaseModel):
    """Configuration for the detection backend.

    Supports split-model architecture for optimal performance:
    - realtime_backend: Detector for live PTZ tracking (default: speciesnet)
    - postprocess_backend: Detector for clip analysis (default: speciesnet)

    The legacy 'backend' field is still supported for backwards compatibility.
    """
    # Legacy single-backend setting (deprecated, use realtime/postprocess instead)
    backend: str = Field(default="speciesnet", pattern="^(yolo|speciesnet|megadetector)$")

    # Split-model architecture
    realtime_backend: str = Field(default="speciesnet", pattern="^(yolo|speciesnet|megadetector)$", description="Detector for live streaming/PTZ tracking")
    postprocess_backend: str = Field(default="speciesnet", pattern="^(yolo|speciesnet|megadetector)$", description="Detector for post-clip analysis")
    
    # YOLO settings
    model_path: str = "yolov8n.pt"
    # SpeciesNet settings
    speciesnet_version: str = "v4.0.3a"  # v4.0.3a (crop) or v4.0.3b (full-image)
    country: Optional[str] = None  # ISO 3166-1 alpha-3 (e.g., "USA")
    admin1_region: Optional[str] = None  # State code for US (e.g., "TX")
    # Geospatial priors for improved species accuracy
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Camera latitude for species range filtering")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Camera longitude for species range filtering")
    # Tiered confidence thresholds (SpeciesNet only)
    # Generic categories (animal, bird, mammalia) require higher confidence
    # Specific species (cardinal, blue_jay) use the camera's normal threshold
    generic_confidence: Optional[float] = Field(default=0.9, ge=0, le=1, description="Higher threshold for generic categories like 'animal', 'bird'")


class RetentionSettings(BaseModel):
    min_days: int = Field(default=7, ge=1)
    max_days: int = Field(default=30, ge=1)
    max_utilization_pct: int = Field(default=80, ge=1, le=99)


class PushoverDestination(BaseModel):
    """One Pushover recipient a camera can be pointed at.

    The user (or group) key itself stays in the environment like every
    other secret; the config only names the variable that holds it.
    """
    id: str = Field(description="Short, permanent id cameras refer to (letters, digits, '-' or '_', up to 32 characters)")
    name: Optional[str] = Field(default=None, description="Label shown in the settings page; the id when empty")
    user_key_env: str = Field(description="Environment variable holding the Pushover user or group key. A comma-separated list sends to each key.")
    app_token_env: Optional[str] = Field(default=None, description="Environment variable holding a Pushover application token to send with instead of pushover_app_token_env (a destination on another Pushover account)")

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        safe = (value or "").strip()
        if not re.match(DESTINATION_ID_PATTERN, safe):
            raise ValueError("use letters, digits, '-' or '_' (up to 32 characters)")
        return safe

    @field_validator("name")
    @classmethod
    def _blank_name_is_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        safe = value.strip()
        return safe or None

    @field_validator("user_key_env")
    @classmethod
    def _validate_env_name(cls, value: str) -> str:
        safe = (value or "").strip()
        if not re.match(ENV_NAME_PATTERN, safe):
            raise ValueError("must be an environment variable name (letters, digits and underscores)")
        return safe

    @field_validator("app_token_env")
    @classmethod
    def _validate_optional_env_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        safe = value.strip()
        if not safe:
            return None
        if not re.match(ENV_NAME_PATTERN, safe):
            raise ValueError("must be an environment variable name (letters, digits and underscores)")
        return safe

    @property
    def label(self) -> str:
        return self.name or self.id


class NotificationSettings(BaseModel):
    pushover_app_token_env: str
    pushover_user_key_env: Optional[str] = Field(default=None, description="Fallback: the variable holding the user key(s) every camera alerts while no destinations are defined. Several keys can be comma-separated.")
    destinations: List[PushoverDestination] = Field(default_factory=list, description="Named Pushover recipients. A camera alerts every one of them unless its own notification.destinations lists a subset.")
    web_base_url: Optional[str] = Field(default=None, description="Base URL for web UI (e.g., http://192.168.1.195:8080). Used for clickable links in notifications.")

    @field_validator("pushover_user_key_env")
    @classmethod
    def _blank_key_env_is_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        safe = value.strip()
        return safe or None

    @field_validator("destinations")
    @classmethod
    def _unique_destination_ids(cls, value: List[PushoverDestination]) -> List[PushoverDestination]:
        seen: set[str] = set()
        for dest in value:
            if dest.id in seen:
                raise ValueError(f"destination id '{dest.id}' appears twice")
            seen.add(dest.id)
        return value

    @model_validator(mode="after")
    def _needs_a_recipient(self) -> "NotificationSettings":
        if not self.destinations and not self.pushover_user_key_env:
            raise ValueError("set pushover_user_key_env or define at least one destination")
        return self

    def resolved_destinations(self) -> List[PushoverDestination]:
        """The recipients a camera without its own list alerts.

        The configured destinations when there are any; otherwise the
        fallback variable, presented as one implicit destination so the
        notifier has a single code path. Not selectable by cameras: their
        lists are checked against ``destinations`` only.
        """
        if self.destinations:
            return list(self.destinations)
        if self.pushover_user_key_env:
            return [PushoverDestination(id="default", name="Default", user_key_env=self.pushover_user_key_env)]
        return []


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
    frame_skip: int = Field(default=1, ge=0, description="Process every Nth frame (0 or 1=all frames, 2=half, etc)")
    transport: str = Field(default="tcp", pattern="^(tcp|udp)$")
    latency_ms: int = Field(default=0, ge=0)
    hwaccel: bool = Field(default=False, description="Use NVDEC GPU hardware decoding (requires GStreamer)")


class ThresholdSettings(BaseModel):
    confidence: float = Field(default=0.5, ge=0, le=1)
    generic_confidence: float = Field(default=0.9, ge=0, le=1, description="Higher threshold for generic categories like 'animal', 'bird'")
    min_frames: int = Field(default=3, ge=1)
    min_duration: float = Field(default=2.0, ge=0)
    min_detection_area: float = Field(default=0.005, ge=0.0, le=0.5, description="Ignore detections smaller than this fraction of frame area (0.005 = 0.5%%, filters leaves/noise)")
    tracking_min_detection_area: float = Field(default=0.0005, ge=0.0, le=0.5, description="Relaxed min area used instead of min_detection_area while the PTZ tracker is already TRACKING. A subject being followed routinely shrinks below min_detection_area in the wide view; dropping it there starves the PTZ controller and makes it return to patrol on a target it can still see. Also the floor for a box that overlaps the subject of an open event, so an animal that sits down or walks away keeps its clip going.")
    blur_threshold: float = Field(default=50.0, ge=0.0, le=1000.0, description="Laplacian variance below this value = blurry frame, skip detection. Applies only to a camera the PTZ tracker moves, within 3 s of a move. 0 = disabled. 50-100 works for most PTZ cameras.")
    ptz_settle_time: float = Field(default=0.5, ge=0.0, le=5.0, description="Seconds to wait after PTZ movement before processing detections (0 = disabled)")


class PTZTrackingSettings(BaseModel):
    """Settings for PTZ auto-tracking (follow detected objects with zoom camera).

    Optimized defaults for real-time tracking with split-model architecture:
    - update_interval: 0.1s (10 updates/sec) for responsive tracking
    - smoothing: 0.15 for faster response with minimal jitter
    - patrol_return_delay: 5.0s before returning to patrol
    """
    enabled: bool = Field(default=False, description="Enable PTZ auto-tracking")
    target_camera_id: Optional[str] = Field(default=None, description="Camera ID to send PTZ commands to (for linked cameras)")
    self_track: bool = Field(default=False, description="Enable self-tracking (camera centers on its own detections)")
    # Multi-camera tracking: allow target camera's detections to take over tracking
    multi_camera_tracking: bool = Field(default=True, description="When enabled, target camera (cam2) detections can take over tracking for finer control")
    target_fill_pct: float = Field(default=0.6, ge=0.1, le=0.95, description="Target object fill percentage (0.6 = 60%)")
    min_detection_area: float = Field(default=0.005, ge=0.0, le=0.1, description="Ignore detections smaller than this fraction of frame (0.005 = 0.5%, filters leaves/noise)")
    pan_scale: float = Field(default=0.8, ge=0.1, le=2.0, description="PTZ pan range as fraction of wide-angle FOV")
    tilt_scale: float = Field(default=0.6, ge=0.1, le=2.0, description="PTZ tilt range as fraction of wide-angle FOV")
    # Optimized for real-time tracking with YOLO backend (~50-150ms inference)
    smoothing: float = Field(default=0.15, ge=0.0, le=0.9, description="Movement smoothing (0=instant, 0.9=very smooth). Lower = faster response")
    update_interval: float = Field(default=0.1, ge=0.05, le=2.0, description="Seconds between PTZ updates (0.1 = 10 updates/sec)")
    # Calibration offsets (where PTZ 0,0 appears on wide-angle, as fraction)
    pan_center_x: float = Field(default=0.5, ge=0.0, le=1.0, description="X position where PTZ center appears on wide-angle")
    tilt_center_y: float = Field(default=0.5, ge=0.0, le=1.0, description="Y position where PTZ center appears on wide-angle")
    # Patrol mode settings (scan for objects)
    patrol_enabled: bool = Field(default=True, description="Enable patrol sweep when no objects detected")
    patrol_speed: float = Field(default=0.08, ge=0.02, le=1.0, description="Patrol sweep speed (0.08 = slow for detection)")
    patrol_return_delay: float = Field(default=5.0, ge=0.5, le=30.0, description="Seconds without a sighting from ANY contributing camera before returning to patrol. 2.0 was routinely tripped by the PTZ settle gate on a target that was still visible.")
    # Track mode settings (follow detected objects)
    track_enabled: bool = Field(default=True, description="Enable tracking of detected objects")
    # Preset-based patrol (instead of continuous sweep)
    patrol_presets: list[str] = Field(default=[], description="List of preset tokens/names for patrol. Empty = continuous sweep")
    patrol_dwell_time: float = Field(default=10.0, ge=2.0, le=120.0, description="Seconds to stay at each preset position")
    # Tracking-stability tunables
    move_min_duration: float = Field(default=0.6, ge=0.0, le=5.0, description="Minimum seconds a tracking ContinuousMove is allowed to run before it can be ptz_stop'd by a no-detection tick")
    tracking_step_duration: float = Field(default=0.35, ge=0.05, le=2.0, description="Maximum seconds a tracking ContinuousMove pulse may run before an automatic Stop is issued")
    low_fill_threshold: float = Field(default=0.30, ge=0.01, le=1.0, description="Apply low-fill pan/tilt velocity caps when target max dimension is below this fraction of the frame")
    low_fill_velocity_cap: float = Field(default=0.30, ge=0.01, le=1.0, description="Maximum pan/tilt velocity for low-fill targets, before scaling by overall offset magnitude")
    low_fill_cap_full_offset: float = Field(default=0.40, ge=0.01, le=1.0, description="Overall offset magnitude at which the low-fill velocity cap reaches its full configured value")
    cam1_fallback_delay: float = Field(default=3.0, ge=0.0, le=30.0, description="Seconds to suppress source-camera (cam1) repositioning after the target camera (cam2) drove tracking. Prevents miscalibrated cam1->cam2 swings on a single dropped cam2 frame")
    # Visibility-aware cam1/cam2 recovery after cam2 loses a target.
    zoom_fov_calibration_path: Optional[str] = Field(default="config/zoom_fov_calibration.json", description="Optional JSON calibration mapping cam2 zoom FOV into cam1 coordinates. Used to decide whether cam2 should recenter, zoom out, or cautiously zoom in after target loss")
    visibility_recovery_enabled: bool = Field(default=True, description="Use cam1 plus zoom FOV calibration to recover when cam2 recently had a target but loses it")
    visibility_recovery_min_overlap: float = Field(default=0.50, ge=0.0, le=1.0, description="Minimum fraction of the cam1 detection that must overlap predicted cam2 FOV to count as visible")
    visibility_recovery_edge_margin: float = Field(default=0.12, ge=0.0, le=0.5, description="Fraction of predicted cam2 FOV treated as edge risk; edge targets trigger recenter + zoom-out")
    visibility_recovery_zoom_out_velocity: float = Field(default=-0.25, ge=-1.0, le=0.0, description="Zoom-out velocity used during cam2 lost-target recovery")
    visibility_recovery_zoom_in_velocity: float = Field(default=0.15, ge=0.0, le=1.0, description="Small zoom-in velocity used only when cam1 says the target is centered in cam2 FOV and cam2 is still wide")
    visibility_recovery_zoom_in_max_zoom: float = Field(default=0.35, ge=0.0, le=1.0, description="Maximum current cam2 zoom where recovery may choose zoom-in instead of zoom-out")
    visibility_recovery_zoom_in_fill_threshold: float = Field(default=0.03, ge=0.0, le=0.5, description="Maximum cam1 bbox fill considered tiny enough to justify cautious recovery zoom-in")
    visibility_recovery_velocity_cap: float = Field(default=0.20, ge=0.01, le=1.0, description="Maximum pan/tilt velocity for calibrated visibility recovery pulses")
    # Investigate mode (opt-in): zoom in on small wide-angle candidates
    investigate_enabled: bool = Field(default=False, description="If true, small cam1 detections (below min_detection_area but above investigate_min_area) cause cam2 to slew over and try to confirm with its zoom view")
    investigate_min_area: float = Field(default=0.0005, ge=0.0, le=0.1, description="Minimum normalized area for a cam1 detection to be treated as an investigate candidate (0.0005 = 0.05% of frame)")
    investigate_timeout: float = Field(default=4.0, ge=0.5, le=30.0, description="Seconds cam2 has to confirm an investigate candidate before it's marked as a reject")
    investigate_cooldown: float = Field(default=30.0, ge=0.0, le=600.0, description="Seconds to suppress re-investigating a previously-rejected cam1 location")
    investigate_cooldown_radius: float = Field(default=0.10, ge=0.0, le=0.5, description="Normalized radius around a rejected location considered the same spot")


class CameraNotificationSettings(BaseModel):
    priority: int = 0
    sound: Optional[str] = None
    destinations: Optional[List[str]] = Field(default=None, description="Ids of the Pushover destinations this camera alerts. Absent: every destination. Empty list: none.")

    @field_validator("destinations")
    @classmethod
    def _clean_destination_ids(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        out: List[str] = []
        for item in value:
            safe = str(item if item is not None else "").strip()
            if not safe:
                raise ValueError("destination ids cannot be empty")
            if safe in out:
                raise ValueError(f"destination '{safe}' appears twice")
            out.append(safe)
        return out


class CameraConfig(BaseModel):
    id: str
    name: str
    location: Optional[str] = None
    rtsp: RTSPSettings
    onvif: Optional[ONVIFSettings] = None
    thresholds: ThresholdSettings = ThresholdSettings()
    detect_enabled: bool = True
    ptz_tracking: PTZTrackingSettings = PTZTrackingSettings()
    include_species: List[str] = Field(default_factory=list)
    exclude_species: List[str] = Field(default_factory=list)
    notification: CameraNotificationSettings = CameraNotificationSettings()
    inference_max_width: int = Field(
        default=0,
        ge=0,
        description=(
            "If > 0 and the captured frame is wider than this, downscale the "
            "frame (preserving aspect ratio) before realtime inference. "
            "Returned bboxes are scaled back to the original frame size, so "
            "trackers, clips, and PTZ math are unaffected. Use to reduce CPU "
            "prep cost on a high-resolution stream. Note: most of the GPU "
            "inference cost is fixed (the model letterboxes internally), so "
            "savings are modest. 0 = disabled."
        ),
    )

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
    timezone: Optional[str] = None  # e.g., "America/Chicago", "US/Central", "UTC"


class RuntimeConfig(BaseModel):
    general: GeneralSettings
    cameras: List[CameraConfig]

    @field_validator("cameras")
    @classmethod
    def _unique_camera_ids(cls, value: List[CameraConfig]) -> List[CameraConfig]:
        """Two cameras with one id validated, and then everything keyed by id
        (the pipeline's workers, the settings page's save) silently kept one."""
        seen = set()
        for cam in value:
            if cam.id in seen:
                raise ValueError(f"camera id '{cam.id}' is used more than once")
            seen.add(cam.id)
        return value

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
