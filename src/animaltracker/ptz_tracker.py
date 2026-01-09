"""PTZ auto-tracking: Center and zoom on detected objects."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .onvif_client import OnvifClient
    from .detector import Detection

LOGGER = logging.getLogger(__name__)


@dataclass
class PTZCalibration:
    """Calibration mapping between wide-angle pixels and PTZ coordinates.
    
    The wide-angle camera shows a fixed view. The zoom camera can pan/tilt
    within that view. This maps pixel positions to PTZ coordinates.
    """
    # Wide-angle frame dimensions
    frame_width: int = 2560
    frame_height: int = 1440
    
    # PTZ coordinate ranges (camera-specific, typically -1.0 to 1.0)
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    zoom_min: float = 0.0
    zoom_max: float = 1.0
    
    # Offset calibration: where PTZ (0,0) appears on wide-angle frame
    # (as fraction of frame, 0.5 = center)
    pan_center_x: float = 0.5
    tilt_center_y: float = 0.5
    
    # Scale factors: how much of wide-angle FOV the PTZ can cover
    # (1.0 means PTZ range covers entire wide-angle view)
    pan_scale: float = 0.8  # PTZ covers 80% of wide-angle horizontal FOV
    tilt_scale: float = 0.6  # PTZ covers 60% of wide-angle vertical FOV
    
    def pixel_to_ptz(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert wide-angle pixel coordinates to PTZ pan/tilt values.
        
        Args:
            pixel_x: X coordinate on wide-angle frame (0 = left)
            pixel_y: Y coordinate on wide-angle frame (0 = top)
            
        Returns:
            (pan, tilt) values for PTZ absolute positioning
        """
        # Normalize pixel to 0-1 range
        norm_x = pixel_x / self.frame_width
        norm_y = pixel_y / self.frame_height
        
        # Calculate offset from center
        offset_x = (norm_x - self.pan_center_x) / self.pan_scale
        offset_y = (self.tilt_center_y - norm_y) / self.tilt_scale  # Y inverted (up = positive tilt)
        
        # Map to PTZ range
        pan_range = self.pan_max - self.pan_min
        tilt_range = self.tilt_max - self.tilt_min
        
        pan = offset_x * pan_range
        tilt = offset_y * tilt_range
        
        # Clamp to valid range
        pan = max(self.pan_min, min(self.pan_max, pan))
        tilt = max(self.tilt_min, min(self.tilt_max, tilt))
        
        return pan, tilt
    
    def bbox_to_zoom(self, bbox: List[float], target_fill: float = 0.6) -> float:
        """Calculate zoom level to make bounding box fill target percentage of frame.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box in pixels
            target_fill: Target fill percentage (0.6 = 60%)
            
        Returns:
            Zoom value (0.0 to 1.0)
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate current fill ratio (using larger dimension)
        width_fill = bbox_width / self.frame_width
        height_fill = bbox_height / self.frame_height
        current_fill = max(width_fill, height_fill)
        
        if current_fill <= 0:
            return 0.0
        
        # Calculate zoom needed
        # More zoom = higher value, less zoom = lower value
        zoom_factor = target_fill / current_fill
        
        # Map to zoom range (logarithmic feels more natural)
        # zoom_factor of 1 = no zoom needed
        # zoom_factor > 1 = need to zoom in
        import math
        zoom = math.log2(max(1.0, zoom_factor)) / 4.0  # Divide by 4 to normalize
        zoom = max(self.zoom_min, min(self.zoom_max, zoom))
        
        return zoom


from enum import Enum


class PTZMode(Enum):
    """PTZ operating mode."""
    IDLE = "idle"           # Not active
    PATROL = "patrol"       # Scanning for objects
    TRACKING = "tracking"   # Following detected object


@dataclass
class PTZTracker:
    """Auto-tracking controller that moves PTZ to follow detections."""
    
    onvif_client: 'OnvifClient'
    profile_token: str
    calibration: PTZCalibration = field(default_factory=PTZCalibration)
    
    # Tracking behavior
    target_fill_pct: float = 0.6  # Target 60% frame fill
    min_move_threshold: float = 0.05  # Don't move if offset < 5% of range
    smoothing: float = 0.3  # Movement smoothing (0=instant, 1=no movement)
    update_interval: float = 0.2  # Seconds between PTZ updates
    
    # Patrol settings
    patrol_enabled: bool = True  # Enable patrol when no detections
    patrol_speed: float = 0.15  # Patrol pan speed (slow sweep)
    patrol_tilt: float = 0.0    # Tilt position during patrol
    patrol_zoom: float = 0.0    # Zoom level during patrol (wide)
    patrol_return_delay: float = 3.0  # Seconds after losing object before returning to patrol
    
    # Preset-based patrol
    patrol_presets: list = field(default_factory=list)  # List of preset tokens
    patrol_dwell_time: float = 10.0  # Seconds at each preset
    _preset_tokens: list = field(default_factory=list, init=False)  # Resolved preset tokens
    _current_preset_index: int = field(default=0, init=False)
    _preset_arrival_time: float = field(default=0.0, init=False)
    
    # State
    _last_update: float = field(default=0.0, init=False)
    _target_pan: float = field(default=0.0, init=False)
    _target_tilt: float = field(default=0.0, init=False)
    _target_zoom: float = field(default=0.0, init=False)
    _tracking_active: bool = field(default=False, init=False)
    _mode: PTZMode = field(default=PTZMode.IDLE, init=False)
    _patrol_direction: int = field(default=1, init=False)  # 1 = right, -1 = left
    _last_detection_time: float = field(default=0.0, init=False)
    _patrol_reverse_time: float = field(default=0.0, init=False)
    
    def _resolve_presets(self) -> None:
        """Resolve preset names to tokens."""
        if not self.patrol_presets:
            return
            
        try:
            available = self.onvif_client.ptz_get_presets(self.profile_token)
            preset_map = {}
            for p in available:
                if p.get('token'):
                    preset_map[p['token']] = p['token']
                    if p.get('name'):
                        preset_map[p['name']] = p['token']
            
            self._preset_tokens = []
            for preset in self.patrol_presets:
                if preset in preset_map:
                    self._preset_tokens.append(preset_map[preset])
                else:
                    LOGGER.warning("Preset '%s' not found on camera", preset)
            
            if self._preset_tokens:
                LOGGER.info("Patrol will use %d presets: %s", 
                           len(self._preset_tokens), self._preset_tokens)
            else:
                LOGGER.warning("No valid presets found, falling back to continuous sweep")
                
        except Exception as e:
            LOGGER.error("Failed to resolve presets: %s", e)
            self._preset_tokens = []
    
    def start_tracking(self) -> None:
        """Enable auto-tracking with patrol mode."""
        self._tracking_active = True
        
        # Resolve presets if configured
        if self.patrol_presets:
            self._resolve_presets()
        
        if self.patrol_enabled:
            self._mode = PTZMode.PATROL
            if self._preset_tokens:
                LOGGER.info("PTZ preset patrol enabled - cycling %d positions", len(self._preset_tokens))
                # Go to first preset
                self._goto_current_preset()
            else:
                LOGGER.info("PTZ patrol mode enabled - continuous sweep")
        else:
            self._mode = PTZMode.IDLE
            LOGGER.info("PTZ tracking enabled (patrol disabled)")
    
    def _goto_current_preset(self) -> None:
        """Move to current preset in the patrol sequence."""
        if not self._preset_tokens:
            return
        try:
            preset = self._preset_tokens[self._current_preset_index]
            self.onvif_client.ptz_goto_preset(self.profile_token, preset, speed=0.3)
            self._preset_arrival_time = time.time()
            LOGGER.info("Moving to patrol preset %d/%d: %s", 
                       self._current_preset_index + 1, len(self._preset_tokens), preset)
        except Exception as e:
            LOGGER.error("Failed to go to preset: %s", e)
    
    def stop_tracking(self) -> None:
        """Disable auto-tracking."""
        self._tracking_active = False
        self._mode = PTZMode.IDLE
        try:
            self.onvif_client.ptz_stop(self.profile_token)
        except Exception:
            pass
        LOGGER.info("PTZ tracking disabled")
    
    def get_mode(self) -> str:
        """Get current PTZ mode as string."""
        return self._mode.value
    
    def _do_patrol(self) -> None:
        """Execute patrol pattern - either preset-based or continuous sweep."""
        now = time.time()
        
        # Preset-based patrol
        if self._preset_tokens:
            # Check if dwell time has elapsed
            if now - self._preset_arrival_time > self.patrol_dwell_time:
                # Move to next preset
                self._current_preset_index = (self._current_preset_index + 1) % len(self._preset_tokens)
                self._goto_current_preset()
            return
        
        # Continuous sweep patrol (fallback if no presets)
        # At very slow speed 0.08, need ~90 seconds to cover full pan range
        sweep_duration = 90.0  # seconds per sweep direction
        
        if now - self._patrol_reverse_time > sweep_duration:
            self._patrol_direction *= -1
            self._patrol_reverse_time = now
            LOGGER.info("Patrol reversing direction: %s", 
                        "right" if self._patrol_direction > 0 else "left")
        
        # Pan sweep at patrol speed (0.08 = very slow, good for small/distant animals)
        pan_vel = self.patrol_speed * self._patrol_direction
        
        try:
            self.onvif_client.ptz_move(
                self.profile_token,
                pan_vel,
                0.0,  # No tilt during patrol
                0.0   # No zoom change during patrol
            )
        except Exception as e:
            LOGGER.error("Patrol move error: %s", e)
    
    def update_calibration(self, pan_scale: float, tilt_scale: float, 
                           pan_center_x: float, tilt_center_y: float) -> None:
        """Update calibration parameters (e.g., from auto-calibration results)."""
        self.calibration.pan_scale = pan_scale
        self.calibration.tilt_scale = tilt_scale
        self.calibration.pan_center_x = pan_center_x
        self.calibration.tilt_center_y = tilt_center_y
        LOGGER.info(
            "PTZ calibration updated: pan_scale=%.3f, tilt_scale=%.3f, center=(%.3f, %.3f)",
            pan_scale, tilt_scale, pan_center_x, tilt_center_y
        )
    
    def update(self, detections: List['Detection'], frame_width: int, frame_height: int) -> bool:
        """Process detections and move PTZ if needed.
        
        State machine:
        - PATROL: Sweeping to find objects. On detection -> TRACKING
        - TRACKING: Following object. On lost object -> wait, then PATROL
        - IDLE: Tracking disabled
        
        Args:
            detections: List of Detection objects from wide-angle camera
            frame_width: Width of the detection frame
            frame_height: Height of the detection frame
            
        Returns:
            True if PTZ was moved, False otherwise
        """
        if not self._tracking_active:
            return False
        
        # Rate limit updates
        now = time.time()
        if now - self._last_update < self.update_interval:
            return False
        
        self._last_update = now
        
        # Handle state transitions based on detections
        if detections:
            # We have detections - switch to tracking mode
            self._last_detection_time = now
            
            if self._mode != PTZMode.TRACKING:
                self._mode = PTZMode.TRACKING
                LOGGER.info("PTZ switching to TRACKING mode - object detected")
            
            return self._do_tracking(detections, frame_width, frame_height)
        else:
            # No detections
            if self._mode == PTZMode.TRACKING:
                # Was tracking, check if we should return to patrol
                time_since_detection = now - self._last_detection_time
                
                if time_since_detection > self.patrol_return_delay:
                    # Object lost for long enough, return to patrol
                    if self.patrol_enabled:
                        self._mode = PTZMode.PATROL
                        LOGGER.info("PTZ returning to PATROL mode - object lost for %.1fs", 
                                   time_since_detection)
                    else:
                        # Stop and wait
                        self._mode = PTZMode.IDLE
                        try:
                            self.onvif_client.ptz_stop(self.profile_token)
                        except Exception:
                            pass
                else:
                    # Still within delay, stop and wait for object to reappear
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception:
                        pass
                    return False
            
            if self._mode == PTZMode.PATROL:
                self._do_patrol()
                return True
            
            return False
    
    def _do_tracking(self, detections: List['Detection'], frame_width: int, frame_height: int) -> bool:
        """Execute object tracking logic."""
        # Update calibration with actual frame size
        self.calibration.frame_width = frame_width
        self.calibration.frame_height = frame_height
        
        # Find best detection to track (highest confidence)
        best = max(detections, key=lambda d: d.confidence)
        bbox = best.bbox
        
        # Calculate bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert to PTZ coordinates
        target_pan, target_tilt = self.calibration.pixel_to_ptz(center_x, center_y)
        target_zoom = self.calibration.bbox_to_zoom(bbox, self.target_fill_pct)
        
        # Apply smoothing
        self._target_pan = self._target_pan * self.smoothing + target_pan * (1 - self.smoothing)
        self._target_tilt = self._target_tilt * self.smoothing + target_tilt * (1 - self.smoothing)
        self._target_zoom = self._target_zoom * self.smoothing + target_zoom * (1 - self.smoothing)
        
        # Calculate how far we need to move (normalized center offset)
        norm_center_x = center_x / frame_width
        norm_center_y = center_y / frame_height
        
        # How far from center? (0.5, 0.5) = centered
        offset_x = norm_center_x - 0.5  # Positive = object is right of center
        offset_y = 0.5 - norm_center_y  # Positive = object is above center (inverted Y)
        
        # Calculate offset magnitude
        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5
        
        # Only move if offset is significant
        if offset_magnitude < self.min_move_threshold:
            # Object is centered enough, stop movement
            try:
                self.onvif_client.ptz_stop(self.profile_token)
            except Exception:
                pass
            return False
        
        # Non-linear velocity curve:
        # - Small offset (< 0.1): slow tracking to maintain center
        # - Medium offset (0.1-0.25): moderate speed
        # - Large offset (> 0.25): fast catch-up mode
        
        def velocity_curve(offset: float) -> float:
            """Convert offset to velocity with non-linear response."""
            abs_offset = abs(offset)
            
            if abs_offset < 0.1:
                # Fine tracking: slow and smooth
                speed = abs_offset * 3.0  # Max 0.3 at 0.1 offset
            elif abs_offset < 0.25:
                # Normal tracking: moderate speed
                speed = 0.3 + (abs_offset - 0.1) * 3.0  # 0.3 to 0.75
            else:
                # Catch-up mode: fast movement
                speed = 0.75 + (abs_offset - 0.25) * 1.0  # 0.75 to 1.0
            
            # Apply sign and clamp
            return max(-1.0, min(1.0, speed if offset >= 0 else -speed))
        
        pan_velocity = velocity_curve(offset_x)
        tilt_velocity = velocity_curve(offset_y)
        
        # Calculate zoom velocity based on current vs target fill
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        current_fill = max(bbox_width / frame_width, bbox_height / frame_height)
        
        if current_fill > 0:
            fill_error = self.target_fill_pct - current_fill
            # Slower zoom adjustments - zoom changes are more jarring
            zoom_velocity = fill_error * 1.5
            zoom_velocity = max(-0.3, min(0.3, zoom_velocity))
        else:
            zoom_velocity = 0.0
        
        LOGGER.debug(
            "PTZ tracking %s: offset=(%.2f, %.2f) mag=%.2f, vel=(%.2f, %.2f, %.2f), fill=%.1f%%",
            best.species, offset_x, offset_y, offset_magnitude, 
            pan_velocity, tilt_velocity, zoom_velocity, current_fill * 100
        )
        
        try:
            self.onvif_client.ptz_move(
                self.profile_token,
                pan_velocity,
                tilt_velocity,
                zoom_velocity
            )
            return True
        except Exception as e:
            LOGGER.error("PTZ tracking error: %s", e)
            return False
    
    def center_on_bbox(self, bbox: List[float], frame_width: int, frame_height: int, auto_zoom: bool = True) -> None:
        """Immediately center PTZ on a bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            frame_width: Frame width for calibration
            frame_height: Frame height for calibration
            auto_zoom: Whether to also adjust zoom
        """
        self.calibration.frame_width = frame_width
        self.calibration.frame_height = frame_height
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        pan, tilt = self.calibration.pixel_to_ptz(center_x, center_y)
        zoom = self.calibration.bbox_to_zoom(bbox, self.target_fill_pct) if auto_zoom else 0.0
        
        LOGGER.info("PTZ centering on bbox: pan=%.3f, tilt=%.3f, zoom=%.3f", pan, tilt, zoom)
        self.onvif_client.ptz_move_absolute(self.profile_token, pan, tilt, zoom)
        
        self._target_pan = pan
        self._target_tilt = tilt
        self._target_zoom = zoom


def create_ptz_tracker(
    onvif_client: 'OnvifClient',
    profile_token: str,
    config: Optional[Dict] = None
) -> PTZTracker:
    """Create a PTZ tracker with optional configuration.
    
    Args:
        onvif_client: ONVIF client for PTZ control
        profile_token: ONVIF profile token
        config: Optional dict with calibration/tracking settings:
            - pan_scale: PTZ pan range as fraction of wide-angle FOV
            - tilt_scale: PTZ tilt range as fraction of wide-angle FOV
            - target_fill_pct: Target object fill percentage (default 0.6)
            - smoothing: Movement smoothing factor (default 0.3)
            - patrol_enabled: Enable patrol mode when no detections (default True)
            - patrol_speed: Patrol sweep speed (default 0.15)
            - patrol_return_delay: Seconds to wait before returning to patrol (default 3.0)
            - patrol_presets: List of preset tokens/names for patrol (default [])
            - patrol_dwell_time: Seconds to stay at each preset (default 10.0)
            
    Returns:
        Configured PTZTracker instance
    """
    config = config or {}
    
    calibration = PTZCalibration(
        pan_scale=config.get('pan_scale', 0.8),
        tilt_scale=config.get('tilt_scale', 0.6),
        pan_center_x=config.get('pan_center_x', 0.5),
        tilt_center_y=config.get('tilt_center_y', 0.5),
    )
    
    tracker = PTZTracker(
        onvif_client=onvif_client,
        profile_token=profile_token,
        calibration=calibration,
        target_fill_pct=config.get('target_fill_pct', 0.6),
        smoothing=config.get('smoothing', 0.3),
        update_interval=config.get('update_interval', 0.2),
        patrol_enabled=config.get('patrol_enabled', True),
        patrol_speed=config.get('patrol_speed', 0.15),
        patrol_return_delay=config.get('patrol_return_delay', 3.0),
        patrol_presets=config.get('patrol_presets', []),
        patrol_dwell_time=config.get('patrol_dwell_time', 10.0),
    )
    
    return tracker
