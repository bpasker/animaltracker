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
    
    # State
    _last_update: float = field(default=0.0, init=False)
    _target_pan: float = field(default=0.0, init=False)
    _target_tilt: float = field(default=0.0, init=False)
    _target_zoom: float = field(default=0.0, init=False)
    _tracking_active: bool = field(default=False, init=False)
    
    def start_tracking(self) -> None:
        """Enable auto-tracking."""
        self._tracking_active = True
        LOGGER.info("PTZ auto-tracking enabled")
    
    def stop_tracking(self) -> None:
        """Disable auto-tracking."""
        self._tracking_active = False
        LOGGER.info("PTZ auto-tracking disabled")
    
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
        
        Args:
            detections: List of Detection objects from wide-angle camera
            frame_width: Width of the detection frame
            frame_height: Height of the detection frame
            
        Returns:
            True if PTZ was moved, False otherwise
        """
        if not self._tracking_active or not detections:
            return False
        
        # Rate limit updates
        now = time.time()
        if now - self._last_update < self.update_interval:
            return False
        
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
        
        # Check if movement is significant enough
        try:
            current = self.onvif_client.ptz_get_position(self.profile_token)
            pan_diff = abs(self._target_pan - current['pan'])
            tilt_diff = abs(self._target_tilt - current['tilt'])
            zoom_diff = abs(self._target_zoom - current['zoom'])
            
            if pan_diff < self.min_move_threshold and tilt_diff < self.min_move_threshold and zoom_diff < self.min_move_threshold:
                return False
            
            # Move PTZ
            LOGGER.debug(
                "PTZ tracking: moving to pan=%.3f, tilt=%.3f, zoom=%.3f (tracking %s)",
                self._target_pan, self._target_tilt, self._target_zoom, best.species
            )
            self.onvif_client.ptz_move_absolute(
                self.profile_token,
                self._target_pan,
                self._target_tilt,
                self._target_zoom
            )
            self._last_update = now
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
    )
    
    return tracker
