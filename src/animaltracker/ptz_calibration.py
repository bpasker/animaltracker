"""PTZ Auto-Calibration: Automatically map zoom camera FOV to wide-angle coordinates."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from .onvif_client import OnvifClient
    from .pipeline import StreamWorker

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """A single calibration measurement."""
    pan: float
    tilt: float
    zoom: float
    # Where the zoom camera center appears on wide-angle (as fraction 0-1)
    wide_x: float
    wide_y: float
    confidence: float  # Match quality


@dataclass
class CalibrationResult:
    """Result of auto-calibration."""
    pan_scale: float
    tilt_scale: float
    pan_center_x: float
    tilt_center_y: float
    points: List[CalibrationPoint]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'pan_scale': self.pan_scale,
            'tilt_scale': self.tilt_scale,
            'pan_center_x': self.pan_center_x,
            'tilt_center_y': self.tilt_center_y,
            'num_points': len(self.points),
            'error': self.error,
        }


class PTZAutoCalibrator:
    """Automatically calibrate PTZ-to-pixel mapping using image matching."""
    
    def __init__(
        self,
        onvif_client: 'OnvifClient',
        profile_token: str,
        min_match_confidence: float = 0.3,
    ):
        self.onvif_client = onvif_client
        self.profile_token = profile_token
        self.min_match_confidence = min_match_confidence
        
        # Feature detector for image matching
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def find_zoom_in_wide(
        self, 
        wide_frame: np.ndarray, 
        zoom_frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Find where zoom camera view appears in wide-angle frame.
        
        Uses ORB feature matching to locate the zoom view within the wide view.
        
        Args:
            wide_frame: Frame from wide-angle camera
            zoom_frame: Frame from zoom camera
            
        Returns:
            (center_x, center_y, confidence) as fractions of wide frame dimensions,
            or None if match failed
        """
        # Convert to grayscale
        wide_gray = cv2.cvtColor(wide_frame, cv2.COLOR_BGR2GRAY)
        zoom_gray = cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize zoom frame to approximate expected size in wide frame
        # (zoom view is typically a small portion of wide view)
        scale_factors = [0.3, 0.4, 0.5, 0.2]  # Try different scales
        
        best_match = None
        best_confidence = 0
        
        for scale in scale_factors:
            scaled_zoom = cv2.resize(
                zoom_gray, 
                None, 
                fx=scale, 
                fy=scale, 
                interpolation=cv2.INTER_AREA
            )
            
            # Detect features
            kp_wide, desc_wide = self.orb.detectAndCompute(wide_gray, None)
            kp_zoom, desc_zoom = self.orb.detectAndCompute(scaled_zoom, None)
            
            if desc_wide is None or desc_zoom is None or len(kp_wide) < 10 or len(kp_zoom) < 10:
                continue
            
            # Match features
            try:
                matches = self.bf.match(desc_zoom, desc_wide)
            except cv2.error:
                continue
            
            if len(matches) < 4:
                continue
            
            # Sort by distance (quality)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use top matches to find homography
            good_matches = matches[:min(50, len(matches))]
            
            if len(good_matches) < 4:
                continue
            
            # Get matched point coordinates
            src_pts = np.float32([kp_zoom[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_wide[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except cv2.error:
                continue
            
            if H is None:
                continue
            
            # Calculate inlier ratio as confidence
            inliers = mask.ravel().sum() if mask is not None else 0
            confidence = inliers / len(good_matches)
            
            if confidence > best_confidence:
                # Find center of zoom view in wide coordinates
                h, w = scaled_zoom.shape
                center_zoom = np.float32([[w/2, h/2]]).reshape(-1, 1, 2)
                center_wide = cv2.perspectiveTransform(center_zoom, H)
                
                cx = center_wide[0, 0, 0] / wide_frame.shape[1]
                cy = center_wide[0, 0, 1] / wide_frame.shape[0]
                
                # Sanity check
                if 0 <= cx <= 1 and 0 <= cy <= 1:
                    best_match = (cx, cy, confidence)
                    best_confidence = confidence
        
        if best_match and best_match[2] >= self.min_match_confidence:
            return best_match
        return None
    
    def calibrate(
        self,
        get_wide_frame: callable,
        get_zoom_frame: callable,
        num_points: int = 9,
        settle_time: float = 1.0,
    ) -> CalibrationResult:
        """Run auto-calibration by moving PTZ and matching images.
        
        Args:
            get_wide_frame: Function that returns current wide-angle frame
            get_zoom_frame: Function that returns current zoom camera frame
            num_points: Number of calibration points (9 = 3x3 grid)
            settle_time: Seconds to wait after PTZ move before capturing
            
        Returns:
            CalibrationResult with computed parameters
        """
        points: List[CalibrationPoint] = []
        
        # Generate grid of PTZ positions to test
        if num_points == 9:
            # 3x3 grid
            pan_values = [-0.5, 0.0, 0.5]
            tilt_values = [-0.3, 0.0, 0.3]
        elif num_points == 5:
            # Cross pattern
            pan_values = [-0.5, 0.0, 0.5, 0.0, 0.0]
            tilt_values = [0.0, 0.0, 0.0, -0.3, 0.3]
        else:
            # Just center and corners
            pan_values = [-0.5, 0.0, 0.5]
            tilt_values = [0.0]
        
        LOGGER.info("Starting PTZ auto-calibration with %d points", num_points)
        
        # First, test if PTZ actually works by doing a small move
        try:
            initial_pos = self.onvif_client.ptz_get_position(self.profile_token)
            LOGGER.info("Initial PTZ position: %s", initial_pos)
            
            # Try a relative move first (more compatible)
            LOGGER.info("Testing PTZ with relative move...")
            self.onvif_client.ptz_move_relative(self.profile_token, 0.1, 0.0, 0.0)
            time.sleep(1.5)
            
            test_pos = self.onvif_client.ptz_get_position(self.profile_token)
            LOGGER.info("After test move: %s", test_pos)
            
            if test_pos['pan'] == initial_pos['pan'] and test_pos['tilt'] == initial_pos['tilt']:
                LOGGER.warning("PTZ did not move! Trying continuous move instead...")
                # Try continuous move as fallback
                self.onvif_client.ptz_move(self.profile_token, 0.5, 0.0, 0.0)
                time.sleep(0.5)
                self.onvif_client.ptz_stop(self.profile_token)
                time.sleep(1.0)
                test_pos2 = self.onvif_client.ptz_get_position(self.profile_token)
                LOGGER.info("After continuous move: %s", test_pos2)
                
                if test_pos2['pan'] == initial_pos['pan']:
                    LOGGER.error("PTZ still not moving! Check ONVIF profile token.")
                    return CalibrationResult(
                        pan_scale=0.8, tilt_scale=0.6,
                        pan_center_x=0.5, tilt_center_y=0.5,
                        points=[],
                        error=f"PTZ not responding. Profile token '{self.profile_token}' may not support PTZ."
                    )
            
            # Return to center before starting calibration
            self.onvif_client.ptz_move_absolute(self.profile_token, 0.0, 0.0, 0.0)
            time.sleep(1.0)
            
        except Exception as e:
            LOGGER.error("PTZ test failed: %s", e)
            return CalibrationResult(
                pan_scale=0.8, tilt_scale=0.6,
                pan_center_x=0.5, tilt_center_y=0.5,
                points=[],
                error=f"PTZ test failed: {e}"
            )
        
        for pan in pan_values:
            for tilt in tilt_values:
                try:
                    # Move PTZ to position
                    LOGGER.info("Moving PTZ to pan=%.2f, tilt=%.2f", pan, tilt)
                    self.onvif_client.ptz_move_absolute(
                        self.profile_token, pan, tilt, 0.0
                    )
                    
                    # Wait for camera to settle
                    time.sleep(settle_time)
                    
                    # Get actual position (may differ slightly)
                    actual_pos = self.onvif_client.ptz_get_position(self.profile_token)
                    actual_pan = actual_pos['pan']
                    actual_tilt = actual_pos['tilt']
                    actual_zoom = actual_pos['zoom']
                    LOGGER.info("PTZ actual position: pan=%.3f, tilt=%.3f, zoom=%.3f", 
                               actual_pan, actual_tilt, actual_zoom)
                    
                    # Capture frames
                    wide_frame = get_wide_frame()
                    zoom_frame = get_zoom_frame()
                    
                    if wide_frame is None:
                        LOGGER.warning("Wide frame is None at pan=%.2f, tilt=%.2f", pan, tilt)
                        continue
                    if zoom_frame is None:
                        LOGGER.warning("Zoom frame is None at pan=%.2f, tilt=%.2f", pan, tilt)
                        continue
                    
                    LOGGER.info("Got frames: wide=%s, zoom=%s", wide_frame.shape, zoom_frame.shape)
                    
                    # Find zoom view in wide frame
                    match = self.find_zoom_in_wide(wide_frame, zoom_frame)
                    
                    if match:
                        wide_x, wide_y, confidence = match
                        points.append(CalibrationPoint(
                            pan=actual_pan,
                            tilt=actual_tilt,
                            zoom=actual_zoom,
                            wide_x=wide_x,
                            wide_y=wide_y,
                            confidence=confidence,
                        ))
                        LOGGER.info(
                            "Calibration point: PTZ(%.3f, %.3f) -> Wide(%.3f, %.3f) conf=%.2f",
                            actual_pan, actual_tilt, wide_x, wide_y, confidence
                        )
                    else:
                        LOGGER.warning("No match found at pan=%.2f, tilt=%.2f", pan, tilt)
                        
                except Exception as e:
                    LOGGER.error("Calibration error at pan=%.2f, tilt=%.2f: %s", pan, tilt, e)
        
        # Return to center
        try:
            self.onvif_client.ptz_move_absolute(self.profile_token, 0.0, 0.0, 0.0)
        except Exception:
            pass
        
        if len(points) < 3:
            return CalibrationResult(
                pan_scale=0.8,
                tilt_scale=0.6,
                pan_center_x=0.5,
                tilt_center_y=0.5,
                points=points,
                error=f"Not enough calibration points ({len(points)}/3 minimum)"
            )
        
        # Calculate calibration from points
        return self._compute_calibration(points)
    
    def _compute_calibration(self, points: List[CalibrationPoint]) -> CalibrationResult:
        """Compute calibration parameters from measured points using linear regression."""
        
        # Extract data
        pans = np.array([p.pan for p in points])
        tilts = np.array([p.tilt for p in points])
        wide_xs = np.array([p.wide_x for p in points])
        wide_ys = np.array([p.wide_y for p in points])
        weights = np.array([p.confidence for p in points])
        
        # Weighted linear regression for pan -> wide_x
        # wide_x = pan_center_x + pan * pan_scale_inv
        # Solve: wide_x - 0.5 = pan * pan_scale_inv + offset
        
        # For pan: wide_x = mx * pan + bx
        try:
            # Use numpy's polyfit with weights
            pan_coeffs = np.polyfit(pans, wide_xs, 1, w=weights)
            mx, bx = pan_coeffs  # slope and intercept
            
            # For tilt: wide_y = my * tilt + by (note: tilt is inverted)
            tilt_coeffs = np.polyfit(tilts, wide_ys, 1, w=weights)
            my, by = tilt_coeffs
            
            # Convert to our calibration format
            # pan_scale is how much of wide FOV the PTZ range covers
            # If pan goes from -1 to 1, and that moves wide_x from 0.1 to 0.9, scale = 0.4/1 = 0.4
            pan_range = pans.max() - pans.min()
            wide_x_range = wide_xs.max() - wide_xs.min()
            
            tilt_range = tilts.max() - tilts.min()
            wide_y_range = wide_ys.max() - wide_ys.min()
            
            # pan_scale: PTZ range / corresponding wide FOV fraction
            # If pan 1.0 moves wide_x by 0.4, then pan_scale = 1.0 / 0.4 = 2.5
            # But we want it the other way: fraction of wide FOV that PTZ covers
            if pan_range > 0 and wide_x_range > 0:
                pan_scale = (pan_range / 2.0) / (wide_x_range / 2.0)  # Normalize to full range
            else:
                pan_scale = 0.8
            
            if tilt_range > 0 and wide_y_range > 0:
                tilt_scale = (tilt_range / 2.0) / (wide_y_range / 2.0)
            else:
                tilt_scale = 0.6
            
            # Center: where does PTZ (0,0) appear on wide?
            pan_center_x = bx  # intercept when pan=0
            tilt_center_y = by  # intercept when tilt=0
            
            # Clamp to reasonable values
            pan_scale = max(0.2, min(3.0, pan_scale))
            tilt_scale = max(0.2, min(3.0, tilt_scale))
            pan_center_x = max(0.1, min(0.9, pan_center_x))
            tilt_center_y = max(0.1, min(0.9, tilt_center_y))
            
            LOGGER.info(
                "Calibration computed: pan_scale=%.3f, tilt_scale=%.3f, center=(%.3f, %.3f)",
                pan_scale, tilt_scale, pan_center_x, tilt_center_y
            )
            
            return CalibrationResult(
                pan_scale=pan_scale,
                tilt_scale=tilt_scale,
                pan_center_x=pan_center_x,
                tilt_center_y=tilt_center_y,
                points=points,
            )
            
        except Exception as e:
            LOGGER.error("Calibration computation failed: %s", e)
            return CalibrationResult(
                pan_scale=0.8,
                tilt_scale=0.6,
                pan_center_x=0.5,
                tilt_center_y=0.5,
                points=points,
                error=str(e)
            )


def run_auto_calibration(
    wide_worker: 'StreamWorker',
    zoom_worker: 'StreamWorker',
    num_points: int = 9,
) -> CalibrationResult:
    """Run auto-calibration between two camera workers.
    
    Args:
        wide_worker: StreamWorker for wide-angle camera (detection source)
        zoom_worker: StreamWorker for zoom camera (PTZ target)
        num_points: Number of calibration points
        
    Returns:
        CalibrationResult with computed parameters
    """
    LOGGER.info("Starting auto-calibration: wide=%s, zoom=%s", 
                wide_worker.camera.id, zoom_worker.camera.id)
    
    if not zoom_worker.onvif_client:
        LOGGER.error("Zoom camera has no ONVIF client")
        return CalibrationResult(
            pan_scale=0.8, tilt_scale=0.6,
            pan_center_x=0.5, tilt_center_y=0.5,
            points=[],
            error="Zoom camera has no ONVIF client configured"
        )
    
    if not zoom_worker.onvif_profile_token:
        LOGGER.error("Zoom camera has no ONVIF profile token")
        return CalibrationResult(
            pan_scale=0.8, tilt_scale=0.6,
            pan_center_x=0.5, tilt_center_y=0.5,
            points=[],
            error="Zoom camera has no ONVIF profile token"
        )
    
    LOGGER.info("Using ONVIF profile token: %s", zoom_worker.onvif_profile_token)
    
    # Check if we have frames
    if wide_worker.latest_frame is None:
        LOGGER.error("Wide camera has no frames yet")
        return CalibrationResult(
            pan_scale=0.8, tilt_scale=0.6,
            pan_center_x=0.5, tilt_center_y=0.5,
            points=[],
            error="Wide camera has no frames - is it streaming?"
        )
    
    if zoom_worker.latest_frame is None:
        LOGGER.error("Zoom camera has no frames yet")
        return CalibrationResult(
            pan_scale=0.8, tilt_scale=0.6,
            pan_center_x=0.5, tilt_center_y=0.5,
            points=[],
            error="Zoom camera has no frames - is it streaming?"
        )
    
    calibrator = PTZAutoCalibrator(
        onvif_client=zoom_worker.onvif_client,
        profile_token=zoom_worker.onvif_profile_token,
    )
    
    def get_wide_frame():
        return wide_worker.latest_frame.copy() if wide_worker.latest_frame is not None else None
    
    def get_zoom_frame():
        return zoom_worker.latest_frame.copy() if zoom_worker.latest_frame is not None else None
    
    return calibrator.calibrate(
        get_wide_frame=get_wide_frame,
        get_zoom_frame=get_zoom_frame,
        num_points=num_points,
    )


# ============================================================================
# Zoom FOV Mapping - Maps what area of cam1 is visible in cam2 at different zoom levels
# ============================================================================

@dataclass
class ZoomFOVPoint:
    """A single zoom level FOV measurement.

    Stores the bounding box of what's visible in the zoom camera (cam2)
    as normalized coordinates (0-1) relative to the wide camera (cam1) frame.
    """
    zoom_level: float  # 0.0 = full wide, 1.0 = full zoom
    # Bounding box in wide-angle frame (normalized 0-1)
    x1: float  # Left edge
    y1: float  # Top edge
    x2: float  # Right edge
    y2: float  # Bottom edge
    confidence: float  # Match quality

    @property
    def width(self) -> float:
        """Width of FOV as fraction of wide frame."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of FOV as fraction of wide frame."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Center X as fraction of wide frame."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Center Y as fraction of wide frame."""
        return (self.y1 + self.y2) / 2


@dataclass
class ZoomFOVCalibration:
    """Calibration data mapping zoom levels to visible FOV in wide frame.

    Allows predicting whether a detection in cam1 would be visible in cam2
    at the current zoom level.
    """
    points: List[ZoomFOVPoint] = field(default_factory=list)
    wide_frame_width: int = 0
    wide_frame_height: int = 0
    error: Optional[str] = None

    def get_fov_at_zoom(self, zoom: float) -> Optional[Tuple[float, float, float, float]]:
        """Interpolate FOV bounds at a given zoom level.

        Args:
            zoom: Zoom level (0.0 = wide, 1.0 = full zoom)

        Returns:
            (x1, y1, x2, y2) normalized coordinates, or None if no calibration
        """
        if not self.points:
            return None

        # Sort points by zoom level
        sorted_points = sorted(self.points, key=lambda p: p.zoom_level)

        # Find bracketing points for interpolation
        lower = None
        upper = None

        for p in sorted_points:
            if p.zoom_level <= zoom:
                lower = p
            if p.zoom_level >= zoom and upper is None:
                upper = p

        if lower is None and upper is None:
            return None

        if lower is None:
            # Use lowest point
            return (upper.x1, upper.y1, upper.x2, upper.y2)

        if upper is None:
            # Use highest point
            return (lower.x1, lower.y1, lower.x2, lower.y2)

        if lower.zoom_level == upper.zoom_level:
            return (lower.x1, lower.y1, lower.x2, lower.y2)

        # Linear interpolation
        t = (zoom - lower.zoom_level) / (upper.zoom_level - lower.zoom_level)
        x1 = lower.x1 + t * (upper.x1 - lower.x1)
        y1 = lower.y1 + t * (upper.y1 - lower.y1)
        x2 = lower.x2 + t * (upper.x2 - lower.x2)
        y2 = lower.y2 + t * (upper.y2 - lower.y2)

        return (x1, y1, x2, y2)

    def is_detection_visible(
        self,
        bbox: Tuple[float, float, float, float],
        current_zoom: float,
        min_overlap: float = 0.5,
    ) -> bool:
        """Check if a detection from cam1 would be visible in cam2 at current zoom.

        Args:
            bbox: Detection bounding box (x1, y1, x2, y2) in pixels from cam1
            current_zoom: Current zoom level of cam2 (0.0-1.0)
            min_overlap: Minimum overlap ratio required (0.5 = 50% of detection visible)

        Returns:
            True if detection would be visible in cam2's current FOV
        """
        fov = self.get_fov_at_zoom(current_zoom)
        if fov is None:
            return True  # No calibration, assume visible

        fov_x1, fov_y1, fov_x2, fov_y2 = fov

        # Normalize detection bbox to 0-1
        if self.wide_frame_width == 0 or self.wide_frame_height == 0:
            return True

        det_x1 = bbox[0] / self.wide_frame_width
        det_y1 = bbox[1] / self.wide_frame_height
        det_x2 = bbox[2] / self.wide_frame_width
        det_y2 = bbox[3] / self.wide_frame_height

        # Calculate intersection
        inter_x1 = max(fov_x1, det_x1)
        inter_y1 = max(fov_y1, det_y1)
        inter_x2 = min(fov_x2, det_x2)
        inter_y2 = min(fov_y2, det_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False  # No overlap

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)

        if det_area <= 0:
            return False

        overlap_ratio = inter_area / det_area
        return overlap_ratio >= min_overlap

    def to_dict(self) -> Dict:
        """Serialize calibration for saving."""
        return {
            'wide_frame_width': self.wide_frame_width,
            'wide_frame_height': self.wide_frame_height,
            'points': [
                {
                    'zoom_level': p.zoom_level,
                    'x1': p.x1, 'y1': p.y1,
                    'x2': p.x2, 'y2': p.y2,
                    'confidence': p.confidence,
                }
                for p in self.points
            ],
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ZoomFOVCalibration':
        """Deserialize calibration from saved data."""
        points = [
            ZoomFOVPoint(
                zoom_level=p['zoom_level'],
                x1=p['x1'], y1=p['y1'],
                x2=p['x2'], y2=p['y2'],
                confidence=p['confidence'],
            )
            for p in data.get('points', [])
        ]
        return cls(
            points=points,
            wide_frame_width=data.get('wide_frame_width', 0),
            wide_frame_height=data.get('wide_frame_height', 0),
            error=data.get('error'),
        )


class ZoomFOVCalibrator:
    """Calibrates zoom-level to FOV mapping by capturing at different zoom levels."""

    def __init__(
        self,
        onvif_client: 'OnvifClient',
        profile_token: str,
        min_match_confidence: float = 0.25,
    ):
        self.onvif_client = onvif_client
        self.profile_token = profile_token
        self.min_match_confidence = min_match_confidence

        # Feature detector for image matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def find_zoom_bounds_in_wide(
        self,
        wide_frame: np.ndarray,
        zoom_frame: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Find the bounding box of zoom camera view within wide-angle frame.

        Uses feature matching and homography to find where the zoom frame
        appears in the wide frame.

        Args:
            wide_frame: Frame from wide-angle camera (cam1)
            zoom_frame: Frame from zoom camera (cam2)

        Returns:
            (x1, y1, x2, y2, confidence) as fractions of wide frame (0-1),
            or None if match failed
        """
        # Convert to grayscale
        wide_gray = cv2.cvtColor(wide_frame, cv2.COLOR_BGR2GRAY)
        zoom_gray = cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2GRAY)

        # Try different scale factors since zoom view may be various sizes
        scale_factors = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

        best_result = None
        best_confidence = 0

        for scale in scale_factors:
            scaled_zoom = cv2.resize(
                zoom_gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

            # Detect features
            kp_wide, desc_wide = self.orb.detectAndCompute(wide_gray, None)
            kp_zoom, desc_zoom = self.orb.detectAndCompute(scaled_zoom, None)

            if desc_wide is None or desc_zoom is None:
                continue
            if len(kp_wide) < 10 or len(kp_zoom) < 10:
                continue

            # Match features
            try:
                matches = self.bf.match(desc_zoom, desc_wide)
            except cv2.error:
                continue

            if len(matches) < 4:
                continue

            # Sort by distance (quality)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(100, len(matches))]

            if len(good_matches) < 4:
                continue

            # Get matched point coordinates
            src_pts = np.float32([kp_zoom[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_wide[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except cv2.error:
                continue

            if H is None:
                continue

            # Calculate inlier ratio as confidence
            inliers = mask.ravel().sum() if mask is not None else 0
            confidence = inliers / len(good_matches)

            if confidence > best_confidence:
                # Transform zoom frame corners to get bounding box in wide frame
                h, w = scaled_zoom.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                try:
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                except cv2.error:
                    continue

                # Get bounding box of transformed corners
                pts = transformed_corners.reshape(-1, 2)
                x1 = pts[:, 0].min() / wide_frame.shape[1]
                x2 = pts[:, 0].max() / wide_frame.shape[1]
                y1 = pts[:, 1].min() / wide_frame.shape[0]
                y2 = pts[:, 1].max() / wide_frame.shape[0]

                # Sanity check - bounds should be within frame
                if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                    best_result = (x1, y1, x2, y2, confidence)
                    best_confidence = confidence

        if best_result and best_result[4] >= self.min_match_confidence:
            return best_result
        return None

    def calibrate_zoom_fov(
        self,
        get_wide_frame: callable,
        get_zoom_frame: callable,
        zoom_levels: List[float] = None,
        settle_time: float = 2.0,
    ) -> ZoomFOVCalibration:
        """Calibrate FOV at multiple zoom levels.

        Args:
            get_wide_frame: Function returning current wide-angle frame
            get_zoom_frame: Function returning current zoom camera frame
            zoom_levels: List of zoom levels to calibrate (0.0-1.0)
            settle_time: Seconds to wait after zoom change before capturing

        Returns:
            ZoomFOVCalibration with measured FOV at each zoom level
        """
        if zoom_levels is None:
            zoom_levels = [0.0, 0.5, 1.0]  # Wide, 50%, full zoom

        points: List[ZoomFOVPoint] = []
        wide_frame = get_wide_frame()

        if wide_frame is None:
            return ZoomFOVCalibration(error="Wide camera has no frames")

        frame_height, frame_width = wide_frame.shape[:2]

        # Get current pan/tilt position - we'll keep these fixed
        # For zoom-only calibration, we don't need to move pan/tilt at all
        current_pan = 0.0
        current_tilt = 0.0
        try:
            current_pos = self.onvif_client.ptz_get_position(self.profile_token)
            if current_pos:
                # Handle None values - use 0.0 as default
                pan_val = current_pos.get('pan')
                tilt_val = current_pos.get('tilt')
                current_pan = pan_val if pan_val is not None else 0.0
                current_tilt = tilt_val if tilt_val is not None else 0.0
            LOGGER.info("Zoom FOV calibration starting at pan=%.2f, tilt=%.2f",
                       current_pan, current_tilt)
        except Exception as e:
            LOGGER.warning("Could not get PTZ position, using defaults: %s", e)

        for zoom in zoom_levels:
            LOGGER.info("Calibrating zoom level %.1f%%", zoom * 100)

            try:
                # Move to zoom level only (don't change pan/tilt)
                # Use zoom-only method for better camera compatibility
                self.onvif_client.ptz_set_zoom(self.profile_token, zoom)

                # Wait for zoom to settle
                time.sleep(settle_time)

                # Capture frames
                wide_frame = get_wide_frame()
                zoom_frame = get_zoom_frame()

                if wide_frame is None or zoom_frame is None:
                    LOGGER.warning("Missing frame at zoom %.1f%%", zoom * 100)
                    continue

                LOGGER.info("Captured frames: wide=%s, zoom=%s",
                           wide_frame.shape, zoom_frame.shape)

                # Find zoom view bounds in wide frame
                result = self.find_zoom_bounds_in_wide(wide_frame, zoom_frame)

                if result:
                    x1, y1, x2, y2, confidence = result
                    points.append(ZoomFOVPoint(
                        zoom_level=zoom,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=confidence,
                    ))
                    LOGGER.info(
                        "Zoom %.0f%%: FOV = (%.2f, %.2f) to (%.2f, %.2f), "
                        "size=%.1f%% x %.1f%%, confidence=%.2f",
                        zoom * 100, x1, y1, x2, y2,
                        (x2 - x1) * 100, (y2 - y1) * 100, confidence
                    )
                else:
                    LOGGER.warning("Could not match zoom view at %.0f%%", zoom * 100)

            except Exception as e:
                LOGGER.error("Error calibrating zoom %.0f%%: %s", zoom * 100, e)

        # Return to wide zoom
        try:
            self.onvif_client.ptz_set_zoom(self.profile_token, 0.0)
        except Exception:
            pass

        if len(points) == 0:
            return ZoomFOVCalibration(
                wide_frame_width=frame_width,
                wide_frame_height=frame_height,
                error="No zoom levels could be calibrated"
            )

        return ZoomFOVCalibration(
            points=points,
            wide_frame_width=frame_width,
            wide_frame_height=frame_height,
        )


def run_zoom_fov_calibration(
    wide_worker: 'StreamWorker',
    zoom_worker: 'StreamWorker',
    zoom_levels: List[float] = None,
) -> ZoomFOVCalibration:
    """Run zoom FOV calibration between two camera workers.

    Captures frames at each zoom level (0%, 50%, 100% by default) and
    determines what portion of the wide-angle view (cam1) is visible
    in the zoom camera (cam2) at each level.

    Args:
        wide_worker: StreamWorker for wide-angle camera (cam1)
        zoom_worker: StreamWorker for zoom camera (cam2)
        zoom_levels: List of zoom levels to calibrate (default: [0.0, 0.5, 1.0])

    Returns:
        ZoomFOVCalibration with measured FOV at each zoom level
    """
    LOGGER.info("Starting zoom FOV calibration: wide=%s, zoom=%s",
                wide_worker.camera.id, zoom_worker.camera.id)

    if not zoom_worker.onvif_client:
        return ZoomFOVCalibration(error="Zoom camera has no ONVIF client")

    if not zoom_worker.onvif_profile_token:
        return ZoomFOVCalibration(error="Zoom camera has no ONVIF profile token")

    if wide_worker.latest_frame is None:
        return ZoomFOVCalibration(error="Wide camera has no frames - is it streaming?")

    if zoom_worker.latest_frame is None:
        return ZoomFOVCalibration(error="Zoom camera has no frames - is it streaming?")

    calibrator = ZoomFOVCalibrator(
        onvif_client=zoom_worker.onvif_client,
        profile_token=zoom_worker.onvif_profile_token,
    )

    def get_wide_frame():
        return wide_worker.latest_frame.copy() if wide_worker.latest_frame is not None else None

    def get_zoom_frame():
        return zoom_worker.latest_frame.copy() if zoom_worker.latest_frame is not None else None

    return calibrator.calibrate_zoom_fov(
        get_wide_frame=get_wide_frame,
        get_zoom_frame=get_zoom_frame,
        zoom_levels=zoom_levels,
    )
