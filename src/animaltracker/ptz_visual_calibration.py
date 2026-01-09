"""PTZ Dead-Reckoning Calibration: Map pixel positions using timed moves and image matching.

This calibration approach works WITHOUT PTZ position feedback by:
1. Using timed continuous moves to explore the PTZ range
2. Using ORB feature matching to find where zoom view appears in wide view
3. Building a mapping from relative movement commands to pixel positions
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from .onvif_client import OnvifClient

LOGGER = logging.getLogger(__name__)


@dataclass
class VisualCalibrationPoint:
    """A calibration point based on visual matching."""
    # Cumulative move commands from home (in arbitrary units based on move duration)
    pan_offset: float  # negative = left, positive = right
    tilt_offset: float  # negative = down, positive = up
    # Where the zoom camera center appears on wide-angle (as fraction 0-1)
    wide_x: float
    wide_y: float
    confidence: float  # Match quality (0-1)


@dataclass 
class VisualCalibrationResult:
    """Result of visual calibration."""
    # Linear fit parameters: wide_x = pan_offset * pan_to_pixel_x + center_x
    pan_to_pixel_x: float  # How much wide_x changes per pan unit
    tilt_to_pixel_y: float  # How much wide_y changes per tilt unit
    center_x: float  # Wide pixel x when PTZ is at home
    center_y: float  # Wide pixel y when PTZ is at home
    points: List[VisualCalibrationPoint] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'pan_to_pixel_x': self.pan_to_pixel_x,
            'tilt_to_pixel_y': self.tilt_to_pixel_y,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'num_points': len(self.points),
            'error': self.error,
        }
    
    def pixel_to_pan_tilt(self, wide_x: float, wide_y: float) -> Tuple[float, float]:
        """Convert wide-angle pixel position to PTZ pan/tilt offset from home.
        
        Args:
            wide_x: X position in wide frame (0-1)
            wide_y: Y position in wide frame (0-1)
            
        Returns:
            (pan_offset, tilt_offset) - movement needed from home position
        """
        if self.pan_to_pixel_x == 0 or self.tilt_to_pixel_y == 0:
            return (0.0, 0.0)
        
        pan_offset = (wide_x - self.center_x) / self.pan_to_pixel_x
        tilt_offset = (wide_y - self.center_y) / self.tilt_to_pixel_y
        return (pan_offset, tilt_offset)


class VisualPTZCalibrator:
    """Calibrate PTZ-to-pixel mapping using image matching (no position feedback needed)."""
    
    def __init__(
        self,
        onvif_client: 'OnvifClient',
        profile_token: str,
        move_speed: float = 0.3,
        move_duration: float = 0.4,
        settle_time: float = 0.8,
        min_match_confidence: float = 0.25,
    ):
        """
        Args:
            onvif_client: ONVIF client for PTZ control
            profile_token: ONVIF profile token
            move_speed: Speed for continuous moves (0-1)
            move_duration: Duration of each move step in seconds
            settle_time: Wait time after move before capturing
            min_match_confidence: Minimum confidence for valid match
        """
        self.onvif_client = onvif_client
        self.profile_token = profile_token
        self.move_speed = move_speed
        self.move_duration = move_duration
        self.settle_time = settle_time
        self.min_match_confidence = min_match_confidence
        
        # Feature detector for image matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Current estimated position (in move units from home)
        self._pan_offset = 0.0
        self._tilt_offset = 0.0
    
    def _timed_move(self, pan_dir: float, tilt_dir: float, duration: float) -> None:
        """Execute a timed continuous move.
        
        Args:
            pan_dir: Pan direction (-1 to 1, negative=left)
            tilt_dir: Tilt direction (-1 to 1, negative=down)
            duration: Move duration in seconds
        """
        pan_vel = pan_dir * self.move_speed
        tilt_vel = tilt_dir * self.move_speed
        
        self.onvif_client.ptz_move(self.profile_token, pan_vel, tilt_vel, 0.0)
        time.sleep(duration)
        self.onvif_client.ptz_stop(self.profile_token)
        
        # Update our estimated position
        self._pan_offset += pan_dir * duration
        self._tilt_offset += tilt_dir * duration
    
    def _go_home(self) -> None:
        """Return PTZ to home/center position."""
        try:
            # Try GotoHomePosition first
            ptz_service = self.onvif_client._camera.create_ptz_service()
            ptz_service.GotoHomePosition({'ProfileToken': self.profile_token})
            LOGGER.info("Sent GotoHomePosition command")
        except Exception as e:
            LOGGER.debug("GotoHomePosition failed (%s), trying absolute move", e)
            try:
                self.onvif_client.ptz_move_absolute(self.profile_token, 0.0, 0.0, 0.5)
            except Exception as e2:
                LOGGER.warning("Absolute move failed: %s", e2)
        
        self._pan_offset = 0.0
        self._tilt_offset = 0.0
        time.sleep(2.0)  # Wait for move to complete
    
    def find_zoom_in_wide(
        self, 
        wide_frame: np.ndarray, 
        zoom_frame: np.ndarray,
        debug: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        """Find where zoom camera view appears in wide-angle frame.
        
        Uses ORB feature matching to locate the zoom view within the wide view.
        
        Returns:
            (center_x, center_y, confidence) as fractions of wide frame dimensions,
            or None if match failed
        """
        # Convert to grayscale
        wide_gray = cv2.cvtColor(wide_frame, cv2.COLOR_BGR2GRAY)
        zoom_gray = cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        wide_gray = clahe.apply(wide_gray)
        zoom_gray = clahe.apply(zoom_gray)
        
        # Try different scale factors - zoom view appears as portion of wide view
        scale_factors = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        best_match = None
        best_confidence = 0
        best_inliers = 0
        
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
            
            if debug:
                LOGGER.debug("Scale %.2f: %d matches, %d inliers, conf=%.2f", 
                            scale, len(good_matches), inliers, confidence)
            
            if inliers > best_inliers:
                # Find center of zoom view in wide coordinates
                h, w = scaled_zoom.shape
                corners = np.float32([
                    [0, 0], [w, 0], [w, h], [0, h]
                ]).reshape(-1, 1, 2)
                
                try:
                    corners_wide = cv2.perspectiveTransform(corners, H)
                    # Center is average of corners
                    cx = corners_wide[:, 0, 0].mean() / wide_frame.shape[1]
                    cy = corners_wide[:, 0, 1].mean() / wide_frame.shape[0]
                    
                    # Sanity check - should be within frame
                    if 0 <= cx <= 1 and 0 <= cy <= 1:
                        best_match = (cx, cy, confidence)
                        best_confidence = confidence
                        best_inliers = inliers
                except cv2.error:
                    continue
        
        if best_match and best_confidence >= self.min_match_confidence:
            return best_match
        
        LOGGER.debug("No match found (best confidence: %.2f)", best_confidence)
        return None
    
    def calibrate(
        self,
        get_wide_frame: Callable[[], np.ndarray],
        get_zoom_frame: Callable[[], np.ndarray],
        grid_size: int = 3,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VisualCalibrationResult:
        """Run visual calibration by moving PTZ and matching images.
        
        Args:
            get_wide_frame: Function returning current wide-angle frame
            get_zoom_frame: Function returning current zoom camera frame  
            grid_size: Grid dimensions (3 = 3x3 = 9 points)
            progress_callback: Optional callback(message, progress_pct)
            
        Returns:
            VisualCalibrationResult with computed parameters
        """
        points: List[VisualCalibrationPoint] = []
        
        def report(msg: str, pct: float):
            LOGGER.info("%s (%.0f%%)", msg, pct * 100)
            if progress_callback:
                progress_callback(msg, pct)
        
        report("Starting visual PTZ calibration", 0.0)
        
        # Step 1: Go to home position
        report("Moving to home position...", 0.05)
        self._go_home()
        
        # Step 2: Capture center point
        report("Capturing center position...", 0.1)
        time.sleep(self.settle_time)
        
        wide_frame = get_wide_frame()
        zoom_frame = get_zoom_frame()
        
        if wide_frame is None or zoom_frame is None:
            return VisualCalibrationResult(
                pan_to_pixel_x=0.0, tilt_to_pixel_y=0.0,
                center_x=0.5, center_y=0.5,
                error="Could not capture frames from cameras"
            )
        
        # Find initial match at home position
        center_match = self.find_zoom_in_wide(wide_frame, zoom_frame, debug=True)
        if center_match:
            cx, cy, conf = center_match
            LOGGER.info("Center match: wide_x=%.3f, wide_y=%.3f, conf=%.2f", cx, cy, conf)
            points.append(VisualCalibrationPoint(
                pan_offset=0.0, tilt_offset=0.0,
                wide_x=cx, wide_y=cy, confidence=conf
            ))
        else:
            LOGGER.warning("Could not match center position - using default")
            cx, cy = 0.5, 0.5
        
        # Step 3: Explore grid positions using timed moves
        # Move pattern: from center, go to each grid position
        total_steps = grid_size * grid_size
        step = 0
        
        # Generate grid offsets (in move units)
        half_range = (grid_size - 1) / 2
        move_steps = 2  # How many timed moves to reach edge
        
        for row in range(grid_size):
            for col in range(grid_size):
                if row == grid_size // 2 and col == grid_size // 2:
                    # Skip center, already captured
                    step += 1
                    continue
                
                step += 1
                progress = 0.1 + 0.8 * (step / total_steps)
                
                # Calculate target offset from center
                target_pan = (col - half_range) * move_steps * self.move_duration
                target_tilt = (half_range - row) * move_steps * self.move_duration  # Invert Y
                
                report(f"Moving to grid position ({col+1}, {row+1})...", progress)
                
                # Return to home first for consistent moves
                self._go_home()
                time.sleep(0.5)
                
                # Move to target position
                if target_pan != 0:
                    pan_dir = 1.0 if target_pan > 0 else -1.0
                    for _ in range(abs(int(target_pan / self.move_duration))):
                        self._timed_move(pan_dir, 0, self.move_duration)
                        time.sleep(0.1)
                
                if target_tilt != 0:
                    tilt_dir = 1.0 if target_tilt > 0 else -1.0
                    for _ in range(abs(int(target_tilt / self.move_duration))):
                        self._timed_move(0, tilt_dir, self.move_duration)
                        time.sleep(0.1)
                
                # Wait for camera to settle
                time.sleep(self.settle_time)
                
                # Capture and match
                wide_frame = get_wide_frame()
                zoom_frame = get_zoom_frame()
                
                if wide_frame is None or zoom_frame is None:
                    LOGGER.warning("Failed to capture frame at position (%d, %d)", col, row)
                    continue
                
                match = self.find_zoom_in_wide(wide_frame, zoom_frame)
                if match:
                    mx, my, conf = match
                    LOGGER.info("Position (%d,%d): pan_off=%.2f, tilt_off=%.2f -> wide=(%.3f, %.3f), conf=%.2f",
                               col, row, self._pan_offset, self._tilt_offset, mx, my, conf)
                    points.append(VisualCalibrationPoint(
                        pan_offset=self._pan_offset,
                        tilt_offset=self._tilt_offset,
                        wide_x=mx, wide_y=my,
                        confidence=conf
                    ))
                else:
                    LOGGER.warning("No match at position (%d, %d)", col, row)
        
        # Step 4: Return to home
        report("Returning to home...", 0.95)
        self._go_home()
        
        # Step 5: Compute linear fit
        report("Computing calibration parameters...", 0.98)
        
        if len(points) < 3:
            return VisualCalibrationResult(
                pan_to_pixel_x=0.0, tilt_to_pixel_y=0.0,
                center_x=cx, center_y=cy,
                points=points,
                error=f"Not enough calibration points ({len(points)}/3 minimum)"
            )
        
        # Linear regression: wide_x = pan_offset * pan_to_pixel + center_x
        pan_offsets = np.array([p.pan_offset for p in points])
        tilt_offsets = np.array([p.tilt_offset for p in points])
        wide_xs = np.array([p.wide_x for p in points])
        wide_ys = np.array([p.wide_y for p in points])
        
        # Weighted by confidence
        weights = np.array([p.confidence for p in points])
        
        # Fit pan -> x relationship
        if np.std(pan_offsets) > 0.01:
            pan_to_pixel_x = np.average(
                (wide_xs - np.average(wide_xs, weights=weights)) / 
                (pan_offsets - np.average(pan_offsets, weights=weights) + 1e-6),
                weights=weights
            )
        else:
            pan_to_pixel_x = 0.5  # Default
        
        # Fit tilt -> y relationship
        if np.std(tilt_offsets) > 0.01:
            tilt_to_pixel_y = np.average(
                (wide_ys - np.average(wide_ys, weights=weights)) /
                (tilt_offsets - np.average(tilt_offsets, weights=weights) + 1e-6),
                weights=weights
            )
        else:
            tilt_to_pixel_y = 0.5  # Default
        
        # Find center point (where pan_offset=0, tilt_offset=0)
        center_point = next((p for p in points if abs(p.pan_offset) < 0.01 and abs(p.tilt_offset) < 0.01), None)
        if center_point:
            center_x = center_point.wide_x
            center_y = center_point.wide_y
        else:
            center_x = np.average(wide_xs, weights=weights)
            center_y = np.average(wide_ys, weights=weights)
        
        report(f"Calibration complete! {len(points)} points", 1.0)
        
        result = VisualCalibrationResult(
            pan_to_pixel_x=float(pan_to_pixel_x),
            tilt_to_pixel_y=float(tilt_to_pixel_y),
            center_x=float(center_x),
            center_y=float(center_y),
            points=points,
        )
        
        LOGGER.info("Calibration result: %s", result.to_dict())
        return result


class DeadReckoningPTZTracker:
    """Track objects by estimating PTZ position from movement commands."""
    
    def __init__(
        self,
        onvif_client: 'OnvifClient',
        profile_token: str,
        calibration: VisualCalibrationResult,
        move_speed: float = 0.3,
    ):
        self.onvif_client = onvif_client
        self.profile_token = profile_token
        self.calibration = calibration
        self.move_speed = move_speed
        
        # Current estimated position (move units from home)
        self._pan_offset = 0.0
        self._tilt_offset = 0.0
        self._is_moving = False
    
    def go_home(self) -> None:
        """Return to home position and reset tracking."""
        try:
            ptz_service = self.onvif_client._camera.create_ptz_service()
            ptz_service.GotoHomePosition({'ProfileToken': self.profile_token})
        except Exception:
            try:
                self.onvif_client.ptz_move_absolute(self.profile_token, 0.0, 0.0, 0.5)
            except Exception:
                pass
        
        self._pan_offset = 0.0
        self._tilt_offset = 0.0
    
    def move_to_pixel(self, wide_x: float, wide_y: float, duration: float = 0.3) -> None:
        """Move PTZ to center on a pixel position in the wide-angle view.
        
        Args:
            wide_x: X position in wide frame (0-1)
            wide_y: Y position in wide frame (0-1)
            duration: Duration of movement
        """
        # Calculate required offset from home
        target_pan, target_tilt = self.calibration.pixel_to_pan_tilt(wide_x, wide_y)
        
        # Calculate delta from current position
        delta_pan = target_pan - self._pan_offset
        delta_tilt = target_tilt - self._tilt_offset
        
        # Convert to velocity direction
        pan_vel = np.clip(delta_pan * 2, -1, 1) * self.move_speed
        tilt_vel = np.clip(delta_tilt * 2, -1, 1) * self.move_speed
        
        if abs(pan_vel) < 0.05 and abs(tilt_vel) < 0.05:
            return  # Already close enough
        
        # Execute move
        self._is_moving = True
        self.onvif_client.ptz_move(self.profile_token, pan_vel, tilt_vel, 0.0)
        time.sleep(duration)
        self.onvif_client.ptz_stop(self.profile_token)
        self._is_moving = False
        
        # Update estimated position
        self._pan_offset += pan_vel * duration / self.move_speed
        self._tilt_offset += tilt_vel * duration / self.move_speed
    
    def get_estimated_position(self) -> Tuple[float, float]:
        """Get estimated PTZ position in move units from home."""
        return (self._pan_offset, self._tilt_offset)


async def run_visual_calibration(
    wide_worker: 'StreamWorker',
    zoom_worker: 'StreamWorker',
    grid_size: int = 3,
) -> Dict:
    """Run visual auto-calibration between wide and zoom cameras.
    
    Args:
        wide_worker: StreamWorker for wide-angle camera
        zoom_worker: StreamWorker for zoom/PTZ camera
        grid_size: Calibration grid size (3 = 3x3)
        
    Returns:
        Dict with calibration results
    """
    import asyncio
    
    if not zoom_worker.onvif_client or not zoom_worker.onvif_profile_token:
        return {'error': 'Zoom camera ONVIF not configured'}
    
    calibrator = VisualPTZCalibrator(
        onvif_client=zoom_worker.onvif_client,
        profile_token=zoom_worker.onvif_profile_token,
    )
    
    def get_wide_frame():
        return wide_worker.latest_frame
    
    def get_zoom_frame():
        return zoom_worker.latest_frame
    
    # Run calibration in executor (blocking)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: calibrator.calibrate(get_wide_frame, get_zoom_frame, grid_size)
    )
    
    return result.to_dict()
