"""PTZ auto-tracking: Center and zoom on detected objects."""
from __future__ import annotations

import logging
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .onvif_client import OnvifClient
    from .detector import Detection
    from .ptz_calibration import ZoomFOVCalibration

LOGGER = logging.getLogger(__name__)

# Dedicated PTZ decision logger for debugging tracking behavior
# Enable with: logging.getLogger('ptz.decisions').setLevel(logging.DEBUG)
PTZ_LOGGER = logging.getLogger('ptz.decisions')


def load_zoom_fov_calibration(path: Optional[str]) -> Optional['ZoomFOVCalibration']:
    """Load an optional cam2-in-cam1 zoom/FOV calibration file."""
    if not path:
        return None
    calibration_path = Path(path).expanduser()
    if not calibration_path.exists():
        LOGGER.info("Zoom FOV calibration file not found: %s", calibration_path)
        return None
    try:
        from .ptz_calibration import ZoomFOVCalibration
        with calibration_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        calibration = ZoomFOVCalibration.from_dict(data)
    except Exception as exc:
        LOGGER.warning("Could not load zoom FOV calibration %s: %s", calibration_path, exc)
        return None
    if calibration.error:
        LOGGER.warning(
            "Zoom FOV calibration %s contains error: %s",
            calibration_path, calibration.error,
        )
        return None
    LOGGER.info(
        "Loaded zoom FOV calibration %s with %d points",
        calibration_path, len(calibration.points),
    )
    return calibration


@dataclass
class PTZDecisionEntry:
    """A single PTZ decision log entry for storage."""
    timestamp: float
    event: str  # mode_change, move, deadzone, rate_limit, tracking_lost, etc.
    mode: str  # idle, patrol, tracking
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'event': self.event,
            'mode': self.mode,
            'details': self.details,
        }


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
        fw = self.frame_width if self.frame_width > 0 else 1
        fh = self.frame_height if self.frame_height > 0 else 1
        norm_x = pixel_x / fw
        norm_y = pixel_y / fh

        # Calculate offset from center. Guard against pan_scale/tilt_scale
        # being misconfigured to 0 (would otherwise raise ZeroDivisionError
        # and crash the streaming executor thread).
        pan_scale = self.pan_scale if self.pan_scale > 1e-6 else 1.0
        tilt_scale = self.tilt_scale if self.tilt_scale > 1e-6 else 1.0
        offset_x = (norm_x - self.pan_center_x) / pan_scale
        offset_y = (self.tilt_center_y - norm_y) / tilt_scale  # Y inverted (up = positive tilt)
        
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
    IDLE = "idle"               # Not active
    PATROL = "patrol"           # Scanning for objects
    INVESTIGATE = "investigate" # Pointed cam2 at a small cam1 candidate, awaiting confirmation
    TRACKING = "tracking"       # Following detected object


@dataclass
class PTZTracker:
    """Auto-tracking controller that moves PTZ to follow detections.
    
    Optimized for split-model architecture where YOLO provides fast detections
    for real-time tracking (~50-150ms inference). Default values are tuned for
    responsive tracking with minimal latency.
    """
    
    onvif_client: 'OnvifClient'
    profile_token: str
    calibration: PTZCalibration = field(default_factory=PTZCalibration)
    
    # Tracking behavior
    target_fill_pct: float = 0.6  # Target 60% frame fill
    min_move_threshold: float = 0.05  # Don't move if offset < 5% of range
    min_detection_area: float = 0.005  # Ignore detections smaller than 0.5% of frame (filters leaves/noise)
    # Reject detections whose bbox covers more than this fraction of the
    # frame. A real wildlife subject in a zoomed PTZ view does not fill the
    # whole frame; high-fill bboxes are MegaDetector mis-fires on motion
    # blur, lens artefacts, or stationary high-contrast features (stone
    # walls, brick edges). Driving the PTZ on these caused multi-second
    # full-velocity overshoots once the velocity-cap (which only applies
    # below low_fill_threshold) was bypassed.
    max_detection_area: float = 0.70
    # Optimized defaults for real-time tracking with YOLO
    smoothing: float = 0.15  # Lower = faster response (was 0.3)
    update_interval: float = 0.1  # 10 updates/sec for responsive tracking (was 0.2)

    # Velocity capping for slow / small / distant targets.
    # When the bbox fills less than ``low_fill_threshold`` of the frame, the
    # commanded |pan|/|tilt| velocity is clamped to ``low_fill_velocity_cap``.
    # ContinuousMove keeps slewing at the commanded velocity until the next
    # decision; for slow-moving animals (deer, etc.) the previous full-speed
    # corrections caused overshoot followed by a swing back. Capping prevents
    # the camera from outpacing the target between detections.
    # Raised threshold (was 0.15) so distant animals filling 15-30% of the
    # frame still get the velocity cap applied -- previously a dog at ~17%
    # fill bypassed the cap, got vel ~0.6 on recovery, and the camera flew
    # past the target between detection ticks.
    low_fill_threshold: float = 0.30        # bbox max-dim / frame-dim fraction
    # Raised back to 0.30 (was 0.15). The timer-backed step stop below makes
    # each pulse reliably bounded, so the cap no longer has to double as
    # overshoot protection. At 0.15 the recorded controller only ever
    # commanded ~22% of the velocity it computed and the offset grew
    # monotonically while a coyote walked out of cam2's FOV.
    #
    # NOTE: this dataclass default is NOT what production uses -- config.py's
    # PTZTrackingSettings and config/cameras.yml both override it via
    # create_ptz_tracker(). Change all three together or the tuning is dead
    # (that is how the 505be72 raise silently never took effect).
    low_fill_velocity_cap: float = 0.30     # cap on |pan|/|tilt| when below threshold
    # Offset magnitude at which the low-fill cap is allowed to reach its
    # full value. Below this, the cap is scaled proportionally to the overall
    # |offset| so we taper toward zero velocity as we approach center,
    # instead of slamming the same 0.22 pulse for both 0.13 and 0.46
    # offsets (which causes overshoot of small offsets and undershoot of
    # large ones with sparse detection ticks).
    low_fill_cap_full_offset: float = 0.40
    # When the target is more than this far off-center, suppress positive
    # (zoom-in) zoom velocity. Zooming in while still chasing narrows the
    # FOV exactly when we need it wide. Zoom-out is still allowed.
    # Tightened (was 0.10) because borderline offsets like 0.08 were
    # squeaking through the gate and the next detection often showed the
    # animal had already left the now-narrower FOV.
    zoom_in_offset_gate: float = 0.05
    
    # Patrol settings
    patrol_enabled: bool = True  # Enable patrol when no detections
    patrol_speed: float = 0.15  # Patrol pan speed (slow sweep)
    patrol_tilt: float = 0.0    # Tilt position during patrol
    patrol_zoom: float = 0.0    # Zoom level during patrol (wide)
    patrol_return_delay: float = 5.0  # Seconds with no sighting from any camera before patrol

    # Minimum time (seconds) a ContinuousMove issued by tracking should be
    # allowed to run before _handle_no_detections is permitted to ptz_stop it.
    # With sparse detections (e.g. SpeciesNet at ~2 fps on a small bird that
    # only matches a fraction of frames) we frequently issue a move and then
    # have an immediate "no detections" tick which used to halt the camera
    # before it physically moved. Holding the move for at least this long
    # lets the slew actually happen.
    # Lowered (was 0.6) because at ~2 fps detection cadence a 0.6 s floor
    # meant every gap was a full 0.6 s of free slew at the last commanded
    # velocity, causing repeated overshoot of slow-moving targets.
    # Lowered again (was 0.3) -- combined with tracking_step_duration we
    # never want a slew to free-run for more than ~150 ms after the last
    # detection, otherwise mechanical Stop latency carries the camera past
    # the target.
    move_min_duration: float = 0.15

    # Hard cap on how long a *tracking* ContinuousMove is allowed to run
    # before the controller proactively issues a Stop, regardless of
    # whether new detections have arrived. With sparse detection cadence
    # (e.g. SpeciesNet ~1 Hz on small targets) the previous behaviour was
    # to let the last commanded velocity run until either a new detection
    # or the no-detections path expired, which routinely overshot. By
    # auto-stopping after a short step, each detection only authorises a
    # bounded amount of slew, then the camera waits for the next detection.
    # A timer enforces this duration even if the detector does not call back
    # into the tracker until much later.
    tracking_step_duration: float = 0.35

    # When the target camera (cam2) has been driving tracking, suppress
    # cam1-driven repositioning for this many seconds after the last cam2
    # detection. Cam2's view is more accurate for fine tracking, and the
    # cam1->cam2 PTZ mapping is only valid when cam2 is near its calibration
    # zero pose -- once cam2 has slewed, cam1 pixel offsets miscompute and
    # produce large erroneous slews. Only fall back to cam1 if cam2 has
    # truly lost the object for longer than this window.
    cam1_fallback_delay: float = 3.0

    # --- Investigate mode (opt-in) ---
    # When the source (wide) camera has a *small* detection that is below
    # min_detection_area but still above investigate_min_area, treat it as
    # a candidate worth checking with the zoom camera instead of dropping
    # it as noise. Cam2 is slewed to the candidate and given
    # investigate_timeout seconds to confirm with its own detection. If
    # cam2 confirms, we transition into normal TRACKING. If not, the
    # candidate location is added to a cooldown list so we don't keep
    # re-investigating the same patch of leaves.
    investigate_enabled: bool = False
    investigate_min_area: float = 0.0005   # 0.05% of frame; below this is pure noise
    investigate_timeout: float = 4.0       # how long to wait for cam2 confirmation
    investigate_cooldown: float = 30.0     # don't re-investigate same spot for this long
    investigate_cooldown_radius: float = 0.10  # normalized distance considered "same spot"
    # Stepped slew during investigate: instead of one full-velocity move that
    # overshoots a small target (especially when cam2 is already zoomed in),
    # break the approach into short capped pulses with stop-and-detect gaps
    # so cam2 can confirm before we move further.
    investigate_velocity_cap: float = 0.25  # max |pan|/|tilt| velocity during investigate (vs. ~1.0 normally)
    investigate_step_duration: float = 0.35  # max ContinuousMove duration per step before auto-stop
    investigate_settle_delay: float = 0.25   # pause after stop to let cam2 detect before next step
    # Zoom out while investigating so the candidate is more likely to fall in
    # cam2's frame after a coarse slew. Cam2 will zoom back in once tracking
    # confirms (normal _do_tracking handles zoom-in).
    investigate_zoom_out: bool = True
    investigate_zoom_velocity: float = -0.5  # negative = zoom out
    
    # Preset-based patrol
    patrol_presets: list = field(default_factory=list)  # List of preset tokens
    patrol_dwell_time: float = 10.0  # Seconds at each preset
    _preset_tokens: list = field(default_factory=list, init=False)  # Resolved preset tokens
    _current_preset_index: int = field(default=0, init=False)
    _preset_arrival_time: float = field(default=0.0, init=False)
    
    # Multi-camera tracking: secondary cameras that can contribute detections
    # When the target camera (cam2) detects an object, use those detections for fine tracking
    secondary_cameras: list = field(default_factory=list)  # Camera IDs that can contribute

    # Visibility-aware recovery: when cam2 recently had the target but loses
    # it, use cam1 plus a calibrated cam2 FOV footprint to decide whether to
    # hold/zoom, recenter, or zoom out before blindly falling back to cam1.
    zoom_fov_calibration: Optional['ZoomFOVCalibration'] = None
    visibility_recovery_enabled: bool = True
    visibility_recovery_min_overlap: float = 0.50
    visibility_recovery_edge_margin: float = 0.12
    visibility_recovery_zoom_out_velocity: float = -0.25
    visibility_recovery_zoom_in_velocity: float = 0.15
    visibility_recovery_zoom_in_max_zoom: float = 0.35
    visibility_recovery_zoom_in_fill_threshold: float = 0.03
    visibility_recovery_velocity_cap: float = 0.20

    # State
    _last_update: float = field(default=0.0, init=False)
    _target_pan: float = field(default=0.0, init=False)
    _target_tilt: float = field(default=0.0, init=False)
    _target_zoom: float = field(default=0.0, init=False)
    _patrol_active: bool = field(default=False, init=False)  # Patrol toggle state
    _track_active: bool = field(default=False, init=False)   # Tracking toggle state
    _mode: PTZMode = field(default=PTZMode.IDLE, init=False)
    _patrol_direction: int = field(default=1, init=False)  # 1 = right, -1 = left
    _last_detection_time: float = field(default=0.0, init=False)
    # Wall-clock time at which *any* contributing camera last saw the subject,
    # regardless of whether that sighting went on to drive a move. Distinct
    # from ``_last_detection_time``, which records when a camera last *drove*
    # tracking and is paired with ``_last_detection_source`` to arbitrate the
    # cam1/cam2 handoff (``cam1_fallback_delay``). Conflating the two would
    # make the cam1 suppression window never expire. The return-to-patrol
    # decision must use this one: sightings that were rate-limited, held by
    # the deadzone, skipped as duplicate frames, blanked by the PTZ settle
    # gate on the moved camera, or consumed by visibility recovery still
    # prove the animal is there.
    _last_target_seen_time: float = field(default=0.0, init=False)
    _patrol_reverse_time: float = field(default=0.0, init=False)
    _tracking_lost_logged_at: float = field(default=0.0, init=False)  # When we last logged "tracking lost"
    _last_tracked_species: str = field(default="", init=False)  # Species we were tracking when lost
    _last_detection_source: str = field(default="", init=False)  # Which camera provided the detection
    _holding_position: bool = field(default=False, init=False)  # True after we've issued a Stop while waiting for return-to-patrol

    # PTZ movement timing: track when last move command was sent
    # Used by pipeline to skip detection during camera settle period.
    # NOTE: only updated for *discrete* repositions (preset goto, absolute move,
    # tracking velocity changes). Continuous patrol velocity does NOT refresh
    # this timestamp, otherwise the worker's settle gate would permanently
    # suppress inference and tracking could never engage.
    _last_move_time: float = field(default=0.0, init=False)

    # Investigate-mode state
    _investigate_started_at: float = field(default=0.0, init=False)
    _investigate_target: Optional[Tuple[float, float]] = field(default=None, init=False)  # normalized cam1 center being investigated
    _investigate_rejects: list = field(default_factory=list, init=False)  # list of (norm_x, norm_y, expires_at)
    _investigate_step_started_at: float = field(default=0.0, init=False)  # when current capped slew began
    _investigate_step_stopped_at: float = field(default=0.0, init=False)  # when last step was halted (for settle delay)
    _investigate_step_active: bool = field(default=False, init=False)     # is a capped slew in flight?

    # Cached patrol velocity so we don't re-issue identical ContinuousMove
    # commands every tick (which both hammers the camera and refreshes
    # _last_move_time, causing the settle deadlock).
    _patrol_velocity: Optional[Tuple[float, float, float]] = field(default=None, init=False)

    # Wall-clock time at which the current tracking ContinuousMove should
    # be auto-stopped (0 = no pending stop). See ``tracking_step_duration``.
    _tracking_step_stop_at: float = field(default=0.0, init=False)
    _tracking_step_timer: Optional[threading.Timer] = field(default=None, init=False)

    # Diagnostic context for decision log enrichment.
    # _current_capture_ts: capture_ts of the frame whose detections we are
    #   reacting to in the *current* update_locked / update_multi_camera_locked
    #   call. Set at the top of those methods, cleared at end. Used by the
    #   move/deadzone log sites to record frame_age_ms (decision_ts - capture_ts)
    #   so we can tell when the controller acted on stale frames.
    # _last_move_capture_ts: capture_ts of the frame that drove the most
    #   recent issued ContinuousMove. Lets us compute the inter-detection gap
    #   (capture-to-capture) at the next move, which is the actual interval
    #   the camera was free-slewing under the previous velocity.
    # _tracking_step_armed_at: wall-clock time _arm_tracking_step() was called.
    #   Logged on tracking_step_stop so we can see how long the auto-stop
    #   actually took to fire vs the configured tracking_step_duration.
    _current_capture_ts: float = field(default=0.0, init=False)
    _last_move_capture_ts: float = field(default=0.0, init=False)
    _tracking_step_armed_at: float = field(default=0.0, init=False)
    _last_move_source: str = field(default="", init=False)
    _last_move_bbox_signature: Optional[Tuple[int, int, int, int]] = field(default=None, init=False)

    # Track persistence: Once we lock onto a target, keep tracking it
    # to prevent jitter from switching between detections every frame.
    _locked_track_id: Optional[int] = field(default=None, init=False)  # Currently locked track ID
    _locked_bbox_center: Optional[Tuple[float, float]] = field(default=None, init=False)  # Last known center (normalized)
    _locked_source_camera: Optional[str] = field(default=None, init=False)  # Camera id whose frame _locked_bbox_center is in
    _locked_species: Optional[str] = field(default=None, init=False)  # Species we are locked onto (for handoff continuity)
    _lock_start_time: float = field(default=0.0, init=False)  # When we locked onto current target
    _consecutive_lock_misses: int = field(default=0, init=False)  # Frames since locked target was last seen
    _lock_miss_limit: int = field(default=3, init=False)  # Misses before releasing lock (was 5; lower = faster recovery on mismatched stale lock)
    # Hysteresis for switching to a *different* (non-locked) max-confidence target.
    # H4: prevents flickering between two similar-confidence detections.
    _challenger_track_id: Optional[int] = field(default=None, init=False)
    _challenger_streak: int = field(default=0, init=False)
    _challenger_required_streak: int = field(default=3, init=False)
    _challenger_required_margin: float = field(default=0.10, init=False)
    # Cam handoff: require N target-camera frames in a row matching the
    # locked species/aim before declaring takeover (H8).
    _pending_takeover_frames: int = field(default=0, init=False)
    _pending_takeover_required: int = field(default=2, init=False)
    # Static-target watchdog: real animals breathe / sway / walk. If a
    # locked target's bbox center has not moved at all for this long,
    # treat it as a stuck false positive (leaf, shadow, branch, log)
    # and release the lock so patrol can resume. Without this guard a
    # 50%-confidence "animal" detected on a stationary tree limb can
    # keep the system "successfully tracking" indefinitely.
    _lock_motion_anchor: Optional[Tuple[float, float]] = field(default=None, init=False)
    _lock_motion_anchor_time: float = field(default=0.0, init=False)
    _lock_static_release_sec: float = field(default=45.0, init=False)
    _lock_motion_threshold: float = field(default=0.02, init=False)

    # Decision log buffer (for storing with clips)
    _decision_log: List[PTZDecisionEntry] = field(default_factory=list, init=False)
    _decision_log_max_entries: int = field(default=1000, init=False)  # Prevent unbounded growth

    # Thread lock for multi-camera access
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _log_decision(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a PTZ decision for later retrieval."""
        entry = PTZDecisionEntry(
            timestamp=time.time(),
            event=event,
            mode=self._mode.value,
            details=details or {},
        )
        self._decision_log.append(entry)
        # Trim if too large
        if len(self._decision_log) > self._decision_log_max_entries:
            self._decision_log = self._decision_log[-self._decision_log_max_entries:]
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get all logged PTZ decisions as dicts."""
        with self._lock:
            return [entry.to_dict() for entry in self._decision_log]
    
    def get_decisions_in_window(self, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        """Get PTZ decisions within a time window (for event finalization).
        
        This is safe for shared trackers - doesn't clear the log, just returns
        decisions that fall within the event's time window.
        """
        with self._lock:
            return [
                entry.to_dict()
                for entry in self._decision_log
                if start_ts <= entry.timestamp <= end_ts
            ]
    
    def clear_decision_log(self) -> List[Dict[str, Any]]:
        """Get and clear all logged PTZ decisions (for event finalization).
        
        DEPRECATED: Use get_decisions_in_window() for shared trackers.
        """
        with self._lock:
            log = [entry.to_dict() for entry in self._decision_log]
            self._decision_log = []
            return log
    
    def trim_old_decisions(self, cutoff_ts: float) -> int:
        """Remove decisions older than cutoff timestamp.
        
        Returns number of entries removed.
        """
        with self._lock:
            original_len = len(self._decision_log)
            self._decision_log = [e for e in self._decision_log if e.timestamp >= cutoff_ts]
            return original_len - len(self._decision_log)

    def get_last_move_time(self) -> float:
        """Return timestamp of last PTZ move command.
        
        Used by pipeline to implement settle delay - skip detections
        while the camera is still moving/stabilizing after a PTZ command.
        """
        return self._last_move_time

    def _arm_tracking_step(self, now: float) -> None:
        """Schedule an automatic Stop after ``tracking_step_duration``.

        Called immediately after issuing a tracking ContinuousMove so the
        camera only slews a bounded amount per detection. Without this,
        the last commanded velocity continues to run between detections
        (especially with sparse SpeciesNet cadence on small targets) and
        the camera flies past the animal.
        """
        if self._tracking_step_timer is not None:
            self._tracking_step_timer.cancel()
            self._tracking_step_timer = None

        if self.tracking_step_duration <= 0:
            self._tracking_step_stop_at = 0.0
            self._tracking_step_armed_at = 0.0
            return

        scheduled_at = now + self.tracking_step_duration
        self._tracking_step_stop_at = scheduled_at
        self._tracking_step_armed_at = now

        timer = threading.Timer(
            self.tracking_step_duration,
            self._tracking_step_timer_fire,
            args=(scheduled_at, now),
        )
        timer.daemon = True
        self._tracking_step_timer = timer
        timer.start()

    def _tracking_step_timer_fire(self, scheduled_at: float, armed_at: float) -> None:
        """Stop an in-flight tracking step from a timer thread.

        The worker loop may not call back into the tracker for several hundred
        milliseconds when realtime inference is sparse. A timer keeps a short
        tracking pulse short instead of letting it free-run until the next
        detection tick.
        """
        with self._lock:
            if self._tracking_step_stop_at != scheduled_at:
                return
            self._stop_tracking_step_locked(time.time(), scheduled_at, armed_at)

    def _stop_tracking_step_locked(self, now: float, scheduled_at: float, armed_at: float) -> None:
        """Stop the current tracking step and log timing diagnostics.

        Caller must hold ``self._lock``.
        """
        self._tracking_step_stop_at = 0.0
        self._tracking_step_armed_at = 0.0
        self._tracking_step_timer = None
        if self._holding_position:
            return
        try:
            self.onvif_client.ptz_stop(self.profile_token)
        except Exception as e:
            PTZ_LOGGER.warning("[STEP_STOP_FAIL] %s", e)
            return
        self._holding_position = True
        late_ms = max(0.0, (now - scheduled_at)) * 1000.0
        slew_ms = max(0.0, (now - armed_at)) * 1000.0 if armed_at > 0 else 0.0
        PTZ_LOGGER.debug(
            "[TRACKING_STEP_STOP] auto-stopped after %.2fs step (late %.0fms)",
            self.tracking_step_duration, late_ms,
        )
        self._log_decision('tracking_step_stop', {
            'step_duration_s': round(self.tracking_step_duration, 3),
            'actual_slew_ms': round(slew_ms, 1),
            'dispatch_late_ms': round(late_ms, 1),
        })

    def _check_tracking_step_expiry(self, now: float) -> None:
        """If a previously-armed tracking step has elapsed, Stop the PTZ.

        Must be called with ``self._lock`` held. Idempotent: clears the
        scheduled stop time so subsequent ticks don't re-issue Stop.
        """
        if self._tracking_step_stop_at <= 0.0:
            return
        if now < self._tracking_step_stop_at:
            return
        self._stop_tracking_step_locked(now, self._tracking_step_stop_at, self._tracking_step_armed_at)

    @staticmethod
    def _bbox_signature(bbox: List[float]) -> Tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in bbox)  # type: ignore[return-value]

    def _is_duplicate_move_frame_locked(self, bbox: List[float], source: str) -> bool:
        """Return True when this move is for the exact frame already commanded.

        Multi-camera co-drivers can both tick the shared tracker. If they read
        the same published cam2 detection, issuing a second ContinuousMove just
        refreshes the velocity and extends the slew for stale evidence.
        """
        if self._current_capture_ts <= 0 or self._last_move_capture_ts <= 0:
            return False
        if abs(self._current_capture_ts - self._last_move_capture_ts) > 1e-3:
            return False
        signature = self._bbox_signature(bbox)
        if source != self._last_move_source:
            return False
        PTZ_LOGGER.info(
            "[DUPLICATE_FRAME_SKIP] Skipping duplicate move for source=%s capture_ts=%.3f bbox=%s previous_bbox=%s",
            source, self._current_capture_ts, signature, self._last_move_bbox_signature,
        )
        self._log_decision('duplicate_frame_skip', {
            'source': source,
            'capture_ts': round(self._current_capture_ts, 3),
            'bbox_px': list(signature),
            'previous_bbox_px': list(self._last_move_bbox_signature) if self._last_move_bbox_signature else None,
        })
        return True

    @staticmethod
    def _velocity_curve(offset: float) -> float:
        """Map a normalized offset in [-1, 1] to a ContinuousMove velocity.

        Softer low-end response than the previous curve to avoid overshooting
        slow-moving targets. The previous curve jumped to 0.30 at |offset|=0.10
        which, combined with continuous-velocity actuation between detection
        ticks, caused the camera to outpace slow-moving deer.

        Profile (|offset| -> |velocity|):
            0.00 -> 0.00
            0.10 -> 0.15   (was 0.30)
            0.25 -> 0.51   (was 0.75)
            1.00 -> 1.00
        """
        abs_offset = abs(offset)
        if abs_offset < 0.10:
            speed = abs_offset * 1.5            # 0.00 -> 0.15
        elif abs_offset < 0.25:
            speed = 0.15 + (abs_offset - 0.10) * 2.4   # 0.15 -> 0.51
        else:
            speed = 0.51 + (abs_offset - 0.25) * 0.6533  # 0.51 -> 1.00
        speed = max(-1.0, min(1.0, speed if offset >= 0 else -speed))
        return speed

    def _apply_low_fill_cap(
        self,
        pan_velocity: float,
        tilt_velocity: float,
        current_fill: float,
        offset_x: float = 1.0,
        offset_y: float = 1.0,
    ) -> Tuple[float, float, bool]:
        """Cap |pan|/|tilt| velocity when the target is small in frame.

        The cap scales proportionally to the *overall* offset magnitude: at
        |offset| >= ``low_fill_cap_full_offset`` the full cap is allowed;
        below that, the cap is reduced linearly so the commanded velocity
        tapers toward zero as we approach center. This prevents the
        controller from issuing the same maximum pulse for a 0.13 offset and
        a 0.46 offset (which produced overshoot of small offsets and
        undershoot of large ones with sparse detection ticks).

        The scale deliberately uses the offset *magnitude* rather than each
        axis's own offset. Scaling per-axis crushed the axis the animal was
        actually moving along whenever the other axis happened to dominate:
        with the subject at offset (-0.074, +0.479) the pan cap collapsed to
        0.15 * 0.074/0.40 = 0.028, a 4x cut on the axis that needed the
        correction, purely because it was more off-centre vertically.

        Returns the (possibly clamped) velocities and whether a cap was applied.
        """
        if current_fill <= 0 or current_fill >= self.low_fill_threshold:
            return pan_velocity, tilt_velocity, False
        base_cap = max(0.0, min(1.0, self.low_fill_velocity_cap))
        full_off = max(1e-3, self.low_fill_cap_full_offset)
        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5
        cap_scale = max(0.0, min(1.0, offset_magnitude / full_off))
        cap_pan = base_cap * cap_scale
        cap_tilt = base_cap * cap_scale
        capped = False
        if abs(pan_velocity) > cap_pan:
            pan_velocity = cap_pan if pan_velocity > 0 else -cap_pan
            capped = True
        if abs(tilt_velocity) > cap_tilt:
            tilt_velocity = cap_tilt if tilt_velocity > 0 else -cap_tilt
            capped = True
        return pan_velocity, tilt_velocity, capped

    @staticmethod
    def _bbox_overlap_ratio(
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
    ) -> float:
        """Return fraction of bbox_a covered by bbox_b."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox_area = max((ax2 - ax1) * (ay2 - ay1), 1e-9)
        return inter_area / bbox_area

    def _get_current_target_ptz_locked(self) -> Optional[Dict[str, float]]:
        """Read current PTZ position, requiring pan, tilt, and zoom."""
        try:
            position = self.onvif_client.ptz_get_position(self.profile_token)
        except Exception as exc:
            PTZ_LOGGER.debug("[VIS_RECOVERY_POSITION_FAIL] %s", exc)
            return None
        if not position.get('available'):
            return None
        pan = position.get('pan')
        tilt = position.get('tilt')
        zoom = position.get('zoom')
        if pan is None or tilt is None or zoom is None:
            return None
        return {
            'pan': float(pan),
            'tilt': float(tilt),
            'zoom': max(0.0, min(1.0, float(zoom))),
        }

    def _predict_current_target_fov_locked(
        self,
        ptz_position: Dict[str, float],
    ) -> Optional[Tuple[float, float, float, float]]:
        """Predict cam2's current footprint in normalized cam1 coordinates."""
        if self.zoom_fov_calibration is None:
            return None
        base_fov = self.zoom_fov_calibration.get_fov_at_zoom(ptz_position['zoom'])
        if base_fov is None:
            return None

        base_width = max(1e-6, base_fov[2] - base_fov[0])
        base_height = max(1e-6, base_fov[3] - base_fov[1])

        pan_range = max(1e-6, self.calibration.pan_max - self.calibration.pan_min)
        tilt_range = max(1e-6, self.calibration.tilt_max - self.calibration.tilt_min)
        center_x = self.calibration.pan_center_x + (
            ptz_position['pan'] / pan_range
        ) * self.calibration.pan_scale
        center_y = self.calibration.tilt_center_y - (
            ptz_position['tilt'] / tilt_range
        ) * self.calibration.tilt_scale
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))

        half_width = base_width / 2.0
        half_height = base_height / 2.0
        return (
            max(0.0, center_x - half_width),
            max(0.0, center_y - half_height),
            min(1.0, center_x + half_width),
            min(1.0, center_y + half_height),
        )

    def _do_visibility_recovery_from_source(
        self,
        detections: List['Detection'],
        frame_width: int,
        frame_height: int,
        source_camera_id: str,
        target_camera_id: str,
        time_since_target: float,
    ) -> Optional[bool]:
        """Use cam1 plus cam2's calibrated footprint to recover a lost target.

        Returns None when calibration/position data is unavailable and callers
        should use the legacy source-camera fallback path.
        """
        if not self.visibility_recovery_enabled or self.zoom_fov_calibration is None:
            return None
        if not detections or frame_width <= 0 or frame_height <= 0:
            return None

        ptz_position = self._get_current_target_ptz_locked()
        if ptz_position is None:
            self._log_decision('visibility_recovery_unavailable', {
                'reason': 'ptz_position_unavailable',
                'source_camera': source_camera_id,
                'target_camera': target_camera_id,
            })
            return None
        fov = self._predict_current_target_fov_locked(ptz_position)
        if fov is None:
            self._log_decision('visibility_recovery_unavailable', {
                'reason': 'fov_unavailable',
                'source_camera': source_camera_id,
                'target_camera': target_camera_id,
                'zoom': round(ptz_position['zoom'], 3),
            })
            return None

        best = max(detections, key=lambda detection: detection.confidence)
        bbox = best.bbox
        norm_bbox = (
            max(0.0, min(1.0, bbox[0] / frame_width)),
            max(0.0, min(1.0, bbox[1] / frame_height)),
            max(0.0, min(1.0, bbox[2] / frame_width)),
            max(0.0, min(1.0, bbox[3] / frame_height)),
        )
        center_x = (norm_bbox[0] + norm_bbox[2]) / 2.0
        center_y = (norm_bbox[1] + norm_bbox[3]) / 2.0
        fov_center_x = (fov[0] + fov[2]) / 2.0
        fov_center_y = (fov[1] + fov[3]) / 2.0
        fov_width = max(1e-6, fov[2] - fov[0])
        fov_height = max(1e-6, fov[3] - fov[1])
        overlap = self._bbox_overlap_ratio(norm_bbox, fov)
        source_fill = max(
            (bbox[2] - bbox[0]) / frame_width,
            (bbox[3] - bbox[1]) / frame_height,
        )

        inside = overlap >= self.visibility_recovery_min_overlap
        margin_x = fov_width * self.visibility_recovery_edge_margin
        margin_y = fov_height * self.visibility_recovery_edge_margin
        near_edge = inside and (
            center_x <= fov[0] + margin_x
            or center_x >= fov[2] - margin_x
            or center_y <= fov[1] + margin_y
            or center_y >= fov[3] - margin_y
        )

        pan_offset = (center_x - fov_center_x) / max(0.1, self.calibration.pan_scale)
        tilt_offset = (fov_center_y - center_y) / max(0.1, self.calibration.tilt_scale)
        pan_velocity = self._velocity_curve(max(-1.0, min(1.0, pan_offset)))
        tilt_velocity = self._velocity_curve(max(-1.0, min(1.0, tilt_offset)))
        cap = max(0.01, min(1.0, self.visibility_recovery_velocity_cap))
        pan_velocity = max(-cap, min(cap, pan_velocity))
        tilt_velocity = max(-cap, min(cap, tilt_velocity))
        offset_magnitude = (pan_offset ** 2 + tilt_offset ** 2) ** 0.5

        if inside and not near_edge:
            if offset_magnitude < self.min_move_threshold:
                pan_velocity = 0.0
                tilt_velocity = 0.0
            if (
                source_fill <= self.visibility_recovery_zoom_in_fill_threshold
                and ptz_position['zoom'] <= self.visibility_recovery_zoom_in_max_zoom
            ):
                zoom_velocity = max(0.0, min(1.0, self.visibility_recovery_zoom_in_velocity))
                event = (
                    'cam2_lost_cam1_inside_fov_zoom_in'
                    if offset_magnitude < self.min_move_threshold
                    else 'cam2_lost_cam1_inside_fov_recenter_zoom_in'
                )
            else:
                zoom_velocity = min(0.0, max(-1.0, self.visibility_recovery_zoom_out_velocity))
                event = (
                    'cam2_lost_cam1_inside_fov_zoom_out'
                    if offset_magnitude < self.min_move_threshold
                    else 'cam2_lost_cam1_inside_fov_recenter_zoom_out'
                )
        elif inside:
            zoom_velocity = min(0.0, max(-1.0, self.visibility_recovery_zoom_out_velocity))
            event = 'cam2_lost_cam1_edge_recenter_zoom_out'
        else:
            zoom_velocity = min(0.0, max(-1.0, self.visibility_recovery_zoom_out_velocity))
            event = 'cam2_lost_cam1_outside_fov_recenter'

        if self._is_duplicate_move_frame_locked(bbox, f'{source_camera_id}_visibility_recovery'):
            return False

        details = {
            'source_camera': source_camera_id,
            'target_camera': target_camera_id,
            'species': best.species,
            'confidence': round(best.confidence, 3),
            'time_since_target': round(time_since_target, 2),
            'source_fill_pct': round(source_fill * 100.0, 2),
            'target_overlap': round(overlap, 3),
            'near_edge': bool(near_edge),
            'offset_magnitude': round(offset_magnitude, 3),
            'current_ptz': {
                'pan': round(ptz_position['pan'], 3),
                'tilt': round(ptz_position['tilt'], 3),
                'zoom': round(ptz_position['zoom'], 3),
            },
            'predicted_fov': [round(value, 3) for value in fov],
            'source_bbox_norm': [round(value, 3) for value in norm_bbox],
            'velocity': {
                'pan': round(pan_velocity, 3),
                'tilt': round(tilt_velocity, 3),
                'zoom': round(zoom_velocity, 3),
            },
        }
        self._last_tracked_species = best.species
        self._tracking_lost_logged_at = 0.0
        PTZ_LOGGER.info(
            "[VISIBILITY_RECOVERY] %s overlap=%.2f edge=%s fill=%.1f%% "
            "vel=(%.2f, %.2f, %.2f)",
            event, overlap, near_edge, source_fill * 100.0,
            pan_velocity, tilt_velocity, zoom_velocity,
        )
        self._log_decision(event, details)

        try:
            self.onvif_client.ptz_move(
                self.profile_token, pan_velocity, tilt_velocity, zoom_velocity,
            )
            self._last_move_time = time.time()
            self._last_move_capture_ts = self._current_capture_ts
            self._last_move_source = f'{source_camera_id}_visibility_recovery'
            self._last_move_bbox_signature = self._bbox_signature(bbox)
            self._holding_position = False
            self._arm_tracking_step(self._last_move_time)
            return True
        except Exception as exc:
            PTZ_LOGGER.error("[VISIBILITY_RECOVERY_ONVIF_ERROR] %s", exc)
            self._log_decision('error', {
                'command': 'visibility_recovery_move',
                'error': str(exc),
                **details,
            })
            return False

    def is_settling(self, settle_time: float = 0.5) -> bool:
        """Check if the PTZ camera is still settling after a move.
        
        Args:
            settle_time: Seconds to wait after last move before considering
                         the camera stable. Default 0.5s.
        
        Returns:
            True if the camera moved within the last settle_time seconds.
        """
        if settle_time <= 0 or self._last_move_time == 0:
            return False
        return (time.time() - self._last_move_time) < settle_time
    
    def _resolve_presets(self) -> None:
        """Resolve preset names to tokens."""
        if not self.patrol_presets:
            return
            
        try:
            available = self.onvif_client.ptz_get_presets(self.profile_token)
            # Build token map first so a name->token lookup can never collide
            # with another preset's token (token wins on conflict).
            token_set = {p.get('token') for p in available if p.get('token')}
            preset_map: Dict[str, str] = {tok: tok for tok in token_set}
            for p in available:
                tok = p.get('token')
                name = p.get('name')
                if tok and name and name not in token_set:
                    preset_map[name] = tok

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
        """Enable auto-tracking with patrol mode (legacy - enables both)."""
        self.set_patrol_enabled(True)
        self.set_track_enabled(True)
    
    def set_patrol_enabled(self, enabled: bool) -> None:
        """Enable or disable patrol mode independently."""
        with self._lock:
            self._patrol_active = enabled

            if enabled:
                # Resolve presets if configured
                if self.patrol_presets and not self._preset_tokens:
                    self._resolve_presets()

                # Start patrol if not currently tracking
                if self._mode != PTZMode.TRACKING:
                    self._mode = PTZMode.PATROL
                    self._patrol_velocity = None  # Force re-issue on next patrol tick
                    if self._preset_tokens:
                        LOGGER.info("PTZ preset patrol enabled - cycling %d positions", len(self._preset_tokens))
                        self._goto_current_preset()
                    else:
                        LOGGER.info("PTZ patrol mode enabled - continuous sweep")
            else:
                # If patrol disabled and not tracking, go idle
                if self._mode == PTZMode.PATROL:
                    self._mode = PTZMode.IDLE
                    self._patrol_velocity = None
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception:
                        pass
                LOGGER.info("PTZ patrol disabled")
    
    def set_track_enabled(self, enabled: bool) -> None:
        """Enable or disable object tracking independently."""
        with self._lock:
            self._track_active = enabled

            if enabled:
                LOGGER.info("PTZ tracking enabled")
            else:
                LOGGER.info("PTZ tracking disabled")
                # Clear lock state so it can't collide with a new ByteTrack
                # id of 1/2/... when tracking is re-enabled later.
                self._reset_lock_state_locked()
                # If currently tracking, either return to patrol or go idle
                if self._mode == PTZMode.TRACKING:
                    if self._patrol_active:
                        self._mode = PTZMode.PATROL
                        self._patrol_velocity = None
                        if self._preset_tokens:
                            self._goto_current_preset()
                        LOGGER.info("PTZ returning to patrol (tracking disabled)")
                    else:
                        self._mode = PTZMode.IDLE
                        try:
                            self.onvif_client.ptz_stop(self.profile_token)
                        except Exception:
                            pass

    def _reset_lock_state_locked(self) -> None:
        """Reset all target-lock state. Caller must hold self._lock."""
        self._locked_track_id = None
        self._locked_bbox_center = None
        self._locked_source_camera = None
        self._locked_species = None
        self._consecutive_lock_misses = 0
        self._challenger_track_id = None
        self._challenger_streak = 0
        self._pending_takeover_frames = 0
        self._lock_motion_anchor = None
        self._lock_motion_anchor_time = 0.0

    def clear_lock(self) -> None:
        """Public: clear the target lock (e.g. on event boundary)."""
        with self._lock:
            self._reset_lock_state_locked()
    
    def _goto_current_preset(self) -> None:
        """Move to current preset in the patrol sequence."""
        if not self._preset_tokens:
            return
        try:
            preset = self._preset_tokens[self._current_preset_index]
            self.onvif_client.ptz_goto_preset(self.profile_token, preset, speed=0.3)
            self._preset_arrival_time = time.time()
            self._last_move_time = self._preset_arrival_time
            LOGGER.info("Moving to patrol preset %d/%d: %s", 
                       self._current_preset_index + 1, len(self._preset_tokens), preset)
        except Exception as e:
            LOGGER.error("Failed to go to preset: %s", e)
    
    def stop_tracking(self) -> None:
        """Disable auto-tracking (legacy - disables both)."""
        self.set_patrol_enabled(False)
        self.set_track_enabled(False)
        with self._lock:
            self._mode = PTZMode.IDLE
            self._patrol_velocity = None
            self._reset_lock_state_locked()
            try:
                self.onvif_client.ptz_stop(self.profile_token)
            except Exception:
                pass
        LOGGER.info("PTZ tracking disabled")
    
    def is_patrol_enabled(self) -> bool:
        """Check if patrol is enabled."""
        return self._patrol_active
    
    def is_track_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._track_active
    
    def get_mode(self) -> str:
        """Get current PTZ mode as string."""
        return self._mode.value
    
    def _do_patrol(self) -> None:
        """Execute patrol pattern - either preset-based or continuous sweep."""
        now = time.time()
        
        # Preset-based patrol
        if self._preset_tokens:
            time_at_preset = now - self._preset_arrival_time
            # Log every 5 seconds to avoid spam
            if int(time_at_preset) % 5 == 0 and int(time_at_preset) > 0:
                LOGGER.debug(
                    "Patrol: at preset %d/%d for %.0fs (dwell=%.0fs)",
                    self._current_preset_index + 1, len(self._preset_tokens),
                    time_at_preset, self.patrol_dwell_time
                )
            # Check if dwell time has elapsed
            if time_at_preset > self.patrol_dwell_time:
                # Move to next preset
                old_index = self._current_preset_index
                self._current_preset_index = (self._current_preset_index + 1) % len(self._preset_tokens)
                LOGGER.info(
                    "Patrol advancing: preset %d -> %d (was at preset for %.1fs)",
                    old_index + 1, self._current_preset_index + 1, time_at_preset
                )
                self._goto_current_preset()
            return
        
        # Continuous sweep patrol (fallback if no presets)
        # At very slow speed 0.08, need ~90 seconds to cover full pan range
        sweep_duration = 90.0  # seconds per sweep direction
        direction_changed = False

        if now - self._patrol_reverse_time > sweep_duration:
            self._patrol_direction *= -1
            self._patrol_reverse_time = now
            direction_changed = True
            LOGGER.info("Patrol reversing direction: %s",
                        "right" if self._patrol_direction > 0 else "left")

        # Pan sweep at patrol speed (0.08 = very slow, good for small/distant animals)
        pan_vel = self.patrol_speed * self._patrol_direction
        desired = (pan_vel, 0.0, 0.0)

        # Only re-issue ContinuousMove when the velocity actually changes.
        # ONVIF ContinuousMove runs until Stop or another Move; spamming it
        # every tick (a) hammers the camera and (b) keeps refreshing
        # _last_move_time which would permanently trip the worker's settle
        # gate and prevent any inference from running.
        if self._patrol_velocity == desired and not direction_changed:
            return

        try:
            self.onvif_client.ptz_move(
                self.profile_token,
                pan_vel,
                0.0,  # No tilt during patrol
                0.0   # No zoom change during patrol
            )
            self._patrol_velocity = desired
            # Intentionally do NOT update _last_move_time for ongoing patrol
            # velocity. Patrol is a sustained motion, not a discrete reposition;
            # the settle gate is meant for discrete jumps. Direction reversals
            # do count as a discrete change.
            if direction_changed:
                self._last_move_time = time.time()
        except Exception as e:
            LOGGER.error("Patrol move error: %s", e)
            self._patrol_velocity = None
    
    def update_calibration(self, pan_scale: float, tilt_scale: float, 
                           pan_center_x: float, tilt_center_y: float) -> None:
        """Update calibration parameters (e.g., from auto-calibration results)."""
        with self._lock:
            self.calibration.pan_scale = pan_scale
            self.calibration.tilt_scale = tilt_scale
            self.calibration.pan_center_x = pan_center_x
            self.calibration.tilt_center_y = tilt_center_y
        LOGGER.info(
            "PTZ calibration updated: pan_scale=%.3f, tilt_scale=%.3f, center=(%.3f, %.3f)",
            pan_scale, tilt_scale, pan_center_x, tilt_center_y
        )
    
    def update(
        self,
        detections: List['Detection'],
        frame_width: int,
        frame_height: int,
        frame_capture_ts: Optional[float] = None,
    ) -> bool:
        """Process detections and move PTZ if needed.

        State machine:
        - PATROL: Sweeping to find objects. On detection -> TRACKING
        - TRACKING: Following object. On lost object -> wait, then PATROL
        - IDLE: Tracking disabled

        Args:
            detections: List of Detection objects from wide-angle camera
            frame_width: Width of the detection frame
            frame_height: Height of the detection frame
            frame_capture_ts: Wall-clock timestamp when the frame these
                detections came from was pulled off the RTSP stream. Used
                only for diagnostic logging (frame_age_ms on each move /
                deadzone decision). Pass None when unknown.

        Returns:
            True if PTZ was moved, False otherwise
        """
        with self._lock:
            self._current_capture_ts = frame_capture_ts or 0.0
            try:
                return self._update_locked(detections, frame_width, frame_height)
            finally:
                self._current_capture_ts = 0.0

    def _update_locked(self, detections: List['Detection'], frame_width: int, frame_height: int) -> bool:
        """Internal update method, must be called with lock held."""
        # Need at least one of patrol or track enabled
        if not self._patrol_active and not self._track_active:
            return False

        # Rate limit updates
        now = time.time()
        # Auto-stop expired tracking step BEFORE rate-limiting so it always
        # fires within ~1 worker tick of the scheduled time.
        self._check_tracking_step_expiry(now)
        # Record the sighting BEFORE the rate limit can discard this tick.
        # A detection that arrives inside the update_interval window still
        # proves the subject is present, and must not count toward the
        # return-to-patrol timer.
        if detections and self._track_active:
            self._last_target_seen_time = now
        if now - self._last_update < self.update_interval:
            PTZ_LOGGER.debug(
                "[RATE_LIMIT] Skipping update, %.2fs since last (interval=%.2fs)",
                now - self._last_update, self.update_interval
            )
            return False

        self._last_update = now

        # Filter out small detections (likely leaves, noise, distant objects)
        # but keep the currently-locked target even if it shrunk (H2).
        detections = self._filter_small_detections(
            detections, frame_width, frame_height,
            protect_track_id=self._locked_track_id,
            protect_center=self._locked_bbox_center,
        )

        # Handle state transitions based on detections
        if detections and self._track_active:
            # We have detections and tracking is enabled - switch to tracking mode
            self._last_detection_time = now
            self._tracking_lost_logged_at = 0.0  # Reset - we have detections again
            self._last_tracked_species = detections[0].species  # Remember what we're tracking
            self._last_detection_source = "single"  # Single camera mode

            PTZ_LOGGER.debug(
                "[DETECTIONS] %d objects detected, track_active=%s",
                len(detections), self._track_active
            )
            for i, det in enumerate(detections):
                PTZ_LOGGER.debug(
                    "  [DET %d] species=%s conf=%.1f%% bbox=[%.0f,%.0f,%.0f,%.0f] track_id=%s",
                    i, det.species, det.confidence * 100,
                    det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                    getattr(det, 'track_id', 'N/A')
                )

            if self._mode != PTZMode.TRACKING:
                PTZ_LOGGER.info(
                    "[MODE_CHANGE] %s -> TRACKING (detected %s at %.1f%%)",
                    self._mode.value, detections[0].species, detections[0].confidence * 100
                )
                self._log_decision('mode_change', {
                    'from': self._mode.value,
                    'to': 'tracking',
                    'trigger': f"{detections[0].species} ({detections[0].confidence*100:.1f}%)",
                    'detection_count': len(detections),
                })
                self._mode = PTZMode.TRACKING
                LOGGER.info("PTZ switching to TRACKING mode - object detected")

            return self._do_tracking(detections, frame_width, frame_height)
        else:
            # No detections or tracking disabled
            if detections and not self._track_active:
                # Detections exist but tracking is disabled - log this prominently
                PTZ_LOGGER.warning(
                    "[TRACK_DISABLED] %d detections ignored - tracking not enabled (track_active=%s)",
                    len(detections), self._track_active
                )
                # Log this as a decision so it shows in the web UI
                self._log_decision('track_disabled', {
                    'detection_count': len(detections),
                    'species': detections[0].species if detections else None,
                    'reason': 'tracking not enabled in config',
                })

            # Use consolidated no-detection handler
            return self._handle_no_detections(now)

    def _filter_small_detections(
        self, detections: List['Detection'], frame_width: int, frame_height: int,
        protect_track_id: Optional[int] = None,
        protect_center: Optional[Tuple[float, float]] = None,
    ) -> List['Detection']:
        """Filter out detections smaller than min_detection_area.

        H2: the locked target is exempted from filtering. Otherwise an animal
        that walks farther away (and shrinks below min_detection_area) is
        silently dropped, the lock is released after _lock_miss_limit, and
        tracking gives up exactly when the operator most wants it to hold.
        """
        if not detections or (self.min_detection_area <= 0 and self.max_detection_area <= 0):
            return detections

        frame_area = frame_width * frame_height
        min_area_pixels = self.min_detection_area * frame_area
        max_area_pixels = self.max_detection_area * frame_area if self.max_detection_area > 0 else None
        filtered: List['Detection'] = []
        for det in detections:
            det_width = det.bbox[2] - det.bbox[0]
            det_height = det.bbox[3] - det.bbox[1]
            det_area = det_width * det_height

            # Reject impossibly-large detections (no exemption -- a real
            # subject does not fill 70%+ of a zoomed PTZ frame; this is a
            # MegaDetector misfire on motion blur or a static feature).
            if max_area_pixels is not None and det_area > max_area_pixels:
                PTZ_LOGGER.debug(
                    "[SIZE_FILTER] Rejecting oversized detection: %s area=%.0fpx (%.1f%%) > max=%.1f%%",
                    det.species, det_area, (det_area/frame_area)*100,
                    self.max_detection_area*100
                )
                continue

            if det_area >= min_area_pixels:
                filtered.append(det)
                continue

            # Exemption: keep this detection if it matches the current lock,
            # either by track_id or by spatial proximity to the last locked
            # center. Use a generous radius (20% of frame) since the lock may
            # have drifted over the last few frames.
            tid = getattr(det, 'track_id', None)
            if protect_track_id is not None and tid == protect_track_id:
                PTZ_LOGGER.debug(
                    "[SIZE_FILTER_EXEMPT] Keeping shrunk locked target track_id=%s",
                    tid
                )
                filtered.append(det)
                continue
            if protect_center is not None:
                cx = (det.bbox[0] + det.bbox[2]) / 2 / max(frame_width, 1)
                cy = (det.bbox[1] + det.bbox[3]) / 2 / max(frame_height, 1)
                dist = ((cx - protect_center[0]) ** 2 + (cy - protect_center[1]) ** 2) ** 0.5
                if dist < 0.20:
                    PTZ_LOGGER.debug(
                        "[SIZE_FILTER_EXEMPT] Keeping shrunk lock-neighbor at dist=%.3f",
                        dist
                    )
                    filtered.append(det)
                    continue

            PTZ_LOGGER.debug(
                "[SIZE_FILTER] Ignoring small detection: %s area=%.0fpx (%.2f%%) < min=%.0fpx (%.2f%%)",
                det.species, det_area, (det_area/frame_area)*100,
                min_area_pixels, self.min_detection_area*100
            )
        return filtered

    def update_multi_camera(
        self,
        camera_detections: Dict[str, Tuple[List['Detection'], int, int]],
        source_camera_id: str,
        target_camera_id: str,
        frame_capture_ts: Optional[float] = None,
    ) -> bool:
        """Process detections from multiple cameras for PTZ tracking.

        This method enables cam2 (the zoom/PTZ camera) to take over tracking
        once it can see the object. Logic:
        1. If target camera (cam2) has detections → use those for fine tracking
        2. Else if source camera (cam1) has detections → use those to reposition
        3. Else → no detections, handle patrol/idle transition

        Args:
            camera_detections: Dict mapping camera_id -> (detections, frame_width, frame_height)
            source_camera_id: ID of the wide-angle source camera (typically 'cam1')
            target_camera_id: ID of the PTZ target camera (typically 'cam2')
            frame_capture_ts: Wall-clock timestamp of the freshest contributing
                frame. Used only for diagnostic frame_age_ms logging.

        Returns:
            True if PTZ was moved, False otherwise
        """
        with self._lock:
            self._current_capture_ts = frame_capture_ts or 0.0
            try:
                return self._update_multi_camera_locked(
                    camera_detections, source_camera_id, target_camera_id
                )
            finally:
                self._current_capture_ts = 0.0

    def _update_multi_camera_locked(
        self,
        camera_detections: Dict[str, Tuple[List['Detection'], int, int]],
        source_camera_id: str,
        target_camera_id: str,
    ) -> bool:
        """Internal multi-camera update, must be called with lock held."""
        # Need at least one of patrol or track enabled
        if not self._patrol_active and not self._track_active:
            return False

        # Rate limit updates
        now = time.time()
        # Auto-stop expired tracking step BEFORE rate-limiting so it always
        # fires within ~1 worker tick of the scheduled time.
        self._check_tracking_step_expiry(now)

        # Extract detections from each camera. Done BEFORE the rate limit so a
        # sighting that lands inside the update_interval window still refreshes
        # the "target seen" clock instead of counting toward return-to-patrol.
        source_data = camera_detections.get(source_camera_id)
        target_data = camera_detections.get(target_camera_id)

        source_detections = source_data[0] if source_data else []
        target_detections = target_data[0] if target_data else []

        if (source_detections or target_detections) and self._track_active:
            self._last_target_seen_time = now

        if now - self._last_update < self.update_interval:
            return False

        self._last_update = now

        # Capture "investigate candidates" BEFORE size filtering removes them.
        # These are source-camera detections that fall in the size band
        # [investigate_min_area, min_detection_area). Too small to drive
        # tracking confidently from cam1 alone, but big enough that they
        # might be a distant animal worth zooming in on.
        investigate_candidates: List['Detection'] = []
        if (
            self.investigate_enabled
            and source_detections
            and source_data
            and self.min_detection_area > 0
            and self.investigate_min_area > 0
            and self.investigate_min_area < self.min_detection_area
        ):
            sw, sh = source_data[1], source_data[2]
            sf_area = max(sw * sh, 1)
            min_px = self.min_detection_area * sf_area
            inv_min_px = self.investigate_min_area * sf_area
            for d in source_detections:
                w = max(d.bbox[2] - d.bbox[0], 0)
                h = max(d.bbox[3] - d.bbox[1], 0)
                a = w * h
                if inv_min_px <= a < min_px:
                    investigate_candidates.append(d)

        # Filter out small detections using shared method (exempt locked target).
        #
        # While we're already in TRACKING mode we deliberately skip the size
        # filter for the SOURCE (wide) camera. The intent of the filter is to
        # ignore wind-blown leaves / noise during patrol; once we have an
        # actual subject being tracked, the same animal at distance often
        # shrinks below min_detection_area in cam1's wide view (e.g. a dog
        # walking away). Dropping it then makes the tracker think the
        # object disappeared and prematurely return to PATROL even though
        # cam1 is detecting the subject continuously. The locked-track
        # exemption alone doesn't help when the lock was previously held by
        # the target (cam2) and the subject only re-appears in cam1.
        skip_source_size_filter = (
            self._mode == PTZMode.TRACKING
            and self._track_active
        )
        if source_detections and source_data and not skip_source_size_filter:
            source_detections = self._filter_small_detections(
                source_detections, source_data[1], source_data[2],
                protect_track_id=self._locked_track_id if self._locked_source_camera == source_camera_id else None,
                protect_center=self._locked_bbox_center if self._locked_source_camera == source_camera_id else None,
            )
        if target_detections and target_data:
            target_detections = self._filter_small_detections(
                target_detections, target_data[1], target_data[2],
                protect_track_id=self._locked_track_id if self._locked_source_camera == target_camera_id else None,
                protect_center=self._locked_bbox_center if self._locked_source_camera == target_camera_id else None,
            )

        # Log what we have
        PTZ_LOGGER.debug(
            "[MULTI_CAM] source(%s)=%d dets, target(%s)=%d dets",
            source_camera_id, len(source_detections),
            target_camera_id, len(target_detections)
        )

        # Determine which detections to use
        # Priority: target camera (cam2) > source camera (cam1)
        # H8: cam2 must demonstrate continuity (matching species or
        # near-center detection) for _pending_takeover_required consecutive
        # frames before its detections are allowed to drive PTZ. Otherwise
        # any unrelated cam2 detection (e.g. a different animal that
        # wandered into the zoom view) silently steals the lock.
        target_can_take_over = False
        if not target_detections:
            # Target lost the object (or hasn't seen it yet) -- any partial
            # takeover sequence we were accumulating is broken; require it
            # to start over from zero on the next confirmed sighting.
            self._pending_takeover_frames = 0
        if target_detections and self._track_active:
            # If we already had cam2 driving, or no existing lock at all, no
            # extra confirmation needed.
            if (self._locked_source_camera == target_camera_id
                    or self._locked_source_camera is None):
                target_can_take_over = True
            else:
                # Continuity check: prefer same-species match; if no species
                # lock, require a near-frame-center detection (cam1's PTZ
                # command was meant to put the locked target near the center
                # of cam2's frame).
                tw, th = target_data[1], target_data[2]
                ok = False
                if self._locked_species:
                    locked = self._locked_species
                    for d in target_detections:
                        ds = d.species or ''
                        # Accept exact match, OR a more specific
                        # specialization (e.g. lock='animal',
                        # detection='animal_mammalia_carnivora_ursidae')
                        # which commonly happens once cam2 zooms in and
                        # the post-classifier sharpens the label. Also
                        # accept the inverse: locked species is more
                        # specific than what cam2 currently reports
                        # (briefly degraded classification).
                        if (ds == locked
                                or (locked and ds.startswith(locked + '_'))
                                or (ds and locked.startswith(ds + '_'))
                                or locked == 'animal'
                                or ds == 'animal'):
                            ok = True
                            break
                if not ok:
                    for d in target_detections:
                        cx = (d.bbox[0] + d.bbox[2]) / 2 / max(tw, 1)
                        cy = (d.bbox[1] + d.bbox[3]) / 2 / max(th, 1)
                        # Within 25% of cam2 frame center
                        if abs(cx - 0.5) < 0.25 and abs(cy - 0.5) < 0.25:
                            ok = True
                            break
                if ok:
                    self._pending_takeover_frames += 1
                    if self._pending_takeover_frames >= self._pending_takeover_required:
                        target_can_take_over = True
                        self._pending_takeover_frames = 0
                        PTZ_LOGGER.info(
                            "[CAM_TAKEOVER_CONFIRMED] target=%s passed continuity check",
                            target_camera_id
                        )
                    else:
                        PTZ_LOGGER.debug(
                            "[CAM_TAKEOVER_PENDING] %d/%d frames",
                            self._pending_takeover_frames,
                            self._pending_takeover_required
                        )
                else:
                    self._pending_takeover_frames = 0
                    PTZ_LOGGER.debug(
                        "[CAM_TAKEOVER_REJECT] no continuity match in %s detections",
                        target_camera_id
                    )

        if target_can_take_over:
            # Target camera can see the object - use its detections for fine tracking
            frame_width, frame_height = target_data[1], target_data[2]

            # Log whether this is takeover (cam1 also sees) or cam2-only tracking
            if source_detections:
                PTZ_LOGGER.info(
                    "[CAM_TAKEOVER] %s has %d detections (cam1 has %d) - using target for fine tracking",
                    target_camera_id, len(target_detections), len(source_detections)
                )
            else:
                PTZ_LOGGER.info(
                    "[CAM2_ONLY_TRACK] %s has %d detections, %s has none - tracking from target camera only",
                    target_camera_id, len(target_detections), source_camera_id
                )
                self._log_decision('cam2_only_track', {
                    'target_camera': target_camera_id,
                    'source_camera': source_camera_id,
                    'detection_count': len(target_detections),
                    'species': target_detections[0].species,
                    'confidence': round(target_detections[0].confidence * 100, 1),
                })

            self._last_detection_time = now
            self._tracking_lost_logged_at = 0.0
            self._last_tracked_species = target_detections[0].species
            self._last_detection_source = target_camera_id

            if self._mode != PTZMode.TRACKING:
                PTZ_LOGGER.info(
                    "[MODE_CHANGE] %s -> TRACKING (detected by %s: %s at %.1f%%)",
                    self._mode.value, target_camera_id,
                    target_detections[0].species, target_detections[0].confidence * 100
                )
                self._log_decision('mode_change', {
                    'from': self._mode.value,
                    'to': 'tracking',
                    'trigger': f"{target_detections[0].species} ({target_detections[0].confidence*100:.1f}%)",
                    'detection_count': len(target_detections),
                    'source_camera': target_camera_id,
                })
                # If we got here from INVESTIGATE, the candidate has been
                # confirmed by cam2 -- clear the investigate state so we
                # don't subsequently log a timeout / cooldown.
                if self._mode == PTZMode.INVESTIGATE:
                    self._maybe_finish_investigate(now, confirmed=True)
                # Fresh event: drop any lock state left over from a previous
                # subject. Otherwise the new sighting must spatially match
                # the stale _locked_bbox_center within 15% of frame, or burn
                # _lock_miss_limit ticks before a fresh lock can form.
                else:
                    self._reset_lock_state_locked()
                self._mode = PTZMode.TRACKING

            # Use target camera's detections - these are most accurate since
            # they show where the object is in the PTZ camera's current view
            return self._do_tracking_from_target(
                target_detections, frame_width, frame_height,
                source_camera=target_camera_id,
            )

        elif source_detections and self._track_active:
            # Only source camera (cam1) sees the object - need to reposition PTZ.
            #
            # If cam2 was driving recently, prefer calibrated visibility
            # recovery over the legacy source-camera fallback. This lets cam1
            # tell us whether cam2 should zoom out, recenter, or cautiously
            # zoom in instead of treating every cam2 miss as the same case.
            frame_width, frame_height = source_data[1], source_data[2]
            if (
                self._last_detection_source == target_camera_id
                and self._last_detection_time > 0.0
            ):
                time_since_target = now - self._last_detection_time
                visibility_recovery = self._do_visibility_recovery_from_source(
                    source_detections,
                    frame_width,
                    frame_height,
                    source_camera_id,
                    target_camera_id,
                    time_since_target,
                )
                if visibility_recovery is not None:
                    return visibility_recovery

            # IMPORTANT: without a calibrated/current cam2 footprint, if the
            # target camera was driving tracking very recently, suppress
            # cam1-driven repositioning for a short window. Cam2's view is
            # more accurate once it has the object framed, and the legacy
            # cam1->cam2 mapping is only reliable near the calibration pose.
            if (
                self._last_detection_source == target_camera_id
                and self._last_detection_time > 0.0
                and (now - self._last_detection_time) < self.cam1_fallback_delay
            ):
                PTZ_LOGGER.info(
                    "[CAM1_SUPPRESSED] %s drove tracking %.2fs ago (<%.1fs); "
                    "ignoring %d cam1 detections, holding cam2 position",
                    target_camera_id,
                    now - self._last_detection_time,
                    self.cam1_fallback_delay,
                    len(source_detections),
                )
                self._log_decision('cam1_suppressed', {
                    'source_camera': source_camera_id,
                    'target_camera': target_camera_id,
                    'time_since_target': round(now - self._last_detection_time, 2),
                    'fallback_delay': self.cam1_fallback_delay,
                    'detection_count': len(source_detections),
                })
                # Hold position rather than slewing on stale cam1 data.
                # Respect move_min_duration so we don't kill an in-flight
                # cam2-driven ContinuousMove.
                time_since_move = now - self._last_move_time
                if (
                    not self._holding_position
                    and time_since_move >= self.move_min_duration
                ):
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception:
                        pass
                    self._holding_position = True
                return False

            PTZ_LOGGER.info(
                "[SOURCE_TRACKING] Only %s sees object - repositioning PTZ",
                source_camera_id
            )

            self._last_detection_time = now
            self._tracking_lost_logged_at = 0.0
            self._last_tracked_species = source_detections[0].species
            self._last_detection_source = source_camera_id

            if self._mode != PTZMode.TRACKING:
                PTZ_LOGGER.info(
                    "[MODE_CHANGE] %s -> TRACKING (detected by %s: %s at %.1f%%)",
                    self._mode.value, source_camera_id,
                    source_detections[0].species, source_detections[0].confidence * 100
                )
                self._log_decision('mode_change', {
                    'from': self._mode.value,
                    'to': 'tracking',
                    'trigger': f"{source_detections[0].species} ({source_detections[0].confidence*100:.1f}%)",
                    'detection_count': len(source_detections),
                    'source_camera': source_camera_id,
                })
                # If we got here from INVESTIGATE, treat the upgraded
                # source detection as confirmation.
                if self._mode == PTZMode.INVESTIGATE:
                    self._maybe_finish_investigate(now, confirmed=True)
                # Fresh event: clear stale lock so the new subject can
                # establish its own lock immediately (see counterpart in
                # the target-camera branch above).
                else:
                    self._reset_lock_state_locked()
                self._mode = PTZMode.TRACKING

            # Use the original tracking method for source camera detections
            return self._do_tracking(
                source_detections, frame_width, frame_height,
                source_camera=source_camera_id,
            )

        else:
            # No qualifying full-size detections from either camera.
            # Before falling through to patrol/idle handling, check whether
            # the source camera has an "investigate candidate" worth zooming
            # in on with cam2.
            if (
                self.investigate_enabled
                and self._track_active
                and investigate_candidates
                and source_data is not None
                and self._mode in (PTZMode.PATROL, PTZMode.IDLE, PTZMode.INVESTIGATE)
            ):
                handled = self._maybe_investigate(
                    investigate_candidates,
                    source_data[1], source_data[2],
                    source_camera_id, target_camera_id,
                    now,
                )
                if handled:
                    return True

            # If we were INVESTIGATE-ing and cam2 hasn't confirmed in time,
            # mark the candidate as a reject and fall back to patrol logic.
            if self._mode == PTZMode.INVESTIGATE:
                self._maybe_finish_investigate(now, confirmed=False)

            # No detections from either camera
            return self._handle_no_detections(now)

    def _do_tracking_from_target(
        self, detections: List['Detection'], frame_width: int, frame_height: int,
        source_camera: Optional[str] = None,
    ) -> bool:
        """Execute tracking using detections from the target/PTZ camera itself.

        When the target camera (cam2) sees the object, we use its detections
        for precise centering. The object's position in cam2's frame directly
        tells us how to move the PTZ.
        """
        if frame_width <= 0 or frame_height <= 0:
            PTZ_LOGGER.error(
                "[INVALID_FRAME_SIZE] _do_tracking_from_target called with frame=%dx%d; skipping",
                frame_width, frame_height
            )
            return False
        PTZ_LOGGER.info(
            "[DO_TRACKING_TARGET] Called with %d detections from PTZ camera, frame=%dx%d",
            len(detections), frame_width, frame_height
        )

        # Select best detection with track persistence (prevents jitter).
        # Pass source_camera explicitly so cross-camera lock state
        # (track_id space, spatial center) is correctly invalidated when the
        # source changes between cam1 (wide) and cam2 (zoom).
        best = self._select_best_detection(
            detections, frame_width, frame_height,
            source_camera=source_camera,
        )
        if best is None:
            # Spatial fallback failed within miss budget -- hold position.
            # Must explicitly issue ptz_stop or the camera keeps moving with
            # whatever velocity the previous ContinuousMove set.
            try:
                self.onvif_client.ptz_stop(self.profile_token)
            except Exception:
                pass
            self._holding_position = True
            return False
        # We're about to move; clear the held flag so the next deadzone hit
        # actually issues a Stop instead of being optimized away.
        self._holding_position = False
        bbox = best.bbox

        PTZ_LOGGER.info(
            "[TARGET_SELECT] Selected %s (%.1f%%) bbox=[%.0f,%.0f,%.0f,%.0f]",
            best.species, best.confidence * 100,
            bbox[0], bbox[1], bbox[2], bbox[3]
        )

        # Calculate bbox center in target camera's frame
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # For target camera tracking, the offset from center directly tells us
        # how to move the PTZ to center the object
        norm_center_x = center_x / frame_width
        norm_center_y = center_y / frame_height

        # How far from center? (0.5, 0.5) = centered
        offset_x = norm_center_x - 0.5  # Positive = object is right of center
        offset_y = 0.5 - norm_center_y  # Positive = object is above center (inverted Y)

        # L1: apply same exponential smoothing as _do_tracking so cam2
        # takeover doesn't feel jerky compared to cam1-driven motion.
        s = self.smoothing
        self._target_pan = self._target_pan * s + offset_x * (1 - s)
        self._target_tilt = self._target_tilt * s + offset_y * (1 - s)
        offset_x = self._target_pan
        offset_y = self._target_tilt

        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5

        PTZ_LOGGER.info(
            "[TARGET_OFFSET] center=(%.0f, %.0f), norm=(%.3f, %.3f), offset=(%.3f, %.3f), mag=%.3f",
            center_x, center_y, norm_center_x, norm_center_y, offset_x, offset_y, offset_magnitude
        )

        # Only move if offset is significant
        if offset_magnitude < self.min_move_threshold:
            PTZ_LOGGER.info(
                "[DEADZONE] Target centered in cam2 - offset=%.3f < threshold=%.3f",
                offset_magnitude, self.min_move_threshold
            )
            _now = time.time()
            _frame_age_ms = (
                round((_now - self._current_capture_ts) * 1000.0, 1)
                if self._current_capture_ts > 0 else None
            )
            self._log_decision('deadzone', {
                'species': best.species,
                'track_id': getattr(best, 'track_id', None),
                'offset_magnitude': round(offset_magnitude, 4),
                'threshold': self.min_move_threshold,
                'source': 'target_camera',
                # Diagnostic context (H, B): bbox in pixels + frame size lets
                # us overlay the decision on the saved MP4 and prove the
                # detection was where the controller thought it was.
                'bbox_px': [round(bbox[0], 1), round(bbox[1], 1),
                            round(bbox[2], 1), round(bbox[3], 1)],
                'frame_size': [frame_width, frame_height],
                'frame_age_ms': _frame_age_ms,
            })
            if not self._holding_position:
                try:
                    self.onvif_client.ptz_stop(self.profile_token)
                    self._holding_position = True
                except Exception as e:
                    PTZ_LOGGER.warning("[DEADZONE_STOP_FAIL] %s", e)
                    # Leave _holding_position False so next frame retries.
            return False

        # Shared velocity curve (softened low-end to avoid overshoot on
        # slow-moving / distant targets).
        pan_velocity = self._velocity_curve(offset_x)
        tilt_velocity = self._velocity_curve(offset_y)
        # Capture pre-cap values so the decision log can show whether the
        # low-fill cap actually intervened (C). If raw == capped, the cap
        # is loose; if they diverge often, the cap is doing the work.
        pan_velocity_raw = pan_velocity
        tilt_velocity_raw = tilt_velocity

        # Calculate zoom velocity based on current vs target fill
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        current_fill = max(bbox_width / frame_width, bbox_height / frame_height)

        # Cap pan/tilt velocity when target is small in frame to prevent
        # the camera from outpacing slow-moving distant animals between
        # detection ticks. Cap is scaled by |offset| so velocity tapers
        # as we near center.
        pan_velocity, tilt_velocity, _capped = self._apply_low_fill_cap(
            pan_velocity, tilt_velocity, current_fill,
            offset_x=offset_x, offset_y=offset_y,
        )

        if current_fill > 0:
            fill_error = self.target_fill_pct - current_fill
            zoom_velocity = fill_error * 1.5
            zoom_velocity = max(-0.3, min(0.3, zoom_velocity))
            # Suppress zoom-in while target is off-center (see _do_tracking).
            if zoom_velocity > 0 and offset_magnitude > self.zoom_in_offset_gate:
                zoom_velocity = 0.0
        else:
            zoom_velocity = 0.0

        move_source = source_camera or 'target_camera'
        if self._is_duplicate_move_frame_locked(bbox, move_source):
            return False

        PTZ_LOGGER.info(
            "[MOVE_TARGET] %s: vel=(pan=%.2f, tilt=%.2f, zoom=%.2f) | "
            "offset=(%.1f%%, %.1f%%) | fill=%.0f%% (target=%.0f%%)",
            best.species, pan_velocity, tilt_velocity, zoom_velocity,
            offset_x * 100, offset_y * 100,
            current_fill * 100, self.target_fill_pct * 100
        )

        try:
            _now = time.time()
            _frame_age_ms = (
                round((_now - self._current_capture_ts) * 1000.0, 1)
                if self._current_capture_ts > 0 else None
            )
            # D: time since previous *issued* move. Captures how long the
            # camera was free-slewing under the prior velocity, including\n            # tracking_step_stop dispatch slop. 0 means first move.
            _gap_since_last_move_ms = (
                round((_now - self._last_move_time) * 1000.0, 1)
                if self._last_move_time > 0 else None
            )
            # Capture-to-capture gap between the frame that drove the last
            # move and the frame driving this one. This is the *actual*
            # interval the controller had to react across, independent of
            # ONVIF dispatch latency.
            _gap_capture_to_capture_ms = (
                round((self._current_capture_ts - self._last_move_capture_ts) * 1000.0, 1)
                if (self._current_capture_ts > 0 and self._last_move_capture_ts > 0)
                else None
            )
            self._log_decision('move', {
                'species': best.species,
                'track_id': getattr(best, 'track_id', None),
                'confidence': round(best.confidence, 3),
                'velocity': {
                    'pan': round(pan_velocity, 3),
                    'tilt': round(tilt_velocity, 3),
                    'zoom': round(zoom_velocity, 3),
                },
                # C: pre-cap velocity + cap-active flag. If raw and capped
                # diverge, the low-fill cap actively reduced the command;
                # if equal, the cap was inert.
                'velocity_raw': {
                    'pan': round(pan_velocity_raw, 3),
                    'tilt': round(tilt_velocity_raw, 3),
                },
                'cap_active': bool(_capped),
                'offset': {
                    'x': round(offset_x, 3),
                    'y': round(offset_y, 3),
                    'magnitude': round(offset_magnitude, 3),
                },
                'fill_pct': round(current_fill * 100, 1),
                'source': 'target_camera',
                # A: pixel-space bbox + frame size for direct overlay.
                'bbox_px': [round(bbox[0], 1), round(bbox[1], 1),
                            round(bbox[2], 1), round(bbox[3], 1)],
                'frame_size': [frame_width, frame_height],
                # B: how stale was the frame at the moment of decision?
                'frame_age_ms': _frame_age_ms,
                # D: inter-move timing.
                'gap_since_last_move_ms': _gap_since_last_move_ms,
                'gap_capture_to_capture_ms': _gap_capture_to_capture_ms,
            })
            self.onvif_client.ptz_move(
                self.profile_token,
                pan_velocity,
                tilt_velocity,
                zoom_velocity
            )
            self._last_move_time = time.time()
            self._last_move_capture_ts = self._current_capture_ts
            self._last_move_source = move_source
            self._last_move_bbox_signature = self._bbox_signature(bbox)
            # We just issued an active move -- we are no longer 'holding'.
            # Without this, a subsequent return to the deadzone would skip
            # ptz_stop because the debounce flag was still latched True.
            self._holding_position = False
            # Bound the slew distance per detection by scheduling an auto-Stop.
            self._arm_tracking_step(self._last_move_time)
            return True
        except Exception as e:
            PTZ_LOGGER.error("[ONVIF_ERROR] ContinuousMove failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # INVESTIGATE mode helpers
    # ------------------------------------------------------------------
    def _purge_investigate_rejects(self, now: float) -> None:
        """Drop expired entries from the investigate-reject cooldown list."""
        if not self._investigate_rejects:
            return
        self._investigate_rejects = [
            r for r in self._investigate_rejects if r[2] > now
        ]

    def _is_in_investigate_cooldown(
        self, norm_x: float, norm_y: float, now: float
    ) -> bool:
        """True if this normalized cam1 location is within cooldown radius
        of any recently-rejected investigation."""
        self._purge_investigate_rejects(now)
        r = self.investigate_cooldown_radius
        for rx, ry, _exp in self._investigate_rejects:
            if abs(rx - norm_x) <= r and abs(ry - norm_y) <= r:
                return True
        return False

    def _maybe_investigate(
        self,
        candidates: List['Detection'],
        frame_width: int,
        frame_height: int,
        source_camera_id: str,
        target_camera_id: str,
        now: float,
    ) -> bool:
        """Either continue an in-flight investigation or start a new one
        targeting the largest non-cooldown candidate. Returns True if PTZ
        was driven this tick."""
        # If we're already investigating, run the stepped-slew state machine:
        # capped pulse -> stop -> settle (let cam2 detect) -> next pulse if
        # cam1 still sees the candidate. This avoids the original behaviour
        # of slewing once at full velocity and overshooting a small target.
        if self._mode == PTZMode.INVESTIGATE:
            elapsed = now - self._investigate_started_at
            if elapsed >= self.investigate_timeout:
                # Caller will mark reject + return to patrol.
                return False

            # If a capped slew is in flight and its step duration has elapsed,
            # halt it so cam2 has a still frame to detect on.
            if self._investigate_step_active:
                step_elapsed = now - self._investigate_step_started_at
                if step_elapsed >= self.investigate_step_duration:
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception as e:
                        PTZ_LOGGER.warning("[INVESTIGATE_STOP_FAIL] %s", e)
                    self._investigate_step_active = False
                    self._investigate_step_stopped_at = now
                    self._holding_position = True
                    PTZ_LOGGER.debug(
                        "[INVESTIGATE_STEP_END] stopped after %.2fs", step_elapsed
                    )
                return True

            # Step is not active. Wait for settle delay, then issue next
            # capped pulse if cam1 still has a candidate near our target.
            since_stop = now - self._investigate_step_stopped_at
            if since_stop < self.investigate_settle_delay:
                return True  # holding, letting cam2 detect

            # Try to refine: re-evaluate candidates and slew toward the one
            # closest to our original investigation target. If none survive
            # the cooldown filter, just hold and wait for confirmation/timeout.
            tgt = self._investigate_target
            if tgt is not None and candidates:
                fw = max(frame_width, 1)
                fh = max(frame_height, 1)
                best = None
                best_dist = 1e9
                for d in candidates:
                    cx = (d.bbox[0] + d.bbox[2]) / 2.0 / fw
                    cy = (d.bbox[1] + d.bbox[3]) / 2.0 / fh
                    dist = ((cx - tgt[0]) ** 2 + (cy - tgt[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best = d
                # Only re-slew if candidate is reasonably close to original
                # target (avoid chasing unrelated noise that pops up elsewhere).
                if best is not None and best_dist <= self.investigate_cooldown_radius:
                    self._do_investigate_slew(
                        best, frame_width, frame_height, source_camera_id, now,
                    )
            return True

        # Pick the largest candidate not in cooldown.
        best = None
        best_area = -1.0
        best_cx = 0.5
        best_cy = 0.5
        fw = max(frame_width, 1)
        fh = max(frame_height, 1)
        for d in candidates:
            cx = (d.bbox[0] + d.bbox[2]) / 2.0 / fw
            cy = (d.bbox[1] + d.bbox[3]) / 2.0 / fh
            if self._is_in_investigate_cooldown(cx, cy, now):
                continue
            a = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            if a > best_area:
                best_area = a
                best = d
                best_cx = cx
                best_cy = cy
        if best is None:
            return False

        # Slew cam2 to point at the candidate using the existing cam1->PTZ
        # mapping. _do_tracking will issue a ContinuousMove and update
        # _last_move_time so move_min_duration applies.
        PTZ_LOGGER.info(
            "[INVESTIGATE] Pointing %s at small %s candidate from %s "
            "(conf=%.0f%%, norm_center=(%.2f, %.2f), area=%.3f%%)",
            target_camera_id, best.species, source_camera_id,
            best.confidence * 100, best_cx, best_cy,
            (best_area / (fw * fh)) * 100.0,
        )
        self._log_decision('investigate_start', {
            'source_camera': source_camera_id,
            'target_camera': target_camera_id,
            'species': best.species,
            'confidence': round(best.confidence, 3),
            'norm_center': [round(best_cx, 3), round(best_cy, 3)],
            'area_pct': round((best_area / (fw * fh)) * 100.0, 3),
            'timeout': self.investigate_timeout,
        })
        prev_mode = self._mode
        self._mode = PTZMode.INVESTIGATE
        self._investigate_started_at = now
        self._investigate_target = (best_cx, best_cy)
        self._investigate_step_stopped_at = 0.0
        self._investigate_step_active = False
        self._holding_position = False
        # Use a capped, stepped slew (with optional zoom-out) instead of the
        # normal full-velocity tracking move so we don't overshoot a small
        # candidate when cam2 is already zoomed in.
        self._do_investigate_slew(
            best, frame_width, frame_height, source_camera_id, now,
        )
        if prev_mode != PTZMode.INVESTIGATE:
            PTZ_LOGGER.info(
                "[MODE_CHANGE] %s -> INVESTIGATE", prev_mode.value
            )
        return True

    def _do_investigate_slew(
        self,
        det: 'Detection',
        frame_width: int,
        frame_height: int,
        source_camera_id: str,
        now: float,
    ) -> None:
        """Issue a single capped pulse toward `det`, optionally zooming out.

        Uses the same pixel->offset math as `_do_tracking` but caps the
        pan/tilt velocity at `investigate_velocity_cap` so the camera takes
        a short, controlled step and can stop in time for cam2's detector
        to see the candidate. The step is auto-halted by `_maybe_investigate`
        once `investigate_step_duration` elapses.
        """
        fw = max(frame_width, 1)
        fh = max(frame_height, 1)
        bbox = det.bbox
        norm_cx = ((bbox[0] + bbox[2]) / 2.0) / fw
        norm_cy = ((bbox[1] + bbox[3]) / 2.0) / fh

        ref_x = self.calibration.pan_center_x
        ref_y = self.calibration.tilt_center_y
        raw_off_x = norm_cx - ref_x
        raw_off_y = ref_y - norm_cy  # Y inverted

        pan_scale = max(0.1, self.calibration.pan_scale)
        tilt_scale = max(0.1, self.calibration.tilt_scale)
        off_x = max(-1.0, min(1.0, raw_off_x / pan_scale))
        off_y = max(-1.0, min(1.0, raw_off_y / tilt_scale))

        # Velocity proportional to offset, but capped so we move in small
        # increments. Sign preserved.
        cap = max(0.05, min(1.0, self.investigate_velocity_cap))
        def _capped(off: float) -> float:
            v = abs(off) * 1.5  # mild proportional response
            v = min(cap, v)
            return v if off >= 0 else -v

        pan_v = _capped(off_x)
        tilt_v = _capped(off_y)
        zoom_v = self.investigate_zoom_velocity if self.investigate_zoom_out else 0.0

        try:
            self.onvif_client.ptz_move(
                self.profile_token, pan_v, tilt_v, zoom_v,
            )
            self._investigate_step_started_at = now
            self._investigate_step_active = True
            self._last_move_time = now
            self._holding_position = False
            PTZ_LOGGER.info(
                "[INVESTIGATE_STEP] cam=%s vel=(pan=%.2f, tilt=%.2f, zoom=%.2f) "
                "offset=(%.2f, %.2f) cap=%.2f dur=%.2fs",
                source_camera_id, pan_v, tilt_v, zoom_v, off_x, off_y,
                cap, self.investigate_step_duration,
            )
            self._log_decision('investigate_step', {
                'velocity': {
                    'pan': round(pan_v, 3),
                    'tilt': round(tilt_v, 3),
                    'zoom': round(zoom_v, 3),
                },
                'offset': {'x': round(off_x, 3), 'y': round(off_y, 3)},
                'cap': cap,
                'step_duration': self.investigate_step_duration,
            })
        except Exception as e:
            PTZ_LOGGER.error("[INVESTIGATE_MOVE_FAIL] %s", e)
            self._investigate_step_active = False

    def _maybe_finish_investigate(self, now: float, confirmed: bool) -> None:
        """Close out an in-flight investigation. If not confirmed, record
        the candidate location in the cooldown list so we don't keep
        re-investigating the same patch of leaves."""
        if self._investigate_target is None:
            return
        cx, cy = self._investigate_target
        if confirmed:
            PTZ_LOGGER.info(
                "[INVESTIGATE_CONFIRMED] cam2 confirmed candidate at (%.2f, %.2f)",
                cx, cy
            )
            self._log_decision('investigate_confirmed', {
                'norm_center': [round(cx, 3), round(cy, 3)],
            })
        else:
            self._investigate_rejects.append(
                (cx, cy, now + self.investigate_cooldown)
            )
            PTZ_LOGGER.info(
                "[INVESTIGATE_TIMEOUT] cam2 did not confirm (%.2f, %.2f) "
                "within %.1fs; cooldown %.0fs",
                cx, cy, self.investigate_timeout, self.investigate_cooldown
            )
            self._log_decision('investigate_rejected', {
                'norm_center': [round(cx, 3), round(cy, 3)],
                'timeout': self.investigate_timeout,
                'cooldown': self.investigate_cooldown,
            })
        self._investigate_target = None
        self._investigate_started_at = 0.0
        self._investigate_step_active = False
        self._investigate_step_started_at = 0.0
        self._investigate_step_stopped_at = 0.0

    def _handle_no_detections(self, now: float) -> bool:
        """Handle case when no detections from any camera."""
        PTZ_LOGGER.debug(
            "[NO_DETECTION] No detections from any camera, mode=%s",
            self._mode.value
        )

        if self._mode == PTZMode.TRACKING:
            # Measure from the most recent *sighting* by any camera, not from
            # the last sighting that drove a move. Moves blank the moved
            # camera's detections for ptz_settle_time, so keying off
            # _last_detection_time alone made the timer run faster the harder
            # we were tracking -- 12 false TRACKING_LOSTs and 3 returns to
            # patrol in 55s on a coyote that post-processing shows was
            # continuously visible.
            time_since_detection = now - max(
                self._last_detection_time, self._last_target_seen_time
            )

            if self._tracking_lost_logged_at == 0.0:
                self._tracking_lost_logged_at = now
                self._holding_position = False  # Reset so we issue one Stop
                PTZ_LOGGER.info(
                    "[TRACKING_LOST] Lost %s (last seen by %s) - waiting %.1fs before patrol",
                    self._last_tracked_species or "object",
                    self._last_detection_source or "unknown",
                    self.patrol_return_delay
                )
                self._log_decision('tracking_lost', {
                    'species': self._last_tracked_species,
                    'last_source': self._last_detection_source,
                    'return_delay': self.patrol_return_delay,
                })

            if time_since_detection > self.patrol_return_delay or not self._track_active:
                if self._patrol_active:
                    PTZ_LOGGER.info(
                        "[MODE_CHANGE] TRACKING -> PATROL (%s lost for %.1fs)",
                        self._last_tracked_species or "object", time_since_detection
                    )
                    self._log_decision('mode_change', {
                        'from': 'tracking',
                        'to': 'patrol',
                        'reason': 'object_lost',
                        'species': self._last_tracked_species,
                        'time_since_detection': round(time_since_detection, 2),
                    })
                    self._mode = PTZMode.PATROL
                    self._tracking_lost_logged_at = 0.0
                    # Reset track lock so we pick fresh when tracking resumes
                    self._reset_lock_state_locked()
                    if self._preset_tokens:
                        self._goto_current_preset()
                else:
                    self._mode = PTZMode.IDLE
                    self._tracking_lost_logged_at = 0.0
                    # Reset track lock
                    self._reset_lock_state_locked()
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception:
                        pass
            else:
                # Still within delay, hold position. Only issue Stop once;
                # repeating it every tick spams the camera unnecessarily.
                #
                # IMPORTANT: when detections are sparse (e.g. a bird showing up
                # every ~1s) we will fall into this branch on the very next
                # tick after issuing a ContinuousMove. Calling ptz_stop
                # immediately kills that move before the camera has had a
                # chance to physically reposition, which makes tracking
                # ineffective. Give the in-flight ContinuousMove a minimum
                # run-time (move_min_duration) before halting it so the
                # camera can actually slew toward the target.
                time_since_move = now - self._last_move_time
                if (
                    not self._holding_position
                    and time_since_move >= self.move_min_duration
                ):
                    try:
                        self.onvif_client.ptz_stop(self.profile_token)
                    except Exception:
                        pass
                    self._holding_position = True
                return False

        # Start patrol if enabled
        if self._patrol_active and self._mode != PTZMode.PATROL:
            self._mode = PTZMode.PATROL
            self._holding_position = False
            self._patrol_velocity = None
            if self._preset_tokens:
                self._current_preset_index = 0
                self._goto_current_preset()
            else:
                self._patrol_reverse_time = time.time()

        if self._mode == PTZMode.PATROL:
            self._do_patrol()
            return True

        return False

    def _select_best_detection(
        self, detections: List['Detection'], frame_width: int, frame_height: int,
        source_camera: Optional[str] = None,
    ) -> Optional['Detection']:
        """Select the best detection to track, with track persistence.
        
        Once locked onto a target (by track_id or spatial proximity), keep
        tracking it to prevent jitter from switching targets every frame.
        Only switch targets when the locked target is truly lost.
        """
        if not detections:
            return None

        # If caller didn't specify, infer source from latest update path.
        if source_camera is None:
            source_camera = self._last_detection_source or None

        # Static-target watchdog: real animals breathe / sway / walk. If
        # the current lock has not seen real motion for a long time, it is
        # almost certainly a stuck false positive (a leaf, a log, a fence
        # post mis-classified as 'animal'). Releasing here lets patrol
        # resume instead of "successfully tracking" a stationary blob
        # forever (the production hang we observed).
        if (self._lock_motion_anchor is not None
                and self._lock_motion_anchor_time > 0):
            stuck_for = time.time() - self._lock_motion_anchor_time
            if stuck_for > self._lock_static_release_sec:
                PTZ_LOGGER.warning(
                    "[LOCK_STATIC_RELEASE] Lock has not moved >%.0fpx norm for %.1fs; "
                    "releasing as suspected false positive (species=%s, track_id=%s)",
                    self._lock_motion_threshold * 100, stuck_for,
                    self._locked_species, self._locked_track_id,
                )
                self._reset_lock_state_locked()

        # H1: a lock center stored in a *different* camera's normalized
        # coordinates is meaningless (cam1 wide vs cam2 zoom). On a true
        # source change, drop the spatial center AND the track_id (each
        # camera has its own ByteTrack instance with an overlapping ID
        # space, so cam1's id=1 colliding with cam2's id=1 would otherwise
        # produce a false LOCK_PERSIST on a totally unrelated animal).
        if (source_camera is not None
                and self._locked_source_camera is not None
                and source_camera != self._locked_source_camera):
            PTZ_LOGGER.info(
                "[LOCK_CROSSCAM] source camera changed %s->%s; invalidating spatial lock and track_id",
                self._locked_source_camera, source_camera
            )
            self._locked_bbox_center = None
            self._locked_track_id = None
            self._challenger_track_id = None
            self._challenger_streak = 0

        # Build a map of track_id -> detection for quick lookup
        track_id_map: Dict[int, 'Detection'] = {}
        for det in detections:
            tid = getattr(det, 'track_id', None)
            if tid is not None:
                track_id_map[tid] = det

        def _record_lock(det: 'Detection', is_new: bool) -> None:
            tid = getattr(det, 'track_id', None)
            self._locked_track_id = tid
            cx = (det.bbox[0] + det.bbox[2]) / 2 / max(frame_width, 1)
            cy = (det.bbox[1] + det.bbox[3]) / 2 / max(frame_height, 1)
            self._locked_bbox_center = (cx, cy)
            self._locked_source_camera = source_camera
            self._locked_species = det.species
            self._consecutive_lock_misses = 0
            self._challenger_track_id = None
            self._challenger_streak = 0
            now_t = time.time()
            # Static-target watchdog: update the motion anchor if the target
            # has moved more than the threshold OR if we don't have an
            # anchor yet (new lock). Otherwise keep the old anchor so its
            # age can grow until release.
            if (self._lock_motion_anchor is None
                    or is_new
                    or abs(cx - self._lock_motion_anchor[0]) > self._lock_motion_threshold
                    or abs(cy - self._lock_motion_anchor[1]) > self._lock_motion_threshold):
                self._lock_motion_anchor = (cx, cy)
                self._lock_motion_anchor_time = now_t
            if is_new:
                self._lock_start_time = time.time()
                # Reset smoothing so the first command after acquiring a new
                # lock isn't blended with whatever residual offset was left
                # over from tracking the previous (now-released) target.
                self._target_pan = 0.0
                self._target_tilt = 0.0
                self._target_zoom = 0.0
                # Clear stale 'we already issued a stop' debounce flag; the
                # new target is at a different location and any prior stop
                # no longer reflects current intent.
                self._holding_position = False

        # 1. If we have a locked track_id, try to keep tracking it
        if self._locked_track_id is not None and self._locked_track_id in track_id_map:
            locked_det = track_id_map[self._locked_track_id]

            # H4: hysteresis on switching to a higher-confidence challenger.
            # Only switch if a *different* track sustains a confidence margin
            # for several consecutive frames. The challenger MUST have a
            # valid track_id; an untracked (None tid) detection just barely
            # above margin would otherwise win the streak and steal the lock.
            challenger = max(detections, key=lambda d: d.confidence)
            ch_tid = getattr(challenger, 'track_id', None)
            if (ch_tid is not None
                    and ch_tid != self._locked_track_id
                    and challenger.confidence >= locked_det.confidence + self._challenger_required_margin):
                if self._challenger_track_id == ch_tid:
                    self._challenger_streak += 1
                else:
                    self._challenger_track_id = ch_tid
                    self._challenger_streak = 1
                if self._challenger_streak >= self._challenger_required_streak:
                    PTZ_LOGGER.info(
                        "[LOCK_SWITCH] Switching lock %s->%s (challenger sustained %.0f%% > locked %.0f%% for %d frames)",
                        self._locked_track_id, ch_tid,
                        challenger.confidence * 100, locked_det.confidence * 100,
                        self._challenger_streak
                    )
                    _record_lock(challenger, is_new=True)
                    return challenger
            else:
                # Challenger condition not met; reset streak
                self._challenger_track_id = None
                self._challenger_streak = 0

            _record_lock(locked_det, is_new=False)
            PTZ_LOGGER.debug(
                "[LOCK_PERSIST] Continuing to track locked target track_id=%d (%s, %.1f%%)",
                self._locked_track_id, locked_det.species, locked_det.confidence * 100
            )
            return locked_det

        # 2. Locked track ID not found - try spatial proximity to last known position
        if self._locked_bbox_center is not None:
            self._consecutive_lock_misses += 1

            if self._consecutive_lock_misses <= self._lock_miss_limit:
                # Find detection closest to last known position
                best_dist = float('inf')
                best_nearby: Optional['Detection'] = None
                for det in detections:
                    cx = (det.bbox[0] + det.bbox[2]) / 2 / max(frame_width, 1)
                    cy = (det.bbox[1] + det.bbox[3]) / 2 / max(frame_height, 1)
                    dist = ((cx - self._locked_bbox_center[0]) ** 2 +
                            (cy - self._locked_bbox_center[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_nearby = det

                # If nearest detection is spatially close (within 15% of frame), use it.
                # Prefer same species if known (handoff continuity).
                if best_nearby is not None and best_dist < 0.15:
                    if (self._locked_species is None
                            or best_nearby.species == self._locked_species):
                        new_tid = getattr(best_nearby, 'track_id', None)
                        PTZ_LOGGER.info(
                            "[LOCK_SPATIAL] Locked track %s lost, nearest detection at dist=%.3f "
                            "(track_id=%s, %s, %.1f%%) - continuing",
                            self._locked_track_id, best_dist,
                            new_tid, best_nearby.species, best_nearby.confidence * 100
                        )
                        _record_lock(best_nearby, is_new=False)
                        return best_nearby
                    else:
                        PTZ_LOGGER.debug(
                            "[LOCK_SPATIAL_REJECT] Nearby detection species %s != locked %s; holding",
                            best_nearby.species, self._locked_species
                        )

                # H3: Spatial fallback failed but we still have miss budget.
                # Do NOT silently re-lock to whatever has the highest confidence
                # (that flips us to an unrelated leaf/animal). Return None to
                # signal "hold position"; the caller will treat this like
                # 'no_detection' and let _handle_no_detections run the
                # patrol_return_delay timer.
                PTZ_LOGGER.info(
                    "[LOCK_HOLD] Lock %s missing (%d/%d misses) and no nearby detection; holding position",
                    self._locked_track_id, self._consecutive_lock_misses, self._lock_miss_limit
                )
                return None
            else:
                # Too many misses - release lock
                PTZ_LOGGER.info(
                    "[LOCK_RELEASE] Releasing lock on track %s after %d misses",
                    self._locked_track_id, self._consecutive_lock_misses
                )
                self._reset_lock_state_locked()

        # 3. No lock - pick highest confidence and establish new lock.
        # Prefer a detection that ByteTrack has actually confirmed
        # (track_id is not None). An unconfirmed detection (track_id=None)
        # is typically a single-frame flash -- locking onto one of those
        # produced the multi-hour stuck-on-static-leaf hang we saw in
        # production: the spatial fallback kept matching the same dead
        # pixel region every frame, so the system "tracked successfully"
        # forever and never returned to patrol.
        tracked = [d for d in detections if getattr(d, 'track_id', None) is not None]
        if tracked:
            best = max(tracked, key=lambda d: d.confidence)
        else:
            # No confirmed tracks -- only lock if confidence is very high
            # (real, strong detection) AND log it as suspicious.
            best = max(detections, key=lambda d: d.confidence)
            if best.confidence < 0.75:
                PTZ_LOGGER.info(
                    "[LOCK_SKIP] No tracked detections and best untracked is only %.1f%% (%s); not locking",
                    best.confidence * 100, best.species
                )
                return None
            PTZ_LOGGER.warning(
                "[LOCK_UNTRACKED] No ByteTrack-confirmed detections; locking onto untracked %s at %.1f%%",
                best.species, best.confidence * 100
            )
        _record_lock(best, is_new=True)
        PTZ_LOGGER.info(
            "[LOCK_NEW] Locked onto new target: track_id=%s, %s (%.1f%%)",
            getattr(best, 'track_id', None), best.species, best.confidence * 100
        )
        return best

    def _do_tracking(
        self, detections: List['Detection'], frame_width: int, frame_height: int,
        source_camera: Optional[str] = None,
    ) -> bool:
        """Execute object tracking logic."""
        if frame_width <= 0 or frame_height <= 0:
            PTZ_LOGGER.error(
                "[INVALID_FRAME_SIZE] _do_tracking called with frame=%dx%d; skipping",
                frame_width, frame_height
            )
            return False
        PTZ_LOGGER.info(
            "[DO_TRACKING] Called with %d detections, frame=%dx%d",
            len(detections), frame_width, frame_height
        )

        # Use the per-call frame size for pixel->normalized math instead of
        # mutating self.calibration. The calibration is shared across cameras
        # in multi-camera setups; mutating its frame_width/height per call
        # would corrupt subsequent calls from a camera with different dims.

        # Select best detection with track persistence (prevents jitter).
        best = self._select_best_detection(
            detections, frame_width, frame_height,
            source_camera=source_camera,
        )
        if best is None:
            # Hold position: stop the camera so it doesn't drift on stale
            # ContinuousMove velocity. _holding_position prevents log/stop
            # spam on subsequent frames.
            if not self._holding_position:
                try:
                    self.onvif_client.ptz_stop(self.profile_token)
                except Exception:
                    pass
                self._holding_position = True
            return False
        # About to issue a move; allow next deadzone/lock-hold to issue Stop.
        self._holding_position = False
        bbox = best.bbox

        PTZ_LOGGER.info(
            "[TARGET_SELECT] Selected %s (%.1f%%) bbox=[%.0f,%.0f,%.0f,%.0f] track_id=%s",
            best.species, best.confidence * 100,
            bbox[0], bbox[1], bbox[2], bbox[3],
            getattr(best, 'track_id', 'N/A')
        )
        
        # Calculate bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        PTZ_LOGGER.info(
            "[BBOX_CENTER] center=(%.0f, %.0f) in frame %dx%d",
            center_x, center_y, frame_width, frame_height
        )

        # Compute zoom target directly from bbox using passed-in frame dims
        # (avoid relying on shared calibration.frame_width/height mutation).
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        current_fill_pre = max(bbox_w / max(frame_width, 1), bbox_h / max(frame_height, 1))
        if current_fill_pre > 0:
            import math
            zoom_factor = self.target_fill_pct / current_fill_pre
            target_zoom = math.log2(max(1.0, zoom_factor)) / 4.0
            target_zoom = max(self.calibration.zoom_min,
                              min(self.calibration.zoom_max, target_zoom))
        else:
            target_zoom = 0.0

        PTZ_LOGGER.info(
            "[COORD_CALC] Pixel center=(%.0f, %.0f) -> zoom target=%.3f",
            center_x, center_y, target_zoom
        )
        
        # Calculate how far we need to move (normalized center offset)
        norm_center_x = center_x / frame_width
        norm_center_y = center_y / frame_height

        # Compute offset from the *PTZ's* optical-center reference point on the
        # source frame (not the geometric center). For self-tracking this is
        # 0.5/0.5; for cross-camera tracking pan_center_x/tilt_center_y
        # compensates for the wide camera and PTZ camera not being perfectly
        # bore-sighted.
        ref_x = self.calibration.pan_center_x
        ref_y = self.calibration.tilt_center_y
        raw_offset_x = norm_center_x - ref_x  # Positive = object is right of PTZ axis
        raw_offset_y = ref_y - norm_center_y  # Positive = object is above PTZ axis (Y inverted)

        # pan_scale / tilt_scale describe what fraction of the wide FOV the PTZ
        # can cover. A smaller scale means a given pixel offset corresponds to
        # a *larger* PTZ angular movement, so divide by the scale to amplify
        # velocity for narrow-coverage PTZs. Guard against degenerate values.
        pan_scale = max(0.1, self.calibration.pan_scale)
        tilt_scale = max(0.1, self.calibration.tilt_scale)
        offset_x = raw_offset_x / pan_scale
        offset_y = raw_offset_y / tilt_scale

        # Clamp to [-1, 1] before velocity curve maps to ContinuousMove range
        offset_x = max(-1.0, min(1.0, offset_x))
        offset_y = max(-1.0, min(1.0, offset_y))

        # Exponential smoothing on the *velocity-driving offset* itself.
        # smoothing=0 -> instant response; smoothing=0.9 -> very smooth.
        s = self.smoothing
        self._target_pan = self._target_pan * s + offset_x * (1 - s)
        self._target_tilt = self._target_tilt * s + offset_y * (1 - s)
        self._target_zoom = self._target_zoom * s + target_zoom * (1 - s)
        offset_x = self._target_pan
        offset_y = self._target_tilt
        
        # Calculate offset magnitude
        offset_magnitude = (offset_x ** 2 + offset_y ** 2) ** 0.5

        PTZ_LOGGER.info(
            "[OFFSET_CALC] offset_x=%.3f, offset_y=%.3f, magnitude=%.3f, threshold=%.3f",
            offset_x, offset_y, offset_magnitude, self.min_move_threshold
        )

        # Only move if offset is significant
        if offset_magnitude < self.min_move_threshold:
            # Object is centered enough, stop movement
            PTZ_LOGGER.info(
                "[DEADZONE] Target centered - offset=%.3f < threshold=%.3f, stopping PTZ",
                offset_magnitude, self.min_move_threshold
            )
            _now = time.time()
            _frame_age_ms = (
                round((_now - self._current_capture_ts) * 1000.0, 1)
                if self._current_capture_ts > 0 else None
            )
            self._log_decision('deadzone', {
                'species': best.species,
                'track_id': getattr(best, 'track_id', None),
                'offset_magnitude': round(offset_magnitude, 4),
                'threshold': self.min_move_threshold,
                # H, B: pixel context + frame staleness for forensics.
                'bbox_px': [round(bbox[0], 1), round(bbox[1], 1),
                            round(bbox[2], 1), round(bbox[3], 1)],
                'frame_size': [frame_width, frame_height],
                'frame_age_ms': _frame_age_ms,
            })
            if not self._holding_position:
                try:
                    self.onvif_client.ptz_stop(self.profile_token)
                    self._holding_position = True
                except Exception as e:
                    PTZ_LOGGER.warning("[DEADZONE_STOP_FAIL] %s", e)
                    # Leave _holding_position False so next frame retries.
            return False

        PTZ_LOGGER.info(
            "[WILL_MOVE] Target NOT centered - offset=%.3f >= threshold=%.3f, will send MOVE command",
            offset_magnitude, self.min_move_threshold
        )

        pan_velocity = self._velocity_curve(offset_x)
        tilt_velocity = self._velocity_curve(offset_y)
        # Capture pre-cap velocity (C) so we can see whether the low-fill
        # cap actually constrained the command vs being inert.
        pan_velocity_raw = pan_velocity
        tilt_velocity_raw = tilt_velocity

        # Calculate zoom velocity based on current vs target fill
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        current_fill = max(bbox_width / frame_width, bbox_height / frame_height)

        # Cap pan/tilt velocity when target is small in frame to prevent
        # overshooting slow-moving distant animals between detection ticks.
        # Cap is scaled by |offset| so velocity tapers as we near center.
        pan_velocity, tilt_velocity, _capped = self._apply_low_fill_cap(
            pan_velocity, tilt_velocity, current_fill,
            offset_x=offset_x, offset_y=offset_y,
        )

        if current_fill > 0:
            fill_error = self.target_fill_pct - current_fill
            # Slower zoom adjustments - zoom changes are more jarring
            zoom_velocity = fill_error * 1.5
            zoom_velocity = max(-0.3, min(0.3, zoom_velocity))
            # Suppress zoom-in while target is off-center: chasing with a
            # narrowing FOV is how we lose small fast-walking animals.
            if zoom_velocity > 0 and offset_magnitude > self.zoom_in_offset_gate:
                zoom_velocity = 0.0
        else:
            zoom_velocity = 0.0

        move_source = source_camera or 'source_camera'
        if self._is_duplicate_move_frame_locked(bbox, move_source):
            return False
        
        LOGGER.debug(
            "PTZ tracking %s: offset=(%.2f, %.2f) mag=%.2f, vel=(%.2f, %.2f, %.2f), fill=%.1f%%",
            best.species, offset_x, offset_y, offset_magnitude, 
            pan_velocity, tilt_velocity, zoom_velocity, current_fill * 100
        )
        
        PTZ_LOGGER.info(
            "[MOVE] %s (track=%s): vel=(pan=%.2f, tilt=%.2f, zoom=%.2f) | "
            "offset=(%.1f%%, %.1f%%) | fill=%.0f%% (target=%.0f%%)",
            best.species, getattr(best, 'track_id', 'N/A'),
            pan_velocity, tilt_velocity, zoom_velocity,
            offset_x * 100, offset_y * 100, 
            current_fill * 100, self.target_fill_pct * 100
        )
        
        try:
            PTZ_LOGGER.debug(
                "[ONVIF_CMD] ContinuousMove: profile=%s, pan=%.3f, tilt=%.3f, zoom=%.3f",
                self.profile_token, pan_velocity, tilt_velocity, zoom_velocity
            )
            self._log_decision('move', {
                'species': best.species,
                'track_id': getattr(best, 'track_id', None),
                'confidence': round(best.confidence, 3),
                'velocity': {
                    'pan': round(pan_velocity, 3),
                    'tilt': round(tilt_velocity, 3),
                    'zoom': round(zoom_velocity, 3),
                },
                # C: pre-cap velocity + cap-active flag.
                'velocity_raw': {
                    'pan': round(pan_velocity_raw, 3),
                    'tilt': round(tilt_velocity_raw, 3),
                },
                'cap_active': bool(_capped),
                'offset': {
                    'x': round(offset_x, 3),
                    'y': round(offset_y, 3),
                    'magnitude': round(offset_magnitude, 3),
                },
                'fill_pct': round(current_fill * 100, 1),
                # A, B, D: pixel bbox + frame staleness + inter-move gaps.
                'bbox_px': [round(bbox[0], 1), round(bbox[1], 1),
                            round(bbox[2], 1), round(bbox[3], 1)],
                'frame_size': [frame_width, frame_height],
                'frame_age_ms': (
                    round((time.time() - self._current_capture_ts) * 1000.0, 1)
                    if self._current_capture_ts > 0 else None
                ),
                'gap_since_last_move_ms': (
                    round((time.time() - self._last_move_time) * 1000.0, 1)
                    if self._last_move_time > 0 else None
                ),
                'gap_capture_to_capture_ms': (
                    round((self._current_capture_ts - self._last_move_capture_ts) * 1000.0, 1)
                    if (self._current_capture_ts > 0 and self._last_move_capture_ts > 0)
                    else None
                ),
            })
            self.onvif_client.ptz_move(
                self.profile_token,
                pan_velocity,
                tilt_velocity,
                zoom_velocity
            )
            self._last_move_time = time.time()
            self._last_move_capture_ts = self._current_capture_ts
            self._last_move_source = move_source
            self._last_move_bbox_signature = self._bbox_signature(bbox)
            self._holding_position = False
            # Bound the slew distance per detection by scheduling an auto-Stop.
            self._arm_tracking_step(self._last_move_time)
            return True
        except Exception as e:
            PTZ_LOGGER.error("[ONVIF_ERROR] ContinuousMove failed: %s", e)
            self._log_decision('error', {
                'command': 'ContinuousMove',
                'error': str(e),
            })
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
        # Snapshot calibration & compute target under the lock so the
        # streaming executor thread can't observe partially-written
        # _target_pan/_target_tilt/_target_zoom values.
        with self._lock:
            cal = PTZCalibration(
                frame_width=frame_width,
                frame_height=frame_height,
                pan_min=self.calibration.pan_min,
                pan_max=self.calibration.pan_max,
                tilt_min=self.calibration.tilt_min,
                tilt_max=self.calibration.tilt_max,
                zoom_min=self.calibration.zoom_min,
                zoom_max=self.calibration.zoom_max,
                pan_center_x=self.calibration.pan_center_x,
                tilt_center_y=self.calibration.tilt_center_y,
                pan_scale=self.calibration.pan_scale,
                tilt_scale=self.calibration.tilt_scale,
            )

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            pan, tilt = cal.pixel_to_ptz(center_x, center_y)
            zoom = cal.bbox_to_zoom(bbox, self.target_fill_pct) if auto_zoom else 0.0

            self._target_pan = pan
            self._target_tilt = tilt
            self._target_zoom = zoom
            self._last_move_time = time.time()

        LOGGER.info("PTZ centering on bbox: pan=%.3f, tilt=%.3f, zoom=%.3f", pan, tilt, zoom)
        # ONVIF call outside the lock (network I/O); onvif_client has its own RLock.
        try:
            self.onvif_client.ptz_move_absolute(self.profile_token, pan, tilt, zoom)
        except Exception as e:
            LOGGER.error("center_on_bbox: ptz_move_absolute failed: %s", e)


def create_ptz_tracker(
    onvif_client: 'OnvifClient',
    profile_token: str,
    config: Optional[Dict] = None
) -> PTZTracker:
    """Create a PTZ tracker with optional configuration.

    Default values are optimized for split-model architecture where YOLO
    provides fast detections (~50-150ms) for responsive real-time tracking.

    Args:
        onvif_client: ONVIF client for PTZ control
        profile_token: ONVIF profile token
        config: Optional dict with calibration/tracking settings:
            - pan_scale: PTZ pan range as fraction of wide-angle FOV
            - tilt_scale: PTZ tilt range as fraction of wide-angle FOV
            - target_fill_pct: Target object fill percentage (default 0.6)
            - smoothing: Movement smoothing factor (default 0.15 for fast response)
            - update_interval: Seconds between PTZ updates (default 0.1 = 10/sec)
            - patrol_enabled: Enable patrol mode when no detections (default True)
            - patrol_speed: Patrol sweep speed (default 0.15)
            - patrol_return_delay: Seconds to wait before returning to patrol (default 5.0)
            - patrol_presets: List of preset tokens/names for patrol (default [])
            - patrol_dwell_time: Seconds to stay at each preset (default 10.0)
            - secondary_cameras: List of camera IDs that can contribute detections
              for multi-camera tracking (e.g., ['cam2'] when cam2 is the PTZ camera)

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
        min_detection_area=config.get('min_detection_area', 0.005),  # Filter small detections (leaves/noise)
        smoothing=config.get('smoothing', 0.15),  # Fast response (was 0.3)
        update_interval=config.get('update_interval', 0.1),  # 10 updates/sec (was 0.2)
        patrol_enabled=config.get('patrol_enabled', True),
        patrol_speed=config.get('patrol_speed', 0.15),
        patrol_return_delay=config.get('patrol_return_delay', 5.0),
        move_min_duration=config.get('move_min_duration', 0.6),
        tracking_step_duration=config.get('tracking_step_duration', 0.35),
        low_fill_threshold=config.get('low_fill_threshold', 0.30),
        low_fill_velocity_cap=config.get('low_fill_velocity_cap', 0.30),
        low_fill_cap_full_offset=config.get('low_fill_cap_full_offset', 0.40),
        cam1_fallback_delay=config.get('cam1_fallback_delay', 3.0),
        investigate_enabled=config.get('investigate_enabled', False),
        investigate_min_area=config.get('investigate_min_area', 0.0005),
        investigate_timeout=config.get('investigate_timeout', 4.0),
        investigate_cooldown=config.get('investigate_cooldown', 30.0),
        investigate_cooldown_radius=config.get('investigate_cooldown_radius', 0.10),
        investigate_velocity_cap=config.get('investigate_velocity_cap', 0.25),
        investigate_step_duration=config.get('investigate_step_duration', 0.35),
        investigate_settle_delay=config.get('investigate_settle_delay', 0.25),
        investigate_zoom_out=config.get('investigate_zoom_out', True),
        investigate_zoom_velocity=config.get('investigate_zoom_velocity', -0.5),
        patrol_presets=config.get('patrol_presets', []),
        patrol_dwell_time=config.get('patrol_dwell_time', 10.0),
        secondary_cameras=config.get('secondary_cameras', []),
        zoom_fov_calibration=load_zoom_fov_calibration(
            config.get('zoom_fov_calibration_path')
        ),
        visibility_recovery_enabled=config.get('visibility_recovery_enabled', True),
        visibility_recovery_min_overlap=config.get('visibility_recovery_min_overlap', 0.50),
        visibility_recovery_edge_margin=config.get('visibility_recovery_edge_margin', 0.12),
        visibility_recovery_zoom_out_velocity=config.get('visibility_recovery_zoom_out_velocity', -0.25),
        visibility_recovery_zoom_in_velocity=config.get('visibility_recovery_zoom_in_velocity', 0.15),
        visibility_recovery_zoom_in_max_zoom=config.get('visibility_recovery_zoom_in_max_zoom', 0.35),
        visibility_recovery_zoom_in_fill_threshold=config.get('visibility_recovery_zoom_in_fill_threshold', 0.03),
        visibility_recovery_velocity_cap=config.get('visibility_recovery_velocity_cap', 0.20),
    )

    return tracker
