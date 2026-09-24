"""A cheap motion test that decides whether a frame is worth the live detector.

Every camera shares one live MegaDetector at ~114 ms a check, and it runs
75-92% busy although the yards are still most of the day and night. A
frame in which nothing moved can hold nothing new: whatever the detector
would find there, it found in the last frame that did move. So each frame
is first compared with a small, slowly updated background of the scene,
and only frames with enough changed pixels need the detector.

Three safety valves keep an animal from slipping past it:

* a check every ``heartbeat_s`` whatever the gate says, which also catches
  an animal that walked in and then stood still long enough to fade into
  the background;
* every frame while an event is open, so the detector decides when it ends;
* the first frame after a gap (startup, a reconnect), which has no
  background to compare against.

``MotionGate.measure`` costs well under a millisecond (a resize to
160 pixels wide, a blur and a difference), so it runs on every frame the
camera would have checked. The ``thresholds.motion_gate`` setting chooses
what the verdict does: ``observe`` (the default) only counts what would
have been skipped, and any skipped frame in which the detector did find
something, which is the number that says whether ``on`` is safe.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

MODES = ("off", "observe", "on")


@dataclass
class MotionReading:
    changed_px: int          # pixels of the small image that differ from the background
    total_px: int
    fresh: bool              # no usable background yet: always check

    @property
    def fraction(self) -> float:
        return self.changed_px / self.total_px if self.total_px else 0.0


class MotionGate:
    WIDTH = 160              # the small image; a 1280x720 frame becomes 160x90
    PIXEL_DELTA = 12         # grey levels a pixel must move by to count as changed
    MIN_CHANGED_PX = 6       # changed pixels that make a frame worth checking
    BACKGROUND_TAU_S = 3.0   # how fast a change is absorbed into the background
    HEARTBEAT_S = 5.0        # a check at least this often, motion or not
    STALE_AFTER_S = 10.0     # a background older than this is started again
    SUMMARY_EVERY_S = 60.0

    def __init__(self, camera_id: str = "", clock=time.monotonic) -> None:
        self.camera_id = camera_id
        self._clock = clock
        self._lock = threading.Lock()
        self._background: Optional[np.ndarray] = None
        self._background_at = 0.0
        self._last_check_at: Optional[float] = None
        self._window_start = clock()
        self._window = self._empty_window()
        self.last_summary: dict = {}
        self._last_miss_log = 0.0
        self._frame_size = (0, 0)       # (w, h) of the last frame measured

    # --- measuring ------------------------------------------------------

    def measure(self, frame: np.ndarray) -> MotionReading:
        """Compare ``frame`` with the background, then fold it in."""
        small = self._small_grey(frame)
        self._frame_size = (frame.shape[1], frame.shape[0])
        now = self._clock()
        with self._lock:
            bg = self._background
            stale = bg is None or bg.shape != small.shape or (now - self._background_at) > self.STALE_AFTER_S
            if stale:
                self._background = small.astype(np.float32)
                self._background_at = now
                return MotionReading(changed_px=0, total_px=small.size, fresh=True)
            diff = cv2.absdiff(small, bg.astype(np.uint8))
            changed = int(np.count_nonzero(diff > self.PIXEL_DELTA))
            # Time-based weight, so the background keeps the same memory
            # whether this camera is measured three times a second or ten.
            alpha = 1.0 - math.exp(-max(0.0, now - self._background_at) / self.BACKGROUND_TAU_S)
            cv2.accumulateWeighted(small.astype(np.float32), self._background, alpha)
            self._background_at = now
            return MotionReading(changed_px=changed, total_px=small.size, fresh=False)

    def _small_grey(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        height = max(1, int(round(h * self.WIDTH / float(w))))
        small = cv2.resize(frame, (self.WIDTH, height), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Night infrared noise is per pixel; a small blur keeps it under
        # PIXEL_DELTA without hiding anything animal-sized.
        return cv2.GaussianBlur(small, (5, 5), 0)

    # --- deciding -------------------------------------------------------

    def wants_check(self, reading: MotionReading, event_open: bool) -> str:
        """Why this frame should reach the detector, or "" if it need not."""
        now = self._clock()
        if reading.fresh:
            return "first"
        if event_open:
            return "event"
        if reading.changed_px >= self.MIN_CHANGED_PX:
            return "motion"
        if self._last_check_at is None or (now - self._last_check_at) >= self.HEARTBEAT_S:
            return "heartbeat"
        return ""

    def note_check(self) -> None:
        """A frame went to the detector (or would have, when observing)."""
        self._last_check_at = self._clock()

    # --- accounting -----------------------------------------------------

    @staticmethod
    def _empty_window() -> dict:
        return {"frames": 0, "would_skip": 0, "skipped": 0, "missed": 0,
                "changed_px": [], "hit_px": [], "missed_at": []}

    def record(self, reading: MotionReading, reason: str, skipped: bool, found: int,
               detections=None) -> None:
        """Count one frame; ``found`` is how many detections passed the filters
        (unknown, pass 0, when the frame was skipped for real). ``detections``
        are those detections, so a miss can say where it was: one fixed spot
        all night is a still object the gate is right to skip, boxes that
        wander are animals."""
        w = self._window
        w["frames"] += 1
        if not reason:
            w["would_skip"] += 1
            if skipped:
                w["skipped"] += 1
            elif found:
                w["missed"] += 1
                where = self._describe(detections)
                if where and len(w["missed_at"]) < 5:
                    w["missed_at"].append(where)
                self._log_miss(reading, found, where)
        if not reading.fresh:
            w["changed_px"].append(reading.changed_px)
            if found:
                w["hit_px"].append(reading.changed_px)
        now = self._clock()
        if now - self._window_start >= self.SUMMARY_EVERY_S:
            self._roll_window(now)

    def _describe(self, detections) -> str:
        """The best detection as "species conf at (x,y) size wxh", in fractions of the frame."""
        if not detections:
            return ""
        best = max(detections, key=lambda d: getattr(d, "confidence", 0.0))
        fw, fh = self._frame_size
        bbox = getattr(best, "bbox", None)
        if not bbox or not fw or not fh:
            return f"{getattr(best, 'species', '?')} {getattr(best, 'confidence', 0.0):.2f}"
        x1, y1, x2, y2 = bbox
        return "%s %.2f at (%.2f,%.2f) size %.2fx%.2f" % (
            getattr(best, "species", "?"), getattr(best, "confidence", 0.0),
            (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, (x2 - x1) / fw, (y2 - y1) / fh,
        )

    def _log_miss(self, reading: MotionReading, found: int, where: str = "") -> None:
        now = self._clock()
        if now - self._last_miss_log < 60.0:
            return
        self._last_miss_log = now
        LOGGER.info(
            "[MOTION_GATE] %s: the gate would have skipped a frame with %d detection(s) "
            "(changed_px=%d, needs %d)%s",
            self.camera_id, found, reading.changed_px, self.MIN_CHANGED_PX,
            f": {where}" if where else "",
        )

    def _roll_window(self, now: float) -> None:
        w = self._window
        frames = w["frames"]
        px = sorted(w["changed_px"])
        summary = {
            "window_s": round(now - self._window_start, 1),
            "frames": frames,
            "would_skip": w["would_skip"],
            "skipped": w["skipped"],
            "missed": w["missed"],
            "skip_pct": round(w["would_skip"] / frames * 100.0, 1) if frames else 0.0,
            "changed_px_median": px[len(px) // 2] if px else None,
            "changed_px_p90": px[min(len(px) - 1, int(len(px) * 0.9))] if px else None,
            "hit_px_min": min(w["hit_px"]) if w["hit_px"] else None,
            "missed_at": list(w["missed_at"]),
            "at": time.time(),
        }
        self.last_summary = summary
        if frames:
            LOGGER.info(
                "[MOTION_GATE] %s: %d frames, %s%.0f%% without motion, %d with a detection among them | "
                "changed_px median=%s p90=%s, fewest on a frame with a detection=%s (threshold %d)",
                self.camera_id, frames,
                "skipped " if w["skipped"] else "would skip ", summary["skip_pct"], w["missed"],
                summary["changed_px_median"], summary["changed_px_p90"], summary["hit_px_min"],
                self.MIN_CHANGED_PX,
            )
            if w["missed_at"]:
                LOGGER.info("[MOTION_GATE] %s: skipped frames with a detection: %s",
                            self.camera_id, "; ".join(w["missed_at"]))
        self._window_start = now
        self._window = self._empty_window()
