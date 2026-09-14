"""Finish the post-processing that a restart interrupted.

Key frames and the species label of a clip come only from SpeciesNet
post-processing, which runs in a thread of the pipeline process and writes
its results (thumbnails, then the ``.log.json`` sidecar) at the very end of
a job. The queue is process memory: a restart from the settings page or a
deploy kills whatever is mid-analysis and nothing ever comes back to it, so
the clip stays ``<epoch>_animal.mp4`` with no sidecar and the archive shows
it as "No frame" for good. On the production host a job takes minutes (a
six-minute clip close to an hour while three cameras share the GPU), so the
window a restart can hit is wide.

Two pieces close that gap:

``ClipAnalysisRegistry``
    Every analysis in flight — a live event's post-processing, a reanalysis
    from the clip page, a recovery — registers the clip path here first, so
    no two of them ever work on one file and the web UI can say "analyzing"
    instead of "no frame". It also counts live jobs, which the sweeper
    defers to.

``RecoverySweeper``
    A daemon thread that, shortly after startup and then periodically, looks
    for clips whose analysis never finished (``find_unfinished_clips``) and
    runs them through the caller's ``process_clip``, one at a time, newest
    first, and only while no live event is waiting for the post-processor.
    It never notifies and never deletes: a recovered clip gets its species,
    key frames and sidecar, and a false positive ends up as an "Animal" clip
    with sample frames rather than being removed hours after the fact.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".log.json"

# Labels the real-time detector gives a clip before post-processing names
# the species. A clip whose name carries only these was never finished.
UNCLASSIFIED_LABELS = frozenset({"animal"})

# A clip written in the last two minutes belongs to a live event that may be
# between its transcode and the start of its analysis; the registry already
# protects that window, this is the belt to its braces.
DEFAULT_MIN_AGE_S = 120.0
DEFAULT_INITIAL_DELAY_S = 90.0
DEFAULT_INTERVAL_S = 30 * 60.0
DEFAULT_LIVE_WAIT_S = 5.0


def sidecar_path(clip_path: Path) -> Path:
    """The post-processor's ``.log.json`` for ``clip_path``."""
    return clip_path.with_name(clip_path.stem + SIDECAR_SUFFIX)


def is_unclassified_clip(clip_path: Path) -> bool:
    """True when the clip still carries the real-time detector's generic label.

    Clip names are ``<epoch>_<label>.mp4``; ``<label>`` may join several
    labels with ``+``. Only names made entirely of generic labels count, so
    a clip the post-processor renamed to a species, or a ``person`` clip,
    is never touched.
    """
    stem = clip_path.stem
    parts = stem.split("_", 1)
    if len(parts) < 2 or not parts[0].isdigit():
        return False
    labels = [p.strip().lower() for p in parts[1].split("+")]
    return bool(labels) and all(label in UNCLASSIFIED_LABELS for label in labels)


def find_unfinished_clips(
    clips_dir: Path,
    *,
    min_age_s: float = DEFAULT_MIN_AGE_S,
    now: Optional[float] = None,
) -> List[Path]:
    """Clips under ``clips_dir/<camera>/`` whose post-processing never finished.

    A finished job always writes the sidecar, whether or not it renamed the
    clip, so "unclassified name and no sidecar" is exactly "never finished".
    The sidecar is checked by exact path, not from a listing: on the NFS
    archive a directory listing can be stale while the file opens fine.
    Manual clips in the root of ``clips_dir`` are not post-processed and are
    skipped. Newest first, so the clip someone is waiting for comes first.
    """
    if not clips_dir.is_dir():
        return []
    now = time.time() if now is None else now
    found: List[tuple] = []
    for camera_dir in sorted(clips_dir.iterdir()):
        if not camera_dir.is_dir():
            continue
        for clip in camera_dir.rglob("*.mp4"):
            if not is_unclassified_clip(clip):
                continue
            try:
                mtime = clip.stat().st_mtime
            except OSError:
                continue
            if now - mtime < min_age_s:
                continue
            if sidecar_path(clip).exists():
                continue
            found.append((mtime, clip))
    found.sort(key=lambda item: item[0], reverse=True)
    return [clip for _, clip in found]


def _key(path) -> str:
    return os.path.abspath(str(path))


class ClipAnalysisRegistry:
    """Thread-safe record of every clip analysis in flight in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, dict] = {}
        self._live = 0

    def begin(self, clip_path, source: str) -> bool:
        """Claim ``clip_path``; False when another job already holds it."""
        key = _key(clip_path)
        with self._lock:
            if key in self._active:
                return False
            self._active[key] = {"path": str(clip_path), "source": source, "started": time.time()}
            return True

    def end(self, clip_path) -> None:
        with self._lock:
            self._active.pop(_key(clip_path), None)

    def is_active(self, clip_path) -> bool:
        with self._lock:
            return _key(clip_path) in self._active

    def source_of(self, clip_path) -> Optional[str]:
        with self._lock:
            entry = self._active.get(_key(clip_path))
            return entry["source"] if entry else None

    def active(self) -> List[dict]:
        """Snapshot of the jobs in flight, oldest first."""
        with self._lock:
            entries = [dict(e) for e in self._active.values()]
        entries.sort(key=lambda e: e["started"])
        return entries

    # --- live-event accounting: the sweeper yields to these -------------

    def live_begin(self) -> None:
        with self._lock:
            self._live += 1

    def live_end(self) -> None:
        with self._lock:
            self._live = max(0, self._live - 1)

    @property
    def live_count(self) -> int:
        with self._lock:
            return self._live

    def live_busy(self) -> bool:
        return self.live_count > 0


class RecoverySweeper(threading.Thread):
    """Background thread that finishes interrupted clip analyses.

    ``process_clip(path) -> bool`` does the work (the caller supplies the
    detector, settings and concurrency limit); ``enabled()`` is read before
    every sweep so the setting applies live. Each clip is attempted once per
    process: a clip that fails (unreadable video, say) is logged and left
    for the next start rather than retried every half hour.
    """

    def __init__(
        self,
        clips_dir: Path,
        registry: ClipAnalysisRegistry,
        process_clip: Callable[[Path], bool],
        *,
        enabled: Callable[[], bool] = lambda: True,
        interval_s: float = DEFAULT_INTERVAL_S,
        initial_delay_s: float = DEFAULT_INITIAL_DELAY_S,
        min_age_s: float = DEFAULT_MIN_AGE_S,
        live_wait_s: float = DEFAULT_LIVE_WAIT_S,
    ) -> None:
        super().__init__(name="analysis-recovery", daemon=True)
        self.clips_dir = Path(clips_dir)
        self.registry = registry
        self.process_clip = process_clip
        self.enabled = enabled
        self.interval_s = float(interval_s)
        self.initial_delay_s = float(initial_delay_s)
        self.min_age_s = float(min_age_s)
        self.live_wait_s = float(live_wait_s)
        # Not ``_stop``: threading.Thread has a private method of that name
        # that ``join`` calls, and shadowing it breaks the join.
        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._attempted: set = set()
        self._current: Optional[str] = None
        self._last_sweep_at: Optional[float] = None
        self._last_candidates = 0
        self._processed = 0
        self._failed = 0

    # --- control ----------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def request_sweep(self) -> None:
        """Run the next sweep as soon as the thread is idle."""
        self._wake.set()

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": bool(self._safe_enabled()),
                "running": self.is_alive() and not self._stop_event.is_set(),
                "current": self._current,
                "last_sweep_at": self._last_sweep_at,
                "last_candidates": self._last_candidates,
                "processed": self._processed,
                "failed": self._failed,
            }

    # --- loop -------------------------------------------------------------

    def run(self) -> None:  # pragma: no cover - exercised through sweep()
        if self._wait(self.initial_delay_s):
            return
        while not self._stop_event.is_set():
            try:
                self.sweep()
            except Exception:  # noqa: BLE001 - the thread must outlive a bad sweep
                LOGGER.exception("Recovery sweep failed")
            if self._wait(self.interval_s):
                return

    def _wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds`` unless woken; True when stopping."""
        self._wake.wait(seconds)
        self._wake.clear()
        return self._stop_event.is_set()

    def _safe_enabled(self) -> bool:
        try:
            return bool(self.enabled())
        except Exception:  # noqa: BLE001
            return False

    def sweep(self) -> int:
        """One pass: process every unfinished clip not already claimed.

        Returns the number of clips processed successfully.
        """
        with self._lock:
            self._last_sweep_at = time.time()
        if not self._safe_enabled():
            LOGGER.debug("Recovery sweep skipped: disabled")
            return 0

        candidates = find_unfinished_clips(self.clips_dir, min_age_s=self.min_age_s)
        with self._lock:
            self._last_candidates = len(candidates)
        if not candidates:
            LOGGER.debug("Recovery sweep: every clip has finished analysis")
            return 0

        todo = [c for c in candidates if _key(c) not in self._attempted and not self.registry.is_active(c)]
        LOGGER.info(
            "Recovery sweep: %d clip(s) never finished analysis, %d to process now%s",
            len(candidates), len(todo),
            "" if len(todo) == len(candidates) else " (the rest are in flight or already attempted)",
        )

        done = 0
        for clip in todo:
            if self._stop_event.is_set():
                break
            if not self._wait_for_idle():
                break
            if self.registry.is_active(clip):
                continue
            self._attempted.add(_key(clip))
            with self._lock:
                self._current = str(clip)
            try:
                ok = bool(self.process_clip(clip))
            except Exception:  # noqa: BLE001
                LOGGER.exception("Recovery of %s raised", clip)
                ok = False
            with self._lock:
                self._current = None
                if ok:
                    self._processed += 1
                else:
                    self._failed += 1
            if ok:
                done += 1
        if todo:
            LOGGER.info("Recovery sweep finished: %d of %d clip(s) recovered", done, len(todo))
        return done

    def _wait_for_idle(self) -> bool:
        """Block while a live event needs the post-processor; False when stopping."""
        waited = False
        while self.registry.live_busy():
            if not waited:
                LOGGER.debug("Recovery waits for live post-processing to finish")
                waited = True
            if self._stop_event.wait(self.live_wait_s):
                return False
        return True


__all__ = [
    "ClipAnalysisRegistry",
    "RecoverySweeper",
    "find_unfinished_clips",
    "is_unclassified_clip",
    "sidecar_path",
]
