#!/usr/bin/env python
"""Serve the web UI from the working tree with fake cameras, for UI work.

No RTSP, no models, no GPU: every camera in the config becomes a stand-in
worker (the first one reports as live, the rest as offline), and the config
is COPIED into a scratch directory so saves from the settings page never
touch the file you started from.

    .venv/bin/python scripts/dev_web.py                      # cameras.sample.yml
    .venv/bin/python scripts/dev_web.py config/cameras.yml   # a real config
    .venv/bin/python scripts/dev_web.py --port 8081 --scratch /tmp/at-dev
    .venv/bin/python scripts/dev_web.py --analyzing cam1/2026/09/14/1789385669_animal.mp4

Then open http://localhost:8081/app/settings. The Claude Code Browser pane
starts this through .claude/launch.json ("settings-dev").

The archive's analysis states can be exercised too: an ``<epoch>_animal.mp4``
stub without a ``.log.json`` beside it shows as "Awaiting analysis" (a stand-in
recovery sweeper reports itself enabled), and ``--analyzing <path>`` marks a
clip as in flight so it shows as "Analyzing…".
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from animaltracker import configstore  # noqa: E402
from animaltracker.analysis_recovery import ClipAnalysisRegistry  # noqa: E402
from animaltracker.detector import MODEL_BUSY  # noqa: E402
from animaltracker.web import WebServer  # noqa: E402


class FakeRecovery:
    """Stands in for the pipeline's RecoverySweeper: enabled and running."""

    def status(self) -> dict:
        return {"enabled": True, "running": True, "current": None, "last_sweep_at": None,
                "last_candidates": 0, "processed": 0, "failed": 0}


class FakeClipBuffer:
    """The pre-roll buffer's figures, as /api/monitor reads them."""

    def __init__(self, live: bool) -> None:
        self.max_frames = 200
        self.max_seconds = 10
        self.frame_count = 120 if live else 0
        self.duration = 6.0 if live else 0.0


class FakeDetector:
    """The detector's name, which /api/monitor reports."""

    backend_name = "megadetector (dev stand-in)"


class FakeWorker:
    """Just enough of StreamWorker for the web handlers."""

    def __init__(self, cam, live: bool, runtime=None) -> None:
        self.camera = cam
        self.runtime = runtime
        self.detector = FakeDetector()
        self.live = live
        self.latest_frame = object() if live else None
        self.latest_frame_ts = time.time() if live else 0.0
        self.stream_connected = live
        self.onvif_client = object() if cam.onvif else None
        self.onvif_profile_token = "Profile_1" if cam.onvif else None
        self.ptz_tracker = object() if (cam.ptz_tracking.enabled or cam.ptz_tracking.self_track) else None
        self.latest_detections = []
        self.latest_detection_ts = 0.0
        self.latest_frame_size = (0, 0)
        self.perf_last_snapshot = {}
        # /api/monitor: no event is ever in flight here, but the buffer is
        # read unconditionally.
        self.clip_buffer = FakeClipBuffer(live)
        self.event_state = None
        self.tracking_enabled = bool(cam.ptz_tracking.enabled or cam.ptz_tracking.self_track)
        self.storage = None   # set by main(), for save_manual_clip
        self.blurry = False   # --blurry: every frame turned away by the blur filter

    def get_perf_stats(self):
        """The monitor's detector card reads this; a live camera checks
        about one frame in six, a --blurry one none at all."""
        if not self.live and not self.blurry:
            return {}
        checked, blurry = (0.0, 4.7) if self.blurry else (3.5, 0.0)
        return {
            'window_sec': 10.0, 'capture_fps': 20.0, 'inferred_fps': checked,
            'frames_dropped_busy': 150 if self.blurry else 165,
            'skipped_blur_fps': blurry, 'skipped_settle_fps': 0.0,
            'frame_age_avg_ms': 160.0, 'at': time.time(),
        }

    def save_manual_clip(self):
        """A stub clip, so the Live page's "Save clip" and the palette work here."""
        if not self.live or self.storage is None:
            return None
        name = f"manual_{self.camera.id}_{int(time.time())}.mp4"
        path = self.storage / "clips" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(0.5)                       # the real write takes seconds
        path.write_bytes(b"\x00" * 4096)
        return name


def fake_model_work(analysing: bool) -> None:
    """Feed the detector card's meter: a 126 ms live check every 285 ms, and
    with --analyzing a 300 ms species-analysis pass every second."""
    last_analysis = 0.0
    while True:
        start = MODEL_BUSY.now()
        MODEL_BUSY.record("live", start, start + 0.126)
        if analysing and start - last_analysis >= 1.0:
            MODEL_BUSY.record("analysis", start + 0.05, start + 0.35)
            last_analysis = start
        time.sleep(0.285)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(REPO / "config" / "cameras.sample.yml"))
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--scratch", default=None, help="directory for the config copy, storage and logs")
    ap.add_argument("--analyzing", action="append", default=[], metavar="CLIP",
                    help="clip path (relative to storage/clips) to show as being analysed; repeatable")
    ap.add_argument("--blurry", action="append", default=[], metavar="CAMERA",
                    help="camera id whose frames the blur filter rejects (monitor card); repeatable")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scratch = Path(args.scratch) if args.scratch else Path(tempfile.gettempdir()) / "animaltracker-dev-web"
    cfg_path = scratch / "config" / "cameras.yml"
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, cfg_path)
        # Point the paths at the scratch dir so nothing real is touched.
        raw = configstore.load_raw(cfg_path)
        raw.setdefault("general", {})["storage_root"] = str(scratch / "storage")
        raw["general"]["logs_root"] = str(scratch / "logs")
        cfg_path.write_text(configstore.dump_yaml(raw), encoding="utf-8")
        logging.info("Copied %s to %s", args.config, cfg_path)
    else:
        logging.info("Reusing %s (delete it to start over)", cfg_path)

    runtime = configstore.validate(configstore.load_raw(cfg_path))
    (scratch / "storage" / "clips").mkdir(parents=True, exist_ok=True)
    workers = {cam.id: FakeWorker(cam, live=(i == 0), runtime=runtime)
               for i, cam in enumerate(runtime.cameras)}
    for w in workers.values():
        w.storage = scratch / "storage"
        w.blurry = w.camera.id in args.blurry

    registry = ClipAnalysisRegistry()
    for rel in args.analyzing:
        registry.begin(scratch / "storage" / "clips" / rel, source="demo")

    server = WebServer(
        workers,
        storage_root=scratch / "storage",
        logs_root=scratch / "logs",
        port=args.port,
        config_path=cfg_path,
        runtime=runtime,
        analysis_registry=registry,
        recovery=FakeRecovery(),
    )

    async def run() -> None:
        async def heartbeat() -> None:
            while True:
                for w in workers.values():
                    if w.live:
                        w.latest_frame_ts = time.time()
                await asyncio.sleep(1)
        asyncio.create_task(heartbeat())
        threading.Thread(target=fake_model_work, args=(bool(args.analyzing),), daemon=True).start()
        await server.start()

    asyncio.run(run())


if __name__ == "__main__":
    main()
