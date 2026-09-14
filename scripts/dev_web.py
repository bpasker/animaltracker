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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from animaltracker import configstore  # noqa: E402
from animaltracker.analysis_recovery import ClipAnalysisRegistry  # noqa: E402
from animaltracker.web import WebServer  # noqa: E402


class FakeRecovery:
    """Stands in for the pipeline's RecoverySweeper: enabled and running."""

    def status(self) -> dict:
        return {"enabled": True, "running": True, "current": None, "last_sweep_at": None,
                "last_candidates": 0, "processed": 0, "failed": 0}


class FakeWorker:
    """Just enough of StreamWorker for the web handlers."""

    def __init__(self, cam, live: bool) -> None:
        self.camera = cam
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(REPO / "config" / "cameras.sample.yml"))
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--scratch", default=None, help="directory for the config copy, storage and logs")
    ap.add_argument("--analyzing", action="append", default=[], metavar="CLIP",
                    help="clip path (relative to storage/clips) to show as being analysed; repeatable")
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
    workers = {cam.id: FakeWorker(cam, live=(i == 0)) for i, cam in enumerate(runtime.cameras)}

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
        await server.start()

    asyncio.run(run())


if __name__ == "__main__":
    main()
