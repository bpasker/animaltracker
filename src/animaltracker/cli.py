"""Command-line entrypoints."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from .config import load_runtime_config
from .onvif_client import OnvifClient
from .pipeline import PipelineOrchestrator
from .storage import StorageManager

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_secrets(config_path: str) -> None:
    """Load secrets from secrets.env in the same directory as config."""
    if load_dotenv is None:
        LOGGER.warning("python-dotenv not installed; skipping automatic .env loading")
        return
        
    config_dir = Path(config_path).parent
    secrets_path = config_dir / "secrets.env"
    if secrets_path.exists():
        LOGGER.info("Loading secrets from %s", secrets_path)
        load_dotenv(secrets_path)
    else:
        LOGGER.debug("No secrets.env found at %s", secrets_path)


def cmd_run(args: argparse.Namespace) -> None:
    _load_secrets(args.config)
    runtime = load_runtime_config(args.config)
    orchestrator = PipelineOrchestrator(
        runtime=runtime,
        model_path=args.model,
        camera_filter=args.camera if args.camera else None,
    )
    target_cams = args.camera if args.camera else [cam.id for cam in runtime.cameras]
    LOGGER.info("Launching pipeline for cameras: %s", ", ".join(target_cams))
    asyncio.run(orchestrator.run())


def cmd_discover(args: argparse.Namespace) -> None:
    _load_secrets(args.config)
    runtime = load_runtime_config(args.config)
    for camera in runtime.cameras:
        username, password = camera.onvif.credentials()
        if not username or not password:
            LOGGER.warning("Camera %s missing ONVIF credentials", camera.id)
            continue
        client = OnvifClient(camera.onvif.host, camera.onvif.port, username, password)
        status = client.get_status()
        profiles = client.get_profiles()
        LOGGER.info(
            "Camera %s (%s) -> %s | Profiles: %d",
            camera.id,
            camera.name,
            status,
            len(profiles),
        )
        if args.inspect:
            for profile in profiles:
                LOGGER.info("- %s", profile)


def cmd_cleanup(args: argparse.Namespace) -> None:
    _load_secrets(args.config)
    runtime = load_runtime_config(args.config)
    storage = StorageManager(
        storage_root=Path(runtime.general.storage_root),
        logs_root=Path(runtime.general.logs_root),
    )
    deleted = storage.cleanup(runtime.general.retention.max_days, dry_run=args.dry_run)
    if args.dry_run:
        LOGGER.info("[dry-run] would delete %d clips", len(deleted))
    else:
        LOGGER.info("Deleted %d old clips", len(deleted))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Animal Tracker CLI")
    parser.add_argument("--config", default="config/cameras.yml", help="Config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run streaming pipeline")
    run_cmd.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    run_cmd.add_argument(
        "--camera",
        action="append",
        help="Camera id to run (repeatable); default=all",
    )
    run_cmd.set_defaults(func=cmd_run)

    discover_cmd = sub.add_parser("discover", help="Run ONVIF discovery")
    discover_cmd.add_argument("--inspect", action="store_true", help="Print profile details")
    discover_cmd.set_defaults(func=cmd_discover)

    cleanup_cmd = sub.add_parser("cleanup", help="Prune old clips")
    cleanup_cmd.add_argument("--dry-run", action="store_true")
    cleanup_cmd.set_defaults(func=cmd_cleanup)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
