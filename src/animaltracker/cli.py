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
        config_path=Path(args.config),
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


def cmd_reprocess(args: argparse.Namespace) -> None:
    """Reprocess clips to improve species classifications."""
    from .detector import create_detector
    from .postprocess import process_all_clips, ClipPostProcessor
    
    _load_secrets(args.config)
    runtime = load_runtime_config(args.config)
    
    # Create detector
    detector_cfg = runtime.general.detector
    detector = create_detector(
        backend=detector_cfg.backend,
        model_path=args.model or detector_cfg.model_path,
        model_version=detector_cfg.speciesnet_version,
        country=detector_cfg.country,
        admin1_region=detector_cfg.admin1_region,
        latitude=detector_cfg.latitude,
        longitude=detector_cfg.longitude,
    )
    LOGGER.info("Using %s detector for reprocessing", detector.backend_name)
    
    storage_root = Path(runtime.general.storage_root)
    
    if args.clip:
        # Process single clip
        clip_path = Path(args.clip)
        if not clip_path.is_absolute():
            clip_path = storage_root / 'clips' / clip_path
        
        processor = ClipPostProcessor(
            detector=detector,
            storage_root=storage_root,
            sample_rate=args.sample_rate,
        )
        
        result = processor.process_clip(
            clip_path,
            update_filename=not args.no_rename,
            regenerate_thumbnails=not args.no_thumbnails,
        )
        
        if result.success:
            LOGGER.info("Processed: %s", result.original_path.name)
            LOGGER.info("  Original species: %s", result.original_species)
            LOGGER.info("  New species: %s (%.1f%% confidence)", 
                       result.new_species, result.confidence * 100)
            LOGGER.info("  Frames analyzed: %d/%d", 
                       result.frames_analyzed, result.total_frames)
            LOGGER.info("  Species found: %s", 
                       ", ".join(result.species_results.keys()) or "None")
            if result.new_path:
                LOGGER.info("  Renamed to: %s", result.new_path.name)
            LOGGER.info("  Thumbnails saved: %d", len(result.thumbnails_saved))
        else:
            LOGGER.error("Failed: %s - %s", result.original_path, result.error)
    else:
        # Process all clips
        results = process_all_clips(
            storage_root=storage_root,
            detector=detector,
            camera_filter=args.camera if args.camera else None,
            update_filenames=not args.no_rename,
            regenerate_thumbnails=not args.no_thumbnails,
            sample_rate=args.sample_rate,
        )
        
        # Print summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        updated = [r for r in results if r.new_path is not None]
        
        LOGGER.info("=" * 50)
        LOGGER.info("Reprocessing Summary:")
        LOGGER.info("  Total clips: %d", len(results))
        LOGGER.info("  Successful: %d", len(successful))
        LOGGER.info("  Failed: %d", len(failed))
        LOGGER.info("  Classifications updated: %d", len(updated))
        
        if updated:
            LOGGER.info("\nUpdated classifications:")
            for r in updated:
                LOGGER.info("  %s: %s -> %s", 
                           r.original_path.name, r.original_species, r.new_species)


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

    reprocess_cmd = sub.add_parser("reprocess", help="Reprocess clips to improve classifications")
    reprocess_cmd.add_argument("--model", help="Model path (overrides config)")
    reprocess_cmd.add_argument(
        "--camera",
        action="append",
        help="Only reprocess clips from this camera (repeatable); default=all",
    )
    reprocess_cmd.add_argument(
        "--clip",
        help="Reprocess a single clip (path relative to clips/ or absolute)",
    )
    reprocess_cmd.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Analyze every Nth frame (default=5, lower=more thorough but slower)",
    )
    reprocess_cmd.add_argument(
        "--no-rename",
        action="store_true",
        help="Don't rename clips even if species changes",
    )
    reprocess_cmd.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="Don't regenerate detection thumbnails",
    )
    reprocess_cmd.set_defaults(func=cmd_reprocess)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
