"""Storage management for clips and logs."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


@dataclass
class StorageManager:
    storage_root: Path
    logs_root: Path

    def __post_init__(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def build_clip_path(self, camera_id: str, species: str, event_ts: float, ext: str = "mp4") -> Path:
        ts = time.strftime("%Y/%m/%d", time.localtime(event_ts))
        directory = self.storage_root / "clips" / camera_id / ts
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{int(event_ts)}_{species}.{ext}"
        return directory / filename

    def save_snapshot(self, camera_id: str, frame) -> Path:
        import cv2
        path = self.logs_root / f"startup_{camera_id}.jpg"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Saved startup snapshot to %s", path)
        return path

    def write_clip(self, frames: List, output_path: Path, fps: int = 15) -> None:
        """Encode frames using cv2.VideoWriter."""
        if not frames:
            LOGGER.warning("No frames available for clip %s; skipping", output_path)
            return
            
        height, width = frames[0][1].shape[:2]
        tmp_path = output_path.with_suffix(".tmp.mp4")
        
        # Try H.264 (avc1) first as it's required for web playback
        # Fallback to mp4v (MPEG-4) if H.264 is not available
        codecs = ['avc1', 'h264', 'mp4v']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            if out.isOpened():
                LOGGER.info("Encoding clip to %s using %s (%dx%d @ %d fps)", output_path, codec, width, height, fps)
                break
            out.release()
            
        if not out or not out.isOpened():
            LOGGER.error("Failed to open VideoWriter for %s (tried: %s)", tmp_path, codecs)
            return

        try:
            for _, frame in frames:
                out.write(frame)
        finally:
            out.release()
            
        if tmp_path.exists():
            size = tmp_path.stat().st_size
            if size > 0:
                tmp_path.rename(output_path)
                LOGGER.info("Saved clip %s (%d bytes)", output_path, size)
            else:
                LOGGER.error("Generated clip is empty: %s", tmp_path)
                tmp_path.unlink()

    def disk_usage_pct(self) -> float:
        stat = shutil.disk_usage(self.storage_root)
        if stat.total == 0:
            return 0.0
        return stat.used / stat.total * 100

    def cleanup(self, retention_days: int, dry_run: bool = False) -> list[Path]:
        deleted: list[Path] = []
        cutoff = time.time() - retention_days * 86400
        for path in sorted(self.storage_root.glob("clips/*/*/*/*")):
            if path.is_file() and path.stat().st_mtime < cutoff:
                LOGGER.info("%s old clip %s", "Would remove" if dry_run else "Removing", path)
                if not dry_run:
                    path.unlink()
                deleted.append(path)
        return deleted
