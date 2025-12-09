"""Storage management for clips and logs."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
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

    def write_clip(self, frames: List, output_path: Path, fps: int = 15) -> None:
        """Encode frames using ffmpeg via subprocess."""
        if not frames:
            LOGGER.warning("No frames available for clip %s; skipping", output_path)
            return
        tmp_path = output_path.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(tmp_path),
        ]
        LOGGER.info("Encoding clip to %s", output_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for _, frame in frames:
                proc.stdin.write(frame.tobytes())  # type: ignore[arg-type]
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
        tmp_path.rename(output_path)

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
