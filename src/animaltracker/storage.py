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
        """Encode frames using two-step process for browser compatibility."""
        if not frames:
            LOGGER.warning("No frames available for clip %s; skipping", output_path)
            return
            
        height, width = frames[0][1].shape[:2]
        # 1. Write to temporary AVI using MJPG (fast, safe, widely supported by OpenCV)
        temp_avi = output_path.with_suffix(".temp.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))
        
        if not out.isOpened():
            LOGGER.error("Failed to open MJPG VideoWriter for %s", temp_avi)
            return

        try:
            for _, frame in frames:
                out.write(frame)
        finally:
            out.release()
            
        if not temp_avi.exists() or temp_avi.stat().st_size == 0:
            LOGGER.error("Failed to create intermediate AVI %s", temp_avi)
            if temp_avi.exists(): temp_avi.unlink()
            return

        # 2. Convert to browser-friendly MP4 (H.264 + YUV420p) using FFmpeg CLI
        # This avoids the 'malloc' crash from piping raw frames and ensures web compatibility
        tmp_mp4 = output_path.with_suffix(".tmp.mp4")
        
        # Check if ffmpeg is available
        if shutil.which("ffmpeg") is None:
            LOGGER.warning("ffmpeg not found; falling back to OpenCV H.264 encoding (may not play in all browsers)")
            # Fallback: Try to write directly with OpenCV 'avc1'
            # We reuse the frames we already have, or re-read the AVI? 
            # Re-reading AVI is safer than keeping frames in memory if they were large.
            # But here we already wrote them to AVI. Let's just try to convert AVI -> MP4 using OpenCV
            
            cap = cv2.VideoCapture(str(temp_avi))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (width, height))
            
            if not out.isOpened():
                LOGGER.error("Failed to open fallback VideoWriter")
                temp_avi.unlink()
                return
                
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
            finally:
                cap.release()
                out.release()
                temp_avi.unlink()
                
            if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                tmp_mp4.rename(output_path)
                LOGGER.info("Saved clip %s (fallback encoding)", output_path)
            return

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(temp_avi),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # Critical for browser playback
            "-preset", "veryfast",
            "-crf", "23",
            str(tmp_mp4)
        ]
        
        LOGGER.info("Transcoding clip to %s", output_path)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if tmp_mp4.exists() and tmp_mp4.stat().st_size > 0:
                tmp_mp4.rename(output_path)
                LOGGER.info("Saved clip %s (%d bytes)", output_path, output_path.stat().st_size)
            else:
                LOGGER.error("FFmpeg produced empty file for %s", output_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            err_msg = e.stderr.decode() if isinstance(e, subprocess.CalledProcessError) else str(e)
            LOGGER.error("FFmpeg failed: %s", err_msg)
        finally:
            # Cleanup intermediate file
            if temp_avi.exists():
                temp_avi.unlink()
            if tmp_mp4.exists(): # If rename failed
                tmp_mp4.unlink()

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
