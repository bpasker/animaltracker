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

# Default storage thresholds
DEFAULT_MIN_FREE_BYTES = 500 * 1024 * 1024  # 500 MB minimum free space
DEFAULT_MAX_UTILIZATION_PCT = 80  # Don't use more than 80% of disk


@dataclass
class StorageManager:
    storage_root: Path
    logs_root: Path
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    max_utilization_pct: int = DEFAULT_MAX_UTILIZATION_PCT

    def __post_init__(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def build_clip_path(self, camera_id: str, species: str, event_ts: float, ext: str = "mp4") -> Path:
        ts = time.strftime("%Y/%m/%d", time.localtime(event_ts))
        directory = self.storage_root / "clips" / camera_id / ts
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{int(event_ts)}_{species}.{ext}"
        return directory / filename

    def build_thumbnail_path(self, clip_path: Path, species: str, index: int = 0) -> Path:
        """Build path for a detection thumbnail associated with a clip.
        
        Thumbnails are stored alongside clips with format: 
        {clip_name}_thumb_{species}.jpg (for index 0)
        {clip_name}_thumb_{species}_{index}.jpg (for index > 0)
        """
        # Create thumbnail filename based on clip name
        clip_stem = clip_path.stem  # e.g., "1766587074_animal+bird"
        if index == 0:
            thumb_filename = f"{clip_stem}_thumb_{species}.jpg"
        else:
            thumb_filename = f"{clip_stem}_thumb_{species}_{index}.jpg"
        return clip_path.parent / thumb_filename

    def save_detection_thumbnails(
        self, 
        clip_path: Path, 
        species_frames: dict
    ) -> List[Path]:
        """Save detection thumbnails for each species detected in a clip.
        
        Args:
            clip_path: Path to the clip file
            species_frames: Dict mapping species names to a list of 
                (frame, confidence, bbox) tuples, where bbox is [x1, y1, x2, y2] or None.
                Multiple detections per species are supported.
                
        Returns:
            List of saved thumbnail paths
        """
        saved_paths = []
        
        for species, detections in species_frames.items():
            # Handle both old format (single tuple) and new format (list of tuples)
            if not isinstance(detections, list):
                detections = [detections]
            
            for idx, detection in enumerate(detections):
                frame, confidence, bbox = detection
                if frame is None:
                    continue
                    
                thumb_path = self.build_thumbnail_path(clip_path, species, idx)
                
                try:
                    # Draw bounding box if available
                    if bbox:
                        frame = frame.copy()
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        # Draw box with green color
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add species label with detection number if multiple
                        if len(detections) > 1:
                            label = f"{species} #{idx+1} ({confidence:.0%})"
                        else:
                            label = f"{species} ({confidence:.0%})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        # Background rectangle for text
                        cv2.rectangle(
                            frame, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width + 4, y1), 
                            (0, 255, 0), 
                            -1
                        )
                        cv2.putText(
                            frame, label, (x1 + 2, y1 - 5), 
                            font, font_scale, (0, 0, 0), thickness
                        )
                    
                    # Save thumbnail
                    cv2.imwrite(str(thumb_path), frame)
                    saved_paths.append(thumb_path)
                    LOGGER.info("Saved detection thumbnail: %s", thumb_path)
                    
                except Exception as e:
                    LOGGER.error("Failed to save thumbnail for %s #%d: %s", species, idx, e)
                
        return saved_paths

    def get_clip_thumbnails(self, clip_path: Path) -> List[dict]:
        """Get all thumbnails associated with a clip.
        
        Returns:
            List of dicts with 'path', 'species', 'rel_path' for each thumbnail
        """
        thumbnails = []
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        
        # Look for thumbnails matching this clip
        for thumb_file in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            # Extract species from filename
            # Format: {timestamp}_{original_species}_thumb_{specific_species}.jpg
            # or: {timestamp}_{original_species}_thumb_{specific_species}_{index}.jpg
            parts = thumb_file.stem.split("_thumb_")
            if len(parts) >= 2:
                species_part = parts[-1]
                # Check if there's an index suffix (e.g., "cardinal_1" or "cardinal_2")
                # Try to split off trailing number
                import re
                match = re.match(r'^(.+?)(?:_(\d+))?$', species_part)
                if match:
                    species_name = match.group(1)
                    detection_num = match.group(2)
                    species = species_name.replace("_", " ").title()
                    if detection_num:
                        species = f"{species} #{int(detection_num) + 1}"
                else:
                    species = species_part.replace("_", " ").title()
            else:
                species = "Unknown"
            
            thumbnails.append({
                'path': thumb_file,
                'species': species,
                'rel_path': thumb_file.relative_to(self.storage_root / "clips")
            })
        
        return thumbnails

    def save_snapshot(self, camera_id: str, frame) -> Path:
        import cv2
        path = self.logs_root / f"startup_{camera_id}.jpg"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Saved startup snapshot to %s", path)
        return path

    def write_clip(self, frames: List, output_path: Path, fps: int = 15) -> None:
        """Encode frames using two-step process for browser compatibility.
        
        Ensures sufficient storage space before writing, removing old clips if needed.
        """
        if not frames:
            LOGGER.warning("No frames available for clip %s; skipping", output_path)
            return
        
        # Ensure we have enough space before writing
        estimated_size = self.estimate_clip_size(frames, fps)
        if not self.ensure_space_for_clip(estimated_size):
            LOGGER.error("Skipping clip %s due to insufficient storage space", output_path)
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

    def get_clips_sorted_by_age(self) -> List[Path]:
        """Get all clip files sorted by modification time (oldest first)."""
        clips = list(self.storage_root.glob("clips/*/*/*/*"))
        clips = [p for p in clips if p.is_file() and p.suffix in (".mp4", ".avi", ".mkv")]
        return sorted(clips, key=lambda p: p.stat().st_mtime)

    def get_free_space(self) -> int:
        """Get free space in bytes on the storage volume."""
        stat = shutil.disk_usage(self.storage_root)
        return stat.free

    def get_total_space(self) -> int:
        """Get total space in bytes on the storage volume."""
        stat = shutil.disk_usage(self.storage_root)
        return stat.total

    def estimate_clip_size(self, frames: List, fps: int = 15) -> int:
        """Estimate the size of a clip based on frame count and resolution.
        
        Uses empirical estimates for H.264 compression ratios.
        Returns estimated size in bytes.
        """
        if not frames:
            return 0
        
        # Get frame dimensions from first frame
        height, width = frames[0][1].shape[:2]
        frame_count = len(frames)
        
        # Estimate bytes per frame for H.264 at CRF 23 (medium quality)
        # Rough estimate: 0.1 bits per pixel for decent quality H.264
        bits_per_pixel = 0.1
        bits_per_frame = width * height * bits_per_pixel
        bytes_per_frame = bits_per_frame / 8
        
        # Add 20% overhead for container, headers, etc.
        estimated_size = int(frame_count * bytes_per_frame * 1.2)
        
        # Minimum estimate of 100KB
        return max(estimated_size, 100 * 1024)

    def has_sufficient_space(self, required_bytes: int) -> bool:
        """Check if there's enough free space for a new clip.
        
        Considers both absolute free space and utilization percentage.
        """
        stat = shutil.disk_usage(self.storage_root)
        
        # Check absolute free space
        if stat.free < self.min_free_bytes + required_bytes:
            return False
        
        # Check utilization percentage after writing
        used_after = stat.used + required_bytes
        utilization_after = (used_after / stat.total) * 100
        if utilization_after > self.max_utilization_pct:
            return False
        
        return True

    def ensure_space_for_clip(self, required_bytes: int) -> bool:
        """Ensure sufficient space exists for a new clip, removing old clips if needed.
        
        Removes oldest clips first until enough space is available or no clips remain.
        
        Returns:
            True if sufficient space is available (or was freed)
            False if unable to free enough space
        """
        if self.has_sufficient_space(required_bytes):
            return True
        
        LOGGER.info(
            "Insufficient storage space. Need %d bytes, have %d bytes free. "
            "Cleaning up old clips...",
            required_bytes, self.get_free_space()
        )
        
        # Get clips sorted oldest first
        old_clips = self.get_clips_sorted_by_age()
        
        freed_count = 0
        for clip_path in old_clips:
            if self.has_sufficient_space(required_bytes):
                LOGGER.info("Freed enough space after removing %d old clips", freed_count)
                return True
            
            try:
                size = clip_path.stat().st_size
                clip_path.unlink()
                freed_count += 1
                LOGGER.info("Removed old clip to free space: %s (%d bytes)", clip_path, size)
            except OSError as e:
                LOGGER.warning("Failed to remove clip %s: %s", clip_path, e)
        
        # Check one more time after removing all possible clips
        if self.has_sufficient_space(required_bytes):
            LOGGER.info("Freed enough space after removing %d old clips", freed_count)
            return True
        
        LOGGER.error(
            "Unable to free enough storage space. Still need %d bytes, have %d bytes free",
            required_bytes, self.get_free_space()
        )
        return False
