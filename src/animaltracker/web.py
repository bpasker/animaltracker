import asyncio
import logging
import cv2
import json
import re
import numpy as np
import yaml
from aiohttp import web
from typing import Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone, timedelta

from .species_names import get_common_name, format_species_display

# Central Time with automatic DST handling
try:
    from zoneinfo import ZoneInfo
    CENTRAL_TZ = ZoneInfo("America/Chicago")
except ImportError:
    # Fallback for Python < 3.9: manual DST calculation
    class CentralTime(timezone):
        """Central Time with DST support (CST/CDT)."""
        def __init__(self):
            super().__init__(timedelta(hours=-6), 'CST')
        
        def utcoffset(self, dt):
            if self._is_dst(dt):
                return timedelta(hours=-5)  # CDT
            return timedelta(hours=-6)  # CST
        
        def tzname(self, dt):
            return 'CDT' if self._is_dst(dt) else 'CST'
        
        def _is_dst(self, dt):
            """Check if DST is in effect (2nd Sunday March - 1st Sunday November)."""
            if dt is None:
                return False
            year = dt.year
            # DST starts 2nd Sunday of March at 2:00 AM
            march_start = datetime(year, 3, 8)  # Earliest possible 2nd Sunday
            while march_start.weekday() != 6:  # Find Sunday
                march_start += timedelta(days=1)
            dst_start = march_start.replace(hour=2)
            
            # DST ends 1st Sunday of November at 2:00 AM
            nov_start = datetime(year, 11, 1)
            while nov_start.weekday() != 6:  # Find Sunday
                nov_start += timedelta(days=1)
            dst_end = nov_start.replace(hour=2)
            
            return dst_start <= dt.replace(tzinfo=None) < dst_end
    
    CENTRAL_TZ = CentralTime()

if TYPE_CHECKING:
    from .pipeline import StreamWorker

LOGGER = logging.getLogger(__name__)

class WebServer:
    def __init__(self, workers: Dict[str, 'StreamWorker'], storage_root: Path, logs_root: Path, port: int = 8080, config_path: Path = None, runtime = None):
        self.workers = workers
        self.storage_root = storage_root
        self.logs_root = logs_root
        self.port = port
        self.config_path = config_path
        self.runtime = runtime
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)
        self.app.router.add_post('/save_clip/{camera_id}', self.handle_save_clip)
        self.app.router.add_post('/ptz/{camera_id}', self.handle_ptz)
        self.app.router.add_get('/recordings', self.handle_recordings)
        self.app.router.add_get('/recording/{path:.*}', self.handle_recording_detail)
        self.app.router.add_delete('/recordings', self.handle_delete_recording)
        self.app.router.add_post('/recordings/bulk_delete', self.handle_bulk_delete)
        self.app.router.add_post('/recordings/reprocess', self.handle_reprocess)
        self.app.router.add_get('/recordings/log/{path:.*}', self.handle_get_processing_log)
        # Settings page and API
        self.app.router.add_get('/settings', self.handle_settings_page)
        self.app.router.add_get('/api/settings', self.handle_get_settings)
        self.app.router.add_post('/api/settings', self.handle_update_settings)
        
        # Serve clips directory statically
        clips_path = self.storage_root / 'clips'
        # Ensure it exists so static route doesn't fail on startup
        clips_path.mkdir(parents=True, exist_ok=True)
        self.app.router.add_static('/clips', clips_path, show_index=True)

    async def handle_index(self, request):
        html = """
        <html>
            <head>
                <title>Animal Tracker Dashboard</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <meta name="apple-mobile-web-app-capable" content="yes">
                <meta name="mobile-web-app-capable" content="yes">
                <style>
                    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #1a1a1a; 
                        color: #eee; 
                        margin: 0; 
                        padding: 16px; 
                    }
                    .nav { 
                        display: flex;
                        gap: 12px;
                        margin-bottom: 20px; 
                        padding-bottom: 16px; 
                        border-bottom: 1px solid #333;
                    }
                    .nav a { 
                        color: #fff;
                        background: #333;
                        text-decoration: none; 
                        font-size: 0.95em;
                        font-weight: 500;
                        padding: 10px 16px;
                        border-radius: 8px;
                        transition: background 0.2s;
                    }
                    .nav a:hover, .nav a.active { background: #4CAF50; }
                    h1 { font-size: 1.5em; margin: 0 0 16px 0; font-weight: 600; }
                    h2 { margin: 0 0 12px 0; font-size: 1.1em; font-weight: 500; }
                    .camera-grid { display: flex; flex-direction: column; gap: 16px; }
                    .camera-card { 
                        background: #2a2a2a; 
                        padding: 16px; 
                        border-radius: 12px; 
                    }
                    .camera-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 12px;
                    }
                    .camera-name { font-weight: 600; font-size: 1.1em; }
                    .camera-id { color: #888; font-size: 0.85em; }
                    img { 
                        width: 100%; 
                        height: auto; 
                        border-radius: 8px; 
                        display: block;
                    }
                    button { 
                        background: #4CAF50; 
                        color: white; 
                        border: none; 
                        padding: 12px 20px; 
                        border-radius: 8px; 
                        cursor: pointer; 
                        font-weight: 500;
                        font-size: 0.95em;
                        transition: transform 0.15s, opacity 0.15s;
                        width: 100%;
                        margin-top: 12px;
                    }
                    button:active { transform: scale(0.98); opacity: 0.9; }
                    .ptz-controls { 
                        margin-top: 12px; 
                        background: #222; 
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .ptz-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 12px;
                        cursor: pointer;
                        user-select: none;
                    }
                    .ptz-header:hover { background: #333; }
                    .ptz-header span { font-size: 0.9em; color: #aaa; }
                    .ptz-toggle { 
                        font-size: 0.8em; 
                        color: #666;
                        transition: transform 0.2s;
                    }
                    .ptz-controls.expanded .ptz-toggle { transform: rotate(180deg); }
                    .ptz-content {
                        display: none;
                        padding: 0 12px 12px 12px;
                    }
                    .ptz-controls.expanded .ptz-content { display: block; }
                    .ptz-grid {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 8px;
                        max-width: 200px;
                        margin: 0 auto;
                    }
                    .ptz-controls button { 
                        margin: 0;
                        padding: 14px;
                        font-size: 0.85em;
                        background: #444;
                        width: 100%;
                    }
                    .ptz-controls button:active { background: #666; }
                    .ptz-zoom {
                        display: flex;
                        gap: 8px;
                        margin-top: 12px;
                        padding-top: 12px;
                        border-top: 1px solid #333;
                    }
                    .ptz-zoom button { flex: 1; }
                    .toggle-switch {
                        position: relative;
                        width: 50px;
                        height: 28px;
                    }
                    .toggle-switch input {
                        opacity: 0;
                        width: 0;
                        height: 0;
                    }
                    .toggle-slider {
                        position: absolute;
                        cursor: pointer;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background-color: #555;
                        transition: 0.3s;
                        border-radius: 28px;
                    }
                    .toggle-slider:before {
                        position: absolute;
                        content: "";
                        height: 22px;
                        width: 22px;
                        left: 3px;
                        bottom: 3px;
                        background-color: white;
                        transition: 0.3s;
                        border-radius: 50%;
                    }
                    .toggle-switch input:checked + .toggle-slider {
                        background-color: #4CAF50;
                    }
                    .toggle-switch input:checked + .toggle-slider:before {
                        transform: translateX(22px);
                    }
                    @media (min-width: 768px) {
                        body { padding: 24px; max-width: 1200px; margin: 0 auto; }
                        .camera-grid { flex-direction: row; flex-wrap: wrap; }
                        .camera-card { flex: 1 1 calc(50% - 8px); min-width: 300px; }
                    }
                </style>
                <script>
                    function refreshImages() {
                        const images = document.querySelectorAll('img');
                        images.forEach(img => {
                            const src = img.getAttribute('data-src');
                            img.src = src + '?t=' + new Date().getTime();
                        });
                    }
                    setInterval(refreshImages, 2000);

                    function togglePtz(element) {
                        const ptzControls = element.closest('.ptz-controls');
                        ptzControls.classList.toggle('expanded');
                    }

                    async function saveClip(camId) {
                        try {
                            const response = await fetch('/save_clip/' + camId, { method: 'POST' });
                            const text = await response.text();
                            alert(text);
                        } catch (e) {
                            alert('Error saving clip: ' + e);
                        }
                    }

                    async function sendPtz(camId, action, pan, tilt, zoom) {
                        try {
                            await fetch('/ptz/' + camId, {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    action: action,
                                    pan: pan,
                                    tilt: tilt,
                                    zoom: zoom
                                })
                            });
                        } catch (e) {
                            console.error('PTZ error:', e);
                        }
                    }
                </script>
            </head>
            <body>
                <div class="nav">
                    <a href="/" class="active">Live View</a>
                    <a href="/recordings">Recordings</a>
                    <a href="/settings">Settings</a>
                </div>
                <h1>Live View</h1>
                <div class="camera-grid">
        """
        
        for cam_id, worker in self.workers.items():
            cam_name = worker.camera.name
            ptz_html = ""
            if worker.onvif_client and worker.onvif_profile_token:
                ptz_html = f"""
                    <div class="ptz-controls">
                        <div class="ptz-header" onclick="togglePtz(this)">
                            <span>PTZ Controls</span>
                            <span class="ptz-toggle">▼</span>
                        </div>
                        <div class="ptz-content">
                            <div class="ptz-grid">
                                <div></div>
                                <button onmousedown="sendPtz('{cam_id}', 'move', 0, 1, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', 0, 1, 0)" ontouchend="sendPtz('{cam_id}', 'stop')">▲</button>
                                <div></div>
                                <button onmousedown="sendPtz('{cam_id}', 'move', -1, 0, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', -1, 0, 0)" ontouchend="sendPtz('{cam_id}', 'stop')">◄</button>
                                <div></div>
                                <button onmousedown="sendPtz('{cam_id}', 'move', 1, 0, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', 1, 0, 0)" ontouchend="sendPtz('{cam_id}', 'stop')">►</button>
                                <div></div>
                                <button onmousedown="sendPtz('{cam_id}', 'move', 0, -1, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', 0, -1, 0)" ontouchend="sendPtz('{cam_id}', 'stop')">▼</button>
                                <div></div>
                            </div>
                            <div class="ptz-zoom">
                                <button onmousedown="sendPtz('{cam_id}', 'move', 0, 0, 1)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', 0, 0, 1)" ontouchend="sendPtz('{cam_id}', 'stop')">Zoom +</button>
                                <button onmousedown="sendPtz('{cam_id}', 'move', 0, 0, -1)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')" ontouchstart="sendPtz('{cam_id}', 'move', 0, 0, -1)" ontouchend="sendPtz('{cam_id}', 'stop')">Zoom -</button>
                            </div>
                        </div>
                    </div>
                """

            html += f"""
                    <div class="camera-card">
                        <div class="camera-header">
                            <span class="camera-name">{cam_name}</span>
                            <span class="camera-id">{cam_id}</span>
                        </div>
                        <img src="/snapshot/{cam_id}" data-src="/snapshot/{cam_id}" alt="{cam_name}">
                        <button onclick="saveClip('{cam_id}')">Save Last 30s</button>
                        {ptz_html}
                    </div>
            """
            
        html += """
                </div>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_ptz(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or not worker.onvif_client or not worker.onvif_profile_token:
            return web.Response(status=400, text="Camera not found or ONVIF not configured")
            
        try:
            data = await request.json()
            action = data.get('action')
            
            loop = asyncio.get_running_loop()
            
            if action == 'move':
                pan = float(data.get('pan', 0.0))
                tilt = float(data.get('tilt', 0.0))
                zoom = float(data.get('zoom', 0.0))
                await loop.run_in_executor(
                    None, 
                    worker.onvif_client.ptz_move, 
                    worker.onvif_profile_token, 
                    pan, tilt, zoom
                )
            elif action == 'stop':
                await loop.run_in_executor(
                    None, 
                    worker.onvif_client.ptz_stop, 
                    worker.onvif_profile_token
                )
            else:
                return web.Response(status=400, text="Invalid action")
                
            return web.Response(text="OK")
        except Exception as e:
            LOGGER.error(f"PTZ error: {e}")
            return web.Response(status=500, text=str(e))

    def _scan_recordings(self):
        clips_dir = self.storage_root / 'clips'
        if not clips_dir.exists():
            return []

        clips = []
        
        # 1. Check for manual clips in root
        for clip_file in clips_dir.glob('*.mp4'):
            stat = clip_file.stat()
            rel_path = clip_file.relative_to(clips_dir)
            parts = clip_file.name.split('_')
            camera = parts[1] if len(parts) > 1 else 'unknown'
            
            clips.append({
                'path': str(rel_path),
                'camera': camera,
                'date': 'Manual',
                'filename': clip_file.name,
                'time': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ),
                'size': stat.st_size,
                'species': 'Manual clip',
                'thumbnails': []
            })

        # 2. Check for automated clips in subdirectories
        for cam_dir in clips_dir.iterdir():
            if not cam_dir.is_dir(): continue
            
            # Use rglob to find all mp4 files recursively (handles year/month/day structure)
            for clip_file in cam_dir.rglob('*.mp4'):
                stat = clip_file.stat()
                rel_path = clip_file.relative_to(clips_dir)
                
                # Parse species from filename (format: timestamp_species.mp4)
                species = self._parse_species_from_filename(clip_file.name)
                
                # Find associated thumbnails
                thumbnails = self._get_thumbnails_for_clip(clip_file)
                
                clips.append({
                    'path': str(rel_path),
                    'camera': cam_dir.name,
                    'date': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ).strftime('%Y-%m-%d'),
                    'filename': clip_file.name,
                    'time': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ),
                    'size': stat.st_size,
                    'species': species,
                    'thumbnails': thumbnails
                })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)
        return clips

    def _get_thumbnails_for_clip(self, clip_path: Path) -> list:
        """Get all thumbnails associated with a clip file.
        
        Returns list of dicts with 'path' (relative to clips dir), 'species', 'url'
        """
        clips_dir = self.storage_root / 'clips'
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        thumbnails = []
        
        # Debug: Log what we're looking for
        glob_pattern = f"{clip_stem}_thumb_*.jpg"
        LOGGER.debug("Looking for thumbnails in %s with pattern: %s", clip_dir, glob_pattern)
        
        # Look for thumbnails matching this clip
        for thumb_file in clip_dir.glob(glob_pattern):
            LOGGER.debug("Found thumbnail: %s", thumb_file)
            # Extract species from filename: {timestamp}_{species}_thumb_{specific_species}.jpg
            parts = thumb_file.stem.split("_thumb_")
            if len(parts) >= 2:
                raw_species = parts[-1]
                # Remove trailing index numbers (e.g., "bird_1" -> "bird")
                raw_species = re.sub(r'_\d+$', '', raw_species)
                species = get_common_name(raw_species)
            else:
                species = "Unknown"
            
            rel_path = thumb_file.relative_to(clips_dir)
            thumbnails.append({
                'path': str(rel_path),
                'species': species,
                'url': f"/clips/{rel_path}"
            })
        
        LOGGER.debug("Total thumbnails found: %d", len(thumbnails))
        return thumbnails

    def _parse_species_from_filename(self, filename: str) -> str:
        """Extract clean species name from clip filename.
        
        Filename format: timestamp_species.mp4
        Example: 1766587074_bird_passeriformes_cardinalidae.mp4 -> "Cardinal"
        """
        import re
        
        # Remove extension
        name = filename.rsplit('.', 1)[0]
        
        # Split by underscore, species is after the timestamp
        parts = name.split('_', 1)
        if len(parts) < 2:
            return 'Unknown'
        
        species_part = parts[1]
        
        # Handle complex SpeciesNet format with UUIDs and semicolons
        # Remove UUIDs (8-4-4-4-12 hex pattern)
        species_part = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}[;]*', '', species_part)
        
        # Split by + for multiple species
        species_list = []
        for part in species_part.split('+'):
            # Split by semicolons and get meaningful parts
            segments = [s.strip() for s in part.split(';') if s.strip()]
            
            # Get the raw species identifier (join all meaningful parts)
            raw_species = '_'.join(seg for seg in segments if seg.lower() not in ('no cv result', 'unknown', 'blank', 'empty', ''))
            
            if raw_species:
                # Use the common name mapping
                common_name = get_common_name(raw_species)
                if common_name and common_name not in species_list:
                    species_list.append(common_name)
        
        if not species_list:
            return 'Unknown'
        
        # Deduplicate and join
        seen = set()
        unique = []
        for s in species_list:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        
        return ', '.join(unique[:3])  # Limit to 3 species for display

    async def handle_recordings(self, request):
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings)
        
        if not clips and not (self.storage_root / 'clips').exists():
             return web.Response(text="No recordings found (clips directory missing)", content_type='text/html')

        html = """
        <html>
            <head>
                <title>Recordings - Animal Tracker</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <meta name="apple-mobile-web-app-capable" content="yes">
                <meta name="mobile-web-app-capable" content="yes">
                <style>
                    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #1a1a1a; 
                        color: #eee; 
                        margin: 0; 
                        padding: 16px;
                        padding-bottom: 80px;
                    }
                    .nav { 
                        display: flex;
                        gap: 12px;
                        margin-bottom: 20px; 
                        padding-bottom: 16px; 
                        border-bottom: 1px solid #333;
                    }
                    .nav a { 
                        color: #fff;
                        background: #333;
                        text-decoration: none; 
                        font-size: 0.95em;
                        font-weight: 500;
                        padding: 10px 16px;
                        border-radius: 8px;
                        transition: background 0.2s;
                    }
                    .nav a:hover, .nav a.active { background: #4CAF50; }
                    h1 { font-size: 1.5em; margin: 0 0 16px 0; font-weight: 600; }
                    
                    /* Card-based layout for recordings */
                    .recordings-list { display: flex; flex-direction: column; gap: 12px; }
                    .recording-card {
                        background: #2a2a2a;
                        border-radius: 12px;
                        padding: 16px;
                        display: flex;
                        align-items: center;
                        gap: 14px;
                        transition: transform 0.15s, background 0.15s;
                        cursor: pointer;
                    }
                    .recording-card:active { transform: scale(0.98); background: #333; }
                    .recording-checkbox {
                        width: 22px;
                        height: 22px;
                        accent-color: #4CAF50;
                        flex-shrink: 0;
                    }
                    .recording-info { flex: 1; min-width: 0; }
                    .recording-species {
                        font-weight: 700;
                        font-size: 1.1em;
                        color: #4CAF50;
                        margin-bottom: 4px;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        flex-wrap: wrap;
                    }
                    .thumbnail-badge {
                        font-size: 0.75em;
                        font-weight: 500;
                        background: #333;
                        color: #888;
                        padding: 2px 8px;
                        border-radius: 12px;
                    }
                    .recording-camera {
                        font-weight: 500;
                        font-size: 0.9em;
                        color: #ccc;
                        margin-bottom: 4px;
                    }
                    .recording-time {
                        font-size: 0.85em;
                        color: #aaa;
                        margin-bottom: 2px;
                    }
                    .recording-meta {
                        font-size: 0.8em;
                        color: #777;
                    }
                    .recording-actions {
                        display: flex;
                        gap: 8px;
                        flex-shrink: 0;
                    }
                    .action-btn {
                        width: 44px;
                        height: 44px;
                        border: none;
                        border-radius: 10px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: transform 0.15s, opacity 0.15s;
                    }
                    .action-btn:active { transform: scale(0.9); }
                    .play-btn { background: #4CAF50; color: white; }
                    .download-btn { background: #2196F3; color: white; }
                    .delete-btn { background: #f44336; color: white; }
                    .action-btn svg { width: 20px; height: 20px; }
                    
                    /* Bulk actions bar */
                    .bulk-actions {
                        position: fixed;
                        bottom: 0;
                        left: 0;
                        right: 0;
                        background: #2a2a2a;
                        padding: 12px 16px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        border-top: 1px solid #444;
                        transform: translateY(100%);
                        transition: transform 0.3s ease;
                        z-index: 100;
                    }
                    .bulk-actions.visible { transform: translateY(0); }
                    .bulk-actions .select-all-wrap {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        color: #aaa;
                        font-size: 0.9em;
                    }
                    .bulk-delete-btn {
                        background: #f44336;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 0.95em;
                        cursor: pointer;
                        transition: opacity 0.2s;
                    }
                    .bulk-delete-btn:disabled { background: #555; opacity: 0.6; }
                    
                    /* Video Modal */
                    .video-modal {
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0, 0, 0, 0.95);
                        z-index: 1000;
                        flex-direction: column;
                    }
                    .video-modal.active { display: flex; }
                    .modal-header {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 12px 16px;
                        background: #1a1a1a;
                    }
                    .modal-title {
                        font-size: 0.95em;
                        font-weight: 500;
                        color: #fff;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        flex: 1;
                        margin-right: 12px;
                    }
                    .modal-close {
                        background: #333;
                        border: none;
                        color: #fff;
                        width: 40px;
                        height: 40px;
                        border-radius: 8px;
                        font-size: 1.5em;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;
                    }
                    .modal-close:active { background: #444; }
                    .video-container {
                        flex: 1;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 8px;
                        background: #000;
                    }
                    .video-container video {
                        max-width: 100%;
                        max-height: 100%;
                        border-radius: 8px;
                    }
                    .modal-actions {
                        display: flex;
                        gap: 12px;
                        padding: 16px;
                        background: #1a1a1a;
                    }
                    .modal-action-btn {
                        flex: 1;
                        padding: 14px;
                        border: none;
                        border-radius: 10px;
                        font-weight: 600;
                        font-size: 0.95em;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 8px;
                    }
                    .modal-action-btn.download { background: #2196F3; color: white; }
                    .modal-action-btn.delete { background: #f44336; color: white; }
                    .modal-action-btn:active { opacity: 0.8; }
                    .modal-action-btn svg { width: 18px; height: 18px; }
                    
                    /* Empty state */
                    .empty-state {
                        text-align: center;
                        padding: 60px 20px;
                        color: #666;
                    }
                    .empty-state svg { width: 64px; height: 64px; margin-bottom: 16px; opacity: 0.5; }
                    
                    /* Desktop adjustments */
                    @media (min-width: 768px) {
                        body { padding: 24px; max-width: 900px; margin: 0 auto; }
                        .recordings-list { gap: 8px; }
                        .recording-card { padding: 14px 18px; }
                    }
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/">Live View</a>
                    <a href="/recordings" class="active">Recordings</a>
                    <a href="/settings">Settings</a>
                </div>
                <h1>Recordings</h1>
                
                <div class="recordings-list">
        """
        
        for clip in clips:
            size_mb = clip['size'] / (1024 * 1024)
            escaped_path = clip['path'].replace("'", "\\'")
            url_encoded_path = clip['path'].replace('#', '%23')
            species_display = clip.get('species', 'Unknown')
            thumbnail_count = len(clip.get('thumbnails', []))
            thumbnail_badge = f'<span class="thumbnail-badge">📸 {thumbnail_count}</span>' if thumbnail_count > 0 else ''
            
            html += f"""
                    <div class="recording-card" onclick="window.location.href='/recording/{url_encoded_path}'">
                        <input type="checkbox" class="recording-checkbox" name="clip_select" value="{clip['path']}" onclick="event.stopPropagation(); updateBulkButton();">
                        <div class="recording-info">
                            <div class="recording-species">🐾 {species_display} {thumbnail_badge}</div>
                            <div class="recording-camera">{clip['camera']}</div>
                            <div class="recording-time">{clip['time'].strftime('%b %d, %Y at %I:%M %p')}</div>
                            <div class="recording-meta">{size_mb:.1f} MB</div>
                        </div>
                        <div class="recording-actions">
                            <button class="action-btn play-btn" onclick="event.stopPropagation(); playVideo('/clips/{clip['path']}', '{clip['filename']}', '{escaped_path}');" title="Quick Play">
                                <svg fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                            </button>
                            <button class="action-btn delete-btn" onclick="event.stopPropagation(); deleteClip('{escaped_path}');" title="Delete">
                                <svg fill="currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                            </button>
                        </div>
                    </div>
            """
        
        if not clips:
            html += """
                    <div class="empty-state">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>
                        <p>No recordings yet</p>
                    </div>
            """
            
        html += """
                </div>
                
                <!-- Bulk Actions Bar -->
                <div class="bulk-actions" id="bulkActions">
                    <div class="select-all-wrap">
                        <input type="checkbox" id="selectAll" class="recording-checkbox" onclick="toggleAll(this)">
                        <label for="selectAll">Select All</label>
                    </div>
                    <button id="bulkDeleteBtn" class="bulk-delete-btn" onclick="bulkDelete()" disabled>Delete Selected</button>
                </div>
                
                <!-- Video Modal -->
                <div class="video-modal" id="videoModal" onclick="closeModalOnBackdrop(event)">
                    <div class="modal-header">
                        <div class="modal-title" id="modalTitle"></div>
                        <button class="modal-close" onclick="closeModal()">&times;</button>
                    </div>
                    <div class="video-container" onclick="event.stopPropagation()">
                        <video id="modalVideo" controls playsinline></video>
                    </div>
                    <div class="modal-actions" onclick="event.stopPropagation()">
                        <a class="modal-action-btn download" id="downloadBtn">
                            <svg fill="currentColor" viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                            Download
                        </a>
                        <button class="modal-action-btn delete" onclick="deleteCurrentClip()">
                            <svg fill="currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                            Delete
                        </button>
                    </div>
                </div>
                
                <script>
                    let currentClipPath = null;
                    
                    function playVideo(url, title, clipPath) {
                        const modal = document.getElementById('videoModal');
                        const video = document.getElementById('modalVideo');
                        const titleEl = document.getElementById('modalTitle');
                        const downloadBtn = document.getElementById('downloadBtn');
                        
                        currentClipPath = clipPath;
                        video.src = url;
                        titleEl.textContent = title;
                        downloadBtn.href = url;
                        downloadBtn.download = title;
                        
                        modal.classList.add('active');
                        document.body.style.overflow = 'hidden';
                        video.play().catch(() => {}); // Auto-play, ignore if blocked
                    }
                    
                    function closeModal() {
                        const modal = document.getElementById('videoModal');
                        const video = document.getElementById('modalVideo');
                        
                        video.pause();
                        video.src = '';
                        modal.classList.remove('active');
                        document.body.style.overflow = '';
                        currentClipPath = null;
                    }
                    
                    function closeModalOnBackdrop(event) {
                        if (event.target.id === 'videoModal') {
                            closeModal();
                        }
                    }
                    
                    // Close modal on escape key
                    document.addEventListener('keydown', (e) => {
                        if (e.key === 'Escape') closeModal();
                    });
                    
                    async function deleteCurrentClip() {
                        if (!currentClipPath) return;
                        await deleteClip(currentClipPath);
                    }
                    
                    async function deleteClip(path) {
                        if (!confirm('Are you sure you want to delete this clip?')) return;
                        try {
                            const response = await fetch('/recordings?path=' + encodeURIComponent(path), { method: 'DELETE' });
                            const text = await response.text();
                            if (response.ok) {
                                closeModal();
                                location.reload();
                            } else {
                                alert('Error: ' + text);
                            }
                        } catch (e) {
                            alert('Error deleting clip: ' + e);
                        }
                    }

                    function toggleAll(source) {
                        const checkboxes = document.querySelectorAll('input[name="clip_select"]');
                        checkboxes.forEach(cb => cb.checked = source.checked);
                        updateBulkButton();
                    }

                    function updateBulkButton() {
                        const checked = document.querySelectorAll('input[name="clip_select"]:checked');
                        const btn = document.getElementById('bulkDeleteBtn');
                        const bulkActions = document.getElementById('bulkActions');
                        const count = checked.length;
                        
                        btn.disabled = count === 0;
                        btn.textContent = count > 0 ? `Delete (${count})` : 'Delete Selected';
                        
                        // Show/hide bulk actions bar
                        if (count > 0) {
                            bulkActions.classList.add('visible');
                        } else {
                            bulkActions.classList.remove('visible');
                        }
                    }

                    async function bulkDelete() {
                        const checked = document.querySelectorAll('input[name="clip_select"]:checked');
                        if (checked.length === 0) return;

                        if (!confirm(`Delete ${checked.length} clips?`)) return;

                        const paths = Array.from(checked).map(cb => cb.value);
                        
                        try {
                            const response = await fetch('/recordings/bulk_delete', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ paths: paths })
                            });
                            
                            const result = await response.json();
                            location.reload();
                        } catch (e) {
                            alert('Error: ' + e);
                        }
                    }
                </script>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_snapshot(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or worker.latest_frame is None:
            return web.Response(status=404, text="Camera not found or no frame available")
            
        # Offload image processing to thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        
        def process_image(img):
            height, width = img.shape[:2]
            if width > 640:
                scale = 640 / width
                new_height = int(height * scale)
                img = cv2.resize(img, (640, new_height), interpolation=cv2.INTER_AREA)

            # Encode frame to JPEG with lower quality (70%)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            return cv2.imencode('.jpg', img, encode_param)

        success, buffer = await loop.run_in_executor(None, process_image, worker.latest_frame)
        
        if not success:
            return web.Response(status=500, text="Failed to encode frame")
            
        return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

    async def handle_save_clip(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=404, text="Camera not found")
            
        filename = worker.save_manual_clip()
        if not filename:
            return web.Response(status=500, text="Failed to save clip (buffer empty?)")
            
        return web.Response(text=f"Clip saved: {filename}")

    def _delete_file(self, rel_path: str) -> tuple[bool, str]:
        # Security check: prevent path traversal
        if '..' in rel_path or rel_path.startswith('/'):
             return False, "Invalid path"

        file_path = self.storage_root / 'clips' / rel_path
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return True, f"Deleted {rel_path}"
            else:
                return False, "File not found"
        except Exception as e:
            return False, f"Error deleting file: {e}"

    async def handle_delete_recording(self, request):
        rel_path = request.query.get('path')
        if not rel_path:
            return web.Response(status=400, text="Missing path parameter")
        
        loop = asyncio.get_running_loop()
        success, message = await loop.run_in_executor(None, self._delete_file, rel_path)
        
        if success:
            return web.Response(text=message)
        elif message == "File not found":
            return web.Response(status=404, text=message)
        elif message == "Invalid path":
            return web.Response(status=403, text=message)
        else:
            return web.Response(status=500, text=message)

    async def handle_bulk_delete(self, request):
        try:
            data = await request.json()
            paths = data.get('paths', [])
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")

        if not paths:
            return web.Response(status=400, text="No paths provided")

        loop = asyncio.get_running_loop()
        results = []
        
        for rel_path in paths:
            success, message = await loop.run_in_executor(None, self._delete_file, rel_path)
            results.append({'path': rel_path, 'success': success, 'message': message})

        # Count successes
        deleted_count = sum(1 for r in results if r['success'])
        return web.json_response({
            'deleted_count': deleted_count,
            'total_requested': len(paths),
            'results': results
        })

    async def handle_reprocess(self, request):
        """Reprocess a clip to improve species classification."""
        try:
            data = await request.json()
            clip_path = data.get('path')
            sample_rate = data.get('sample_rate', 5)
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")
        
        if not clip_path:
            return web.Response(status=400, text="Missing 'path' parameter")
        
        # Security check
        if '..' in clip_path:
            return web.Response(status=403, text="Invalid path")
        
        full_path = self.storage_root / 'clips' / clip_path
        if not full_path.exists():
            return web.Response(status=404, text="Clip not found")
        
        # Get detector from first worker (they all share the same one)
        if not self.workers:
            return web.Response(status=500, text="No workers available")
        
        detector = next(iter(self.workers.values())).detector
        
        # Run reprocessing in thread pool
        loop = asyncio.get_running_loop()
        
        # Get post-processing settings from runtime config
        clip_cfg = self.runtime.general.clip if self.runtime else None
        conf_threshold = getattr(clip_cfg, 'post_analysis_confidence', 0.3) if clip_cfg else 0.3
        generic_conf = getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5) if clip_cfg else 0.5
        tracking_enabled = getattr(clip_cfg, 'tracking_enabled', True) if clip_cfg else True
        
        def do_reprocess():
            from .postprocess import ClipPostProcessor
            processor = ClipPostProcessor(
                detector=detector,
                storage_root=self.storage_root,
                sample_rate=sample_rate,
                confidence_threshold=conf_threshold,
                generic_confidence=generic_conf,
                tracking_enabled=tracking_enabled,
            )
            return processor.process_clip(
                full_path,
                update_filename=True,
                regenerate_thumbnails=True,
            )
        
        result = await loop.run_in_executor(None, do_reprocess)
        
        if result.success:
            # Log thumbnail paths for debugging
            LOGGER.info("Reprocess complete. Thumbnails saved: %s", 
                       [str(p) for p in result.thumbnails_saved])
            
            return web.json_response({
                'success': True,
                'original_species': result.original_species,
                'new_species': result.new_species,
                'confidence': result.confidence,
                'frames_analyzed': result.frames_analyzed,
                'total_frames': result.total_frames,
                'raw_detections': result.raw_detections,
                'filtered_detections': result.filtered_detections,
                'species_found': list(result.species_results.keys()),
                'thumbnails_saved': len(result.thumbnails_saved),
                'thumbnail_paths': [str(p) for p in result.thumbnails_saved],  # Add actual paths for debugging
                'renamed': result.new_path is not None,
                'new_path': str(result.new_path.relative_to(self.storage_root / 'clips')) if result.new_path else None,
            })
        else:
            return web.json_response({
                'success': False,
                'error': result.error,
            }, status=500)

    async def handle_get_processing_log(self, request):
        """Get processing log JSON for a recording."""
        rel_path = request.match_info['path']
        
        # Security check
        if '..' in rel_path:
            return web.Response(status=403, text="Invalid path")
        
        # Construct the log file path (replace .mp4 with .log.json)
        clip_path = self.storage_root / 'clips' / rel_path
        if not clip_path.exists():
            return web.Response(status=404, text="Clip not found")
        
        log_path = clip_path.with_suffix('.log.json')
        
        if not log_path.exists():
            return web.json_response({
                'exists': False,
                'message': 'No processing log available. Reanalyze the recording to generate one.'
            })
        
        try:
            import json
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            return web.json_response({
                'exists': True,
                'data': log_data
            })
        except Exception as e:
            LOGGER.warning("Failed to read processing log: %s", e)
            return web.json_response({
                'exists': False,
                'message': f'Error reading log: {str(e)}'
            }, status=500)

    async def handle_recording_detail(self, request):
        """Render a detailed recording page with key detection photos."""
        rel_path = request.match_info['path']
        
        # Security check
        if '..' in rel_path:
            return web.Response(status=403, text="Invalid path")
        
        loop = asyncio.get_running_loop()
        clip_info = await loop.run_in_executor(None, self._get_clip_detail, rel_path)
        
        if clip_info is None:
            return web.Response(status=404, text="Recording not found")
        
        # Build thumbnail gallery HTML
        thumbnails_html = ""
        processing_log_html = """
                    <div class="processing-log">
                        <button class="log-toggle" onclick="toggleLog()">
                            <span class="arrow">▶</span>
                            <span>📋 Processing Details</span>
                        </button>
                        <div class="log-content" id="logContent">
                            <div id="logData" class="log-no-data">Loading...</div>
                        </div>
                    </div>
        """
        
        if clip_info['thumbnails']:
            thumbnails_html = """
                <div class="detection-section">
                    <h2>🔍 Detection Key Frames</h2>
                    <p class="detection-hint">These are the frames used to identify each species.</p>
                    <div class="thumbnail-gallery">
            """
            for thumb in clip_info['thumbnails']:
                thumbnails_html += f"""
                        <div class="thumbnail-card">
                            <img src="{thumb['url']}" alt="{thumb['species']}" onclick="openImage('{thumb['url']}')">
                            <div class="thumbnail-label">{thumb['species']}</div>
                        </div>
                """
            thumbnails_html += """
                    </div>
            """ + processing_log_html + """
                </div>
            """
        else:
            thumbnails_html = """
                <div class="detection-section">
                    <h2>🔍 Detection Key Frames</h2>
                    <p class="no-thumbnails">No detection thumbnails available for this recording.</p>
            """ + processing_log_html + """
                </div>
            """
        
        html = f"""
        <html>
            <head>
                <title>{clip_info['species']} - Recording Detail</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <meta name="apple-mobile-web-app-capable" content="yes">
                <meta name="mobile-web-app-capable" content="yes">
                <style>
                    * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
                    body {{ 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #1a1a1a; 
                        color: #eee; 
                        margin: 0; 
                        padding: 16px;
                        padding-bottom: 100px;
                    }}
                    .nav {{ 
                        display: flex;
                        gap: 12px;
                        margin-bottom: 20px; 
                        padding-bottom: 16px; 
                        border-bottom: 1px solid #333;
                    }}
                    .nav a {{ 
                        color: #fff;
                        background: #333;
                        text-decoration: none; 
                        font-size: 0.95em;
                        font-weight: 500;
                        padding: 10px 16px;
                        border-radius: 8px;
                        transition: background 0.2s;
                    }}
                    .nav a:hover, .nav a.active {{ background: #4CAF50; }}
                    .back-btn {{
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        color: #4CAF50;
                        text-decoration: none;
                        margin-bottom: 16px;
                        font-size: 0.95em;
                    }}
                    .back-btn:hover {{ text-decoration: underline; }}
                    h1 {{ font-size: 1.5em; margin: 0 0 8px 0; font-weight: 600; color: #4CAF50; }}
                    h2 {{ font-size: 1.1em; margin: 20px 0 12px 0; font-weight: 600; color: #ccc; }}
                    
                    .recording-header {{
                        margin-bottom: 20px;
                    }}
                    .recording-meta {{
                        color: #aaa;
                        font-size: 0.9em;
                        margin-bottom: 4px;
                    }}
                    .recording-meta span {{
                        margin-right: 16px;
                    }}
                    
                    .video-section {{
                        background: #000;
                        border-radius: 12px;
                        overflow: hidden;
                        margin-bottom: 20px;
                    }}
                    .video-section video {{
                        width: 100%;
                        display: block;
                    }}
                    
                    .action-buttons {{
                        display: flex;
                        gap: 12px;
                        margin-bottom: 24px;
                    }}
                    .action-btn {{
                        flex: 1;
                        padding: 14px;
                        border: none;
                        border-radius: 10px;
                        font-weight: 600;
                        font-size: 0.95em;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 8px;
                        text-decoration: none;
                        color: white;
                    }}
                    .action-btn.download {{ background: #2196F3; }}
                    .action-btn.delete {{ background: #f44336; }}
                    .action-btn.reprocess {{ background: #9C27B0; }}
                    .action-btn:active {{ opacity: 0.8; }}
                    .action-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
                    .action-btn svg {{ width: 18px; height: 18px; }}
                    
                    .detection-section {{
                        background: #2a2a2a;
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 20px;
                    }}
                    .detection-section h2 {{
                        margin: 0 0 8px 0;
                        color: #fff;
                    }}
                    .detection-hint {{
                        color: #888;
                        font-size: 0.85em;
                        margin: 0 0 16px 0;
                    }}
                    .no-thumbnails {{
                        color: #666;
                        font-style: italic;
                        margin: 0;
                    }}
                    
                    .thumbnail-gallery {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 12px;
                    }}
                    .thumbnail-card {{
                        background: #1a1a1a;
                        border-radius: 8px;
                        overflow: hidden;
                    }}
                    .thumbnail-card img {{
                        width: 100%;
                        aspect-ratio: 16/9;
                        object-fit: cover;
                        cursor: pointer;
                        transition: transform 0.2s;
                    }}
                    .thumbnail-card img:hover {{
                        transform: scale(1.02);
                    }}
                    .thumbnail-label {{
                        padding: 10px 12px;
                        font-size: 0.9em;
                        font-weight: 600;
                        color: #4CAF50;
                        text-align: center;
                    }}
                    
                    /* Image lightbox */
                    .lightbox {{
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0, 0, 0, 0.95);
                        z-index: 1000;
                        justify-content: center;
                        align-items: center;
                        padding: 20px;
                    }}
                    .lightbox.active {{ display: flex; }}
                    .lightbox img {{
                        max-width: 100%;
                        max-height: 100%;
                        border-radius: 8px;
                    }}
                    .lightbox-close {{
                        position: absolute;
                        top: 16px;
                        right: 16px;
                        background: #333;
                        border: none;
                        color: #fff;
                        width: 44px;
                        height: 44px;
                        border-radius: 8px;
                        font-size: 1.5em;
                        cursor: pointer;
                    }}
                    
                    /* Processing log styles */
                    .processing-log {{
                        margin-top: 16px;
                    }}
                    .log-toggle {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        cursor: pointer;
                        color: #888;
                        font-size: 0.9em;
                        padding: 8px 12px;
                        background: #1a1a1a;
                        border-radius: 8px;
                        border: none;
                        width: 100%;
                        text-align: left;
                    }}
                    .log-toggle:hover {{ background: #222; color: #aaa; }}
                    .log-toggle .arrow {{ transition: transform 0.2s; }}
                    .log-toggle.expanded .arrow {{ transform: rotate(90deg); }}
                    .log-content {{
                        display: none;
                        margin-top: 12px;
                        font-size: 0.85em;
                    }}
                    .log-content.visible {{ display: block; }}
                    .log-summary {{
                        background: #1a1a1a;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 12px;
                    }}
                    .log-summary h4 {{ margin: 0 0 8px 0; color: #4CAF50; }}
                    .log-summary-row {{
                        display: flex;
                        justify-content: space-between;
                        padding: 4px 0;
                        border-bottom: 1px solid #333;
                    }}
                    .log-summary-row:last-child {{ border-bottom: none; }}
                    .log-events {{
                        background: #1a1a1a;
                        border-radius: 8px;
                        padding: 12px;
                        max-height: 400px;
                        overflow-y: auto;
                    }}
                    .log-events h4 {{ margin: 0 0 8px 0; color: #2196F3; }}
                    .log-event {{
                        padding: 8px;
                        margin: 4px 0;
                        border-radius: 4px;
                        font-family: 'SF Mono', Monaco, 'Consolas', monospace;
                        font-size: 0.9em;
                    }}
                    .log-event.detection {{ background: rgba(76, 175, 80, 0.1); border-left: 3px solid #4CAF50; }}
                    .log-event.accepted {{ background: rgba(76, 175, 80, 0.1); border-left: 3px solid #4CAF50; }}
                    .log-event.filtered {{ background: rgba(255, 152, 0, 0.1); border-left: 3px solid #FF9800; }}
                    .log-event.detector_filtered {{ background: rgba(244, 67, 54, 0.15); border-left: 3px solid #F44336; }}
                    .log-event.tracked {{ background: rgba(33, 150, 243, 0.1); border-left: 3px solid #2196F3; }}
                    .log-event.track_consolidated {{ background: rgba(156, 39, 176, 0.1); border-left: 3px solid #9C27B0; }}
                    .log-event.selected {{ background: rgba(156, 39, 176, 0.1); border-left: 3px solid #9C27B0; }}
                    .log-event .frame {{ color: #888; font-size: 0.85em; }}
                    .log-event .species {{ color: #fff; font-weight: 600; }}
                    .log-event .confidence {{ color: #4CAF50; }}
                    .log-event .reason {{ color: #aaa; font-style: italic; font-size: 0.9em; }}
                    .log-no-data {{ color: #666; font-style: italic; text-align: center; padding: 20px; }}
                    
                    @media (min-width: 768px) {{
                        body {{ padding: 24px; max-width: 900px; margin: 0 auto; }}
                        .thumbnail-gallery {{ grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); }}
                    }}
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/">Live View</a>
                    <a href="/recordings" class="active">Recordings</a>
                    <a href="/settings">Settings</a>
                </div>
                
                <a href="/recordings" class="back-btn">← Back to Recordings</a>
                
                <div class="recording-header">
                    <h1>🐾 {clip_info['species']}</h1>
                    <div class="recording-meta">
                        <span>📷 {clip_info['camera']}</span>
                        <span>📅 {clip_info['time'].strftime('%b %d, %Y at %I:%M %p')}</span>
                        <span>📦 {clip_info['size_mb']:.1f} MB</span>
                    </div>
                </div>
                
                <div class="video-section">
                    <video controls playsinline autoplay>
                        <source src="/clips/{clip_info['path']}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                </div>
                
                <div class="action-buttons">
                    <a class="action-btn download" href="/clips/{clip_info['path']}" download="{clip_info['filename']}">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                        Download
                    </a>
                    <button class="action-btn reprocess" id="reprocessBtn" onclick="reprocessRecording()">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M12 6v3l4-4-4-4v3c-4.42 0-8 3.58-8 8 0 1.57.46 3.03 1.24 4.26L6.7 14.8c-.45-.83-.7-1.79-.7-2.8 0-3.31 2.69-6 6-6zm6.76 1.74L17.3 9.2c.44.84.7 1.79.7 2.8 0 3.31-2.69 6-6 6v-3l-4 4 4 4v-3c4.42 0 8-3.58 8-8 0-1.57-.46-3.03-1.24-4.26z"/></svg>
                        <span id="reprocessText">Reanalyze</span>
                    </button>
                    <button class="action-btn delete" onclick="deleteRecording()">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                        Delete
                    </button>
                </div>
                
                {thumbnails_html}
                
                <!-- Image Lightbox -->
                <div class="lightbox" id="lightbox" onclick="closeLightbox()">
                    <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
                    <img id="lightbox-img" src="" alt="Detection frame">
                </div>
                
                <script>
                    function openImage(url) {{
                        document.getElementById('lightbox-img').src = url;
                        document.getElementById('lightbox').classList.add('active');
                        document.body.style.overflow = 'hidden';
                    }}
                    
                    function closeLightbox() {{
                        document.getElementById('lightbox').classList.remove('active');
                        document.body.style.overflow = '';
                    }}
                    
                    document.addEventListener('keydown', (e) => {{
                        if (e.key === 'Escape') closeLightbox();
                    }});
                    
                    async function deleteRecording() {{
                        if (!confirm('Are you sure you want to delete this recording?')) return;
                        
                        try {{
                            const response = await fetch('/recordings?path=' + encodeURIComponent('{clip_info['path']}'), {{
                                method: 'DELETE'
                            }});
                            
                            if (response.ok) {{
                                window.location.href = '/recordings';
                            }} else {{
                                const text = await response.text();
                                alert('Error: ' + text);
                            }}
                        }} catch (e) {{
                            alert('Error deleting recording: ' + e);
                        }}
                    }}
                    
                    async function reprocessRecording() {{
                        const btn = document.getElementById('reprocessBtn');
                        const textEl = document.getElementById('reprocessText');
                        
                        btn.disabled = true;
                        textEl.textContent = 'Analyzing...';
                        
                        try {{
                            const response = await fetch('/recordings/reprocess', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{
                                    path: '{clip_info['path']}',
                                    sample_rate: 3  // More thorough analysis
                                }})
                            }});
                            
                            const result = await response.json();
                            
                            if (result.success) {{
                                let msg = `Analysis complete!\\n\\n`;
                                msg += `Species: ${{result.new_species || 'Unknown'}} (${{(result.confidence * 100).toFixed(1)}}% confidence)\\n`;
                                msg += `Frames analyzed: ${{result.frames_analyzed}}/${{result.total_frames}}\\n`;
                                msg += `Raw detections: ${{result.raw_detections}} (filtered: ${{result.filtered_detections}})\\n`;
                                msg += `Valid species found: ${{result.species_found.join(', ') || 'None'}}\\n`;
                                msg += `Thumbnails saved: ${{result.thumbnails_saved}}\\n`;
                                if (result.thumbnail_paths && result.thumbnail_paths.length > 0) {{
                                    msg += `Thumbnail files: ${{result.thumbnail_paths.join(', ')}}`;
                                }}
                                
                                if (result.renamed) {{
                                    msg += `\\n\\nFile was renamed. Redirecting...`;
                                    alert(msg);
                                    window.location.href = '/recording/' + result.new_path;
                                }} else {{
                                    alert(msg);
                                    window.location.reload();
                                }}
                            }} else {{
                                alert('Reprocessing failed: ' + result.error);
                                btn.disabled = false;
                                textEl.textContent = 'Reanalyze';
                            }}
                        }} catch (e) {{
                            alert('Error reprocessing: ' + e);
                            btn.disabled = false;
                            textEl.textContent = 'Reanalyze';
                        }}
                    }}
                    
                    // Processing log functions
                    let logLoaded = false;
                    
                    function toggleLog() {{
                        const toggle = document.querySelector('.log-toggle');
                        const content = document.getElementById('logContent');
                        
                        toggle.classList.toggle('expanded');
                        content.classList.toggle('visible');
                        
                        if (!logLoaded && content.classList.contains('visible')) {{
                            loadProcessingLog();
                        }}
                    }}
                    
                    async function loadProcessingLog() {{
                        const logData = document.getElementById('logData');
                        
                        try {{
                            const response = await fetch('/recordings/log/{clip_info['path']}');
                            const result = await response.json();
                            
                            if (result.exists) {{
                                renderProcessingLog(result.data);
                            }} else {{
                                logData.innerHTML = '<div class="log-no-data">' + result.message + '</div>';
                            }}
                            logLoaded = true;
                        }} catch (e) {{
                            logData.innerHTML = '<div class="log-no-data">Error loading log: ' + e + '</div>';
                        }}
                    }}
                    
                    function renderProcessingLog(data) {{
                        const logData = document.getElementById('logData');
                        let html = '';
                        
                        // Tracking summary if available
                        if (data.tracking_summary) {{
                            const summary = data.tracking_summary;
                            html += '<div class="log-summary">';
                            html += '<h4>🎯 Tracking Summary</h4>';
                            if (summary.total_tracks !== undefined) {{
                                html += '<div class="log-summary-row"><span>Total Tracks</span><span>' + summary.total_tracks + '</span></div>';
                            }}
                            if (summary.species_by_track) {{
                                for (const [trackId, species] of Object.entries(summary.species_by_track)) {{
                                    html += '<div class="log-summary-row"><span>Track ' + trackId + '</span><span>' + species + '</span></div>';
                                }}
                            }}
                            if (summary.track_details) {{
                                for (const [trackId, details] of Object.entries(summary.track_details)) {{
                                    html += '<div class="log-summary-row">';
                                    html += '<span>Track ' + trackId + '</span>';
                                    html += '<span>' + details.best_species + ' (' + (details.confidence * 100).toFixed(1) + '%) - ' + details.frame_count + ' frames</span>';
                                    html += '</div>';
                                }}
                            }}
                            html += '</div>';
                        }}
                        
                        // Log entries
                        if (data.log_entries && data.log_entries.length > 0) {{
                            html += '<div class="log-events">';
                            html += '<h4>📝 Processing Events (' + data.log_entries.length + ')</h4>';
                            
                            for (const entry of data.log_entries) {{
                                const eventClass = entry.event || 'detection';
                                html += '<div class="log-event ' + eventClass + '">';
                                html += '<span class="frame">Frame ' + entry.frame_idx + '</span> ';
                                if (entry.track_id !== null && entry.track_id !== undefined) {{
                                    html += '<span class="track">[Track ' + entry.track_id + ']</span> ';
                                }}
                                html += '<span class="species">' + entry.species + '</span> ';
                                html += '<span class="confidence">(' + (entry.confidence * 100).toFixed(1) + '%)</span>';
                                if (entry.reason) {{
                                    html += '<br><span class="reason">' + entry.reason + '</span>';
                                }}
                                html += '</div>';
                            }}
                            html += '</div>';
                        }} else {{
                            html += '<div class="log-no-data">No processing events recorded.</div>';
                        }}
                        
                        logData.innerHTML = html;
                    }}
                </script>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    def _get_clip_detail(self, rel_path: str) -> dict | None:
        """Get detailed information about a specific clip."""
        clips_dir = self.storage_root / 'clips'
        clip_path = clips_dir / rel_path
        
        if not clip_path.exists() or not clip_path.is_file():
            return None
        
        stat = clip_path.stat()
        species = self._parse_species_from_filename(clip_path.name)
        thumbnails = self._get_thumbnails_for_clip(clip_path)
        
        # Determine camera from path
        parts = rel_path.split('/')
        camera = parts[0] if len(parts) > 1 else 'unknown'
        
        return {
            'path': rel_path,
            'filename': clip_path.name,
            'camera': camera,
            'species': species,
            'time': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ),
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'thumbnails': thumbnails
        }

    async def handle_settings_page(self, request):
        """Render the settings page HTML."""
        html = """
        <html>
            <head>
                <title>Settings - Animal Tracker</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <meta name="apple-mobile-web-app-capable" content="yes">
                <meta name="mobile-web-app-capable" content="yes">
                <style>
                    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #1a1a1a; 
                        color: #eee; 
                        margin: 0; 
                        padding: 16px;
                        padding-bottom: 100px;
                    }
                    .nav { 
                        display: flex;
                        gap: 12px;
                        margin-bottom: 20px; 
                        padding-bottom: 16px; 
                        border-bottom: 1px solid #333;
                    }
                    .nav a { 
                        color: #fff;
                        background: #333;
                        text-decoration: none; 
                        font-size: 0.95em;
                        font-weight: 500;
                        padding: 10px 16px;
                        border-radius: 8px;
                        transition: background 0.2s;
                    }
                    .nav a:hover, .nav a.active { background: #4CAF50; }
                    h1 { font-size: 1.5em; margin: 0 0 16px 0; font-weight: 600; }
                    h2 { font-size: 1.2em; margin: 24px 0 12px 0; font-weight: 600; color: #4CAF50; }
                    
                    .camera-tabs {
                        display: flex;
                        gap: 8px;
                        margin-bottom: 16px;
                        overflow-x: auto;
                        padding-bottom: 8px;
                    }
                    .camera-tab {
                        background: #333;
                        color: #fff;
                        border: none;
                        padding: 10px 16px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 0.9em;
                        font-weight: 500;
                        white-space: nowrap;
                        transition: background 0.2s;
                    }
                    .camera-tab:hover { background: #444; }
                    .camera-tab.active { background: #4CAF50; }
                    
                    .settings-card {
                        background: #2a2a2a;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 16px;
                    }
                    .setting-row {
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                        margin-bottom: 20px;
                    }
                    .setting-row:last-child { margin-bottom: 0; }
                    .setting-label {
                        font-size: 0.9em;
                        color: #aaa;
                        font-weight: 500;
                    }
                    .setting-description {
                        font-size: 0.8em;
                        color: #666;
                        margin-top: 4px;
                    }
                    input[type="number"], input[type="text"], textarea {
                        background: #1a1a1a;
                        border: 1px solid #444;
                        border-radius: 8px;
                        padding: 12px;
                        color: #fff;
                        font-size: 1em;
                        width: 100%;
                    }
                    input[type="number"]:focus, input[type="text"]:focus, textarea:focus {
                        outline: none;
                        border-color: #4CAF50;
                    }
                    textarea {
                        min-height: 80px;
                        resize: vertical;
                        font-family: monospace;
                    }
                    .slider-container {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    }
                    input[type="range"] {
                        flex: 1;
                        height: 6px;
                        -webkit-appearance: none;
                        background: #444;
                        border-radius: 3px;
                        outline: none;
                    }
                    input[type="range"]::-webkit-slider-thumb {
                        -webkit-appearance: none;
                        width: 20px;
                        height: 20px;
                        background: #4CAF50;
                        border-radius: 50%;
                        cursor: pointer;
                    }
                    .slider-value {
                        min-width: 50px;
                        text-align: right;
                        font-weight: 600;
                        color: #4CAF50;
                    }
                    
                    /* Toggle switch */
                    .toggle-switch {
                        position: relative;
                        display: inline-block;
                        width: 50px;
                        height: 28px;
                    }
                    .toggle-switch input {
                        opacity: 0;
                        width: 0;
                        height: 0;
                    }
                    .toggle-slider {
                        position: absolute;
                        cursor: pointer;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background-color: #444;
                        transition: 0.3s;
                        border-radius: 28px;
                    }
                    .toggle-slider:before {
                        position: absolute;
                        content: "";
                        height: 20px;
                        width: 20px;
                        left: 4px;
                        bottom: 4px;
                        background-color: white;
                        transition: 0.3s;
                        border-radius: 50%;
                    }
                    .toggle-switch input:checked + .toggle-slider {
                        background-color: #4CAF50;
                    }
                    .toggle-switch input:checked + .toggle-slider:before {
                        transform: translateX(22px);
                    }
                    
                    /* Select dropdown */
                    select {
                        background: #1a1a1a;
                        border: 1px solid #444;
                        border-radius: 8px;
                        padding: 12px;
                        color: #fff;
                        font-size: 1em;
                        width: 100%;
                        cursor: pointer;
                    }
                    select:focus {
                        outline: none;
                        border-color: #4CAF50;
                    }
                    select option {
                        background: #1a1a1a;
                        color: #fff;
                    }
                    
                    .save-bar {
                        position: fixed;
                        bottom: 0;
                        left: 0;
                        right: 0;
                        background: #2a2a2a;
                        padding: 16px;
                        display: flex;
                        gap: 12px;
                        border-top: 1px solid #444;
                        z-index: 100;
                    }
                    .save-btn {
                        flex: 1;
                        background: #4CAF50;
                        color: white;
                        border: none;
                        padding: 14px;
                        border-radius: 10px;
                        font-weight: 600;
                        font-size: 1em;
                        cursor: pointer;
                        transition: opacity 0.2s;
                    }
                    .save-btn:disabled { background: #555; opacity: 0.6; }
                    .save-btn:active { opacity: 0.8; }
                    .reset-btn {
                        background: #555;
                        color: white;
                        border: none;
                        padding: 14px 20px;
                        border-radius: 10px;
                        font-weight: 500;
                        font-size: 1em;
                        cursor: pointer;
                    }
                    
                    .toast {
                        position: fixed;
                        bottom: 80px;
                        left: 50%;
                        transform: translateX(-50%);
                        background: #333;
                        color: #fff;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-size: 0.9em;
                        opacity: 0;
                        transition: opacity 0.3s;
                        z-index: 200;
                    }
                    .toast.success { background: #4CAF50; }
                    .toast.error { background: #f44336; }
                    .toast.visible { opacity: 1; }
                    
                    .loading {
                        text-align: center;
                        padding: 40px;
                        color: #666;
                    }
                    
                    /* Species selector styles */
                    .species-selector {
                        background: #1a1a1a;
                        border: 1px solid #444;
                        border-radius: 8px;
                        max-height: 250px;
                        overflow-y: auto;
                    }
                    .species-search {
                        position: sticky;
                        top: 0;
                        background: #1a1a1a;
                        padding: 10px;
                        border-bottom: 1px solid #333;
                    }
                    .species-search input {
                        width: 100%;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 0.9em;
                    }
                    .species-grid {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 2px;
                        padding: 8px;
                    }
                    .species-item {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 10px;
                        border-radius: 6px;
                        cursor: pointer;
                        transition: background 0.15s;
                        user-select: none;
                    }
                    .species-item:hover { background: #333; }
                    .species-item.selected { background: #2d4a2d; }
                    .species-item input[type="checkbox"] {
                        width: 18px;
                        height: 18px;
                        accent-color: #4CAF50;
                        cursor: pointer;
                    }
                    .species-item label {
                        cursor: pointer;
                        font-size: 0.9em;
                        flex: 1;
                    }
                    .species-category {
                        grid-column: 1 / -1;
                        padding: 8px 10px 4px;
                        font-size: 0.75em;
                        color: #888;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        border-top: 1px solid #333;
                        margin-top: 4px;
                    }
                    .species-category:first-child {
                        border-top: none;
                        margin-top: 0;
                    }
                    .species-category.recent-category {
                        background: #3d2d1a;
                        color: #ffa500;
                        font-weight: 600;
                        padding: 10px;
                        border-radius: 6px;
                        margin: 0;
                        border-top: none;
                    }
                    .species-item.recent-item {
                        background: #2d2520;
                        border: 1px solid #4a3520;
                    }
                    .species-item.recent-item:hover {
                        background: #3d3020;
                    }
                    .species-item.recent-item.selected {
                        background: #2d4a2d;
                        border-color: #4CAF50;
                    }
                    .detection-count {
                        background: #ff6b35;
                        color: #fff;
                        font-size: 0.75em;
                        font-weight: 600;
                        padding: 2px 8px;
                        border-radius: 10px;
                        min-width: 24px;
                        text-align: center;
                    }
                    .selected-count {
                        font-size: 0.8em;
                        color: #4CAF50;
                        margin-top: 6px;
                    }
                    .clear-btn {
                        background: #444;
                        color: #fff;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 6px;
                        font-size: 0.8em;
                        cursor: pointer;
                        margin-left: 8px;
                    }
                    .clear-btn:hover { background: #555; }
                    
                    @media (min-width: 768px) {
                        body { padding: 24px; max-width: 700px; margin: 0 auto; padding-bottom: 100px; }
                        .species-grid { grid-template-columns: repeat(3, 1fr); }
                    }
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/">Live View</a>
                    <a href="/recordings">Recordings</a>
                    <a href="/settings" class="active">Settings</a>
                </div>
                <h1>Settings</h1>
                
                <div class="camera-tabs" id="cameraTabs">
                    <!-- Populated by JS -->
                </div>
                
                <div id="settingsContent">
                    <div class="loading">Loading settings...</div>
                </div>
                
                <div class="save-bar">
                    <button class="reset-btn" onclick="resetSettings()">Reset</button>
                    <button class="save-btn" id="saveBtn" onclick="saveSettings()">Save Changes</button>
                </div>
                
                <div class="toast" id="toast"></div>
                
                <script>
                    let settings = {};
                    let originalSettings = {};
                    let currentCamera = null;
                    
                    // Predefined species list organized by category
                    const SPECIES_LIST = {
                        'Common Wildlife': [
                            'deer', 'coyote', 'fox', 'raccoon', 'opossum', 'skunk', 'rabbit', 
                            'squirrel', 'chipmunk', 'groundhog', 'armadillo', 'porcupine', 'beaver'
                        ],
                        'Large Mammals': [
                            'bear', 'moose', 'elk', 'mountain lion', 'cougar', 'bobcat', 'lynx',
                            'wolf', 'wild boar', 'javelina', 'bison', 'antelope'
                        ],
                        'Birds': [
                            'bird', 'turkey', 'hawk', 'owl', 'eagle', 'vulture', 'crow', 
                            'heron', 'duck', 'goose', 'pheasant', 'quail', 'dove', 'woodpecker'
                        ],
                        'Farm Animals': [
                            'horse', 'cow', 'sheep', 'goat', 'pig', 'chicken', 'donkey', 'llama'
                        ],
                        'Pets & Domestic': [
                            'dog', 'cat', 'person'
                        ],
                        'Other': [
                            'snake', 'turtle', 'frog', 'lizard', 'alligator', 'fish',
                            'bat', 'mouse', 'rat', 'mole', 'weasel', 'otter', 'mink', 'badger'
                        ]
                    };
                    
                    async function loadSettings() {
                        try {
                            const response = await fetch('/api/settings');
                            settings = await response.json();
                            originalSettings = JSON.parse(JSON.stringify(settings));
                            
                            // Build camera tabs
                            const tabsContainer = document.getElementById('cameraTabs');
                            tabsContainer.innerHTML = '';
                            
                            // Add Global tab first
                            const globalBtn = document.createElement('button');
                            globalBtn.className = 'camera-tab';
                            globalBtn.textContent = '⚙️ Global';
                            globalBtn.onclick = () => selectGlobal();
                            tabsContainer.appendChild(globalBtn);
                            
                            // Add camera tabs
                            Object.keys(settings.cameras).forEach((camId, index) => {
                                const cam = settings.cameras[camId];
                                const btn = document.createElement('button');
                                btn.className = 'camera-tab';
                                btn.textContent = cam.name;
                                btn.onclick = () => selectCamera(camId);
                                tabsContainer.appendChild(btn);
                            });
                            
                            // Select Global tab by default
                            selectGlobal();
                            
                        } catch (e) {
                            document.getElementById('settingsContent').innerHTML = 
                                '<div class="loading">Error loading settings: ' + e.message + '</div>';
                        }
                    }
                    
                    function selectGlobal() {
                        currentCamera = null;
                        
                        // Update tab styling
                        document.querySelectorAll('.camera-tab').forEach((tab, idx) => {
                            tab.classList.toggle('active', idx === 0);
                        });
                        
                        renderGlobalSettings();
                    }
                    
                    function selectCamera(camId) {
                        currentCamera = camId;
                        
                        // Update tab styling - find by camera name
                        const camName = settings.cameras[camId].name;
                        document.querySelectorAll('.camera-tab').forEach(tab => {
                            tab.classList.remove('active');
                            if (tab.textContent === camName) {
                                tab.classList.add('active');
                            }
                        });
                        
                        renderSettings(camId);
                    }
                    
                    function renderGlobalSettings() {
                        const g = settings.global;
                        const html = `
                            <div class="settings-card">
                                <h2>🎯 Detector Settings</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Backend</label>
                                    <input type="text" value="${g.detector.backend}" disabled 
                                           style="opacity: 0.6; cursor: not-allowed;">
                                    <div class="setting-description">Detection backend (requires restart to change)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Location</label>
                                    <input type="text" value="${g.detector.country || ''} ${g.detector.admin1_region || ''}" disabled 
                                           style="opacity: 0.6; cursor: not-allowed;">
                                    <div class="setting-description">Geographic priors (edit cameras.yml to change)</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>🎬 Clip Settings</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Pre-Event Buffer (seconds)</label>
                                    <input type="number" min="1" max="60" step="1"
                                           value="${g.clip.pre_seconds}"
                                           onchange="updateGlobalValue('clip', 'pre_seconds', parseFloat(this.value))">
                                    <div class="setting-description">Seconds of video to keep before detection trigger</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Post-Event Buffer (seconds)</label>
                                    <input type="number" min="1" max="60" step="1"
                                           value="${g.clip.post_seconds}"
                                           onchange="updateGlobalValue('clip', 'post_seconds', parseFloat(this.value))">
                                    <div class="setting-description">Seconds of video to record after detection ends</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Post-Analysis Enabled</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.post_analysis ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'post_analysis', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Re-analyze clips after recording for better species ID</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Post-Analysis Species Confidence</label>
                                    <div class="slider-container">
                                        <input type="range" min="0" max="100" step="5" 
                                               value="${Math.round((g.clip.post_analysis_confidence || 0.3) * 100)}"
                                               oninput="updateGlobalSlider(this, 'clip', 'post_analysis_confidence')">
                                        <span class="slider-value" id="post_analysis_confidence-value">${Math.round((g.clip.post_analysis_confidence || 0.3) * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Species threshold for post-analysis (lower catches more)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Post-Analysis Generic Confidence</label>
                                    <div class="slider-container">
                                        <input type="range" min="0" max="100" step="5" 
                                               value="${Math.round((g.clip.post_analysis_generic_confidence || 0.5) * 100)}"
                                               oninput="updateGlobalSlider(this, 'clip', 'post_analysis_generic_confidence')">
                                        <span class="slider-value" id="post_analysis_generic_confidence-value">${Math.round((g.clip.post_analysis_generic_confidence || 0.5) * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Generic category threshold for post-analysis (animal, bird, etc.)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Object Tracking</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.tracking_enabled ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'tracking_enabled', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Track same animal across frames for consistent species ID (uses ByteTrack)</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>💾 Storage & Retention</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Minimum Retention (days)</label>
                                    <input type="number" min="1" max="365" step="1"
                                           value="${g.retention.min_days}"
                                           onchange="updateGlobalValue('retention', 'min_days', parseInt(this.value))">
                                    <div class="setting-description">Keep clips for at least this many days</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Maximum Retention (days)</label>
                                    <input type="number" min="1" max="365" step="1"
                                           value="${g.retention.max_days}"
                                           onchange="updateGlobalValue('retention', 'max_days', parseInt(this.value))">
                                    <div class="setting-description">Delete clips older than this (unless space is needed sooner)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Max Disk Usage (%)</label>
                                    <div class="slider-container">
                                        <input type="range" min="50" max="95" step="5" 
                                               value="${g.retention.max_utilization_pct}"
                                               oninput="updateGlobalSliderDirect(this, 'retention', 'max_utilization_pct')">
                                        <span class="slider-value" id="max_utilization_pct-value">${g.retention.max_utilization_pct}%</span>
                                    </div>
                                    <div class="setting-description">Start deleting old clips when disk usage exceeds this</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>🐦 eBird Integration</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Enable eBird Filtering</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.ebird?.enabled ? 'checked' : ''}
                                               onchange="updateGlobalValue('ebird', 'enabled', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Filter/flag bird detections based on recent eBird sightings. Requires EBIRD_API_KEY env variable.</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">eBird Region</label>
                                    <input type="text" value="${g.ebird?.region || 'US-MN'}"
                                           placeholder="US-MN"
                                           onchange="updateGlobalValue('ebird', 'region', this.value)"
                                           style="width: 150px;">
                                    <div class="setting-description">Region code (e.g., US-MN, US-CA, CA-ON). See <a href="https://ebird.org/region/world" target="_blank">eBird regions</a></div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Days Back</label>
                                    <input type="number" min="1" max="30" step="1"
                                           value="${g.ebird?.days_back || 14}"
                                           onchange="updateGlobalValue('ebird', 'days_back', parseInt(this.value))">
                                    <div class="setting-description">How many days of recent sightings to check (1-30)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Filter Mode</label>
                                    <select onchange="updateGlobalValue('ebird', 'filter_mode', this.value)">
                                        <option value="flag" ${(g.ebird?.filter_mode || 'flag') === 'flag' ? 'selected' : ''}>Flag Only</option>
                                        <option value="filter" ${(g.ebird?.filter_mode) === 'filter' ? 'selected' : ''}>Filter Out</option>
                                        <option value="boost" ${(g.ebird?.filter_mode) === 'boost' ? 'selected' : ''}>Boost Priority</option>
                                    </select>
                                    <div class="setting-description">Flag: log unusual species. Filter: reject species not seen recently. Boost: prioritize common species.</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Cache Hours</label>
                                    <input type="number" min="1" max="168" step="1"
                                           value="${g.ebird?.cache_hours || 24}"
                                           onchange="updateGlobalValue('ebird', 'cache_hours', parseInt(this.value))">
                                    <div class="setting-description">How long to cache eBird data before refreshing (hours)</div>
                                </div>
                                
                                <div class="setting-description" style="margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                                    <small>Species data provided by <a href="https://ebird.org" target="_blank">eBird.org</a>. Free API key at <a href="https://ebird.org/api/keygen" target="_blank">ebird.org/api/keygen</a></small>
                                </div>
                            </div>
                        `;
                        document.getElementById('settingsContent').innerHTML = html;
                    }
                    
                    function updateGlobalSlider(slider, section, field) {
                        const value = parseInt(slider.value) / 100;
                        document.getElementById(field + '-value').textContent = Math.round(value * 100) + '%';
                        if (!settings.global[section]) settings.global[section] = {};
                        settings.global[section][field] = value;
                    }
                    
                    function updateGlobalSliderDirect(slider, section, field) {
                        const value = parseInt(slider.value);
                        document.getElementById(field + '-value').textContent = value + '%';
                        if (!settings.global[section]) settings.global[section] = {};
                        settings.global[section][field] = value;
                    }
                    
                    function updateGlobalValue(section, field, value) {
                        if (!settings.global[section]) settings.global[section] = {};
                        settings.global[section][field] = value;
                    }
                    
                    function renderSettings(camId) {
                        const cam = settings.cameras[camId];
                        const recentDetections = cam.recent_detections || {};
                        const rtsp = cam.rtsp || {};
                        const notify = cam.notification || {};
                        
                        const html = `
                            <div class="settings-card">
                                <h2>🔍 Detection Settings</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Detection Enabled</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${cam.detect_enabled !== false ? 'checked' : ''}
                                               onchange="updateCameraValue('detect_enabled', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Enable/disable detection for this camera</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Species Confidence</label>
                                    <div class="slider-container">
                                        <input type="range" min="0" max="100" step="5" 
                                               value="${Math.round(cam.thresholds.confidence * 100)}"
                                               oninput="updateSlider(this, 'confidence')"
                                               data-field="confidence">
                                        <span class="slider-value" id="confidence-value">${Math.round(cam.thresholds.confidence * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Threshold for specific species (cardinal, deer, etc.)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Generic Category Confidence</label>
                                    <div class="slider-container">
                                        <input type="range" min="0" max="100" step="5" 
                                               value="${Math.round((cam.thresholds.generic_confidence || 0.9) * 100)}"
                                               oninput="updateSlider(this, 'generic_confidence')"
                                               data-field="generic_confidence">
                                        <span class="slider-value" id="generic_confidence-value">${Math.round((cam.thresholds.generic_confidence || 0.9) * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Higher threshold for vague labels (animal, bird, mammal)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Minimum Frames</label>
                                    <input type="number" min="1" max="30" step="1"
                                           value="${cam.thresholds.min_frames}"
                                           onchange="updateValue('min_frames', this.value)"
                                           data-field="min_frames">
                                    <div class="setting-description">Number of consecutive frames with detection required</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Minimum Duration (seconds)</label>
                                    <input type="number" min="0" max="30" step="0.5"
                                           value="${cam.thresholds.min_duration}"
                                           onchange="updateValue('min_duration', parseFloat(this.value))"
                                           data-field="min_duration">
                                    <div class="setting-description">Minimum event duration before saving clip and notifying</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>📹 Stream Settings</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Frame Skip</label>
                                    <input type="number" min="0" max="30" step="1"
                                           value="${rtsp.frame_skip || 0}"
                                           onchange="updateRtspValue('frame_skip', parseInt(this.value))">
                                    <div class="setting-description">Skip N frames between detections (reduces CPU, 0=analyze every frame)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Hardware Acceleration</label>
                                    <select onchange="updateRtspValue('hwaccel', this.value || null)">
                                        <option value="" ${!rtsp.hwaccel ? 'selected' : ''}>None (CPU)</option>
                                        <option value="nvdec" ${rtsp.hwaccel === 'nvdec' ? 'selected' : ''}>NVDEC (NVIDIA)</option>
                                        <option value="cuda" ${rtsp.hwaccel === 'cuda' ? 'selected' : ''}>CUDA (NVIDIA)</option>
                                        <option value="vaapi" ${rtsp.hwaccel === 'vaapi' ? 'selected' : ''}>VAAPI (Intel/AMD)</option>
                                        <option value="videotoolbox" ${rtsp.hwaccel === 'videotoolbox' ? 'selected' : ''}>VideoToolbox (macOS)</option>
                                    </select>
                                    <div class="setting-description">Hardware decoder for stream (platform-specific)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Latency (ms)</label>
                                    <input type="number" min="0" max="5000" step="100"
                                           value="${rtsp.latency_ms || 200}"
                                           onchange="updateRtspValue('latency_ms', parseInt(this.value))">
                                    <div class="setting-description">Stream latency buffer in milliseconds</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Transport</label>
                                    <select onchange="updateRtspValue('transport', this.value)">
                                        <option value="tcp" ${(rtsp.transport || 'tcp') === 'tcp' ? 'selected' : ''}>TCP (reliable)</option>
                                        <option value="udp" ${rtsp.transport === 'udp' ? 'selected' : ''}>UDP (lower latency)</option>
                                    </select>
                                    <div class="setting-description">RTSP transport protocol</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>🔔 Notification Settings</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Priority</label>
                                    <select onchange="updateNotifyValue('priority', parseInt(this.value))">
                                        <option value="-2" ${notify.priority === -2 ? 'selected' : ''}>Lowest</option>
                                        <option value="-1" ${notify.priority === -1 ? 'selected' : ''}>Low</option>
                                        <option value="0" ${!notify.priority || notify.priority === 0 ? 'selected' : ''}>Normal</option>
                                        <option value="1" ${notify.priority === 1 ? 'selected' : ''}>High</option>
                                    </select>
                                    <div class="setting-description">Pushover notification priority</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Sound</label>
                                    <select onchange="updateNotifyValue('sound', this.value || null)">
                                        <option value="" ${!notify.sound ? 'selected' : ''}>Default</option>
                                        <option value="pushover" ${notify.sound === 'pushover' ? 'selected' : ''}>Pushover</option>
                                        <option value="bike" ${notify.sound === 'bike' ? 'selected' : ''}>Bike</option>
                                        <option value="bugle" ${notify.sound === 'bugle' ? 'selected' : ''}>Bugle</option>
                                        <option value="cashregister" ${notify.sound === 'cashregister' ? 'selected' : ''}>Cash Register</option>
                                        <option value="classical" ${notify.sound === 'classical' ? 'selected' : ''}>Classical</option>
                                        <option value="cosmic" ${notify.sound === 'cosmic' ? 'selected' : ''}>Cosmic</option>
                                        <option value="falling" ${notify.sound === 'falling' ? 'selected' : ''}>Falling</option>
                                        <option value="gamelan" ${notify.sound === 'gamelan' ? 'selected' : ''}>Gamelan</option>
                                        <option value="incoming" ${notify.sound === 'incoming' ? 'selected' : ''}>Incoming</option>
                                        <option value="intermission" ${notify.sound === 'intermission' ? 'selected' : ''}>Intermission</option>
                                        <option value="magic" ${notify.sound === 'magic' ? 'selected' : ''}>Magic</option>
                                        <option value="mechanical" ${notify.sound === 'mechanical' ? 'selected' : ''}>Mechanical</option>
                                        <option value="pianobar" ${notify.sound === 'pianobar' ? 'selected' : ''}>Piano Bar</option>
                                        <option value="siren" ${notify.sound === 'siren' ? 'selected' : ''}>Siren</option>
                                        <option value="spacealarm" ${notify.sound === 'spacealarm' ? 'selected' : ''}>Space Alarm</option>
                                        <option value="tugboat" ${notify.sound === 'tugboat' ? 'selected' : ''}>Tugboat</option>
                                        <option value="none" ${notify.sound === 'none' ? 'selected' : ''}>None (silent)</option>
                                    </select>
                                    <div class="setting-description">Notification sound</div>
                                </div>
                            </div>
                            
                            <div class="settings-card">
                                <h2>✅ Include Species</h2>
                                <div class="setting-description" style="margin-bottom: 12px;">
                                    Select species to detect. Leave all unchecked to detect everything.
                                </div>
                                <div class="selected-count" id="include-count">${getSelectedCountText(cam.include_species, 'include')}</div>
                                ${renderSpeciesSelector('include_species', cam.include_species || [], null)}
                            </div>
                            
                            <div class="settings-card">
                                <h2>🚫 Exclude Species</h2>
                                <div class="setting-description" style="margin-bottom: 12px;">
                                    Select species to always ignore, even if detected.
                                </div>
                                <div class="selected-count" id="exclude-count">${getSelectedCountText(cam.exclude_species, 'exclude')}</div>
                                ${renderSpeciesSelector('exclude_species', cam.exclude_species || [], recentDetections)}
                            </div>
                        `;
                        document.getElementById('settingsContent').innerHTML = html;
                    }
                    
                    function updateCameraValue(field, value) {
                        settings.cameras[currentCamera][field] = value;
                    }
                    
                    function updateRtspValue(field, value) {
                        if (!settings.cameras[currentCamera].rtsp) {
                            settings.cameras[currentCamera].rtsp = {};
                        }
                        settings.cameras[currentCamera].rtsp[field] = value;
                    }
                    
                    function updateNotifyValue(field, value) {
                        if (!settings.cameras[currentCamera].notification) {
                            settings.cameras[currentCamera].notification = {};
                        }
                        settings.cameras[currentCamera].notification[field] = value;
                    }
                    
                    function getSelectedCountText(selected, type) {
                        const count = (selected || []).length;
                        if (type === 'include') {
                            return count === 0 ? '✓ Detecting all species' : `${count} species selected`;
                        } else {
                            return count === 0 ? 'No species excluded' : `${count} species excluded`;
                        }
                    }
                    
                    function renderSpeciesSelector(field, selectedSpecies, recentDetections) {
                        const selectedSet = new Set((selectedSpecies || []).map(s => s.toLowerCase()));
                        let html = `<div class="species-selector">
                            <div class="species-search">
                                <input type="text" placeholder="Search species..." 
                                       oninput="filterSpecies(this.value, '${field}')">
                                <button class="clear-btn" onclick="clearAllSpecies('${field}')">Clear All</button>
                            </div>
                            <div class="species-grid" id="${field}-grid">`;
                        
                        // Show recent detections at the top for exclude_species
                        if (recentDetections && Object.keys(recentDetections).length > 0) {
                            html += `<div class="species-category recent-category">📊 Recent Alerts (tap to exclude)</div>`;
                            
                            // Sort by count descending
                            const sorted = Object.entries(recentDetections).sort((a, b) => b[1] - a[1]);
                            
                            for (const [sp, count] of sorted) {
                                const isSelected = selectedSet.has(sp.toLowerCase());
                                const itemClass = isSelected ? 'species-item selected recent-item' : 'species-item recent-item';
                                const displayName = sp.charAt(0).toUpperCase() + sp.slice(1);
                                html += `
                                    <div class="${itemClass}" onclick="toggleSpecies('${field}', '${sp}', this)" data-species="${sp.toLowerCase()}">
                                        <input type="checkbox" ${isSelected ? 'checked' : ''} 
                                               onclick="event.stopPropagation(); toggleSpecies('${field}', '${sp}', this.parentElement)">
                                        <label>${displayName}</label>
                                        <span class="detection-count">${count}</span>
                                    </div>`;
                            }
                        }
                        
                        // Then show all species by category
                        for (const [category, species] of Object.entries(SPECIES_LIST)) {
                            html += `<div class="species-category">${category}</div>`;
                            for (const sp of species) {
                                const isSelected = selectedSet.has(sp.toLowerCase());
                                const itemClass = isSelected ? 'species-item selected' : 'species-item';
                                html += `
                                    <div class="${itemClass}" onclick="toggleSpecies('${field}', '${sp}', this)" data-species="${sp.toLowerCase()}">
                                        <input type="checkbox" ${isSelected ? 'checked' : ''} 
                                               onclick="event.stopPropagation(); toggleSpecies('${field}', '${sp}', this.parentElement)">
                                        <label>${sp.charAt(0).toUpperCase() + sp.slice(1)}</label>
                                    </div>`;
                            }
                        }
                        
                        html += '</div></div>';
                        return html;
                    }
                    
                    function toggleSpecies(field, species, element) {
                        const speciesList = settings.cameras[currentCamera][field] || [];
                        const speciesLower = species.toLowerCase();
                        const index = speciesList.findIndex(s => s.toLowerCase() === speciesLower);
                        
                        if (index >= 0) {
                            speciesList.splice(index, 1);
                            element.classList.remove('selected');
                            element.querySelector('input').checked = false;
                        } else {
                            speciesList.push(speciesLower);
                            element.classList.add('selected');
                            element.querySelector('input').checked = true;
                        }
                        
                        settings.cameras[currentCamera][field] = speciesList;
                        updateSelectedCount(field);
                    }
                    
                    function clearAllSpecies(field) {
                        settings.cameras[currentCamera][field] = [];
                        document.querySelectorAll(`#${field}-grid .species-item`).forEach(item => {
                            item.classList.remove('selected');
                            item.querySelector('input').checked = false;
                        });
                        updateSelectedCount(field);
                    }
                    
                    function updateSelectedCount(field) {
                        const count = (settings.cameras[currentCamera][field] || []).length;
                        const type = field === 'include_species' ? 'include' : 'exclude';
                        const countEl = document.getElementById(type + '-count');
                        if (countEl) {
                            countEl.textContent = getSelectedCountText(settings.cameras[currentCamera][field], type);
                        }
                    }
                    
                    function filterSpecies(query, field) {
                        const queryLower = query.toLowerCase();
                        const grid = document.getElementById(field + '-grid');
                        const items = grid.querySelectorAll('.species-item');
                        const categories = grid.querySelectorAll('.species-category');
                        
                        // Track which categories have visible items
                        const categoryVisibility = {};
                        
                        items.forEach(item => {
                            const label = item.querySelector('label').textContent.toLowerCase();
                            const matches = label.includes(queryLower);
                            item.style.display = matches ? '' : 'none';
                        });
                        
                        // Hide empty categories
                        categories.forEach(cat => {
                            let nextEl = cat.nextElementSibling;
                            let hasVisible = false;
                            while (nextEl && !nextEl.classList.contains('species-category')) {
                                if (nextEl.style.display !== 'none') {
                                    hasVisible = true;
                                    break;
                                }
                                nextEl = nextEl.nextElementSibling;
                            }
                            cat.style.display = hasVisible ? '' : 'none';
                        });
                    }
                    
                    function updateSlider(slider, field) {
                        const value = parseInt(slider.value);
                        document.getElementById(field + '-value').textContent = value + '%';
                        settings.cameras[currentCamera].thresholds[field] = value / 100;
                    }
                    
                    function updateValue(field, value) {
                        if (field === 'min_frames') {
                            value = parseInt(value);
                        }
                        settings.cameras[currentCamera].thresholds[field] = value;
                    }
                    
                    async function saveSettings() {
                        const btn = document.getElementById('saveBtn');
                        btn.disabled = true;
                        btn.textContent = 'Saving...';
                        
                        try {
                            const response = await fetch('/api/settings', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(settings)
                            });
                            
                            if (response.ok) {
                                originalSettings = JSON.parse(JSON.stringify(settings));
                                showToast('Settings saved successfully!', 'success');
                            } else {
                                const text = await response.text();
                                showToast('Error: ' + text, 'error');
                            }
                        } catch (e) {
                            showToast('Error saving settings: ' + e.message, 'error');
                        } finally {
                            btn.disabled = false;
                            btn.textContent = 'Save Changes';
                        }
                    }
                    
                    function resetSettings() {
                        settings = JSON.parse(JSON.stringify(originalSettings));
                        if (currentCamera) {
                            renderSettings(currentCamera);
                        } else {
                            renderGlobalSettings();
                        }
                        showToast('Settings reset to last saved state', 'success');
                    }
                    
                    function showToast(message, type) {
                        const toast = document.getElementById('toast');
                        toast.textContent = message;
                        toast.className = 'toast ' + type + ' visible';
                        setTimeout(() => {
                            toast.classList.remove('visible');
                        }, 3000);
                    }
                    
                    // Load settings on page load
                    loadSettings();
                </script>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_get_settings(self, request):
        """API endpoint to get current settings for all cameras."""
        # Get recent detections from clips
        loop = asyncio.get_running_loop()
        recent_detections = await loop.run_in_executor(None, self._get_recent_detections)
        
        # Get runtime config for global settings
        runtime = self.runtime
        if runtime is None:
            return web.json_response({'error': 'Runtime not initialized'}, status=500)
        
        cameras = {}
        for cam_id, worker in self.workers.items():
            cam = worker.camera
            cameras[cam_id] = {
                'name': cam.name,
                'id': cam.id,
                'location': getattr(cam, 'location', ''),
                'detect_enabled': getattr(cam, 'detect_enabled', True),
                'thresholds': {
                    'confidence': cam.thresholds.confidence,
                    'generic_confidence': getattr(cam.thresholds, 'generic_confidence', 0.9),
                    'min_frames': cam.thresholds.min_frames,
                    'min_duration': cam.thresholds.min_duration,
                },
                'rtsp': {
                    'frame_skip': getattr(cam.rtsp, 'frame_skip', 0),
                    'hwaccel': getattr(cam.rtsp, 'hwaccel', None),
                    'transport': getattr(cam.rtsp, 'transport', 'tcp'),
                    'latency_ms': getattr(cam.rtsp, 'latency_ms', 200),
                },
                'include_species': list(cam.include_species),
                'exclude_species': list(cam.exclude_species),
                'notification': {
                    'priority': getattr(cam.notification, 'priority', 0),
                    'sound': getattr(cam.notification, 'sound', 'pushover'),
                },
                'recent_detections': recent_detections.get(cam_id, {}),
            }
        
        # Global settings
        detector_cfg = runtime.general.detector
        clip_cfg = runtime.general.clip
        retention_cfg = runtime.general.retention
        ebird_cfg = runtime.general.ebird
        
        global_settings = {
            'detector': {
                'backend': detector_cfg.backend,
                'speciesnet_version': getattr(detector_cfg, 'speciesnet_version', 'v4.0.2a'),
                'country': getattr(detector_cfg, 'country', None),
                'admin1_region': getattr(detector_cfg, 'admin1_region', None),
                'generic_confidence': getattr(detector_cfg, 'generic_confidence', 0.9),
            },
            'clip': {
                'pre_seconds': clip_cfg.pre_seconds,
                'post_seconds': clip_cfg.post_seconds,
                'post_analysis': getattr(clip_cfg, 'post_analysis', True),
                'post_analysis_confidence': getattr(clip_cfg, 'post_analysis_confidence', 0.3),
                'post_analysis_generic_confidence': getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5),
                'tracking_enabled': getattr(clip_cfg, 'tracking_enabled', True),
            },
            'retention': {
                'min_days': retention_cfg.min_days,
                'max_days': retention_cfg.max_days,
                'max_utilization_pct': retention_cfg.max_utilization_pct,
            },
            'ebird': {
                'enabled': getattr(ebird_cfg, 'enabled', False),
                'region': getattr(ebird_cfg, 'region', 'US-MN'),
                'days_back': getattr(ebird_cfg, 'days_back', 14),
                'filter_mode': getattr(ebird_cfg, 'filter_mode', 'flag'),
                'cache_hours': getattr(ebird_cfg, 'cache_hours', 24),
            },
            'exclusion_list': list(runtime.general.exclusion_list),
        }
        
        return web.json_response({
            'cameras': cameras,
            'global': global_settings,
        })

    def _get_recent_detections(self):
        """Scan clips directory to get species detection counts per camera."""
        import re
        from collections import defaultdict
        
        clips_dir = self.storage_root / 'clips'
        if not clips_dir.exists():
            return {}
        
        # Count detections per camera per species
        detections = defaultdict(lambda: defaultdict(int))
        
        for cam_dir in clips_dir.iterdir():
            if not cam_dir.is_dir():
                continue
            
            cam_id = cam_dir.name
            
            # Scan all clips in this camera's directory
            for clip_file in cam_dir.rglob('*.mp4'):
                species = self._extract_species_from_filename(clip_file.name)
                for sp in species:
                    if sp and sp.lower() not in ('unknown', 'manual clip', 'no cv result', 'blank', 'empty'):
                        detections[cam_id][sp.lower()] += 1
        
        # Convert to regular dict and sort by count
        result = {}
        for cam_id, species_counts in detections.items():
            sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
            result[cam_id] = {sp: count for sp, count in sorted_species[:20]}  # Top 20
        
        return result

    def _extract_species_from_filename(self, filename: str) -> list:
        """Extract species names from clip filename."""
        import re
        
        # Remove extension
        name = filename.rsplit('.', 1)[0]
        
        # Split by underscore, species is after the timestamp
        parts = name.split('_', 1)
        if len(parts) < 2:
            return []
        
        species_part = parts[1]
        
        # Remove UUIDs
        species_part = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}[;]*', '', species_part)
        
        species_list = []
        for part in species_part.split('+'):
            segments = [s.strip() for s in part.split(';') if s.strip()]
            for segment in reversed(segments):
                seg_lower = segment.lower()
                if seg_lower in ('no cv result', 'unknown', 'blank', 'empty', ''):
                    continue
                clean_name = segment.replace('_', ' ').lower()
                if clean_name and clean_name not in species_list:
                    species_list.append(clean_name)
                break
        
        return species_list

    async def handle_update_settings(self, request):
        """API endpoint to update settings for cameras."""
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")

        cameras_data = data.get('cameras', {})
        global_data = data.get('global', {})
        
        # Update in-memory settings for each camera
        updated_cameras = []
        for cam_id, cam_settings in cameras_data.items():
            worker = self.workers.get(cam_id)
            if not worker:
                continue
            
            cam = worker.camera
            
            # Update thresholds
            if 'thresholds' in cam_settings:
                thresholds = cam_settings['thresholds']
                if 'confidence' in thresholds:
                    cam.thresholds.confidence = float(thresholds['confidence'])
                if 'generic_confidence' in thresholds:
                    cam.thresholds.generic_confidence = float(thresholds['generic_confidence'])
                if 'min_frames' in thresholds:
                    cam.thresholds.min_frames = int(thresholds['min_frames'])
                if 'min_duration' in thresholds:
                    cam.thresholds.min_duration = float(thresholds['min_duration'])
            
            # Update RTSP settings
            if 'rtsp' in cam_settings:
                rtsp = cam_settings['rtsp']
                if 'frame_skip' in rtsp:
                    cam.rtsp.frame_skip = int(rtsp['frame_skip'])
                if 'hwaccel' in rtsp:
                    # hwaccel can be None, or a string like 'nvdec', 'cuda', etc.
                    cam.rtsp.hwaccel = rtsp['hwaccel'] if rtsp['hwaccel'] else None
                if 'latency_ms' in rtsp:
                    cam.rtsp.latency_ms = int(rtsp['latency_ms'])
                if 'transport' in rtsp:
                    cam.rtsp.transport = rtsp['transport']
            
            # Update species filters
            if 'include_species' in cam_settings:
                cam.include_species = list(cam_settings['include_species'])
            if 'exclude_species' in cam_settings:
                cam.exclude_species = list(cam_settings['exclude_species'])
            
            # Update detection enabled
            if 'detect_enabled' in cam_settings:
                cam.detect_enabled = bool(cam_settings['detect_enabled'])
            
            # Update notification settings
            if 'notification' in cam_settings:
                notif = cam_settings['notification']
                if 'priority' in notif:
                    cam.notification.priority = int(notif['priority'])
                if 'sound' in notif:
                    cam.notification.sound = notif['sound']
            
            updated_cameras.append(cam_id)
        
        # Update global settings
        updated_global = False
        if global_data:
            runtime = self.runtime
            
            # Detector settings
            if 'detector' in global_data:
                det = global_data['detector']
                if 'generic_confidence' in det:
                    runtime.general.detector.generic_confidence = float(det['generic_confidence'])
                updated_global = True
            
            # Clip settings
            if 'clip' in global_data:
                clip = global_data['clip']
                if 'pre_seconds' in clip:
                    runtime.general.clip.pre_seconds = float(clip['pre_seconds'])
                if 'post_seconds' in clip:
                    runtime.general.clip.post_seconds = float(clip['post_seconds'])
                if 'post_analysis' in clip:
                    runtime.general.clip.post_analysis = bool(clip['post_analysis'])
                if 'post_analysis_confidence' in clip:
                    runtime.general.clip.post_analysis_confidence = float(clip['post_analysis_confidence'])
                if 'post_analysis_generic_confidence' in clip:
                    runtime.general.clip.post_analysis_generic_confidence = float(clip['post_analysis_generic_confidence'])
                updated_global = True
            
            # Retention settings
            if 'retention' in global_data:
                ret = global_data['retention']
                if 'min_days' in ret:
                    runtime.general.retention.min_days = int(ret['min_days'])
                if 'max_days' in ret:
                    runtime.general.retention.max_days = int(ret['max_days'])
                if 'max_utilization_pct' in ret:
                    runtime.general.retention.max_utilization_pct = int(ret['max_utilization_pct'])
                updated_global = True
            
            # eBird settings
            if 'ebird' in global_data:
                ebird = global_data['ebird']
                if 'enabled' in ebird:
                    runtime.general.ebird.enabled = bool(ebird['enabled'])
                if 'region' in ebird:
                    runtime.general.ebird.region = str(ebird['region'])
                if 'days_back' in ebird:
                    runtime.general.ebird.days_back = int(ebird['days_back'])
                if 'filter_mode' in ebird:
                    runtime.general.ebird.filter_mode = str(ebird['filter_mode'])
                if 'cache_hours' in ebird:
                    runtime.general.ebird.cache_hours = int(ebird['cache_hours'])
                updated_global = True
            
            # Global exclusion list
            if 'exclusion_list' in global_data:
                runtime.general.exclusion_list = list(global_data['exclusion_list'])
                updated_global = True
        
        # Persist to config file if available
        if self.config_path and self.config_path.exists():
            try:
                await self._save_config_to_file(global_data if updated_global else None)
                LOGGER.info(f"Settings saved to {self.config_path}")
            except Exception as e:
                LOGGER.error(f"Failed to save config file: {e}")
                return web.Response(status=500, text=f"Settings applied but failed to save to file: {e}")
        
        return web.json_response({
            'status': 'ok',
            'updated_cameras': updated_cameras,
            'updated_global': updated_global,
        })

    async def _save_config_to_file(self, global_data=None):
        """Save current settings back to the YAML config file."""
        if not self.config_path:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_config_sync, global_data)

    def _save_config_sync(self, global_data=None):
        """Synchronously save config to file."""
        # Read existing config to preserve non-editable settings
        with self.config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Update global settings
        if global_data:
            if 'general' not in config:
                config['general'] = {}
            
            # Detector settings
            if 'detector' in global_data:
                if 'detector' not in config['general']:
                    config['general']['detector'] = {}
                det = global_data['detector']
                if 'generic_confidence' in det:
                    config['general']['detector']['generic_confidence'] = det['generic_confidence']
            
            # Clip settings
            if 'clip' in global_data:
                if 'clip' not in config['general']:
                    config['general']['clip'] = {}
                clip = global_data['clip']
                if 'pre_seconds' in clip:
                    config['general']['clip']['pre_seconds'] = clip['pre_seconds']
                if 'post_seconds' in clip:
                    config['general']['clip']['post_seconds'] = clip['post_seconds']
                if 'post_analysis' in clip:
                    config['general']['clip']['post_analysis'] = clip['post_analysis']
                if 'post_analysis_confidence' in clip:
                    config['general']['clip']['post_analysis_confidence'] = clip['post_analysis_confidence']
                if 'post_analysis_generic_confidence' in clip:
                    config['general']['clip']['post_analysis_generic_confidence'] = clip['post_analysis_generic_confidence']
            
            # Retention settings
            if 'retention' in global_data:
                if 'retention' not in config['general']:
                    config['general']['retention'] = {}
                ret = global_data['retention']
                if 'min_days' in ret:
                    config['general']['retention']['min_days'] = ret['min_days']
                if 'max_days' in ret:
                    config['general']['retention']['max_days'] = ret['max_days']
                if 'max_utilization_pct' in ret:
                    config['general']['retention']['max_utilization_pct'] = ret['max_utilization_pct']
            
            # eBird settings
            if 'ebird' in global_data:
                if 'ebird' not in config['general']:
                    config['general']['ebird'] = {}
                ebird = global_data['ebird']
                if 'enabled' in ebird:
                    config['general']['ebird']['enabled'] = ebird['enabled']
                if 'region' in ebird:
                    config['general']['ebird']['region'] = ebird['region']
                if 'days_back' in ebird:
                    config['general']['ebird']['days_back'] = ebird['days_back']
                if 'filter_mode' in ebird:
                    config['general']['ebird']['filter_mode'] = ebird['filter_mode']
                if 'cache_hours' in ebird:
                    config['general']['ebird']['cache_hours'] = ebird['cache_hours']
            
            # Exclusion list
            if 'exclusion_list' in global_data:
                config['general']['exclusion_list'] = global_data['exclusion_list']
        
        # Update camera settings
        for cam_id, worker in self.workers.items():
            cam = worker.camera
            # Find matching camera in config
            for cam_cfg in config.get('cameras', []):
                if cam_cfg.get('id') == cam_id:
                    # Update thresholds
                    if 'thresholds' not in cam_cfg:
                        cam_cfg['thresholds'] = {}
                    cam_cfg['thresholds']['confidence'] = cam.thresholds.confidence
                    cam_cfg['thresholds']['generic_confidence'] = getattr(cam.thresholds, 'generic_confidence', 0.9)
                    cam_cfg['thresholds']['min_frames'] = cam.thresholds.min_frames
                    cam_cfg['thresholds']['min_duration'] = cam.thresholds.min_duration
                    
                    # Update RTSP settings
                    if 'rtsp' not in cam_cfg:
                        cam_cfg['rtsp'] = {}
                    cam_cfg['rtsp']['frame_skip'] = getattr(cam.rtsp, 'frame_skip', 0)
                    hwaccel_val = getattr(cam.rtsp, 'hwaccel', None)
                    if hwaccel_val:
                        cam_cfg['rtsp']['hwaccel'] = hwaccel_val
                    elif 'hwaccel' in cam_cfg['rtsp']:
                        del cam_cfg['rtsp']['hwaccel']
                    cam_cfg['rtsp']['latency_ms'] = getattr(cam.rtsp, 'latency_ms', 200)
                    cam_cfg['rtsp']['transport'] = getattr(cam.rtsp, 'transport', 'tcp')
                    
                    # Update detection enabled
                    cam_cfg['detect_enabled'] = getattr(cam, 'detect_enabled', True)
                    
                    # Update species filters
                    cam_cfg['include_species'] = list(cam.include_species)
                    cam_cfg['exclude_species'] = list(cam.exclude_species)
                    
                    # Update notification
                    if 'notification' not in cam_cfg:
                        cam_cfg['notification'] = {}
                    cam_cfg['notification']['priority'] = getattr(cam.notification, 'priority', 0)
                    sound_val = getattr(cam.notification, 'sound', None)
                    if sound_val:
                        cam_cfg['notification']['sound'] = sound_val
                    elif 'sound' in cam_cfg['notification']:
                        del cam_cfg['notification']['sound']
                    break
        
        # Write back to file
        with self.config_path.open('w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    async def start(self):
        # Setup access logger
        access_logger = logging.getLogger('web_access')
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
        
        # Ensure logs directory exists
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_root / 'web_access.log'
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)

        runner = web.AppRunner(self.app, access_log=access_logger)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        LOGGER.info(f"Web server started on http://0.0.0.0:{self.port}")
        
        # Keep the server running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour, repeat forever
        except asyncio.CancelledError:
            LOGGER.info("Web server shutting down...")
            await runner.cleanup()
