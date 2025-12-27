import asyncio
import logging
import cv2
import json
import numpy as np
import yaml
from aiohttp import web
from typing import Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
    def __init__(self, workers: Dict[str, 'StreamWorker'], storage_root: Path, logs_root: Path, port: int = 8080, config_path: Path = None):
        self.workers = workers
        self.storage_root = storage_root
        self.logs_root = logs_root
        self.port = port
        self.config_path = config_path
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)
        self.app.router.add_post('/save_clip/{camera_id}', self.handle_save_clip)
        self.app.router.add_post('/ptz/{camera_id}', self.handle_ptz)
        self.app.router.add_get('/recordings', self.handle_recordings)
        self.app.router.add_get('/recording/{path:.*}', self.handle_recording_detail)
        self.app.router.add_delete('/recordings', self.handle_delete_recording)
        self.app.router.add_post('/recordings/bulk_delete', self.handle_bulk_delete)
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
        
        # Look for thumbnails matching this clip
        for thumb_file in clip_dir.glob(f"{clip_stem}_thumb_*.jpg"):
            # Extract species from filename: {timestamp}_{species}_thumb_{specific_species}.jpg
            parts = thumb_file.stem.split("_thumb_")
            if len(parts) >= 2:
                species = parts[-1].replace("_", " ").title()
            else:
                species = "Unknown"
            
            rel_path = thumb_file.relative_to(clips_dir)
            thumbnails.append({
                'path': str(rel_path),
                'species': species,
                'url': f"/clips/{rel_path}"
            })
        
        return thumbnails

    def _parse_species_from_filename(self, filename: str) -> str:
        """Extract clean species name from clip filename.
        
        Filename format: timestamp_species.mp4
        Example: 1766587074_animal+bird.mp4 -> "Animal, Bird"
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
            
            for segment in reversed(segments):
                seg_lower = segment.lower()
                # Skip useless values
                if seg_lower in ('no cv result', 'unknown', 'blank', 'empty', ''):
                    continue
                # Use the first meaningful segment
                clean_name = segment.replace('_', ' ').title()
                if clean_name and clean_name not in species_list:
                    species_list.append(clean_name)
                break
        
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
                </div>
            """
        else:
            thumbnails_html = """
                <div class="detection-section">
                    <h2>🔍 Detection Key Frames</h2>
                    <p class="no-thumbnails">No detection thumbnails available for this recording.</p>
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
                    .action-btn:active {{ opacity: 0.8; }}
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
                            
                            Object.keys(settings.cameras).forEach((camId, index) => {
                                const cam = settings.cameras[camId];
                                const btn = document.createElement('button');
                                btn.className = 'camera-tab' + (index === 0 ? ' active' : '');
                                btn.textContent = cam.name;
                                btn.onclick = () => selectCamera(camId);
                                tabsContainer.appendChild(btn);
                            });
                            
                            // Select first camera
                            const firstCam = Object.keys(settings.cameras)[0];
                            if (firstCam) selectCamera(firstCam);
                            
                        } catch (e) {
                            document.getElementById('settingsContent').innerHTML = 
                                '<div class="loading">Error loading settings: ' + e.message + '</div>';
                        }
                    }
                    
                    function selectCamera(camId) {
                        currentCamera = camId;
                        
                        // Update tab styling
                        document.querySelectorAll('.camera-tab').forEach(tab => {
                            tab.classList.remove('active');
                            if (tab.textContent === settings.cameras[camId].name) {
                                tab.classList.add('active');
                            }
                        });
                        
                        renderSettings(camId);
                    }
                    
                    function renderSettings(camId) {
                        const cam = settings.cameras[camId];
                        const recentDetections = cam.recent_detections || {};
                        
                        const html = `
                            <div class="settings-card">
                                <h2>Detection Thresholds</h2>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Confidence Threshold</label>
                                    <div class="slider-container">
                                        <input type="range" min="0" max="100" step="5" 
                                               value="${Math.round(cam.thresholds.confidence * 100)}"
                                               oninput="updateSlider(this, 'confidence')"
                                               data-field="confidence">
                                        <span class="slider-value" id="confidence-value">${Math.round(cam.thresholds.confidence * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Minimum detection confidence required to trigger an event (0-100%)</div>
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
                                <h2>Include Species</h2>
                                <div class="setting-description" style="margin-bottom: 12px;">
                                    Select species to detect. Leave all unchecked to detect everything.
                                </div>
                                <div class="selected-count" id="include-count">${getSelectedCountText(cam.include_species, 'include')}</div>
                                ${renderSpeciesSelector('include_species', cam.include_species || [], null)}
                            </div>
                            
                            <div class="settings-card">
                                <h2>Exclude Species</h2>
                                <div class="setting-description" style="margin-bottom: 12px;">
                                    Select species to always ignore, even if detected.
                                </div>
                                <div class="selected-count" id="exclude-count">${getSelectedCountText(cam.exclude_species, 'exclude')}</div>
                                ${renderSpeciesSelector('exclude_species', cam.exclude_species || [], recentDetections)}
                            </div>
                        `;
                        document.getElementById('settingsContent').innerHTML = html;
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
                        if (currentCamera) renderSettings(currentCamera);
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
        
        cameras = {}
        for cam_id, worker in self.workers.items():
            cam = worker.camera
            cameras[cam_id] = {
                'name': cam.name,
                'id': cam.id,
                'thresholds': {
                    'confidence': cam.thresholds.confidence,
                    'min_frames': cam.thresholds.min_frames,
                    'min_duration': cam.thresholds.min_duration,
                },
                'include_species': list(cam.include_species),
                'exclude_species': list(cam.exclude_species),
                'recent_detections': recent_detections.get(cam_id, {}),
            }
        return web.json_response({'cameras': cameras})

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
                if 'min_frames' in thresholds:
                    cam.thresholds.min_frames = int(thresholds['min_frames'])
                if 'min_duration' in thresholds:
                    cam.thresholds.min_duration = float(thresholds['min_duration'])
            
            # Update species filters
            if 'include_species' in cam_settings:
                cam.include_species = list(cam_settings['include_species'])
            if 'exclude_species' in cam_settings:
                cam.exclude_species = list(cam_settings['exclude_species'])
            
            updated_cameras.append(cam_id)
        
        # Persist to config file if available
        if self.config_path and self.config_path.exists():
            try:
                await self._save_config_to_file()
                LOGGER.info(f"Settings saved to {self.config_path}")
            except Exception as e:
                LOGGER.error(f"Failed to save config file: {e}")
                return web.Response(status=500, text=f"Settings applied but failed to save to file: {e}")
        
        return web.json_response({
            'status': 'ok',
            'updated_cameras': updated_cameras
        })

    async def _save_config_to_file(self):
        """Save current settings back to the YAML config file."""
        if not self.config_path:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_config_sync)

    def _save_config_sync(self):
        """Synchronously save config to file."""
        # Read existing config to preserve non-editable settings
        with self.config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
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
                    cam_cfg['thresholds']['min_frames'] = cam.thresholds.min_frames
                    cam_cfg['thresholds']['min_duration'] = cam.thresholds.min_duration
                    
                    # Update species filters
                    cam_cfg['include_species'] = list(cam.include_species)
                    cam_cfg['exclude_species'] = list(cam.exclude_species)
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
