import asyncio
import logging
import cv2
import numpy as np
from aiohttp import web
from typing import Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Central Standard Time (UTC-6)
CST = timezone(timedelta(hours=-6), 'CST')

if TYPE_CHECKING:
    from .pipeline import StreamWorker

LOGGER = logging.getLogger(__name__)

class WebServer:
    def __init__(self, workers: Dict[str, 'StreamWorker'], storage_root: Path, logs_root: Path, port: int = 8080):
        self.workers = workers
        self.storage_root = storage_root
        self.logs_root = logs_root
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)
        self.app.router.add_post('/save_clip/{camera_id}', self.handle_save_clip)
        self.app.router.add_post('/ptz/{camera_id}', self.handle_ptz)
        self.app.router.add_post('/tracking/{camera_id}', self.handle_toggle_tracking)
        self.app.router.add_get('/recordings', self.handle_recordings)
        self.app.router.add_delete('/recordings', self.handle_delete_recording)
        self.app.router.add_post('/recordings/bulk_delete', self.handle_bulk_delete)
        
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
                        padding: 12px; 
                        border-radius: 8px; 
                    }
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
                    .tracking-toggle { 
                        margin-top: 12px; 
                        padding: 12px; 
                        background: #222; 
                        border-radius: 8px; 
                    }
                    .tracking-toggle label { 
                        display: flex; 
                        align-items: center; 
                        justify-content: space-between; 
                        cursor: pointer;
                        font-size: 0.95em;
                    }
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

                    async function saveClip(camId) {
                        try {
                            const response = await fetch('/save_clip/' + camId, { method: 'POST' });
                            const text = await response.text();
                            alert(text);
                        } catch (e) {
                            alert('Error saving clip: ' + e);
                        }
                    }

                    async function toggleTracking(camId, checkbox) {
                        try {
                            const response = await fetch('/tracking/' + camId, {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ enabled: checkbox.checked })
                            });
                            if (!response.ok) {
                                alert('Failed to toggle tracking');
                                checkbox.checked = !checkbox.checked;
                            }
                        } catch (e) {
                            alert('Error toggling tracking: ' + e);
                            checkbox.checked = !checkbox.checked;
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
                """

            tracking_html = ""
            if worker.camera.tracking.target_camera_id:
                checked = "checked" if worker.camera.tracking.enabled else ""
                target_cam = worker.camera.tracking.target_camera_id
                tracking_html = f"""
                    <div class="tracking-toggle">
                        <label>
                            <span>Auto-Track {target_cam}</span>
                            <div class="toggle-switch">
                                <input type="checkbox" {checked} onchange="toggleTracking('{cam_id}', this)">
                                <span class="toggle-slider"></span>
                            </div>
                        </label>
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
                        {tracking_html}
                        {ptz_html}
                    </div>
            """
            
        html += """
                </div>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_toggle_tracking(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=404, text="Camera not found")
            
        try:
            data = await request.json()
            enabled = bool(data.get('enabled'))
            worker.camera.tracking.enabled = enabled
            LOGGER.info(f"Tracking on {camera_id} set to {enabled}")
            return web.Response(text="OK")
        except Exception as e:
            return web.Response(status=400, text=str(e))

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
                'time': datetime.fromtimestamp(stat.st_mtime, tz=CST),
                'size': stat.st_size,
                'species': 'Manual clip'
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
                
                clips.append({
                    'path': str(rel_path),
                    'camera': cam_dir.name,
                    'date': datetime.fromtimestamp(stat.st_mtime, tz=CST).strftime('%Y-%m-%d'),
                    'filename': clip_file.name,
                    'time': datetime.fromtimestamp(stat.st_mtime, tz=CST),
                    'size': stat.st_size,
                    'species': species
                })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)
        return clips

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
                </div>
                <h1>Recordings</h1>
                
                <div class="recordings-list">
        """
        
        for clip in clips:
            size_mb = clip['size'] / (1024 * 1024)
            escaped_path = clip['path'].replace("'", "\\'")
            species_display = clip.get('species', 'Unknown')
            html += f"""
                    <div class="recording-card" onclick="playVideo('/clips/{clip['path']}', '{clip['filename']}', '{escaped_path}')">
                        <input type="checkbox" class="recording-checkbox" name="clip_select" value="{clip['path']}" onclick="event.stopPropagation(); updateBulkButton();">
                        <div class="recording-info">
                            <div class="recording-species">🐾 {species_display}</div>
                            <div class="recording-camera">{clip['camera']}</div>
                            <div class="recording-time">{clip['time'].strftime('%b %d, %Y at %I:%M %p')}</div>
                            <div class="recording-meta">{size_mb:.1f} MB</div>
                        </div>
                        <div class="recording-actions">
                            <button class="action-btn play-btn" onclick="event.stopPropagation(); playVideo('/clips/{clip['path']}', '{clip['filename']}', '{escaped_path}');" title="Play">
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
