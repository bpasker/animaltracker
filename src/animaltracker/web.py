import asyncio
import logging
import cv2
import numpy as np
from aiohttp import web
from typing import Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

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
                <style>
                    body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 20px; }
                    .nav { margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #444; }
                    .nav a { color: #4CAF50; text-decoration: none; font-size: 1.2em; margin-right: 20px; font-weight: bold; }
                    .nav a:hover { text-decoration: underline; }
                    .camera-grid { display: flex; flex-wrap: wrap; gap: 20px; }
                    .camera-card { background: #333; padding: 10px; border-radius: 8px; text-align: center; }
                    img { max-width: 100%; height: auto; border-radius: 4px; }
                    h2 { margin-top: 0; }
                    button { background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; }
                    button:hover { background: #45a049; }
                    .ptz-controls { margin-top: 10px; background: #222; padding: 10px; border-radius: 4px; }
                    .ptz-controls button { margin: 2px; padding: 5px 10px; font-size: 0.9em; background: #555; }
                    .ptz-controls button:hover { background: #666; }
                    .ptz-controls button:active { background: #888; }
                    .tracking-toggle { margin-top: 10px; padding: 10px; background: #222; border-radius: 4px; }
                    .tracking-toggle label { display: flex; align-items: center; justify-content: center; gap: 10px; cursor: pointer; }
                    .tracking-toggle input { width: 20px; height: 20px; }
                </style>
                <script>
                    function refreshImages() {
                        const images = document.querySelectorAll('img');
                        images.forEach(img => {
                            const src = img.getAttribute('data-src');
                            img.src = src + '?t=' + new Date().getTime();
                        });
                    }
                    setInterval(refreshImages, 2000); // Refresh every 2 seconds

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
                                checkbox.checked = !checkbox.checked; // Revert
                            }
                        } catch (e) {
                            alert('Error toggling tracking: ' + e);
                            checkbox.checked = !checkbox.checked; // Revert
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
                    <a href="/">Live View</a>
                    <a href="/recordings">Recordings</a>
                </div>
                <h1>Animal Tracker Live View</h1>
                <div class="camera-grid">
        """
        
        for cam_id, worker in self.workers.items():
            cam_name = worker.camera.name
            ptz_html = ""
            if worker.onvif_client and worker.onvif_profile_token:
                ptz_html = f"""
                    <div class="ptz-controls">
                        <div>
                            <button onmousedown="sendPtz('{cam_id}', 'move', 0, 1, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Up</button>
                        </div>
                        <div>
                            <button onmousedown="sendPtz('{cam_id}', 'move', -1, 0, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Left</button>
                            <button onmousedown="sendPtz('{cam_id}', 'move', 1, 0, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Right</button>
                        </div>
                        <div>
                            <button onmousedown="sendPtz('{cam_id}', 'move', 0, -1, 0)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Down</button>
                        </div>
                        <div style="margin-top: 5px; border-top: 1px solid #444; padding-top: 5px;">
                            <button onmousedown="sendPtz('{cam_id}', 'move', 0, 0, 1)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Zoom +</button>
                            <button onmousedown="sendPtz('{cam_id}', 'move', 0, 0, -1)" onmouseup="sendPtz('{cam_id}', 'stop')" onmouseleave="sendPtz('{cam_id}', 'stop')">Zoom -</button>
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
                            <input type="checkbox" {checked} onchange="toggleTracking('{cam_id}', this)">
                            Auto-Track {target_cam}
                        </label>
                    </div>
                """

            html += f"""
                    <div class="camera-card">
                        <h2>{cam_name} ({cam_id})</h2>
                        <img src="/snapshot/{cam_id}" data-src="/snapshot/{cam_id}" alt="{cam_name}">
                        <br>
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
                'time': datetime.fromtimestamp(stat.st_mtime),
                'size': stat.st_size
            })

        # 2. Check for automated clips in subdirectories
        for cam_dir in clips_dir.iterdir():
            if not cam_dir.is_dir(): continue
            
            # Use rglob to find all mp4 files recursively (handles year/month/day structure)
            for clip_file in cam_dir.rglob('*.mp4'):
                stat = clip_file.stat()
                rel_path = clip_file.relative_to(clips_dir)
                
                clips.append({
                    'path': str(rel_path),
                    'camera': cam_dir.name,
                    'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d'),
                    'filename': clip_file.name,
                    'time': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size
                })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)
        return clips

    async def handle_recordings(self, request):
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings)
        
        if not clips and not (self.storage_root / 'clips').exists():
             return web.Response(text="No recordings found (clips directory missing)", content_type='text/html')

        html = """
        <html>
            <head>
                <title>Recordings - Animal Tracker</title>
                <style>
                    body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 20px; }
                    .nav { margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #444; }
                    .nav a { color: #4CAF50; text-decoration: none; font-size: 1.2em; margin-right: 20px; font-weight: bold; }
                    .nav a:hover { text-decoration: underline; }
                    table { width: 100%; border-collapse: collapse; background: #333; }
                    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #444; }
                    th { background: #2a2a2a; }
                    tr:hover { background: #3a3a3a; }
                    a.clip-link { color: #64B5F6; text-decoration: none; }
                    a.clip-link:hover { text-decoration: underline; }
                    .delete-btn { background: #f44336; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; margin-left: 10px; }
                    .delete-btn:hover { background: #d32f2f; }
                    .bulk-actions { margin: 20px 0; padding: 10px; background: #333; border-radius: 4px; display: flex; align-items: center; gap: 10px; }
                    .bulk-delete-btn { background: #d32f2f; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
                    .bulk-delete-btn:hover { background: #b71c1c; }
                    .bulk-delete-btn:disabled { background: #555; cursor: not-allowed; }
                </style>
                <script>
                    async function deleteClip(path) {
                        if (!confirm('Are you sure you want to delete this clip?')) return;
                        try {
                            const response = await fetch('/recordings?path=' + encodeURIComponent(path), { method: 'DELETE' });
                            const text = await response.text();
                            if (response.ok) {
                                alert(text);
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
                        btn.disabled = checked.length === 0;
                        btn.textContent = checked.length > 0 ? `Delete Selected (${checked.length})` : 'Delete Selected';
                    }

                    async function bulkDelete() {
                        const checked = document.querySelectorAll('input[name="clip_select"]:checked');
                        if (checked.length === 0) return;

                        if (!confirm(`Are you sure you want to delete ${checked.length} clips?`)) return;

                        const paths = Array.from(checked).map(cb => cb.value);
                        
                        try {
                            const response = await fetch('/recordings/bulk_delete', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ paths: paths })
                            });
                            
                            const result = await response.json();
                            alert(`Deleted ${result.deleted_count} of ${result.total_requested} clips.`);
                            location.reload();
                        } catch (e) {
                            alert('Error performing bulk delete: ' + e);
                        }
                    }
                </script>
            </head>
            <body>
                <div class="nav">
                    <a href="/">Live View</a>
                    <a href="/recordings">Recordings</a>
                </div>
                <h1>Recent Recordings</h1>
                
                <div class="bulk-actions">
                    <button id="bulkDeleteBtn" class="bulk-delete-btn" onclick="bulkDelete()" disabled>Delete Selected</button>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th style="width: 40px;"><input type="checkbox" onclick="toggleAll(this)"></th>
                            <th>Time</th>
                            <th>Camera</th>
                            <th>File</th>
                            <th>Size</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for clip in clips:
            size_mb = clip['size'] / (1024 * 1024)
            html += f"""
                        <tr>
                            <td><input type="checkbox" name="clip_select" value="{clip['path']}" onclick="updateBulkButton()"></td>
                            <td>{clip['time'].strftime('%Y-%m-%d %H:%M:%S')}</td>
                            <td>{clip['camera']}</td>
                            <td>{clip['filename']}</td>
                            <td>{size_mb:.1f} MB</td>
                            <td>
                                <a class="clip-link" href="/clips/{clip['path']}" target="_blank">Play</a>
                                <a class="clip-link" href="/clips/{clip['path']}" download style="margin-left: 10px;">Download</a>
                                <button class="delete-btn" onclick="deleteClip('{clip['path']}')">Delete</button>
                            </td>
                        </tr>
            """
            
        html += """
                    </tbody>
                </table>
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
