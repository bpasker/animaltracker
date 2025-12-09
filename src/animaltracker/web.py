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
        self.app.router.add_get('/recordings', self.handle_recordings)
        
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
                    .camera-card { background: #333; padding: 10px; border-radius: 8px; }
                    img { max-width: 100%; height: auto; border-radius: 4px; }
                    h2 { margin-top: 0; }
                    button { background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; }
                    button:hover { background: #45a049; }
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
            html += f"""
                    <div class="camera-card">
                        <h2>{cam_name} ({cam_id})</h2>
                        <img src="/snapshot/{cam_id}" data-src="/snapshot/{cam_id}" alt="{cam_name}">
                        <br>
                        <button onclick="saveClip('{cam_id}')">Save Last 30s</button>
                    </div>
            """
            
        html += """
                </div>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_recordings(self, request):
        clips_dir = self.storage_root / 'clips'
        if not clips_dir.exists():
            return web.Response(text="No recordings found (clips directory missing)", content_type='text/html')

        # Find all mp4 files
        clips = []
        
        # 1. Check for manual clips in root
        for clip_file in clips_dir.glob('*.mp4'):
            stat = clip_file.stat()
            rel_path = clip_file.relative_to(clips_dir)
            # Try to parse camera from filename manual_cam1_123.mp4
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
            for date_dir in cam_dir.iterdir():
                if not date_dir.is_dir(): continue
                for clip_file in date_dir.glob('*.mp4'):
                    stat = clip_file.stat()
                    # Path relative to storage_root/clips for the URL
                    rel_path = clip_file.relative_to(clips_dir)
                    clips.append({
                        'path': str(rel_path),
                        'camera': cam_dir.name,
                        'date': date_dir.name,
                        'filename': clip_file.name,
                        'time': datetime.fromtimestamp(stat.st_mtime),
                        'size': stat.st_size
                    })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)

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
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/">Live View</a>
                    <a href="/recordings">Recordings</a>
                </div>
                <h1>Recent Recordings</h1>
                <table>
                    <thead>
                        <tr>
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
                            <td>{clip['time'].strftime('%Y-%m-%d %H:%M:%S')}</td>
                            <td>{clip['camera']}</td>
                            <td>{clip['filename']}</td>
                            <td>{size_mb:.1f} MB</td>
                            <td><a class="clip-link" href="/clips/{clip['path']}" target="_blank">Play</a></td>
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
            
        # Resize frame for faster web loading (max width 640px)
        frame = worker.latest_frame
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_AREA)

        # Encode frame to JPEG with lower quality (70%)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
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
