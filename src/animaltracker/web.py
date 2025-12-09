import asyncio
import logging
import cv2
import numpy as np
from aiohttp import web
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import StreamWorker

LOGGER = logging.getLogger(__name__)

class WebServer:
    def __init__(self, workers: Dict[str, 'StreamWorker'], port: int = 8080):
        self.workers = workers
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)

    async def handle_index(self, request):
        html = """
        <html>
            <head>
                <title>Animal Tracker Dashboard</title>
                <style>
                    body { font-family: sans-serif; background: #222; color: #eee; }
                    .camera-grid { display: flex; flex-wrap: wrap; gap: 20px; }
                    .camera-card { background: #333; padding: 10px; border-radius: 8px; }
                    img { max-width: 100%; height: auto; border-radius: 4px; }
                    h2 { margin-top: 0; }
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
                </script>
            </head>
            <body>
                <h1>Animal Tracker Live View</h1>
                <div class="camera-grid">
        """
        
        for cam_id, worker in self.workers.items():
            cam_name = worker.camera.name
            html += f"""
                    <div class="camera-card">
                        <h2>{cam_name} ({cam_id})</h2>
                        <img src="/snapshot/{cam_id}" data-src="/snapshot/{cam_id}" alt="{cam_name}">
                    </div>
            """
            
        html += """
                </div>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_snapshot(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or worker.latest_frame is None:
            return web.Response(status=404, text="Camera not found or no frame available")
            
        # Encode frame to JPEG
        success, buffer = cv2.imencode('.jpg', worker.latest_frame)
        if not success:
            return web.Response(status=500, text="Failed to encode frame")
            
        return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        LOGGER.info(f"Web server started on http://0.0.0.0:{self.port}")
