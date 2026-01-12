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

from .species_names import get_common_name, format_species_display, get_species_icon

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
        # State file for persisting PTZ settings across restarts
        # Store in config directory alongside cameras.yml
        config_dir = config_path.parent if config_path else None
        self.state_file = config_dir / 'ptz_state.json' if config_dir else None
        # Track active reprocessing jobs: {clip_path: {'started': timestamp, 'clip_name': name}}
        self.reprocessing_jobs: Dict[str, dict] = {}
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_root_redirect)
        self.app.router.add_get('/live', self.handle_index)
        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)
        self.app.router.add_post('/save_clip/{camera_id}', self.handle_save_clip)
        self.app.router.add_post('/ptz/{camera_id}', self.handle_ptz)
        self.app.router.add_get('/ptz/{camera_id}/position', self.handle_ptz_position)
        self.app.router.add_get('/ptz/{camera_id}/mode', self.handle_ptz_mode)
        self.app.router.add_post('/ptz/{camera_id}/patrol', self.handle_ptz_patrol)
        self.app.router.add_post('/ptz/{camera_id}/track', self.handle_ptz_track)
        self.app.router.add_post('/ptz/{camera_id}/return_delay', self.handle_ptz_return_delay)
        self.app.router.add_get('/ptz/{camera_id}/presets', self.handle_ptz_presets)
        self.app.router.add_post('/ptz/{camera_id}/presets', self.handle_ptz_set_patrol_presets)
        self.app.router.add_post('/ptz/{camera_id}/goto_preset', self.handle_ptz_goto_preset)
        self.app.router.add_post('/ptz/{camera_id}/save_preset', self.handle_ptz_save_preset)
        self.app.router.add_post('/ptz/calibrate', self.handle_ptz_calibrate)
        self.app.router.add_get('/recordings', self.handle_recordings)
        self.app.router.add_get('/recording/{path:.*}', self.handle_recording_detail)
        self.app.router.add_delete('/recordings', self.handle_delete_recording)
        self.app.router.add_post('/recordings/bulk_delete', self.handle_bulk_delete)
        self.app.router.add_post('/recordings/reprocess', self.handle_reprocess)
        self.app.router.add_get('/recordings/log/{path:.*}', self.handle_get_processing_log)
        # Monitor page and API
        self.app.router.add_get('/monitor', self.handle_monitor_page)
        self.app.router.add_get('/api/monitor', self.handle_get_monitor_data)
        self.app.router.add_get('/api/logs', self.handle_get_logs)
        # Settings page and API
        self.app.router.add_get('/settings', self.handle_settings_page)
        self.app.router.add_get('/api/settings', self.handle_get_settings)
        self.app.router.add_post('/api/settings', self.handle_update_settings)
        
        # Calendar API endpoints
        self.app.router.add_get('/api/recordings/calendar', self.handle_calendar_api)
        self.app.router.add_get('/api/recordings/day/{date}', self.handle_day_api)
        
        # Serve clips directory statically
        clips_path = self.storage_root / 'clips'
        # Ensure it exists so static route doesn't fail on startup
        clips_path.mkdir(parents=True, exist_ok=True)
        self.app.router.add_static('/clips', clips_path, show_index=True)

    async def handle_root_redirect(self, request):
        """Redirect root URL to recordings page."""
        raise web.HTTPFound('/recordings')

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
                        width: 100vw;
                        margin-left: calc(-50vw + 50%);
                        padding-left: 16px;
                        padding-right: 16px;
                        box-sizing: border-box;
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
                    .ptz-position {
                        margin-top: 12px;
                        padding-top: 12px;
                        border-top: 1px solid #333;
                        font-family: 'SF Mono', Monaco, monospace;
                        font-size: 0.8em;
                        color: #aaa;
                    }
                    .ptz-position-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr 1fr;
                        gap: 8px;
                        text-align: center;
                    }
                    .ptz-position-value {
                        background: #333;
                        padding: 8px;
                        border-radius: 4px;
                    }
                    .ptz-position-label {
                        font-size: 0.75em;
                        color: #666;
                        margin-bottom: 2px;
                    }
                    .ptz-position-num {
                        color: #4CAF50;
                        font-size: 1.1em;
                    }
                    .ptz-calibrate {
                        margin-top: 12px;
                        padding-top: 12px;
                        border-top: 1px solid #333;
                    }
                    .ptz-calibrate button {
                        background: #2196F3;
                        width: 100%;
                    }
                    .ptz-calibrate button:disabled {
                        background: #666;
                        cursor: wait;
                    }
                    .ptz-calibrate-status {
                        margin-top: 8px;
                        font-size: 0.8em;
                        color: #aaa;
                        line-height: 1.4;
                    }
                    .ptz-mode {
                        margin-top: 12px;
                        padding: 12px;
                        border-top: 1px solid #333;
                        background: #2a2a2a;
                        border-radius: 4px;
                    }
                    .ptz-mode-status {
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 0.85em;
                    }
                    .ptz-mode-status.patrol { background: #1565C0; color: white; }
                    .ptz-mode-status.tracking { background: #2E7D32; color: white; }
                    .ptz-mode-status.idle { background: #555; color: #aaa; }
                    .ptz-presets {
                        margin-top: 12px;
                        padding: 12px;
                        border-top: 1px solid #333;
                        background: #252525;
                        border-radius: 4px;
                    }
                    .preset-list {
                        display: flex;
                        flex-direction: column;
                        gap: 4px;
                    }
                    .preset-item {
                        display: flex;
                        align-items: center;
                        padding: 6px 8px;
                        background: #333;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: background 0.2s;
                    }
                    .preset-item:hover {
                        background: #404040;
                    }
                    .preset-item.selected {
                        background: #1565C0;
                    }
                    .preset-item input[type="checkbox"] {
                        margin-right: 8px;
                    }
                    .preset-item .preset-name {
                        flex: 1;
                        font-size: 13px;
                    }
                    .preset-item .preset-goto {
                        padding: 2px 6px;
                        font-size: 11px;
                        background: #555;
                        border: none;
                        border-radius: 3px;
                        color: white;
                        cursor: pointer;
                    }
                    .preset-item .preset-goto:hover {
                        background: #666;
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

                    function togglePtz(element) {
                        const ptzControls = element.closest('.ptz-controls');
                        const wasExpanded = ptzControls.classList.contains('expanded');
                        ptzControls.classList.toggle('expanded');
                        
                        // Auto-load presets when first expanded
                        if (!wasExpanded) {
                            const camId = ptzControls.dataset.camId;
                            if (camId) {
                                loadPresets(camId);
                            }
                        }
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
                            // Refresh position after move
                            setTimeout(() => updatePtzPosition(camId), 200);
                        } catch (e) {
                            console.error('PTZ error:', e);
                        }
                    }

                    async function updatePtzPosition(camId) {
                        try {
                            const response = await fetch('/ptz/' + camId + '/position');
                            if (response.ok) {
                                const data = await response.json();
                                const container = document.getElementById('ptz-pos-' + camId);
                                if (container) {
                                    if (data.available) {
                                        container.style.display = 'block';
                                        container.querySelector('.pan-val').textContent = data.pan.toFixed(3);
                                        container.querySelector('.tilt-val').textContent = data.tilt.toFixed(3);
                                        container.querySelector('.zoom-val').textContent = data.zoom.toFixed(3);
                                    } else {
                                        // Hide position display if camera doesn't report position
                                        container.style.display = 'none';
                                    }
                                }
                            }
                        } catch (e) {
                            console.error('Position fetch error:', e);
                        }
                    }

                    // Poll positions for expanded PTZ panels
                    function startPositionPolling() {
                        setInterval(() => {
                            document.querySelectorAll('.ptz-controls.expanded').forEach(panel => {
                                const camId = panel.dataset.camId;
                                if (camId) {
                                    updatePtzPosition(camId);
                                    updatePtzMode(camId);
                                }
                            });
                        }, 1000);
                    }
                    startPositionPolling();

                    async function updatePtzMode(camId) {
                        try {
                            const response = await fetch('/ptz/' + camId + '/mode');
                            if (response.ok) {
                                const data = await response.json();
                                const statusEl = document.getElementById('ptz-status-' + camId);
                                const patrolToggleEl = document.getElementById('patrol-toggle-' + camId);
                                const trackToggleEl = document.getElementById('track-toggle-' + camId);
                                
                                if (statusEl) {
                                    statusEl.textContent = data.mode.toUpperCase();
                                    statusEl.className = 'ptz-mode-status ' + data.mode;
                                }
                                if (patrolToggleEl && !patrolToggleEl.matches(':focus')) {
                                    patrolToggleEl.checked = data.patrol_enabled;
                                }
                                if (trackToggleEl && !trackToggleEl.matches(':focus')) {
                                    trackToggleEl.checked = data.track_enabled;
                                }
                                // Sync return delay slider
                                const delaySlider = document.getElementById('return-delay-' + camId);
                                const delayLabel = document.getElementById('return-delay-value-' + camId);
                                if (delaySlider && !delaySlider.matches(':active') && data.patrol_return_delay) {
                                    delaySlider.value = data.patrol_return_delay;
                                    if (delayLabel) delayLabel.textContent = data.patrol_return_delay + 's';
                                }
                            }
                        } catch (e) {
                            console.error('Mode fetch error:', e);
                        }
                    }

                    async function togglePatrol(camId, enabled) {
                        try {
                            await fetch('/ptz/' + camId + '/patrol', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ enabled: enabled })
                            });
                            // Update status immediately
                            setTimeout(() => updatePtzMode(camId), 200);
                        } catch (e) {
                            console.error('Patrol toggle error:', e);
                        }
                    }

                    async function toggleTrack(camId, enabled) {
                        try {
                            await fetch('/ptz/' + camId + '/track', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ enabled: enabled })
                            });
                            // Update status immediately
                            setTimeout(() => updatePtzMode(camId), 200);
                        } catch (e) {
                            console.error('Track toggle error:', e);
                        }
                    }

                    function updateReturnDelayLabel(camId, value) {
                        document.getElementById('return-delay-value-' + camId).textContent = value + 's';
                    }

                    async function setReturnDelay(camId, value) {
                        try {
                            await fetch('/ptz/' + camId + '/return_delay', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ delay: parseFloat(value) })
                            });
                        } catch (e) {
                            console.error('Return delay error:', e);
                        }
                    }

                    async function loadPresets(camId) {
                        const listEl = document.getElementById('preset-list-' + camId);
                        listEl.innerHTML = '<div style="color: #888; font-size: 12px;">Loading presets...</div>';
                        
                        try {
                            const response = await fetch('/ptz/' + camId + '/presets');
                            const data = await response.json();
                            
                            if (data.error) {
                                listEl.innerHTML = '<div style="color: #f44336; font-size: 12px;">Error: ' + data.error + '</div>';
                                return;
                            }
                            
                            if (!data.presets || data.presets.length === 0) {
                                listEl.innerHTML = '<div style="color: #888; font-size: 12px;">No presets found. Create presets in camera settings first.</div>';
                                return;
                            }
                            
                            const activePresets = data.active_patrol_presets || [];
                            
                            let html = '';
                            for (const preset of data.presets) {
                                const token = preset.token || '';
                                const name = preset.name || token;
                                const isActive = activePresets.includes(token);
                                
                                html += `
                                    <div class="preset-item ${isActive ? 'selected' : ''}" data-token="${token}">
                                        <input type="checkbox" ${isActive ? 'checked' : ''} onchange="togglePresetSelection(this, '${camId}', '${token}')">
                                        <span class="preset-name">${name}</span>
                                        <button class="preset-goto" onclick="gotoPreset('${camId}', '${token}')">Go</button>
                                    </div>
                                `;
                            }
                            listEl.innerHTML = html;
                            
                        } catch (e) {
                            console.error('Load presets error:', e);
                            listEl.innerHTML = '<div style="color: #f44336; font-size: 12px;">Failed to load presets</div>';
                        }
                    }

                    function togglePresetSelection(checkbox, camId, token) {
                        const item = checkbox.closest('.preset-item');
                        if (checkbox.checked) {
                            item.classList.add('selected');
                        } else {
                            item.classList.remove('selected');
                        }
                    }

                    async function gotoPreset(camId, presetToken) {
                        try {
                            await fetch('/ptz/' + camId + '/goto_preset', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ preset_token: presetToken })
                            });
                        } catch (e) {
                            console.error('Goto preset error:', e);
                        }
                    }

                    async function applyPatrolPresets(camId) {
                        const listEl = document.getElementById('preset-list-' + camId);
                        const selected = [];
                        
                        listEl.querySelectorAll('.preset-item input[type="checkbox"]:checked').forEach(cb => {
                            const item = cb.closest('.preset-item');
                            selected.push(item.dataset.token);
                        });
                        
                        try {
                            const response = await fetch('/ptz/' + camId + '/presets', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ presets: selected })
                            });
                            const data = await response.json();
                            
                            if (data.success) {
                                alert('Patrol will cycle through ' + selected.length + ' preset(s)');
                            }
                        } catch (e) {
                            console.error('Apply presets error:', e);
                        }
                    }

                    async function clearPatrolPresets(camId) {
                        // Uncheck all
                        const listEl = document.getElementById('preset-list-' + camId);
                        listEl.querySelectorAll('.preset-item').forEach(item => {
                            item.classList.remove('selected');
                            const cb = item.querySelector('input[type="checkbox"]');
                            if (cb) cb.checked = false;
                        });
                        
                        // Apply empty preset list (will use continuous sweep)
                        try {
                            await fetch('/ptz/' + camId + '/presets', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ presets: [] })
                            });
                            alert('Patrol presets cleared. Using continuous sweep.');
                        } catch (e) {
                            console.error('Clear presets error:', e);
                        }
                    }

                    async function savePreset(camId) {
                        const nameInput = document.getElementById('preset-name-' + camId);
                        const name = nameInput.value.trim();
                        
                        if (!name) {
                            alert('Please enter a preset name');
                            return;
                        }
                        
                        try {
                            const response = await fetch('/ptz/' + camId + '/save_preset', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ name: name })
                            });
                            
                            if (response.ok) {
                                const data = await response.json();
                                alert('Preset "' + name + '" saved!');
                                nameInput.value = '';
                                // Refresh presets list
                                loadPresets(camId);
                            } else {
                                const text = await response.text();
                                alert('Failed to save preset: ' + text);
                            }
                        } catch (e) {
                            console.error('Save preset error:', e);
                            alert('Error saving preset');
                        }
                    }

                    async function runCalibration(wideCamId, zoomCamId) {
                        const btn = document.getElementById('calibrate-btn-' + wideCamId);
                        const status = document.getElementById('calibrate-status-' + wideCamId);
                        
                        btn.disabled = true;
                        btn.textContent = 'Calibrating...';
                        status.textContent = 'Moving PTZ to calibration points...';
                        status.style.color = '#ffa500';
                        
                        try {
                            const response = await fetch('/ptz/calibrate', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    wide_camera_id: wideCamId,
                                    zoom_camera_id: zoomCamId,
                                    grid_size: 3
                                })
                            });
                            
                            const result = await response.json();
                            
                            if (result.error) {
                                status.textContent = 'Error: ' + result.error;
                                status.style.color = '#f44336';
                            } else {
                                const panPx = result.pan_to_pixel_x?.toFixed(3) ?? 'N/A';
                                const tiltPx = result.tilt_to_pixel_y?.toFixed(3) ?? 'N/A';
                                const centerX = result.center_x?.toFixed(3) ?? 'N/A';
                                const centerY = result.center_y?.toFixed(3) ?? 'N/A';
                                status.innerHTML = `
                                    <strong>Calibration Complete!</strong><br>
                                    pan_to_pixel: ${panPx}<br>
                                    tilt_to_pixel: ${tiltPx}<br>
                                    center_x: ${centerX}<br>
                                    center_y: ${centerY}<br>
                                    Points: ${result.num_points || 0}
                                `;
                                status.style.color = '#4CAF50';
                            }
                        } catch (e) {
                            status.textContent = 'Error: ' + e;
                            status.style.color = '#f44336';
                        }
                        
                        btn.disabled = false;
                        btn.textContent = 'Auto-Calibrate PTZ';
                    }
                </script>
            </head>
            <body>
                <div class="nav">
                    <a href="/recordings">Recordings</a>
                    <a href="/live" class="active">Live</a>
                    <a href="/monitor">Monitor</a>
                    <a href="/settings">Settings</a>
                </div>
                <h1>Live View</h1>
                <div class="camera-grid">
        """
        
        for cam_id, worker in self.workers.items():
            cam_name = worker.camera.name
            ptz_html = ""
            if worker.onvif_client and worker.onvif_profile_token:
                # Check if this camera has PTZ tracking configured (to show calibration)
                ptz_tracking = getattr(worker.camera, 'ptz_tracking', None)
                target_cam_id = ptz_tracking.target_camera_id if ptz_tracking and ptz_tracking.enabled else None
                
                # Check if this camera has a PTZ tracker attached (for patrol/track controls)
                has_tracker = hasattr(worker, 'ptz_tracker') and worker.ptz_tracker is not None
                
                calibrate_html = ""
                if target_cam_id:
                    calibrate_html = f"""
                            <div class="ptz-calibrate">
                                <button id="calibrate-btn-{cam_id}" onclick="runCalibration('{cam_id}', '{target_cam_id}')">Auto-Calibrate PTZ</button>
                                <div id="calibrate-status-{cam_id}" class="ptz-calibrate-status">
                                    Calibrates PTZ mapping between wide-angle and zoom cameras
                                </div>
                            </div>
                    """
                
                # Patrol/track controls - only show if this camera has a tracker
                patrol_track_html = ""
                if has_tracker:
                    # Determine what camera's PTZ this controls
                    controls_label = ""
                    if target_cam_id:
                        target_name = self.workers.get(target_cam_id, {})
                        if hasattr(target_name, 'camera'):
                            controls_label = f'<div style="font-size: 11px; color: #4CAF50; margin-bottom: 8px;">Controls {target_name.camera.name} PTZ</div>'
                    
                    patrol_track_html = f"""
                            <div class="ptz-mode" id="ptz-mode-{cam_id}">
                                {controls_label}
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span>Patrol</span>
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="patrol-toggle-{cam_id}" onchange="togglePatrol('{cam_id}', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                                <div style="font-size: 11px; color: #888; margin-bottom: 12px;">
                                    Scan for objects using sweep or presets
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span>Track Objects</span>
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="track-toggle-{cam_id}" onchange="toggleTrack('{cam_id}', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                                <div style="font-size: 11px; color: #888; margin-bottom: 12px;">
                                    Follow detected objects automatically
                                </div>
                                <div style="margin-bottom: 12px; padding-top: 8px; border-top: 1px solid #444;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span style="font-size: 12px;">Return to Patrol</span>
                                        <span id="return-delay-value-{cam_id}" style="font-size: 12px; color: #4CAF50;">3.0s</span>
                                    </div>
                                    <input type="range" id="return-delay-{cam_id}" min="0.5" max="30" step="0.5" value="3" 
                                           style="width: 100%; accent-color: #4CAF50;"
                                           oninput="updateReturnDelayLabel('{cam_id}', this.value)"
                                           onchange="setReturnDelay('{cam_id}', this.value)">
                                    <div style="font-size: 10px; color: #666;">Seconds after losing object before resuming patrol</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #888;">Current:</span>
                                    <span class="ptz-mode-status" id="ptz-status-{cam_id}" style="font-weight: bold; color: #888;">--</span>
                                </div>
                            </div>
                            <div class="ptz-presets" id="ptz-presets-{cam_id}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span style="font-weight: bold;">Patrol Presets</span>
                                    <button onclick="loadPresets('{cam_id}')" style="padding: 4px 8px; font-size: 12px;">Refresh</button>
                                </div>
                                <div id="preset-list-{cam_id}" class="preset-list" style="max-height: 200px; overflow-y: auto;">
                                    <div style="color: #888; font-size: 12px;">Click Refresh to load presets...</div>
                                </div>
                                <div style="margin-top: 8px; display: flex; gap: 8px;">
                                    <button onclick="applyPatrolPresets('{cam_id}')" style="flex: 1; padding: 6px;">Apply Selected</button>
                                    <button onclick="clearPatrolPresets('{cam_id}')" style="padding: 6px; background: #666;">Clear</button>
                                </div>
                                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #444;">
                                    <div style="font-size: 12px; color: #888; margin-bottom: 6px;">Save Current Position:</div>
                                    <div style="display: flex; gap: 8px;">
                                        <input type="text" id="preset-name-{cam_id}" placeholder="Preset name" style="flex: 1; padding: 6px; background: #333; border: 1px solid #555; color: white; border-radius: 4px;">
                                        <button onclick="savePreset('{cam_id}')" style="padding: 6px 12px;">Save</button>
                                    </div>
                                </div>
                            </div>
                    """
                
                ptz_html = f"""
                    <div class="ptz-controls" data-cam-id="{cam_id}">
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
                            <div class="ptz-position" id="ptz-pos-{cam_id}">
                                <div class="ptz-position-grid">
                                    <div class="ptz-position-value">
                                        <div class="ptz-position-label">PAN</div>
                                        <div class="ptz-position-num pan-val">--</div>
                                    </div>
                                    <div class="ptz-position-value">
                                        <div class="ptz-position-label">TILT</div>
                                        <div class="ptz-position-num tilt-val">--</div>
                                    </div>
                                    <div class="ptz-position-value">
                                        <div class="ptz-position-label">ZOOM</div>
                                        <div class="ptz-position-num zoom-val">--</div>
                                    </div>
                                </div>
                            </div>
                            {patrol_track_html}
                            {calibrate_html}
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

    async def handle_ptz_position(self, request):
        """Get current PTZ position for a camera."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or not worker.onvif_client or not worker.onvif_profile_token:
            return web.Response(status=400, text="Camera not found or ONVIF not configured")
            
        try:
            loop = asyncio.get_running_loop()
            position = await loop.run_in_executor(
                None,
                worker.onvif_client.ptz_get_position,
                worker.onvif_profile_token
            )
            return web.json_response(position)
        except Exception as e:
            LOGGER.error(f"PTZ position error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_mode(self, request):
        """Get current PTZ tracking mode for a camera."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=400, text="Camera not found")
        
        # Find if this camera has a PTZ tracker
        tracker = getattr(worker, 'ptz_tracker', None)
        if not tracker:
            return web.json_response({
                'mode': 'idle',
                'patrol_enabled': False,
                'track_enabled': False,
                'patrol_return_delay': 3.0
            })
        
        return web.json_response({
            'mode': tracker._mode.value if hasattr(tracker._mode, 'value') else str(tracker._mode),
            'patrol_enabled': tracker.is_patrol_enabled(),
            'track_enabled': tracker.is_track_enabled(),
            'patrol_return_delay': tracker.patrol_return_delay
        })

    async def handle_ptz_patrol(self, request):
        """Toggle patrol mode for a camera's PTZ tracker."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=400, text="Camera not found")
        
        try:
            data = await request.json()
            enabled = data.get('enabled', True)
            
            tracker = getattr(worker, 'ptz_tracker', None)
            if not tracker:
                return web.json_response({'error': 'No PTZ tracker configured for this camera'}, status=400)
            
            # Use the new method
            tracker.set_patrol_enabled(enabled)
            
            # Persist the state
            self._update_ptz_state(camera_id, patrol_enabled=enabled)
            
            return web.json_response({
                'success': True,
                'patrol_enabled': tracker.is_patrol_enabled(),
                'track_enabled': tracker.is_track_enabled(),
                'mode': tracker._mode.value
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ patrol toggle error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_track(self, request):
        """Toggle object tracking for a camera's PTZ tracker."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=400, text="Camera not found")
        
        try:
            data = await request.json()
            enabled = data.get('enabled', True)
            
            tracker = getattr(worker, 'ptz_tracker', None)
            if not tracker:
                return web.json_response({'error': 'No PTZ tracker configured for this camera'}, status=400)
            
            # Use the new method
            tracker.set_track_enabled(enabled)
            
            # Persist the state
            self._update_ptz_state(camera_id, track_enabled=enabled)
            
            return web.json_response({
                'success': True,
                'patrol_enabled': tracker.is_patrol_enabled(),
                'track_enabled': tracker.is_track_enabled(),
                'mode': tracker._mode.value
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ track toggle error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_return_delay(self, request):
        """Set patrol return delay for a camera's PTZ tracker."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=400, text="Camera not found")
        
        try:
            data = await request.json()
            delay = float(data.get('delay', 3.0))
            
            # Clamp to valid range
            delay = max(0.5, min(30.0, delay))
            
            tracker = getattr(worker, 'ptz_tracker', None)
            if not tracker:
                return web.json_response({'error': 'No PTZ tracker configured for this camera'}, status=400)
            
            tracker.patrol_return_delay = delay
            
            # Persist the state
            self._update_ptz_state(camera_id, patrol_return_delay=delay)
            
            LOGGER.info(f"PTZ return delay set to {delay}s for camera {camera_id}")
            
            return web.json_response({
                'success': True,
                'patrol_return_delay': delay
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ return delay error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_presets(self, request):
        """Get available PTZ presets for a camera."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or not worker.onvif_client or not worker.onvif_profile_token:
            return web.json_response({'presets': [], 'error': 'Camera not found or ONVIF not configured'})
        
        try:
            loop = asyncio.get_running_loop()
            presets = await loop.run_in_executor(
                None,
                worker.onvif_client.ptz_get_presets,
                worker.onvif_profile_token
            )
            
            # Get current patrol presets from tracker
            tracker = getattr(worker, 'ptz_tracker', None)
            active_presets = []
            if tracker:
                active_presets = list(tracker._preset_tokens) if tracker._preset_tokens else []
            
            return web.json_response({
                'presets': presets,
                'active_patrol_presets': active_presets
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ presets error: {e}")
            return web.json_response({'presets': [], 'error': str(e)})

    async def handle_ptz_set_patrol_presets(self, request):
        """Set which presets to use for patrol mode."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=400, text="Camera not found")
        
        try:
            data = await request.json()
            preset_tokens = data.get('presets', [])
            
            tracker = getattr(worker, 'ptz_tracker', None)
            if not tracker:
                return web.json_response({'error': 'No PTZ tracker configured for this camera'}, status=400)
            
            # Update the patrol presets
            tracker.patrol_presets = preset_tokens
            tracker._preset_tokens = preset_tokens
            tracker._current_preset_index = 0
            
            # Persist the state
            self._update_ptz_state(camera_id, patrol_presets=preset_tokens)
            
            if preset_tokens:
                LOGGER.info(f"Updated patrol presets for {camera_id}: {preset_tokens}")
                # Go to first preset if patrol is active
                from .ptz_tracker import PTZMode
                if tracker._mode == PTZMode.PATROL:
                    tracker._goto_current_preset()
            else:
                LOGGER.info(f"Cleared patrol presets for {camera_id}, using continuous sweep")
            
            return web.json_response({
                'success': True,
                'presets': preset_tokens
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ set patrol presets error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_goto_preset(self, request):
        """Move camera to a specific preset."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or not worker.onvif_client or not worker.onvif_profile_token:
            return web.Response(status=400, text="Camera not found or ONVIF not configured")
        
        try:
            data = await request.json()
            preset_token = data.get('preset_token')
            
            if not preset_token:
                return web.Response(status=400, text="Missing preset_token")
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                worker.onvif_client.ptz_goto_preset,
                worker.onvif_profile_token,
                preset_token,
                0.5  # speed
            )
            
            LOGGER.info(f"Moving {camera_id} to preset {preset_token}")
            return web.json_response({'success': True})
            
        except Exception as e:
            LOGGER.error(f"PTZ goto preset error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_save_preset(self, request):
        """Save current PTZ position as a preset."""
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker or not worker.onvif_client or not worker.onvif_profile_token:
            return web.Response(status=400, text="Camera not found or ONVIF not configured")
        
        try:
            data = await request.json()
            preset_name = data.get('name', '').strip()
            
            if not preset_name:
                return web.Response(status=400, text="Missing preset name")
            
            loop = asyncio.get_running_loop()
            preset_token = await loop.run_in_executor(
                None,
                worker.onvif_client.ptz_set_preset,
                worker.onvif_profile_token,
                preset_name,
                None  # Create new preset
            )
            
            LOGGER.info(f"Saved preset '{preset_name}' for {camera_id} with token {preset_token}")
            return web.json_response({
                'success': True,
                'token': preset_token,
                'name': preset_name
            })
            
        except Exception as e:
            LOGGER.error(f"PTZ save preset error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ptz_calibrate(self, request):
        """Run PTZ visual auto-calibration between wide and zoom cameras."""
        from .ptz_visual_calibration import run_visual_calibration
        
        try:
            data = await request.json()
            wide_camera_id = data.get('wide_camera_id')
            zoom_camera_id = data.get('zoom_camera_id')
            grid_size = int(data.get('grid_size', 3))
            
            wide_worker = self.workers.get(wide_camera_id)
            zoom_worker = self.workers.get(zoom_camera_id)
            
            if not wide_worker:
                return web.json_response({'error': f'Wide camera {wide_camera_id} not found'}, status=400)
            if not zoom_worker:
                return web.json_response({'error': f'Zoom camera {zoom_camera_id} not found'}, status=400)
            
            # Run visual calibration
            result = await run_visual_calibration(
                wide_worker=wide_worker,
                zoom_worker=zoom_worker,
                grid_size=grid_size
            )
            
            return web.json_response(result)
            
        except Exception as e:
            LOGGER.error(f"PTZ calibration error: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({'error': str(e)}, status=500)

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
                'raw_species': 'manual',
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
                species_display, raw_species = self._parse_species_from_filename(clip_file.name)
                
                # Find associated thumbnails
                thumbnails = self._get_thumbnails_for_clip(clip_file)
                
                clips.append({
                    'path': str(rel_path),
                    'camera': cam_dir.name,
                    'date': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ).strftime('%Y-%m-%d'),
                    'filename': clip_file.name,
                    'time': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ),
                    'size': stat.st_size,
                    'species': species_display,
                    'raw_species': raw_species,
                    'thumbnails': thumbnails
                })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)
        return clips

    def _get_thumbnails_for_clip(self, clip_path: Path) -> list:
        """Get all thumbnails associated with a clip file.
        
        Returns list of dicts with 'path' (relative to clips dir), 'species', 'url', 
        and optionally 'track_index' for per-track thumbnails.
        """
        clips_dir = self.storage_root / 'clips'
        clip_stem = clip_path.stem
        clip_dir = clip_path.parent
        thumbnails = []
        
        # Look for thumbnails matching this clip
        glob_pattern = f"{clip_stem}_thumb_*.jpg"
        for thumb_file in clip_dir.glob(glob_pattern):
            # Extract species from filename: {timestamp}_{species}_thumb_{specific_species}.jpg
            # or: {timestamp}_{species}_thumb_{specific_species}_t{track_idx}.jpg (new format)
            parts = thumb_file.stem.split("_thumb_")
            track_index = None
            
            if len(parts) >= 2:
                raw_species = parts[-1]
                
                # Check for track index suffix (e.g., "corvidae_t0" or "corvidae_t1")
                track_match = re.match(r'^(.+?)_t(\d+)$', raw_species)
                if track_match:
                    raw_species = track_match.group(1)
                    track_index = int(track_match.group(2))
                else:
                    # Remove trailing legacy index numbers (e.g., "bird_1" -> "bird")
                    raw_species = re.sub(r'_\d+$', '', raw_species)
                    
                species = get_common_name(raw_species)
            else:
                species = "Unknown"
            
            rel_path = thumb_file.relative_to(clips_dir)
            thumb_data = {
                'path': str(rel_path),
                'species': species,
                'url': f"/clips/{rel_path}"
            }
            if track_index is not None:
                thumb_data['track_index'] = track_index
                
            thumbnails.append(thumb_data)
        
        # Sort by track_index if present, to maintain consistent order
        thumbnails.sort(key=lambda x: (x.get('track_index', 999), x['path']))
        
        return thumbnails

    def _parse_species_from_filename(self, filename: str) -> tuple:
        """Extract clean species name from clip filename.
        
        Filename format: timestamp_species.mp4
        Example: 1766587074_bird_passeriformes_cardinalidae.mp4 -> ("Cardinal", "bird_passeriformes_cardinalidae")
        
        Returns:
            Tuple of (display_name, raw_species) where raw_species can be used for icon lookup
        """
        import re
        
        # Remove extension
        name = filename.rsplit('.', 1)[0]
        
        # Split by underscore, species is after the timestamp
        parts = name.split('_', 1)
        if len(parts) < 2:
            return ('Unknown', 'unknown')
        
        species_part = parts[1]
        
        # Handle complex SpeciesNet format with UUIDs and semicolons
        # Remove UUIDs (8-4-4-4-12 hex pattern)
        species_part = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}[;]*', '', species_part)
        
        # Split by + for multiple species
        species_list = []
        raw_species_list = []
        for part in species_part.split('+'):
            # Split by semicolons and get meaningful parts
            segments = [s.strip() for s in part.split(';') if s.strip()]
            
            # Get the raw species identifier (join all meaningful parts)
            raw_species = '_'.join(seg for seg in segments if seg.lower() not in ('no cv result', 'unknown', 'blank', 'empty', ''))
            
            if raw_species:
                raw_species_list.append(raw_species)
                # Use the common name mapping
                common_name = get_common_name(raw_species)
                if common_name and common_name not in species_list:
                    species_list.append(common_name)
        
        if not species_list:
            return ('Unknown', 'unknown')
        
        # Deduplicate and join
        seen = set()
        unique = []
        for s in species_list:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        
        display_name = ', '.join(unique[:3])  # Limit to 3 species for display
        # Use first raw species for icon
        first_raw = raw_species_list[0] if raw_species_list else 'unknown'
        
        return (display_name, first_raw)

    def _build_calendar_data(self, clips: list) -> dict:
        """
        Group clips into hierarchical calendar structure.
        
        Returns dict with years -> months -> days structure plus filter options.
        """
        from collections import defaultdict
        
        # Nested defaultdict for year -> month -> day
        years_data = defaultdict(lambda: {'total': 0, 'months': defaultdict(lambda: {'total': 0, 'days': {}})})
        all_cameras = set()
        all_species = set()
        
        for clip in clips:
            clip_time = clip['time']
            year = clip_time.year
            month = clip_time.month
            day = clip_time.day
            camera = clip['camera']
            species = clip.get('species', 'Unknown')
            
            all_cameras.add(camera)
            # Handle comma-separated species
            for sp in species.split(', '):
                if sp and sp != 'Unknown':
                    all_species.add(sp)
            
            # Increment totals
            years_data[year]['total'] += 1
            years_data[year]['months'][month]['total'] += 1
            
            # Initialize or update day data
            day_key = day
            if day_key not in years_data[year]['months'][month]['days']:
                years_data[year]['months'][month]['days'][day_key] = {
                    'count': 0,
                    'species': set(),
                    'cameras': set(),
                    'first_clip_time': clip_time.strftime('%H:%M'),
                    'last_clip_time': clip_time.strftime('%H:%M'),
                    'clips': []  # Store clip refs for quick access
                }
            
            day_data = years_data[year]['months'][month]['days'][day_key]
            day_data['count'] += 1
            day_data['cameras'].add(camera)
            for sp in species.split(', '):
                if sp and sp != 'Unknown':
                    day_data['species'].add(sp)
            
            # Update time range
            clip_time_str = clip_time.strftime('%H:%M')
            if clip_time_str < day_data['first_clip_time']:
                day_data['first_clip_time'] = clip_time_str
            if clip_time_str > day_data['last_clip_time']:
                day_data['last_clip_time'] = clip_time_str
        
        # Convert sets to lists and defaultdicts to regular dicts for JSON serialization
        result_years = {}
        for year, year_data in sorted(years_data.items(), reverse=True):
            result_years[str(year)] = {
                'total': year_data['total'],
                'months': {}
            }
            for month, month_data in sorted(year_data['months'].items(), reverse=True):
                result_years[str(year)]['months'][str(month)] = {
                    'total': month_data['total'],
                    'days': {}
                }
                for day, day_data in sorted(month_data['days'].items(), reverse=True):
                    result_years[str(year)]['months'][str(month)]['days'][str(day)] = {
                        'count': day_data['count'],
                        'species': sorted(list(day_data['species'])),
                        'cameras': sorted(list(day_data['cameras'])),
                        'first_clip_time': day_data['first_clip_time'],
                        'last_clip_time': day_data['last_clip_time']
                    }
        
        return {
            'years': result_years,
            'filters': {
                'cameras': sorted(list(all_cameras)),
                'species': sorted(list(all_species))
            }
        }

    def _get_clips_for_date(self, clips: list, date_str: str, camera: str = None, species: str = None) -> dict:
        """
        Filter clips for a specific date with optional filters.
        
        Args:
            clips: Full list of clips from _scan_recordings
            date_str: "YYYY-MM-DD" format
            camera: Optional camera filter
            species: Optional species filter
        
        Returns:
            Dict with clips list and summary stats
        """
        from collections import defaultdict
        
        filtered = []
        by_species = defaultdict(int)
        by_camera = defaultdict(int)
        by_hour = defaultdict(int)
        
        for clip in clips:
            clip_date = clip['time'].strftime('%Y-%m-%d')
            if clip_date != date_str:
                continue
            
            # Apply camera filter
            if camera and clip['camera'] != camera:
                continue
            
            # Apply species filter
            clip_species = clip.get('species', 'Unknown')
            if species and species.lower() not in clip_species.lower():
                continue
            
            # Build clip response object
            clip_hour = clip['time'].hour
            clip_data = {
                'path': clip['path'],
                'camera': clip['camera'],
                'time': clip['time'].strftime('%H:%M:%S'),
                'time_display': clip['time'].strftime('%I:%M %p'),
                'hour': clip_hour,
                'species': clip_species,
                'raw_species': clip.get('raw_species', 'unknown'),
                'species_icon': get_species_icon(clip.get('raw_species', 'unknown')),
                'size_mb': round(clip['size'] / (1024 * 1024), 2),
                'filename': clip['filename'],
                'thumbnails': [
                    {'url': f"/clips/{t['path']}", 'species': t['species']}
                    for t in clip.get('thumbnails', [])
                ]
            }
            filtered.append(clip_data)
            
            # Update stats
            by_camera[clip['camera']] += 1
            by_hour[clip_hour] += 1
            for sp in clip_species.split(', '):
                if sp:
                    by_species[sp] += 1
        
        # Sort by time
        filtered.sort(key=lambda x: x['time'])
        
        # Find peak hour
        peak_hour = max(by_hour.keys(), key=lambda h: by_hour[h]) if by_hour else None
        
        return {
            'date': date_str,
            'clips': filtered,
            'summary': {
                'total': len(filtered),
                'by_species': dict(by_species),
                'by_camera': dict(by_camera),
                'by_hour': dict(by_hour),
                'peak_hour': peak_hour
            }
        }

    async def handle_calendar_api(self, request):
        """GET /api/recordings/calendar - Returns full calendar structure as JSON"""
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings)
        calendar_data = self._build_calendar_data(clips)
        return web.json_response(calendar_data)

    async def handle_day_api(self, request):
        """GET /api/recordings/day/{date} - Returns clips for specific date"""
        date_str = request.match_info.get('date')
        
        # Validate date format
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return web.json_response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)
        
        # Get optional filters from query params
        camera = request.query.get('camera')
        species = request.query.get('species')
        
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings)
        day_data = self._get_clips_for_date(clips, date_str, camera, species)
        
        return web.json_response(day_data)

    async def handle_recordings(self, request):
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings)
        
        if not clips and not (self.storage_root / 'clips').exists():
             return web.Response(text="No recordings found (clips directory missing)", content_type='text/html')

        # Get current date for default view
        now = datetime.now(CENTRAL_TZ)
        
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
                        width: 100vw;
                        margin-left: calc(-50vw + 50%);
                        padding-left: 16px;
                        padding-right: 16px;
                        box-sizing: border-box;
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
                    
                    /* Calendar Container */
                    .calendar-container { background: #1a1a1a; border-radius: 12px; padding: 0; }
                    .calendar-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
                    .calendar-nav { display: flex; align-items: center; gap: 12px; }
                    .nav-btn { 
                        background: #333; 
                        border: none; 
                        color: #fff; 
                        width: 40px; 
                        height: 40px; 
                        border-radius: 8px; 
                        cursor: pointer; 
                        font-size: 1.2em;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: background 0.2s;
                    }
                    .nav-btn:hover { background: #444; }
                    .nav-btn:active { background: #555; }
                    .current-month { font-size: 1.3em; font-weight: 600; margin: 0; min-width: 180px; text-align: center; }
                    .today-btn {
                        background: #333;
                        border: none;
                        color: #4CAF50;
                        padding: 8px 16px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 0.9em;
                        font-weight: 500;
                        transition: background 0.2s;
                    }
                    .today-btn:hover { background: #444; }

                    /* View Tabs */
                    .view-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
                    .view-tab { 
                        background: #333; 
                        border: none; 
                        color: #aaa; 
                        padding: 10px 20px; 
                        border-radius: 8px; 
                        cursor: pointer;
                        font-size: 0.95em;
                        font-weight: 500;
                        transition: background 0.2s, color 0.2s;
                    }
                    .view-tab:hover { background: #444; color: #fff; }
                    .view-tab.active { background: #4CAF50; color: white; }

                    /* Calendar Grid */
                    .calendar-grid { 
                        display: grid; 
                        grid-template-columns: repeat(7, 1fr); 
                        gap: 4px;
                        margin-bottom: 16px;
                    }
                    .calendar-weekday { 
                        text-align: center; 
                        padding: 12px 8px; 
                        color: #888; 
                        font-size: 0.85em; 
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    .calendar-day { 
                        min-height: 80px; 
                        background: #2a2a2a; 
                        border-radius: 8px; 
                        padding: 8px; 
                        cursor: pointer; 
                        position: relative;
                        transition: background 0.15s, transform 0.15s;
                    }
                    .calendar-day:hover { background: #333; }
                    .calendar-day:active { transform: scale(0.98); }
                    .calendar-day.today { border: 2px solid #4CAF50; }
                    .calendar-day.selected { border: 2px solid #2196F3; background: #1a3a5a; }
                    .calendar-day.selected.has-recordings { background: #1a4a3a; }
                    .calendar-day.has-recordings { background: #1e3a1e; }
                    .calendar-day.has-recordings:hover { background: #264a26; }
                    .calendar-day.other-month { opacity: 0.4; }
                    .calendar-day.other-month:hover { opacity: 0.6; }
                    .day-number { 
                        font-size: 0.95em; 
                        font-weight: 600;
                        color: #ccc;
                    }
                    .calendar-day.today .day-number { color: #4CAF50; }
                    .calendar-day.selected .day-number { color: #2196F3; }
                    .day-count { 
                        position: absolute; 
                        bottom: 6px; 
                        right: 6px; 
                        background: #4CAF50; 
                        color: white; 
                        font-size: 0.75em; 
                        font-weight: 600;
                        padding: 3px 10px; 
                        border-radius: 12px;
                    }
                    .day-species {
                        position: absolute;
                        bottom: 6px;
                        left: 6px;
                        font-size: 0.7em;
                        color: #888;
                        max-width: 60%;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .day-cameras {
                        position: absolute;
                        top: 6px;
                        right: 6px;
                        font-size: 0.7em;
                        color: #666;
                    }
                    .day-time-range {
                        font-size: 0.65em;
                        color: #666;
                        margin-top: 2px;
                    }

                    /* Month Summary Stats */
                    .month-stats {
                        display: flex;
                        gap: 12px;
                        margin-bottom: 16px;
                        padding: 12px;
                        background: #2a2a2a;
                        border-radius: 8px;
                        flex-wrap: wrap;
                    }
                    .month-stat {
                        flex: 1;
                        min-width: 80px;
                        text-align: center;
                        padding: 8px;
                    }
                    .month-stat-value {
                        font-size: 1.4em;
                        font-weight: 700;
                        color: #4CAF50;
                    }
                    .month-stat-label {
                        font-size: 0.75em;
                        color: #888;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-top: 2px;
                    }
                    .month-stat.empty .month-stat-value {
                        color: #555;
                    }

                    /* Year Navigation Mini */
                    .year-nav {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        margin-left: 16px;
                    }
                    .year-btn {
                        background: #333;
                        border: none;
                        color: #888;
                        padding: 4px 8px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 0.8em;
                    }
                    .year-btn:hover { background: #444; color: #fff; }

                    /* Month/Year Picker */
                    .month-picker {
                        position: relative;
                        display: inline-block;
                    }
                    .current-month {
                        cursor: pointer;
                        padding: 4px 8px;
                        border-radius: 6px;
                        transition: background 0.2s;
                    }
                    .current-month:hover {
                        background: #333;
                    }
                    .picker-dropdown {
                        position: absolute;
                        top: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        background: #2a2a2a;
                        border-radius: 12px;
                        padding: 16px;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                        z-index: 200;
                        display: none;
                        min-width: 280px;
                    }
                    .picker-dropdown.visible {
                        display: block;
                    }
                    .picker-year-nav {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        margin-bottom: 12px;
                        padding-bottom: 12px;
                        border-bottom: 1px solid #444;
                    }
                    .picker-year {
                        font-size: 1.1em;
                        font-weight: 600;
                        color: #fff;
                    }
                    .picker-year-btn {
                        background: #333;
                        border: none;
                        color: #fff;
                        width: 32px;
                        height: 32px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 1em;
                    }
                    .picker-year-btn:hover { background: #444; }
                    .picker-months {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 8px;
                    }
                    .picker-month {
                        background: #333;
                        border: none;
                        color: #aaa;
                        padding: 10px 8px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.85em;
                        transition: background 0.15s, color 0.15s;
                    }
                    .picker-month:hover {
                        background: #444;
                        color: #fff;
                    }
                    .picker-month.current {
                        background: #4CAF50;
                        color: white;
                    }
                    .picker-month.selected {
                        border: 2px solid #2196F3;
                    }
                    .picker-overlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        z-index: 199;
                        display: none;
                    }
                    .picker-overlay.visible {
                        display: block;
                    }

                    /* Day Panel (slide-in for day details) */
                    .day-panel-container {
                        position: fixed;
                        top: 0;
                        right: -100%;
                        width: 100%;
                        max-width: 450px;
                        height: 100%;
                        background: #1a1a1a;
                        z-index: 500;
                        transition: right 0.3s ease;
                        overflow-y: auto;
                        box-shadow: -4px 0 20px rgba(0,0,0,0.5);
                    }
                    .day-panel-container.visible { right: 0; }
                    .day-panel-overlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0,0,0,0.5);
                        z-index: 499;
                        opacity: 0;
                        visibility: hidden;
                        transition: opacity 0.3s, visibility 0.3s;
                    }
                    .day-panel-overlay.visible { opacity: 1; visibility: visible; }
                    
                    /* Day Panel Content */
                    .day-panel {
                        height: 100%;
                        display: flex;
                        flex-direction: column;
                    }
                    .day-panel-header {
                        padding: 16px;
                        border-bottom: 1px solid #333;
                        flex-shrink: 0;
                    }
                    .day-panel-header-row {
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                    }
                    .day-panel-title {
                        margin: 0 0 4px 0;
                        font-size: 1.15em;
                        font-weight: 600;
                    }
                    .day-panel-subtitle {
                        color: #888;
                        font-size: 0.9em;
                    }
                    .day-panel-close {
                        background: #333;
                        border: none;
                        color: #fff;
                        width: 36px;
                        height: 36px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 1.3em;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: background 0.15s;
                    }
                    .day-panel-close:hover { background: #444; }
                    
                    .day-panel-summary {
                        display: flex;
                        gap: 16px;
                        margin-top: 12px;
                        flex-wrap: wrap;
                    }
                    .day-summary-stat {
                        display: flex;
                        align-items: center;
                        gap: 6px;
                        font-size: 0.85em;
                        color: #aaa;
                    }
                    .day-summary-stat span { color: #4CAF50; font-weight: 600; }
                    
                    .day-panel-content {
                        flex: 1;
                        overflow-y: auto;
                        padding: 16px;
                    }
                    
                    /* Time Groups */
                    .time-group {
                        margin-bottom: 20px;
                    }
                    .time-group:last-child { margin-bottom: 0; }
                    .time-group-header {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 8px 0;
                        margin-bottom: 8px;
                        border-bottom: 1px solid #333;
                    }
                    .time-group-title {
                        font-weight: 600;
                        font-size: 0.95em;
                        color: #ccc;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .time-group-count {
                        background: #333;
                        color: #888;
                        padding: 2px 10px;
                        border-radius: 12px;
                        font-size: 0.8em;
                    }
                    
                    /* Day Panel Clip Cards */
                    .day-clip-card {
                        background: #2a2a2a;
                        border-radius: 10px;
                        padding: 12px;
                        margin-bottom: 8px;
                        display: flex;
                        align-items: flex-start;
                        gap: 12px;
                        cursor: pointer;
                        transition: background 0.15s, transform 0.1s;
                    }
                    .day-clip-card:hover { background: #333; }
                    .day-clip-card:active { transform: scale(0.99); }
                    .day-clip-card:last-child { margin-bottom: 0; }
                    
                    .day-clip-checkbox {
                        width: 20px;
                        height: 20px;
                        accent-color: #4CAF50;
                        flex-shrink: 0;
                        margin-top: 2px;
                    }
                    
                    .day-clip-thumb {
                        width: 80px;
                        height: 60px;
                        border-radius: 6px;
                        object-fit: cover;
                        background: #1a1a1a;
                        flex-shrink: 0;
                    }
                    .day-clip-thumb-placeholder {
                        width: 80px;
                        height: 60px;
                        border-radius: 6px;
                        background: #1a1a1a;
                        flex-shrink: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #444;
                        font-size: 1.5em;
                    }
                    
                    .day-clip-info {
                        flex: 1;
                        min-width: 0;
                    }
                    .day-clip-species {
                        color: #4CAF50;
                        font-weight: 600;
                        font-size: 1em;
                        margin-bottom: 4px;
                    }
                    .day-clip-meta {
                        color: #888;
                        font-size: 0.85em;
                        margin-bottom: 2px;
                    }
                    .day-clip-size {
                        color: #666;
                        font-size: 0.8em;
                    }
                    
                    .day-clip-actions {
                        display: flex;
                        gap: 6px;
                        flex-shrink: 0;
                    }
                    .day-clip-btn {
                        width: 36px;
                        height: 36px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: background 0.15s, transform 0.1s;
                    }
                    .day-clip-btn:active { transform: scale(0.9); }
                    .day-clip-btn.play {
                        background: #4CAF50;
                        color: white;
                    }
                    .day-clip-btn.play:hover { background: #45a049; }
                    .day-clip-btn.delete {
                        background: #333;
                        color: #888;
                    }
                    .day-clip-btn.delete:hover { background: #f44336; color: white; }
                    .day-clip-btn svg {
                        width: 18px;
                        height: 18px;
                    }
                    
                    /* Day Panel Footer Actions */
                    .day-panel-footer {
                        padding: 12px 16px;
                        border-top: 1px solid #333;
                        background: #1a1a1a;
                        flex-shrink: 0;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                    }
                    .day-select-all {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        color: #aaa;
                        font-size: 0.9em;
                    }
                    .day-select-all input {
                        width: 18px;
                        height: 18px;
                        accent-color: #4CAF50;
                    }
                    .day-delete-btn {
                        background: #333;
                        border: none;
                        color: #888;
                        padding: 8px 16px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.9em;
                        transition: background 0.15s, color 0.15s;
                    }
                    .day-delete-btn:hover:not(:disabled) { background: #f44336; color: white; }
                    .day-delete-btn:disabled { opacity: 0.5; cursor: not-allowed; }
                    .day-delete-btn.has-selection {
                        background: #f44336;
                        color: white;
                    }
                    
                    /* Empty state for day panel */
                    .day-empty-state {
                        text-align: center;
                        padding: 40px 20px;
                        color: #666;
                    }
                    .day-empty-state svg {
                        width: 48px;
                        height: 48px;
                        margin-bottom: 12px;
                        opacity: 0.5;
                    }
                    
                    /* Gallery-style grid layout for recordings (like security camera view) */
                    .recordings-list { 
                        display: grid; 
                        grid-template-columns: repeat(2, 1fr); 
                        gap: 8px;
                    }
                    .recording-card {
                        background: #2a2a2a;
                        border-radius: 12px;
                        overflow: hidden;
                        position: relative;
                        cursor: pointer;
                        transition: transform 0.15s, box-shadow 0.15s;
                        aspect-ratio: 4/3;
                    }
                    .recording-card:hover { transform: scale(1.02); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
                    .recording-card:active { transform: scale(0.98); }
                    .recording-checkbox {
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        width: 22px;
                        height: 22px;
                        accent-color: #4CAF50;
                        z-index: 10;
                        opacity: 0;
                        transition: opacity 0.2s;
                    }
                    .recording-card:hover .recording-checkbox,
                    .recording-checkbox:checked { opacity: 1; }
                    .recording-thumb {
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        background: #1a1a1a;
                    }
                    .recording-thumb-placeholder {
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #444;
                        font-size: 2.5em;
                    }
                    /* Overlay at bottom of card */
                    .recording-overlay {
                        position: absolute;
                        bottom: 0;
                        left: 0;
                        right: 0;
                        background: linear-gradient(transparent, rgba(0,0,0,0.85));
                        padding: 30px 12px 12px 12px;
                        pointer-events: none;
                    }
                    .recording-time-ago {
                        font-weight: 600;
                        font-size: 1em;
                        color: #fff;
                        margin-bottom: 2px;
                        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
                    }
                    .recording-camera {
                        font-weight: 400;
                        font-size: 0.85em;
                        color: rgba(255,255,255,0.8);
                        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
                    }
                    /* Species badge in top-right */
                    .recording-species-badge {
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: rgba(76, 175, 80, 0.9);
                        color: white;
                        font-size: 0.75em;
                        font-weight: 600;
                        padding: 4px 10px;
                        border-radius: 20px;
                        display: flex;
                        align-items: center;
                        gap: 5px;
                        backdrop-filter: blur(4px);
                    }
                    .recording-species-badge .species-icon {
                        font-size: 1.1em;
                    }
                    /* Action buttons overlay (hidden by default, shown on hover) */
                    .recording-actions {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        display: flex;
                        gap: 12px;
                        opacity: 0;
                        transition: opacity 0.2s;
                        pointer-events: none;
                    }
                    .recording-card:hover .recording-actions {
                        opacity: 1;
                        pointer-events: auto;
                    }
                    .action-btn {
                        width: 48px;
                        height: 48px;
                        border: none;
                        border-radius: 50%;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: transform 0.15s, opacity 0.15s;
                        backdrop-filter: blur(8px);
                    }
                    .action-btn:active { transform: scale(0.9); }
                    .play-btn { background: rgba(76, 175, 80, 0.95); color: white; }
                    .play-btn:hover { background: #4CAF50; transform: scale(1.1); }
                    .delete-btn { background: rgba(244, 67, 54, 0.95); color: white; }
                    .delete-btn:hover { background: #f44336; transform: scale(1.1); }
                    .action-btn svg { width: 22px; height: 22px; }
                    
                    /* Legacy fields hidden in grid view */
                    .recording-info { display: none; }
                    .thumbnail-badge { display: none; }
                    .recording-time { display: none; }
                    .recording-meta { display: none; }
                    
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
                        grid-column: 1 / -1;
                        text-align: center;
                        padding: 60px 20px;
                        color: #666;
                    }
                    .empty-state svg { width: 64px; height: 64px; margin-bottom: 16px; opacity: 0.5; }
                    
                    /* Loading spinner */
                    .loading {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 40px;
                        color: #888;
                    }
                    .loading::after {
                        content: '';
                        width: 24px;
                        height: 24px;
                        border: 3px solid #333;
                        border-top-color: #4CAF50;
                        border-radius: 50%;
                        animation: spin 0.8s linear infinite;
                        margin-left: 12px;
                    }
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                    
                    /* Hide elements based on view */
                    .calendar-view-only { display: none; }
                    .list-view-only { display: block; }
                    body.view-month .calendar-view-only { display: block; }
                    body.view-month .list-view-only { display: none; }
                    
                    /* List View Header */
                    .list-header {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        margin-bottom: 16px;
                        flex-wrap: wrap;
                        gap: 12px;
                    }
                    .list-summary {
                        color: #888;
                        font-size: 0.95em;
                    }
                    .list-summary strong {
                        color: #4CAF50;
                        font-weight: 600;
                    }
                    .list-controls {
                        display: flex;
                        gap: 8px;
                        align-items: center;
                    }
                    .sort-select {
                        background: #333;
                        border: none;
                        color: #fff;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 0.9em;
                        cursor: pointer;
                    }
                    .sort-select:focus {
                        outline: 2px solid #4CAF50;
                    }
                    .filter-btn {
                        background: #333;
                        border: none;
                        color: #aaa;
                        padding: 8px 12px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.9em;
                        display: flex;
                        align-items: center;
                        gap: 6px;
                    }
                    .filter-btn:hover { background: #444; color: #fff; }
                    .filter-btn.active { background: #4CAF50; color: #fff; }
                    
                    /* Date Group Headers in List View */
                    .date-group-header {
                        grid-column: 1 / -1;
                        background: transparent;
                        padding: 16px 0 8px 0;
                        margin-top: 8px;
                        font-weight: 600;
                        color: #fff;
                        font-size: 1.1em;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                    }
                    .date-group-header:first-child {
                        margin-top: 0;
                        padding-top: 0;
                    }
                    .date-group-count {
                        background: #333;
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-size: 0.75em;
                        color: #888;
                        font-weight: 500;
                    }
                    
                    /* Filters Sidebar */
                    .filters-sidebar {
                        position: fixed;
                        top: 0;
                        left: -100%;
                        width: 100%;
                        max-width: 320px;
                        height: 100%;
                        background: #1a1a1a;
                        z-index: 600;
                        transition: left 0.3s ease;
                        display: flex;
                        flex-direction: column;
                        box-shadow: 4px 0 20px rgba(0,0,0,0.5);
                    }
                    .filters-sidebar.visible { left: 0; }
                    .filters-overlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0,0,0,0.5);
                        z-index: 599;
                        opacity: 0;
                        visibility: hidden;
                        transition: opacity 0.3s, visibility 0.3s;
                    }
                    .filters-overlay.visible { opacity: 1; visibility: visible; }
                    
                    .filters-header {
                        padding: 16px;
                        border-bottom: 1px solid #333;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        flex-shrink: 0;
                    }
                    .filters-title {
                        font-size: 1.1em;
                        font-weight: 600;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .filters-close {
                        background: #333;
                        border: none;
                        color: #fff;
                        width: 36px;
                        height: 36px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 1.3em;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .filters-close:hover { background: #444; }
                    
                    .filters-content {
                        flex: 1;
                        overflow-y: auto;
                        padding: 16px;
                    }
                    
                    .filter-section {
                        margin-bottom: 24px;
                    }
                    .filter-section:last-child { margin-bottom: 0; }
                    .filter-section-title {
                        font-size: 0.85em;
                        font-weight: 600;
                        color: #888;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 12px;
                    }
                    
                    /* Quick Select Buttons */
                    .quick-filters {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 8px;
                    }
                    .quick-filter-btn {
                        background: #2a2a2a;
                        border: none;
                        color: #aaa;
                        padding: 10px 12px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 0.9em;
                        transition: background 0.15s, color 0.15s;
                    }
                    .quick-filter-btn:hover { background: #333; color: #fff; }
                    .quick-filter-btn.active { background: #4CAF50; color: white; }
                    
                    /* Date Range */
                    .date-range-inputs {
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }
                    .date-input-group {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }
                    .date-input-label {
                        color: #888;
                        font-size: 0.9em;
                        width: 45px;
                    }
                    .date-input {
                        flex: 1;
                        background: #2a2a2a;
                        border: 1px solid #333;
                        color: #fff;
                        padding: 10px 12px;
                        border-radius: 8px;
                        font-size: 0.9em;
                    }
                    .date-input:focus {
                        outline: none;
                        border-color: #4CAF50;
                    }
                    
                    /* Filter Checkbox List */
                    .filter-list {
                        display: flex;
                        flex-direction: column;
                        gap: 6px;
                        max-height: 200px;
                        overflow-y: auto;
                    }
                    .filter-item {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding: 8px 10px;
                        background: #2a2a2a;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: background 0.15s;
                    }
                    .filter-item:hover { background: #333; }
                    .filter-item input[type="checkbox"] {
                        width: 18px;
                        height: 18px;
                        accent-color: #4CAF50;
                        flex-shrink: 0;
                    }
                    .filter-item-label {
                        flex: 1;
                        font-size: 0.95em;
                        color: #ddd;
                    }
                    .filter-item-count {
                        color: #666;
                        font-size: 0.85em;
                        background: #1a1a1a;
                        padding: 2px 8px;
                        border-radius: 10px;
                    }
                    
                    /* Active filter indicator */
                    .filter-active-badge {
                        background: #4CAF50;
                        color: white;
                        font-size: 0.75em;
                        padding: 2px 8px;
                        border-radius: 10px;
                        margin-left: 8px;
                    }
                    
                    /* Filters Footer */
                    .filters-footer {
                        padding: 16px;
                        border-top: 1px solid #333;
                        display: flex;
                        gap: 10px;
                        flex-shrink: 0;
                    }
                    .filter-btn {
                        flex: 1;
                        padding: 12px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 0.95em;
                        font-weight: 500;
                        transition: background 0.15s;
                    }
                    .filter-btn.clear {
                        background: #333;
                        color: #aaa;
                    }
                    .filter-btn.clear:hover { background: #444; color: #fff; }
                    .filter-btn.apply {
                        background: #4CAF50;
                        color: white;
                    }
                    .filter-btn.apply:hover { background: #45a049; }
                    
                    /* Filter Toggle Button in Header */
                    .filter-toggle-btn {
                        background: #333;
                        border: none;
                        color: #aaa;
                        padding: 8px 14px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 0.9em;
                        display: flex;
                        align-items: center;
                        gap: 6px;
                        transition: background 0.15s, color 0.15s;
                    }
                    .filter-toggle-btn:hover { background: #444; color: #fff; }
                    .filter-toggle-btn.has-filters {
                        background: #4CAF50;
                        color: white;
                    }
                    .filter-toggle-btn svg {
                        width: 16px;
                        height: 16px;
                    }
                    
                    /* Desktop adjustments */
                    @media (min-width: 768px) {
                        body { padding: 24px; max-width: 1400px; margin: 0 auto; }
                        .calendar-day { min-height: 90px; }
                        .day-panel-container { max-width: 500px; }
                        .filters-sidebar { max-width: 350px; }
                        .recordings-list { 
                            grid-template-columns: repeat(3, 1fr); 
                            gap: 12px;
                        }
                        .recording-card { border-radius: 16px; }
                    }
                    
                    @media (min-width: 1200px) {
                        .recordings-list { 
                            grid-template-columns: repeat(4, 1fr); 
                            gap: 16px;
                        }
                    }
                    
                    /* Mobile adjustments for calendar */
                    @media (max-width: 480px) {
                        .calendar-day { min-height: 60px; padding: 6px; }
                        .day-number { font-size: 0.85em; }
                        .day-count { font-size: 0.65em; padding: 2px 6px; }
                        .day-species { display: none; }
                        .current-month { font-size: 1.1em; min-width: 140px; }
                        .nav-btn { width: 36px; height: 36px; }
                        .filter-toggle-btn span { display: none; }
                        .filter-toggle-btn { padding: 8px 10px; }
                    }
                </style>
            </head>
            <body class="view-list">
                <div class="nav">
                    <a href="/recordings" class="active">Recordings</a>
                    <a href="/live">Live</a>
                    <a href="/monitor">Monitor</a>
                    <a href="/settings">Settings</a>
                </div>
                
                <!-- Filters Sidebar -->
                <div class="filters-overlay" id="filtersOverlay" onclick="CalendarApp.closeFilters()"></div>
                <div class="filters-sidebar" id="filtersSidebar">
                    <div class="filters-header">
                        <div class="filters-title">
                            <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/></svg>
                            Filters
                        </div>
                        <button class="filters-close" onclick="CalendarApp.closeFilters()">×</button>
                    </div>
                    <div class="filters-content">
                        <!-- Quick Select -->
                        <div class="filter-section">
                            <div class="filter-section-title">Quick Select</div>
                            <div class="quick-filters">
                                <button class="quick-filter-btn" onclick="CalendarApp.quickFilter('today')">Today</button>
                                <button class="quick-filter-btn" onclick="CalendarApp.quickFilter('week')">This Week</button>
                                <button class="quick-filter-btn" onclick="CalendarApp.quickFilter('month')">This Month</button>
                                <button class="quick-filter-btn" onclick="CalendarApp.quickFilter('all')">All Time</button>
                            </div>
                        </div>
                        
                        <!-- Date Range -->
                        <div class="filter-section">
                            <div class="filter-section-title">Date Range</div>
                            <div class="date-range-inputs">
                                <div class="date-input-group">
                                    <label class="date-input-label">From:</label>
                                    <input type="date" class="date-input" id="filterDateFrom" onchange="CalendarApp.updateFilterPreview()">
                                </div>
                                <div class="date-input-group">
                                    <label class="date-input-label">To:</label>
                                    <input type="date" class="date-input" id="filterDateTo" onchange="CalendarApp.updateFilterPreview()">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Cameras -->
                        <div class="filter-section">
                            <div class="filter-section-title">Cameras</div>
                            <div class="filter-list" id="filterCamerasList">
                                <!-- Populated by JavaScript -->
                            </div>
                        </div>
                        
                        <!-- Species -->
                        <div class="filter-section">
                            <div class="filter-section-title">Species</div>
                            <div class="filter-list" id="filterSpeciesList">
                                <!-- Populated by JavaScript -->
                            </div>
                        </div>
                    </div>
                    <div class="filters-footer">
                        <button class="filter-btn clear" onclick="CalendarApp.clearFilters()">Clear All</button>
                        <button class="filter-btn apply" onclick="CalendarApp.applyFilters()">Apply</button>
                    </div>
                </div>
                
                <div class="calendar-container">
                    <!-- View Tabs -->
                    <div class="view-tabs">
                        <button class="view-tab active" id="tabList" onclick="CalendarApp.setView('list')">📋 List</button>
                        <button class="view-tab" id="tabMonth" onclick="CalendarApp.setView('month')">📅 Month</button>
                    </div>
                    
                    <!-- Calendar Header with Navigation (Month view only) -->
                    <div class="calendar-header calendar-view-only">
                        <div class="calendar-nav">
                            <button class="nav-btn" onclick="CalendarApp.prevMonth()" title="Previous Month (← key)">
                                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>
                            </button>
                            <div class="month-picker">
                                <h2 class="current-month" id="currentMonth" onclick="CalendarApp.togglePicker()" title="Click to jump to month">January 2026</h2>
                                <div class="picker-dropdown" id="pickerDropdown">
                                    <div class="picker-year-nav">
                                        <button class="picker-year-btn" onclick="CalendarApp.pickerPrevYear()">‹</button>
                                        <span class="picker-year" id="pickerYear">2026</span>
                                        <button class="picker-year-btn" onclick="CalendarApp.pickerNextYear()">›</button>
                                    </div>
                                    <div class="picker-months" id="pickerMonths">
                                        <!-- Months rendered by JS -->
                                    </div>
                                </div>
                            </div>
                            <button class="nav-btn" onclick="CalendarApp.nextMonth()" title="Next Month (→ key)">
                                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg>
                            </button>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <button class="filter-toggle-btn" id="filterToggleBtn" onclick="CalendarApp.toggleFilters()" title="Filters (F key)">
                                <svg fill="currentColor" viewBox="0 0 24 24"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/></svg>
                                <span>Filters</span>
                            </button>
                            <button class="today-btn" onclick="CalendarApp.goToToday()" title="Go to Today (T key)">Today</button>
                        </div>
                    </div>
                    
                    <!-- Calendar Grid (Month View) -->
                    <div class="calendar-view-only">
                        <!-- Picker Overlay (click outside to close) -->
                        <div class="picker-overlay" id="pickerOverlay" onclick="CalendarApp.closePicker()"></div>
                        
                        <!-- Month Summary Stats -->
                        <div class="month-stats" id="monthStats">
                            <div class="month-stat">
                                <div class="month-stat-value" id="statTotal">-</div>
                                <div class="month-stat-label">Recordings</div>
                            </div>
                            <div class="month-stat">
                                <div class="month-stat-value" id="statDays">-</div>
                                <div class="month-stat-label">Active Days</div>
                            </div>
                            <div class="month-stat">
                                <div class="month-stat-value" id="statTopSpecies">-</div>
                                <div class="month-stat-label">Top Species</div>
                            </div>
                            <div class="month-stat">
                                <div class="month-stat-value" id="statCameras">-</div>
                                <div class="month-stat-label">Cameras</div>
                            </div>
                        </div>
                        
                        <div class="calendar-grid" id="calendarGrid">
                            <!-- Weekday headers -->
                            <div class="calendar-weekday">Sun</div>
                            <div class="calendar-weekday">Mon</div>
                            <div class="calendar-weekday">Tue</div>
                            <div class="calendar-weekday">Wed</div>
                            <div class="calendar-weekday">Thu</div>
                            <div class="calendar-weekday">Fri</div>
                            <div class="calendar-weekday">Sat</div>
                            <!-- Day cells will be rendered by JavaScript -->
                        </div>
                        <div class="loading" id="calendarLoading">Loading calendar...</div>
                    </div>
                </div>
                
                <!-- List View (original recordings list) -->
                <div class="list-view-only">
                    <!-- List Header with Summary and Controls -->
                    <div class="list-header">
                        <div class="list-summary">
                            <strong>""" + str(len(clips)) + """</strong> recording""" + ('' if len(clips) == 1 else 's') + """ total
                        </div>
                        <div class="list-controls">
                            <select class="sort-select" id="sortSelect" onchange="sortRecordings(this.value)">
                                <option value="newest">Newest First</option>
                                <option value="oldest">Oldest First</option>
                                <option value="species">By Species</option>
                                <option value="camera">By Camera</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="recordings-list" id="recordingsList">
        """
        
        # Group clips by date for better organization
        current_date = None
        for clip in clips:
            clip_date = clip['time'].strftime('%Y-%m-%d')
            clip_date_display = clip['time'].strftime('%b %d, %Y')
            
            # Add date group header when date changes
            if clip_date != current_date:
                if current_date is not None:
                    # Count clips for this date
                    pass
                current_date = clip_date
                clips_on_date = sum(1 for c in clips if c['time'].strftime('%Y-%m-%d') == clip_date)
                html += f"""
                        <div class="date-group-header" data-date="{clip_date}">
                            <span>{clip_date_display}</span>
                            <span class="date-group-count">{clips_on_date}</span>
                        </div>
                """
            
            size_mb = clip['size'] / (1024 * 1024)
            escaped_path = clip['path'].replace("'", "\\'")
            url_encoded_path = clip['path'].replace('#', '%23')
            species_display = clip.get('species', 'Unknown')
            raw_species = clip.get('raw_species', 'unknown')
            species_icon = get_species_icon(raw_species)
            thumbnails = clip.get('thumbnails', [])
            thumbnail_url = thumbnails[0]['url'] if thumbnails else ''
            thumbnail_html = f'<img class="recording-thumb" src="{thumbnail_url}" alt="" loading="lazy" onerror="this.parentElement.innerHTML=\'<div class=recording-thumb-placeholder>🎬</div>\'">' if thumbnail_url else '<div class="recording-thumb-placeholder">🎬</div>'
            
            # Calculate time ago
            time_iso = clip['time'].isoformat()
            
            html += f"""
                        <div class="recording-card" data-date="{clip_date}" data-species="{species_display}" data-camera="{clip['camera']}" data-time="{time_iso}" onclick="window.location.href='/recording/{url_encoded_path}'">
                            <input type="checkbox" class="recording-checkbox" name="clip_select" value="{clip['path']}" onclick="event.stopPropagation(); updateBulkButton();">
                            {thumbnail_html}
                            <div class="recording-species-badge">
                                <span class="species-icon">{species_icon}</span>
                                {species_display}
                            </div>
                            <div class="recording-overlay">
                                <div class="recording-time-ago" data-time="{time_iso}">Loading...</div>
                                <div class="recording-camera">{clip['camera']}</div>
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
            
        html += f"""
                    </div>
                </div>
                
                <!-- Day Panel Overlay -->
                <div class="day-panel-overlay" id="dayPanelOverlay" onclick="CalendarApp.closeDayPanel()"></div>
                
                <!-- Day Panel (slide-in) -->
                <div class="day-panel-container" id="dayPanelContainer">
                    <!-- Content will be rendered by JavaScript -->
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
                    // Helper function to calculate "time ago" strings
                    function getTimeAgo(dateString) {{
                        const date = new Date(dateString);
                        const now = new Date();
                        const diffMs = now - date;
                        const diffSec = Math.floor(diffMs / 1000);
                        const diffMin = Math.floor(diffSec / 60);
                        const diffHour = Math.floor(diffMin / 60);
                        const diffDay = Math.floor(diffHour / 24);
                        
                        if (diffMin < 1) return 'Just now';
                        if (diffMin === 1) return '1 minute ago';
                        if (diffMin < 60) return diffMin + ' minutes ago';
                        if (diffHour === 1) return '1 hour ago';
                        if (diffHour < 24) return diffHour + ' hours ago';
                        if (diffDay === 1) return 'Yesterday';
                        if (diffDay < 7) return diffDay + ' days ago';
                        
                        // For older dates, show the actual date
                        return date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
                    }}
                    
                    // Update all time ago displays on the page
                    function updateTimeAgoDisplays() {{
                        document.querySelectorAll('.recording-time-ago[data-time]').forEach(el => {{
                            const time = el.getAttribute('data-time');
                            if (time) {{
                                el.textContent = getTimeAgo(time);
                            }}
                        }});
                    }}
                    
                    // Update time ago on load and periodically
                    document.addEventListener('DOMContentLoaded', () => {{
                        updateTimeAgoDisplays();
                        // Update every minute
                        setInterval(updateTimeAgoDisplays, 60000);
                    }});
                    
                    // CalendarApp - Placeholder for Phase 2B-2E JavaScript
                    const CalendarApp = {{
                        // Application state
                        state: {{
                            view: 'list',            // 'month' or 'list'
                            year: {now.year},
                            month: {now.month},
                            calendarData: null,      // Full calendar data from API
                            selectedDate: null,      // Currently selected date (YYYY-MM-DD)
                            dayClips: null,          // Clips for selected day
                            loading: {{
                                calendar: false,
                                dayClips: false
                            }},
                            filters: {{               // Placeholder for Phase 4
                                cameras: [],
                                species: [],
                                dateRange: {{ start: null, end: null }}
                            }}
                        }},
                        
                        // Month names for display
                        MONTHS: ['January', 'February', 'March', 'April', 'May', 'June', 
                                 'July', 'August', 'September', 'October', 'November', 'December'],
                        
                        // Initialize the application
                        async init() {{
                            this.parseUrlParams();
                            this.render();
                            await this.loadCalendarData();
                        }},
                        
                        // Parse URL parameters to restore state
                        parseUrlParams() {{
                            const params = new URLSearchParams(window.location.search);
                            
                            // View mode
                            const view = params.get('view');
                            if (view === 'month' || view === 'list') {{
                                this.state.view = view;
                            }}
                            
                            // Year (validate it's a reasonable year)
                            const year = parseInt(params.get('year'));
                            if (year && year >= 2000 && year <= 2100) {{
                                this.state.year = year;
                            }}
                            
                            // Month (validate 1-12)
                            const month = parseInt(params.get('month'));
                            if (month && month >= 1 && month <= 12) {{
                                this.state.month = month;
                            }}
                            
                            // Selected date
                            const date = params.get('date');
                            if (date && /^\d{{4}}-\d{{2}}-\d{{2}}$/.test(date)) {{
                                this.state.selectedDate = date;
                            }}
                            
                            // Filters - cameras
                            const cameras = params.get('cameras');
                            if (cameras) {{
                                this.state.filters.cameras = cameras.split(',').filter(c => c);
                            }}
                            
                            // Filters - species
                            const species = params.get('species');
                            if (species) {{
                                this.state.filters.species = species.split(',').filter(s => s);
                            }}
                            
                            // Filters - date range
                            const dateFrom = params.get('from');
                            const dateTo = params.get('to');
                            if (dateFrom) this.state.filters.dateRange.start = dateFrom;
                            if (dateTo) this.state.filters.dateRange.end = dateTo;
                            
                            // Update filter button state
                            setTimeout(() => this.updateFilterButtonState(), 0);
                        }},
                        
                        // Track if this is initial load (to avoid pushing initial state)
                        _initialLoad: true,
                        
                        // Update URL to reflect current state (for bookmarking/sharing)
                        updateUrl() {{
                            const url = new URL(window.location);
                            url.searchParams.set('view', this.state.view);
                            url.searchParams.set('year', this.state.year);
                            url.searchParams.set('month', this.state.month);
                            
                            if (this.state.selectedDate) {{
                                url.searchParams.set('date', this.state.selectedDate);
                            }} else {{
                                url.searchParams.delete('date');
                            }}
                            
                            // Filter params
                            if (this.state.filters.cameras.length > 0) {{
                                url.searchParams.set('cameras', this.state.filters.cameras.join(','));
                            }} else {{
                                url.searchParams.delete('cameras');
                            }}
                            
                            if (this.state.filters.species.length > 0) {{
                                url.searchParams.set('species', this.state.filters.species.join(','));
                            }} else {{
                                url.searchParams.delete('species');
                            }}
                            
                            if (this.state.filters.dateRange.start) {{
                                url.searchParams.set('from', this.state.filters.dateRange.start);
                            }} else {{
                                url.searchParams.delete('from');
                            }}
                            
                            if (this.state.filters.dateRange.end) {{
                                url.searchParams.set('to', this.state.filters.dateRange.end);
                            }} else {{
                                url.searchParams.delete('to');
                            }}
                            
                            // Use pushState for navigation history, replaceState for initial load
                            if (this._initialLoad) {{
                                history.replaceState({{ year: this.state.year, month: this.state.month, view: this.state.view }}, '', url);
                                this._initialLoad = false;
                            }} else {{
                                history.pushState({{ year: this.state.year, month: this.state.month, view: this.state.view }}, '', url);
                            }}
                        }},
                        
                        // Main render function - orchestrates all UI updates
                        render() {{
                            this.updateMonthDisplay();
                            this.applyView();
                            
                            if (this.state.view === 'month' && this.state.calendarData) {{
                                this.renderMonthView();
                            }}
                        }},
                        
                        // Update the month/year display in the header
                        updateMonthDisplay() {{
                            const monthName = this.MONTHS[this.state.month - 1];
                            document.getElementById('currentMonth').textContent = `${{monthName}} ${{this.state.year}}`;
                        }},
                        
                        // Apply the current view mode (month/list)
                        applyView() {{
                            document.body.className = 'view-' + this.state.view;
                            document.getElementById('tabMonth').classList.toggle('active', this.state.view === 'month');
                            document.getElementById('tabList').classList.toggle('active', this.state.view === 'list');
                        }},
                        
                        // Switch between month and list views
                        setView(view) {{
                            if (this.state.view === view) return;
                            this.state.view = view;
                            this.render();
                            this.updateUrl();
                        }},
                        
                        // Navigate to previous month
                        prevMonth() {{
                            if (this.state.month === 1) {{
                                this.state.month = 12;
                                this.state.year--;
                            }} else {{
                                this.state.month--;
                            }}
                            this.state.selectedDate = null;
                            this.render();
                            this.updateUrl();
                        }},
                        
                        // Navigate to next month
                        nextMonth() {{
                            if (this.state.month === 12) {{
                                this.state.month = 1;
                                this.state.year++;
                            }} else {{
                                this.state.month++;
                            }}
                            this.state.selectedDate = null;
                            this.render();
                            this.updateUrl();
                        }},
                        
                        // Jump to today's date
                        goToToday() {{
                            const now = new Date();
                            this.state.year = now.getFullYear();
                            this.state.month = now.getMonth() + 1;
                            this.state.selectedDate = null;
                            this.render();
                            this.updateUrl();
                        }},
                        
                        // Navigate to a specific year/month
                        goToMonth(year, month) {{
                            this.state.year = year;
                            this.state.month = month;
                            this.state.selectedDate = null;
                            this.closePicker();
                            this.render();
                            this.updateUrl();
                        }},
                        
                        // Month/Year Picker state
                        pickerYear: {now.year},
                        
                        // Toggle the month picker dropdown
                        togglePicker() {{
                            const dropdown = document.getElementById('pickerDropdown');
                            const overlay = document.getElementById('pickerOverlay');
                            const isVisible = dropdown.classList.contains('visible');
                            
                            if (isVisible) {{
                                this.closePicker();
                            }} else {{
                                // Initialize picker year to current view year
                                this.pickerYear = this.state.year;
                                this.renderPicker();
                                dropdown.classList.add('visible');
                                overlay.classList.add('visible');
                            }}
                        }},
                        
                        // Close the picker
                        closePicker() {{
                            document.getElementById('pickerDropdown').classList.remove('visible');
                            document.getElementById('pickerOverlay').classList.remove('visible');
                        }},
                        
                        // Navigate picker to previous year
                        pickerPrevYear() {{
                            this.pickerYear--;
                            this.renderPicker();
                        }},
                        
                        // Navigate picker to next year
                        pickerNextYear() {{
                            this.pickerYear++;
                            this.renderPicker();
                        }},
                        
                        // Render the picker months grid
                        renderPicker() {{
                            const yearEl = document.getElementById('pickerYear');
                            const monthsEl = document.getElementById('pickerMonths');
                            const now = new Date();
                            
                            yearEl.textContent = this.pickerYear;
                            
                            const monthsHtml = this.MONTHS.map((name, idx) => {{
                                const monthNum = idx + 1;
                                const isCurrent = this.pickerYear === now.getFullYear() && monthNum === now.getMonth() + 1;
                                const isSelected = this.pickerYear === this.state.year && monthNum === this.state.month;
                                
                                // Check if this month has recordings
                                const hasData = this.state.calendarData?.years?.[String(this.pickerYear)]?.months?.[String(monthNum)];
                                
                                let classes = 'picker-month';
                                if (isCurrent) classes += ' current';
                                if (isSelected) classes += ' selected';
                                
                                const countBadge = hasData?.total ? ` (${{hasData.total}})` : '';
                                
                                return `<button class="${{classes}}" onclick="CalendarApp.goToMonth(${{this.pickerYear}}, ${{monthNum}})">${{name.slice(0, 3)}}${{countBadge}}</button>`;
                            }}).join('');
                            
                            monthsEl.innerHTML = monthsHtml;
                        }},
                        
                        // Get available years from calendar data
                        getAvailableYears() {{
                            const years = Object.keys(this.state.calendarData?.years || {{}}).map(Number).sort((a, b) => b - a);
                            return years.length > 0 ? years : [new Date().getFullYear()];
                        }},

                        // Load calendar data from API
                        async loadCalendarData() {{
                            const loadingEl = document.getElementById('calendarLoading');
                            
                            try {{
                                this.state.loading.calendar = true;
                                loadingEl.style.display = 'flex';
                                loadingEl.textContent = 'Loading calendar...';
                                
                                const res = await fetch('/api/recordings/calendar');
                                
                                if (!res.ok) {{
                                    throw new Error(`HTTP ${{res.status}}: ${{res.statusText}}`);
                                }}
                                
                                this.state.calendarData = await res.json();
                                loadingEl.style.display = 'none';
                                
                                // Render after data is loaded
                                this.render();
                                
                            }} catch (e) {{
                                console.error('Failed to load calendar data:', e);
                                loadingEl.innerHTML = `
                                    <span style="color: #f44336;">Failed to load calendar</span>
                                    <button onclick="CalendarApp.loadCalendarData()" 
                                            style="margin-left: 12px; padding: 6px 12px; background: #333; 
                                                   border: none; color: #fff; border-radius: 4px; cursor: pointer;">
                                        Retry
                                    </button>
                                `;
                            }} finally {{
                                this.state.loading.calendar = false;
                            }}
                        }},
                        
                        // Load clips for a specific day
                        async loadDayClips(dateStr) {{
                            try {{
                                this.state.loading.dayClips = true;
                                
                                const res = await fetch(`/api/recordings/day/${{dateStr}}`);
                                
                                if (!res.ok) {{
                                    throw new Error(`HTTP ${{res.status}}: ${{res.statusText}}`);
                                }}
                                
                                const data = await res.json();
                                this.state.dayClips = data;
                                this.state.selectedDate = dateStr;
                                
                                return data;
                                
                            }} catch (e) {{
                                console.error('Failed to load day clips:', e);
                                this.state.dayClips = null;
                                throw e;
                            }} finally {{
                                this.state.loading.dayClips = false;
                            }}
                        }},
                        
                        // Get summary stats for current view
                        getStats() {{
                            const data = this.state.calendarData;
                            if (!data) return null;
                            
                            const yearData = data.years?.[String(this.state.year)];
                            const monthData = yearData?.months?.[String(this.state.month)];
                            
                            return {{
                                yearTotal: yearData?.total || 0,
                                monthTotal: monthData?.total || 0,
                                cameras: data.filters?.cameras || [],
                                species: data.filters?.species || []
                            }};
                        }},
                        
                        // Format a date string for display
                        formatDate(dateStr) {{
                            const [year, month, day] = dateStr.split('-').map(Number);
                            const date = new Date(year, month - 1, day);
                            return date.toLocaleDateString('en-US', {{ 
                                weekday: 'long', 
                                year: 'numeric', 
                                month: 'long', 
                                day: 'numeric' 
                            }});
                        }},
                        
                        // Build a date string from current state + day
                        buildDateStr(day) {{
                            const month = String(this.state.month).padStart(2, '0');
                            const dayStr = String(day).padStart(2, '0');
                            return `${{this.state.year}}-${{month}}-${{dayStr}}`;
                        }},
                        
                        // Render the month view calendar grid
                        renderMonthView() {{
                            const grid = document.getElementById('calendarGrid');
                            const {{ year, month }} = this.state;
                            
                            // Update month stats
                            this.updateMonthStats();
                            
                            // Clear existing day cells (keep weekday headers)
                            const existingDays = grid.querySelectorAll('.calendar-day');
                            existingDays.forEach(d => d.remove());
                            
                            // Calculate calendar layout
                            const firstDay = new Date(year, month - 1, 1).getDay();  // 0=Sun, 6=Sat
                            const daysInMonth = new Date(year, month, 0).getDate();
                            const daysInPrevMonth = new Date(year, month - 1, 0).getDate();
                            
                            // Determine today for highlighting
                            const today = new Date();
                            const isCurrentMonth = today.getFullYear() === year && today.getMonth() + 1 === month;
                            const todayDate = today.getDate();
                            
                            // Add previous month's trailing days (grayed out)
                            for (let i = firstDay - 1; i >= 0; i--) {{
                                const dayNum = daysInPrevMonth - i;
                                const dayEl = this.createDayCell(dayNum, {{ isOtherMonth: true }});
                                grid.appendChild(dayEl);
                            }}
                            
                            // Add current month's days
                            for (let day = 1; day <= daysInMonth; day++) {{
                                const isToday = isCurrentMonth && day === todayDate;
                                const dayData = this.getDayData(year, month, day);
                                const isSelected = this.state.selectedDate === this.buildDateStr(day);
                                
                                const dayEl = this.createDayCell(day, {{
                                    isToday,
                                    isSelected,
                                    dayData
                                }});
                                grid.appendChild(dayEl);
                            }}
                            
                            // Add next month's leading days to complete the grid (6 rows max)
                            const totalCells = firstDay + daysInMonth;
                            const remainingCells = (7 - (totalCells % 7)) % 7;
                            for (let i = 1; i <= remainingCells; i++) {{
                                const dayEl = this.createDayCell(i, {{ isOtherMonth: true }});
                                grid.appendChild(dayEl);
                            }}
                        }},
                        
                        // Update month summary statistics
                        updateMonthStats() {{
                            const {{ year, month }} = this.state;
                            const yearData = this.state.calendarData?.years?.[String(year)];
                            const monthData = yearData?.months?.[String(month)];
                            
                            // Calculate stats
                            let totalClips = 0;
                            let activeDays = 0;
                            let speciesCount = {{}};
                            let cameraSet = new Set();
                            
                            if (monthData?.days) {{
                                Object.values(monthData.days).forEach(day => {{
                                    if (day.count > 0) {{
                                        totalClips += day.count;
                                        activeDays++;
                                        
                                        // Count species
                                        if (day.species) {{
                                            day.species.forEach(s => {{
                                                speciesCount[s] = (speciesCount[s] || 0) + 1;
                                            }});
                                        }}
                                        
                                        // Collect cameras
                                        if (day.cameras) {{
                                            day.cameras.forEach(c => cameraSet.add(c));
                                        }}
                                    }}
                                }});
                            }}
                            
                            // Find top species
                            let topSpecies = '-';
                            let maxCount = 0;
                            Object.entries(speciesCount).forEach(([species, count]) => {{
                                if (count > maxCount) {{
                                    maxCount = count;
                                    topSpecies = species;
                                }}
                            }});
                            
                            // Update DOM
                            const setStatValue = (id, value, isEmpty = false) => {{
                                const el = document.getElementById(id);
                                if (el) {{
                                    el.textContent = value;
                                    el.parentElement.classList.toggle('empty', isEmpty);
                                }}
                            }};
                            
                            setStatValue('statTotal', totalClips || '-', totalClips === 0);
                            setStatValue('statDays', activeDays || '-', activeDays === 0);
                            setStatValue('statTopSpecies', topSpecies, topSpecies === '-');
                            setStatValue('statCameras', cameraSet.size || '-', cameraSet.size === 0);
                        }},
                        
                        // Create a single day cell element
                        createDayCell(day, options = {{}}) {{
                            const {{ isOtherMonth = false, isToday = false, isSelected = false, dayData = null }} = options;
                            
                            const div = document.createElement('div');
                            div.className = 'calendar-day';
                            
                            // Check if this day passes filters
                            const dateStr = isOtherMonth ? null : this.buildDateStr(day);
                            const passesFilters = !isOtherMonth && dayData && 
                                                  this.dayPassesFilters(dayData) && 
                                                  this.dateInRange(dateStr);
                            
                            // Apply state classes
                            if (isOtherMonth) div.classList.add('other-month');
                            if (isToday) div.classList.add('today');
                            if (isSelected) div.classList.add('selected');
                            if (dayData && dayData.count > 0 && passesFilters) div.classList.add('has-recordings');
                            
                            // Build cell content
                            let html = `<div class="day-number">${{day}}</div>`;
                            
                            if (dayData && dayData.count > 0 && passesFilters) {{
                                // Show camera count icon if multiple cameras
                                if (dayData.cameras && dayData.cameras.length > 1) {{
                                    html += `<div class="day-cameras">📹${{dayData.cameras.length}}</div>`;
                                }}
                                
                                html += `<div class="day-count">${{dayData.count}}</div>`;
                                
                                // Show up to 2 species on larger screens
                                if (dayData.species && dayData.species.length > 0) {{
                                    const speciesText = dayData.species.slice(0, 2).join(', ');
                                    html += `<div class="day-species">${{speciesText}}</div>`;
                                }}
                            }}
                            
                            div.innerHTML = html;
                            
                            // Add click handler for days with recordings that pass filters (not other month)
                            if (!isOtherMonth && dayData && dayData.count > 0 && passesFilters) {{
                                div.onclick = () => this.showDayPanel(day);
                                div.style.cursor = 'pointer';
                            }} else if (!isOtherMonth) {{
                                // Days without recordings or filtered out are not clickable
                                div.style.cursor = 'default';
                            }}
                            
                            return div;
                        }},
                        
                        // Look up recording data for a specific day
                        getDayData(year, month, day) {{
                            const yearData = this.state.calendarData?.years?.[String(year)];
                            const monthData = yearData?.months?.[String(month)];
                            return monthData?.days?.[String(day)] || null;
                        }},
                        
                        // Show the day detail panel
                        async showDayPanel(day) {{
                            const dateStr = this.buildDateStr(day);
                            this.state.selectedDate = dateStr;
                            
                            // Update URL to include selected date
                            this.updateUrl();
                            
                            // Re-render to show selection
                            this.renderMonthView();
                            
                            // Show panel with loading state
                            const overlay = document.getElementById('dayPanelOverlay');
                            const container = document.getElementById('dayPanelContainer');
                            
                            overlay.classList.add('visible');
                            container.classList.add('visible');
                            
                            // Show loading state in panel
                            container.innerHTML = `
                                <div style="padding: 20px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                        <h3 style="margin: 0; font-size: 1.1em;">${{this.formatDate(dateStr)}}</h3>
                                        <button onclick="CalendarApp.closeDayPanel()" 
                                                style="background: #333; border: none; color: #fff; width: 36px; height: 36px; 
                                                       border-radius: 8px; cursor: pointer; font-size: 1.2em;">×</button>
                                    </div>
                                    <div class="loading">Loading clips...</div>
                                </div>
                            `;
                            
                            // Load clips for this day
                            try {{
                                const data = await this.loadDayClips(dateStr);
                                this.renderDayPanel(dateStr, data);
                            }} catch (e) {{
                                container.innerHTML = `
                                    <div style="padding: 20px;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                            <h3 style="margin: 0; font-size: 1.1em;">${{this.formatDate(dateStr)}}</h3>
                                            <button onclick="CalendarApp.closeDayPanel()" 
                                                    style="background: #333; border: none; color: #fff; width: 36px; height: 36px; 
                                                           border-radius: 8px; cursor: pointer; font-size: 1.2em;">×</button>
                                        </div>
                                        <div style="color: #f44336; padding: 20px; text-align: center;">
                                            Failed to load clips
                                            <br><br>
                                            <button onclick="CalendarApp.showDayPanel(${{day}})" 
                                                    style="padding: 8px 16px; background: #333; border: none; color: #fff; 
                                                           border-radius: 4px; cursor: pointer;">
                                                Retry
                                            </button>
                                        </div>
                                    </div>
                                `;
                            }}
                        }},
                        
                        // Group clips by time of day
                        groupByTimeOfDay(clips) {{
                            const groups = {{
                                night: {{ icon: '🌙', label: 'Night', range: '12 AM - 6 AM', clips: [] }},
                                morning: {{ icon: '🌅', label: 'Morning', range: '6 AM - 12 PM', clips: [] }},
                                afternoon: {{ icon: '☀️', label: 'Afternoon', range: '12 PM - 6 PM', clips: [] }},
                                evening: {{ icon: '🌆', label: 'Evening', range: '6 PM - 12 AM', clips: [] }}
                            }};
                            
                            clips.forEach(clip => {{
                                // Parse hour from time string (e.g., "08:23:43" or "8:23 AM")
                                let hour = 0;
                                if (clip.time) {{
                                    const match = clip.time.match(/^(\d{{1,2}})/);
                                    if (match) {{
                                        hour = parseInt(match[1], 10);
                                        // Handle 12-hour format with PM
                                        if (clip.time.toLowerCase().includes('pm') && hour !== 12) {{
                                            hour += 12;
                                        }} else if (clip.time.toLowerCase().includes('am') && hour === 12) {{
                                            hour = 0;
                                        }}
                                    }}
                                }}
                                
                                if (hour >= 0 && hour < 6) {{
                                    groups.night.clips.push(clip);
                                }} else if (hour >= 6 && hour < 12) {{
                                    groups.morning.clips.push(clip);
                                }} else if (hour >= 12 && hour < 18) {{
                                    groups.afternoon.clips.push(clip);
                                }} else {{
                                    groups.evening.clips.push(clip);
                                }}
                            }});
                            
                            return groups;
                        }},
                        
                        // Build summary stats for day
                        buildDaySummary(clips, data) {{
                            const summary = data.summary || {{}};
                            const cameras = Object.keys(summary.by_camera || {{}});
                            const species = Object.keys(summary.by_species || {{}});
                            
                            return {{
                                total: clips.length,
                                cameras: cameras.length > 0 ? cameras.join(', ') : 'Unknown',
                                topSpecies: species.length > 0 ? species[0] : null,
                                speciesCount: species.length
                            }};
                        }},
                        
                        // Render a single clip card for day panel
                        renderDayClipCard(clip, index) {{
                            const thumbUrl = clip.thumbnails && clip.thumbnails.length > 0 
                                ? clip.thumbnails[0].url
                                : null;
                            
                            const thumbHtml = thumbUrl 
                                ? `<img class="day-clip-thumb" src="${{thumbUrl}}" alt="" loading="lazy" onerror="this.outerHTML='<div class=\\'day-clip-thumb-placeholder\\'>🎬</div>'">`
                                : `<div class="day-clip-thumb-placeholder">🎬</div>`;
                            
                            const escapedPath = clip.path.replace(/'/g, "\\'");
                            const urlPath = clip.path.replace(/#/g, '%23');
                            const filename = clip.path.split('/').pop();
                            
                            return `
                                <div class="day-clip-card" data-index="${{index}}" data-path="${{clip.path}}"
                                     onclick="window.location.href='/recording/${{urlPath}}'">
                                    <input type="checkbox" class="day-clip-checkbox" name="day_clip_select" value="${{clip.path}}"
                                           onclick="event.stopPropagation(); CalendarApp.updateDaySelection();">
                                    ${{thumbHtml}}
                                    <div class="day-clip-info">
                                        <div class="day-clip-species">${{clip.species_icon || '🐾'}} ${{clip.species || 'Unknown'}}</div>
                                        <div class="day-clip-meta">${{clip.time}} • ${{clip.camera}}</div>
                                        <div class="day-clip-size">${{clip.size_mb?.toFixed(1) || '?'}} MB</div>
                                    </div>
                                    <div class="day-clip-actions">
                                        <button class="day-clip-btn play" 
                                                onclick="event.stopPropagation(); playVideo('/clips/${{clip.path}}', '${{filename}}', '${{escapedPath}}');"
                                                title="Quick Play">
                                            <svg fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                                        </button>
                                        <button class="day-clip-btn delete"
                                                onclick="event.stopPropagation(); CalendarApp.deleteDayClip('${{escapedPath}}');"
                                                title="Delete">
                                            <svg fill="currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                                        </button>
                                    </div>
                                </div>
                            `;
                        }},
                        
                        // Render time group section
                        renderTimeGroup(group, groupKey) {{
                            if (group.clips.length === 0) return '';
                            
                            const clipsHtml = group.clips.map((clip, i) => this.renderDayClipCard(clip, i)).join('');
                            
                            return `
                                <div class="time-group" data-group="${{groupKey}}">
                                    <div class="time-group-header">
                                        <div class="time-group-title">
                                            ${{group.icon}} ${{group.label}}
                                            <span style="color: #666; font-weight: normal; font-size: 0.85em;">(${{group.range}})</span>
                                        </div>
                                        <span class="time-group-count">${{group.clips.length}} clip${{group.clips.length !== 1 ? 's' : ''}}</span>
                                    </div>
                                    ${{clipsHtml}}
                                </div>
                            `;
                        }},
                        
                        // Render the day panel content
                        renderDayPanel(dateStr, data) {{
                            const container = document.getElementById('dayPanelContainer');
                            const clips = data.clips || [];
                            this.state.dayClips = clips;
                            
                            // Build summary
                            const summary = this.buildDaySummary(clips, data);
                            
                            // Empty state
                            if (clips.length === 0) {{
                                container.innerHTML = `
                                    <div class="day-panel">
                                        <div class="day-panel-header">
                                            <div class="day-panel-header-row">
                                                <div>
                                                    <h3 class="day-panel-title">${{this.formatDate(dateStr)}}</h3>
                                                    <div class="day-panel-subtitle">No recordings</div>
                                                </div>
                                                <button class="day-panel-close" onclick="CalendarApp.closeDayPanel()">×</button>
                                            </div>
                                        </div>
                                        <div class="day-panel-content">
                                            <div class="day-empty-state">
                                                <svg fill="currentColor" viewBox="0 0 24 24"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>
                                                <p>No recordings on this day</p>
                                            </div>
                                        </div>
                                    </div>
                                `;
                                return;
                            }}
                            
                            // Group by time of day
                            const groups = this.groupByTimeOfDay(clips);
                            
                            // Render time groups (in order: morning, afternoon, evening, night)
                            const groupOrder = ['morning', 'afternoon', 'evening', 'night'];
                            const groupsHtml = groupOrder.map(key => this.renderTimeGroup(groups[key], key)).join('');
                            
                            container.innerHTML = `
                                <div class="day-panel">
                                    <div class="day-panel-header">
                                        <div class="day-panel-header-row">
                                            <div>
                                                <h3 class="day-panel-title">${{this.formatDate(dateStr)}}</h3>
                                                <div class="day-panel-subtitle">${{clips.length}} recording${{clips.length !== 1 ? 's' : ''}}</div>
                                            </div>
                                            <button class="day-panel-close" onclick="CalendarApp.closeDayPanel()">×</button>
                                        </div>
                                        <div class="day-panel-summary">
                                            <div class="day-summary-stat">📷 <span>${{summary.cameras}}</span></div>
                                            ${{summary.topSpecies ? `<div class="day-summary-stat">🏆 <span>${{summary.topSpecies}}</span></div>` : ''}}
                                        </div>
                                    </div>
                                    <div class="day-panel-content">
                                        ${{groupsHtml}}
                                    </div>
                                    <div class="day-panel-footer">
                                        <label class="day-select-all">
                                            <input type="checkbox" id="daySelectAll" onclick="CalendarApp.toggleDaySelectAll(this)">
                                            Select All
                                        </label>
                                        <button class="day-delete-btn" id="dayDeleteBtn" onclick="CalendarApp.bulkDeleteDayClips()" disabled>
                                            Delete (0)
                                        </button>
                                    </div>
                                </div>
                            `;
                        }},
                        
                        // Update selection count in day panel
                        updateDaySelection() {{
                            const checkboxes = document.querySelectorAll('input[name="day_clip_select"]');
                            const checked = document.querySelectorAll('input[name="day_clip_select"]:checked');
                            const count = checked.length;
                            const total = checkboxes.length;
                            
                            const btn = document.getElementById('dayDeleteBtn');
                            const selectAllCb = document.getElementById('daySelectAll');
                            
                            if (btn) {{
                                btn.disabled = count === 0;
                                btn.textContent = count > 0 ? `Delete (${{count}})` : 'Delete (0)';
                                btn.classList.toggle('has-selection', count > 0);
                            }}
                            
                            if (selectAllCb) {{
                                selectAllCb.checked = count === total && total > 0;
                                selectAllCb.indeterminate = count > 0 && count < total;
                            }}
                        }},
                        
                        // Toggle select all in day panel
                        toggleDaySelectAll(source) {{
                            const checkboxes = document.querySelectorAll('input[name="day_clip_select"]');
                            checkboxes.forEach(cb => cb.checked = source.checked);
                            this.updateDaySelection();
                        }},
                        
                        // Delete a single clip from day panel
                        async deleteDayClip(path) {{
                            if (!confirm('Delete this clip?')) return;
                            
                            try {{
                                const response = await fetch('/recordings?path=' + encodeURIComponent(path), {{ method: 'DELETE' }});
                                if (response.ok) {{
                                    // Refresh the calendar data and day panel
                                    await this.loadCalendarData();
                                    this.renderMonthView();
                                    
                                    // Reload day panel if still open
                                    if (this.state.selectedDate) {{
                                        const data = await this.loadDayClips(this.state.selectedDate);
                                        this.renderDayPanel(this.state.selectedDate, data);
                                    }}
                                }} else {{
                                    const text = await response.text();
                                    alert('Error: ' + text);
                                }}
                            }} catch (e) {{
                                alert('Error deleting clip: ' + e);
                            }}
                        }},
                        
                        // Bulk delete selected clips in day panel
                        async bulkDeleteDayClips() {{
                            const checked = document.querySelectorAll('input[name="day_clip_select"]:checked');
                            if (checked.length === 0) return;
                            
                            if (!confirm(`Delete ${{checked.length}} clip${{checked.length !== 1 ? 's' : ''}}?`)) return;
                            
                            const paths = Array.from(checked).map(cb => cb.value);
                            
                            try {{
                                const response = await fetch('/recordings/bulk_delete', {{
                                    method: 'POST',
                                    headers: {{ 'Content-Type': 'application/json' }},
                                    body: JSON.stringify({{ paths: paths }})
                                }});
                                
                                if (response.ok) {{
                                    // Refresh the calendar data and day panel
                                    await this.loadCalendarData();
                                    this.renderMonthView();
                                    
                                    // Reload day panel
                                    if (this.state.selectedDate) {{
                                        const data = await this.loadDayClips(this.state.selectedDate);
                                        this.renderDayPanel(this.state.selectedDate, data);
                                    }}
                                }} else {{
                                    alert('Error deleting clips');
                                }}
                            }} catch (e) {{
                                alert('Error: ' + e);
                            }}
                        }},
                        
                        // Navigate to next/previous day in day panel
                        async navigateDay(offset) {{
                            if (!this.state.selectedDate) return;
                            
                            // Parse current date
                            const current = new Date(this.state.selectedDate + 'T12:00:00');
                            current.setDate(current.getDate() + offset);
                            
                            const newYear = current.getFullYear();
                            const newMonth = current.getMonth() + 1;
                            const newDay = current.getDate();
                            const newDateStr = `${{newYear}}-${{String(newMonth).padStart(2, '0')}}-${{String(newDay).padStart(2, '0')}}`;
                            
                            // Update month if needed
                            if (newYear !== this.state.year || newMonth !== this.state.month) {{
                                this.state.year = newYear;
                                this.state.month = newMonth;
                            }}
                            
                            // Load and show the new day
                            this.state.selectedDate = newDateStr;
                            this.updateUrl();
                            this.renderMonthView();
                            
                            // Show loading state
                            const container = document.getElementById('dayPanelContainer');
                            container.innerHTML = `
                                <div class="day-panel">
                                    <div class="day-panel-header">
                                        <div class="day-panel-header-row">
                                            <div>
                                                <h3 class="day-panel-title">${{this.formatDate(newDateStr)}}</h3>
                                                <div class="day-panel-subtitle">Loading...</div>
                                            </div>
                                            <button class="day-panel-close" onclick="CalendarApp.closeDayPanel()">×</button>
                                        </div>
                                    </div>
                                    <div class="day-panel-content">
                                        <div class="loading">Loading clips...</div>
                                    </div>
                                </div>
                            `;
                            
                            try {{
                                const data = await this.loadDayClips(newDateStr);
                                this.renderDayPanel(newDateStr, data);
                            }} catch (e) {{
                                container.innerHTML = `
                                    <div class="day-panel">
                                        <div class="day-panel-header">
                                            <div class="day-panel-header-row">
                                                <div>
                                                    <h3 class="day-panel-title">${{this.formatDate(newDateStr)}}</h3>
                                                    <div class="day-panel-subtitle">Error</div>
                                                </div>
                                                <button class="day-panel-close" onclick="CalendarApp.closeDayPanel()">×</button>
                                            </div>
                                        </div>
                                        <div class="day-panel-content">
                                            <div style="color: #f44336; text-align: center; padding: 40px;">
                                                Failed to load clips
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }}
                        }},
                        
                        // ============== FILTERS (Phase 4) ==============
                        
                        // Toggle filters sidebar
                        toggleFilters() {{
                            const sidebar = document.getElementById('filtersSidebar');
                            const overlay = document.getElementById('filtersOverlay');
                            const isVisible = sidebar.classList.contains('visible');
                            
                            if (isVisible) {{
                                this.closeFilters();
                            }} else {{
                                this.openFilters();
                            }}
                        }},
                        
                        // Open filters sidebar
                        openFilters() {{
                            const sidebar = document.getElementById('filtersSidebar');
                            const overlay = document.getElementById('filtersOverlay');
                            
                            // Populate filter lists from calendar data
                            this.populateFilterLists();
                            
                            sidebar.classList.add('visible');
                            overlay.classList.add('visible');
                        }},
                        
                        // Close filters sidebar
                        closeFilters() {{
                            const sidebar = document.getElementById('filtersSidebar');
                            const overlay = document.getElementById('filtersOverlay');
                            
                            sidebar.classList.remove('visible');
                            overlay.classList.remove('visible');
                        }},
                        
                        // Populate camera and species filter lists from calendar data
                        populateFilterLists() {{
                            const data = this.state.calendarData;
                            if (!data) return;
                            
                            const cameras = data.filters?.cameras || [];
                            const species = data.filters?.species || [];
                            
                            // Count clips per camera and species
                            const cameraCounts = {{}};
                            const speciesCounts = {{}};
                            
                            // Iterate through all days to count
                            Object.values(data.years || {{}}).forEach(yearData => {{
                                Object.values(yearData.months || {{}}).forEach(monthData => {{
                                    Object.values(monthData.days || {{}}).forEach(dayData => {{
                                        // Count by camera
                                        (dayData.cameras || []).forEach(cam => {{
                                            cameraCounts[cam] = (cameraCounts[cam] || 0) + (dayData.count || 0) / (dayData.cameras?.length || 1);
                                        }});
                                        // Count by species
                                        (dayData.species || []).forEach(sp => {{
                                            speciesCounts[sp] = (speciesCounts[sp] || 0) + (dayData.count || 0) / (dayData.species?.length || 1);
                                        }});
                                    }});
                                }});
                            }});
                            
                            // Render camera list
                            const cameraList = document.getElementById('filterCamerasList');
                            if (cameraList) {{
                                cameraList.innerHTML = cameras.map(cam => `
                                    <label class="filter-item">
                                        <input type="checkbox" name="filter_camera" value="${{cam}}"
                                               ${{this.state.filters.cameras.includes(cam) ? 'checked' : ''}}
                                               onchange="CalendarApp.updateFilterPreview()">
                                        <span class="filter-item-label">${{cam}}</span>
                                        <span class="filter-item-count">${{Math.round(cameraCounts[cam] || 0)}}</span>
                                    </label>
                                `).join('') || '<div style="color: #666; font-size: 0.9em;">No cameras found</div>';
                            }}
                            
                            // Render species list
                            const speciesList = document.getElementById('filterSpeciesList');
                            if (speciesList) {{
                                speciesList.innerHTML = species.map(sp => `
                                    <label class="filter-item">
                                        <input type="checkbox" name="filter_species" value="${{sp}}"
                                               ${{this.state.filters.species.includes(sp) ? 'checked' : ''}}
                                               onchange="CalendarApp.updateFilterPreview()">
                                        <span class="filter-item-label">${{sp}}</span>
                                        <span class="filter-item-count">${{Math.round(speciesCounts[sp] || 0)}}</span>
                                    </label>
                                `).join('') || '<div style="color: #666; font-size: 0.9em;">No species found</div>';
                            }}
                            
                            // Set date inputs
                            const dateFrom = document.getElementById('filterDateFrom');
                            const dateTo = document.getElementById('filterDateTo');
                            if (dateFrom) dateFrom.value = this.state.filters.dateRange.start || '';
                            if (dateTo) dateTo.value = this.state.filters.dateRange.end || '';
                        }},
                        
                        // Update filter preview (called when checkboxes change)
                        updateFilterPreview() {{
                            // Could add a preview count here if desired
                        }},
                        
                        // Quick filter buttons
                        quickFilter(type) {{
                            const now = new Date();
                            const today = this.formatDateStr(now.getFullYear(), now.getMonth() + 1, now.getDate());
                            
                            // Clear existing date range
                            let start = null;
                            let end = null;
                            
                            switch (type) {{
                                case 'today':
                                    start = today;
                                    end = today;
                                    break;
                                case 'week':
                                    // Start of week (Sunday)
                                    const weekStart = new Date(now);
                                    weekStart.setDate(now.getDate() - now.getDay());
                                    start = this.formatDateStr(weekStart.getFullYear(), weekStart.getMonth() + 1, weekStart.getDate());
                                    end = today;
                                    break;
                                case 'month':
                                    start = this.formatDateStr(now.getFullYear(), now.getMonth() + 1, 1);
                                    end = today;
                                    break;
                                case 'all':
                                    start = null;
                                    end = null;
                                    break;
                            }}
                            
                            // Update date inputs
                            const dateFrom = document.getElementById('filterDateFrom');
                            const dateTo = document.getElementById('filterDateTo');
                            if (dateFrom) dateFrom.value = start || '';
                            if (dateTo) dateTo.value = end || '';
                            
                            // Highlight active quick filter button
                            document.querySelectorAll('.quick-filter-btn').forEach(btn => btn.classList.remove('active'));
                            event.target.classList.add('active');
                        }},
                        
                        // Format date as YYYY-MM-DD
                        formatDateStr(year, month, day) {{
                            return `${{year}}-${{String(month).padStart(2, '0')}}-${{String(day).padStart(2, '0')}}`;
                        }},
                        
                        // Clear all filters
                        clearFilters() {{
                            this.state.filters = {{
                                cameras: [],
                                species: [],
                                dateRange: {{ start: null, end: null }}
                            }};
                            
                            // Clear checkboxes
                            document.querySelectorAll('input[name="filter_camera"]').forEach(cb => cb.checked = false);
                            document.querySelectorAll('input[name="filter_species"]').forEach(cb => cb.checked = false);
                            
                            // Clear date inputs
                            const dateFrom = document.getElementById('filterDateFrom');
                            const dateTo = document.getElementById('filterDateTo');
                            if (dateFrom) dateFrom.value = '';
                            if (dateTo) dateTo.value = '';
                            
                            // Clear quick filter highlights
                            document.querySelectorAll('.quick-filter-btn').forEach(btn => btn.classList.remove('active'));
                            
                            // Update button state
                            this.updateFilterButtonState();
                            
                            // Re-render
                            this.render();
                            this.closeFilters();
                        }},
                        
                        // Apply filters
                        applyFilters() {{
                            // Collect selected cameras
                            const cameras = Array.from(document.querySelectorAll('input[name="filter_camera"]:checked'))
                                .map(cb => cb.value);
                            
                            // Collect selected species
                            const species = Array.from(document.querySelectorAll('input[name="filter_species"]:checked'))
                                .map(cb => cb.value);
                            
                            // Get date range
                            const dateFrom = document.getElementById('filterDateFrom')?.value || null;
                            const dateTo = document.getElementById('filterDateTo')?.value || null;
                            
                            // Update state
                            this.state.filters = {{
                                cameras: cameras,
                                species: species,
                                dateRange: {{ start: dateFrom, end: dateTo }}
                            }};
                            
                            // Update URL with filters
                            this.updateUrl();
                            
                            // Update button state
                            this.updateFilterButtonState();
                            
                            // Re-render calendar with filters
                            this.render();
                            
                            // Close sidebar
                            this.closeFilters();
                        }},
                        
                        // Update the filter button to show active state
                        updateFilterButtonState() {{
                            const btn = document.getElementById('filterToggleBtn');
                            if (!btn) return;
                            
                            const hasFilters = this.hasActiveFilters();
                            btn.classList.toggle('has-filters', hasFilters);
                            
                            const count = this.getActiveFilterCount();
                            const span = btn.querySelector('span');
                            if (span) {{
                                span.textContent = hasFilters ? `Filters (${{count}})` : 'Filters';
                            }}
                        }},
                        
                        // Check if any filters are active
                        hasActiveFilters() {{
                            const f = this.state.filters;
                            return f.cameras.length > 0 || 
                                   f.species.length > 0 || 
                                   f.dateRange.start || 
                                   f.dateRange.end;
                        }},
                        
                        // Get count of active filters
                        getActiveFilterCount() {{
                            const f = this.state.filters;
                            let count = f.cameras.length + f.species.length;
                            if (f.dateRange.start || f.dateRange.end) count++;
                            return count;
                        }},
                        
                        // Check if a day passes the current filters
                        dayPassesFilters(dayData) {{
                            if (!dayData) return false;
                            const f = this.state.filters;
                            
                            // Camera filter
                            if (f.cameras.length > 0) {{
                                const dayCameras = dayData.cameras || [];
                                if (!f.cameras.some(c => dayCameras.includes(c))) {{
                                    return false;
                                }}
                            }}
                            
                            // Species filter
                            if (f.species.length > 0) {{
                                const daySpecies = dayData.species || [];
                                if (!f.species.some(s => daySpecies.includes(s))) {{
                                    return false;
                                }}
                            }}
                            
                            return true;
                        }},
                        
                        // Check if a date is within the date range filter
                        dateInRange(dateStr) {{
                            const f = this.state.filters;
                            if (!f.dateRange.start && !f.dateRange.end) return true;
                            
                            if (f.dateRange.start && dateStr < f.dateRange.start) return false;
                            if (f.dateRange.end && dateStr > f.dateRange.end) return false;
                            
                            return true;
                        }},
                        
                        // Close the day panel
                        closeDayPanel() {{
                            document.getElementById('dayPanelOverlay').classList.remove('visible');
                            document.getElementById('dayPanelContainer').classList.remove('visible');
                            
                            // Clear selection
                            this.state.selectedDate = null;
                            this.state.dayClips = null;
                            this.updateUrl();
                            this.renderMonthView();
                        }}
                    }};
                    
                    // Initialize on page load
                    document.addEventListener('DOMContentLoaded', () => {{
                        CalendarApp.init();
                        
                        // Handle browser back/forward navigation
                        window.addEventListener('popstate', (e) => {{
                            CalendarApp.parseUrlParams();
                            CalendarApp.render();
                        }});
                        
                        // Set up keyboard navigation
                        document.addEventListener('keydown', (e) => {{
                            // Don't capture keys if focused on input
                            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                            
                            // Close picker and filters on Escape
                            if (e.key === 'Escape') {{
                                CalendarApp.closePicker();
                                CalendarApp.closeFilters();
                            }}
                            
                            // Don't capture if picker is open
                            if (document.getElementById('pickerDropdown').classList.contains('visible')) return;
                            
                            // Don't capture if filters sidebar is open
                            if (document.getElementById('filtersSidebar').classList.contains('visible')) return;
                            
                            // Handle day panel keyboard navigation
                            const dayPanelOpen = document.getElementById('dayPanelContainer').classList.contains('visible');
                            if (dayPanelOpen) {{
                                switch (e.key) {{
                                    case 'Escape':
                                        e.preventDefault();
                                        CalendarApp.closeDayPanel();
                                        break;
                                    case 'ArrowLeft':
                                        e.preventDefault();
                                        CalendarApp.navigateDay(-1);
                                        break;
                                    case 'ArrowRight':
                                        e.preventDefault();
                                        CalendarApp.navigateDay(1);
                                        break;
                                }}
                                return;
                            }}
                            
                            // Only apply in month view
                            if (CalendarApp.state.view !== 'month') return;
                            
                            switch (e.key) {{
                                case 'ArrowLeft':
                                    e.preventDefault();
                                    CalendarApp.prevMonth();
                                    break;
                                case 'ArrowRight':
                                    e.preventDefault();
                                    CalendarApp.nextMonth();
                                    break;
                                case 't':
                                case 'T':
                                    e.preventDefault();
                                    CalendarApp.goToToday();
                                    break;
                                case 'm':
                                case 'M':
                                    e.preventDefault();
                                    CalendarApp.togglePicker();
                                    break;
                                case 'f':
                                case 'F':
                                    e.preventDefault();
                                    CalendarApp.toggleFilters();
                                    break;
                            }}
                        }});
                        
                        // Set up touch swipe navigation for mobile
                        let touchStartX = 0;
                        let touchStartY = 0;
                        const calendarContainer = document.querySelector('.calendar-container');
                        
                        if (calendarContainer) {{
                            calendarContainer.addEventListener('touchstart', (e) => {{
                                touchStartX = e.touches[0].clientX;
                                touchStartY = e.touches[0].clientY;
                            }}, {{ passive: true }});
                            
                            calendarContainer.addEventListener('touchend', (e) => {{
                                if (CalendarApp.state.view !== 'month') return;
                                
                                const touchEndX = e.changedTouches[0].clientX;
                                const touchEndY = e.changedTouches[0].clientY;
                                
                                const deltaX = touchEndX - touchStartX;
                                const deltaY = touchEndY - touchStartY;
                                
                                // Only trigger if horizontal swipe is dominant and long enough
                                if (Math.abs(deltaX) > 80 && Math.abs(deltaX) > Math.abs(deltaY) * 2) {{
                                    if (deltaX > 0) {{
                                        CalendarApp.prevMonth();
                                    }} else {{
                                        CalendarApp.nextMonth();
                                    }}
                                }}
                            }}, {{ passive: true }});
                        }}
                        
                        // Set up touch swipe for day panel (left/right to navigate days)
                        const dayPanelContainer = document.getElementById('dayPanelContainer');
                        let dayPanelTouchStartX = 0;
                        let dayPanelTouchStartY = 0;
                        
                        if (dayPanelContainer) {{
                            dayPanelContainer.addEventListener('touchstart', (e) => {{
                                dayPanelTouchStartX = e.touches[0].clientX;
                                dayPanelTouchStartY = e.touches[0].clientY;
                            }}, {{ passive: true }});
                            
                            dayPanelContainer.addEventListener('touchend', (e) => {{
                                if (!CalendarApp.state.selectedDate) return;
                                
                                const touchEndX = e.changedTouches[0].clientX;
                                const touchEndY = e.changedTouches[0].clientY;
                                
                                const deltaX = touchEndX - dayPanelTouchStartX;
                                const deltaY = touchEndY - dayPanelTouchStartY;
                                
                                // Only trigger if horizontal swipe is dominant and long enough
                                if (Math.abs(deltaX) > 80 && Math.abs(deltaX) > Math.abs(deltaY) * 2) {{
                                    if (deltaX > 0) {{
                                        CalendarApp.navigateDay(-1); // Swipe right = previous day
                                    }} else {{
                                        CalendarApp.navigateDay(1); // Swipe left = next day
                                    }}
                                }}
                            }}, {{ passive: true }});
                        }}
                    }});
                    
                    // Existing functions for video modal and bulk actions
                    let currentClipPath = null;
                    
                    function playVideo(url, title, clipPath) {{
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
                        video.play().catch(() => {{}}); // Auto-play, ignore if blocked
                    }}
                    
                    function closeModal() {{
                        const modal = document.getElementById('videoModal');
                        const video = document.getElementById('modalVideo');
                        
                        video.pause();
                        video.src = '';
                        modal.classList.remove('active');
                        document.body.style.overflow = '';
                        currentClipPath = null;
                    }}
                    
                    function closeModalOnBackdrop(event) {{
                        if (event.target.id === 'videoModal') {{
                            closeModal();
                        }}
                    }}
                    
                    // Close modal on escape key
                    document.addEventListener('keydown', (e) => {{
                        if (e.key === 'Escape') {{
                            closeModal();
                            CalendarApp.closeDayPanel();
                        }}
                    }});
                    
                    async function deleteCurrentClip() {{
                        if (!currentClipPath) return;
                        await deleteClip(currentClipPath);
                    }}
                    
                    async function deleteClip(path) {{
                        if (!confirm('Are you sure you want to delete this clip?')) return;
                        try {{
                            const response = await fetch('/recordings?path=' + encodeURIComponent(path), {{ method: 'DELETE' }});
                            const text = await response.text();
                            if (response.ok) {{
                                closeModal();
                                location.reload();
                            }} else {{
                                alert('Error: ' + text);
                            }}
                        }} catch (e) {{
                            alert('Error deleting clip: ' + e);
                        }}
                    }}

                    function toggleAll(source) {{
                        const checkboxes = document.querySelectorAll('input[name="clip_select"]');
                        checkboxes.forEach(cb => cb.checked = source.checked);
                        updateBulkButton();
                    }}

                    function updateBulkButton() {{
                        const checked = document.querySelectorAll('input[name="clip_select"]:checked');
                        const btn = document.getElementById('bulkDeleteBtn');
                        const bulkActions = document.getElementById('bulkActions');
                        const count = checked.length;
                        
                        btn.disabled = count === 0;
                        btn.textContent = count > 0 ? `Delete (${{count}})` : 'Delete Selected';
                        
                        // Show/hide bulk actions bar
                        if (count > 0) {{
                            bulkActions.classList.add('visible');
                        }} else {{
                            bulkActions.classList.remove('visible');
                        }}
                    }}

                    async function bulkDelete() {{
                        const checked = document.querySelectorAll('input[name="clip_select"]:checked');
                        if (checked.length === 0) return;

                        if (!confirm(`Delete ${{checked.length}} clips?`)) return;

                        const paths = Array.from(checked).map(cb => cb.value);
                        
                        try {{
                            const response = await fetch('/recordings/bulk_delete', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{ paths: paths }})
                            }});
                            
                            const result = await response.json();
                            location.reload();
                        }} catch (e) {{
                            alert('Error: ' + e);
                        }}
                    }}
                    
                    // List view sorting
                    function sortRecordings(sortBy) {{
                        const list = document.getElementById('recordingsList');
                        const cards = Array.from(list.querySelectorAll('.recording-card'));
                        const headers = list.querySelectorAll('.date-group-header');
                        
                        // Remove existing headers (we'll re-add them if needed)
                        headers.forEach(h => h.remove());
                        
                        // Sort cards based on criteria
                        cards.sort((a, b) => {{
                            switch(sortBy) {{
                                case 'newest':
                                    return new Date(b.dataset.time) - new Date(a.dataset.time);
                                case 'oldest':
                                    return new Date(a.dataset.time) - new Date(b.dataset.time);
                                case 'species':
                                    return a.dataset.species.localeCompare(b.dataset.species);
                                case 'camera':
                                    return a.dataset.camera.localeCompare(b.dataset.camera);
                                default:
                                    return 0;
                            }}
                        }});
                        
                        // Re-append cards in sorted order
                        cards.forEach(card => list.appendChild(card));
                        
                        // Re-add date group headers if sorting by date
                        if (sortBy === 'newest' || sortBy === 'oldest') {{
                            let currentDate = null;
                            cards.forEach(card => {{
                                const cardDate = card.dataset.date;
                                if (cardDate !== currentDate) {{
                                    currentDate = cardDate;
                                    const dateObj = new Date(cardDate + 'T12:00:00');
                                    const dateStr = dateObj.toLocaleDateString('en-US', {{ 
                                        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
                                    }});
                                    const clipsOnDate = cards.filter(c => c.dataset.date === cardDate).length;
                                    
                                    const header = document.createElement('div');
                                    header.className = 'date-group-header';
                                    header.dataset.date = cardDate;
                                    header.innerHTML = `<span>📅 ${{dateStr}}</span><span class="date-group-count">${{clipsOnDate}} clip${{clipsOnDate === 1 ? '' : 's'}}</span>`;
                                    
                                    card.parentNode.insertBefore(header, card);
                                }}
                            }});
                        }}
                    }}
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
        
        # Get current detections for overlay
        detections = getattr(worker, 'latest_detections', []) or []
        
        def process_image(img, detections):
            height, width = img.shape[:2]
            
            # Draw bounding boxes and labels for current detections
            for det in detections:
                if det.bbox:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    
                    # Draw bounding box (green)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label with species name and confidence
                    common_name = get_common_name(det.species)
                    label = f"{common_name} {det.confidence*100:.0f}%"
                    
                    # Calculate text size for background rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw background rectangle for label
                    label_y = max(y1 - 10, text_height + 10)
                    cv2.rectangle(img, 
                                  (x1, label_y - text_height - 5), 
                                  (x1 + text_width + 10, label_y + 5), 
                                  (0, 255, 0), -1)
                    
                    # Draw label text (black on green background)
                    cv2.putText(img, label, (x1 + 5, label_y), 
                                font, font_scale, (0, 0, 0), thickness)
            
            # Resize for web display
            if width > 640:
                scale = 640 / width
                new_height = int(height * scale)
                img = cv2.resize(img, (640, new_height), interpolation=cv2.INTER_AREA)

            # Encode frame to JPEG with lower quality (70%)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            return cv2.imencode('.jpg', img, encode_param)

        success, buffer = await loop.run_in_executor(None, process_image, worker.latest_frame.copy(), detections)
        
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
        """Reprocess a clip to improve species classification.
        
        Accepts optional settings overrides in the request body:
        {
            "path": "camera/clip.mp4",
            "settings": {
                "sample_rate": 3,
                "confidence_threshold": 0.3,
                "generic_confidence": 0.5,
                "tracking_enabled": true,
                "merge_enabled": true,
                "same_species_merge_gap": 120,
                "hierarchical_merge_enabled": true,
                "hierarchical_merge_gap": 120,
                "min_specific_detections": 2
            }
        }
        """
        try:
            data = await request.json()
            clip_path = data.get('path')
            settings_override = data.get('settings', {})
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
        
        # Create postprocess detector (SpeciesNet for accurate species ID)
        # This uses the split-model architecture: MegaDetector for real-time, SpeciesNet for post-processing
        from .detector import create_postprocess_detector
        detector_cfg = self.runtime.general.detector if self.runtime else None
        if detector_cfg:
            detector = create_postprocess_detector(detector_cfg)
            LOGGER.info("Reprocess using %s detector (postprocess_backend)", detector.backend_name)
        else:
            # Fallback to worker's detector if no config available
            if not self.workers:
                return web.Response(status=500, text="No workers available")
            detector = next(iter(self.workers.values())).detector
            LOGGER.warning("No detector config, falling back to worker detector: %s", detector.backend_name)
        
        # Prevent duplicate processing - check if already in progress
        job_key = clip_path
        if job_key in self.reprocessing_jobs:
            existing = self.reprocessing_jobs[job_key]
            LOGGER.warning("Duplicate reprocess request for %s, already started at %s", 
                          clip_path, existing.get('started'))
            return web.json_response({
                'success': False,
                'error': f"Processing already in progress (started {existing.get('started')})"
            }, status=409)  # Conflict
        
        # Track this reprocessing job
        self.reprocessing_jobs[job_key] = {
            'started': datetime.now(tz=CENTRAL_TZ).isoformat(),
            'clip_name': full_path.stem,
            'camera': full_path.parent.name,
        }
        
        # Run reprocessing in thread pool
        loop = asyncio.get_running_loop()
        
        # Build ProcessingSettings from defaults + overrides
        from .postprocess import ProcessingSettings
        
        # Start with defaults from runtime config
        clip_cfg = self.runtime.general.clip if self.runtime else None
        default_settings = {
            'sample_rate': getattr(clip_cfg, 'sample_rate', 3) if clip_cfg else 3,
            'confidence_threshold': getattr(clip_cfg, 'post_analysis_confidence', 0.3) if clip_cfg else 0.3,
            'generic_confidence': getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5) if clip_cfg else 0.5,
            'tracking_enabled': getattr(clip_cfg, 'tracking_enabled', True) if clip_cfg else True,
            'merge_enabled': True,
            'same_species_merge_gap': getattr(clip_cfg, 'track_merge_gap', 120) if clip_cfg else 120,
            'spatial_merge_enabled': getattr(clip_cfg, 'spatial_merge_enabled', True) if clip_cfg else True,
            'spatial_merge_iou': getattr(clip_cfg, 'spatial_merge_iou', 0.3) if clip_cfg else 0.3,
            'spatial_merge_gap': 30,  # Default gap for spatial matching
            'hierarchical_merge_enabled': getattr(clip_cfg, 'hierarchical_merge_enabled', True) if clip_cfg else True,
            'hierarchical_merge_gap': 120,
            'min_specific_detections': 2,
            'lost_track_buffer': 120,
            'single_animal_mode': getattr(clip_cfg, 'single_animal_mode', False) if clip_cfg else False,
            'thumbnail_cropped': getattr(clip_cfg, 'thumbnail_cropped', True) if clip_cfg else True,
        }
        
        # Apply overrides from request (allow any key that ProcessingSettings supports)
        for key, value in settings_override.items():
            default_settings[key] = value
        
        settings = ProcessingSettings.from_dict(default_settings)
        
        LOGGER.info("Reprocess settings: spatial_merge_iou=%.2f, spatial_merge_enabled=%s, tracking=%s",
                   settings.spatial_merge_iou, settings.spatial_merge_enabled, settings.tracking_enabled)
        
        def do_reprocess():
            from .postprocess import ClipPostProcessor
            processor = ClipPostProcessor(
                detector=detector,
                storage_root=self.storage_root,
                settings=settings,
            )
            return processor.process_clip(
                full_path,
                update_filename=True,
                regenerate_thumbnails=True,
            )
        
        result = await loop.run_in_executor(None, do_reprocess)
        
        # Remove from active jobs
        self.reprocessing_jobs.pop(job_key, None)
        
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
                'tracks_detected': result.tracks_detected,
                'thumbnails_saved': len(result.thumbnails_saved),
                'thumbnail_paths': [str(p) for p in result.thumbnails_saved],
                'renamed': result.new_path is not None,
                'new_path': str(result.new_path.relative_to(self.storage_root / 'clips')) if result.new_path else None,
                'settings_used': result.settings_used.to_dict() if result.settings_used else None,
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
                        <button class="log-toggle" onclick="toggleLog('processing')">
                            <span class="arrow">▶</span>
                            <span>📋 Processing Details</span>
                        </button>
                        <div class="log-content" id="processingLogContent">
                            <div id="logData" class="log-no-data">Loading...</div>
                        </div>
                    </div>
                    <div class="processing-log ptz-log">
                        <button class="log-toggle" onclick="toggleLog('ptz')">
                            <span class="arrow">▶</span>
                            <span>🎥 PTZ Decisions</span>
                        </button>
                        <div class="log-content" id="ptzLogContent">
                            <div id="ptzLogData" class="log-no-data">Loading...</div>
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
                # Format timing info if available
                timing_html = ""
                if thumb.get('start_time') is not None:
                    start = thumb['start_time']
                    end = thumb['end_time']
                    duration = thumb.get('duration', end - start)
                    
                    # Format as mm:ss
                    start_str = f"{int(start // 60)}:{int(start % 60):02d}"
                    end_str = f"{int(end // 60)}:{int(end % 60):02d}"
                    
                    timing_html = f'<div class="thumbnail-timing" data-start-time="{start}" onclick="seekVideo({start})" title="Click to jump to this time">⏱️ {start_str} - {end_str} ({duration:.1f}s)</div>'
                    
                    # Add confidence if available
                    if thumb.get('confidence'):
                        conf_pct = thumb['confidence'] * 100
                        timing_html += f'<div class="thumbnail-confidence">{conf_pct:.0f}% confidence</div>'
                
                thumbnails_html += f"""
                        <div class="thumbnail-card">
                            <img src="{thumb['url']}" alt="{thumb['species']}" onclick="openImage('{thumb['url']}')">
                            <div class="thumbnail-label">{thumb['species']}</div>
                            {timing_html}
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
                        width: 100vw;
                        margin-left: calc(-50vw + 50%);
                        padding-left: 16px;
                        padding-right: 16px;
                        box-sizing: border-box;
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
                        padding: 10px 12px 4px;
                        font-size: 0.9em;
                        font-weight: 600;
                        color: #4CAF50;
                        text-align: center;
                    }}
                    .thumbnail-timing {{
                        padding: 2px 12px;
                        font-size: 0.8em;
                        color: #aaa;
                        text-align: center;
                        cursor: pointer;
                        transition: color 0.2s, background 0.2s;
                        border-radius: 4px;
                        margin: 2px 8px;
                    }}
                    .thumbnail-timing:hover {{
                        color: #4CAF50;
                        background: rgba(76, 175, 80, 0.1);
                    }}
                    .thumbnail-confidence {{
                        padding: 2px 12px 8px;
                        font-size: 0.75em;
                        color: #888;
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
                    .log-events-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 8px;
                    }}
                    .log-events-header h4 {{ margin: 0; }}
                    .copy-logs-btn {{
                        background: #2196F3;
                        color: white;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 0.85em;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }}
                    .copy-logs-btn:hover {{ background: #1976D2; }}
                    .copy-logs-btn.copied {{ background: #4CAF50; }}
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
                    
                    /* PTZ Log Styles */
                    .ptz-log {{ margin-top: 8px; }}
                    .ptz-event {{
                        padding: 8px;
                        margin: 4px 0;
                        border-radius: 4px;
                        font-family: 'SF Mono', Monaco, 'Consolas', monospace;
                        font-size: 0.85em;
                    }}
                    .ptz-event.mode_change {{ background: rgba(156, 39, 176, 0.15); border-left: 3px solid #9C27B0; }}
                    .ptz-event.move {{ background: rgba(33, 150, 243, 0.1); border-left: 3px solid #2196F3; }}
                    .ptz-event.deadzone {{ background: rgba(255, 193, 7, 0.1); border-left: 3px solid #FFC107; }}
                    .ptz-event.error {{ background: rgba(244, 67, 54, 0.15); border-left: 3px solid #F44336; }}
                    .ptz-event .timestamp {{ color: #888; font-size: 0.8em; }}
                    .ptz-event .event-type {{ color: #fff; font-weight: 600; text-transform: uppercase; font-size: 0.75em; }}
                    .ptz-event .details {{ color: #ccc; margin-top: 4px; }}
                    .ptz-summary {{
                        background: #1a1a1a;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 12px;
                    }}
                    .ptz-summary h4 {{ margin: 0 0 8px 0; color: #9C27B0; }}
                    .ptz-summary-row {{
                        display: flex;
                        justify-content: space-between;
                        padding: 4px 0;
                        border-bottom: 1px solid #333;
                    }}
                    .ptz-summary-row:last-child {{ border-bottom: none; }}
                    
                    /* Settings Panel Styles */
                    .settings-toggle {{
                        background: #607D8B !important;
                    }}
                    .settings-toggle:hover {{
                        background: #455A64 !important;
                    }}
                    .settings-panel {{
                        background: #252525;
                        border-radius: 12px;
                        padding: 20px;
                        margin: 16px 0;
                        display: none;
                        border: 1px solid #333;
                    }}
                    .settings-panel.visible {{
                        display: block;
                    }}
                    .settings-panel h3 {{
                        margin: 0 0 8px 0;
                        font-size: 1.1em;
                        color: #fff;
                    }}
                    .settings-description {{
                        color: #888;
                        font-size: 0.9em;
                        margin-bottom: 16px;
                    }}
                    .settings-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                        gap: 16px;
                    }}
                    .setting-group {{
                        display: flex;
                        flex-direction: column;
                        gap: 4px;
                    }}
                    .setting-group label {{
                        font-weight: 500;
                        color: #ddd;
                        font-size: 0.9em;
                    }}
                    .setting-group input[type="number"] {{
                        background: #1a1a1a;
                        border: 1px solid #444;
                        border-radius: 6px;
                        padding: 8px 12px;
                        color: #fff;
                        font-size: 1em;
                        width: 100%;
                        box-sizing: border-box;
                    }}
                    .setting-group input[type="number"]:focus {{
                        outline: none;
                        border-color: #4CAF50;
                    }}
                    .setting-help {{
                        font-size: 0.75em;
                        color: #666;
                    }}
                    .checkbox-group label {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        cursor: pointer;
                    }}
                    .checkbox-group input[type="checkbox"] {{
                        width: 18px;
                        height: 18px;
                        accent-color: #4CAF50;
                    }}
                    .settings-actions {{
                        display: flex;
                        gap: 12px;
                        margin-top: 16px;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    .reset-settings-btn {{
                        background: transparent;
                        border: 1px solid #666;
                        color: #888;
                        padding: 8px 16px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.9em;
                    }}
                    .reset-settings-btn:hover {{
                        border-color: #888;
                        color: #fff;
                    }}
                    .apply-settings-btn {{
                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                        border: none;
                        color: white;
                        padding: 10px 20px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 1em;
                        font-weight: 600;
                        transition: transform 0.1s, box-shadow 0.2s;
                    }}
                    .apply-settings-btn:hover {{
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
                    }}
                    
                    /* Mobile: 2x2 grid for action buttons */
                    @media (max-width: 480px) {{
                        .action-buttons {{
                            flex-wrap: wrap;
                        }}
                        .action-btn {{
                            flex: 1 1 calc(50% - 6px);
                            min-width: calc(50% - 6px);
                            padding: 12px 8px;
                            font-size: 0.85em;
                        }}
                        .action-btn svg {{
                            width: 16px;
                            height: 16px;
                        }}
                    }}
                    
                    @media (min-width: 768px) {{
                        body {{ padding: 24px; max-width: 900px; margin: 0 auto; }}
                        .thumbnail-gallery {{ grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); }}
                    }}
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/recordings" class="active">Recordings</a>
                    <a href="/live">Live</a>
                    <a href="/monitor">Monitor</a>
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
                    <video id="clipVideo" controls playsinline autoplay>
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
                    <button class="action-btn settings-toggle" onclick="toggleSettings()">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>
                        Settings
                    </button>
                    <button class="action-btn delete" onclick="deleteRecording()">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                        Delete
                    </button>
                </div>
                
                <!-- Processing Settings Panel -->
                <div class="settings-panel" id="settingsPanel">
                    <h3>⚙️ Processing Settings</h3>
                    <p class="settings-description">These settings are initialized from your global config. Changes are applied per-run, not saved permanently.</p>
                    
                    <div class="settings-grid">
                        <div class="setting-group">
                            <label for="sampleRate">Sample Rate</label>
                            <input type="number" id="sampleRate" value="{clip_info['global_settings']['sample_rate']}" min="1" max="30">
                            <span class="setting-help">Analyze every Nth frame (lower = more thorough)</span>
                        </div>
                        
                        <div class="setting-group">
                            <label for="confidenceThreshold">Confidence Threshold</label>
                            <input type="number" id="confidenceThreshold" value="{clip_info['global_settings']['confidence_threshold']:.2f}" min="0.1" max="1.0" step="0.05">
                            <span class="setting-help">Min confidence for species detection (0.3 recommended)</span>
                        </div>
                        
                        <div class="setting-group">
                            <label for="genericConfidence">Generic Confidence</label>
                            <input type="number" id="genericConfidence" value="{clip_info['global_settings']['generic_confidence']:.2f}" min="0.1" max="1.0" step="0.05">
                            <span class="setting-help">Min confidence for generic categories (animal, bird)</span>
                        </div>
                        
                        <div class="setting-group">
                            <label for="mergeGap">Track Merge Gap</label>
                            <input type="number" id="mergeGap" value="{clip_info['global_settings']['track_merge_gap']}" min="10" max="500">
                            <span class="setting-help">Max frame gap to merge same-species tracks</span>
                        </div>
                        
                        <div class="setting-group checkbox-group">
                            <label>
                                <input type="checkbox" id="trackingEnabled" {"checked" if clip_info['global_settings']['tracking_enabled'] else ""}>
                                Enable Object Tracking
                            </label>
                            <span class="setting-help">Track animals across frames for better accuracy</span>
                        </div>
                        
                        <div class="setting-group checkbox-group">
                            <label>
                                <input type="checkbox" id="spatialMerge" {"checked" if clip_info['global_settings']['spatial_merge_enabled'] else ""}>
                                Spatial Merge (Recommended)
                            </label>
                            <span class="setting-help">Merge tracks in same location - ignores species misclassifications</span>
                        </div>
                        
                        <div class="setting-group">
                            <label for="spatialIoU">Spatial Overlap (IoU)</label>
                            <input type="number" id="spatialIoU" value="{clip_info['global_settings']['spatial_merge_iou']:.2f}" min="0.1" max="0.9" step="0.05">
                            <span class="setting-help">Min bounding box overlap to merge (0.3 = 30%)</span>
                        </div>
                        
                        <div class="setting-group checkbox-group">
                            <label>
                                <input type="checkbox" id="hierarchicalMerge" {"checked" if clip_info['global_settings']['hierarchical_merge_enabled'] else ""}>
                                Hierarchical Merging
                            </label>
                            <span class="setting-help">Merge "animal" tracks into specific species tracks</span>
                        </div>
                        
                        <div class="setting-group checkbox-group">
                            <label>
                                <input type="checkbox" id="singleAnimalMode" {"checked" if clip_info['global_settings']['single_animal_mode'] else ""}>
                                Single Animal Mode
                            </label>
                            <span class="setting-help">Force merge ALL non-overlapping tracks into one</span>
                        </div>
                        
                        <div class="setting-group checkbox-group">
                            <label>
                                <input type="checkbox" id="thumbnailCropped" {"checked" if clip_info['global_settings']['thumbnail_cropped'] else ""}>
                                Cropped Thumbnails
                            </label>
                            <span class="setting-help">Zoom thumbnails to detection area (off = full frame with bbox)</span>
                        </div>
                    </div>
                    
                    <div class="settings-actions">
                        <button class="reset-settings-btn" onclick="resetSettings()">Reset to Defaults</button>
                        <button class="apply-settings-btn" onclick="reprocessRecording()">🔄 Apply & Reanalyze</button>
                    </div>
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
                    
                    function seekVideo(seconds) {{
                        const video = document.getElementById('clipVideo');
                        if (video) {{
                            video.currentTime = seconds;
                            video.play().catch(() => {{}});
                            // Scroll video into view
                            video.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
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
                        
                        // Gather settings from the panel
                        const settings = {{
                            sample_rate: parseInt(document.getElementById('sampleRate').value) || 3,
                            confidence_threshold: parseFloat(document.getElementById('confidenceThreshold').value) || 0.3,
                            generic_confidence: parseFloat(document.getElementById('genericConfidence').value) || 0.5,
                            same_species_merge_gap: parseInt(document.getElementById('mergeGap').value) || 120,
                            hierarchical_merge_gap: parseInt(document.getElementById('mergeGap').value) || 120,
                            spatial_merge_enabled: document.getElementById('spatialMerge').checked,
                            spatial_merge_iou: parseFloat(document.getElementById('spatialIoU').value) || 0.3,
                            spatial_merge_gap: 30,
                            tracking_enabled: document.getElementById('trackingEnabled').checked,
                            hierarchical_merge_enabled: document.getElementById('hierarchicalMerge').checked,
                            single_animal_mode: document.getElementById('singleAnimalMode').checked,
                            thumbnail_cropped: document.getElementById('thumbnailCropped').checked,
                            merge_enabled: true
                        }};
                        
                        try {{
                            const response = await fetch('/recordings/reprocess', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{
                                    path: '{clip_info['path']}',
                                    settings: settings
                                }})
                            }});
                            
                            const result = await response.json();
                            
                            if (result.success) {{
                                let msg = `Analysis complete!\\n\\n`;
                                msg += `Species: ${{result.new_species || 'Unknown'}} (${{(result.confidence * 100).toFixed(1)}}% confidence)\\n`;
                                msg += `Tracks detected: ${{result.tracks_detected || 1}} animal(s)\\n`;
                                msg += `Frames analyzed: ${{result.frames_analyzed}}/${{result.total_frames}}\\n`;
                                msg += `Raw detections: ${{result.raw_detections}} (filtered: ${{result.filtered_detections}})\\n`;
                                msg += `Valid species found: ${{result.species_found.join(', ') || 'None'}}\\n`;
                                msg += `Thumbnails saved: ${{result.thumbnails_saved}}`;
                                
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
                    
                    function toggleSettings() {{
                        document.getElementById('settingsPanel').classList.toggle('visible');
                    }}
                    
                    function resetSettings() {{
                        // Reset to global settings from config
                        document.getElementById('sampleRate').value = {clip_info['global_settings']['sample_rate']};
                        document.getElementById('confidenceThreshold').value = {clip_info['global_settings']['confidence_threshold']:.2f};
                        document.getElementById('genericConfidence').value = {clip_info['global_settings']['generic_confidence']:.2f};
                        document.getElementById('mergeGap').value = {clip_info['global_settings']['track_merge_gap']};
                        document.getElementById('trackingEnabled').checked = {'true' if clip_info['global_settings']['tracking_enabled'] else 'false'};
                        document.getElementById('spatialMerge').checked = {'true' if clip_info['global_settings']['spatial_merge_enabled'] else 'false'};
                        document.getElementById('spatialIoU').value = {clip_info['global_settings']['spatial_merge_iou']:.2f};
                        document.getElementById('hierarchicalMerge').checked = {'true' if clip_info['global_settings']['hierarchical_merge_enabled'] else 'false'};
                        document.getElementById('singleAnimalMode').checked = {'true' if clip_info['global_settings']['single_animal_mode'] else 'false'};
                        document.getElementById('thumbnailCropped').checked = {'true' if clip_info['global_settings']['thumbnail_cropped'] else 'false'};
                    }}
                    
                    // Processing log functions
                    let processingLogLoaded = false;
                    let ptzLogLoaded = false;
                    let cachedLogData = null;
                    
                    function toggleLog(logType) {{
                        const isProcessing = logType === 'processing';
                        const toggles = document.querySelectorAll('.log-toggle');
                        const toggle = isProcessing ? toggles[0] : toggles[1];
                        const content = document.getElementById(isProcessing ? 'processingLogContent' : 'ptzLogContent');
                        
                        toggle.classList.toggle('expanded');
                        content.classList.toggle('visible');
                        
                        // Load data on first expand
                        if (content.classList.contains('visible')) {{
                            if (cachedLogData) {{
                                // Use cached data
                                if (isProcessing && !processingLogLoaded) {{
                                    renderProcessingLog(cachedLogData);
                                    processingLogLoaded = true;
                                }} else if (!isProcessing && !ptzLogLoaded) {{
                                    renderPtzLog(cachedLogData);
                                    ptzLogLoaded = true;
                                }}
                            }} else {{
                                loadAllLogs(logType);
                            }}
                        }}
                    }}
                    
                    async function loadAllLogs(requestedType) {{
                        try {{
                            const response = await fetch('/recordings/log/{clip_info['path']}');
                            const result = await response.json();
                            
                            if (result.exists) {{
                                cachedLogData = result.data;
                                if (requestedType === 'processing') {{
                                    renderProcessingLog(result.data);
                                    processingLogLoaded = true;
                                }} else {{
                                    renderPtzLog(result.data);
                                    ptzLogLoaded = true;
                                }}
                            }} else {{
                                const target = requestedType === 'processing' ? 'logData' : 'ptzLogData';
                                document.getElementById(target).innerHTML = '<div class="log-no-data">' + result.message + '</div>';
                            }}
                        }} catch (e) {{
                            const target = requestedType === 'processing' ? 'logData' : 'ptzLogData';
                            document.getElementById(target).innerHTML = '<div class="log-no-data">Error loading log: ' + e + '</div>';
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
                            // Store log entries globally for copy function
                            window.currentLogEntries = data.log_entries;
                            
                            html += '<div class="log-events">';
                            html += '<div class="log-events-header">';
                            html += '<h4>📝 Processing Events (' + data.log_entries.length + ')</h4>';
                            html += '<button class="copy-logs-btn" onclick="copyLogs()">📋 Copy Logs</button>';
                            html += '</div>';
                            
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
                    
                    function copyLogs() {{
                        if (!window.currentLogEntries || window.currentLogEntries.length === 0) {{
                            return;
                        }}
                        
                        // Format logs as text
                        let logText = '';
                        for (const entry of window.currentLogEntries) {{
                            let line = 'Frame ' + entry.frame_idx;
                            if (entry.track_id !== null && entry.track_id !== undefined) {{
                                line += ' [Track ' + entry.track_id + ']';
                            }}
                            line += ' ' + entry.species + ' (' + (entry.confidence * 100).toFixed(1) + '%)';
                            if (entry.reason) {{
                                line += '\\n' + entry.reason;
                            }}
                            logText += line + '\\n';
                        }}
                        
                        // Copy to clipboard with fallback for non-HTTPS contexts
                        function showCopiedFeedback() {{
                            const btn = document.querySelector('.copy-logs-btn');
                            const originalText = btn.innerHTML;
                            btn.innerHTML = '✓ Copied!';
                            btn.classList.add('copied');
                            setTimeout(() => {{
                                btn.innerHTML = originalText;
                                btn.classList.remove('copied');
                            }}, 2000);
                        }}
                        
                        if (navigator.clipboard && navigator.clipboard.writeText) {{
                            navigator.clipboard.writeText(logText).then(() => {{
                                showCopiedFeedback();
                            }}).catch(err => {{
                                console.error('Failed to copy logs:', err);
                                fallbackCopy(logText);
                            }});
                        }} else {{
                            fallbackCopy(logText);
                        }}
                        
                        function fallbackCopy(text) {{
                            // Fallback for non-secure contexts (HTTP)
                            const textarea = document.createElement('textarea');
                            textarea.value = text;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-9999px';
                            document.body.appendChild(textarea);
                            textarea.select();
                            try {{
                                document.execCommand('copy');
                                showCopiedFeedback();
                            }} catch (err) {{
                                console.error('Fallback copy failed:', err);
                                alert('Failed to copy logs to clipboard');
                            }}
                            document.body.removeChild(textarea);
                        }}
                    }}
                    
                    // PTZ Log rendering
                    function renderPtzLog(data) {{
                        const ptzLogData = document.getElementById('ptzLogData');
                        
                        if (!data.ptz_decisions || data.ptz_decisions.length === 0) {{
                            ptzLogData.innerHTML = '<div class="log-no-data">No PTZ decisions recorded for this clip. PTZ tracking may not have been active.</div>';
                            return;
                        }}
                        
                        const decisions = data.ptz_decisions;
                        window.currentPtzDecisions = decisions;
                        
                        let html = '';
                        
                        // PTZ Summary
                        const modeChanges = decisions.filter(d => d.event === 'mode_change').length;
                        const moves = decisions.filter(d => d.event === 'move').length;
                        const deadzones = decisions.filter(d => d.event === 'deadzone').length;
                        const errors = decisions.filter(d => d.event === 'error').length;
                        
                        html += '<div class="ptz-summary">';
                        html += '<h4>🎥 PTZ Summary</h4>';
                        html += '<div class="ptz-summary-row"><span>Total Decisions</span><span>' + decisions.length + '</span></div>';
                        html += '<div class="ptz-summary-row"><span>Mode Changes</span><span>' + modeChanges + '</span></div>';
                        html += '<div class="ptz-summary-row"><span>Move Commands</span><span>' + moves + '</span></div>';
                        html += '<div class="ptz-summary-row"><span>In Deadzone</span><span>' + deadzones + '</span></div>';
                        if (errors > 0) {{
                            html += '<div class="ptz-summary-row"><span style="color: #F44336;">Errors</span><span style="color: #F44336;">' + errors + '</span></div>';
                        }}
                        html += '</div>';
                        
                        // PTZ Events
                        html += '<div class="log-events">';
                        html += '<div class="log-events-header">';
                        html += '<h4>📝 PTZ Events (' + decisions.length + ')</h4>';
                        html += '<button class="copy-logs-btn" onclick="copyPtzLogs()">📋 Copy Logs</button>';
                        html += '</div>';
                        
                        for (const entry of decisions) {{
                            const eventClass = entry.event || 'move';
                            const time = new Date(entry.timestamp * 1000).toLocaleTimeString();
                            
                            html += '<div class="ptz-event ' + eventClass + '">';
                            html += '<span class="timestamp">' + time + '</span> ';
                            html += '<span class="event-type">' + entry.event + '</span> ';
                            html += '<span class="mode">[' + entry.mode + ']</span>';
                            
                            // Format details based on event type
                            if (entry.details) {{
                                html += '<div class="details">';
                                if (entry.event === 'move') {{
                                    const d = entry.details;
                                    html += 'Species: ' + (d.species || 'unknown') + ' (' + ((d.confidence || 0) * 100).toFixed(1) + '%) ';
                                    if (d.velocity) {{
                                        html += '| Vel: pan=' + d.velocity.pan.toFixed(2) + ', tilt=' + d.velocity.tilt.toFixed(2);
                                    }}
                                    if (d.offset) {{
                                        html += ' | Offset: ' + (d.offset.magnitude * 100).toFixed(1) + '%';
                                    }}
                                    if (d.fill_pct) {{
                                        html += ' | Fill: ' + d.fill_pct.toFixed(0) + '%';
                                    }}
                                }} else if (entry.event === 'mode_change') {{
                                    html += entry.details.from + ' → ' + entry.details.to;
                                    if (entry.details.reason) {{
                                        html += ' (' + entry.details.reason + ')';
                                    }}
                                    if (entry.details.trigger) {{
                                        html += ' | Trigger: ' + entry.details.trigger;
                                    }}
                                }} else if (entry.event === 'deadzone') {{
                                    html += 'In center zone (offset: ' + (entry.details.offset_magnitude * 100).toFixed(1) + '% < ' + (entry.details.threshold * 100).toFixed(0) + '%)';
                                }} else if (entry.event === 'error') {{
                                    html += '<span style="color: #F44336;">' + (entry.details.error || 'Unknown error') + '</span>';
                                }} else {{
                                    html += JSON.stringify(entry.details);
                                }}
                                html += '</div>';
                            }}
                            html += '</div>';
                        }}
                        html += '</div>';
                        
                        ptzLogData.innerHTML = html;
                    }}
                    
                    function copyPtzLogs() {{
                        if (!window.currentPtzDecisions || window.currentPtzDecisions.length === 0) {{
                            return;
                        }}
                        
                        let logText = 'PTZ Decision Log\\n================\\n\\n';
                        for (const entry of window.currentPtzDecisions) {{
                            const time = new Date(entry.timestamp * 1000).toLocaleTimeString();
                            let line = time + ' [' + entry.mode + '] ' + entry.event.toUpperCase();
                            
                            if (entry.details) {{
                                if (entry.event === 'move') {{
                                    const d = entry.details;
                                    line += ': ' + (d.species || 'unknown');
                                    if (d.velocity) {{
                                        line += ' vel=(pan=' + d.velocity.pan.toFixed(2) + ', tilt=' + d.velocity.tilt.toFixed(2) + ')';
                                    }}
                                }} else if (entry.event === 'mode_change') {{
                                    line += ': ' + entry.details.from + ' -> ' + entry.details.to;
                                }} else {{
                                    line += ': ' + JSON.stringify(entry.details);
                                }}
                            }}
                            logText += line + '\\n';
                        }}
                        
                        // Use same copy logic as processing logs
                        function showCopiedFeedback() {{
                            const btns = document.querySelectorAll('.copy-logs-btn');
                            const btn = btns[btns.length - 1]; // Last button is PTZ copy
                            const originalText = btn.innerHTML;
                            btn.innerHTML = '✓ Copied!';
                            btn.classList.add('copied');
                            setTimeout(() => {{
                                btn.innerHTML = originalText;
                                btn.classList.remove('copied');
                            }}, 2000);
                        }}
                        
                        if (navigator.clipboard && navigator.clipboard.writeText) {{
                            navigator.clipboard.writeText(logText).then(() => {{
                                showCopiedFeedback();
                            }}).catch(err => {{
                                fallbackCopyPtz(logText, showCopiedFeedback);
                            }});
                        }} else {{
                            fallbackCopyPtz(logText, showCopiedFeedback);
                        }}
                        
                        function fallbackCopyPtz(text, callback) {{
                            const textarea = document.createElement('textarea');
                            textarea.value = text;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-9999px';
                            document.body.appendChild(textarea);
                            textarea.select();
                            try {{
                                document.execCommand('copy');
                                callback();
                            }} catch (err) {{
                                alert('Failed to copy PTZ logs');
                            }}
                            document.body.removeChild(textarea);
                        }}
                    }}
                </script>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    def _get_clip_detail(self, rel_path: str) -> dict | None:
        """Get detailed information about a specific clip."""
        import json
        
        clips_dir = self.storage_root / 'clips'
        clip_path = clips_dir / rel_path
        
        if not clip_path.exists() or not clip_path.is_file():
            return None
        
        stat = clip_path.stat()
        species = self._parse_species_from_filename(clip_path.name)
        thumbnails = self._get_thumbnails_for_clip(clip_path)
        
        # Try to load processing log for track timing info
        log_path = clip_path.with_suffix('.log.json')
        track_info = {}
        video_fps = 15.0  # Default
        
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Get video FPS for time calculation
                if log_data.get('video', {}).get('fps'):
                    video_fps = log_data['video']['fps']
                
                # Build track info - now keyed by track INDEX (order) not species
                # This preserves individual track timing even for same-species tracks
                tracking_summary = log_data.get('tracking_summary', {})
                tracks_by_index = {}  # Index -> track data
                tracks_by_species = {}  # Fallback for old-format thumbnails
                
                if tracking_summary and tracking_summary.get('tracks'):
                    # Sort tracks by first_frame to match thumbnail generation order
                    sorted_tracks = sorted(
                        tracking_summary['tracks'],
                        key=lambda t: t.get('first_frame', 0)
                    )
                    
                    for track_idx, track in enumerate(sorted_tracks):
                        species_name = track.get('best_species', '')
                        if species_name:
                            # Convert frames to timestamps
                            first_frame = track.get('first_frame', 0)
                            last_frame = track.get('last_frame', 0)
                            start_sec = first_frame / video_fps
                            end_sec = last_frame / video_fps
                            
                            common_name = get_common_name(species_name)
                            
                            # Store by track index for new-format thumbnails
                            tracks_by_index[track_idx] = {
                                'track_id': track.get('track_id'),
                                'start_time': start_sec,
                                'end_time': end_sec,
                                'duration': end_sec - start_sec,
                                'confidence': track.get('best_confidence', 0),
                                'species': common_name,
                            }
                            
                            # Also store by species for old-format thumbnail fallback
                            # But DON'T merge - keep the first one (highest confidence usually)
                            if common_name not in tracks_by_species:
                                tracks_by_species[common_name] = tracks_by_index[track_idx]
                                
            except Exception as e:
                LOGGER.warning("Failed to load processing log: %s", e)
        
        # Enrich thumbnails with track timing
        for thumb in thumbnails:
            # First try to match by track_index (new format thumbnails)
            track_idx = thumb.get('track_index')
            if track_idx is not None and track_idx in tracks_by_index:
                ti = tracks_by_index[track_idx]
                thumb['start_time'] = ti['start_time']
                thumb['end_time'] = ti['end_time']
                thumb['duration'] = ti['duration']
                thumb['track_id'] = ti['track_id']
                thumb['confidence'] = ti['confidence']
            else:
                # Fallback: match by species name (old format thumbnails)
                thumb_species = thumb.get('species', '')
                if thumb_species in tracks_by_species:
                    ti = tracks_by_species[thumb_species]
                    thumb['start_time'] = ti['start_time']
                    thumb['end_time'] = ti['end_time']
                    thumb['duration'] = ti['duration']
                    thumb['track_id'] = ti['track_id']
                    thumb['confidence'] = ti['confidence']
        
        # Determine camera from path
        parts = rel_path.split('/')
        camera = parts[0] if len(parts) > 1 else 'unknown'
        
        # Get global processing settings from runtime config
        clip_cfg = self.runtime.general.clip if self.runtime else None
        global_settings = {
            'sample_rate': getattr(clip_cfg, 'sample_rate', 3) if clip_cfg else 3,
            'confidence_threshold': getattr(clip_cfg, 'post_analysis_confidence', 0.3) if clip_cfg else 0.3,
            'generic_confidence': getattr(clip_cfg, 'post_analysis_generic_confidence', 0.5) if clip_cfg else 0.5,
            'tracking_enabled': getattr(clip_cfg, 'tracking_enabled', True) if clip_cfg else True,
            'track_merge_gap': getattr(clip_cfg, 'track_merge_gap', 120) if clip_cfg else 120,
            'spatial_merge_enabled': getattr(clip_cfg, 'spatial_merge_enabled', True) if clip_cfg else True,
            'spatial_merge_iou': getattr(clip_cfg, 'spatial_merge_iou', 0.3) if clip_cfg else 0.3,
            'hierarchical_merge_enabled': getattr(clip_cfg, 'hierarchical_merge_enabled', True) if clip_cfg else True,
            'single_animal_mode': getattr(clip_cfg, 'single_animal_mode', False) if clip_cfg else False,
            'thumbnail_cropped': getattr(clip_cfg, 'thumbnail_cropped', True) if clip_cfg else True,
        }
        
        return {
            'path': rel_path,
            'filename': clip_path.name,
            'camera': camera,
            'species': species,
            'time': datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ),
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'thumbnails': thumbnails,
            'fps': video_fps,
            'global_settings': global_settings,
        }

    async def handle_get_monitor_data(self, request):
        """Get real-time pipeline monitoring data as JSON."""
        import psutil
        
        cameras = []
        for camera_id, worker in self.workers.items():
            # Get camera status
            camera_data = {
                'id': camera_id,
                'name': worker.camera.name,
                'location': worker.camera.location,
                'status': 'connected' if worker.latest_frame is not None else 'disconnected',
                'buffer_frames': worker.clip_buffer.frame_count,
                'buffer_max_frames': worker.clip_buffer.max_frames,
                'buffer_seconds': round(worker.clip_buffer.duration, 1),
                'buffer_max_seconds': worker.clip_buffer.max_seconds,
                'event_active': worker.event_state is not None,
                'event_species': list(worker.event_state.species) if worker.event_state else [],
                'event_duration': round(worker.event_state.duration, 1) if worker.event_state else 0,
                'event_confidence': round(worker.event_state.max_confidence, 3) if worker.event_state else 0,
                'tracking_enabled': worker.tracking_enabled,
                'tracks_active': len(worker.event_state.tracker.tracks) if worker.event_state and worker.event_state.tracker else 0,
            }
            cameras.append(camera_data)
        
        # System stats
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.storage_root))
            
            system = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 1),
                'memory_total_gb': round(memory.total / (1024**3), 1),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 1),
                'disk_total_gb': round(disk.total / (1024**3), 1),
            }
        except Exception:
            system = {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'disk_percent': 0,
                'disk_used_gb': 0,
                'disk_total_gb': 0,
            }
        
        # GPU stats (NVIDIA)
        gpu = {
            'available': False,
            'name': None,
            'utilization': 0,
            'memory_percent': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 0,
            'temperature': 0,
            'power_draw': 0,
            'power_limit': 0,
        }
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu['available'] = True
            gpu['name'] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu['name'], bytes):
                gpu['name'] = gpu['name'].decode('utf-8')
            
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu['utilization'] = util.gpu
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu['memory_used_mb'] = round(mem_info.used / (1024**2))
            gpu['memory_total_mb'] = round(mem_info.total / (1024**2))
            gpu['memory_percent'] = round((mem_info.used / mem_info.total) * 100, 1)
            
            try:
                gpu['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            try:
                gpu['power_draw'] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)  # mW to W
                gpu['power_limit'] = round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000, 1)
            except:
                pass
            
            pynvml.nvmlShutdown()
        except ImportError:
            pass  # pynvml not installed
        except Exception as e:
            LOGGER.debug("GPU monitoring unavailable: %s", e)
        
        # Detector info
        detector_info = {
            'backend': 'unknown',
            'country': None,
        }
        if self.workers:
            worker = next(iter(self.workers.values()))
            detector_info['backend'] = worker.detector.backend_name
            if hasattr(worker.detector, 'country'):
                detector_info['country'] = worker.detector.country
        
        # Recent clips (last 5)
        recent_clips = []
        clips_dir = self.storage_root / 'clips'
        if clips_dir.exists():
            all_clips = []
            for camera_dir in clips_dir.iterdir():
                if camera_dir.is_dir():
                    for clip in camera_dir.glob('*.mp4'):
                        all_clips.append((clip, clip.stat().st_mtime))
            all_clips.sort(key=lambda x: x[1], reverse=True)
            for clip, mtime in all_clips[:5]:
                species = self._parse_species_from_filename(clip.name)
                recent_clips.append({
                    'path': str(clip.relative_to(clips_dir)),
                    'species': species,
                    'time': datetime.fromtimestamp(mtime, tz=CENTRAL_TZ).strftime('%H:%M:%S'),
                    'camera': clip.parent.name,
                })
        
        # Active reprocessing jobs
        reprocessing = list(self.reprocessing_jobs.values())
        
        return web.json_response({
            'timestamp': datetime.now(tz=CENTRAL_TZ).isoformat(),
            'cameras': cameras,
            'system': system,
            'gpu': gpu,
            'detector': detector_info,
            'recent_clips': recent_clips,
            'reprocessing_jobs': reprocessing,
        })

    async def handle_get_logs(self, request):
        """Get recent logs from journalctl or log files."""
        import subprocess
        import re as regex
        
        # Get query params
        camera_id = request.query.get('camera', None)
        minutes = int(request.query.get('minutes', 30))
        level = request.query.get('level', 'all')  # all, error, warning
        log_type = request.query.get('type', 'all')  # all, no-http, detection, tracking, events, clips, errors
        
        # Server-side filter patterns (match client-side)
        LOG_TYPE_FILTERS = {
            'all': None,
            'no-http': {
                'exclude': [r'GET /', r'POST /', r'DELETE /', r'PUT /', r'HTTP/\d', r'\d{3} \d+ bytes', r'aiohttp']
            },
            'detection': {
                'include': [r'detect', r'species', r'confidence', r'infer', r'YOLO', r'SpeciesNet']
            },
            'tracking': {
                'include': [r'track', r'ByteTrack', r'lost_buffer', r'merge']
            },
            'events': {
                'include': [r'event', r'started tracking', r'closed', r'clip at']
            },
            'clips': {
                'include': [r'clip', r'recording', r'write_clip', r'storage', r'\.mp4']
            },
            'errors': {
                'include': [r'error', r'warning', r'failed', r'exception', r'traceback']
            }
        }
        
        def matches_filter(message, filter_type):
            """Check if message matches the filter criteria."""
            filter_config = LOG_TYPE_FILTERS.get(filter_type)
            if not filter_config:
                return True
            
            if 'exclude' in filter_config:
                for pattern in filter_config['exclude']:
                    if regex.search(pattern, message, regex.IGNORECASE):
                        return False
                return True
            
            if 'include' in filter_config:
                for pattern in filter_config['include']:
                    if regex.search(pattern, message, regex.IGNORECASE):
                        return True
                return False
            
            return True
        
        logs = []
        source = 'none'
        error_msg = None
        skipped_count = 0
        
        # Try journalctl first (for systemd systems)
        try:
            # Build journalctl command - simpler approach without unit filtering
            # to capture all system logs, then filter in Python
            # Fetch more logs when filtering to ensure we get enough matches
            fetch_limit = 2000 if log_type != 'all' else 500
            cmd = [
                'journalctl',
                '--since', f'{minutes} minutes ago',
                '--no-pager',
                '-o', 'json',
                '-n', str(fetch_limit),
            ]
            
            # Add unit filter if specific camera requested
            if camera_id:
                cmd.extend(['-u', f'detector@{camera_id}.service'])
            # Otherwise get all detector units using glob (works on Ubuntu 24.04)
            else:
                cmd.extend(['-u', 'detector@*.service'])
            
            # Add priority filter
            if level == 'error':
                cmd.extend(['-p', 'err'])
            elif level == 'warning':
                cmd.extend(['-p', 'warning'])
            
            LOGGER.debug("Running journalctl: %s", ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            LOGGER.debug("journalctl returned: code=%d, stdout_len=%d, stderr=%s", 
                        result.returncode, len(result.stdout), result.stderr[:200] if result.stderr else '')
            
            if result.returncode == 0 and result.stdout.strip():
                source = 'journalctl'
                import json
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            entry = json.loads(line)
                            # Parse journalctl JSON format
                            timestamp = entry.get('__REALTIME_TIMESTAMP', '')
                            if timestamp:
                                # Convert microseconds to datetime
                                ts = int(timestamp) / 1_000_000
                                time_str = datetime.fromtimestamp(ts, tz=CENTRAL_TZ).strftime('%H:%M:%S')
                            else:
                                time_str = '--:--:--'
                            
                            message = entry.get('MESSAGE', '')
                            priority = int(entry.get('PRIORITY', 6))
                            unit = entry.get('_SYSTEMD_UNIT', '')
                            
                            # Map priority to level
                            if priority <= 3:
                                log_level = 'error'
                            elif priority <= 4:
                                log_level = 'warning'
                            else:
                                log_level = 'info'
                            
                            # Extract camera from unit name
                            cam = ''
                            if 'detector@' in unit:
                                cam = unit.replace('detector@', '').replace('.service', '')
                            
                            # Apply server-side type filter
                            if matches_filter(message, log_type):
                                logs.append({
                                    'time': time_str,
                                    'level': log_level,
                                    'camera': cam,
                                    'message': message,
                                })
                            else:
                                skipped_count += 1
                        except json.JSONDecodeError:
                            continue
            elif result.returncode != 0:
                error_msg = result.stderr[:200] if result.stderr else f'Exit code {result.returncode}'
                LOGGER.warning("journalctl failed: %s", error_msg)
        except FileNotFoundError:
            error_msg = 'journalctl not found'
        except subprocess.TimeoutExpired:
            error_msg = 'journalctl timed out'
        except Exception as e:
            error_msg = str(e)
            LOGGER.debug("journalctl failed: %s", e)
        
        # Also read from log files and merge (not just fallback)
        log_files_found = 0
        if self.logs_root and self.logs_root.exists():
            cutoff = datetime.now(tz=CENTRAL_TZ) - timedelta(minutes=minutes)
            
            # Look for log files
            log_patterns = ['*.log', 'detector*.log', 'animaltracker*.log']
            for pattern in log_patterns:
                for log_file in self.logs_root.glob(pattern):
                    log_files_found += 1
                    try:
                        with open(log_file, 'r') as f:
                            # Read last 500 lines
                            lines = f.readlines()[-500:]
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                # Parse common log formats
                                log_level = 'info'
                                if 'ERROR' in line or 'error' in line.lower():
                                    log_level = 'error'
                                elif 'WARNING' in line or 'warning' in line.lower():
                                    log_level = 'warning'
                                
                                # Filter by level
                                if level == 'error' and log_level != 'error':
                                    continue
                                if level == 'warning' and log_level not in ('error', 'warning'):
                                    continue
                                
                                # Try to extract timestamp
                                time_str = '--:--:--'
                                time_match = regex.search(r'(\\d{2}:\\d{2}:\\d{2})', line)
                                if time_match:
                                    time_str = time_match.group(1)
                                
                                # Apply server-side type filter
                                if matches_filter(line, log_type):
                                    logs.append({
                                        'time': time_str,
                                        'level': log_level,
                                        'camera': '',
                                        'message': line[:500],  # Truncate long lines
                                    })
                                else:
                                    skipped_count += 1
                    except Exception:
                        continue
            
            # Update source based on what we found
            if log_files_found > 0:
                if source == 'journalctl':
                    source = 'journalctl+logfile'
                else:
                    source = 'logfile'
        
        # Sort by time descending and limit
        logs = logs[-200:]  # Keep last 200
        logs.reverse()  # Most recent first
        
        # Determine final source description
        if source == 'none' and len(logs) > 0:
            source = 'unknown'  # We have logs but don't know where from
        
        response_data = {
            'source': source,
            'camera': camera_id,
            'minutes': minutes,
            'level': level,
            'type': log_type,
            'count': len(logs),
            'skipped': skipped_count,
            'logs': logs,
        }
        if error_msg and source in ('none', 'unknown'):
            response_data['error'] = error_msg
        
        return web.json_response(response_data)

    async def handle_monitor_page(self, request):
        """Render the pipeline monitor page HTML."""
        html = """
        <html>
            <head>
                <title>Monitor - Animal Tracker</title>
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
                        width: 100vw;
                        margin-left: calc(-50vw + 50%);
                        padding-left: 16px;
                        padding-right: 16px;
                        box-sizing: border-box;
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
                    .nav a:hover { background: #444; }
                    .nav a.active { background: #4CAF50; }
                    
                    h1 { margin: 0 0 20px 0; font-size: 1.5em; }
                    
                    .section {
                        background: #2a2a2a;
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 16px;
                    }
                    .section h2 {
                        margin: 0 0 12px 0;
                        font-size: 1.1em;
                        color: #4CAF50;
                    }
                    
                    /* System stats */
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 12px;
                    }
                    .stat-card {
                        background: #1a1a1a;
                        border-radius: 8px;
                        padding: 12px;
                        text-align: center;
                    }
                    .stat-value {
                        font-size: 1.5em;
                        font-weight: bold;
                        color: #fff;
                    }
                    .stat-label {
                        font-size: 0.8em;
                        color: #888;
                        margin-top: 4px;
                    }
                    .stat-bar {
                        height: 4px;
                        background: #333;
                        border-radius: 2px;
                        margin-top: 8px;
                        overflow: hidden;
                    }
                    .stat-bar-fill {
                        height: 100%;
                        background: #4CAF50;
                        transition: width 0.3s;
                    }
                    .stat-bar-fill.warning { background: #FF9800; }
                    .stat-bar-fill.danger { background: #F44336; }
                    
                    /* Camera cards */
                    .camera-grid {
                        display: grid;
                        gap: 12px;
                    }
                    .camera-card {
                        background: #1a1a1a;
                        border-radius: 8px;
                        padding: 12px;
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                    }
                    .camera-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    .camera-name {
                        font-weight: 600;
                        font-size: 1em;
                    }
                    .camera-location {
                        color: #888;
                        font-size: 0.85em;
                    }
                    .status-badge {
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 0.75em;
                        font-weight: 600;
                        text-transform: uppercase;
                    }
                    .status-connected { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
                    .status-disconnected { background: rgba(244, 67, 54, 0.2); color: #F44336; }
                    .status-detecting { background: rgba(255, 152, 0, 0.2); color: #FF9800; animation: pulse 1s infinite; }
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.6; }
                    }
                    
                    .camera-stats {
                        display: flex;
                        gap: 16px;
                        font-size: 0.85em;
                        color: #aaa;
                    }
                    .camera-stats span {
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }
                    
                    .event-info {
                        background: rgba(255, 152, 0, 0.1);
                        border: 1px solid rgba(255, 152, 0, 0.3);
                        border-radius: 6px;
                        padding: 8px;
                        font-size: 0.85em;
                    }
                    .event-species {
                        color: #FF9800;
                        font-weight: 600;
                    }
                    
                    /* Recent clips */
                    .clip-list {
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                    }
                    .clip-item {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 8px 12px;
                        background: #1a1a1a;
                        border-radius: 6px;
                        text-decoration: none;
                        color: inherit;
                        transition: background 0.2s;
                    }
                    .clip-item:hover { background: #222; }
                    .clip-species { color: #4CAF50; font-weight: 500; }
                    .clip-meta { color: #888; font-size: 0.85em; }
                    
                    /* Detector info */
                    .detector-info {
                        display: flex;
                        gap: 16px;
                        font-size: 0.9em;
                    }
                    .detector-badge {
                        background: #333;
                        padding: 4px 12px;
                        border-radius: 6px;
                    }
                    
                    /* GPU section */
                    .gpu-header {
                        font-size: 0.9em;
                        color: #76B900;
                        font-weight: 600;
                        margin-bottom: 12px;
                    }
                    .gpu-details {
                        display: flex;
                        gap: 16px;
                        margin-top: 12px;
                        font-size: 0.85em;
                        color: #888;
                    }
                    .gpu-details span {
                        background: #1a1a1a;
                        padding: 4px 10px;
                        border-radius: 4px;
                    }
                    
                    /* Log viewer */
                    .log-controls {
                        display: flex;
                        gap: 8px;
                        margin-bottom: 12px;
                        flex-wrap: wrap;
                    }
                    .log-controls select {
                        background: #1a1a1a;
                        color: #fff;
                        border: 1px solid #444;
                        padding: 6px 10px;
                        border-radius: 6px;
                        font-size: 0.85em;
                    }
                    .refresh-logs-btn {
                        background: #333;
                        color: #fff;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.85em;
                    }
                    .refresh-logs-btn:hover { background: #444; }
                    /* Reprocessing jobs */
                    .reprocessing-list {
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                    }
                    .reprocessing-item {
                        background: #2a2a2a;
                        border-radius: 8px;
                        padding: 12px;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    }
                    .reprocessing-spinner {
                        width: 20px;
                        height: 20px;
                        border: 2px solid #333;
                        border-top-color: #4CAF50;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    }
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                    .reprocessing-info {
                        flex: 1;
                    }
                    .reprocessing-clip {
                        font-weight: 500;
                        color: #fff;
                    }
                    .reprocessing-meta {
                        font-size: 0.85em;
                        color: #888;
                    }
                    .log-info {
                        font-size: 0.8em;
                        color: #888;
                        margin-bottom: 8px;
                    }
                    .log-container {
                        background: #111;
                        border-radius: 8px;
                        padding: 12px;
                        max-height: 400px;
                        overflow-y: auto;
                        font-family: 'SF Mono', Monaco, 'Consolas', monospace;
                        font-size: 0.8em;
                    }
                    .log-entry {
                        padding: 4px 0;
                        border-bottom: 1px solid #222;
                        display: flex;
                        gap: 8px;
                        flex-wrap: wrap;
                    }
                    .log-entry:last-child { border-bottom: none; }
                    .log-time { color: #666; min-width: 70px; }
                    .log-camera { color: #2196F3; min-width: 60px; }
                    .log-level { min-width: 50px; font-weight: 600; }
                    .log-level.error { color: #F44336; }
                    .log-level.warning { color: #FF9800; }
                    .log-level.info { color: #4CAF50; }
                    .log-message { color: #ccc; flex: 1; word-break: break-word; }
                    .log-empty { color: #666; text-align: center; padding: 20px; }
                    
                    /* Auto-refresh indicator */
                    .refresh-indicator {
                        position: fixed;
                        bottom: 16px;
                        right: 16px;
                        background: #333;
                        padding: 8px 12px;
                        border-radius: 8px;
                        font-size: 0.8em;
                        color: #888;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .refresh-dot {
                        width: 8px;
                        height: 8px;
                        background: #4CAF50;
                        border-radius: 50%;
                        animation: blink 2s infinite;
                    }
                    @keyframes blink {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.3; }
                    }
                    
                    @media (min-width: 768px) {
                        body { padding: 24px; max-width: 1000px; margin: 0 auto; }
                        .camera-grid { grid-template-columns: repeat(2, 1fr); }
                    }
                </style>
            </head>
            <body>
                <div class="nav">
                    <a href="/recordings">Recordings</a>
                    <a href="/live">Live</a>
                    <a href="/monitor" class="active">Monitor</a>
                    <a href="/settings">Settings</a>
                </div>
                
                <h1>🖥️ Pipeline Monitor</h1>
                
                <div class="section">
                    <h2>📊 System Resources</h2>
                    <div class="stats-grid" id="systemStats">
                        <div class="stat-card">
                            <div class="stat-value" id="cpuValue">--%</div>
                            <div class="stat-label">CPU Usage</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="cpuBar" style="width: 0%"></div></div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="memValue">--%</div>
                            <div class="stat-label">Memory</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="memBar" style="width: 0%"></div></div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="diskValue">--%</div>
                            <div class="stat-label">Disk</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="diskBar" style="width: 0%"></div></div>
                        </div>
                    </div>
                </div>
                
                <div class="section" id="gpuSection" style="display: none;">
                    <h2>🎮 GPU</h2>
                    <div class="gpu-header" id="gpuName">--</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="gpuUtil">--%</div>
                            <div class="stat-label">GPU Usage</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="gpuUtilBar" style="width: 0%"></div></div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="gpuMem">--%</div>
                            <div class="stat-label">VRAM</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="gpuMemBar" style="width: 0%"></div></div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="gpuTemp">--°C</div>
                            <div class="stat-label">Temperature</div>
                            <div class="stat-bar"><div class="stat-bar-fill" id="gpuTempBar" style="width: 0%"></div></div>
                        </div>
                    </div>
                    <div class="gpu-details" id="gpuDetails"></div>
                </div>
                
                <div class="section">
                    <h2>🔍 Detector</h2>
                    <div class="detector-info" id="detectorInfo">
                        <span class="detector-badge">Backend: <strong>--</strong></span>
                        <span class="detector-badge">Location: <strong>--</strong></span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📷 Camera Pipelines</h2>
                    <div class="camera-grid" id="cameraGrid">
                        <!-- Populated by JS -->
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎬 Recent Clips</h2>
                    <div class="clip-list" id="recentClips">
                        <!-- Populated by JS -->
                    </div>
                </div>
                
                <div class="section" id="reprocessingSection" style="display: none;">
                    <h2>🔄 Active Reprocessing</h2>
                    <div class="reprocessing-list" id="reprocessingList">
                        <!-- Populated by JS -->
                    </div>
                </div>
                
                <div class="section">
                    <h2>📜 System Logs</h2>
                    <div class="log-controls">
                        <select id="logCamera" onchange="loadLogs()">
                            <option value="">All Cameras</option>
                        </select>
                        <select id="logLevel" onchange="loadLogs()">
                            <option value="all">All Levels</option>
                            <option value="warning">Warnings & Errors</option>
                            <option value="error">Errors Only</option>
                        </select>
                        <select id="logType" onchange="loadLogs()">
                            <option value="all">All Types</option>
                            <option value="no-http">Hide HTTP Traffic</option>
                            <option value="detection">Detections Only</option>
                            <option value="tracking">Tracking Only</option>
                            <option value="events">Events Only</option>
                            <option value="clips">Clips Only</option>
                            <option value="errors">Errors/Warnings</option>
                        </select>
                        <select id="logMinutes" onchange="loadLogs()">
                            <option value="15">Last 15 min</option>
                            <option value="30" selected>Last 30 min</option>
                            <option value="60">Last 1 hour</option>
                            <option value="120">Last 2 hours</option>
                        </select>
                        <button class="refresh-logs-btn" onclick="loadLogs()">↻ Refresh</button>
                    </div>
                    <div class="log-info" id="logInfo">Loading...</div>
                    <div class="log-container" id="logContainer">
                        <!-- Populated by JS -->
                    </div>
                </div>
                
                <div class="refresh-indicator">
                    <div class="refresh-dot"></div>
                    <span>Auto-refresh: <span id="countdown">2</span>s</span>
                </div>
                
                <script>
                    let refreshInterval = 2000;
                    let countdown = 2;
                    
                    function getBarClass(value) {
                        if (value > 90) return 'danger';
                        if (value > 70) return 'warning';
                        return '';
                    }
                    
                    function updateUI(data) {
                        // System stats
                        const sys = data.system;
                        document.getElementById('cpuValue').textContent = sys.cpu_percent.toFixed(0) + '%';
                        document.getElementById('cpuBar').style.width = sys.cpu_percent + '%';
                        document.getElementById('cpuBar').className = 'stat-bar-fill ' + getBarClass(sys.cpu_percent);
                        
                        document.getElementById('memValue').textContent = sys.memory_percent.toFixed(0) + '%';
                        document.getElementById('memBar').style.width = sys.memory_percent + '%';
                        document.getElementById('memBar').className = 'stat-bar-fill ' + getBarClass(sys.memory_percent);
                        
                        document.getElementById('diskValue').textContent = sys.disk_percent.toFixed(0) + '%';
                        document.getElementById('diskBar').style.width = sys.disk_percent + '%';
                        document.getElementById('diskBar').className = 'stat-bar-fill ' + getBarClass(sys.disk_percent);
                        
                        // GPU stats
                        const gpu = data.gpu;
                        if (gpu && gpu.available) {
                            document.getElementById('gpuSection').style.display = 'block';
                            document.getElementById('gpuName').textContent = '🟢 ' + gpu.name;
                            
                            document.getElementById('gpuUtil').textContent = gpu.utilization + '%';
                            document.getElementById('gpuUtilBar').style.width = gpu.utilization + '%';
                            document.getElementById('gpuUtilBar').className = 'stat-bar-fill ' + getBarClass(gpu.utilization);
                            
                            document.getElementById('gpuMem').textContent = gpu.memory_percent.toFixed(0) + '%';
                            document.getElementById('gpuMemBar').style.width = gpu.memory_percent + '%';
                            document.getElementById('gpuMemBar').className = 'stat-bar-fill ' + getBarClass(gpu.memory_percent);
                            
                            // Temperature (warning at 70C, danger at 85C)
                            const tempClass = gpu.temperature > 85 ? 'danger' : (gpu.temperature > 70 ? 'warning' : '');
                            document.getElementById('gpuTemp').textContent = gpu.temperature + '°C';
                            document.getElementById('gpuTempBar').style.width = Math.min(gpu.temperature, 100) + '%';
                            document.getElementById('gpuTempBar').className = 'stat-bar-fill ' + tempClass;
                            
                            // Details row
                            let details = `<span>VRAM: ${gpu.memory_used_mb}/${gpu.memory_total_mb} MB</span>`;
                            if (gpu.power_draw > 0) {
                                details += `<span>Power: ${gpu.power_draw}W / ${gpu.power_limit}W</span>`;
                            }
                            document.getElementById('gpuDetails').innerHTML = details;
                        } else {
                            document.getElementById('gpuSection').style.display = 'none';
                        }
                        
                        // Detector info
                        const det = data.detector;
                        document.getElementById('detectorInfo').innerHTML = `
                            <span class="detector-badge">Backend: <strong>${det.backend}</strong></span>
                            <span class="detector-badge">Location: <strong>${det.country || 'Not set'}</strong></span>
                        `;
                        
                        // Camera grid
                        const cameraGrid = document.getElementById('cameraGrid');
                        cameraGrid.innerHTML = data.cameras.map(cam => {
                            let statusClass = 'status-' + cam.status;
                            let statusText = cam.status;
                            if (cam.event_active) {
                                statusClass = 'status-detecting';
                                statusText = 'detecting';
                            }
                            
                            let eventHtml = '';
                            if (cam.event_active) {
                                eventHtml = `
                                    <div class="event-info">
                                        🎯 <span class="event-species">${cam.event_species.join(', ')}</span>
                                        (${(cam.event_confidence * 100).toFixed(0)}%) • 
                                        ${cam.event_duration}s • 
                                        ${cam.tracks_active} track${cam.tracks_active !== 1 ? 's' : ''}
                                    </div>
                                `;
                            }
                            
                            return `
                                <div class="camera-card">
                                    <div class="camera-header">
                                        <div>
                                            <div class="camera-name">${cam.name}</div>
                                            <div class="camera-location">${cam.location || cam.id}</div>
                                        </div>
                                        <span class="status-badge ${statusClass}">${statusText}</span>
                                    </div>
                                    <div class="camera-stats">
                                        <span title="${cam.buffer_frames}/${cam.buffer_max_frames} frames">⏱️ ${cam.buffer_seconds.toFixed(1)}s / ${cam.buffer_max_seconds}s</span>
                                        <span>${cam.tracking_enabled ? '🎯 Tracking' : '📍 No tracking'}</span>
                                    </div>
                                    ${eventHtml}
                                </div>
                            `;
                        }).join('');
                        
                        // Recent clips
                        const clipList = document.getElementById('recentClips');
                        if (data.recent_clips.length > 0) {
                            clipList.innerHTML = data.recent_clips.map(clip => `
                                <a class="clip-item" href="/recording/${clip.path}">
                                    <span class="clip-species">${clip.species}</span>
                                    <span class="clip-meta">${clip.camera} • ${clip.time}</span>
                                </a>
                            `).join('');
                        } else {
                            clipList.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No recent clips</div>';
                        }
                        
                        // Reprocessing jobs
                        const reprocessSection = document.getElementById('reprocessingSection');
                        const reprocessList = document.getElementById('reprocessingList');
                        if (data.reprocessing_jobs && data.reprocessing_jobs.length > 0) {
                            reprocessSection.style.display = 'block';
                            reprocessList.innerHTML = data.reprocessing_jobs.map(job => {
                                const startTime = new Date(job.started).toLocaleTimeString();
                                return `
                                    <div class="reprocessing-item">
                                        <div class="reprocessing-spinner"></div>
                                        <div class="reprocessing-info">
                                            <div class="reprocessing-clip">${job.clip_name}</div>
                                            <div class="reprocessing-meta">${job.camera} • Started ${startTime}</div>
                                        </div>
                                    </div>
                                `;
                            }).join('');
                        } else {
                            reprocessSection.style.display = 'none';
                        }
                        
                        // Update camera dropdown for logs
                        updateCameraDropdown(data.cameras);
                    }
                    
                    async function fetchData() {
                        try {
                            const response = await fetch('/api/monitor');
                            const data = await response.json();
                            updateUI(data);
                        } catch (e) {
                            console.error('Failed to fetch monitor data:', e);
                        }
                    }
                    
                    // Initial fetch
                    fetchData();
                    
                    // Auto-refresh every 2 seconds
                    setInterval(fetchData, refreshInterval);
                    
                    // Countdown display
                    setInterval(() => {
                        countdown--;
                        if (countdown <= 0) countdown = 2;
                        document.getElementById('countdown').textContent = countdown;
                    }, 1000);
                    
                    // Log viewer functions
                    let logsLoaded = false;
                    
                    // Log type filter patterns
                    const LOG_FILTERS = {
                        'all': null,
                        'no-http': {
                            exclude: [/GET \//i, /POST \//i, /DELETE \//i, /PUT \//i, /HTTP\/\d/i, /\d{3} \d+ bytes/i, /aiohttp/i]
                        },
                        'detection': {
                            include: [/detect/i, /species/i, /confidence/i, /infer/i, /YOLO/i, /SpeciesNet/i]
                        },
                        'tracking': {
                            include: [/track/i, /ByteTrack/i, /lost_buffer/i, /merge/i]
                        },
                        'events': {
                            include: [/event/i, /started tracking/i, /closed/i, /clip at/i]
                        },
                        'clips': {
                            include: [/clip/i, /recording/i, /write_clip/i, /storage/i, /mp4/i]
                        },
                        'errors': {
                            include: [/error/i, /warning/i, /failed/i, /exception/i, /traceback/i]
                        }
                    };
                    
                    function filterLog(log, filterType) {
                        const filter = LOG_FILTERS[filterType];
                        if (!filter) return true;
                        
                        const msg = log.message || '';
                        
                        if (filter.exclude) {
                            for (const pattern of filter.exclude) {
                                if (pattern.test(msg)) return false;
                            }
                            return true;
                        }
                        
                        if (filter.include) {
                            for (const pattern of filter.include) {
                                if (pattern.test(msg)) return true;
                            }
                            return false;
                        }
                        
                        return true;
                    }
                    
                    async function loadLogs() {
                        const camera = document.getElementById('logCamera').value;
                        const level = document.getElementById('logLevel').value;
                        const logType = document.getElementById('logType').value;
                        const minutes = document.getElementById('logMinutes').value;
                        
                        const logContainer = document.getElementById('logContainer');
                        const logInfo = document.getElementById('logInfo');
                        
                        logInfo.textContent = 'Loading...';
                        
                        try {
                            const params = new URLSearchParams({
                                minutes: minutes,
                                level: level,
                                type: logType,  // Pass filter type to server
                            });
                            if (camera) params.set('camera', camera);
                            
                            const response = await fetch('/api/logs?' + params);
                            const data = await response.json();
                            
                            // Server already filtered, use logs directly
                            const logs = data.logs;
                            
                            // Update info
                            let sourceText = data.source === 'journalctl' ? 'systemd journal' : 
                                            (data.source === 'journalctl+logfile' ? 'journal + logs' :
                                            (data.source === 'logfile' ? 'log files' : 'no source'));
                            let filterText = data.skipped > 0 ? ` (${data.skipped} filtered out)` : '';
                            let errorText = data.error ? ` [${data.error}]` : '';
                            logInfo.textContent = `${logs.length} entries from ${sourceText}${filterText}${errorText}`;
                            
                            // Render logs
                            if (logs.length === 0) {
                                logContainer.innerHTML = '<div class="log-empty">No log entries found</div>';
                            } else {
                                logContainer.innerHTML = logs.map(log => `
                                    <div class="log-entry">
                                        <span class="log-time">${log.time}</span>
                                        ${log.camera ? `<span class="log-camera">${log.camera}</span>` : ''}
                                        <span class="log-level ${log.level}">${log.level.toUpperCase()}</span>
                                        <span class="log-message">${escapeHtml(log.message)}</span>
                                    </div>
                                `).join('');
                            }
                            
                            logsLoaded = true;
                        } catch (e) {
                            logInfo.textContent = 'Error loading logs';
                            logContainer.innerHTML = '<div class="log-empty">Failed to load logs: ' + e + '</div>';
                        }
                    }
                    
                    function escapeHtml(text) {
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }
                    
                    // Populate camera dropdown from monitor data
                    function updateCameraDropdown(cameras) {
                        const select = document.getElementById('logCamera');
                        const currentValue = select.value;
                        
                        // Keep "All Cameras" option
                        let html = '<option value="">All Cameras</option>';
                        cameras.forEach(cam => {
                            html += `<option value="${cam.id}" ${cam.id === currentValue ? 'selected' : ''}>${cam.name} (${cam.id})</option>`;
                        });
                        select.innerHTML = html;
                    }
                    
                    // Load logs on page load
                    setTimeout(loadLogs, 500);
                    
                    // Auto-refresh logs every 30 seconds
                    setInterval(() => {
                        if (logsLoaded) loadLogs();
                    }, 30000);
                </script>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

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
                        width: 100vw;
                        margin-left: calc(-50vw + 50%);
                        padding-left: 16px;
                        padding-right: 16px;
                        box-sizing: border-box;
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
                    <a href="/recordings">Recordings</a>
                    <a href="/live">Live</a>
                    <a href="/monitor">Monitor</a>
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
                                    <label class="setting-label">Sample Rate</label>
                                    <input type="number" min="1" max="30" step="1"
                                           value="${g.clip.sample_rate || 3}"
                                           onchange="updateGlobalValue('clip', 'sample_rate', parseInt(this.value))">
                                    <div class="setting-description">Analyze every Nth frame (lower = more thorough, slower)</div>
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
                                
                                <div class="setting-row">
                                    <label class="setting-label">Track Merge Gap</label>
                                    <input type="number" min="10" max="500" step="10"
                                           value="${g.clip.track_merge_gap || 120}"
                                           onchange="updateGlobalValue('clip', 'track_merge_gap', parseInt(this.value))">
                                    <div class="setting-description">Max frame gap to merge same-species tracks</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Spatial Merge</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.spatial_merge_enabled !== false ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'spatial_merge_enabled', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Merge tracks in same location (ignores species misclassifications)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Spatial Overlap (IoU)</label>
                                    <div class="slider-container">
                                        <input type="range" min="10" max="90" step="5" 
                                               value="${Math.round((g.clip.spatial_merge_iou || 0.3) * 100)}"
                                               oninput="updateGlobalSlider(this, 'clip', 'spatial_merge_iou')">
                                        <span class="slider-value" id="spatial_merge_iou-value">${Math.round((g.clip.spatial_merge_iou || 0.3) * 100)}%</span>
                                    </div>
                                    <div class="setting-description">Min bounding box overlap to merge (30% recommended)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Hierarchical Merging</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.hierarchical_merge_enabled !== false ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'hierarchical_merge_enabled', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Merge "animal" tracks into specific species tracks</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Single Animal Mode</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.single_animal_mode ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'single_animal_mode', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Force merge ALL non-overlapping tracks into one (use when confident there's only one animal)</div>
                                </div>
                                
                                <div class="setting-row">
                                    <label class="setting-label">Cropped Thumbnails</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${g.clip.thumbnail_cropped !== false ? 'checked' : ''}
                                               onchange="updateGlobalValue('clip', 'thumbnail_cropped', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div class="setting-description">Zoom thumbnails to detection area (off = full frame with bounding box)</div>
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
                'sample_rate': getattr(clip_cfg, 'sample_rate', 3),
                'tracking_enabled': getattr(clip_cfg, 'tracking_enabled', True),
                'track_merge_gap': getattr(clip_cfg, 'track_merge_gap', 120),
                'spatial_merge_enabled': getattr(clip_cfg, 'spatial_merge_enabled', True),
                'spatial_merge_iou': getattr(clip_cfg, 'spatial_merge_iou', 0.3),
                'hierarchical_merge_enabled': getattr(clip_cfg, 'hierarchical_merge_enabled', True),
                'single_animal_mode': getattr(clip_cfg, 'single_animal_mode', False),
                'thumbnail_cropped': getattr(clip_cfg, 'thumbnail_cropped', True),
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
                if 'sample_rate' in clip:
                    runtime.general.clip.sample_rate = int(clip['sample_rate'])
                if 'tracking_enabled' in clip:
                    runtime.general.clip.tracking_enabled = bool(clip['tracking_enabled'])
                if 'track_merge_gap' in clip:
                    runtime.general.clip.track_merge_gap = int(clip['track_merge_gap'])
                if 'spatial_merge_enabled' in clip:
                    runtime.general.clip.spatial_merge_enabled = bool(clip['spatial_merge_enabled'])
                if 'spatial_merge_iou' in clip:
                    runtime.general.clip.spatial_merge_iou = float(clip['spatial_merge_iou'])
                if 'hierarchical_merge_enabled' in clip:
                    runtime.general.clip.hierarchical_merge_enabled = bool(clip['hierarchical_merge_enabled'])
                if 'single_animal_mode' in clip:
                    runtime.general.clip.single_animal_mode = bool(clip['single_animal_mode'])
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
                if 'sample_rate' in clip:
                    config['general']['clip']['sample_rate'] = clip['sample_rate']
                if 'tracking_enabled' in clip:
                    config['general']['clip']['tracking_enabled'] = clip['tracking_enabled']
                if 'track_merge_gap' in clip:
                    config['general']['clip']['track_merge_gap'] = clip['track_merge_gap']
                if 'spatial_merge_enabled' in clip:
                    config['general']['clip']['spatial_merge_enabled'] = clip['spatial_merge_enabled']
                if 'spatial_merge_iou' in clip:
                    config['general']['clip']['spatial_merge_iou'] = clip['spatial_merge_iou']
                if 'hierarchical_merge_enabled' in clip:
                    config['general']['clip']['hierarchical_merge_enabled'] = clip['hierarchical_merge_enabled']
                if 'single_animal_mode' in clip:
                    config['general']['clip']['single_animal_mode'] = clip['single_animal_mode']
            
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

    def _load_ptz_state(self) -> dict:
        """Load persisted PTZ state from file."""
        if not self.state_file or not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load PTZ state: {e}")
            return {}
    
    def _save_ptz_state(self, state: dict) -> None:
        """Save PTZ state to file."""
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            LOGGER.debug(f"Saved PTZ state to {self.state_file}")
        except Exception as e:
            LOGGER.error(f"Failed to save PTZ state: {e}")
    
    def _apply_ptz_state(self) -> None:
        """Apply persisted PTZ state to trackers after startup."""
        state = self._load_ptz_state()
        if not state:
            return
        
        for cam_id, cam_state in state.items():
            worker = self.workers.get(cam_id)
            if not worker:
                continue
            
            tracker = getattr(worker, 'ptz_tracker', None)
            if not tracker:
                continue
            
            # Restore patrol presets
            if 'patrol_presets' in cam_state:
                preset_tokens = cam_state['patrol_presets']
                tracker.patrol_presets = preset_tokens
                tracker._preset_tokens = preset_tokens
                tracker._current_preset_index = 0
                if preset_tokens:
                    LOGGER.info(f"Restored patrol presets for {cam_id}: {preset_tokens}")
                    # Initialize patrol if it's already active - move to first preset
                    from .ptz_tracker import PTZMode
                    if tracker._mode == PTZMode.PATROL or tracker._patrol_active:
                        tracker._goto_current_preset()
                        LOGGER.info(f"Started patrol for {cam_id} - moving to first preset")
            
            # Restore patrol return delay
            if 'patrol_return_delay' in cam_state:
                tracker.patrol_return_delay = cam_state['patrol_return_delay']
                LOGGER.info(f"Restored patrol return delay for {cam_id}: {cam_state['patrol_return_delay']}s")
            
            # Restore patrol/track enabled states
            if 'patrol_enabled' in cam_state:
                tracker.set_patrol_enabled(cam_state['patrol_enabled'])
            if 'track_enabled' in cam_state:
                tracker.set_track_enabled(cam_state['track_enabled'])
    
    def _update_ptz_state(self, cam_id: str, **kwargs) -> None:
        """Update and save PTZ state for a camera."""
        state = self._load_ptz_state()
        if cam_id not in state:
            state[cam_id] = {}
        state[cam_id].update(kwargs)
        self._save_ptz_state(state)

    async def start(self):
        # Apply persisted PTZ state to trackers
        self._apply_ptz_state()
        
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
