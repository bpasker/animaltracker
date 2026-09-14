import asyncio
import hashlib
import logging
import logging.handlers
import cv2
import json
import re
import numpy as np
from aiohttp import web
from typing import Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

from .species_names import get_common_name, get_species_icon
from .analysis_recovery import is_unclassified_clip, sidecar_path
from . import configstore

# Server's timezone (configurable or auto-detected)
import threading
import time as _time

# Automated clips are named "<epoch>_<species>.mp4"; the epoch is the event's
# first detection.
_CLIP_EPOCH_RE = re.compile(r'^(\d{9,10})_')


def clip_start_time(clip_file: Path, stat) -> datetime:
    """When a clip's event began, as an aware datetime in the display zone.

    The filename epoch is the first detection. st_mtime is when the transcode
    finished — the END of the event plus encoding time, minutes later for a
    long event — so it is only the fallback for files without an epoch prefix
    (manual clips, imports).
    """
    match = _CLIP_EPOCH_RE.match(clip_file.name)
    if match:
        epoch = int(match.group(1))
        # Reject garbage prefixes: before 2001, or later than the file itself
        # (allowing a day of clock skew between the camera host and storage).
        if 1_000_000_000 <= epoch <= stat.st_mtime + 86400:
            return datetime.fromtimestamp(epoch, tz=CENTRAL_TZ)
    return datetime.fromtimestamp(stat.st_mtime, tz=CENTRAL_TZ)


def primary_thumbnail(clip: dict):
    """URL of the thumbnail that shows what the card says.

    Thumbnails are one per track, in track order, and the first track is often
    a generic "Animal" fragment or a different visitor. Prefer the first track
    that was classified as the clip's own species; fall back to the first.
    """
    thumbs = clip.get('thumbnails') or []
    species = clip.get('species')
    for thumb in thumbs:
        if thumb.get('species') == species and thumb.get('url'):
            return thumb['url']
    return thumbs[0].get('url') if thumbs else None

def _get_timezone(configured_tz: str = None):
    """Get timezone - use configured value if provided, otherwise auto-detect.

    Args:
        configured_tz: Timezone name like "America/Chicago", "US/Central", "UTC"
    """
    try:
        from zoneinfo import ZoneInfo

        # Use configured timezone if provided
        if configured_tz:
            try:
                return ZoneInfo(configured_tz)
            except Exception as e:
                # LOGGER may not be defined yet at module load time
                import logging
                logging.getLogger(__name__).warning(f"Invalid timezone '{configured_tz}': {e}, falling back to auto-detect")

        # Auto-detect from environment/system
        import os
        tz_name = os.environ.get('TZ')
        if tz_name:
            return ZoneInfo(tz_name)

        # Try to read from /etc/timezone (Linux)
        try:
            with open('/etc/timezone', 'r') as f:
                tz_name = f.read().strip()
                if tz_name:
                    return ZoneInfo(tz_name)
        except (FileNotFoundError, IOError):
            pass

        # Try to read from /etc/localtime symlink (Linux/macOS)
        try:
            import os.path
            localtime = os.path.realpath('/etc/localtime')
            # Extract timezone from path like /usr/share/zoneinfo/America/Chicago
            if 'zoneinfo/' in localtime:
                tz_name = localtime.split('zoneinfo/')[-1]
                return ZoneInfo(tz_name)
        except (FileNotFoundError, IOError, OSError):
            pass

    except ImportError:
        pass

    # Fallback: create a fixed offset timezone from system's UTC offset
    # This handles DST at the moment of detection but won't auto-update
    offset_seconds = -_time.timezone if _time.daylight == 0 else -_time.altzone
    offset = timedelta(seconds=offset_seconds)
    tz_name = _time.tzname[1] if _time.daylight and _time.localtime().tm_isdst else _time.tzname[0]
    return timezone(offset, tz_name)

def _get_timezone_display_name(tz):
    """Get a human-readable timezone name for display."""
    try:
        # Try to get the zone name from ZoneInfo
        if hasattr(tz, 'key'):
            return tz.key  # e.g., "America/Chicago"
        # Fallback to tzname
        now = datetime.now(tz)
        return tz.tzname(now) or str(tz)
    except Exception:
        return str(tz)

# Initialize with auto-detect (will be updated when WebServer starts with config)
LOCAL_TZ = _get_timezone()
CENTRAL_TZ = LOCAL_TZ  # Backwards compatibility alias
TIMEZONE_DISPLAY = _get_timezone_display_name(LOCAL_TZ)

def configure_timezone(tz_name: str = None):
    """Configure the global timezone. Called when config is loaded."""
    global LOCAL_TZ, CENTRAL_TZ, TIMEZONE_DISPLAY
    LOCAL_TZ = _get_timezone(tz_name)
    CENTRAL_TZ = LOCAL_TZ
    TIMEZONE_DISPLAY = _get_timezone_display_name(LOCAL_TZ)

if TYPE_CHECKING:
    from .pipeline import StreamWorker

LOGGER = logging.getLogger(__name__)

# A PTZ 'move' starts continuous motion that only ends when a 'stop' arrives. If that
# stop is lost — dropped request, closed tab, backgrounded phone, a touchend that never
# fires — the camera slews until it hits a limit. This is the server-side backstop: any
# move is automatically stopped after this long unless another PTZ command supersedes it.
# Generous on purpose, so it only ever catches the failure case and never interrupts a
# deliberate long pan.
PTZ_DEADMAN_SECONDS = 10.0

# A frame older than this means the stream is no longer live: the snapshot and MJPEG
# paths dim the frame and draw the STREAM DOWN banner, and /api/cameras reports 'stale'.
# Defined once -- it was previously a local in two handlers, with a comment elsewhere
# claiming a different value.
STALE_AFTER_SECONDS = 5.0

# Every archive read walks the whole clips tree and stats each file, and a single page
# load does it twice. The detection pipeline writes new clips outside this process, so
# explicit invalidation alone cannot keep a cache honest -- hence a short TTL. New
# recordings therefore appear within this many seconds, which is well inside what anyone
# notices, while repeat reads within a burst become free.
RECORDINGS_CACHE_TTL = 3.0

# Static assets live next to this module and are served straight from the git
# checkout (production runs an editable install), so a deploy is just `git pull`.
STATIC_DIR = Path(__file__).parent / 'static'


def _compute_asset_version(static_dir: Path) -> str:
    """Content hash of every static asset, used to bust far-future caches.

    Hashing content rather than mtime matters here: `git pull` rewrites mtimes on
    every deploy, which would invalidate the cache even when nothing changed.
    """
    h = hashlib.sha256()
    if static_dir.is_dir():
        for path in sorted(static_dir.rglob('*')):
            if path.is_file():
                h.update(path.relative_to(static_dir).as_posix().encode())
                h.update(path.read_bytes())
    return h.hexdigest()[:12]


@web.middleware
async def _static_cache_middleware(request, handler):
    """Far-future cache for versioned asset URLs, revalidate for unversioned ones."""
    response = await handler(request)
    if request.path.startswith('/static/'):
        try:
            if request.query.get('v'):
                response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
            else:
                response.headers['Cache-Control'] = 'no-cache'
        except (AttributeError, TypeError):
            pass
    return response


# --- Pre-compiled regex patterns for /api/logs (compiled once at import) ---
# HTTP access-log noise we always want to exclude.
_HTTP_EXCLUDE_PATTERNS = [
    r'GET /', r'POST /', r'DELETE /', r'PUT /',
    r'HTTP/\d', r'\d{3} \d+ bytes', r'aiohttp',
]

# Level and logger prefix the app puts on every line: the current
# 'LEVEL name: message' form and the older 'LEVEL:name:message' form (journal
# entries written before the app tagged stderr lines with a syslog priority
# carry the old form and are all priority 6). Type filters run on the message
# *body* only: the logger name 'animaltracker.*' would otherwise satisfy the
# 'track' pattern on every single line.
_APP_LINE_RE = re.compile(
    r'^(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)'
    r'(?::(?P<logger_old>[A-Za-z_][\w.]*):|\s(?P<logger>[A-Za-z_][\w.]*):\s?)'
)
# The same prefix as a PCRE fragment for `journalctl --grep`. Possessive (?+):
# once the prefix is consumed the body search cannot backtrack into it, so
# 'track' cannot match inside 'animaltracker'.
_JOURNAL_PREFIX_PCRE = r'^(?:(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)(?::[\w.]+:| [\w.]+: ?))?+'
# Timestamped app-log file line: '[2026-09-09 06:19:31,477] LEVEL name: message'.
_FILE_LINE_RE = re.compile(r'^\[[^\]]*\]\s+(?=(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s)')
_LEVEL_NAME_TO_KEY = {
    'DEBUG': 'debug', 'INFO': 'info', 'WARNING': 'warning',
    'ERROR': 'error', 'CRITICAL': 'error',
}

_LOG_TYPE_FILTERS_RAW = {
    'all': None,
    'no-http': {'exclude': _HTTP_EXCLUDE_PATTERNS},
    'realtime': {
        'include': [r'\[REALTIME\]', r'raw detections'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'detection': {
        'include': [r'detect', r'species', r'confidence', r'infer', r'YOLO', r'SpeciesNet', r'\[REALTIME\]'],
        'loggers': ['animaltracker.detector'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'tracking': {
        # Word-bounded so 'track' cannot match inside a path such as
        # /opt/speciesnet/animaltracker/storage/... in the body.
        'include': [r'\btrack(?:s|ed|er|ers|ing)?\b', r'ByteTrack', r'lost_buffer',
                    r'\bmerg(?:e|ed|es|ing)\b'],
        'loggers': ['animaltracker.tracker'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'ptz': {
        'include': [r'ptz', r'\[MOVE', r'\[MODE_CHANGE', r'\[TRACKING', r'\[COORD', r'\[OFFSET',
                    r'patrol', r'preset'],
        'loggers': ['ptz.decisions', 'animaltracker.ptz_tracker', 'animaltracker.onvif_client',
                    'animaltracker.ptz_calibration', 'animaltracker.ptz_visual_calibration'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'events': {
        'include': [r'event', r'started tracking', r'closed', r'clip at'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'clips': {
        'include': [r'clip', r'recording', r'write_clip', r'storage', r'\.mp4'],
        'loggers': ['animaltracker.storage', 'animaltracker.clip_buffer'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
    'errors': {
        # By level first (an ERROR line's body rarely says "error"), then the
        # words, which also catch FFmpeg/OpenCV lines that carry no level.
        'levels': ['error', 'warning'],
        'include': [r'error', r'warning', r'failed', r'exception', r'traceback'],
        'exclude': _HTTP_EXCLUDE_PATTERNS,
    },
}

_LOG_TYPE_FILTERS_COMPILED = {}
for _name, _cfg in _LOG_TYPE_FILTERS_RAW.items():
    if _cfg is None:
        _LOG_TYPE_FILTERS_COMPILED[_name] = None
        continue
    _LOG_TYPE_FILTERS_COMPILED[_name] = {
        'include': [re.compile(p, re.IGNORECASE) for p in _cfg.get('include', [])] or None,
        'exclude': [re.compile(p, re.IGNORECASE) for p in _cfg.get('exclude', [])] or None,
        'loggers': frozenset(_cfg.get('loggers', [])),
        'levels': frozenset(_cfg.get('levels', [])),
    }

# journalctl --grep for the level filter. Not -p: entries from before the
# app tagged stderr lines with a priority are all priority 6, so the level
# has to come from the prefix. systemd's own failure notices carry no prefix.
_SYSTEMD_FAILURE_PCRE = r'Main process exited|Failed with result'
_LEVEL_GREP = {
    'error': r'^(?:ERROR|CRITICAL)[: ]|' + _SYSTEMD_FAILURE_PCRE,
    'warning': r'^(?:WARNING|ERROR|CRITICAL)[: ]|' + _SYSTEMD_FAILURE_PCRE,
}

# Timestamp parsers used by file-log fallback.
_TS_FULL_RE = re.compile(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})')
_TS_TIME_RE = re.compile(r'(\d{2}:\d{2}:\d{2})')


def _classify_log_line(text, priority=None):
    """Split one log line into ``(level, logger, body)``.

    ``level`` is debug/info/warning/error. It comes from the app's own prefix
    when the line has one -- the truth for journal entries whose priority
    predates the prefix -- and otherwise from the journal ``priority``
    (systemd's own notices, FFmpeg/OpenCV lines written straight to stderr).
    For file lines without a prefix the old keyword heuristic applies.
    ``logger`` is '' for lines without a prefix; ``body`` has the prefix (and
    a file line's timestamp) removed.
    """
    if text.startswith('['):
        m = _FILE_LINE_RE.match(text)
        if m:
            text = text[m.end():]
    m = _APP_LINE_RE.match(text)
    if m:
        return (
            _LEVEL_NAME_TO_KEY[m.group('level')],
            m.group('logger') or m.group('logger_old') or '',
            text[m.end():],
        )
    if priority is not None:
        if priority <= 3:
            level = 'error'
        elif priority <= 4:
            level = 'warning'
        elif priority >= 7:
            level = 'debug'
        else:
            level = 'info'
    else:
        lower = text.lower()
        level = 'error' if 'error' in lower else ('warning' if 'warning' in lower else 'info')
    return level, '', text


def _matches_log_filter(message, filter_type, logger='', level='info'):
    """Apply the type filter to one line.

    ``message`` is the body without the app's level/logger prefix, ``logger``
    the prefix's logger name ('' when there was none) and ``level`` the
    debug/info/warning/error key from :func:`_classify_log_line`.
    """
    cfg = _LOG_TYPE_FILTERS_COMPILED.get(filter_type)
    if cfg is None:
        return True
    excludes = cfg.get('exclude')
    if excludes:
        for pat in excludes:
            if pat.search(message):
                return False
    includes = cfg.get('include')
    if includes is None and not cfg['loggers'] and not cfg['levels']:
        # Only excludes defined -> pass everything not excluded.
        return True
    if logger and logger in cfg['loggers']:
        return True
    if level in cfg['levels']:
        return True
    for pat in includes or ():
        if pat.search(message):
            return True
    return False


def _journal_grep_pattern(log_type, level):
    """PCRE for ``journalctl --grep`` admitting every entry the Python filters
    could accept (a superset), so the journal does the bulk of the filtering
    and ``-n`` counts matching entries rather than the noise around them.

    A sparse type (tracking lines from one event a day) used to be crowded
    out: ``-n 8000`` fetched the newest 8000 lines of the window *before*
    filtering, and on a busy day that was a few hours of [PERF] lines. When
    both a type and a level are set only the type is pushed down; the level
    is enforced in Python. Returns None when nothing useful can be pushed.
    """
    cfg = _LOG_TYPE_FILTERS_RAW.get(log_type)
    if cfg and (cfg.get('include') or cfg.get('loggers') or cfg.get('levels')):
        parts = []
        loggers = cfg.get('loggers') or []
        if loggers:
            alt = '|'.join(re.escape(name) for name in loggers)
            parts.append(r'^(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)[: ](?:%s)[: ]' % alt)
        levels = cfg.get('levels') or ()
        if levels:
            names = [n for n, k in _LEVEL_NAME_TO_KEY.items() if k in levels]
            parts.append(r'^(?:%s)[: ]' % '|'.join(names))
            parts.append(_SYSTEMD_FAILURE_PCRE)
        includes = cfg.get('include') or []
        if includes:
            parts.append(_JOURNAL_PREFIX_PCRE + r'.*?(?:%s)' % '|'.join('(?:%s)' % p for p in includes))
        return '|'.join(parts)
    return _LEVEL_GREP.get(level)


_EMPTY_SYSTEM_STATS = {
    'cpu_percent': 0, 'memory_percent': 0, 'memory_used_gb': 0, 'memory_total_gb': 0,
    'disk_percent': 0, 'disk_used_gb': 0, 'disk_total_gb': 0,
}
_EMPTY_GPU_STATS = {
    'available': False, 'name': None, 'utilization': 0, 'memory_percent': 0,
    'memory_used_mb': 0, 'memory_total_mb': 0, 'temperature': 0, 'power_draw': 0, 'power_limit': 0,
}


class WebServer:
    def __init__(self, workers: Dict[str, 'StreamWorker'], storage_root: Path, logs_root: Path, port: int = 8080, config_path: Path = None, runtime = None,
                 analysis_registry=None, recovery=None):
        self.workers = workers
        self.storage_root = storage_root
        self.logs_root = logs_root
        self.port = port
        self.config_path = config_path
        self.runtime = runtime
        # The pipeline's record of clip analyses in flight and its recovery
        # sweeper (analysis_recovery.py); both None when the web server runs
        # without a pipeline, and every clip is then simply "unfinished".
        self.analysis_registry = analysis_registry
        self.recovery = recovery

        # Configure timezone from config (or auto-detect if not specified)
        configured_tz = None
        if runtime and hasattr(runtime, 'general') and hasattr(runtime.general, 'timezone'):
            configured_tz = runtime.general.timezone
        configure_timezone(configured_tz)
        LOGGER.info(f"Timezone configured: {TIMEZONE_DISPLAY}" + (f" (from config)" if configured_tz else " (auto-detected)"))

        # State file for persisting PTZ settings across restarts
        # Store in config directory alongside cameras.yml
        config_dir = config_path.parent if config_path else None
        self.state_file = config_dir / 'ptz_state.json' if config_dir else None
        # Track active reprocessing jobs: {clip_path: {'started': timestamp, 'clip_name': name}}
        self.reprocessing_jobs: Dict[str, dict] = {}
        # Per-camera PTZ dead-man timers: {camera_id: asyncio.Task}
        self._ptz_deadman: Dict[str, 'asyncio.Task'] = {}
        # Short-lived cache of the archive scan; see RECORDINGS_CACHE_TTL.
        self._scan_cache = None
        self._scan_cache_ts = 0.0
        self.asset_version = _compute_asset_version(STATIC_DIR)
        LOGGER.info("Static asset version: %s", self.asset_version)
        self.app = web.Application(middlewares=[_static_cache_middleware])
        # Every path the server-rendered pages once answered sends its
        # bookmark, query string intact, into the client-side app below.
        self.app.router.add_get('/', self.handle_root_redirect)
        self.app.router.add_get('/live', self._redirect_to('/app/live'))
        self.app.router.add_get('/recordings', self._redirect_to('/app/recordings'))
        self.app.router.add_get('/recording/{path:.*}', self.handle_recording_redirect)
        self.app.router.add_get('/monitor', self._redirect_to('/app/monitor'))
        self.app.router.add_get('/settings', self._redirect_to('/app/settings'))

        self.app.router.add_get('/snapshot/{camera_id}', self.handle_snapshot)
        self.app.router.add_get('/stream/{camera_id}', self.handle_stream)
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
        self.app.router.add_post('/ptz/zoom-fov-calibrate', self.handle_zoom_fov_calibrate)
        self.app.router.add_get('/ptz/debug', self.handle_get_ptz_debug)
        self.app.router.add_post('/ptz/debug', self.handle_set_ptz_debug)
        self.app.router.add_delete('/recordings', self.handle_delete_recording)
        self.app.router.add_post('/recordings/bulk_delete', self.handle_bulk_delete)
        self.app.router.add_post('/recordings/reprocess', self.handle_reprocess)
        self.app.router.add_get('/recordings/log/{path:.*}', self.handle_get_processing_log)
        self.app.router.add_get('/api/monitor', self.handle_get_monitor_data)
        self.app.router.add_get('/api/logs', self.handle_get_logs)
        # Configuration editor behind /app/settings: the file is the source of
        # truth, saves are validated + backed up, restart-only fields are
        # reported as pending. See configstore.py.
        self.app.router.add_get('/api/config', self.handle_get_config)
        self.app.router.add_post('/api/config', self.handle_save_config)
        self.app.router.add_post('/api/config/probe', self.handle_probe_camera)
        self.app.router.add_post('/api/system/restart', self.handle_restart)
        self.app.router.add_post('/api/secrets', self.handle_set_secret)

        # The client-side app (static/): one shell for every /app path, so a
        # deep link works, and the JSON API below is everything it talks to.
        self.app.router.add_get('/app', self.handle_app_shell)
        self.app.router.add_get('/app/{tail:.*}', self.handle_app_shell)

        # JSON API for the front-end
        self.app.router.add_get('/api/recordings', self.handle_recordings_api)
        self.app.router.add_get('/api/clip/{path:.*}', self.handle_clip_api)
        self.app.router.add_get('/api/cameras', self.handle_cameras_api)

        # Calendar API endpoints
        self.app.router.add_get('/api/recordings/calendar', self.handle_calendar_api)
        self.app.router.add_get('/api/recordings/day/{date}', self.handle_day_api)
        
        # Front-end assets (stylesheet, modules, icon sprite)
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        self.app.router.add_static('/static', STATIC_DIR)

        # Serve clips directory statically
        clips_path = self.storage_root / 'clips'
        # Ensure it exists so static route doesn't fail on startup
        clips_path.mkdir(parents=True, exist_ok=True)
        self.app.router.add_static('/clips', clips_path, show_index=True)

    async def handle_root_redirect(self, request):
        """The root lands in the app's archive."""
        raise web.HTTPFound('/app/recordings')

    @staticmethod
    def _redirect_to(target: str):
        """A handler that sends an old page path to `target`, query intact.

        The pages that lived at /live, /recordings, /monitor and /settings are
        gone; their query strings were the only thing anyone bookmarked, and
        the app reads the same parameters, so nothing saved is lost.
        """
        async def handler(request):
            qs = request.query_string
            raise web.HTTPFound(target + ('?' + qs if qs else ''))
        return handler

    async def handle_recording_redirect(self, request):
        """/recording/<path> was the clip page; the app's is /app/clips/<path>.

        match_info arrives decoded, so the path is re-encoded per segment.
        """
        rel_path = request.match_info.get('path', '')
        raise web.HTTPFound('/app/clips/' + quote(rel_path, safe='/'))

    def _cancel_ptz_deadman(self, camera_id: str) -> None:
        """Cancel any pending auto-stop for this camera."""
        task = self._ptz_deadman.pop(camera_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _arm_ptz_deadman(self, camera_id: str, worker) -> None:
        """(Re)arm the auto-stop watchdog for a camera that was just told to move."""
        self._cancel_ptz_deadman(camera_id)

        async def _auto_stop():
            try:
                await asyncio.sleep(PTZ_DEADMAN_SECONDS)
            except asyncio.CancelledError:
                return
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, worker.onvif_client.ptz_stop, worker.onvif_profile_token
                )
                LOGGER.warning(
                    "PTZ dead-man stopped %s after %.0fs with no follow-up command "
                    "(a 'stop' was probably lost)",
                    camera_id, PTZ_DEADMAN_SECONDS,
                )
            except Exception as e:
                LOGGER.error("PTZ dead-man stop failed for %s: %s", camera_id, e, exc_info=True)
            finally:
                self._ptz_deadman.pop(camera_id, None)

        self._ptz_deadman[camera_id] = asyncio.create_task(_auto_stop())

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
                # Motion is now continuous; guarantee it ends even if the stop never arrives.
                self._arm_ptz_deadman(camera_id, worker)
            elif action == 'stop':
                self._cancel_ptz_deadman(camera_id)
                await loop.run_in_executor(
                    None, 
                    worker.onvif_client.ptz_stop, 
                    worker.onvif_profile_token
                )
            else:
                return web.Response(status=400, text="Invalid action")
                
            return web.Response(text="OK")
        except Exception as e:
            LOGGER.error(f"PTZ error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ position error: {e}", exc_info=True)
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
                'patrol_return_delay': 5.0
            })
        
        import time
        now = time.time()
        last_detection_age = now - tracker._last_detection_time if tracker._last_detection_time > 0 else None
        last_update_age = now - tracker._last_update if tracker._last_update > 0 else None

        return web.json_response({
            'mode': tracker._mode.value if hasattr(tracker._mode, 'value') else str(tracker._mode),
            'patrol_enabled': tracker.is_patrol_enabled(),
            'track_enabled': tracker.is_track_enabled(),
            'patrol_return_delay': tracker.patrol_return_delay,
            # Debug info
            'debug': {
                'last_detection_age_seconds': round(last_detection_age, 1) if last_detection_age else None,
                'last_update_age_seconds': round(last_update_age, 1) if last_update_age else None,
                'patrol_presets': tracker.patrol_presets,
                'preset_tokens': tracker._preset_tokens,
                'current_preset_index': tracker._current_preset_index,
                'patrol_active': tracker._patrol_active,
                'track_active': tracker._track_active,
                'last_tracked_species': tracker._last_tracked_species,
                'last_detection_source': tracker._last_detection_source,
            }
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
            LOGGER.error(f"PTZ patrol toggle error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ track toggle error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ return delay error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ presets error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ set patrol presets error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ goto preset error: {e}", exc_info=True)
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
            LOGGER.error(f"PTZ save preset error: {e}", exc_info=True)
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
            LOGGER.exception("PTZ calibration error: %s", e)
            return web.json_response({'error': str(e)}, status=500)

    async def handle_zoom_fov_calibrate(self, request):
        """Run zoom FOV calibration to map what area of wide cam is visible at each zoom level."""
        from .ptz_calibration import ZoomFOVCalibrator
        import json
        from pathlib import Path

        try:
            data = await request.json()
            wide_camera_id = data.get('wide_camera_id')
            zoom_camera_id = data.get('zoom_camera_id')
            zoom_levels_str = data.get('zoom_levels', '0.0,0.5,1.0')
            settle_time = float(data.get('settle_time', 2.0))

            zoom_levels = [float(x.strip()) for x in zoom_levels_str.split(',')]

            wide_worker = self.workers.get(wide_camera_id)
            zoom_worker = self.workers.get(zoom_camera_id)

            if not wide_worker:
                return web.json_response({'error': f'Wide camera {wide_camera_id} not found'}, status=400)
            if not zoom_worker:
                return web.json_response({'error': f'Zoom camera {zoom_camera_id} not found'}, status=400)

            if not zoom_worker.onvif_client:
                return web.json_response({'error': 'Zoom camera has no ONVIF client'}, status=400)

            if not zoom_worker.onvif_profile_token:
                return web.json_response({'error': 'Zoom camera has no ONVIF profile token'}, status=400)

            if wide_worker.latest_frame is None:
                return web.json_response({'error': 'Wide camera has no frames - is it streaming?'}, status=400)

            if zoom_worker.latest_frame is None:
                return web.json_response({'error': 'Zoom camera has no frames - is it streaming?'}, status=400)

            # Create calibrator
            calibrator = ZoomFOVCalibrator(
                onvif_client=zoom_worker.onvif_client,
                profile_token=zoom_worker.onvif_profile_token,
            )

            # Track frame timestamps to ensure we get fresh frames after zoom changes
            import time as time_module

            def get_fresh_frame(worker, max_wait=3.0):
                """Wait for a fresh frame that's newer than when this function was called."""
                start_time = time_module.time()
                initial_ts = worker.latest_detection_ts

                # Wait for a frame with a newer timestamp
                while time_module.time() - start_time < max_wait:
                    if worker.latest_detection_ts > initial_ts and worker.latest_frame is not None:
                        LOGGER.debug("Got fresh frame: ts=%.3f (was %.3f)",
                                    worker.latest_detection_ts, initial_ts)
                        return worker.latest_frame.copy()
                    time_module.sleep(0.1)

                # Fallback to current frame if no new frame arrived
                LOGGER.warning("Timeout waiting for fresh frame from %s (waited %.1fs)",
                              worker.camera.id, max_wait)
                return worker.latest_frame.copy() if worker.latest_frame is not None else None

            def get_wide_frame():
                return get_fresh_frame(wide_worker)

            def get_zoom_frame():
                return get_fresh_frame(zoom_worker)

            # Calibration physically drives the PTZ and busy-waits for fresh frames at
            # each zoom level, so it runs for tens of seconds. Off the event loop it goes:
            # on the loop it stalls every MJPEG stream, every PTZ command and every request
            # for the whole run. The frame callbacks read worker.latest_frame, which the
            # capture threads keep updating, so they are safe to call from the executor.
            LOGGER.info(f"Starting zoom FOV calibration: wide={wide_camera_id}, zoom={zoom_camera_id}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: calibrator.calibrate_zoom_fov(
                    get_wide_frame=get_wide_frame,
                    get_zoom_frame=get_zoom_frame,
                    zoom_levels=zoom_levels,
                    settle_time=settle_time,
                ),
            )

            if result.error:
                return web.json_response({'error': result.error}, status=400)

            # Save to config directory
            output_path = Path(self.storage_root).parent / 'config' / 'zoom_fov_calibration.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            # Format results for response
            points_info = []
            for p in result.points:
                points_info.append({
                    'zoom_level': p.zoom_level,
                    'zoom_pct': f"{p.zoom_level * 100:.0f}%",
                    'fov_bounds': f"({p.x1:.2f}, {p.y1:.2f}) to ({p.x2:.2f}, {p.y2:.2f})",
                    'fov_size': f"{p.width * 100:.1f}% x {p.height * 100:.1f}%",
                    'confidence': p.confidence,
                })

            return web.json_response({
                'success': True,
                'message': f'Calibration complete. Saved to {output_path}',
                'wide_frame_size': f"{result.wide_frame_width}x{result.wide_frame_height}",
                'points': points_info,
            })

        except Exception as e:
            LOGGER.exception("Zoom FOV calibration error: %s", e)
            return web.json_response({'error': str(e)}, status=500)

    async def handle_get_ptz_debug(self, request):
        """Get current PTZ debug logging state."""
        import logging
        ptz_logger = logging.getLogger('ptz.decisions')
        # A logger's own level is NOTSET (0) until something sets it, and
        # 0 <= DEBUG, so checking .level reported "enabled" by default.
        is_enabled = ptz_logger.getEffectiveLevel() <= logging.DEBUG
        return web.json_response({'enabled': is_enabled})

    async def handle_set_ptz_debug(self, request):
        """Enable or disable PTZ debug logging."""
        import logging
        try:
            data = await request.json()
            enabled = data.get('enabled', False)

            ptz_logger = logging.getLogger('ptz.decisions')
            if enabled:
                ptz_logger.setLevel(logging.DEBUG)
                # Also ensure the root logger can show debug
                logging.getLogger('animaltracker.ptz_tracker').setLevel(logging.DEBUG)
                LOGGER.info("PTZ debug logging ENABLED")
            else:
                ptz_logger.setLevel(logging.INFO)
                # Back to inheriting from 'animaltracker' so a --debug run
                # keeps DEBUG for this module.
                logging.getLogger('animaltracker.ptz_tracker').setLevel(logging.NOTSET)
                LOGGER.info("PTZ debug logging DISABLED")

            return web.json_response({'enabled': enabled, 'success': True})
        except Exception as e:
            LOGGER.error(f"Error setting PTZ debug: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    def _invalidate_scan_cache(self) -> None:
        """Drop the archive cache after a mutation this process performed."""
        self._scan_cache = None

    def _scan_recordings_cached(self):
        """TTL-cached archive scan for request handlers.

        _scan_recordings itself stays uncached: it is the characterized primitive and
        callers that need guaranteed-fresh results (and the tests) use it directly.
        """
        now = _time.monotonic()
        if self._scan_cache is not None and (now - self._scan_cache_ts) < RECORDINGS_CACHE_TTL:
            return self._scan_cache
        clips = self._scan_recordings()
        self._scan_cache = clips
        self._scan_cache_ts = now
        return clips

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
                'thumbnails': [],
                'unfinished': False,
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

                # Still named by the real-time detector and without the
                # post-processor's sidecar: its analysis never finished.
                # Checked by exact path (see _get_thumbnails_for_clip on
                # stale NFS listings).
                unfinished = is_unclassified_clip(clip_file) and not sidecar_path(clip_file).exists()

                when = clip_start_time(clip_file, stat)
                clips.append({
                    'path': str(rel_path),
                    'camera': cam_dir.name,
                    'date': when.strftime('%Y-%m-%d'),
                    'filename': clip_file.name,
                    'time': when,
                    'size': stat.st_size,
                    'species': species_display,
                    'raw_species': raw_species,
                    'thumbnails': thumbnails,
                    'unfinished': unfinished,
                })

        # Sort by time descending
        clips.sort(key=lambda x: x['time'], reverse=True)
        return clips

    def _get_thumbnails_for_clip(self, clip_path: Path, log_data: dict | None = None) -> list:
        """Get all thumbnails associated with a clip file.

        Returns list of dicts with 'path' (relative to clips dir), 'species', 'url',
        and optionally 'track_index' for per-track thumbnails.

        Two sources are merged: the directory listing, and the ``thumbnails``
        the post-processor recorded in the clip's ``.log.json``, each checked
        by name. On the NFS archive a listing fetched between the clip rename
        and the sidecar writes can stay cached for good and hide files that
        open fine by path; the sidecar is what keeps those key frames visible.
        It is read here only when the listing shows nothing, so the archive
        scan stays cheap; ``_get_clip_detail`` passes the sidecar it loads anyway.
        """
        clip_dir = clip_path.parent
        found: Dict[str, Path] = {}
        for thumb_file in clip_dir.glob(f"{clip_path.stem}_thumb_*.jpg"):
            found[thumb_file.name] = thumb_file

        if log_data is None and not found:
            log_data = self._read_sidecar(clip_path)
        for name in self._sidecar_thumbnail_names(log_data, clip_path):
            if name in found:
                continue
            candidate = clip_dir / name
            if candidate.is_file():
                found[name] = candidate

        thumbnails = [self._thumbnail_entry(path) for path in found.values()]
        # Sort by track_index if present, to maintain consistent order
        thumbnails.sort(key=lambda x: (x.get('track_index', 999), x['path']))
        return thumbnails

    def _read_sidecar(self, clip_path: Path) -> dict:
        """The clip's ``.log.json`` as a dict; ``{}`` when absent or unreadable."""
        log_path = clip_path.with_suffix('.log.json')
        if not log_path.exists():
            return {}
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load processing log: %s", e)
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _sidecar_thumbnail_names(log_data, clip_path: Path) -> list:
        """Thumbnail file names the sidecar lists for this clip.

        Entries are ``{"file": name}`` dicts. Only plain names of this clip's
        own thumbnails count: the sidecar is data read from disk, and after a
        rename without regeneration it still names the old stem.
        """
        if not isinstance(log_data, dict):
            return []
        entries = log_data.get('thumbnails')
        if not isinstance(entries, list):
            return []
        prefix = f"{clip_path.stem}_thumb_"
        names = []
        for entry in entries:
            name = entry.get('file') if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name.startswith(prefix) or not name.endswith('.jpg'):
                continue
            if Path(name).name != name:
                continue
            names.append(name)
        return names

    def _thumbnail_entry(self, thumb_file: Path) -> dict:
        """The API's description of one thumbnail file, parsed from its name."""
        clips_dir = self.storage_root / 'clips'
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
        return thumb_data

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
            day_thumbs = [
                {'url': f"/clips/{t['path']}", 'species': t['species']}
                for t in clip.get('thumbnails', [])
            ]
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
                'thumbnails': day_thumbs,
                'thumbnail': primary_thumbnail({'species': clip_species, 'thumbnails': day_thumbs}),
            }
            analysis = self._analysis_state(clip['path'], clip.get('unfinished'))
            if analysis:
                clip_data['analysis'] = analysis
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

    def _camera_identity(self) -> Dict[str, Dict[str, str]]:
        """Camera id -> {'name', 'location'} for every configured camera.

        The running workers come first: the settings page applies a live
        ``name`` or ``location`` edit to their camera objects. The runtime
        config then covers a camera that is configured but not running
        (``run --camera cam1``). Resolved per request rather than at scan
        time, because the archive scan is cached and identity is not part
        of the archive. A retired camera whose clips remain is absent here.
        """
        out: Dict[str, Dict[str, str]] = {}
        sources = [getattr(w, 'camera', None) for w in self.workers.values()]
        sources.extend(getattr(self.runtime, 'cameras', None) or [])
        for cam in sources:
            cam_id = getattr(cam, 'id', None)
            if not cam_id or cam_id in out:
                continue
            name = getattr(cam, 'name', None) or cam_id
            location = getattr(cam, 'location', None) or ''
            out[cam_id] = {'name': str(name), 'location': str(location).strip()}
        return out

    def _analysis_state(self, rel_path: str, unfinished) -> 'str | None':
        """What the archive should say about a clip that has no key frames yet.

        ``running``: a job holds the clip right now (a live event, a
        reanalysis or the recovery sweep). ``queued``: its analysis never
        finished and the sweeper will get to it. ``unfinished``: never
        finished and nothing will pick it up (recovery off, or no pipeline
        behind this server). None for every clip that has been analysed.
        The registry is consulted per request because a job starts and ends
        within the archive cache's lifetime.
        """
        registry = self.analysis_registry
        if registry is not None and registry.is_active(self.storage_root / 'clips' / rel_path):
            return 'running'
        if not unfinished:
            return None
        recovery = self.recovery
        if recovery is not None:
            try:
                status = recovery.status()
            except Exception:  # noqa: BLE001 - a status hiccup must not break the archive
                status = {}
            if status.get('enabled') and status.get('running'):
                return 'queued'
        return 'unfinished'

    def _clip_to_json(self, clip: dict, identity: dict = None) -> dict:
        """Serialise a _scan_recordings entry for the API.

        `time` is emitted as ISO-8601 with offset so the client never has to guess
        a timezone, alongside the epoch for cheap sorting and relative formatting.
        `camera_name` and `location` come from `identity` (`_camera_identity`);
        a retired camera keeps its id as its name and has no location.
        `analysis` is present only while there is something to say about a
        clip's missing key frames (`_analysis_state`).
        """
        when = clip['time']
        meta = (identity or {}).get(clip['camera']) or {}
        payload = {
            'path': clip['path'],
            'filename': clip['filename'],
            'camera': clip['camera'],
            'camera_name': meta.get('name') or clip['camera'],
            'location': meta.get('location') or '',
            'species': clip['species'],
            'raw_species': clip.get('raw_species', 'unknown'),
            'date': clip['date'],
            'time': when.isoformat(),
            'epoch': when.timestamp(),
            'size': clip['size'],
            'size_mb': round(clip['size'] / (1024 * 1024), 2),
            'thumbnails': clip.get('thumbnails', []),
            'thumbnail': primary_thumbnail(clip),
        }
        analysis = self._analysis_state(clip['path'], clip.get('unfinished'))
        if analysis:
            payload['analysis'] = analysis
        return payload

    def _filter_clips(self, clips: list, query: dict) -> list:
        """Apply camera / location / species / date-range / free-text filters.

        ``camera`` and ``location`` are one scope: a location stands for every
        camera configured there, and the two select their union, so
        ``camera=cam1&location=Otteson`` is cam1 plus the Otteson cameras. A
        location no camera carries matches nothing. Free text also matches a
        camera's configured name and location.
        """
        cameras = {c for c in (query.get('camera') or '').split(',') if c}
        locations = {loc.strip() for loc in (query.get('location') or '').split(',') if loc.strip()}
        species = {sp for sp in (query.get('species') or '').split(',') if sp}
        date_from = query.get('from') or None
        date_to = query.get('to') or None
        text = (query.get('q') or '').strip().lower()

        identity = self._camera_identity() if (locations or text) else {}
        scope = set(cameras)
        if locations:
            scope.update(cid for cid, meta in identity.items() if meta['location'] in locations)
        scoped = bool(cameras or locations)

        out = []
        for clip in clips:
            if scoped and clip['camera'] not in scope:
                continue
            if species and clip['species'] not in species:
                continue
            # 'Manual' clips have no calendar date; a date range excludes them.
            if date_from and (clip['date'] == 'Manual' or clip['date'] < date_from):
                continue
            if date_to and (clip['date'] == 'Manual' or clip['date'] > date_to):
                continue
            if text:
                meta = identity.get(clip['camera']) or {}
                haystack = (
                    f"{clip['species']} {clip['camera']} {clip['filename']} "
                    f"{meta.get('name', '')} {meta.get('location', '')}"
                ).lower()
                if text not in haystack:
                    continue
            out.append(clip)
        return out

    async def handle_app_shell(self, request):
        """The single HTML document behind every client-side route.

        Deep links work because every app path returns this same shell and the
        router reads location.pathname on boot.
        """
        v = self.asset_version
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Animal Tracker</title>
<link rel="stylesheet" href="/static/tokens.css?v={v}">
<link rel="stylesheet" href="/static/app.css?v={v}">
<script>
  /* Resolve the theme before first paint so a manual choice never flashes the
     other palette. Storage can throw in a private window; the OS preference is
     the fallback and the stylesheet already handles it. */
  try {{
    var t = localStorage.getItem('at:theme');
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  }} catch (e) {{}}
</script>
</head>
<body>
<a class="skiplink" href="#main">Skip to content</a>
<div id="app" class="shell" aria-busy="true"></div>
<script type="module" src="/static/app.js?v={v}"></script>
<noscript>
  <p style="padding:24px">Animal Tracker needs JavaScript enabled.</p>
</noscript>
</body>
</html>"""
        return web.Response(
            text=html,
            content_type='text/html',
            headers={'Cache-Control': 'no-cache'},
        )

    async def handle_recordings_api(self, request):
        """GET /api/recordings — filtered, sorted, paginated clip list plus facets."""
        q = dict(request.query)
        try:
            limit = max(1, min(500, int(q.get('limit', 60))))
            offset = max(0, int(q.get('offset', 0)))
        except ValueError:
            return web.json_response({'error': 'limit and offset must be integers'}, status=400)

        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings_cached)
        matched = self._filter_clips(clips, q)

        sort = q.get('sort', 'newest')
        if sort == 'oldest':
            matched = sorted(matched, key=lambda c: c['time'])
        elif sort == 'species':
            matched = sorted(matched, key=lambda c: (c['species'].lower(), -c['time'].timestamp()))
        elif sort == 'camera':
            matched = sorted(matched, key=lambda c: (c['camera'], -c['time'].timestamp()))
        elif sort == 'largest':
            matched = sorted(matched, key=lambda c: -c['size'])
        # 'newest' is the order _scan_recordings already returns.

        # Facets describe the whole match, not the current page, so counts stay
        # stable while paging. Each camera carries its configured name and
        # location so the client can group the chips by place; the location
        # facet counts only clips from a camera that has one.
        identity = self._camera_identity()
        cameras: Dict[str, int] = {}
        locations: Dict[str, int] = {}
        species: Dict[str, int] = {}
        for clip in matched:
            cam = clip['camera']
            cameras[cam] = cameras.get(cam, 0) + 1
            species[clip['species']] = species.get(clip['species'], 0) + 1
            location = (identity.get(cam) or {}).get('location') or ''
            if location:
                locations[location] = locations.get(location, 0) + 1

        page = matched[offset:offset + limit]
        return web.json_response({
            'clips': [self._clip_to_json(c, identity) for c in page],
            'total': len(matched),
            'archive_total': len(clips),
            'offset': offset,
            'limit': limit,
            'has_more': offset + limit < len(matched),
            'facets': {
                'cameras': sorted(
                    ({'value': k, 'count': v,
                      'name': (identity.get(k) or {}).get('name') or k,
                      'location': (identity.get(k) or {}).get('location') or ''}
                     for k, v in cameras.items()),
                    key=lambda x: x['value'],
                ),
                'locations': sorted(
                    ({'value': k, 'count': v} for k, v in locations.items()),
                    key=lambda x: x['value'],
                ),
                'species': sorted(
                    ({'value': k, 'count': v} for k, v in species.items()),
                    key=lambda x: (-x['count'], x['value']),
                ),
            },
        })

    async def handle_clip_api(self, request):
        """GET /api/clip/{path} — detail for one clip, including track timing."""
        rel_path = request.match_info.get('path', '')
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, self._get_clip_detail, rel_path)
        if info is None:
            renamed_to = await loop.run_in_executor(None, self._renamed_clip, rel_path)
            body = {'error': 'Clip not found'}
            if renamed_to:
                body['renamed_to'] = renamed_to
            return web.json_response(body, status=404)

        payload = dict(info)
        meta = self._camera_identity().get(info.get('camera')) or {}
        payload['camera_name'] = meta.get('name') or info.get('camera')
        payload['location'] = meta.get('location') or ''
        payload['time'] = info['time'].isoformat()
        payload['epoch'] = info['time'].timestamp()
        payload['url'] = f"/clips/{rel_path}"
        analysis = self._analysis_state(rel_path, info.get('unfinished'))
        payload['reprocessing'] = rel_path in self.reprocessing_jobs or analysis == 'running'
        if analysis:
            payload['analysis'] = analysis
        payload.pop('unfinished', None)
        return web.json_response(payload)

    async def handle_cameras_api(self, request):
        """GET /api/cameras — identity plus live health for the shell chrome."""
        now = _time.time()
        cameras = []
        for cam_id, worker in self.workers.items():
            last_ts = float(getattr(worker, 'latest_frame_ts', 0.0) or 0.0)
            age = (now - last_ts) if last_ts > 0 else None
            connected = bool(getattr(worker, 'stream_connected', False))
            has_frame = getattr(worker, 'latest_frame', None) is not None
            if not has_frame:
                state = 'offline'
            elif not connected or (age is not None and age > STALE_AFTER_SECONDS):
                state = 'stale'
            else:
                state = 'live'
            ptz = getattr(worker.camera, 'ptz_tracking', None)
            cameras.append({
                'id': cam_id,
                'name': getattr(worker.camera, 'name', cam_id),
                'location': getattr(worker.camera, 'location', ''),
                'state': state,
                'frame_age': round(age, 1) if age is not None else None,
                'stream_url': f'/stream/{cam_id}',
                'snapshot_url': f'/snapshot/{cam_id}',
                'has_ptz': bool(
                    getattr(worker, 'onvif_client', None)
                    and getattr(worker, 'onvif_profile_token', None)
                ),
                'has_tracker': getattr(worker, 'ptz_tracker', None) is not None,
                'ptz_target': getattr(ptz, 'target_camera_id', None) if ptz else None,
            })
        return web.json_response({'cameras': cameras, 'timezone': TIMEZONE_DISPLAY})

    async def handle_calendar_api(self, request):
        """GET /api/recordings/calendar - Returns full calendar structure as JSON"""
        loop = asyncio.get_running_loop()
        clips = await loop.run_in_executor(None, self._scan_recordings_cached)
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
        clips = await loop.run_in_executor(None, self._scan_recordings_cached)
        day_data = self._get_clips_for_date(clips, date_str, camera, species)
        
        return web.json_response(day_data)

    async def handle_snapshot(self, request):
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)
        
        if not worker:
            return web.Response(status=404, text="Camera not found")

        # Determine stream health: stale if no fresh frame in the last 10s,
        # or if the capture has been disconnected and we have no frame at all.
        now = _time.time()
        last_ts = float(getattr(worker, 'latest_frame_ts', 0.0) or 0.0)
        stream_connected = bool(getattr(worker, 'stream_connected', False))
        age = now - last_ts if last_ts > 0 else float('inf')
        is_stale = (not stream_connected) or age > STALE_AFTER_SECONDS

        # If we have no frame at all, return a generated "STREAM DOWN" placeholder
        if worker.latest_frame is None:
            placeholder = self._render_stream_down_placeholder(camera_id, age=age)
            return web.Response(
                body=placeholder,
                content_type='image/jpeg',
                headers={
                    'X-Stream-Status': 'down',
                    'X-Frame-Age-Seconds': 'inf',
                    'Cache-Control': 'no-store',
                },
            )

        # Offload image processing to thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        
        # Get current detections for overlay
        detections = getattr(worker, 'latest_detections', []) or []
        
        success, buffer = await loop.run_in_executor(
            None, self._render_frame_jpeg, worker.latest_frame.copy(), detections, is_stale, age,
        )

        if not success:
            return web.Response(status=500, text="Failed to encode frame")

        headers = {
            'X-Stream-Status': 'down' if is_stale else 'ok',
            'X-Frame-Age-Seconds': f"{age:.1f}" if age != float('inf') else 'inf',
            'Cache-Control': 'no-store',
        }
        return web.Response(body=buffer.tobytes(), content_type='image/jpeg', headers=headers)

    def _render_frame_jpeg(self, img, detections, stale, age_seconds, quality=70):
        """Draw overlays, downscale to 640px wide, encode JPEG.

        Shared by the single-shot /snapshot handler and the /stream MJPEG
        handler so both render identically. Pure function of its arguments;
        safe to call in a worker thread.
        """
        height, width = img.shape[:2]

        # Draw bounding boxes only if the stream is live (otherwise the boxes
        # would be drawn on a stale frame and look misleading).
        if not stale:
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

        # When the stream is stale, dim the frame and overlay a clear banner
        # so the operator immediately sees there is no live video.
        if stale:
            # Dim the underlying (stale) image
            img = cv2.addWeighted(img, 0.45, np.zeros_like(img), 0.0, 0)

            banner_color = (0, 0, 220)  # red (BGR)
            if age_seconds == float('inf') or age_seconds > 86400:
                age_text = "no frames received"
            elif age_seconds < 60:
                age_text = f"last frame {int(age_seconds)}s ago"
            elif age_seconds < 3600:
                age_text = f"last frame {int(age_seconds // 60)}m ago"
            else:
                age_text = f"last frame {int(age_seconds // 3600)}h ago"

            title = "STREAM DOWN"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Title sized roughly proportional to image width
            title_scale = max(0.9, width / 700.0)
            title_thickness = max(2, int(title_scale * 2))
            (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thickness)

            sub_scale = max(0.5, title_scale * 0.55)
            sub_thickness = max(1, int(sub_scale * 2))
            (sw, sh), _ = cv2.getTextSize(age_text, font, sub_scale, sub_thickness)

            pad = int(20 * title_scale)
            box_w = max(tw, sw) + pad * 2
            box_h = th + sh + pad * 3
            x0 = (width - box_w) // 2
            y0 = (height - box_h) // 2

            # Solid banner background
            cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), banner_color, -1)
            cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 2)

            cv2.putText(
                img, title,
                (x0 + (box_w - tw) // 2, y0 + pad + th),
                font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA,
            )
            cv2.putText(
                img, age_text,
                (x0 + (box_w - sw) // 2, y0 + pad * 2 + th + sh),
                font, sub_scale, (255, 255, 255), sub_thickness, cv2.LINE_AA,
            )

        # Resize for web display
        if width > 640:
            scale = 640 / width
            new_height = int(height * scale)
            img = cv2.resize(img, (640, new_height), interpolation=cv2.INTER_AREA)

        # Encode frame to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        return cv2.imencode('.jpg', img, encode_param)

    async def _write_mjpeg_part(self, response, boundary: str, payload: bytes) -> None:
        """Write one multipart/x-mixed-replace part."""
        header = (
            f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n"
        ).encode('ascii')
        await response.write(header + payload + b"\r\n")

    async def handle_stream(self, request):
        """MJPEG stream for a camera: multipart/x-mixed-replace.

        Replaces the old 2-second snapshot polling on /live. The browser holds
        one connection open and frames are pushed as the capture loop produces
        them, so the view runs at the camera's real rate instead of 0.5 fps.

        ``latest_frame`` is updated on every decoded frame (pipeline.py, before
        the frame_skip gate), so this is independent of the detector's rate.
        """
        camera_id = request.match_info['camera_id']
        worker = self.workers.get(camera_id)

        if not worker:
            return web.Response(status=404, text="Camera not found")

        boundary = 'frame'
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': f'multipart/x-mixed-replace; boundary={boundary}',
                'Cache-Control': 'no-store, no-cache, must-revalidate',
                'Pragma': 'no-cache',
            },
        )
        await response.prepare(request)

        loop = asyncio.get_running_loop()
        last_sent_ts = 0.0
        # Bound the encode load: every part costs a resize + JPEG encode in a
        # worker thread, and the capture loop can run well above this.
        MAX_FPS = 15.0
        min_interval = 1.0 / MAX_FPS
        idle_poll = 0.05

        try:
            while True:
                now = _time.time()
                last_ts = float(getattr(worker, 'latest_frame_ts', 0.0) or 0.0)
                stream_connected = bool(getattr(worker, 'stream_connected', False))
                age = now - last_ts if last_ts > 0 else float('inf')
                is_stale = (not stream_connected) or age > STALE_AFTER_SECONDS

                frame = worker.latest_frame

                if frame is None:
                    # No frame at all: send the placeholder slowly rather than
                    # hot-looping on a camera that may never come up.
                    payload = self._render_stream_down_placeholder(camera_id, age=age)
                    await self._write_mjpeg_part(response, boundary, payload)
                    await asyncio.sleep(1.0)
                    continue

                # Nothing new to send. Stale frames still get resent so the
                # STREAM DOWN banner's age counter keeps ticking.
                if last_ts <= last_sent_ts and not is_stale:
                    await asyncio.sleep(idle_poll)
                    continue

                detections = getattr(worker, 'latest_detections', []) or []
                success, buffer = await loop.run_in_executor(
                    None, self._render_frame_jpeg,
                    frame.copy(), detections, is_stale, age,
                )
                if not success:
                    await asyncio.sleep(idle_poll)
                    continue

                await self._write_mjpeg_part(response, boundary, buffer.tobytes())
                last_sent_ts = last_ts
                await asyncio.sleep(min_interval if not is_stale else 1.0)

        except (ConnectionResetError, ConnectionAbortedError, asyncio.CancelledError):
            pass  # viewer closed the tab / navigated away
        except Exception as e:  # noqa: BLE001
            LOGGER.debug("MJPEG stream for %s ended: %s", camera_id, e)
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass

        return response

    def _render_stream_down_placeholder(self, camera_id: str, age: float = float('inf')) -> bytes:
        """Generate a black JPEG with a STREAM DOWN banner for a camera with no frame."""
        width, height = 640, 360
        img = np.zeros((height, width, 3), dtype=np.uint8)

        title = "STREAM DOWN"
        sub = f"camera: {camera_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 1.2
        title_thickness = 3
        (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
        sub_scale = 0.6
        sub_thickness = 2
        (sw, sh), _ = cv2.getTextSize(sub, font, sub_scale, sub_thickness)

        pad = 24
        box_w = max(tw, sw) + pad * 2
        box_h = th + sh + pad * 3
        x0 = (width - box_w) // 2
        y0 = (height - box_h) // 2
        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 220), -1)
        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 2)
        cv2.putText(img, title, (x0 + (box_w - tw) // 2, y0 + pad + th),
                    font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
        cv2.putText(img, sub, (x0 + (box_w - sw) // 2, y0 + pad * 2 + th + sh),
                    font, sub_scale, (255, 255, 255), sub_thickness, cv2.LINE_AA)

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b''

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
        self._invalidate_scan_cache()
        # Resolve and require containment in the clips directory. A substring test for
        # '..' both misses encodings it should catch and rejects legitimate names that
        # merely contain two dots. Both sides are resolved so a symlinked storage root
        # (e.g. an SSD mount) compares correctly.
        clips_root = (self.storage_root / 'clips').resolve()
        try:
            file_path = (clips_root / rel_path).resolve()
            file_path.relative_to(clips_root)
        except (ValueError, OSError):
            return False, "Invalid path"
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
        
        # Prevent duplicate processing - check if already in progress, here
        # or in the pipeline (a live event or the recovery sweep).
        job_key = clip_path
        registry = self.analysis_registry
        if job_key in self.reprocessing_jobs:
            existing = self.reprocessing_jobs[job_key]
            LOGGER.warning("Duplicate reprocess request for %s, already started at %s", 
                          clip_path, existing.get('started'))
            return web.json_response({
                'success': False,
                'error': f"Processing already in progress (started {existing.get('started')})"
            }, status=409)  # Conflict
        if registry is not None and not registry.begin(full_path, source='reanalyze'):
            LOGGER.warning("Reprocess request for %s while the pipeline is analysing it (%s)",
                           clip_path, registry.source_of(full_path))
            return web.json_response({
                'success': False,
                'error': "The pipeline is already analysing this clip; its result will appear shortly"
            }, status=409)  # Conflict
        
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
                if registry is not None:
                    registry.end(full_path)
                return web.Response(status=500, text="No workers available")
            detector = next(iter(self.workers.values())).detector
            LOGGER.warning("No detector config, falling back to worker detector: %s", detector.backend_name)
        
        # Track this reprocessing job
        self.reprocessing_jobs[job_key] = {
            'started': datetime.now(tz=CENTRAL_TZ).isoformat(),
            'clip_name': full_path.stem,
            'camera': full_path.parent.name,
        }
        
        # Run reprocessing in thread pool
        loop = asyncio.get_running_loop()
        
        # Settings from config plus the request's overrides, by the same
        # mapping the live path and the recovery sweep use.
        from .postprocess import build_processing_settings
        clip_cfg = self.runtime.general.clip if self.runtime else None
        settings = build_processing_settings(clip_cfg, settings_override)
        
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
        
        try:
            result = await loop.run_in_executor(None, do_reprocess)
        finally:
            # Remove from active jobs
            self.reprocessing_jobs.pop(job_key, None)
            if registry is not None:
                registry.end(full_path)
        
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

    def _get_clip_detail(self, rel_path: str) -> dict | None:
        """Get detailed information about a specific clip."""
        import json
        
        clips_dir = self.storage_root / 'clips'
        clip_path = clips_dir / rel_path
        
        if not clip_path.exists() or not clip_path.is_file():
            return None
        
        stat = clip_path.stat()
        species_display, raw_species = self._parse_species_from_filename(clip_path.name)
        # The sidecar names the thumbnails and carries the track timing, so
        # it is read once and shared with the thumbnail lookup.
        log_data = self._read_sidecar(clip_path)
        thumbnails = self._get_thumbnails_for_clip(clip_path, log_data=log_data)

        track_info = {}
        video_fps = 15.0  # Default
        tracks_by_index = {}  # Index -> track data
        tracks_by_species = {}  # Fallback for old-format thumbnails

        if log_data:
            try:
                # Get video FPS for time calculation
                if log_data.get('video', {}).get('fps'):
                    video_fps = log_data['video']['fps']
                
                # Build track info - now keyed by track INDEX (order) not species
                # This preserves individual track timing even for same-species tracks
                tracking_summary = log_data.get('tracking_summary', {})
                
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
            'species': species_display,
            'raw_species': raw_species,
            'time': clip_start_time(clip_path, stat),
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'thumbnails': thumbnails,
            'fps': video_fps,
            'global_settings': global_settings,
            'unfinished': is_unclassified_clip(clip_path) and not sidecar_path(clip_path).exists(),
        }

    def _renamed_clip(self, rel_path: str) -> 'str | None':
        """Where a clip that is no longer at ``rel_path`` went, if it was renamed.

        Post-processing renames ``<epoch>_animal.mp4`` to the species it
        found, in place. A page that was watching the old name can follow it
        when exactly one clip with that epoch remains in the directory.
        """
        clip_path = self.storage_root / 'clips' / rel_path
        if clip_path.suffix.lower() != '.mp4' or not is_unclassified_clip(clip_path):
            return None
        epoch = clip_path.stem.split('_', 1)[0]
        try:
            matches = [p for p in clip_path.parent.glob(f"{epoch}_*.mp4") if p.name != clip_path.name]
        except OSError:
            return None
        if len(matches) != 1:
            return None
        return str(matches[0].relative_to(self.storage_root / 'clips'))

    def _monitor_host_stats(self):
        """System, GPU, detector and recent-clip figures for /api/monitor.

        Runs on an executor thread, never on the event loop: the disk
        figure is a statvfs on the storage root and the recent-clip scan
        lists it, and on the production host that root is an NFS mount.
        Done on the loop, a slow NFS server froze every camera worker and
        every request in this process for as long as it took to answer.
        Returns ``(system, gpu, detector_info, recent_clips)``.
        """
        import psutil

        # System stats
        try:
            # interval=None compares with the previous call instead of
            # sleeping 100 ms; the first call after startup reads 0.
            cpu_percent = psutil.cpu_percent(interval=None)
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
            # Get location from config, not detector object
            detector_cfg = worker.runtime.general.detector
            country = getattr(detector_cfg, 'country', None) or ''
            region = getattr(detector_cfg, 'admin1_region', None) or ''
            if country and region:
                detector_info['country'] = f"{country} {region}"
            elif country:
                detector_info['country'] = country
        
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
        
        return system, gpu, detector_info, recent_clips

    async def handle_get_monitor_data(self, request):
        """Get real-time pipeline monitoring data as JSON."""
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
        
        # Host figures are gathered off the loop (see _monitor_host_stats) on
        # a thread of their own: a stalled NFS call must not take one of the
        # default pool's threads, which the camera capture loops run on, and
        # psutil keeps its "since the last call" CPU sample per thread, so
        # the figure is only meaningful when every poll lands on the same one.
        executor = getattr(self, '_stats_executor', None)
        if executor is None:
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='monitor-stats')
            self._stats_executor = executor
        loop = asyncio.get_running_loop()
        try:
            system, gpu, detector_info, recent_clips = await loop.run_in_executor(
                executor, self._monitor_host_stats
            )
        except Exception as err:  # noqa: BLE001 - telemetry is best effort
            LOGGER.debug("Host stats unavailable: %s", err)
            system, gpu, detector_info, recent_clips = (
                dict(_EMPTY_SYSTEM_STATS), dict(_EMPTY_GPU_STATS), {'backend': 'unknown', 'country': None}, []
            )

        # Active reprocessing jobs
        reprocessing = list(self.reprocessing_jobs.values())
        analysis_active = self.analysis_registry.active() if self.analysis_registry is not None else []
        recovery = None
        if self.recovery is not None:
            try:
                recovery = self.recovery.status()
            except Exception:  # noqa: BLE001 - telemetry is best effort
                recovery = None
        
        return web.json_response({
            'timestamp': datetime.now(tz=CENTRAL_TZ).isoformat(),
            'cameras': cameras,
            'system': system,
            'gpu': gpu,
            'detector': detector_info,
            'recent_clips': recent_clips,
            'reprocessing_jobs': reprocessing,
            'analysis_active': analysis_active,
            'recovery': recovery,
        })

    async def handle_get_logs(self, request):
        """Get recent logs from journalctl or log files."""
        from collections import deque

        # Get query params
        camera_id = request.query.get('camera', None)
        minutes = int(request.query.get('minutes', 30))
        level = request.query.get('level', 'all')  # all, error, warning
        log_type = request.query.get('type', 'all')  # all, no-http, detection, tracking, events, clips, errors
        limit = min(int(request.query.get('limit', 200)), 2000)  # Cap at 2000 to prevent memory issues

        # Custom time range support (overrides minutes if provided)
        start_time = request.query.get('start', None)  # ISO format: 2026-01-17T10:00
        end_time = request.query.get('end', None)  # ISO format: 2026-01-17T12:00

        logs = []
        source = 'none'
        error_msg = None
        skipped_count = 0

        # Known camera IDs for message-body camera attribution and filtering
        # (the whole pipeline now runs in a single systemd unit, so we can't
        # rely on the unit name).
        known_camera_ids = list(self.workers.keys()) if hasattr(self, 'workers') else []
        _camera_token_re_cache: dict = {}

        # Parse custom time range if provided
        time_range_start = None
        time_range_end = None
        if start_time and end_time:
            try:
                # Parse ISO format datetime-local (2026-01-17T10:00)
                time_range_start = datetime.fromisoformat(start_time).replace(tzinfo=CENTRAL_TZ)
                time_range_end = datetime.fromisoformat(end_time).replace(tzinfo=CENTRAL_TZ)
            except ValueError as e:
                LOGGER.warning("Invalid time range format: %s", e)

        # Try journalctl first (for systemd systems)
        try:
            # Build journalctl command
            cmd = [
                'journalctl',
                '--no-pager',
                '-o', 'json',
            ]

            # Add time range - either custom or relative
            relative_window = False
            if time_range_start and time_range_end:
                # For custom time range, fetch ALL logs in the range (no -n limit)
                # We'll apply the limit after filtering
                cmd.extend(['--since', time_range_start.strftime('%Y-%m-%d %H:%M:%S')])
                cmd.extend(['--until', time_range_end.strftime('%Y-%m-%d %H:%M:%S')])
            else:
                relative_window = True
                cmd.extend(['--since', f'{minutes} minutes ago'])

            # Single unified service runs the whole pipeline. Per-camera
            # filtering is done via message substring match below since all
            # cameras log to the same systemd unit.
            cmd.extend(['-u', 'animaltracker.service'])

            # Push the type/level filter down into journalctl (a superset of
            # what the Python filters accept), so that with a relative window
            # -n counts matching entries and a sparse type is not crowded out
            # by the periodic lines around it.
            grep = _journal_grep_pattern(log_type, level)
            plain_cmd = list(cmd)
            if grep:
                cmd.extend(['--grep', grep, '--case-sensitive=false'])
            if relative_window:
                if grep:
                    fetch_limit = max(limit * 2, 1000)
                elif log_type == 'all':
                    fetch_limit = max(limit * 2, 500)
                else:
                    fetch_limit = max(limit * 4, 2000)
                cmd.extend(['-n', str(fetch_limit)])
                plain_cmd.extend(['-n', str(max(limit * 4, 2000))])

            async def _run_journalctl(argv):
                LOGGER.debug("Running journalctl: %s", ' '.join(argv))
                # Run journalctl asynchronously so we don't block the event loop
                # while it executes (can take seconds on busy systems).
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    raise
                code = proc.returncode if proc.returncode is not None else -1
                return (code, out_b.decode('utf-8', errors='replace'),
                        err_b.decode('utf-8', errors='replace'))

            returncode, stdout, stderr = await _run_journalctl(cmd)
            if returncode != 0 and grep:
                # A journalctl built without pattern-matching support rejects
                # --grep: fetch more and let the Python filters do the work.
                LOGGER.debug("journalctl --grep failed (%s); retrying without it",
                             stderr.strip()[:200])
                returncode, stdout, stderr = await _run_journalctl(plain_cmd)

            LOGGER.debug("journalctl returned: code=%d, stdout_len=%d, stderr=%s",
                        returncode, len(stdout), stderr[:200] if stderr else '')

            if returncode == 0 and stdout.strip():
                source = 'journalctl'
                for line in stdout.strip().split('\n'):
                    if line:
                        try:
                            entry = json.loads(line)
                            # Parse journalctl JSON format
                            timestamp = entry.get('__REALTIME_TIMESTAMP', '')
                            if timestamp:
                                # Convert microseconds to seconds (Unix epoch)
                                ts = int(timestamp) / 1_000_000
                                time_str = datetime.fromtimestamp(ts, tz=CENTRAL_TZ).strftime('%H:%M:%S')
                            else:
                                ts = None
                                time_str = '--:--:--'

                            message = entry.get('MESSAGE', '')
                            if isinstance(message, list):
                                # journalctl emits non-UTF-8 payloads as byte arrays.
                                message = bytes(message).decode('utf-8', errors='replace')
                            try:
                                priority = int(entry.get('PRIORITY', 6))
                            except (TypeError, ValueError):
                                priority = 6

                            log_level, logger_name, body = _classify_log_line(message, priority)

                            # Level filter, from the prefix rather than -p:
                            # pre-prefix entries are all priority 6.
                            if level == 'error' and log_level != 'error':
                                skipped_count += 1
                                continue
                            if level == 'warning' and log_level not in ('error', 'warning'):
                                skipped_count += 1
                                continue

                            # Extract camera ID from the message body. The
                            # whole pipeline now runs in one systemd unit, so
                            # we look for any known camera id appearing as a
                            # whole-word token in the message.
                            cam = ''
                            if known_camera_ids:
                                for cid in known_camera_ids:
                                    if _camera_token_re_cache.setdefault(
                                        cid, re.compile(rf'\b{re.escape(cid)}\b')
                                    ).search(body):
                                        cam = cid
                                        break

                            # Per-camera filter: skip messages that don't
                            # mention the requested camera.
                            if camera_id and cam != camera_id:
                                skipped_count += 1
                                continue

                            # Apply server-side type filter (uses pre-compiled patterns)
                            if _matches_log_filter(body, log_type, logger_name, log_level):
                                logs.append({
                                    'time': time_str,
                                    'timestamp': ts,  # Unix epoch for client-side timezone conversion
                                    'level': log_level,
                                    'logger': logger_name,
                                    'camera': cam,
                                    'message': body,
                                })
                            else:
                                skipped_count += 1
                        except json.JSONDecodeError:
                            continue
            elif returncode != 0:
                error_msg = stderr[:200] if stderr else f'Exit code {returncode}'
                LOGGER.warning("journalctl failed: %s", error_msg)
        except FileNotFoundError:
            error_msg = 'journalctl not found'
        except asyncio.TimeoutError:
            error_msg = 'journalctl timed out'
        except Exception as e:
            error_msg = str(e)
            LOGGER.debug("journalctl failed: %s", e)

        # Also read from log files and merge (not just fallback)
        log_files_found = 0
        if self.logs_root and self.logs_root.exists():
            # Use custom time range or fall back to minutes
            if time_range_start:
                cutoff = time_range_start
                cutoff_end = time_range_end
            else:
                cutoff = datetime.now(tz=CENTRAL_TZ) - timedelta(minutes=minutes)
                cutoff_end = None

            # Helper to parse log timestamps (uses pre-compiled regexes)
            def parse_log_timestamp(line):
                """Try to extract datetime from log line. Returns (datetime, time_str, unix_ts) or (None, time_str, None)."""
                full_match = _TS_FULL_RE.search(line)
                if full_match:
                    try:
                        dt = datetime.strptime(f"{full_match.group(1)} {full_match.group(2)}", '%Y-%m-%d %H:%M:%S')
                        dt = dt.replace(tzinfo=CENTRAL_TZ)
                        return dt, full_match.group(2), dt.timestamp()
                    except ValueError:
                        pass

                # Time-only fallback. Naively combining with "today" misorders
                # entries written shortly before midnight (they look like
                # they're in the future). If the resulting datetime is more
                # than ~1 minute ahead of "now", assume the line was written
                # yesterday.
                time_match = _TS_TIME_RE.search(line)
                if time_match:
                    time_str = time_match.group(1)
                    try:
                        now = datetime.now(tz=CENTRAL_TZ)
                        t = datetime.strptime(time_str, '%H:%M:%S').time()
                        dt = datetime.combine(now.date(), t, tzinfo=CENTRAL_TZ)
                        if dt - now > timedelta(minutes=1):
                            dt -= timedelta(days=1)
                        return dt, time_str, dt.timestamp()
                    except ValueError:
                        return None, time_str, None

                return None, '--:--:--', None

            # App log file only: animaltracker.log is what cli.attach_file_log()
            # writes when stderr is not the systemd journal. The bare '*.log'
            # glob would also match 'web_access.log' (HTTP access log), which
            # can be huge and is pure noise.
            log_patterns = ['animaltracker*.log']
            # For custom time range, read more lines to ensure we capture the full range
            max_lines_per_file = 10000 if time_range_start else 500
            for pattern in log_patterns:
                for log_file in self.logs_root.glob(pattern):
                    log_files_found += 1
                    try:
                        # deque(..., maxlen=N) keeps only the last N lines in
                        # constant memory (vs f.readlines()[-N:] which loads
                        # the whole file).
                        with open(log_file, 'r', errors='replace') as f:
                            lines = deque(f, maxlen=max_lines_per_file)
                        for line in lines:
                            line = line.rstrip('\n').rstrip('\r')
                            if not line:
                                continue

                            log_level, logger_name, body = _classify_log_line(line)

                            # Filter by level
                            if level == 'error' and log_level != 'error':
                                continue
                            if level == 'warning' and log_level not in ('error', 'warning'):
                                continue

                            # Extract and filter by timestamp
                            log_dt, time_str, unix_ts = parse_log_timestamp(line)

                            # Apply time range filter if we could parse the timestamp
                            if log_dt:
                                if log_dt < cutoff:
                                    continue  # Before start time
                                if cutoff_end and log_dt > cutoff_end:
                                    continue  # After end time

                            # Camera attribution from message body
                            cam = ''
                            if known_camera_ids:
                                for cid in known_camera_ids:
                                    if _camera_token_re_cache.setdefault(
                                        cid, re.compile(rf'\b{re.escape(cid)}\b')
                                    ).search(body):
                                        cam = cid
                                        break

                            if camera_id and cam != camera_id:
                                skipped_count += 1
                                continue

                            # Apply server-side type filter
                            if _matches_log_filter(body, log_type, logger_name, log_level):
                                logs.append({
                                    'time': time_str,
                                    'timestamp': unix_ts,  # Unix epoch for client-side timezone conversion
                                    'level': log_level,
                                    'logger': logger_name,
                                    'camera': cam,
                                    'message': body[:500],  # Truncate long lines
                                })
                            else:
                                skipped_count += 1
                    except (OSError, UnicodeDecodeError) as e:
                        LOGGER.warning("Failed to read log file %s: %s", log_file, e)
                        continue

            # Update source based on what we found
            if log_files_found > 0:
                if source == 'journalctl':
                    source = 'journalctl+logfile'
                else:
                    source = 'logfile'
        
        # Sort by timestamp (oldest first), then take last N and reverse for most recent first
        # This ensures we get the most recent logs within the time range
        logs.sort(key=lambda x: x.get('timestamp') or 0)  # Sort by timestamp ascending
        logs = logs[-limit:]  # Keep last N (most recent)
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
            'limit': limit,
            'count': len(logs),
            'skipped': skipped_count,
            'timezone': TIMEZONE_DISPLAY,
            'logs': logs,
        }
        # Include time range info if custom range was used
        if time_range_start and time_range_end:
            response_data['time_range'] = {
                'start': time_range_start.isoformat(),
                'end': time_range_end.isoformat(),
            }
        if error_msg and source in ('none', 'unknown'):
            response_data['error'] = error_msg

        return web.json_response(response_data)

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

    # ------------------------------------------------------------------
    # /api/config — the settings editor's contract (see configstore.py)
    # ------------------------------------------------------------------

    def _stream_state(self, worker) -> dict:
        """The same live/stale/offline verdict /api/cameras gives."""
        now = _time.time()
        last_ts = float(getattr(worker, 'latest_frame_ts', 0.0) or 0.0)
        age = (now - last_ts) if last_ts > 0 else None
        connected = bool(getattr(worker, 'stream_connected', False))
        has_frame = getattr(worker, 'latest_frame', None) is not None
        if not has_frame:
            state = 'offline'
        elif not connected or (age is not None and age > STALE_AFTER_SECONDS):
            state = 'stale'
        else:
            state = 'live'
        return {'state': state, 'frame_age': round(age, 1) if age is not None else None}

    def _config_runtime_annotation(self, camera_id: str):
        """What the running process knows about one configured camera."""
        worker = self.workers.get(camera_id)
        if worker is None:
            return {'running': False}
        info = {'running': True}
        info.update(self._stream_state(worker))
        info['onvif_connected'] = bool(
            getattr(worker, 'onvif_client', None) and getattr(worker, 'onvif_profile_token', None)
        )
        info['profile_token'] = getattr(worker, 'onvif_profile_token', None)
        info['has_tracker'] = getattr(worker, 'ptz_tracker', None) is not None
        return info

    def _config_error_response(self, err: 'configstore.ConfigError', status: int):
        return web.json_response({
            'error': str(err),
            'problems': list(err.problems),
            'config_path': str(self.config_path) if self.config_path else None,
        }, status=status)

    async def handle_get_config(self, request):
        """GET /api/config — the validated file plus runtime annotations."""
        if not self.config_path:
            return web.json_response({'error': 'The server was started without a config path.'}, status=500)
        loop = asyncio.get_running_loop()
        recent = await loop.run_in_executor(None, self._get_recent_detections)
        try:
            payload = await loop.run_in_executor(
                None,
                lambda: configstore.describe(
                    self.config_path, self.runtime, self.workers,
                    self._config_runtime_annotation, recent,
                ),
            )
        except configstore.ConfigError as err:
            return self._config_error_response(err, 500)
        except OSError as err:
            return web.json_response({'error': f'Could not read {self.config_path}: {err}'}, status=500)
        return web.json_response(payload)

    async def handle_save_config(self, request):
        """POST /api/config — merge, validate, back up, write, apply live."""
        if not self.config_path:
            return web.json_response({'error': 'The server was started without a config path.'}, status=500)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON body'}, status=400)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: configstore.save(self.config_path, body, self.runtime, self.workers)
            )
        except configstore.ConfigError as err:
            return self._config_error_response(err, 400)
        except OSError as err:
            LOGGER.error("Failed to write %s: %s", self.config_path, err, exc_info=True)
            return web.json_response({
                'error': f'{self.config_path.name} was not written: {err}',
                'problems': [],
            }, status=500)

        for cam_id, patch in result.get('ptz_state', {}).items():
            try:
                self._update_ptz_state(cam_id, **patch)
            except Exception as err:  # noqa: BLE001 - the config is already saved
                LOGGER.warning("Could not sync ptz_state.json for %s: %s", cam_id, err)

        changes = result['changes']
        if result['written']:
            LOGGER.info(
                "Settings saved to %s (general: %s; cameras: %s; added: %s; removed: %s; live: %d) backup=%s",
                self.config_path, changes['general'], list(changes['cameras'].keys()),
                changes['added'], changes['removed'], len(result['applied_live']), result['backup'],
            )
        reasons = configstore.pending_restart(result['config'], self.runtime, self.workers)
        support = configstore.restart_support()
        return web.json_response({
            'status': 'ok',
            'written': result['written'],
            'backup': result['backup'],
            'applied_live': result['applied_live'],
            'changes': changes,
            'restart': {
                'required': bool(reasons),
                'reasons': reasons,
                'supported': support['supported'],
                'unit': support['unit'],
            },
        })

    async def handle_probe_camera(self, request):
        """POST /api/config/probe — test a stream URI, an ONVIF endpoint, or env names.

        Body: {"rtsp": {"uri", "transport"}, "onvif": {"host", "port",
        "username_env", "password_env"}, "env": ["NAME", ...]} — any subset.
        Secrets never travel: ONVIF credentials are looked up by env name.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON body'}, status=400)
        if not isinstance(body, dict):
            return web.json_response({'error': 'Body must be an object'}, status=400)
        loop = asyncio.get_running_loop()
        out = {}
        tasks = {}
        rtsp = body.get('rtsp')
        if isinstance(rtsp, dict):
            tasks['rtsp'] = loop.run_in_executor(
                None, lambda: configstore.probe_rtsp(str(rtsp.get('uri') or ''), str(rtsp.get('transport') or 'tcp'))
            )
        onvif = body.get('onvif')
        if isinstance(onvif, dict):
            tasks['onvif'] = loop.run_in_executor(
                None, lambda: configstore.probe_onvif(
                    str(onvif.get('host') or ''), onvif.get('port') or 80,
                    str(onvif.get('username_env') or ''), str(onvif.get('password_env') or ''),
                )
            )
        names = body.get('env')
        if isinstance(names, list):
            out['env'] = configstore.env_presence([str(n) for n in names][:50])
        for key, task in tasks.items():
            out[key] = await task
        return web.json_response(out)

    async def handle_restart(self, request):
        """POST /api/system/restart — restart the service through systemd."""
        support = configstore.restart_support()
        if not support['supported']:
            return web.json_response({
                'error': 'Not running under systemd, so the process cannot restart itself. '
                         'Restart it by hand (for example: sudo systemctl restart animaltracker).',
                'supported': False,
            }, status=501)
        unit = support['unit']
        LOGGER.warning("Restart requested from the settings page; restarting %s", unit)

        def _go():
            configstore.restart_now(unit)

        loop = asyncio.get_running_loop()
        # Answer first so the client sees the acknowledgement, then let go.
        loop.call_later(0.5, lambda: threading.Thread(target=_go, name='service-restart', daemon=True).start())
        return web.json_response({'status': 'restarting', 'unit': unit})

    async def handle_set_secret(self, request):
        """POST /api/secrets — write one variable into config/secrets.env and this process.

        Body: ``{"name": "PUSHOVER_USER_KEY_X", "value": "..."}``; an empty
        or missing value removes the variable. Only names the saved
        configuration references are accepted (configstore.set_secret), and
        the value is never logged or echoed: the response carries the name,
        whether it is now set, and where the backup went.
        """
        if not self.config_path:
            return web.json_response({'error': 'The server was started without a config path.'}, status=500)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON body'}, status=400)
        if not isinstance(body, dict):
            return web.json_response({'error': 'Body must be an object', 'problems': []}, status=400)
        name = str(body.get('name') or '').strip()
        value = body.get('value')
        if value is not None and not isinstance(value, str):
            return web.json_response({'error': 'The value must be a string.', 'problems': []}, status=400)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: configstore.set_secret(self.config_path, name, value)
            )
        except configstore.ConfigError as err:
            return self._config_error_response(err, 400)
        except OSError as err:
            LOGGER.error("The secrets file was not written for %s: %s", name or '?', err)
            return web.json_response({
                'error': f'The secrets file was not written: {err}',
                'problems': [],
            }, status=500)
        result['status'] = 'ok'
        result['env'] = configstore.env_presence([name])
        return web.json_response(result)

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
            LOGGER.error(f"Failed to save PTZ state: {e}", exc_info=True)
    
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
        
        # aiohttp's access line already carries the request time, so no
        # second timestamp. Rotate in-process so installs without logrotate
        # (macOS, Windows) don't grow this file forever; same 50 MB x 4
        # policy as systemd/animaltracker-logrotate.
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=4, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        access_logger.addHandler(handler)

        runner = web.AppRunner(self.app, access_log=access_logger)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        LOGGER.info(f"Web server started on http://0.0.0.0:{self.port}")
        LOGGER.info(f"Server timezone: {TIMEZONE_DISPLAY}")
        
        # Keep the server running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour, repeat forever
        except asyncio.CancelledError:
            LOGGER.info("Web server shutting down...")
            await runner.cleanup()
