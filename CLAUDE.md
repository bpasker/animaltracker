# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered multi-camera wildlife detection system using YOLO/SpeciesNet + RTSP + ONVIF for animal monitoring with PTZ camera tracking.

## Common Commands

```bash
# Run detection pipeline
python -m animaltracker.cli --config config/cameras.yml run

# Run specific camera only
python -m animaltracker.cli --config config/cameras.yml run --camera cam1

# Enable PTZ debug logging
python -m animaltracker.cli --config config/cameras.yml run --ptz-debug

# ONVIF camera discovery (find RTSP URLs, PTZ profiles)
python -m animaltracker.cli --config config/cameras.yml discover --inspect --presets

# Test PTZ controls
python -m animaltracker.cli --config config/cameras.yml ptz-test --camera cam1 --find-working

# Reprocess clips with SpeciesNet
python -m animaltracker.cli --config config/cameras.yml reprocess --camera cam1

# Cleanup old clips (preview first)
python -m animaltracker.cli --config config/cameras.yml cleanup --dry-run

# Calibrate zoom FOV mapping between cam1 (wide) and cam2 (zoom)
python -m animaltracker.cli --config config/cameras.yml zoom-calibrate -w cam1 -z cam2

# Install in dev mode
pip install -e .
```

## Architecture

### Split-Model Detection Pipeline

The system uses a two-stage detection approach:

1. **Real-time detection** (pipeline.py → StreamWorker): Fast MegaDetector/YOLO (~50-150ms) for immediate PTZ tracking triggers
2. **Post-processing** (postprocess.py → ClipPostProcessor): Accurate SpeciesNet (~200-500ms) for species classification on saved clips

### Core Components

- `cli.py` - Entry point, command parsing
- `pipeline.py` - `PipelineOrchestrator` manages all cameras, spawns `StreamWorker` per camera
  Each worker runs under `_run_worker_supervised`: an unexpected error in one
  camera's `run()` is logged, its open event is closed, and that worker
  restarts after a growing pause while the others carry on. Never gather
  `worker.run()` bare; one camera's exception then ends the whole process.
  The web server is deliberately not supervised (a second instance must fail).
  The live `ObjectTracker` is ticked through `_tick_live_tracker`, which
  forgets tracks ByteTrack has given up on, but only between events.
- `detector.py` - Detection backends: MegaDetector, YOLO, SpeciesNet
- `tracker.py` - ByteTrack object tracking with persistent IDs
- `postprocess.py` - Clip post-analysis, track merging, species finalization
  `PostProcessResult.detection_frames` is what `clip.min_detection_frames` is
  measured against: sampled frames with an accepted detection that is neither
  a person's shadow nor part of a track dropped as one (`_evidence_frames`).
  The live path's false-positive gate must use it, never a recount of the
  processing log, which still holds the shadow frames.
- `ptz_tracker.py` - PTZ auto-tracking controller, pixel-to-PTZ coordinate mapping
  `PTZTracker._lock` is held for the whole of an update, ONVIF requests
  included, and cam1 and cam2 share one tracker. Nothing the event loop
  calls may wait for it: call lock-taking methods through `run_in_executor`.
  The decision log has its own `_decision_lock`, never held across I/O, so
  the pipeline can read and trim it on the loop, and `clear_lock()` never
  blocks (a contended reset is done by the next update before it looks at a
  detection).
  Two guards against false locks: a first lock on a detection ByteTrack has
  not confirmed needs 0.75 until the episode has locked (`_episode_locked`;
  never test `_mode` for this, every update path sets TRACKING before it
  selects), and the static-target watchdog remembers the spot it released
  (`_static_rejects`, per camera, ten minutes) so the blob is dropped before
  it counts as a sighting and patrol can resume.
  `_do_patrol` re-issues its sweep only when the cached `_patrol_velocity`
  differs, so every way into PATROL, and every Stop the tracker issues, must
  clear that cache or the camera sits still until the 90 s reversal.
- `analysis_recovery.py` - `ClipAnalysisRegistry` (every clip analysis in
  flight: live event, reanalysis, recovery) and `RecoverySweeper`, a daemon
  thread that finishes the post-processing a restart interrupted: shortly
  after startup and every 30 min it finds `<epoch>_animal.mp4` clips with
  no analysis sidecar and runs them, newest first, one at a time, only while
  no live event needs the post-processor; no alert, no false-positive delete.
  "No analysis sidecar" is `has_analysis_sidecar`: no `.log.json`, or one
  holding only the PTZ decisions the pipeline parks there, which is what a
  failed analysis leaves on a PTZ camera. A job whose detector raised on half
  or more of the sampled frames returns `success=False` and writes nothing,
  so it lands here; a result with any `inference_errors` is never grounds
  for the false-positive delete.
  The API reports `analysis: running|queued|unfinished` for such clips and
  the archive shows "Analyzing…" / "Awaiting analysis" / "Not analyzed"
  instead of "No frame". `clip.recover_unfinished_clips` switches it off.
- `storage.py` - `StorageManager` and `StreamingClipWriter`. An event records
  to `logs_root/event_temp/<camera>_<epoch>_<tag>.temp.avi`, the only copy
  until it is transcoded. `StreamWorker._save_event_clip` does that as soon as
  the event closes, *before* the job waits for a post-processing slot, so a
  restart costs the analysis (which the sweep redoes) and never the clip.
  The finalize job also drains the writer (`writer.close()`); the coroutine
  that closes the event must never wait for it, because it runs inside the
  camera's inference task and the read loop drops every frame until that
  task ends.
  Once the clip is saved, the analysis and the alert are queued on
  `StreamWorker._analysis_workers` (`AnalysisWorkers`: daemon threads, one
  per post-processing slot). Never leave a job waiting for a slot on the
  event loop's default executor: that pool is `cpu_count + 4` threads and
  every camera read, inference call and web request needs one.
  Building a `StorageManager` touches nothing there (the `cleanup` command
  builds one next to the running service). At startup the pipeline turns
  whatever a previous run left behind into `<epoch>_animal.mp4` clips
  (`recover_orphan_event_temp`), which the recovery sweep then analyses.
- `species_names.py` - display names, and the one specificity scale:
  `species_lineage` / `species_rank` read a label's taxonomy depth (0 animal,
  1 class, 2 order, 3 family) and `pick_species_by_lineage` is the vote every
  species choice goes through: a specific label beats its own generic
  ancestors, labels that contradict each other are settled by their votes.
  `TrackInfo`, `ObjectTracker` and the post-processor all rank through these;
  never add a keyword list of families or orders, an unlisted family then
  scores differently from a listed one at the same level.
- `ptz_calibration.py` - Auto-calibration via ORB feature matching between wide/zoom frames
- `onvif_client.py` - ONVIF camera control and discovery
- `web.py` - aiohttp JSON API, MJPEG/snapshot streams, PTZ control and the shell of the client-side app (which lives in `static/`, served at `/app`)
  Every handler that takes a clip path from a request resolves it through
  `_confined_clip_path`, which refuses anything that leaves `clips/` (an
  absolute path, a climbing `..`, a symlink out); never join a request path
  onto the clips directory directly or test it for the substring `..`.
  A reanalysis (`POST /recordings/reprocess`) runs on the pipeline's cached
  post-processing detector (`_postprocess_detector_for`), fetched on the
  executor thread; never build a detector in a handler, the model load
  stalls every camera. It claims the clip in the analysis registry only once
  nothing before the `try` can fail.
- `configstore.py` - the settings editor's transaction on `config/cameras.yml`:
  read + validate, deep-merge the managed keys, back up, atomic write, apply
  live-safe fields to the running process, diff file vs. runtime for the
  pending-restart banner, RTSP/ONVIF probes, systemd restart

### Multi-Camera PTZ Tracking

Cam1 (wide-angle) detects animals and controls cam2 (zoom) PTZ movements. With `multi_camera_tracking` enabled (default), cam2 can take over tracking once the object is in its frame for finer control:

```yaml
cameras:
  - id: cam1
    ptz_tracking:
      enabled: true
      target_camera_id: cam2          # cam1 detections drive cam2's PTZ
      multi_camera_tracking: true     # cam2 can take over tracking (default: true)
  - id: cam2
    ptz_tracking:
      self_track: true                # Allows cam2 to contribute to tracking
```

**How it works:**
1. cam1 (wide-angle) detects an object and moves cam2's PTZ to point at it
2. Once the object is in cam2's frame, cam2 detects it too
3. cam2's detections take priority for fine tracking (since they show exactly where the object is in cam2's view)
4. If cam2 loses the object but cam1 still sees it, cam1 repositions the PTZ

Both cameras share pan/tilt hardware but cam2 has zoom control. The `PTZTracker` converts pixel coordinates from cam1's frame to PTZ commands for cam2, and uses `update_multi_camera()` when both cameras contribute detections.

### Key Data Flow

```
RTSP Stream → StreamWorker → Real-time Detector → ObjectTracker → PTZTracker
                                                          ↓
                                                    ClipBuffer
                                                          ↓
                                               Save MP4 → PostProcessor Queue
                                                                    ↓
                                              ClipPostProcessor → SpeciesNet → Rename/Notify
```

## Configuration

- `config/cameras.yml` - Camera RTSP URIs, detection thresholds, PTZ settings
- `config/secrets.env` - ONVIF credentials, Pushover tokens, Kaggle API keys
- `config/backups/` - the last 20 versions of `cameras.yml` written by the
  settings page (gitignored)

### Settings page (`/app/settings`)

`static/views/settings.js` talks to `GET/POST /api/config`,
`POST /api/config/probe` and `POST /api/system/restart` (all in `web.py`,
backed by `configstore.py`). Rules that keep it safe:

- The file is the source of truth: GET returns the validated file with every
  schema default filled in, plus runtime annotations per camera.
- `GENERAL_FIELDS` / `CAMERA_FIELDS` in `configstore.py` list the managed keys
  and whether the pipeline reads each one live (`True`) or at startup
  (`False`). The client inventory (`GENERAL_SECTIONS` / `CAMERA_GROUPS` in
  `settings.js`) must carry the matching `restart` flag; adding a field means
  touching both lists.
- POST deep-merges into the parsed file, validates the result with
  `RuntimeConfig`, backs up, writes atomically, then `setattr`s live fields on
  the running models. Restart-only fields are never applied live, because the
  pending-restart banner is the diff between file and runtime.
- Values equal to the schema default are not written for keys the file lacks,
  so hand-kept files stay compact.
- Pushover recipients: `general.notification.destinations` is a list of
  `{id, name, user_key_env, app_token_env?}` and each camera's
  `notification.destinations` names ids (absent = every destination, `[]` =
  none). `PushoverNotifier` holds the live `NotificationSettings` object
  and resolves recipients on every send, so these and the variable-name
  fields apply without a restart; with no destinations the fallback
  `pushover_user_key_env` (comma-separated keys allowed) is used. The store
  refuses a camera naming an unknown destination (`check_destination_refs`);
  the pipeline only warns and skips it.
- Secrets are write-only: `POST /api/secrets` (`configstore.set_secret`)
  writes `NAME=value` into `config/secrets.env` beside the config (backup,
  atomic replace, 0600 when created) and into `os.environ`, only for names
  the saved configuration references (`env_references`) or any `PUSHOVER_…`
  name (so the add-destination dialog can take the key). The value is
  never logged or returned; `describe()` lists the variables under
  `secrets.variables` with who uses them and whether the process reads them
  live (Pushover) or at startup (ONVIF).
- The server-rendered pages are gone: `/`, `/live`, `/recordings`,
  `/recording/<path>`, `/monitor` and `/settings` redirect into the app
  (`/app/...`), query string intact, so old bookmarks and alert links
  still land on the right screen. `/api/settings` went with them.

PTZ calibration parameters in cameras.yml:
```yaml
ptz_tracking:
  pan_scale: 0.8      # PTZ range as fraction of wide FOV
  tilt_scale: 0.6
  pan_center_x: 0.5   # Where PTZ (0,0) appears on wide frame
  tilt_center_y: 0.5
```

## Key Classes

- `PTZTracker` (ptz_tracker.py) - Auto-tracking controller with `update()` for single-camera and `update_multi_camera()` for multi-camera tracking
- `PTZCalibration` (ptz_tracker.py) - Stores pan/tilt/zoom mapping parameters
- `PTZAutoCalibrator` (ptz_calibration.py) - Finds zoom view within wide frame using ORB features
- `ZoomFOVCalibration` (ptz_calibration.py) - Maps what area of cam1 is visible in cam2 at different zoom levels
- `ZoomFOVCalibrator` (ptz_calibration.py) - Calibrates zoom-level to FOV mapping
- `CalibrationPoint` / `CalibrationResult` - Calibration data structures

## Zoom FOV Calibration

The `zoom-calibrate` command maps what portion of cam1 (wide) is visible in cam2 (zoom) at different zoom levels (0%, 50%, 100%). This enables checking if a detection in cam1 would be visible in cam2's current FOV:

```python
from animaltracker.ptz_calibration import ZoomFOVCalibration
import json

# Load saved calibration
with open('config/zoom_fov_calibration.json') as f:
    calib = ZoomFOVCalibration.from_dict(json.load(f))

# Check if detection would be visible at current zoom
bbox = (100, 200, 300, 400)  # Detection from cam1 in pixels
current_zoom = 0.5  # 50% zoom on cam2
if calib.is_detection_visible(bbox, current_zoom):
    print("Detection is in cam2's FOV")
```

## SpeciesNet Setup

Requires Kaggle credentials for model download (~1.5GB):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Set geographic filtering in cameras.yml for accurate species ID:
```yaml
detector:
  country: USA
  admin1_region: MN
```

## Logging

- Configured by `logging_setup.configure_logging()`, called from `cli.py`
  **after** the pipeline imports: `onvif-zeep` calls `logging.basicConfig`
  at import time, so the call uses `force=True` or it would be a no-op.
- Under systemd (`JOURNAL_STREAM` set) lines go only to the journal as
  `LEVEL name: message` with a `<N>` syslog prefix, so app levels become
  journal priorities: `journalctl -u animaltracker -p err` works and the
  web UI Logs page level filter depends on it. Set
  `ANIMALTRACKER_LOG_JOURNAL=0/1` to override the detection.
- Elsewhere: timestamped stderr plus a rotating `logs/animaltracker.log`
  (attached by `cmd_run` once the config is loaded); `logs/web_access.log`
  is the aiohttp access log, rotated in-process at 50 MB.
- Volume controls: `[PERF]` per-camera lines are DEBUG every 10 s with an
  INFO roll-up every 60 s; `[REALTIME]` raw detections are INFO for the
  first frame after a 10 s quiet gap and DEBUG after; RTSP open failures
  log ERROR once, then a WARNING roll-up per minute, then INFO on reconnect.
  `run --debug` turns everything to DEBUG.
