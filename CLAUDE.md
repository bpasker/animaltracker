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
- `detector.py` - Detection backends: MegaDetector, YOLO, SpeciesNet
- `tracker.py` - ByteTrack object tracking with persistent IDs
- `postprocess.py` - Clip post-analysis, track merging, species finalization
- `ptz_tracker.py` - PTZ auto-tracking controller, pixel-to-PTZ coordinate mapping
- `ptz_calibration.py` - Auto-calibration via ORB feature matching between wide/zoom frames
- `onvif_client.py` - ONVIF camera control and discovery
- `web.py` - Flask web UI for clip browsing

### Multi-Camera PTZ Tracking

Cam1 (wide-angle) detects animals and controls cam2 (zoom) PTZ movements:

```yaml
cameras:
  - id: cam1
    ptz_tracking:
      target_camera_id: cam2  # cam1 detections drive cam2's PTZ
```

Both cameras share pan/tilt hardware but cam2 has zoom control. The `PTZTracker` converts pixel coordinates from cam1's frame to PTZ commands for cam2.

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

PTZ calibration parameters in cameras.yml:
```yaml
ptz_tracking:
  pan_scale: 0.8      # PTZ range as fraction of wide FOV
  tilt_scale: 0.6
  pan_center_x: 0.5   # Where PTZ (0,0) appears on wide frame
  tilt_center_y: 0.5
```

## Key Classes

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
