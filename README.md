# Animal Tracker RTSP Monitoring

YOLO + RTSP + ONVIF pipeline for multi-camera animal detection on a Jetson Nano, developed from macOS and deployed remotely over SSH.

## Repository Layout
- `rtsp_animal_monitoring_plan.md` – high-level implementation plan and phases.
- `config/` – camera and runtime configuration files.
- `scripts/` – setup helpers and test scripts.
- `src/animaltracker/` – Python package containing the streaming pipeline, ONVIF helpers, clip buffer, notifications, and cleaners.
- `systemd/` – unit files for running the services on-device.

## Jetson Nano Setup (Ubuntu 20.04 / Python 3.10)

### 1. System Dependencies
```bash
sudo apt update && sudo apt install -y \
    python3-venv python3-pip \
    ffmpeg libsm6 libxext6 \
    libxml2-dev libxslt1-dev \
    git curl wget
```

### 2. Project Setup
```bash
# Clone repo (if not already done)
git clone https://github.com/bpasker/animaltracker.git
cd animaltracker

# Create directories
mkdir -p ~/animaltracker/storage ~/animaltracker/logs ~/animaltracker/models

# Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install Dependencies (Pinned for compatibility)
pip install --upgrade pip wheel setuptools
# Note: We pin numpy<2 and opencv-python-headless<4.10 for compatibility
pip install -e .
```

### 3. Download Model
Download the YOLO11 Nano model (optimized for edge devices):
```bash
wget -O models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### 4. Configuration
1.  **Secrets**: Copy `config/secrets.sample.env` to `config/secrets.env` and edit:
    ```bash
    cp config/secrets.sample.env config/secrets.env
    nano config/secrets.env
    ```
2.  **Cameras**: Copy `config/cameras.sample.yml` to `config/cameras.yml` and edit:
    ```bash
    cp config/cameras.sample.yml config/cameras.yml
    nano config/cameras.yml
    ```
    *Tip: For Reolink TrackMix, use the sub-stream (`h264Preview_01_sub`) and TCP transport.*

### 5. Camera Discovery (Finding RTSP URLs)
If you don't know the RTSP URL for your camera, you can use the `discover` command to query the camera via ONVIF and list available profiles and stream URIs.

1.  Configure the camera's IP and ONVIF credentials in `config/cameras.yml`.
2.  Run the discovery command:
    ```bash
    python -m animaltracker.cli --config config/cameras.yml discover --inspect
    ```
3.  The output will list all available profiles and their RTSP URIs. Copy the appropriate URI (prefer H.264/sub-streams for Jetson) back into your `config/cameras.yml`.

## Running the Application

### Manual Run (Testing)
```bash
source .venv/bin/activate
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt
```

### Test Notifications
```bash
source .venv/bin/activate
python scripts/test_pushover.py
```

### Systemd Service (Auto-start)
1.  Edit the service files in `systemd/` to match your user/paths if different from default.
2.  Install services:
    ```bash
    sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    
    # Enable and start for specific cameras
    sudo systemctl enable --now detector@cam1.service
    sudo systemctl enable --now detector@cam2.service
    
    # Enable cleanup timer
    sudo systemctl enable --now ssd-cleaner.timer
    ```

## Troubleshooting

### "numpy.core.multiarray failed to import"
Ensure you have compatible versions installed:
```bash
pip uninstall -y opencv-python opencv-python-headless numpy
pip install "numpy<2" "opencv-python-headless<4.10"
```

### "Unable to open RTSP stream"
- Check if the camera URL is correct (use VLC to verify).
- Ensure `transport: tcp` is set in `cameras.yml`.
- Try using the sub-stream (lower resolution, H.264) which is more stable on Jetson.

### "PPS changed between slices" (FFmpeg/OpenCV)
This usually happens with H.265 streams on software decoding. Switch to the H.264 sub-stream in your camera config.

## Next Steps
- Flesh out YOLO model training/export pipeline.
- Hook storage cleanup alerts into Pushover for disk thresholds.
- Expand CLI to cover PTZ patrol scripts and remote firmware health checks.
