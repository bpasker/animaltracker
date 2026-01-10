# Animal Tracker RTSP Monitoring

🦌 AI-powered wildlife detection system using YOLO + RTSP + ONVIF for multi-camera animal monitoring. Get instant push notifications when animals appear on your cameras!

**Features:**
- Real-time animal detection on IP cameras (RTSP streams)
- **SpeciesNet integration** - Google's AI trained on 65M+ camera trap images, identifying 2000+ wildlife species
- Geographic filtering - eliminates impossible species for your location
- Push notifications via Pushover with video clips & thumbnails
- PTZ camera tracking support (follows detected animals)
- Web UI for browsing recorded clips
- Runs on Jetson Nano, Linux, macOS, or Windows

---

## 📋 Table of Contents
- [Quick Start (5 minutes)](#-quick-start-5-minutes)
- [Full Setup by Platform](#full-setup-by-platform)
  - [macOS](#macos-setup)
  - [Linux / Jetson Nano](#linux--jetson-nano-setup)
  - [Windows](#windows-setup)
- [Configuration Guide](#-configuration-guide)
  - [Detector Backends: SpeciesNet vs YOLO](#detector-backends-speciesnet-vs-yolo)
- [CLI Commands Reference](#-cli-commands-reference)
- [Running as a Service](#-running-as-a-service)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+ (3.10+ recommended)
- Git
- FFmpeg
- An IP camera with RTSP support
- Kaggle account (free, for SpeciesNet model download): https://www.kaggle.com/

### One-Line Install (macOS/Linux)
```bash
git clone https://github.com/bpasker/animaltracker.git && cd animaltracker && ./scripts/quick_setup.sh
```

Or follow the manual steps below:

### Manual Quick Start
```bash
# 1. Clone and enter directory
git clone https://github.com/bpasker/animaltracker.git
cd animaltracker

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -e .

# 4. Set up Kaggle credentials (required for SpeciesNet model download)
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
# Or create ~/.kaggle/kaggle.json with: {"username":"...","key":"..."}

# 5. Copy and edit config files
cp config/secrets.sample.env config/secrets.env
cp config/cameras.sample.yml config/cameras.yml
# Edit both files with your settings (see Configuration Guide below)
# Important: Set your country code in cameras.yml for geographic filtering!

# 6. Run! (SpeciesNet downloads ~1.5GB model on first run)
python -m animaltracker.cli --config config/cameras.yml run
```

> **Note:** The YOLO model (`models/yolo11n.pt`) is only needed if you use `backend: yolo`. SpeciesNet downloads its own models automatically.

---

## Full Setup by Platform

### macOS Setup

#### 1. Install Prerequisites
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11 git ffmpeg
```

#### 2. Project Setup
```bash
# Clone repository
git clone https://github.com/bpasker/animaltracker.git
cd animaltracker

# Create directories
mkdir -p storage logs models

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install --upgrade pip wheel setuptools
pip install -e .
```

#### 3. Download Model
```bash
curl -L -o models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

#### 4. Configure (see [Configuration Guide](#-configuration-guide) below)
```bash
cp config/secrets.sample.env config/secrets.env
cp config/cameras.sample.yml config/cameras.yml
# Edit both files with your camera and notification settings
```

#### 5. Run
```bash
# Use the mac-specific config or edit cameras.yml
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt
```

---

### Linux / Jetson Nano Setup

#### 1. Install System Dependencies
```bash
sudo apt update && sudo apt install -y \
    python3-venv python3-pip python3-dev \
    ffmpeg libsm6 libxext6 \
    libxml2-dev libxslt1-dev \
    git curl wget
```

#### 2. Project Setup
```bash
# Clone repository
git clone https://github.com/bpasker/animaltracker.git
cd animaltracker

# Create directories
mkdir -p storage logs models

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install --upgrade pip wheel setuptools
pip install -e .
```

#### 3. Download Model
```bash
wget -O models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

#### 4. Configure (see [Configuration Guide](#-configuration-guide) below)
```bash
cp config/secrets.sample.env config/secrets.env
cp config/cameras.sample.yml config/cameras.yml
nano config/secrets.env     # Add your Pushover keys
nano config/cameras.yml     # Add your camera settings
```

#### 5. Run
```bash
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt
```

---

### Windows Setup

#### 1. Install Prerequisites
- **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/) ✅ Check "Add to PATH" during install
- **Git**: Download from [git-scm.com](https://git-scm.com/download/win)
- **FFmpeg**: Install via one of these methods:
  - [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`
  - [Scoop](https://scoop.sh/): `scoop install ffmpeg`
  - Manual: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

#### 2. Project Setup (PowerShell)
```powershell
# Clone repository
git clone https://github.com/bpasker/animaltracker.git
cd animaltracker

# Create directories
mkdir storage, logs, models -Force

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the package
pip install --upgrade pip wheel setuptools
pip install -e .
```

#### 3. Download Model
```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -OutFile "models\yolo11n.pt"
```

#### 4. Configure (see [Configuration Guide](#-configuration-guide) below)
```powershell
copy config\secrets.sample.env config\secrets.env
copy config\cameras.sample.yml config\cameras.yml
# Edit both files with your settings using notepad or VS Code
```

#### 5. Run
```powershell
python -m animaltracker.cli --config config\cameras.yml run --model models\yolo11n.pt
```

---

## 📝 Configuration Guide

### Step 1: Configure Secrets (`config/secrets.env`)

Copy from sample and edit:
```bash
cp config/secrets.sample.env config/secrets.env
```

**Required settings:**
```env
# Pushover Notifications (get keys at https://pushover.net)
PUSHOVER_APP_TOKEN=your-app-token-here
PUSHOVER_USER_KEY=your-user-key-here

# Camera credentials (used in RTSP URLs and ONVIF)
DEFAULT_RTSP_USERNAME=admin
DEFAULT_RTSP_PASSWORD=your-camera-password

# Per-camera ONVIF credentials (match camera IDs in cameras.yml)
CAM1_ONVIF_USER=admin
CAM1_ONVIF_PASS=your-camera-password
```

### Step 2: Configure Cameras (`config/cameras.yml`)

Copy from sample and edit:
```bash
cp config/cameras.sample.yml config/cameras.yml
```

**Key settings to update:**

```yaml
general:
  # Where to store recorded clips and logs
  storage_root: /path/to/animaltracker/storage   # Update this path!
  logs_root: /path/to/animaltracker/logs         # Update this path!
  
  clip:
    pre_seconds: 5     # Seconds to record before detection
    post_seconds: 5    # Seconds to record after detection
  
  detector:
    backend: speciesnet              # "speciesnet" (recommended) or "yolo" (fast)
    model_path: models/yolo11n.pt    # Only used for YOLO backend
    # SpeciesNet settings (see Detector Backends section below)
    speciesnet_version: v4.0.2a      # v4.0.2a (crop-based) or v4.0.2b (full-image)
    country: USA                      # Your country code for geographic filtering
    admin1_region: MN                 # State/province code (optional)
    generic_confidence: 0.9           # Higher threshold for generic labels
  
  # Species to ignore globally
  exclusion_list:
    - person
    - car

cameras:
  - id: cam1                         # Unique ID (used in systemd service names)
    name: "Front Yard"               # Display name
    rtsp:
      # Replace with YOUR camera's RTSP URL
      uri: "rtsp://admin:password@192.168.1.100:554/stream1"
      transport: tcp                 # tcp or udp
    onvif:
      host: 192.168.1.100           # Camera IP
      port: 80                       # ONVIF port (usually 80 or 8000)
      username_env: CAM1_ONVIF_USER  # References secrets.env
      password_env: CAM1_ONVIF_PASS
    thresholds:
      confidence: 0.5                # Detection confidence (0.0-1.0)
      min_frames: 3                  # Frames before triggering
```

### Detector Backends: SpeciesNet vs YOLO

Animal Tracker supports two detection backends. **SpeciesNet is recommended** for wildlife monitoring.

#### SpeciesNet (Recommended for Wildlife)

[Google SpeciesNet](https://github.com/google/speciesnet) is an AI model trained specifically for camera trap images, combining:
- **MegaDetector** - Detects animals, people, and vehicles in camera trap images
- **Species Classifier** - Identifies 2000+ wildlife species trained on 65M+ images
- **Geographic Filtering** - Filters out species impossible for your location

**Installation:**
```bash
# SpeciesNet is included in the default installation
pip install -e .

# Or install separately:
pip install speciesnet
```

**First Run Note:** SpeciesNet automatically downloads model weights (~1.5GB) from Kaggle on first use. This requires:
1. A Kaggle account (free): https://www.kaggle.com/
2. Kaggle API credentials configured

**Setting up Kaggle credentials:**
```bash
# Option 1: Environment variables (recommended)
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key

# Option 2: Credentials file
# Create ~/.kaggle/kaggle.json with:
# {"username":"your_username","key":"your_api_key"}
```

**Configuration (`cameras.yml`):**
```yaml
general:
  detector:
    backend: speciesnet
    speciesnet_version: v4.0.2a    # v4.0.2a (crop, faster) or v4.0.2b (full-image)
    
    # Geographic filtering - IMPORTANT for accurate species ID
    country: USA                    # ISO 3166-1 alpha-3 code
    admin1_region: MN               # US state code (optional but recommended)
    # latitude: 45.0                # Camera coordinates (optional, for future use)
    # longitude: -93.0
    
    # Confidence thresholds
    generic_confidence: 0.9         # Threshold for generic labels (animal, bird, mammalia)
                                    # Specific species use the camera's normal threshold
```

**Country Codes (ISO 3166-1 alpha-3):**
| Country | Code | Country | Code |
|---------|------|---------|------|
| United States | `USA` | Canada | `CAN` |
| United Kingdom | `GBR` | Germany | `DEU` |
| Australia | `AUS` | Mexico | `MEX` |
| France | `FRA` | Spain | `ESP` |

**US State Codes (`admin1_region`):**
Use standard 2-letter state abbreviations: `MN`, `CA`, `TX`, `NY`, `FL`, etc.

**How Geographic Filtering Works:**
- SpeciesNet uses your location to filter out impossible species
- Example: A "tiger" detection in Minnesota is rejected as impossible
- Built-in blocklists for North America, Europe, Australia
- Reduces false positives significantly

**Model Versions:**
| Version | Description | Use Case |
|---------|-------------|----------|
| `v4.0.2a` | Crop-based classifier | Faster, good for real-time |
| `v4.0.2b` | Full-image classifier | More context, slightly slower |

---

#### YOLO (Fast General Detection)

YOLO is faster but detects general object classes (bird, cat, dog, bear) rather than specific species.

**When to use YOLO:**
- You only need general categories (not specific species)
- Running on very limited hardware
- Processing many cameras simultaneously

**Configuration:**
```yaml
general:
  detector:
    backend: yolo
    model_path: models/yolo11n.pt  # Downloaded during setup
```

**Available YOLO Models:**
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolo11n.pt` | 6MB | Fastest | Good |
| `yolo11s.pt` | 22MB | Fast | Better |
| `yolo11m.pt` | 42MB | Medium | Best |

---

### Finding Your Camera's RTSP URL

If you don't know your RTSP URL, use the discover command:

```bash
# First, add your camera's IP and ONVIF credentials to cameras.yml
# Then run:
python -m animaltracker.cli --config config/cameras.yml discover --inspect
```

**Common RTSP URL formats:**
| Brand | URL Format |
|-------|------------|
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Amcrest | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Generic ONVIF | Use the `discover --inspect` command |

---

## 🔧 CLI Commands Reference

All commands follow the pattern:
```bash
python -m animaltracker.cli --config config/cameras.yml <command> [options]
```

### `run` - Start Detection Pipeline
```bash
# Run all cameras
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt

# Run specific camera only
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt --camera cam1

# Enable PTZ debug logging
python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt --ptz-debug
```

### `discover` - Find Camera Settings via ONVIF
```bash
# Basic discovery
python -m animaltracker.cli --config config/cameras.yml discover

# Show detailed profile info (RTSP URLs)
python -m animaltracker.cli --config config/cameras.yml discover --inspect

# Show PTZ presets
python -m animaltracker.cli --config config/cameras.yml discover --presets
```

### `ptz-test` - Test PTZ Camera Controls
```bash
# List PTZ profiles
python -m animaltracker.cli --config config/cameras.yml ptz-test --camera cam1

# Find working PTZ profile (moves camera!)
python -m animaltracker.cli --config config/cameras.yml ptz-test --camera cam1 --find-working
```

### `cleanup` - Remove Old Clips
```bash
# Preview what would be deleted
python -m animaltracker.cli --config config/cameras.yml cleanup --dry-run

# Actually delete old clips
python -m animaltracker.cli --config config/cameras.yml cleanup
```

### `reprocess` - Re-analyze Recorded Clips

Re-analyze existing clips to improve species identification. Useful when:
- You've upgraded to SpeciesNet from YOLO
- You've changed your geographic settings
- You want to re-identify clips with improved confidence

```bash
# Reprocess all clips using detector settings from cameras.yml
python -m animaltracker.cli --config config/cameras.yml reprocess

# Reprocess only clips from a specific camera
python -m animaltracker.cli --config config/cameras.yml reprocess --camera cam1

# Reprocess a single clip
python -m animaltracker.cli --config config/cameras.yml reprocess --clip storage/clips/myclip.mp4

# More thorough analysis (analyze more frames, slower)
python -m animaltracker.cli --config config/cameras.yml reprocess --sample-rate 2

# Keep original filenames (don't rename based on new species)
python -m animaltracker.cli --config config/cameras.yml reprocess --no-rename
```

**Options:**
| Option | Description |
|--------|-------------|
| `--camera CAM_ID` | Only reprocess clips from this camera |
| `--clip PATH` | Reprocess a single clip file |
| `--sample-rate N` | Analyze every Nth frame (default=5, lower=more thorough) |
| `--no-rename` | Don't rename files even if species changes |
| `--no-thumbnails` | Don't regenerate detection thumbnails |

---

## 🔄 Running as a Service

### Linux (systemd) - Recommended for Jetson/Servers

```bash
# 1. Copy service files
sudo cp systemd/detector@.service /etc/systemd/system/
sudo cp systemd/ssd-cleaner.service systemd/ssd-cleaner.timer /etc/systemd/system/

# 2. Reload systemd
sudo systemctl daemon-reload

# 3. Enable and start for your camera (use camera ID from cameras.yml)
sudo systemctl enable detector@cam1
sudo systemctl start detector@cam1

# 4. Check status
sudo systemctl status detector@cam1

# 5. View logs
journalctl -u detector@cam1 -f

# 6. Enable automatic cleanup (optional)
sudo systemctl enable --now ssd-cleaner.timer
```

### Windows (NSSM Service)

```powershell
# Install NSSM
choco install nssm

# Create service (update paths!)
nssm install AnimalTracker "C:\path\to\animaltracker\.venv\Scripts\python.exe"
nssm set AnimalTracker AppParameters "-m animaltracker.cli --config config\cameras.yml run --model models\yolo11n.pt"
nssm set AnimalTracker AppDirectory "C:\path\to\animaltracker"
nssm start AnimalTracker
```

### macOS (launchd)

Create `~/Library/LaunchAgents/com.animaltracker.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.animaltracker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/animaltracker/.venv/bin/python</string>
        <string>-m</string>
        <string>animaltracker.cli</string>
        <string>--config</string>
        <string>/path/to/animaltracker/config/cameras.yml</string>
        <string>run</string>
        <string>--model</string>
        <string>/path/to/animaltracker/models/yolo11n.pt</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/animaltracker</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/path/to/animaltracker/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/animaltracker/logs/stderr.log</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.animaltracker.plist
```

---

## 🐛 Troubleshooting

### SpeciesNet Issues

#### "SpeciesNet not installed" error
```bash
pip install speciesnet
# Or reinstall the full package:
pip install -e .
```

#### Model download fails / Kaggle authentication error
SpeciesNet downloads models from Kaggle on first run. You need Kaggle credentials:

```bash
# 1. Create a Kaggle account at https://www.kaggle.com/
# 2. Go to Account Settings → API → Create New Token
# 3. Set credentials via environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Or create credentials file:
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

#### SpeciesNet is slow
- SpeciesNet is more accurate but slower than YOLO
- Use `speciesnet_version: v4.0.2a` (crop-based, faster) instead of `v4.0.2b`
- Consider using lower-resolution camera streams
- For real-time needs with many cameras, use YOLO and `reprocess` clips later with SpeciesNet

#### Getting generic labels like "animal" or "bird" instead of species
1. **Set your location** - Geographic filtering dramatically improves species ID:
   ```yaml
   detector:
     country: USA           # Required!
     admin1_region: MN      # Recommended for US
   ```
2. **Adjust generic_confidence** - Lower it if you're getting too many generic labels:
   ```yaml
   detector:
     generic_confidence: 0.8  # Default is 0.9
   ```
3. **Check confidence thresholds** - Specific species need lower confidence to pass:
   ```yaml
   cameras:
     - id: cam1
       thresholds:
         confidence: 0.4    # Try lowering from 0.5
   ```

#### Wrong species identified
- Ensure `country` and `admin1_region` are set correctly
- Species impossible for your region are automatically filtered
- Run `reprocess` on clips to re-analyze with updated settings

---

### General Issues

### "numpy.core.multiarray failed to import"
```bash
pip uninstall -y opencv-python opencv-python-headless numpy
pip install "numpy<2" "opencv-python-headless<4.10"
```

### Camera connection fails / No video
1. Test RTSP URL directly with VLC: `vlc rtsp://user:pass@ip:554/stream`
2. Ensure camera and computer are on same network
3. Check firewall allows port 554 (RTSP) and ONVIF port (80 or 8000)
4. Try `transport: tcp` instead of `udp` in cameras.yml

### ONVIF discovery fails
1. Verify ONVIF is enabled in camera settings
2. Check username/password in secrets.env
3. Try different ONVIF ports (80, 8000, 8080)

### No notifications received
1. Run `python scripts/test_pushover.py` to test
2. Verify PUSHOVER_APP_TOKEN and PUSHOVER_USER_KEY in secrets.env
3. Check Pushover app is installed on your phone

### High CPU usage / Slow detection
1. Use sub-stream instead of main stream (lower resolution)
2. Increase `confidence` threshold to reduce false positives
3. On Jetson: Enable NVDEC (`hwaccel: true`) for GPU decoding
4. Consider YOLO backend for real-time, then `reprocess` with SpeciesNet

### "ModuleNotFoundError: No module named 'animaltracker'"
```bash
# Make sure you're in the project directory and venv is activated
cd animaltracker
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

---

## 📁 Repository Layout

```
animaltracker/
├── config/                 # Configuration files
│   ├── cameras.yml        # Your camera settings (create from sample)
│   ├── cameras.sample.yml # Example camera config
│   ├── secrets.env        # Your API keys (create from sample)
│   └── secrets.sample.env # Example secrets
├── models/                # YOLO model files (only needed for YOLO backend)
├── storage/               # Recorded clips and thumbnails
│   └── clips/
├── logs/                  # Application logs
├── src/animaltracker/     # Python source code
├── systemd/               # Linux service files
├── scripts/               # Helper scripts
└── tests/                 # Test files
```

---

## 📄 License

MIT License - See LICENSE file for details.

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

#CHECK GPU Usage
nvidia-smi -l 1

systemctl restart detector@cam2


journalctl -u detector@cam2 --since "30 minutes ago"