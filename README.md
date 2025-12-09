# Animal Tracker RTSP Monitoring

YOLO + RTSP + ONVIF pipeline for multi-camera animal detection on a Jetson Nano, developed from macOS and deployed remotely over SSH.

## Repository Layout
- `rtsp_animal_monitoring_plan.md` – high-level implementation plan and phases.
- `config/` – camera and runtime configuration files (sample included).
- `scripts/` – setup helpers for provisioning the Jetson.
- `src/animaltracker/` – Python package containing the streaming pipeline, ONVIF helpers, clip buffer, notifications, and cleaners.
- `systemd/` – unit files for running the services on-device.

## macOS Development Workflow
1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```
2. **Configure cameras** by duplicating `config/cameras.sample.yml` to `config/cameras.yml` and filling in RTSP/ONVIF credentials.
3. **Run local lint/tests** (optional, CPU-only):
   ```bash
   python -m animaltracker.cli discover --config config/cameras.yml
   python -m animaltracker.cli run --config config/cameras.yml --dry-run
   ```
4. **Sync to Jetson** once ready:
   ```bash
   rsync -avz --exclude '.venv' . jetson-nano:/home/jetson/animaltracker
   ```

## Jetson Nano Setup Commands
SSH into the Jetson (`ssh jetson-nano`) and execute the following once per device.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} \
    gstreamer1.0-nvcodec ffmpeg libssl-dev libffi-dev pkg-config \
    libyaml-dev git curl wget

# Optional but recommended: install ONVIF client dependencies
sudo apt install -y libxml2-dev libxslt1-dev

# Mount SSD (example assumes already partitioned as /dev/nvme0n1p1)
sudo mkdir -p /mnt/wildlife_ssd
sudo blkid /dev/nvme0n1p1 | sudo tee /etc/wildlife_ssd.blkid
sudo sh -c "echo '/dev/nvme0n1p1 /mnt/wildlife_ssd ext4 defaults,noatime 0 2' >> /etc/fstab"
sudo mount -a
sudo chown jetson:jetson /mnt/wildlife_ssd

# Create project user directories
mkdir -p ~/animaltracker ~/animaltracker/config ~/animaltracker/logs

# Python environment
cd ~/animaltracker
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .[jetson]

# Install YOLO runtime (Ultralytics)
pip install ultralytics==8.3.4
# Jetson-specific torch wheel (example for JetPack 5.x / PyTorch 2.1)
wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.1.0-cp38-cp38-linux_aarch64.whl
pip install torch-2.1.0-cp38-cp38-linux_aarch64.whl torchvision torchaudio

# Set up environment files
cp config/cameras.sample.yml config/cameras.yml  # edit with real values
cp config/secrets.sample.env config/secrets.env  # contains Pushover/ONVIF secrets

# Enable systemd units
sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable detector@cam1.service
sudo systemctl enable ssd-cleaner.service ssd-cleaner.timer
sudo systemctl start ssd-cleaner.timer
```

> Adjust the torch wheel URL to match your JetPack + Python version; refer to NVIDIA PyTorch for Jetson docs.

## Key Commands During Operation
- Manual pipeline run: `source .venv/bin/activate && python -m animaltracker.cli run --config config/cameras.yml`
- ONVIF sweep: `python -m animaltracker.cli discover --iface eth0`
- Cleanup job dry-run: `python -m animaltracker.cli cleanup --config config/cameras.yml --dry-run`
- View metrics: `curl http://localhost:9500/metrics`

## Configuration Files
- `config/cameras.yml` – list of cameras, RTSP pipeline strings, thresholds, exclusion lists.
- `config/secrets.env` – exported environment variables consumed by systemd units for Pushover keys, ONVIF passwords, etc.

## Systemd Deployment Cheatsheet
```bash
sudo systemctl enable detector@cam1.service detector@cam2.service
sudo systemctl restart detector@cam1.service
sudo journalctl -u detector@cam1.service -f
```

## Testing & Validation
- Use `tests/fixtures/*.mp4` (add your own) with `rtsp-simple-server` locally to simulate streams.
- Run `pytest` (if you add tests) before deploying.
- Validate ONVIF control via `python -m animaltracker.cli discover --inspect cam1`.

## Next Steps
- Flesh out YOLO model training/export pipeline.
- Hook storage cleanup alerts into Pushover for disk thresholds.
- Expand CLI to cover PTZ patrol scripts and remote firmware health checks.
