#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -eq 0 ]]; then
  echo "Run as non-root user with sudo privileges." >&2
  exit 1
fi

echo "Updating apt repositories..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-nvcodec ffmpeg git curl wget \
  libssl-dev libffi-dev libxml2-dev libxslt1-dev pkg-config

echo "Ensuring SSD mount point exists..."
sudo mkdir -p /mnt/wildlife_ssd
sudo chown $USER:$USER /mnt/wildlife_ssd

PROJECT_DIR=${PROJECT_DIR:-"$HOME/animaltracker"}
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .[jetson]
# Install CameraTrapAI runtime by default; uncomment the Yolo extra if you need Ultralytics
pip install -e .[cameratrapai]
# pip install -e .[yolo]  # optional: enables ultralytics

if [[ ! -f config/cameras.yml ]]; then
  cp config/cameras.sample.yml config/cameras.yml
  echo "Please edit config/cameras.yml with real RTSP/ONVIF values."
fi

if [[ ! -f config/secrets.env ]]; then
  cp config/secrets.sample.env config/secrets.env
  echo "Populate config/secrets.env with tokens and passwords."
fi

echo "Setup complete. Edit configuration files and deploy systemd units from systemd/."
