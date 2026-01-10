#!/bin/bash
#
# Animal Tracker Quick Setup Script
# Automates the installation process for macOS and Linux
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           🦌 Animal Tracker - Quick Setup                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo -e "${GREEN}✓ Detected: macOS${NC}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}✓ Detected: Linux${NC}"
else
    echo -e "${RED}✗ Unsupported OS: $OSTYPE${NC}"
    echo "  This script supports macOS and Linux only."
    echo "  For Windows, please follow the manual instructions in README.md"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}→ Project directory: ${PROJECT_DIR}${NC}"
cd "$PROJECT_DIR"

# Check for Python
echo ""
echo -e "${BLUE}[1/6] Checking Python installation...${NC}"
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}✓ Found $cmd (version $version)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗ Python 3.8+ not found!${NC}"
    if [ "$OS" == "macos" ]; then
        echo "  Install with: brew install python@3.11"
    else
        echo "  Install with: sudo apt install python3 python3-venv python3-pip"
    fi
    exit 1
fi

# Check for FFmpeg
echo ""
echo -e "${BLUE}[2/6] Checking FFmpeg installation...${NC}"
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg is installed${NC}"
else
    echo -e "${YELLOW}⚠ FFmpeg not found - attempting to install...${NC}"
    if [ "$OS" == "macos" ]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo -e "${RED}✗ Homebrew not found. Please install FFmpeg manually:${NC}"
            echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "  2. Install FFmpeg: brew install ffmpeg"
            exit 1
        fi
    else
        sudo apt update && sudo apt install -y ffmpeg libsm6 libxext6 libxml2-dev libxslt1-dev
    fi
fi

# Create directories
echo ""
echo -e "${BLUE}[3/6] Creating directories...${NC}"
mkdir -p storage logs models
echo -e "${GREEN}✓ Created: storage/, logs/, models/${NC}"

# Create virtual environment
echo ""
echo -e "${BLUE}[4/6] Setting up Python virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
else
    $PYTHON_CMD -m venv .venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
fi

# Activate and install dependencies
echo ""
echo -e "${BLUE}[5/6] Installing Python dependencies...${NC}"
source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q
pip install -e . -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download YOLO model
echo ""
echo -e "${BLUE}[6/6] Downloading YOLO model...${NC}"
if [ -f "models/yolo11n.pt" ]; then
    echo -e "${YELLOW}⚠ Model already exists at models/yolo11n.pt${NC}"
else
    echo "  Downloading yolo11n.pt (~6MB)..."
    curl -L -o models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi

# Copy config files if they don't exist
echo ""
echo -e "${BLUE}Setting up configuration files...${NC}"
if [ ! -f "config/secrets.env" ]; then
    cp config/secrets.sample.env config/secrets.env
    echo -e "${GREEN}✓ Created config/secrets.env (needs editing)${NC}"
else
    echo -e "${YELLOW}⚠ config/secrets.env already exists${NC}"
fi

if [ ! -f "config/cameras.yml" ]; then
    cp config/cameras.sample.yml config/cameras.yml
    echo -e "${GREEN}✓ Created config/cameras.yml (needs editing)${NC}"
else
    echo -e "${YELLOW}⚠ config/cameras.yml already exists${NC}"
fi

# Update storage paths in cameras.yml for this machine
if [ "$OS" == "macos" ]; then
    # Use sed with backup for macOS compatibility
    sed -i.bak "s|storage_root:.*|storage_root: ${PROJECT_DIR}/storage|g" config/cameras.yml
    sed -i.bak "s|logs_root:.*|logs_root: ${PROJECT_DIR}/logs|g" config/cameras.yml
    rm -f config/cameras.yml.bak
    echo -e "${GREEN}✓ Updated storage paths in cameras.yml${NC}"
fi

# Print success message
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              ✓ Setup Complete!                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${YELLOW}⚠ NEXT STEPS - You must configure these files:${NC}"
echo ""
echo "  1. Edit config/secrets.env with your Pushover API keys:"
echo "     ${BLUE}nano config/secrets.env${NC}"
echo ""
echo "  2. Edit config/cameras.yml with your camera settings:"
echo "     ${BLUE}nano config/cameras.yml${NC}"
echo "     - Add your camera's RTSP URL"
echo "     - Set ONVIF credentials"
echo ""
echo "  3. Test your setup:"
echo "     ${BLUE}source .venv/bin/activate${NC}"
echo "     ${BLUE}python -m animaltracker.cli --config config/cameras.yml discover --inspect${NC}"
echo ""
echo "  4. Run the detector:"
echo "     ${BLUE}python -m animaltracker.cli --config config/cameras.yml run --model models/yolo11n.pt${NC}"
echo ""
echo -e "${BLUE}For more help, see README.md or run:${NC}"
echo "     python -m animaltracker.cli --help"
echo ""
