#!/usr/bin/env bash
# ── Default device parameters ──────────────
DEFAULT_IP=192.168.0.5
DEFAULT_WIDTH=1280
DEFAULT_HEIGHT=720
UART=/dev/ttyTHS0
VIDEO=/dev/video0
PORT_VIDEO=5000
PORT_CTRL=5001
# ───────────────────────────────────────────

BIN_DIR="/opt/jvp/bin"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}       JetsonTracker Launcher             ${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# ── Check binaries ─────────────────────────
BIN_V1="$BIN_DIR/JetsonTracker_constant_robotics_lib"
BIN_V2="$BIN_DIR/JetsonTracker_cuda_library"

echo -e "${CYAN}Available executables:${NC}"
echo ""
[ -f "$BIN_V1" ] && echo -e "  ${GREEN}[1] JetsonTracker_constant_robotics_lib${NC}" || echo -e "  ${RED}[1] JetsonTracker_constant_robotics_lib — missing${NC}"
[ -f "$BIN_V2" ] && echo -e "  ${GREEN}[2] JetsonTracker_cuda_library${NC}" || echo -e "  ${RED}[2] JetsonTracker_cuda_library — missing${NC}"
echo ""

if [ ! -f "$BIN_V1" ] && [ ! -f "$BIN_V2" ]; then
    echo -e "${RED}No binaries found. Rebuild the runtime image.${NC}"
    exit 1
fi

if [ ! -f "$BIN_V1" ] || [ ! -f "$BIN_V2" ]; then
    echo -e "${YELLOW}One binary missing. Rebuild the runtime image.${NC}"
    echo ""
fi

read -p "Select algorithm [1/2]: " choice

case $choice in
    1) BIN="$BIN_V1" ;;
    2) BIN="$BIN_V2" ;;
    *) echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
esac

if [ ! -f "$BIN" ]; then
    echo -e "${RED}Binary not available. Rebuild the runtime image.${NC}"
    exit 1
fi

# ── Network settings ───────────────────────
echo ""
read -p "Stream IP [${DEFAULT_IP}]: " IP
IP=${IP:-$DEFAULT_IP}

read -p "Width [${DEFAULT_WIDTH}]: " WIDTH
WIDTH=${WIDTH:-$DEFAULT_WIDTH}

read -p "Height [${DEFAULT_HEIGHT}]: " HEIGHT
HEIGHT=${HEIGHT:-$DEFAULT_HEIGHT}

# ── Fix UART permission ────────────────────
sudo fuser -k 5000/udp 2>/dev/null || true
sudo fuser -k 5001/udp 2>/dev/null || true
sudo chmod 666 $UART 2>/dev/null || true

# ── Launch ─────────────────────────────────
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${GREEN}  Starting $(basename $BIN)${NC}"
echo -e "${CYAN}  Stream to : ${IP}:${PORT_VIDEO}${NC}"
echo -e "${CYAN}  Resolution: ${WIDTH}x${HEIGHT}${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

$BIN $IP $PORT_VIDEO $PORT_CTRL $WIDTH $HEIGHT $UART $VIDEO