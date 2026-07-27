#!/usr/bin/env bash
# HiFT tracker build. This fork replaces the correlation filter with HiFT, so
# there is only one thing to build — no engine menu.
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

sudo chmod 666 /dev/ttyTHS0 2>/dev/null || true

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   JetsonTracker — HiFT build${NC}"
echo -e "${CYAN}==========================================${NC}"

# WHY: auto-detect Jetson board from device tree.
COMPATIBLE=$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' '\n')
if echo "$COMPATIBLE" | grep -qiE "p3767|tegra234"; then
    BOARD="orin"
    echo -e "${GREEN}   Board detected : Jetson Orin NX${NC}"
elif echo "$COMPATIBLE" | grep -qiE "p3668|tegra194"; then
    BOARD="xavier"
    echo -e "${GREEN}   Board detected : Jetson Xavier NX${NC}"
else
    BOARD="orin"
    echo -e "${YELLOW}   Board not detected — defaulting to Orin NX${NC}"
fi
echo -e "${CYAN}==========================================${NC}"
echo ""

DIR="$(dirname "$0")"

echo -e "${YELLOW}Clearing build/ for a fully clean build...${NC}"
sudo rm -rf "$DIR/build"

# HiFT reuses the cvtracker submodule for the VTracker/Frame base symbols, then
# swaps the tracker to HiFTTracker via -DENABLE_HIFT=ON (see common/globals.h).
cmake -B "$DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NATIVE_CUDA_TRACKER=ON \
      -DJETSON_BOARD:STRING=${BOARD} \
      -DTRACKER_VERSION:STRING=hift \
      -DTRACKER_ENGINE_DIR:STRING=cvtracker \
      -DENABLE_HIFT=ON \
      "$DIR"

make -C "$DIR/build" -j$(nproc)

echo ""
echo -e "${CYAN}Built binaries:${NC}"
ls "$DIR/build/bin/"

# Apply capabilities to real-time process threads.
BIN_DIR="$DIR/build/bin"
for bin in "$BIN_DIR"/JetsonTracker_*; do
    [ -f "$bin" ] || continue
    sudo setcap cap_sys_nice+ep "$bin" && \
        echo -e "${GREEN}   cap_sys_nice set: $(basename $bin)${NC}"
done
