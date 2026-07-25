#!/usr/bin/env bash
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
echo -e "${CYAN}   JetsonTracker Build${NC}"
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   [1] constant_robotics_lib${NC}"
echo -e "${CYAN}   [2] cuda_library${NC}"
echo -e "${CYAN}   [3] both${NC}"
echo -e "${CYAN}   [4] lockon (cvtracker-lockon engine)${NC}"
echo -e "${CYAN}   [5] all three${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

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
read -p "Select [1/2/3/4/5]: " choice

echo ""
echo -e "${YELLOW}Clearing build/ for a fully clean build...${NC}"
sudo rm -rf "$(dirname "$0")/build"

build_one() {
    local VERSION=$1
    # Which vendored engine directory provides the `cvtracker` target.
    # Baseline builds use cvtracker/; the lockon build uses cvtracker-lockon/.
    local ENGINE_DIR=${2:-cvtracker}

    # Map the text tracking variant directly to our updated CMake option toggle
    local TOGGLE_VAL="OFF"
    if [ "$VERSION" != "constant_robotics_lib" ]; then
        TOGGLE_VAL="ON"
    fi

    # WHY: clean CMake cache before each build to ensure dynamic changes load cleanly
    sudo rm -rf "$(dirname "$0")/build/CMakeFiles" \
                "$(dirname "$0")/build/CMakeCache.txt" \
                "$(dirname "$0")/build/cvtracker" \
                "$(dirname "$0")/build/cvtracker-lockon" \
                "$(dirname "$0")/build/cvtracker_submodule" \
                "$(dirname "$0")/build/src"

    # WHY: preserve other binaries when executing sequentially (option 3/5)
    sudo rm -f "$(dirname "$0")/build/bin/JetsonTracker_${VERSION}_${BOARD}"

    # Pass TRACKER_VERSION so each sub-build produces a distinctly-named
    # binary (JetsonTracker_${VERSION}_${BOARD}). Without this, both
    # option 3 sub-builds fell through to the CMake default "cuda_library",
    # so the const_robotics binary silently overwrote the cuda_library one
    # and error messages mislabeled which build was actually failing.
    cmake -B "$(dirname "$0")/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DUSE_NATIVE_CUDA_TRACKER=${TOGGLE_VAL} \
          -DJETSON_BOARD:STRING=${BOARD} \
          -DTRACKER_VERSION:STRING=${VERSION} \
          -DTRACKER_ENGINE_DIR:STRING=${ENGINE_DIR} \
          "$(dirname "$0")"

    make -C "$(dirname "$0")/build" -j$(nproc)
}

case $choice in
    1) build_one constant_robotics_lib ;;
    2) build_one cuda_library ;;
    3) build_one constant_robotics_lib && build_one cuda_library ;;
    4) build_one lockon cvtracker-lockon ;;
    5) build_one constant_robotics_lib && build_one cuda_library && build_one lockon cvtracker-lockon ;;
    *) echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
esac

echo ""
echo -e "${CYAN}Built binaries:${NC}"
ls "$(dirname "$0")/build/bin/"

# Apply capabilities to real-time process threads
BIN_DIR="$(dirname "$0")/build/bin"
for bin in "$BIN_DIR"/JetsonTracker_*; do
    [ -f "$bin" ] || continue
    sudo setcap cap_sys_nice+ep "$bin" && \
        echo -e "${GREEN}   cap_sys_nice set: $(basename $bin)${NC}"
done