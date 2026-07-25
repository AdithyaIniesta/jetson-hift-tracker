#!/usr/bin/env bash
# Dataset replay launcher — feeds JPG frames from a dataset folder into
# the tracker via multifilesrc, streams annotated H264 to the ground
# station on Windows. YOU click the template on the GUI, tracker
# locks, follows the target through the sequence.
#
# Bypasses the run_jp5.sh camera-detection / v4l2 dance. Uses the
# TRACKER_FILE_SRC_LEFT env var the tracker already understands.
#
# Usage:
#   ./run_dataset.sh <frames_dir> [gui_ip] [start_index] [engine]
#
# Example (bike1 from UAV123):
#   ./run_dataset.sh \
#       /home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123/bike1 \
#       192.168.0.100
set -e

if [ -z "$1" ]; then
    echo "usage: $0 <frames_dir> [gui_ip] [start_index] [engine]"
    echo "  frames_dir  : folder containing 000001.jpg, 000002.jpg, ..."
    echo "  gui_ip      : where to stream the annotated feed (default 192.168.0.100)"
    echo "  start_index : first frame number (default 1)"
    echo "  engine      : lockon | cuda_library | constant_robotics_lib (default lockon)"
    exit 1
fi

FRAMES_DIR="$1"
GUI_IP="${2:-192.168.0.100}"
START_INDEX="${3:-1}"
ENGINE="${4:-lockon}"

# Auto-detect the JPG naming pattern by peeking at the first file.
FIRST_FILE=$(ls "$FRAMES_DIR" | grep -E "\.jpg$|\.jpeg$" | sort | head -1)
if [ -z "$FIRST_FILE" ]; then
    echo "ERROR: no .jpg files in $FRAMES_DIR"
    exit 1
fi
DIGITS=$(echo "${FIRST_FILE%.*}" | wc -c)
DIGITS=$((DIGITS - 1))   # subtract newline
PATTERN="%0${DIGITS}d.jpg"
FULL_PATH="$FRAMES_DIR/$PATTERN"

# Pick the tracker binary
BIN_DIR="$(dirname "$0")/build/bin"
BOARD=$(uname -m | sed 's/aarch64/orin/')
BIN="$BIN_DIR/JetsonTracker_${ENGINE}_${BOARD}"
if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first with ./build.sh"
    exit 1
fi

# Kill any leftover tracker so CTRL ports 5001/5003 are free.
sudo pkill -9 -f JetsonTracker 2>/dev/null || true
sleep 0.3

echo "=========================================="
echo "  Dataset replay"
echo "=========================================="
echo "  Frames    : $FULL_PATH (start=$START_INDEX)"
echo "  Engine    : $ENGINE"
echo "  GUI IP    : $GUI_IP"
echo "  Ports     : L video 5000  ctrl 5001"
echo "=========================================="
echo ""
echo "On Windows: launch cvtracker_ground_station.py, target IP = this Jetson's IP,"
echo "S1 video 5000, S1 cmd 5001. Drag a rect on the target when the first frame appears."
echo ""

export TRACKER_FILE_SRC_LEFT="$FULL_PATH"
export TRACKER_FILE_SRC_START_INDEX="$START_INDEX"

# main.cpp positional args, matching run_jp5.sh's order:
#   ip  L_vid L_ctrl R_vid R_ctrl W H uart L_dev R_dev fmt fps camModel recBase recEnabled downsample hfov vfov target_depth
"$BIN" "$GUI_IP" \
       5000 5001 5002 5003 \
       1280 720 \
       /dev/null \
       /dev/video0 "" \
       MJPG 60 \
       2 \
       /tmp 0 \
       0 \
       104.6 61.6 \
       0
