#!/usr/bin/env bash
# ── Port mapping ───────────────────────────
DEFAULT_IP=192.168.0.100
# UART is hardcoded per JetPack — this is run_jp6.sh, use it only on a
# JetPack 6 Jetson. On JetPack 5, launch run_jp5.sh instead. Autodetect
# between ttyTHS0/1/2 has burned us before: the kernel exposes multiple
# nodes but only one is actually wired to the airframe UART, so the
# wrong pick silently kills angle output. Keep it static.
UART=/dev/ttyTHS1
LEFT_VIDEO_PORT=5000
LEFT_CTRL_PORT=5001
RIGHT_VIDEO_PORT=5002
RIGHT_CTRL_PORT=5003
# ───────────────────────────────────────────

BIN_DIR="$(cd "$(dirname "$0")" && pwd)/build/bin"
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

# ── Select binary ──────────────────────────
echo -e "${CYAN}Available executables:${NC}"
echo ""
BINS=()
IDX=1
for f in "$BIN_DIR"/JetsonTracker_*; do
    [ -f "$f" ] || continue
    echo -e "  ${GREEN}[$IDX] $(basename $f)${NC}"
    BINS+=("$f")
    IDX=$((IDX + 1))
done

if [ ${#BINS[@]} -eq 0 ]; then
    echo -e "${RED}No binaries found. Run ./build.sh first.${NC}"; exit 1
fi

# New logic: If only one binary, select it automatically
if [ ${#BINS[@]} -eq 1 ]; then
    echo -e "${CYAN}Only one binary found, auto-selecting...${NC}"
    choice=1
else
    # Only prompt if there are multiple binaries
    read -p "Select algorithm [1-$((IDX-1))] (default: 2): " choice
    choice=${choice:-2}
fi

# Validation logic remains the same
if [ -z "$choice" ] || [ "$choice" -lt 1 ] || \
   [ "$choice" -gt $((IDX-1)) ] 2>/dev/null; then
    echo -e "${RED}Invalid selection.${NC}"; exit 1
fi

BIN="${BINS[$((choice-1))]}"
if [ ! -f "$BIN" ]; then
    echo -e "${RED}Binary not available.${NC}"; exit 1
fi


# ── Recording ──────────────────────────────
DEFAULT_REC_PATH="$HOME/recordings"
echo ""

read -p "Enable recording? [Y/n]: " REC_ENABLE
REC_ENABLE=${REC_ENABLE:-Y}
REC_ENABLE=${REC_ENABLE:-N}
if [ "$REC_ENABLE" = "y" ] || [ "$REC_ENABLE" = "Y" ]; then
    RECORDING=1
    read -p "Recording path [${DEFAULT_REC_PATH}]: " REC_PATH
    REC_PATH=${REC_PATH:-$DEFAULT_REC_PATH}
    if [ ! -d "$REC_PATH" ]; then
        read -p "  Create '$REC_PATH'? [y/N]: " CREATE_DIR
        if [ "$CREATE_DIR" = "y" ] || [ "$CREATE_DIR" = "Y" ]; then
            mkdir -p "$REC_PATH"
            echo -e "${GREEN}  Created: $REC_PATH${NC}"
        else
            echo -e "${RED}  Directory not found. Exiting.${NC}"; exit 1
        fi
    fi
    echo -e "${GREEN}  Recording enabled: ${REC_PATH}${NC}"
else
    RECORDING=0
    REC_PATH="/tmp"
    echo -e "${YELLOW}  Recording disabled.${NC}"
fi

# ── Network ────────────────────────────────
echo ""
read -p "Stream IP [${DEFAULT_IP}]: " IP
IP=${IP:-$DEFAULT_IP}

# ── Aircraft camera pinning (if set up) ────
# If scripts/setup_cameras.sh has been run on this Jetson, /dev/boresight
# and /dev/depression are stable symlinks. Convention: boresight = LEFT,
# depression = RIGHT. Skip the whole autodetect flow.
PINNED_LEFT=""
PINNED_RIGHT=""
[ -e /dev/boresight  ] && PINNED_LEFT=$(readlink -f /dev/boresight)
[ -e /dev/depression ] && PINNED_RIGHT=$(readlink -f /dev/depression)

# ── Detect video devices ───────────────────
echo ""
echo -e "${CYAN}Detecting video devices...${NC}"
for attempt in 1 2 3 4 5; do
    FOUND=0
    for node in 0 1 2 3 4 5; do
        DEV="/dev/video$node"
        [ ! -e "$DEV" ] && continue
        FMTS=$(v4l2-ctl --device="$DEV" --list-formats 2>/dev/null \
               | grep -oP "'\w+'" | tr -d "'" | tr '\n' ' ')
        [ -n "$FMTS" ] && FOUND=1 && break
    done
    [ "$FOUND" = "1" ] && break
    echo -e "${YELLOW}  No devices yet, waiting 2s (attempt $attempt/5)...${NC}"
    sleep 2
done

echo ""
echo -e "${CYAN}Detected video devices:${NC}"
echo ""
VIDEO_DEVICES=()
VIDEO_LABELS=()
IDX=1
for node in 0 1 2 3 4 5; do
    DEV="/dev/video$node"
    [ ! -e "$DEV" ] && continue
    FMTS=$(v4l2-ctl --device="$DEV" --list-formats 2>/dev/null \
           | grep -oP "'\w+'" | tr -d "'" | tr '\n' ' ')
    [ -z "$FMTS" ] && continue
    if echo "$FMTS" | grep -q "UYVY"; then
        CAM_LABEL="ECON See3CAM"
    elif echo "$FMTS" | grep -q "YUYV"; then
        CAM_LABEL="Arducam B0510"
    else
        CAM_LABEL="Unknown"
    fi
    echo -e "  ${GREEN}[$IDX]  $DEV  —  $CAM_LABEL  —  formats: $FMTS${NC}"
    VIDEO_DEVICES+=("$DEV")
    VIDEO_LABELS+=("$CAM_LABEL")
    IDX=$((IDX + 1))
done
if [ ${#VIDEO_DEVICES[@]} -eq 0 ]; then
    echo -e "${RED}No video devices found.${NC}"; exit 1
fi

# ── Camera mode selection ──────────────────
# WHY: left/right device always passed as argv[9]/argv[10].
# Empty string "" keeps argv positions fixed when one side unused.
LEFT_DEV=""
RIGHT_DEV=""

# If aircraft pinning is present, use it and skip the L/R prompt.
if [ -n "$PINNED_LEFT" ] || [ -n "$PINNED_RIGHT" ]; then
    LEFT_DEV="$PINNED_LEFT"
    RIGHT_DEV="$PINNED_RIGHT"
    SELECTED_CAM="ECON See3CAM"
    echo ""
    echo -e "${CYAN}  Aircraft pinning detected — using stable symlinks:${NC}"
    [ -n "$LEFT_DEV"  ] && echo -e "${GREEN}    boresight  (LEFT)  : /dev/boresight  -> $LEFT_DEV${NC}"
    [ -n "$RIGHT_DEV" ] && echo -e "${GREEN}    depression (RIGHT) : /dev/depression -> $RIGHT_DEV${NC}"
    # Skip the interactive selection block below by short-circuiting.
    SKIP_CAM_PROMPT=1
fi
echo ""
if [ -n "$SKIP_CAM_PROMPT" ]; then
    :  # pinning already assigned LEFT_DEV / RIGHT_DEV
elif [ ${#VIDEO_DEVICES[@]} -ge 2 ]; then
    echo -e "${CYAN}  Two cameras detected.${NC}"
    echo -e "  ${GREEN}[1]  Left only  : ${VIDEO_DEVICES[0]} — ${VIDEO_LABELS[0]}${NC}"
    echo -e "  ${GREEN}[2]  Right only : ${VIDEO_DEVICES[1]} — ${VIDEO_LABELS[1]}${NC}"
    echo -e "  ${GREEN}[3]  Dual       : both cameras${NC}"
    echo ""
    read -p "  Select camera mode [1/2/3] (default: 3 dual): " CAM_MODE
    CAM_MODE=${CAM_MODE:-3}
    case $CAM_MODE in
        1)  LEFT_DEV="${VIDEO_DEVICES[0]}"; RIGHT_DEV=""
            SELECTED_CAM="${VIDEO_LABELS[0]}"
            echo -e "${CYAN}  Left only : $LEFT_DEV${NC}" ;;
        2)  LEFT_DEV=""; RIGHT_DEV="${VIDEO_DEVICES[1]}"
            SELECTED_CAM="${VIDEO_LABELS[1]}"
            echo -e "${CYAN}  Right only : $RIGHT_DEV${NC}" ;;
        3)  LEFT_DEV="${VIDEO_DEVICES[0]}"; RIGHT_DEV="${VIDEO_DEVICES[1]}"
            SELECTED_CAM="${VIDEO_LABELS[0]}"
            echo -e "${CYAN}  Dual : $LEFT_DEV + $RIGHT_DEV${NC}" ;;
        *)  echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
    esac
else
    # WHY: with only one camera, the L/R prompt was harmful — picking "R"
    # routed the app to the right-side pipeline while the device sits on
    # the left slot, wedging V4L2 until reboot. Auto-assign to LEFT
    # (the well-tested single-camera path) and skip the prompt.
    LEFT_DEV="${VIDEO_DEVICES[0]}"; RIGHT_DEV=""
    SELECTED_CAM="${VIDEO_LABELS[0]}"
    echo -e "${CYAN}  One camera detected — auto-assigned to LEFT: $LEFT_DEV — ${VIDEO_LABELS[0]}${NC}"
fi

# ── Device readiness check ─────────────────
# WHY: check ALL selected cameras — not just the first one.
# If kernel has not released a device, abort immediately.
# This prevents silent single-camera operation in dual mode.
DEVICES_TO_CHECK=()
[ -n "$LEFT_DEV"  ] && DEVICES_TO_CHECK+=("$LEFT_DEV")
[ -n "$RIGHT_DEV" ] && DEVICES_TO_CHECK+=("$RIGHT_DEV")

for DEV in "${DEVICES_TO_CHECK[@]}"; do
    echo -e "${CYAN}  Checking $DEV...${NC}"
    READY=0
    for i in $(seq 1 10); do
        # Read-only probe, explicit close. RW open (3<>) could wedge
        # V4L2 when the driver was mid-init, requiring a reboot.
        if timeout 1 bash -c "exec 3< '$DEV'; exec 3<&-" 2>/dev/null; then
            echo -e "${GREEN}  $DEV ready.${NC}"
            READY=1; break
        fi
        echo -e "${YELLOW}  $DEV not ready (attempt $i/10)...${NC}"
        sleep 1
    done
    if [ "$READY" -eq 0 ]; then
        echo -e "${RED}  $DEV never became ready — kernel has not released it.${NC}"
        echo -e "${RED}  Cannot start in selected camera mode. Aborting.${NC}"
        exit 1
    fi
done

# ── Camera configuration ───────────────────
echo ""
echo -e "${CYAN}Select camera configuration:${NC}"
echo ""
if [ "$SELECTED_CAM" = "Arducam B0510" ]; then
    echo -e "  ${GREEN}[1]  Arducam B0510  — YUYV   1920x1080   60fps${NC}"
    echo ""
    read -p "Select configuration [1]: " CAM_CHOICE
    case $CAM_CHOICE in
        1)  STREAM_FORMAT=YUYV; WIDTH=1920; HEIGHT=1080; FRAMERATE=60; CAM_MODEL=1 ;;
        *)  echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
    esac
else
    echo -e "  ${GREEN}[1]  ECON See3CAM   — UYVY   1280x720    60fps${NC}"
    echo -e "  ${GREEN}[2]  ECON See3CAM   — UYVY   1920x1080   60fps${NC}"
    echo -e "  ${GREEN}[3]  ECON See3CAM   — MJPG   1280x720    60fps${NC}"
    echo -e "  ${GREEN}[4]  ECON See3CAM   — MJPG   1920x1080   60fps${NC}"
    echo -e "  ${GREEN}[5]  ECON See3CAM   — MJPG   1920x1200   60fps${NC}"
    echo ""
    read -p "Select configuration [1-5] (default: 1): " CAM_CHOICE
    CAM_CHOICE=${CAM_CHOICE:-1}
    case $CAM_CHOICE in
        1)  STREAM_FORMAT=UYVY; WIDTH=1280;  HEIGHT=720;  FRAMERATE=60; CAM_MODEL=2 ;;
        2)  STREAM_FORMAT=UYVY; WIDTH=1920;  HEIGHT=1080; FRAMERATE=60; CAM_MODEL=2 ;;
        3)  STREAM_FORMAT=MJPG; WIDTH=1280;  HEIGHT=720;  FRAMERATE=60; CAM_MODEL=2 ;;
        4)  STREAM_FORMAT=MJPG; WIDTH=1920;  HEIGHT=1080; FRAMERATE=60; CAM_MODEL=2 ;;
        5)  STREAM_FORMAT=MJPG; WIDTH=1920;  HEIGHT=1200; FRAMERATE=60; CAM_MODEL=2 ;;
        *)  echo -e "${RED}Invalid selection.${NC}"; exit 1 ;;
    esac
fi

# ── UART permission ────────────────────────
sudo fuser -k ${LEFT_VIDEO_PORT}/udp  2>/dev/null || true
sudo fuser -k ${LEFT_CTRL_PORT}/udp   2>/dev/null || true
sudo fuser -k ${RIGHT_VIDEO_PORT}/udp 2>/dev/null || true
sudo fuser -k ${RIGHT_CTRL_PORT}/udp  2>/dev/null || true

# WHY: force-release camera device files left locked by an unclean
# shutdown of a previous run (e.g. kill -9, crash). Extra insurance
# on top of the GST_STATE_NULL wait added in main.cpp.
for dev in "$LEFT_DEV" "$RIGHT_DEV"; do
    [ -n "$dev" ] && [ -e "$dev" ] && sudo fuser -k "$dev" 2>/dev/null || true
done

sleep 1
if [ -e "$UART" ]; then
    sudo chmod 666 "$UART" 2>/dev/null || true
else
    echo -e "${YELLOW}  [WARN] UART $UART not found — angle output disabled.${NC}"
fi

# ── Downsampling ───────────────────────────
echo ""
read -p "Enable tracker downsampling to 640x480? [Y/n]: " DS_ENABLE
DS_ENABLE=${DS_ENABLE:-Y}
if [ "$DS_ENABLE" = "y" ] || [ "$DS_ENABLE" = "Y" ]; then
    DOWNSAMPLE=1
    echo -e "${GREEN}  Downsampling enabled.${NC}"
else
    DOWNSAMPLE=0
    echo -e "${YELLOW}  Downsampling disabled.${NC}"
fi


# ── Camera Field of View ───────────────────
# WHY: show camModel defaults so operator can just press Enter.
# Must match constants in src/common/globals.h.
if [ "$CAM_MODEL" = "2" ]; then
    DEFAULT_HFOV=104.6
    DEFAULT_VFOV=61.6
else
    DEFAULT_HFOV=68.0
    DEFAULT_VFOV=54.0
fi
echo ""
echo -e "${CYAN}Camera Field of View:${NC}"
echo ""
read -p "  Horizontal FOV (HFOV) [default: ${DEFAULT_HFOV}°]: " HFOV_IN
read -p "  Vertical FOV (VFOV)   [default: ${DEFAULT_VFOV}°]: " VFOV_IN
HFOV=${HFOV_IN:-$DEFAULT_HFOV}
VFOV=${VFOV_IN:-$DEFAULT_VFOV}

# ── Handoff target depth ───────────────────
# Distance from the boresight camera's optical origin to the plane on
# which the target sits, in millimetres. Indoor: measure with a ruler
# (typical 2000 mm for a tabletop test). Airframe: will be replaced
# by barometer AGL in a later change. Blank / 0 disables handoff.
echo ""
echo -e "${CYAN}Handoff target depth (mm):${NC}"
DEFAULT_DEPTH=2000
read -p "  Depth to target plane [default: ${DEFAULT_DEPTH}, blank to disable]: " DEPTH_IN
TARGET_DEPTH_MM=${DEPTH_IN:-$DEFAULT_DEPTH}

# ── Handoff stereo calibration ─────────────
# stereo_calib.json is committed to the repo, generated from the real
# ChArUco stereo calibration in camcalib/combined/. Only the target
# depth (above) is per-run operator input. The old ruler-based
# regeneration was removed — it would have overwritten the real
# calibration on every launch with hand-measured guesses.
if [ -n "$TARGET_DEPTH_MM" ] && [ "$TARGET_DEPTH_MM" != "0" ]; then
    if [ "$DOWNSAMPLE" = "1" ]; then
        TRACKER_W=640; TRACKER_H=480
    else
        TRACKER_W=$WIDTH; TRACKER_H=$HEIGHT
    fi
    SIZED_CALIB="stereo_calib_${TRACKER_W}x${TRACKER_H}.json"
    if [ -f "$SIZED_CALIB" ]; then
        echo -e "${GREEN}  Using ${SIZED_CALIB} (resolution-specific).${NC}"
    elif [ -f stereo_calib.json ]; then
        echo -e "${YELLOW}  ${SIZED_CALIB} missing, falling back to stereo_calib.json.${NC}"
    else
        echo -e "${YELLOW}  WARNING: neither ${SIZED_CALIB} nor stereo_calib.json found — handoff disabled.${NC}"
        TARGET_DEPTH_MM=0
    fi
fi

# ── Appearance verifier (DNN) ──────────────
# Which feature extractor the tracker's drift / re-capture verifier uses.
# Runtime choice (not a build variant) — all engines share the same bank.
#   dino     = DINOv2-small CLS token (default, best in field; needs
#              models/dinov2_small.onnx — see tools/export_dinov2.py. HEAVY ViT.)
#   embedder = trained re-id embedder (discriminative)
#   resnet   = frozen ResNet18 block2 (scale-invariant; needs
#              models/resnet18_block2.onnx — see tools/export_resnet18_block2.py)
#   none     = disable DNN verifier (built-in fallback)
echo ""
echo -e "${CYAN}Appearance verifier:${NC}"
echo -e "  ${GREEN}[1] dino${NC} (default)   ${GREEN}[2] embedder${NC}   ${GREEN}[3] resnet${NC}   ${GREEN}[4] none${NC}"
read -p "  Select verifier [1-4] (default: 1): " VER_CHOICE
case "${VER_CHOICE:-1}" in
    2) VERIFIER=embedder ;;
    3) VERIFIER=resnet ;;
    4) VERIFIER=none ;;
    *) VERIFIER=dino ;;
esac
echo -e "${GREEN}  Verifier: ${VERIFIER}${NC}"

# ── Launch summary ─────────────────────────
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${GREEN}  Starting $(basename "$BIN")${NC}"
echo -e "${CYAN}  Left camera  : ${LEFT_DEV:-disabled}${NC}"
echo -e "${CYAN}  Right camera : ${RIGHT_DEV:-disabled}${NC}"
echo -e "${CYAN}  Resolution   : ${WIDTH}x${HEIGHT}  format:${STREAM_FORMAT}  fps:${FRAMERATE}${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# ── Launch binary ──────────────────────────
# Arguments match main.cpp expectation (16 total arguments)
# TRACKER_VERIFIER selects the appearance verifier model (see main.cpp).
TRACKER_VERIFIER="$VERIFIER" "$BIN" "$IP" \
    "$LEFT_VIDEO_PORT"  "$LEFT_CTRL_PORT" \
    "$RIGHT_VIDEO_PORT" "$RIGHT_CTRL_PORT" \
    "$WIDTH" "$HEIGHT" \
    "$UART" \
    "$LEFT_DEV" "$RIGHT_DEV" \
    "$STREAM_FORMAT" "$FRAMERATE" \
    "$CAM_MODEL" \
    "$( [ "$RECORDING" = "1" ] && realpath "$REC_PATH" || echo "/tmp" )" \
    "$RECORDING" \
    "$DOWNSAMPLE" \
    "$HFOV" "$VFOV" \
    "$TARGET_DEPTH_MM"