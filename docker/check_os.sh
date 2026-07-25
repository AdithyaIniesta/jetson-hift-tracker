#!/usr/bin/env bash
# =============================================================================
# check_os.sh — Pre-flight OS validation for JetsonTracker Docker deployment
# Run this BEFORE loading or running the Docker image on any Jetson device.
# =============================================================================

PASS=0
FAIL=0
WARN=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✔ $*${NC}"; ((PASS++)); }
fail() { echo -e "  ${RED}✘ $*${NC}"; ((FAIL++)); }
warn() { echo -e "  ${YELLOW}⚠ $*${NC}"; ((WARN++)); }
info() { echo -e "  ${CYAN}ℹ $*${NC}"; }
section() { echo ""; echo -e "${CYAN}── $* ──────────────────────────────────────${NC}"; }

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}   JetsonTracker — OS Pre-flight Check      ${NC}"
echo -e "${CYAN}============================================${NC}"

# ── 1. JetPack / L4T Version ─────────────────────────────────────────────────
section "JetPack / L4T"

if [ -f /etc/nv_tegra_release ]; then
    L4T=$(cat /etc/nv_tegra_release)
    REVISION=$(echo "$L4T" | grep -oP 'REVISION: \K[0-9]+\.[0-9]+')
    RELEASE=$(echo "$L4T" | grep -oP 'R\K[0-9]+')
    pass "L4T detected: R${RELEASE}.${REVISION}"
    if [ "$RELEASE" -ge 35 ]; then
        pass "JetPack 5.x confirmed (R${RELEASE}) — supported"
    else
        fail "JetPack version R${RELEASE} not supported — requires R35 or higher"
    fi
else
    fail "/etc/nv_tegra_release not found — is this a Jetson device?"
fi

# ── 2. Kernel ─────────────────────────────────────────────────────────────────
section "Kernel"

KERNEL=$(uname -r)
echo "  Kernel: $KERNEL"
if echo "$KERNEL" | grep -q "tegra"; then
    pass "NVIDIA tegra kernel running — NVIDIA modules available"
else
    fail "Custom kernel detected ($KERNEL) — NVIDIA modules may be missing"
    fail "Run: sudo cp /boot/Image.t19x /boot/Image && sudo reboot"
fi

# ── 3. NVIDIA Kernel Modules ──────────────────────────────────────────────────
section "NVIDIA Kernel Modules"

if [ -e /dev/nvhost-msenc ]; then
    pass "NVIDIA multimedia device /dev/nvhost-msenc found — HW encoder ready"
else
    MODULES=$(ls /lib/modules/$(uname -r)/extra/ 2>/dev/null)
    if [ -z "$MODULES" ]; then
        fail "/dev/nvhost-msenc not found — NVIDIA multimedia kernel modules missing"
    else
        fail "/dev/nvhost-msenc not found — HW encoder will not work"
    fi
fi

# ── 4. CUDA ───────────────────────────────────────────────────────────────────
section "CUDA"

if [ -f /usr/local/cuda/bin/nvcc ]; then
    CUDA_VER=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    pass "CUDA $CUDA_VER found at /usr/local/cuda"
    if [[ "$CUDA_VER" == 11.* ]]; then
        pass "CUDA 11.x confirmed — compatible with JetPack 5"
    else
        warn "CUDA $CUDA_VER detected — expected 11.x for JetPack 5"
    fi
else
    fail "nvcc not found — CUDA not installed or not in PATH"
fi

# ── 5. Docker ─────────────────────────────────────────────────────────────────
section "Docker"

if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | grep -oP '[0-9]+.[0-9]+.[0-9]+' | head -1)
    pass "Docker $DOCKER_VER installed"
else
    fail "Docker not installed — install with: sudo apt-get install docker.io"
fi

if docker info &>/dev/null; then
    pass "Docker daemon running"
else
    fail "Docker daemon not running — run: sudo systemctl start docker"
fi

# ── 6. NVIDIA Container Toolkit ───────────────────────────────────────────────
section "NVIDIA Container Toolkit"

if dpkg -l | grep -q nvidia-container-toolkit; then
    TOOLKIT_VER=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
    pass "nvidia-container-toolkit $TOOLKIT_VER installed"
else
    fail "nvidia-container-toolkit not installed"
    fail "Install: sudo apt-get install nvidia-container-toolkit"
fi

if docker info 2>/dev/null | grep -q "nvidia"; then
    pass "NVIDIA runtime registered with Docker"
else
    fail "NVIDIA runtime not registered — run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# ── 7. GStreamer NVIDIA Plugins ───────────────────────────────────────────────
section "GStreamer NVIDIA Plugins"

if gst-inspect-1.0 nvv4l2h264enc &>/dev/null; then
    pass "nvv4l2h264enc (HW encoder) available"
else
    fail "nvv4l2h264enc not found — hardware encoder missing"
    fail "This device cannot run the tracker pipeline"
fi

if gst-inspect-1.0 nvv4l2decoder &>/dev/null; then
    pass "nvv4l2decoder (HW decoder) available"
else
    warn "nvv4l2decoder not found — hardware decoder missing"
fi

if gst-inspect-1.0 nvvidconv &>/dev/null; then
    pass "nvvidconv available"
else
    fail "nvvidconv not found"
fi

# ── 8. USB Camera ─────────────────────────────────────────────────────────────
section "USB Camera"

if ls /dev/video* &>/dev/null; then
    for dev in /dev/video*; do
        CAM=$(v4l2-ctl --device=$dev --info 2>/dev/null | grep "Card type" | awk -F: '{print $2}' | xargs)
        if [ -n "$CAM" ]; then
            pass "$dev — $CAM"
        fi
    done
else
    fail "No video devices found — USB camera not connected"
fi

# ── 9. UART ───────────────────────────────────────────────────────────────────
section "UART"

if ls /dev/ttyTHS* &>/dev/null; then
    for uart in /dev/ttyTHS*; do
        PERM=$(ls -la $uart | awk '{print $1}')
        if echo "$PERM" | grep -q "rw-rw-rw-"; then
            pass "$uart — permission OK"
        else
            warn "$uart — permission restricted ($PERM)"
            info "Auto-fixing: sudo chmod 666 $uart"
        sudo chmod 666 $uart 2>/dev/null && pass "Auto-fixed: $uart permission set to 666"
        fi
    done
else
    warn "No UART devices found at /dev/ttyTHS*"
fi

# ── 10. Network ───────────────────────────────────────────────────────────────
section "Network"

IP=$(hostname -I | awk '{print $1}')
if [ -n "$IP" ]; then
    pass "Device IP: $IP"
else
    fail "No network interface detected"
fi

for port in 5000 5001; do
    if sudo fuser $port/udp &>/dev/null 2>&1; then
        warn "Port $port/udp already in use — run: sudo fuser -k $port/udp"
    else
        pass "Port $port/udp is free"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "  Results: ${GREEN}${PASS} passed${NC}  ${YELLOW}${WARN} warnings${NC}  ${RED}${FAIL} failed${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Device is NOT ready. Fix the failed checks before running Docker.${NC}"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "${YELLOW}Device is ready with warnings. Review warnings before running.${NC}"
    exit 0
else
    echo -e "${GREEN}Device is fully ready. Run the Docker image.${NC}"
    exit 0
fi
