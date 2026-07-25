#!/usr/bin/env bash
# port_check.sh — passive UDP traffic monitor for JetsonTracker ports
# Like a multimeter in parallel — listens without interfering
# Run on the JETSON while video16.py is running on the ground station

LEFT_VIDEO=5000
LEFT_CTRL=5001
RIGHT_VIDEO=5002
RIGHT_CTRL=5003
DURATION=5  # seconds to sample each port

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  JetsonTracker port monitor${NC}"
echo -e "${CYAN}  Sampling each port for ${DURATION}s...${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

check_port() {
    local port=$1
    local label=$2
    local direction=$3

    # Count packets seen on this port for DURATION seconds
    count=$(timeout $DURATION tcpdump -i any -q "udp port $port" 2>/dev/null | wc -l)

    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}[OK]${NC}  Port $port  ($label / $direction)  — $count packets in ${DURATION}s"
    else
        echo -e "  ${RED}[--]${NC}  Port $port  ($label / $direction)  — no traffic detected"
    fi
}

echo -e "${CYAN}Checking video stream ports (Jetson → Ground station):${NC}"
echo ""
check_port $LEFT_VIDEO  "left video"  "outbound RTP/H264"
check_port $RIGHT_VIDEO "right video" "outbound RTP/H264"

echo ""
echo -e "${CYAN}Checking control/telemetry ports (Ground station ↔ Jetson):${NC}"
echo ""
check_port $LEFT_CTRL  "left ctrl/telem"  "inbound CMD / outbound TELEM"
check_port $RIGHT_CTRL "right ctrl/telem" "inbound CMD / outbound TELEM"

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${YELLOW}Note: run with sudo if tcpdump requires it${NC}"
echo -e "${YELLOW}      ./port_check.sh  or  sudo ./port_check.sh${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""