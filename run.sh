#!/usr/bin/env bash
# Deprecated. Use the per-JetPack launcher — see project_jetpack_uart_mapping.
#
#   JetPack 5  →  ./run_jp5.sh   (UART /dev/ttyTHS0)
#   JetPack 6  →  ./run_jp6.sh   (UART /dev/ttyTHS2)
#
# The old single run.sh tried to autodetect the UART node and silently
# picked the wrong one on some Jetsons, so angle output went nowhere.
# Split it, hardcoded the mapping, and made the operator pick the script.
RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo -e "${RED}run.sh is retired.${NC}" >&2
echo -e "${YELLOW}Use ./run_jp5.sh on JetPack 5, ./run_jp6.sh on JetPack 6.${NC}" >&2
exit 2
