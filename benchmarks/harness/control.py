"""CmdPacket sender.

Mirrors the on-the-wire struct from src/common/globals.h:

    #pragma pack(push, 1)
    struct CmdPacket {
        uint32_t magic;
        uint32_t type;
        float    arg1;
        float    arg2;
        float    arg3;
    };
    static constexpr uint32_t CMD_MAGIC = 0x54524B43;

Total 20 bytes, little-endian on the Jetson. CMD_CAPTURE (type=1) uses
arg1=x, arg2=y, arg3=rect_size — same call the operator's mouse click
produces from the GUI.
"""

from __future__ import annotations

import socket
import struct

CMD_MAGIC = 0x54524B43

CMD_CAPTURE = 1
CMD_RESET = 2
CMD_CHANGE_SIZE = 3
CMD_SET_RECT_POS = 4
CMD_SET_PARAM = 5
CMD_SAVE_PARAMS = 6
CMD_HANDOFF = 7
CMD_SET_CAMERA_PARAM = 8
CMD_GET_PARAMS = 9
CMD_CONFIRM_TARGET = 10

_PACKET_FMT = "<IIfff"  # 20 bytes
assert struct.calcsize(_PACKET_FMT) == 20


def send_cmd(host: str, port: int, cmd_type: int,
             arg1: float = 0.0, arg2: float = 0.0, arg3: float = 0.0) -> None:
    """One-shot UDP send. Non-blocking, no ACK wait — the tracker's
    controlThread handles the packet on receipt."""
    pkt = struct.pack(_PACKET_FMT, CMD_MAGIC, cmd_type,
                      float(arg1), float(arg2), float(arg3))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(pkt, (host, port))


def send_capture(host: str, port: int,
                 x: float, y: float, frame_id: float = -1.0) -> None:
    """Send CMD_CAPTURE(x, y, frame_id).

    Correction: arg3 is the frame id in the tracker's ring buffer, not
    the rect size. -1 = 'capture on the latest frame'. Rect size is set
    via SET_PARAM RECT_WIDTH / RECT_HEIGHT before CAPTURE — see
    send_set_rect_size below."""
    send_cmd(host, port, CMD_CAPTURE, x, y, frame_id)


# Param IDs (must match VTrackerParam in cvtracker/VTracker.h). The
# first three are the SEARCH_WINDOW/RECT trio at the top of the enum
# and are stable across engine variants.
PARAM_SEARCH_WINDOW_WIDTH  = 1
PARAM_SEARCH_WINDOW_HEIGHT = 2
PARAM_RECT_WIDTH           = 3
PARAM_RECT_HEIGHT          = 4


def send_set_param(host: str, port: int, param_id: int, value: float) -> None:
    """Send CMD_SET_PARAM(param_id, value)."""
    send_cmd(host, port, CMD_SET_PARAM, float(param_id), float(value))


def send_set_rect_size(host: str, port: int,
                       width: float, height: float) -> None:
    """Convenience: two SET_PARAM calls, RECT_WIDTH then RECT_HEIGHT."""
    send_set_param(host, port, PARAM_RECT_WIDTH,  width)
    send_set_param(host, port, PARAM_RECT_HEIGHT, height)
