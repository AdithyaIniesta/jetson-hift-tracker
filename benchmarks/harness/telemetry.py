"""TelemetryPacket receiver.

Layout mirrors src/common/globals.h. Fields in order (all 4 bytes each,
little-endian). Total struct size = 160 bytes. Only a handful of fields
are useful for benchmarking (frame_id, mode, det_prob, rect_*, object_*);
the rest are decoded for completeness / future analysis and dumped raw.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional


TELEMETRY_MAGIC = 0x544C4D54

# Field-by-field format — kept explicit so the layout is auditable against
# the C++ struct without cross-referencing.
_FMT_TOKENS = (
    ("I", "magic"),
    ("I", "frame_id"),
    ("I", "mode"),
    ("f", "det_prob"),
    ("I", "proc_time_us"),
    ("i", "rect_x"),
    ("i", "rect_y"),
    ("i", "rect_width"),
    ("i", "rect_height"),
    ("i", "object_x"),
    ("i", "object_y"),
    ("i", "object_width"),
    ("i", "object_height"),
    ("f", "angle_x_deg"),
    ("f", "angle_y_deg"),
    ("i", "pixel_offset_x"),
    ("i", "pixel_offset_y"),
    ("i", "search_win_width"),
    ("i", "search_win_height"),
    ("f", "lost_mode_option"),
    ("i", "frame_buffer_size"),
    ("i", "max_frames_lost"),
    ("f", "rect_auto_size"),
    ("f", "rect_auto_position"),
    ("i", "multiple_threads"),
    ("i", "num_channels"),
    ("i", "tracker_type"),
    ("f", "custom_1"),
    ("f", "custom_2"),
    ("f", "custom_3"),
    ("f", "ekf_sigma_a"),
    ("f", "ekf_sigma_alpha"),
    ("f", "ekf_r_base"),
    ("i", "dnn_verify_interval"),
    ("f", "dnn_veto_threshold"),
    ("f", "dnn_accept_threshold"),
    ("f", "dnn_similarity"),
    ("i", "frame_width"),
    ("i", "frame_height"),
    ("I", "reserved"),
)

_PACKET_FMT = "<" + "".join(t[0] for t in _FMT_TOKENS)
_FIELD_NAMES = tuple(t[1] for t in _FMT_TOKENS)
PACKET_SIZE = struct.calcsize(_PACKET_FMT)
assert PACKET_SIZE == 160, f"expected 160 bytes, got {PACKET_SIZE}"


@dataclass
class Telemetry:
    """A decoded packet with the fields we actually care about at the top.

    All fields from the wire are stored in `.raw`. The convenience fields
    exposed as attributes are the ones the benchmark metrics touch."""

    frame_id: int
    mode: int
    det_prob: float
    rect_x: int
    rect_y: int
    rect_width: int
    rect_height: int
    raw: dict

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["Telemetry"]:
        if len(data) != PACKET_SIZE:
            return None
        values = struct.unpack(_PACKET_FMT, data)
        if values[0] != TELEMETRY_MAGIC:
            return None
        raw = dict(zip(_FIELD_NAMES, values))
        return cls(
            frame_id=raw["frame_id"],
            mode=raw["mode"],
            det_prob=raw["det_prob"],
            rect_x=raw["rect_x"],
            rect_y=raw["rect_y"],
            rect_width=raw["rect_width"],
            rect_height=raw["rect_height"],
            raw=raw,
        )


class TelemetryListener:
    """Background UDP receiver. Every valid TelemetryPacket that arrives
    is handed to `on_packet(Telemetry)`. Runs until `stop()` or the
    context-manager exit."""

    def __init__(self, bind_port: int, on_packet: Callable[[Telemetry], None],
                 bind_host: str = "0.0.0.0"):
        self.bind_addr = (bind_host, bind_port)
        self.on_packet = on_packet
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def __enter__(self) -> "TelemetryListener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.bind_addr)
        # Short recv timeout so stop() responds quickly.
        self._sock.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            tp = Telemetry.from_bytes(data)
            if tp is not None:
                try:
                    self.on_packet(tp)
                except Exception:  # pragma: no cover — never crash the listener
                    pass

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def telemetry_to_dict(t: Telemetry) -> dict:
    """Serialisable form for JSONL output. Includes wall-clock ts."""
    d = {"ts": time.time(), "frame_id": t.frame_id, "mode": t.mode,
         "det_prob": t.det_prob,
         "rect_x": t.rect_x, "rect_y": t.rect_y,
         "rect_width": t.rect_width, "rect_height": t.rect_height}
    return d
