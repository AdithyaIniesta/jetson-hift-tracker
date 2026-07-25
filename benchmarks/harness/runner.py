"""Orchestrates one sequence run start-to-finish, via TRACKER_FILE_SRC_LEFT.

  1. Kill any zombie tracker holding CTRL ports.
  2. Spawn tracker binary with TRACKER_FILE_SRC_LEFT pointing at the
     sequence's JPG folder pattern. Pipeline reads via multifilesrc,
     no v4l2 involved.
  3. Start telemetry listener on ctrl_port + 10000 (offset avoids the
     tracker's own CTRL socket on localhost).
  4. Wait a beat for the ring buffer, then SET_PARAM RECT_WIDTH/HEIGHT
     and CMD_CAPTURE(x, y, -1) using the sequence's init bbox.
  5. Wait for multifilesrc EOS (or a max-wait timeout).
  6. Stop tracker (SIGINT), stop listener.
  7. Write predictions.jsonl.

The listener keeps every TelemetryPacket. Frame-index alignment happens
at metric time by counting from the first packet whose mode >= 1.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import subprocess

from benchmarks.datasets.base import Sequence
from benchmarks.harness.control import send_capture, send_set_rect_size
from benchmarks.harness.telemetry import Telemetry, TelemetryListener, telemetry_to_dict
from benchmarks.harness.tracker_process import TrackerProcess


DEFAULT_LOOPBACK = "/dev/video20"


@dataclass
class RunConfig:
    engine: str = "lockon"
    client_ip: str = "127.0.0.1"
    left_ctrl_port: int = 5001
    left_video_port: int = 5000
    right_ctrl_port: int = 5003
    right_video_port: int = 5002
    tracker_width: int = 1280
    tracker_height: int = 720
    tracker_fps: int = 60
    hfov: float = 104.6
    vfov: float = 61.6
    cam_model: int = 2
    downsample: bool = True
    loopback: str = DEFAULT_LOOPBACK
    bin_dir: Path = Path("build/bin")

    # tuning knobs
    tracker_ready_wait: float = 5.0
    frames_before_capture: int = 30       # ~ 0.5 s at 60fps
    max_wait_seconds: float = 600.0       # per-sequence cap


def _bbox_to_capture_args(bbox: tuple, sx: float, sy: float) -> tuple:
    """Convert dataset (x, y, w, h) at the frame's native size into
    (centre_x, centre_y, rect_w, rect_h) at the tracker's OP resolution.
    Rect size is set via SET_PARAM RECT_WIDTH/HEIGHT before CAPTURE.
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) * sx
    cy = (y + h / 2.0) * sy
    return cx, cy, w * sx, h * sy


def run_sequence(seq: Sequence, out_dir: Path,
                 cfg: RunConfig = RunConfig(),
                 tracker_log: Optional[Path] = None) -> Path:
    """Run one sequence end-to-end via TRACKER_FILE_SRC_LEFT. Returns
    path to the predictions JSONL. The v4l2loopback + feeder path was
    removed — datasets must expose a direct_source (frames_dir, pattern,
    start_index) so the tracker reads JPGs via multifilesrc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"

    packets: List[Telemetry] = []

    def on_packet(pkt: Telemetry) -> None:
        packets.append(pkt)

    tracker_log = tracker_log or (out_dir / "tracker.log")

    # cfg.tracker_width/_height are the capture dims. Under downsample
    # the tracker's internal coordinate space is 640x480, so CAPTURE
    # args must be in that space too.
    op_w = 640 if cfg.downsample else cfg.tracker_width
    op_h = 480 if cfg.downsample else cfg.tracker_height
    sx = op_w / float(seq.width)
    sy = op_h / float(seq.height)

    if seq.direct_source is None:
        raise RuntimeError(
            f"Sequence '{seq.name}' has no direct_source. The v4l2loopback "
            "feeder was removed; datasets must supply a JPG folder + "
            "printf-style pattern for multifilesrc.")
    src_dir, pattern, start_index = seq.direct_source
    file_src_left = str(Path(src_dir) / pattern)
    file_src_start_index = start_index

    # On localhost the tracker's CTRL binds cfg.left_ctrl_port for
    # command receive. Route telemetry to a distinct port via
    # TRACKER_TELEM_PORT_OVERRIDE so the harness's listener doesn't
    # collide with the tracker's CTRL socket.
    telem_port = cfg.left_ctrl_port + 10000

    tracker = TrackerProcess(
        engine=cfg.engine, client_ip=cfg.client_ip,
        left_video_port=cfg.left_video_port,
        left_ctrl_port=cfg.left_ctrl_port,
        right_video_port=cfg.right_video_port,
        right_ctrl_port=cfg.right_ctrl_port,
        width=cfg.tracker_width, height=cfg.tracker_height,
        left_dev="/dev/video0", right_dev="",
        stream_format="MJPG",
        fps=cfg.tracker_fps,
        cam_model=cfg.cam_model,
        hfov=cfg.hfov, vfov=cfg.vfov,
        downsample=cfg.downsample,
        bin_dir=cfg.bin_dir, log_path=tracker_log,
        file_src_left=file_src_left,
        file_src_start_index=file_src_start_index,
        telem_port_override=telem_port)

    listener = TelemetryListener(bind_port=telem_port,
                                 on_packet=on_packet)

    start_ts = time.time()
    try:
        # Kill any zombie tracker holding CTRL ports 5001/5003.
        try:
            subprocess.run(["pkill", "-9", "-f", "JetsonTracker"],
                           check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except FileNotFoundError:
            pass

        tracker.start()
        tracker.wait_ready(cfg.tracker_ready_wait)
        listener.start()

        # Give the tracker's ring buffer a few frames before firing CAPTURE.
        time.sleep(cfg.frames_before_capture / float(cfg.tracker_fps))

        cx, cy, rw, rh = _bbox_to_capture_args(seq.init_bbox, sx, sy)
        send_set_rect_size(cfg.client_ip, cfg.left_ctrl_port, rw, rh)
        time.sleep(0.05)
        send_capture(cfg.client_ip, cfg.left_ctrl_port, cx, cy, -1.0)

        # Wait for the tracker to exit on its own (multifilesrc EOS),
        # or the max-wait timer to fire.
        deadline = time.time() + cfg.max_wait_seconds
        while time.time() < deadline:
            if tracker._proc is None or tracker._proc.poll() is not None:
                break
            time.sleep(0.2)
    finally:
        time.sleep(0.5)
        listener.stop()
        tracker.stop()

    # Persist predictions.
    with predictions_path.open("w") as f:
        for pkt in packets:
            f.write(json.dumps(telemetry_to_dict(pkt)) + "\n")

    meta = {
        "sequence": seq.name,
        "engine": cfg.engine,
        "frames_in_sequence": len(seq.frames),
        "predictions_recorded": len(packets),
        "scale_x": sx, "scale_y": sy,
        "init_bbox_dataset": list(seq.init_bbox),
        "init_bbox_tracker": [cx, cy, rw, rh],
        "wallclock_seconds": time.time() - start_ts,
        "tracker_width": cfg.tracker_width,
        "tracker_height": cfg.tracker_height,
        "dataset_width": seq.width,
        "dataset_height": seq.height,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return predictions_path
