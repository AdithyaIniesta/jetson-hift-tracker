"""Spawn the tracker binary as a subprocess with the argv layout that
main.cpp expects. The whole point is to avoid touching C++, so we mimic
what run_jp6.sh does — just with recording disabled, UART pointed at a
dummy device, and the input camera pointing at the loopback.

argv order (must match main.cpp):
  [0]  binary path
  [1]  clientIp
  [2]  leftVideoPort   [3] leftCtrlPort
  [4]  rightVideoPort  [5] rightCtrlPort
  [6]  W               [7] H
  [8]  uartDev
  [9]  leftDev         [10] rightDev
  [11] streamFormat    [12] fps
  [13] camModel        [14] recBasePath   [15] recEnabled
  [16] downsample      [17] hfov_deg      [18] vfov_deg
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional


DEFAULT_BIN_DIR = Path("build/bin")

ENGINE_TO_BINARY = {
    "constant_robotics_lib": "JetsonTracker_constant_robotics_lib_orin",
    "cuda_library":          "JetsonTracker_cuda_library_orin",
    "lockon":                "JetsonTracker_lockon_orin",
}


def resolve_binary(engine: str, bin_dir: Path = DEFAULT_BIN_DIR) -> Path:
    if engine not in ENGINE_TO_BINARY:
        raise KeyError(
            f"unknown engine {engine!r}; known: {sorted(ENGINE_TO_BINARY)}"
        )
    path = bin_dir / ENGINE_TO_BINARY[engine]
    if not path.is_file():
        raise FileNotFoundError(
            f"tracker binary {path} not found. Have you run ./build.sh?"
        )
    return path


class TrackerProcess:
    """Wrapper around the tracker binary. Never touches its stdin. Prints
    to stdout are captured to `log_path` if given."""

    def __init__(self, engine: str, client_ip: str = "127.0.0.1",
                 left_video_port: int = 5000, left_ctrl_port: int = 5001,
                 right_video_port: int = 5002, right_ctrl_port: int = 5003,
                 width: int = 1280, height: int = 720,
                 left_dev: str = "/dev/video20", right_dev: str = "",
                 stream_format: str = "UYVY", fps: int = 60,
                 cam_model: int = 2,
                 hfov: float = 104.6, vfov: float = 61.6,
                 downsample: bool = True,
                 bin_dir: Path = DEFAULT_BIN_DIR,
                 log_path: Optional[Path] = None,
                 file_src_left: Optional[str] = None,
                 file_src_right: Optional[str] = None,
                 file_src_start_index: int = 1,
                 telem_port_override: Optional[int] = None):
        self.binary = resolve_binary(engine, bin_dir=bin_dir)
        self.client_ip = client_ip
        self.ports = (left_video_port, left_ctrl_port,
                      right_video_port, right_ctrl_port)
        self.width, self.height = width, height
        self.left_dev, self.right_dev = left_dev, right_dev
        self.stream_format, self.fps = stream_format, fps
        self.cam_model = cam_model
        self.hfov, self.vfov = hfov, vfov
        self.downsample = downsample
        self.log_path = log_path
        self.file_src_left = file_src_left
        self.file_src_right = file_src_right
        self.file_src_start_index = file_src_start_index
        self.telem_port_override = telem_port_override
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    def __enter__(self) -> "TrackerProcess":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _argv(self) -> list:
        # UART -> /dev/null: no airframe. Tracker prints WARN and moves on.
        # recBasePath -> /tmp, recEnabled=0: no recording.
        return [str(self.binary),
                self.client_ip,
                str(self.ports[0]), str(self.ports[1]),
                str(self.ports[2]), str(self.ports[3]),
                str(self.width), str(self.height),
                "/dev/null",
                self.left_dev, self.right_dev,
                self.stream_format, str(self.fps),
                str(self.cam_model),
                "/tmp", "0",
                "1" if self.downsample else "0",
                str(self.hfov), str(self.vfov)]

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("wb")
            stdout = stderr = self._log_fh
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        env = os.environ.copy()
        if self.file_src_left:
            env["TRACKER_FILE_SRC_LEFT"] = self.file_src_left
            env["TRACKER_FILE_SRC"] = self.file_src_left
        if self.file_src_right:
            env["TRACKER_FILE_SRC_RIGHT"] = self.file_src_right
        if self.file_src_left or self.file_src_right:
            env["TRACKER_FILE_SRC_START_INDEX"] = str(self.file_src_start_index)
        if self.telem_port_override is not None:
            env["TRACKER_TELEM_PORT_OVERRIDE"] = str(self.telem_port_override)
        # Isolate in its own process group so we can SIGINT the whole
        # tree even if the tracker spawns children (it does: GStreamer).
        self._proc = subprocess.Popen(self._argv(),
                                      stdout=stdout, stderr=stderr,
                                      env=env,
                                      preexec_fn=os.setsid)

    def wait_ready(self, seconds: float = 5.0) -> None:
        """Give the tracker time to bind its UDP sockets before we start
        firing CMDs at it. There's no real ready signal; a fixed sleep is
        pragmatic."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("tracker exited before becoming ready")
            time.sleep(0.1)

    def stop(self, sigint_grace: float = 3.0) -> int:
        """SIGINT (clean shutdown path in main.cpp), then SIGTERM if it
        won't leave. Returns the exit code, or -1 if we had to KILL."""
        if self._proc is None:
            return 0
        rc = 0
        try:
            if self._proc.poll() is None:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    rc = self._proc.wait(timeout=sigint_grace)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        rc = self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        rc = -1
            else:
                rc = self._proc.returncode
        finally:
            self._proc = None
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except OSError:
                    pass
                self._log_fh = None
        return rc
