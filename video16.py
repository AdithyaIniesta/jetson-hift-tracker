#!/usr/bin/env python3
"""
CvTracker ground-station UI  -  two-column layout
  LEFT  : Live Telemetry + ACK/Command Log
  RIGHT : Video feed (fills available space) + Controls panel below it

Talks to the JetsonTracker app:
  - receives the H.264/RTP video stream      (UDP  port 5000)
  - sends CmdPacket control commands          (UDP  port 5001)
  - receives TelemetryPacket + ACKs           (UDP  port 5001, same socket)

Interactions:
  - Left click  on video : CAPTURE at that pixel
  - Left drag   on video : SET_PARAM RECT_WIDTH/HEIGHT then CAPTURE at centre
  - Right click on video : RESET
  - Buttons / param panel: size, position, any VTrackerParam
"""

import os
import queue
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk

# --------------------------------------------------------------------------
# Protocol constants  (must match main.cpp)
# --------------------------------------------------------------------------
CMD_MAGIC       = 0x54524B43
ACK_MAGIC       = 0x41434B00
TELEMETRY_MAGIC = 0x544C4D54

CMD_CAPTURE      = 1
CMD_RESET        = 2
CMD_CHANGE_SIZE  = 3
CMD_SET_RECT_POS = 4
CMD_SET_PARAM    = 5
CMD_SAVE_PARAMS  = 6

CMD_NAMES = {
    CMD_CAPTURE:      "CAPTURE",
    CMD_RESET:        "RESET",
    CMD_CHANGE_SIZE:  "CHANGE_SIZE",
    CMD_SET_RECT_POS: "SET_RECT_POS",
    CMD_SET_PARAM:    "SET_PARAM",
    CMD_SAVE_PARAMS:  "SAVE_PARAMS",
}

PARAMS = {
    "SEARCH_WINDOW_WIDTH":      1,
    "SEARCH_WINDOW_HEIGHT":     2,
    "RECT_WIDTH":               3,
    "RECT_HEIGHT":              4,
    "LOST_MODE_OPTION":         5,
    "FRAME_BUFFER_SIZE":        6,
    "MAX_FRAMES_IN_LOST_MODE":  7,
    "RECT_AUTO_SIZE":           8,
    "RECT_AUTO_POSITION":       9,
    "MULTIPLE_THREADS":         10,
    "NUM_CHANNELS":             11,
    "TYPE":                     12,
    "CUSTOM_1 (loss thr)":      13,
    "CUSTOM_2 (recapture thr)": 14,
    "CUSTOM_3 (learning rate)": 15,
}
PARAM_NAME_BY_ID = {v: k for k, v in PARAMS.items()}

TELEM_FMT  = ("<IIIfI"
              "iiii"
              "iiii"
              "ffii"
              "ii"
              "fii"
              "ff"
              "iii"
              "fff"
              "ii"
              "I")
TELEM_SIZE = struct.calcsize(TELEM_FMT)   # 132 bytes

TELEM_FIELDS = [
    "magic", "frame_id", "mode", "det_prob", "proc_time_us",
    "rect_x", "rect_y", "rect_width", "rect_height",
    "object_x", "object_y", "object_width", "object_height",
    "angle_x_deg", "angle_y_deg", "pixel_offset_x", "pixel_offset_y",
    "search_win_width", "search_win_height",
    "lost_mode_option", "frame_buffer_size", "max_frames_lost",
    "rect_auto_size", "rect_auto_position",
    "multiple_threads", "num_channels", "tracker_type",
    "custom_1", "custom_2", "custom_3",
    "frame_width", "frame_height",
    "reserved",
]

MODE_NAMES = {0: "FREE", 1: "TRACKING", 2: "LOST", 3: "INERTIAL", 4: "STATIC"}

TELEM_DISPLAY = [
    ("Frame ID",          "frame_id",           "{:d}"),
    ("Mode",              "mode",               "{}"),
    ("Det. Probability",  "det_prob",           "{:.3f}"),
    ("Proc Time (us)",    "proc_time_us",       "{:d}"),
    ("Rect X",            "rect_x",             "{:d}"),
    ("Rect Y",            "rect_y",             "{:d}"),
    ("Rect Width",        "rect_width",         "{:d}"),
    ("Rect Height",       "rect_height",        "{:d}"),
    ("Object X",          "object_x",           "{:d}"),
    ("Object Y",          "object_y",           "{:d}"),
    ("Object Width",      "object_width",       "{:d}"),
    ("Object Height",     "object_height",      "{:d}"),
    ("Angle X (deg)",     "angle_x_deg",        "{:.2f}"),
    ("Angle Y (deg)",     "angle_y_deg",        "{:.2f}"),
    ("Pixel Offset X",    "pixel_offset_x",     "{:d}"),
    ("Pixel Offset Y",    "pixel_offset_y",     "{:d}"),
    ("Search Win W",      "search_win_width",   "{:d}"),
    ("Search Win H",      "search_win_height",  "{:d}"),
    ("Lost Mode Option",  "lost_mode_option",   "{:.0f}"),
    ("Frame Buffer Size", "frame_buffer_size",  "{:d}"),
    ("Max Frames Lost",   "max_frames_lost",    "{:d}"),
    ("Rect Auto Size",    "rect_auto_size",     "{:.0f}"),
    ("Rect Auto Pos",     "rect_auto_position", "{:.0f}"),
    ("Threads",           "multiple_threads",   "{:d}"),
    ("Channels",          "num_channels",       "{:d}"),
    ("Tracker Type",      "tracker_type",       "{:d}"),
    ("Custom 1 (loss)",   "custom_1",           "{:.4f}"),
    ("Custom 2 (recap.)", "custom_2",           "{:.4f}"),
    ("Custom 3 (lr)",     "custom_3",           "{:.4f}"),
    ("Frame Width",       "frame_width",        "{:d}"),
    ("Frame Height",      "frame_height",       "{:d}"),
]


# ==========================================================================
# Control  -  UDP command sender + ACK + Telemetry receiver (all on port 5001)
# ==========================================================================
class Control:
    def __init__(self, log_cb, telem_cb=None, listen_port=5001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', listen_port))
        self.sock.settimeout(0.2)

        self.addr     = ("192.168.0.20", 5001)
        self.log_cb   = log_cb
        self.telem_cb = telem_cb
        self.running  = True
        threading.Thread(target=self._rx_loop, daemon=True).start()
        log_cb(f"[ctrl] listening for telem/ACK on UDP :{listen_port}")

    def set_target(self, ip, port):
        self.addr = (ip, int(port))

    def send(self, ctype, a1=0.0, a2=0.0, a3=0.0):
        pkt = struct.pack("<IIfff", CMD_MAGIC, ctype, a1, a2, a3)
        try:
            self.sock.sendto(pkt, self.addr)
            self.log_cb(f"-> {CMD_NAMES.get(ctype, ctype)} "
                        f"({a1:g}, {a2:g}, {a3:g})")
        except OSError as e:
            self.log_cb(f"!! send failed: {e}")

    def _rx_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                continue

            if len(data) == 24:
                try:
                    magic, ack_type, param_id, value, success, _ = \
                        struct.unpack("<IIIfII", data)
                except struct.error:
                    continue
                if magic != ACK_MAGIC:
                    continue
                ok = "OK" if success else "FAIL"
                if ack_type == CMD_SET_PARAM:
                    pname = PARAM_NAME_BY_ID.get(param_id, f"id {param_id}")
                    self.log_cb(f"<- ACK SET_PARAM {pname}: {ok}, val={value:g}")
                else:
                    self.log_cb(f"<- ACK {CMD_NAMES.get(ack_type, ack_type)}: {ok}")

            elif len(data) == TELEM_SIZE:
                try:
                    values = struct.unpack_from(TELEM_FMT, data)
                except struct.error:
                    continue
                if values[0] != TELEMETRY_MAGIC:
                    continue
                telem = dict(zip(TELEM_FIELDS, values))
                if self.telem_cb:
                    self.telem_cb(telem)

    def shutdown(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


# ==========================================================================
# Video receiver  -  H.264 / RTP via GStreamer or ffmpeg subprocess
# ==========================================================================
class VideoReceiver:
    def __init__(self, on_frame, on_status, on_log=None):
        self.on_frame  = on_frame
        self.on_status = on_status
        self.on_log    = on_log or (lambda s: None)
        self.port    = 5000
        self.width   = 640
        self.height  = 480
        self.running = True
        self.restart = threading.Event()
        self._controls_synced = False
        self.proc    = None
        self._lock   = threading.Lock()
        self._stderr_lines: list = []
        threading.Thread(target=self._loop, daemon=True).start()

    def configure(self, port, width, height):
        self.port   = int(port)
        self.width  = int(width)
        self.height = int(height)
        self.restart.set()
        self._kill()

    @staticmethod
    def _find_gst():
        import shutil
        exe = shutil.which("gst-launch-1.0")
        if exe:
            return exe
        for c in (
            r"C:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe",
            r"C:\gstreamer\1.0\mingw_x86_64\bin\gst-launch-1.0.exe",
            r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe",
        ):
            if os.path.exists(c):
                return c
        return None

    @staticmethod
    def _find_ffmpeg():
        import shutil
        return shutil.which("ffmpeg")

    def _gst_cmd(self, exe):
        return [exe, "-q",
                "udpsrc", f"port={self.port}",
                "caps=application/x-rtp,media=video,encoding-name=H264,payload=96",
                "!", "rtph264depay", "!", "avdec_h264",
                "!", "videoconvert", "!", "videoscale",
                "!", f"video/x-raw,format=BGR,width={self.width},height={self.height}",
                "!", "fdsink", "fd=1"]

    def _ffmpeg_cmd(self, exe):
        sdp = ("v=0\no=- 0 0 IN IP4 0.0.0.0\ns=cvtracker\n"
               "c=IN IP4 0.0.0.0\nt=0 0\n"
               f"m=video {self.port} RTP/AVP 96\n"
               "a=rtpmap:96 H264/90000\n")
        path = os.path.join(tempfile.gettempdir(), "cvtracker_stream.sdp")
        with open(path, "w") as f:
            f.write(sdp)
        return [exe, "-loglevel", "error",
                "-protocol_whitelist", "file,udp,rtp",
                "-analyzeduration", "3000000", "-probesize", "3000000",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-i", path,
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-"]

    def _kill(self):
        with self._lock:
            if self.proc is not None:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
                except (OSError, Exception):
                    pass
                finally:
                    self.proc = None

    def shutdown(self):
        self.running = False
        self.restart.set()
        self._kill()

    def _loop(self):
        import subprocess
        import numpy as np
        while self.running:
            self.restart.clear()
            gst = self._find_gst()
            ff  = self._find_ffmpeg()
            if gst:
                cmd, name = self._gst_cmd(gst), "GStreamer"
            elif ff:
                cmd, name = self._ffmpeg_cmd(ff), "ffmpeg"
            else:
                self.on_status("install GStreamer or ffmpeg (not found in PATH)")
                time.sleep(3.0)
                continue

            self.on_status(f"{name}: listening on UDP :{self.port} ...")
            flags = 0x08000000 if os.name == "nt" else 0
            try:
                with self._lock:
                    if not self.running:
                        break
                    self.proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, creationflags=flags)
                self._stderr_lines = []
                threading.Thread(target=self._drain_stderr,
                                  args=(self.proc,), daemon=True).start()
            except OSError as e:
                self.on_status(f"{name} failed to start: {e}")
                time.sleep(3.0)
                continue

            nbytes = self.width * self.height * 3
            got_any = False
            start_t = time.monotonic()
            while self.running and not self.restart.is_set():
                buf = b""
                while len(buf) < nbytes:
                    chunk = self.proc.stdout.read(nbytes - len(buf))
                    if not chunk:
                        buf = None
                        break
                    buf += chunk
                if buf is None:
                    break
                if not got_any:
                    got_any = True
                    self.on_status(f"{name}: stream connected")
                self.on_frame(buf, self.width, self.height)

            self._kill()
            if self.running and not self.restart.is_set():
                if not got_any and (time.monotonic() - start_t) < 2.0:
                    err_tail = "\n".join(self._stderr_lines[-5:]).strip()
                    if err_tail:
                        self.on_log(f"!! {name} exited immediately:\n{err_tail}")
                    else:
                        self.on_log(f"!! {name} exited immediately with no "
                                    f"output - check that UDP :{self.port} "
                                    f"isn't already in use")
                self.on_status(f"{name}: stream ended - reconnecting")
                time.sleep(1.0)

        self.on_status("Video receiver stopped - port released")

    def _drain_stderr(self, proc):
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 50:
                        del self._stderr_lines[:-50]
        except (OSError, ValueError):
            pass


# ==========================================================================
# Main application
# ==========================================================================
class App:
    WIN_W = 1400
    WIN_H = 860

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CvTracker Ground Station")
        self.root.configure(bg="#f0f0f0")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.minsize(900, 600)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            try:
                style.theme_use("default")
            except Exception:
                pass

        self.frame_q    = queue.Queue(maxsize=2)
        self.telem_q    = queue.Queue(maxsize=4)
        self.photo      = None
        self.frame_size = (1280, 720)

        self._drag_start   = None
        self._drag_current = None
        self._dragging     = False

        self._telem_vars = {}
        self._telem_last = {}
        self._telem_rows = {}
        self._fps_times: list = []
        self._closed = False

        self._build_ui()

        self.ctrl  = Control(self.log, telem_cb=self._telem_cb, listen_port=5001)
        self.video = VideoReceiver(self._frame_cb, self._status_cb, on_log=self.log)

        self._apply_connection()

        self.root.after(15,  self._poll_frames)
        self.root.after(50,  self._poll_telem)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        signal.signal(signal.SIGINT, self._handle_signal)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, AttributeError):
            pass
        self.root.after(150, self._signal_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = self.root

        # Top connection bar
        top = ttk.Frame(root, padding=(6, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        def lbl(t):
            ttk.Label(top, text=t).pack(side=tk.LEFT, padx=(4, 1))

        lbl("Jetson IP")
        self.ip_var = tk.StringVar(value="192.168.0.20")
        ttk.Combobox(top, textvariable=self.ip_var,
                     values=["192.168.0.2", "192.168.0.20"],
                     width=14, state="normal").pack(side=tk.LEFT, padx=2)

        lbl("ctrl/telem")
        self.cport_var = tk.StringVar(value="5001")
        ttk.Entry(top, textvariable=self.cport_var, width=6).pack(side=tk.LEFT, padx=2)

        lbl("video")
        self.vport_var = tk.StringVar(value="5000")
        ttk.Entry(top, textvariable=self.vport_var, width=6).pack(side=tk.LEFT, padx=2)

        lbl("W×H")
        self.w_var = tk.StringVar(value="1280")
        ttk.Entry(top, textvariable=self.w_var, width=5).pack(side=tk.LEFT, padx=1)
        self.h_var = tk.StringVar(value="720")
        ttk.Entry(top, textvariable=self.h_var, width=5).pack(side=tk.LEFT, padx=1)

        ttk.Button(top, text="Apply",
                   command=self._apply_connection).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="-")
        ttk.Label(top, textvariable=self.status_var,
                  foreground="#555555").pack(side=tk.LEFT, padx=8)

        ttk.Separator(root, orient="horizontal").pack(side=tk.TOP, fill=tk.X)

        # Body: two columns
        body = tk.Frame(root, bg="#f0f0f0")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        body.columnconfigure(0, weight=0, minsize=380)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # LEFT PANEL
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.columnconfigure(0, weight=1)

        self._build_telem_panel(left)
        self._build_log_panel(left)

        # RIGHT PANEL
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        self._build_video_panel(right)
        self._build_controls_panel(right)

        # Status bar
        self.bye_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.bye_var,
                  font=("Consolas", 10, "bold"),
                  foreground="#c00000").pack(side=tk.BOTTOM, anchor="w", padx=6, pady=2)

    # ----------------------------------------------------------------
    def _build_telem_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Live Telemetry", padding=6)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        hdr = ttk.Frame(frame)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        hdr.columnconfigure(1, weight=1)

        self._mode_var = tk.StringVar(value="MODE: -")
        ttk.Label(hdr, textvariable=self._mode_var,
                  font=("Consolas", 11, "bold")).grid(row=0, column=0, sticky="w")

        self._fps_var = tk.StringVar(value="- fps")
        ttk.Label(hdr, textvariable=self._fps_var,
                  font=("Consolas", 10)).grid(row=0, column=1, sticky="w", padx=12)

        ttk.Button(hdr, text="Reset History",
                   command=self._reset_telem).grid(row=0, column=2, sticky="e")

        grid = ttk.Frame(frame)
        grid.grid(row=1, column=0, sticky="nsew")

        sections = [
            ("Status",        ["frame_id", "mode", "det_prob", "proc_time_us"]),
            ("Tracking Rect", ["rect_x", "rect_y", "rect_width", "rect_height"]),
            ("Object Rect",   ["object_x", "object_y",
                               "object_width", "object_height"]),
            ("Angle Offsets", ["angle_x_deg", "angle_y_deg",
                               "pixel_offset_x", "pixel_offset_y"]),
            ("Search Window", ["search_win_width", "search_win_height"]),
            ("Parameters",    ["lost_mode_option", "frame_buffer_size",
                               "max_frames_lost", "rect_auto_size",
                               "rect_auto_position", "multiple_threads",
                               "num_channels", "tracker_type",
                               "custom_1", "custom_2", "custom_3"]),
            ("Resolution",    ["frame_width", "frame_height"]),
        ]

        disp_map = {k: (lbl, fmt) for lbl, k, fmt in TELEM_DISPLAY}

        per_row = 2
        for col in range(per_row):
            grid.columnconfigure(col, weight=1)

        for idx, (section_name, keys) in enumerate(sections):
            r, c = divmod(idx, per_row)
            block = ttk.Frame(grid, padding=(0, 0, 12, 8))
            block.grid(row=r, column=c, sticky="nsew")

            ttk.Label(block, text=section_name.upper(),
                      font=("Consolas", 8, "bold"),
                      foreground="#666666").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 3))

            rr = 1
            for key in keys:
                if key not in disp_map:
                    continue
                lbl_text, _fmt = disp_map[key]
                ttk.Label(block, text=lbl_text, width=17,
                          anchor="w").grid(row=rr, column=0, sticky="w")
                sv = tk.StringVar(value="-")
                val_lbl = ttk.Label(block, textvariable=sv,
                                    font=("Consolas", 10, "bold"),
                                    width=10, anchor="e")
                val_lbl.grid(row=rr, column=1, sticky="e")
                self._telem_vars[key] = sv
                self._telem_rows[key] = val_lbl
                rr += 1

    # ----------------------------------------------------------------
    def _build_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="ACK / Command Log", padding=6)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.logbox = tk.Text(
            frame, width=58, height=18,
            state=tk.DISABLED,
            bg="#ffffff", fg="#000000",
            font=("Consolas", 9),
            relief="solid", bd=1,
            wrap=tk.NONE,
        )
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.logbox.yview)
        self.logbox.configure(yscrollcommand=sb.set)
        self.logbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ----------------------------------------------------------------
    def _build_video_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Video", padding=2)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            frame,
            bg="#1a1a1a",
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.root.bind("<B1-Motion>",          self._on_drag)
        self.root.bind("<ButtonRelease-1>",    self._on_release)
        self.canvas.bind("<Button-3>",
                         lambda e: self.ctrl.send(CMD_RESET))

    # ----------------------------------------------------------------
    def _build_controls_panel(self, parent):
        """Controls - fixed height row below the video."""
        frame = ttk.LabelFrame(parent, text="Controls", padding=6)
        frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        # Row 0: RESET button
        ttk.Button(frame, text="RESET  (or right-click video)",
                   command=lambda: self.ctrl.send(CMD_RESET)).grid(
            row=0, column=0, columnspan=6, sticky="we", pady=(0, 4))

        # Row 1: PERSIST button
        ttk.Button(
            frame,
            text="  PERSIST  ",
            command=self._persist_params
        ).grid(row=1, column=0, columnspan=6, sticky="we", pady=(0, 6))

        # Row 2: Rect size + auto checkboxes
        ttk.Button(frame, text="Rect −8",
                   command=lambda: self.ctrl.send(CMD_CHANGE_SIZE, -8, -8)
                   ).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(frame, text="Rect +8",
                   command=lambda: self.ctrl.send(CMD_CHANGE_SIZE, 8, 8)
                   ).grid(row=2, column=1, padx=2, pady=2)

        self.autosize_var = tk.IntVar(value=0)
        ttk.Checkbutton(frame, text="Auto size",
                        variable=self.autosize_var,
                        command=lambda: self.ctrl.send(
                            CMD_SET_PARAM, PARAMS["RECT_AUTO_SIZE"],
                            float(self.autosize_var.get()))
                        ).grid(row=2, column=2, padx=8, sticky="w")

        self.autopos_var = tk.IntVar(value=0)
        ttk.Checkbutton(frame, text="Auto position",
                        variable=self.autopos_var,
                        command=lambda: self.ctrl.send(
                            CMD_SET_PARAM, PARAMS["RECT_AUTO_POSITION"],
                            float(self.autopos_var.get()))
                        ).grid(row=2, column=3, padx=8, sticky="w")

        # Row 3: Param selector
        ttk.Label(frame, text="Param").grid(row=3, column=0, sticky="w",
                                             pady=(8, 2), padx=2)
        self.param_var = tk.StringVar(value="CUSTOM_1 (loss thr)")
        ttk.Combobox(frame, textvariable=self.param_var,
                     values=list(PARAMS.keys()), state="readonly",
                     width=26).grid(row=3, column=1, columnspan=3,
                                    sticky="we", pady=(8, 2))

        # Row 4: Value entry + Set button
        ttk.Label(frame, text="Value").grid(row=4, column=0, sticky="w",
                                             pady=2, padx=2)
        self.pval_var = tk.StringVar(value="0.1")
        ttk.Entry(frame, textvariable=self.pval_var, width=12).grid(
            row=4, column=1, sticky="w", pady=2)
        ttk.Button(frame, text="Set",
                   command=self._send_param).grid(row=4, column=2,
                                                   sticky="w", pady=2)

    # ================================================================ helpers
    def _canvas_to_frame(self, cx, cy):
        fw, fh = self.frame_size
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / fw, ch / fh)
        ox = (cw - fw * scale) / 2
        oy = (ch - fh * scale) / 2
        x = (cx - ox) / scale
        y = (cy - oy) / scale
        if 0 <= x < fw and 0 <= y < fh:
            return int(x), int(y)
        return None

    def _frame_to_canvas(self, fx, fy):
        fw, fh = self.frame_size
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / fw, ch / fh)
        ox = (cw - fw * scale) / 2
        oy = (ch - fh * scale) / 2
        return fx * scale + ox, fy * scale + oy

    # ============================================== mouse event handlers
    def _on_press(self, event):
        pt = self._canvas_to_frame(event.x, event.y)
        if pt:
            self._drag_start   = pt
            self._drag_current = pt
            self._dragging     = False

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        cvx = event.x_root - self.canvas.winfo_rootx()
        cvy = event.y_root - self.canvas.winfo_rooty()
        cx0, cy0 = self._frame_to_canvas(*self._drag_start)
        if abs(cvx - cx0) < 5 and abs(cvy - cy0) < 5:
            return
        self._dragging = True
        pt = self._canvas_to_frame(cvx, cvy)
        if pt is None:
            return
        self._drag_current = pt
        fw, fh = self.frame_size
        x1 = max(0, min(fw - 1, pt[0]))
        y1 = max(0, min(fh - 1, pt[1]))
        cx1, cy1 = self._frame_to_canvas(x1, y1)
        self.canvas.delete("rubberband")
        self.canvas.create_rectangle(cx0-1, cy0-1, cx1+1, cy1+1,
                                      outline="black", width=1,
                                      tag="rubberband")
        self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                      outline="yellow", width=2,
                                      tag="rubberband")
        for hx, hy in ((cx0, cy0), (cx1, cy0), (cx0, cy1), (cx1, cy1)):
            self.canvas.create_rectangle(hx-3, hy-3, hx+3, hy+3,
                                          outline="yellow", fill="black",
                                          tag="rubberband")

    def _on_release(self, event):
        self.canvas.delete("rubberband")
        if self._drag_start is None:
            return
        cvx = event.x_root - self.canvas.winfo_rootx()
        cvy = event.y_root - self.canvas.winfo_rooty()
        pt  = self._canvas_to_frame(cvx, cvy) or self._drag_current

        if self._dragging and pt:
            x0, y0 = self._drag_start
            fw, fh  = self.frame_size
            x1 = max(0, min(fw - 1, pt[0]))
            y1 = max(0, min(fh - 1, pt[1]))
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w > 5 and h > 5:
                cx = (min(x0, x1) + max(x0, x1)) / 2.0
                cy = (min(y0, y1) + max(y0, y1)) / 2.0
                self.ctrl.send(CMD_SET_PARAM,
                               float(PARAMS["RECT_WIDTH"]),  float(w))
                self.ctrl.send(CMD_SET_PARAM,
                               float(PARAMS["RECT_HEIGHT"]), float(h))
                self.ctrl.send(CMD_CAPTURE,
                               float(int(cx)), float(int(cy)), -1.0)
                self.log(f"- drag->capture ({w:.0f}x{h:.0f}) @ "
                         f"({int(cx)},{int(cy)})")
            else:
                self.log("- drag too small, ignored")
        else:
            if pt:
                self.ctrl.send(CMD_CAPTURE,
                               float(pt[0]), float(pt[1]), -1.0)

        self._drag_start   = None
        self._drag_current = None
        self._dragging     = False

    # ================================================================= apply
    def _apply_connection(self):
        try:
            ip   = self.ip_var.get().strip()
            cp   = self.cport_var.get()
            vp   = self.vport_var.get()
            w, h = self.w_var.get(), self.h_var.get()
            self.ctrl.set_target(ip, cp)
            self.video.configure(vp, w, h)
            self._controls_synced = False  
            self.log(f"target {ip}:{cp}, video :{vp} {w}x{h}")
        except ValueError:
            self.log("!! invalid port or dimensions")

    def _send_param(self):
        try:
            value = float(self.pval_var.get())
        except ValueError:
            self.log("!! invalid value")
            return
        pid = PARAMS[self.param_var.get()]
        self.ctrl.send(CMD_SET_PARAM, float(pid), value)

    def _persist_params(self):
        self.ctrl.send(CMD_SAVE_PARAMS, 0.0, 0.0, 0.0)
        self.log("- PERSIST sent: params saved to disk, will survive next restart")

    # -------------------------------------------- video plumbing
    def _frame_cb(self, raw_bytes, w, h):
        try:
            self.frame_q.put_nowait((raw_bytes, w, h))
        except queue.Full:
            pass

    def _status_cb(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _poll_frames(self):
        import cv2
        import numpy as np
        item = None
        while True:
            try:
                item = self.frame_q.get_nowait()
            except queue.Empty:
                break
        if item is not None:
            raw, fw, fh = item
            self.frame_size = (fw, fh)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw, 3)
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            scale = min(cw / fw, ch / fh)
            dw = max(1, int(fw * scale))
            dh = max(1, int(fh * scale))
            disp = cv2.resize(frame, (dw, dh))
            ok, ppm = cv2.imencode(".ppm", disp)
            if ok:
                self.photo = tk.PhotoImage(data=ppm.tobytes())
                self.canvas.delete("all")
                self.canvas.create_image(cw // 2, ch // 2, image=self.photo)
        self.root.after(15, self._poll_frames)

    # -------------------------------------------- telemetry plumbing
    def _telem_cb(self, telem: dict):
        try:
            self.telem_q.put_nowait(telem)
        except queue.Full:
            pass

    def _poll_telem(self):
        telem = None
        while True:
            try:
                telem = self.telem_q.get_nowait()
            except queue.Empty:
                break
        if telem is not None:
            self._update_telem_panel(telem)
        self.root.after(50, self._poll_telem)

    def _update_telem_panel(self, t: dict):
        now = time.monotonic()
        self._fps_times.append(now)
        self._fps_times = [x for x in self._fps_times if now - x < 2.0]
        fps = len(self._fps_times) / 2.0
        self._fps_var.set(f"{fps:.1f} fps")

        mode  = int(t.get("mode", 0))
        mname = MODE_NAMES.get(mode, "?")
        self._mode_var.set(f"MODE: {mname}")

        disp_map = {k: (l, f) for l, k, f in TELEM_DISPLAY}
        for key, sv in self._telem_vars.items():
            if key not in t:
                continue
            raw = t[key]
            _, fmt = disp_map.get(key, ("", "{}"))
            if key == "mode":
                display_val = MODE_NAMES.get(int(raw), str(raw))
            else:
                try:
                    display_val = fmt.format(raw)
                except (ValueError, TypeError):
                    display_val = str(raw)
            sv.set(display_val)
            self._telem_last[key] = raw
            if not self._controls_synced:
                self._sync_controls_from_telem(t)
                self._controls_synced = True

    def _sync_controls_from_telem(self, t: dict):
   
        mapping = {
            "lost_mode_option":  "LOST_MODE_OPTION",
            "frame_buffer_size": "FRAME_BUFFER_SIZE",
            "max_frames_lost":   "MAX_FRAMES_IN_LOST_MODE",
            "rect_auto_size":    "RECT_AUTO_SIZE",
            "rect_auto_position":"RECT_AUTO_POSITION",
            "multiple_threads":  "MULTIPLE_THREADS",
            "num_channels":      "NUM_CHANNELS",
            "tracker_type":      "TYPE",
            "custom_1":          "CUSTOM_1 (loss thr)",
            "custom_2":          "CUSTOM_2 (recapture thr)",
            "custom_3":          "CUSTOM_3 (learning rate)",
        }

    # Sync the param entry box to show custom_1 value by default
        if "custom_1" in t:
            self.pval_var.set(f"{t['custom_1']:.4f}")

        # Sync checkboxes
        if "rect_auto_size" in t:
            self.autosize_var.set(int(t["rect_auto_size"]))
        if "rect_auto_position" in t:
            self.autopos_var.set(int(t["rect_auto_position"]))

        self.log(f"- synced controls from Jetson: "
                f"custom1={t.get('custom_1', '?'):.4f}  "
                f"custom2={t.get('custom_2', '?'):.4f}  "
                f"autosize={int(t.get('rect_auto_size', 0))}  "
                f"autopos={int(t.get('rect_auto_position', 0))}")     
    def _reset_telem(self):
        for sv in self._telem_vars.values():
            sv.set("-")
        self._telem_last.clear()
        self._fps_times.clear()
        self._fps_var.set("- fps")
        self._mode_var.set("MODE: -")
        self.log("- telemetry history reset")

    # ----------------------------------------------------------------- log
    def log(self, text):
        def _append():
            self.logbox.configure(state=tk.NORMAL)
            self.logbox.insert(tk.END, time.strftime("%H:%M:%S ") + text + "\n")
            self.logbox.see(tk.END)
            self.logbox.configure(state=tk.DISABLED)
        self.root.after(0, _append)

    # --------------------------------------------------------------- close
    def _signal_tick(self):
        if not self._closed:
            self.root.after(150, self._signal_tick)

    def _handle_signal(self, signum, frame):
        self.log(f"!! signal {signum} received - releasing ports and exiting")
        self._on_close()
        sys.exit(0)

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        self.video.shutdown()
        self.ctrl.shutdown()
        time.sleep(0.3)
        try:
            self.status_var.set("Ports released")
            self.bye_var.set("Ports 5000/5001 released - closing")
        except tk.TclError:
            pass
        print("Port 5000 (video)      - RELEASED")
        print("Port 5001 (ctrl+telem) - RELEASED")
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_close()


if __name__ == "__main__":
    App().run()