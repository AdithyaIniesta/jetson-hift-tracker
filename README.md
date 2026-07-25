# 🎯 Jetson Tracking Perception

> Real-time object tracking pipeline for NVIDIA Jetson devices.
> USB camera → CvTracker → H.264 hardware encoding → UDP stream to ground station.

---

## 📚 Table of Contents

- [Supported Devices](#-supported-devices)
- [Tracker Algorithms](#-tracker-algorithms)
- [Quick Start](#-quick-start)
  - [Build](#step-1--build)
  - [Run with run.sh](#step-2--run-with-runsh)
  - [Run Manually](#step-3--run-manually)
- [Ground Station](#️-ground-station-windows)
- [Troubleshooting](#-troubleshooting)
  - [Camera not detected](#-camera-not-detected)
  - [Port already in use](#-port-already-in-use)
  - [UART permission denied](#-uart-permission-denied)
  - [No video on ground station](#-no-video-on-ground-station)
- [Project Structure](#-project-structure)
- [Docker](#-docker)
  - [Image Architecture](#image-architecture)
  - [Build Docker Image](#build-docker-image)
  - [Transfer and Deploy](#transfer-and-deploy)
  - [Run in Docker](#run-in-docker)

---

## 📋 Supported Devices

| Device | JetPack | CUDA |
|--------|---------|------|
| Orin NX | 5.x (L4T R35.x) | 11.4 |
| Xavier NX | 5.x (L4T R35.x) | 11.4 |

---

## 🧠 Tracker Algorithms

| Binary | Library | Version | Backend |
|--------|---------|---------|---------|
| `JetsonTracker_constant_robotics_lib` | libCvTracker (Constant Robotics) | 10.0.2 | CPU |
| `JetsonTracker_cuda_library` | cvtracker (open source) | 1.0.0 | CUDA cuFFT (11.4) |

---

## ⚡ Quick Start

### Step 1 — Build

```bash
./build.sh
```

You will be prompted to select:
```
[1] constant_robotics_lib
[2] cuda_library
[3] both
Select [1/2/3]:
```

Both binaries are placed in `build/bin/`.

---

### Step 2 — Run with run.sh

```bash
./run.sh
```

The script will:
- Show available binaries in green, missing ones in red
- Prompt for algorithm selection
- Prompt for IP and resolution (press Enter for defaults)
- Fix UART permissions automatically
- Clean ports before starting

---

### Step 3 — Run Manually

```bash
# Fix UART permission
sudo chmod 666 /dev/ttyTHS0

# Clean ports
sudo fuser -k 5000/udp
sudo fuser -k 5001/udp

# Run constant_robotics_lib
./build/bin/JetsonTracker_constant_robotics_lib 192.168.0.100 5000 5001 1280 720 /dev/ttyTHS0 /dev/video0

# Run cuda_library
./build/bin/JetsonTracker_cuda_library 192.168.0.100 5000 5001 1280 720 /dev/ttyTHS0 /dev/video0
```

**Arguments:**
```
<client_ip> <video_port> <ctrl_port> <width> <height> <uart_dev> <video_dev>
```

---

## 🖥️ Ground Station (Windows)

Receive video stream and send tracking commands:

```powershell
pip install opencv-python numpy
python tracker_ui.py
```

- Set Jetson IP and click **Apply**
- **Left click** on video to capture object
- **Right click** to reset tracking

Receive raw stream with GStreamer:

```powershell
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

---

## 🔧 Troubleshooting

### 📷 Camera not detected

Check if camera is recognized:
```bash
ls /dev/video*
```
Expected:
```
/dev/video0
```

Check if another process is holding the camera:
```bash
fuser -v /dev/video0
```
Expected:
```
(no output)
```

Test camera frames directly:
```bash
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=10
```
Expected:
```
<<<<<<<<<<
```

List supported formats:
```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

---

### 🔌 Port already in use

Check which process holds port 5000 or 5001:
```bash
sudo fuser -v 5000/udp
sudo fuser -v 5001/udp
```

Kill the process:
```bash
sudo fuser -k 5000/udp
sudo fuser -k 5001/udp
```

Expected after kill:
```
(no output from fuser -v)
```

---

### 🔗 UART permission denied

Check available UART devices:
```bash
ls /dev/ttyTHS*
```

Fix permission:
```bash
sudo chmod 666 /dev/ttyTHS0
```

Verify permission is set:
```bash
ls -la /dev/ttyTHS0
```
Expected:
```
crw-rw-rw- 1 root dialout ... /dev/ttyTHS0
```

Add user to dialout group permanently (no sudo needed after reboot):
```bash
sudo usermod -aG dialout $USER
sudo reboot
```

---

### 📡 No video on ground station

Verify packets are leaving the Jetson:
```bash
sudo tcpdump -i any udp port 5000 -c 10
```
Expected:
```
10 packets captured
```

- If no packets — tracker pipeline failed. Check tracker terminal for pipeline errors.
- Check Windows firewall allows inbound UDP on port 5000 and 5001.

---

## 📁 Project Structure

```
JetsonTracker2048/
├── src/
│   ├── main.cpp                  # Main pipeline
│   ├── CMakeLists.txt            # Build config with v1/v2 switch
│   └── compat_shim.cpp/c/S       # GCC compatibility shims
├── cvtracker/                    # Open-source tracker library (v1.0.0 + CUDA)
├── Limitations 2048/             # Constant Robotics prebuilt library (v10.0.2)
├── docker/
│   ├── Dockerfile.jp5            # Three-stage Docker build
│   └── build.sh                  # Docker build helper
├── build.sh                      # Local build script
├── run.sh                        # Local run script
└── run_docker.sh                 # Docker run script
```

---

## 🐳 Docker

### Image Architecture

The Docker image is built in three stages:

```
l4t-jetpack:r35.4.1  (NVIDIA base — Ubuntu 20.04)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  base image  jvp/base:4.5.4-jp5        │
  │  CUDA 11.4 · OpenCV 4.5.4              │
  │  GStreamer 1.16.3 · HW encoder          │
  │  CUDA kernels: CC 7.2 + CC 8.7         │
  └─────────────┬───────────────┬───────────┘
                │               │
                ▼               ▼
   ┌────────────────┐  ┌─────────────────────────────┐
   │  dev image     │  │  runtime image               │
   │  + CMake       │  │  jvp/jetsontracker:v2.0-jp5  │
   │  + Ninja       │  │  + JetsonTracker_v1 binary   │
   │  + GCC         │  │  + JetsonTracker_v2 binary   │
   │  + GDB         │  │  + run_docker.sh             │
   │  (build only)  │  │  (deploy this one)           │
   └────────────────┘  └─────────────────────────────┘
```

- **base** — built once (~90 min), cached forever
- **dev** — used only to compile source code, never deployed
- **runtime** — the only image deployed to Jetson devices

---

### Build Docker Image

Run from `docker/` folder:

```bash
cd docker/

# Step 1 — Build base image (once, ~90 min)
./build.sh base

# Step 2 — Build dev image
./build.sh dev

# Step 3 — Compile both tracker binaries inside Docker
./build.sh compile ~/JetsonTracker2048

# Step 4 — Package runtime image
./build.sh runtime ~/JetsonTracker2048

# Step 5 — Save to tar.gz
./build.sh save
```

---

### Transfer and Deploy

Transfer to another Jetson:
```bash
scp jetsontracker-v2-jp5.tar.gz nvidia@<device-ip>:~/
```

Load on target device:
```bash
docker load < jetsontracker-v2-jp5.tar.gz
```

---

### Run in Docker

```bash
docker run --rm -it --runtime nvidia --network host \
  --device /dev/video0 \
  --device /dev/ttyTHS0 \
  jvp/jetsontracker:v2.0-jp5-orin-xavier \
  /opt/jvp/run_docker.sh
```

| Flag | Purpose |
|------|---------|
| `--rm` | Remove container after exit |
| `-it` | Interactive terminal for run_docker.sh prompt |
| `--runtime nvidia` | Enable GPU and HW encoder access |
| `--network host` | Direct network access for UDP streaming |
| `--device /dev/video0` | Pass USB camera into container |
| `--device /dev/ttyTHS0` | Pass UART into container |

---

## 📝 Notes

- UART output transmits angle offsets every frame while tracking
- Hardware encoder `nvv4l2h264enc` required — not available on Orin Nano
- Default resolution: 1280x720 @ 60fps MJPG
- Stream: H.264 RTP/UDP, MTU 1400
