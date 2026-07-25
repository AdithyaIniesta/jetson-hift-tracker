# Integrating CvTracker on Jetson Xavier NX

## 1. Prerequisites (on the NX)

JetPack 5.x (L4T 35.x) ships everything needed: GCC 9+, CUDA 11.4, cuFFT.

```bash
sudo apt update && sudo apt install -y cmake build-essential
nvcc --version    # confirm CUDA if you want the cuFFT path
```

## 2. Build and install the library

Copy the `cvtracker/` folder to the NX (or git clone if you put it in a repo).

```bash
cd cvtracker

# CPU-only (recommended first run - zero dependencies):
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Or with cuFFT acceleration (Xavier NX = SM 7.2, already in the preset):
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCVTRACKER_WITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=72

cmake --build build -j6          # NX has 6 cores
sudo cmake --install build       # -> /usr/local/lib/libcvtracker.a + headers
```

Sanity check on the device:

```bash
cmake -B build -DCVTRACKER_BUILD_TESTS=ON && cmake --build build -j6
./build/test_tracker             # expect: ALL TESTS PASSED
```

Add `-DBUILD_SHARED_LIBS=ON` if you prefer `libcvtracker.so`.

## 3. Link it into your project

CMake:

```cmake
find_package(cvtracker REQUIRED)            # after cmake --install
target_link_libraries(my_app PRIVATE cvtracker::cvtracker)
```

or without installing: `add_subdirectory(cvtracker)` and link `cvtracker`.
Plain Makefile: `-I<path>/include ... -lcvtracker -lpthread` (add
`-lcufft -lcudart` for the CUDA build).

## 4. Feed it camera frames

The tracker is camera agnostic - it only consumes `cr::video::Frame`. On the
NX the two usual sources:

**CSI camera (Argus) via GStreamer, NV12 - zero conversion needed:**

```
nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1
  ! nvvidconv ! video/x-raw,format=NV12 ! appsink
```

**USB camera via V4L2, YUYV - also consumed directly:**

```
v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=1280,height=720 ! appsink
```

In the appsink callback, copy the mapped buffer into a reusable `Frame`:

```cpp
#include <cvtracker/CvTracker.h>
using namespace cr::vtracker;
using namespace cr::video;

CvTracker tracker;
Frame frame(1920, 1080, Fourcc::NV12);   // allocate once
int32_t nextId = 0;

// per buffer (GstMapInfo map):
std::memcpy(frame.data, map.data, frame.size);
frame.frameId = nextId++;                 // unique, every frame, no drops
tracker.processFrame(frame);              // call in ALL modes, incl. FREE
```

Rules that matter: every frame goes to `processFrame()` without dropping,
each with a unique incrementing `frameId` (this drives the stop-frame buffer
and delay compensation). Keep one producer thread calling `processFrame()`;
commands/params can come from any other thread.

## 5. Initialize and run the tracking loop

```cpp
VTrackerParams p;
p.rectWidth = 72;  p.rectHeight = 72;     // expected object size
p.searchWindowWidth = 256;  p.searchWindowHeight = 256;
p.frameBufferSize = 32;                   // 32 NV12 1080p lumas ~ 64 MB
p.maxFramesInLostMode = 90;               // ~3 s at 30 FPS
p.lostModeOption = 1;                     // predict trajectory when lost
p.multipleThreads = true;
tracker.initVTracker(p);
```

Operator clicks at pixel (x, y):

```cpp
tracker.executeCommand(VTrackerCommand::CAPTURE, x, y, -1);          // live
tracker.executeCommand(VTrackerCommand::CAPTURE, x, y, shownFrameId); // stop-frame
```

After each processed frame, read results for your OSD / gimbal loop:

```cpp
VTrackerParams r;
tracker.getParams(r);
// r.mode (1 = TRACKING, 2 = LOST), r.rectX/rectY, r.velX/velY,
// r.objectWidth/objectHeight, r.detectionProbability
```

`RESET` releases the object. For remote control over a link, use
`VTracker::encodeCommand()` on the ground side and
`tracker.decodeAndExecuteCommand()` on the NX, and ship
`r.encode()` telemetry back.

## 6. Xavier NX performance checklist

- `sudo nvpmodel -m 8` (20W 6-core) or `-m 0`, then `sudo jetson_clocks`.
- Expected: ~10-25 ms/frame at a 256x256 search window on CPU - real-time at
  30 FPS. If you need margin: search window 128x128 is ~4x cheaper;
  `MULTIPLE_THREADS = 1`; or the CUDA build.
- Pin the capture/tracking thread if you run DNNs alongside:
  `taskset -c 4,5 ./my_app`.
- `r.processingTimeMks` tells you the actual per-frame cost at runtime.

## 7. Troubleshooting

- `processFrame()` returns false -> wrong fourcc/size, or frame dimensions
  changed mid-stream (tracker auto-resets; re-init with the new size).
- Tracking resets immediately -> capture rectangle mostly background; match
  `rectWidth/Height` to the object and capture centered.
- Loses small low-contrast objects -> lower loss threshold
  (`setParam(VTrackerParam::CUSTOM_1, 0.05f)`), raise re-capture
  threshold logic via `CUSTOM_2` as needed.
- Drifts onto background during slow appearance change -> reduce learning
  rate: `setParam(VTrackerParam::CUSTOM_3, 0.03f)`.
