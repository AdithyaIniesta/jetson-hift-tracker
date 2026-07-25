# CvTracker - correlation video tracker (C++)

Self-contained C++17 object tracker with the same interface, operating modes
and workflow as the RapidPixel CvTracker C++ library
(https://rapidpixel.constantrobotics.com/docs/VideoTracking/CvTracker.html).

## Robustness

- **Lighting changes** - features are log-transformed and zero-mean /
  unit-norm normalized, so global gain and offset changes do not affect the
  correlation surface.
- **Orientation changes** - at capture the filter is trained on rotated
  (+-4, +-8 deg) and scaled (+-5%) perturbations of the reference; during
  tracking the pattern adapts continuously (confidence-weighted update), and
  a periodic scale probe follows object size changes.
- **Occlusions** - detection probability combines the response peak value
  with the peak-to-sidelobe ratio. When it falls below the loss threshold the
  pattern update stops (no corruption), the tracker switches to LOST mode and
  predicts the trajectory; the object is re-captured automatically when the
  probability exceeds the re-capture threshold.

## Hardware and camera agnosticism

- Plain C++17, **zero external dependencies** (FFT included). Builds and runs
  identically on every Jetson (Nano/TX1/TX2/Xavier/Orin), any ARM or x86
  Linux/Windows machine.
- Optional CUDA (cuFFT) acceleration: `-DCVTRACKER_WITH_CUDA=ON`. Runtime
  fallback to the CPU path if no CUDA device is present, so one binary works
  everywhere.
- No camera code: you push frames in via `processFrame()`. Supported RAW
  pixel formats: GRAY, YUV24, YUYV, UYVY, NV12, NV21, YU12, YV12 (8 bit) -
  everything V4L2 / Argus / GStreamer pipelines typically deliver. Processing
  uses the luma plane, so daylight and thermal cameras both work.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release        # CPU only
cmake -B build -DCVTRACKER_WITH_CUDA=ON          # with cuFFT acceleration
cmake --build build -j
sudo cmake --install build                       # installs lib + headers
```

Options: `BUILD_SHARED_LIBS=ON` for a `.so`, `CVTRACKER_BUILD_TESTS=ON` for
the synthetic-sequence test (`build/test_tracker`). Cross-compiling for
Jetson works with a standard aarch64 toolchain file; CUDA architectures for
the whole Jetson series (5.3/6.2/7.2/8.7) are preset.

## Usage

```cpp
#include <cvtracker/CvTracker.h>
using namespace cr::vtracker;

CvTracker tracker;

VTrackerParams params;            // 1. set parameters
params.rectWidth = 72;
params.rectHeight = 72;
params.searchWindowWidth = 256;
params.searchWindowHeight = 256;
tracker.initVTracker(params);     // 2. init

cr::video::Frame frame(width, height, cr::video::Fourcc::NV12);
// fill frame.data, set unique frame.frameId for each frame ...

tracker.processFrame(frame);      // 3. every frame, without dropping

// 4. capture object at pixel (x, y) on the last frame:
tracker.executeCommand(VTrackerCommand::CAPTURE, x, y, -1);
// or on a buffered past frame (stop-frame / channel delay compensation):
tracker.executeCommand(VTrackerCommand::CAPTURE, x, y, (float)pastFrameId);

VTrackerParams results;           // 5. read tracking results
tracker.getParams(results);
// results.mode: 0 FREE, 1 TRACKING, 2 LOST, 3 INERTIAL, 4 STATIC
// results.rectX/rectY, objectX/Y/Width/Height, velX/velY,
// results.detectionProbability

tracker.executeCommand(VTrackerCommand::RESET);
```

All of `setParam / getParam / getParams / executeCommand /
decodeAndExecuteCommand` are thread-safe. Remote control:
`VTracker::encodeCommand()` / `encodeSetParamCommand()` on the control side,
`decodeAndExecuteCommand()` on the tracker side;
`VTrackerParams::encode()/decode()` for telemetry.

Custom parameters: `custom1` - loss detection probability threshold
(default 0.1), `custom2` - re-capture probability threshold (default 0.4),
`custom3` - pattern learning rate (default 0.075).

`getImage()` returns internal matrices: 0 - object reference image,
1 - object mask, 2 - correlation surface.

## Performance

1080p input, 256x256 search window, x86 CPU (single core): ~10 ms/frame.
Reduce the search window to 128x128 for ~4x faster processing on small
Jetson modules, or enable `MULTIPLE_THREADS` / build with CUDA.

## Layout

```
include/cvtracker/   Frame.h, VTracker.h (interface + params/commands),
                     CvTracker.h, CvTrackerVersion.h
src/                 implementation (FFT, correlation core, state machine)
src/cuda/            optional cuFFT backend
test/                synthetic-sequence verification test
```
