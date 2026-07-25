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
- **Predictive filtering (optional EKF)** - an Extended Kalman Filter with a
  constant-turn-rate-and-velocity (CTRV) motion model can be enabled to
  smooth position/velocity output and coast through occlusions and sudden
  camera motion along a physically plausible trajectory. A Mahalanobis
  outlier gate rejects measurement jumps that violate the motion model (e.g.
  the correlation peak sliding along a straight edge). Opt-in via
  `ENABLE_EKF`; default off preserves the raw correlation behavior. See
  [CVTRACKER_API.md](CVTRACKER_API.md#13a-ekf-predictive-filtering-optional).
- **DNN appearance verifier (optional)** - a pluggable embedding extractor
  (TensorRT on Jetson, or any user implementation of `FeatureExtractor`)
  maintains a template bank of the object's appearance across time. It
  verifies at a low cadence that the tracker still follows the captured
  object (blocks pattern learning and forces LOST on sustained mismatch)
  and rejects LOST-mode re-captures onto similar-looking distractors.
  Opt-in via `ENABLE_DNN_VERIFIER` + `setFeatureExtractor()`; default off.
  See [CVTRACKER_API.md](CVTRACKER_API.md#13b-dnn-appearance-verifier-optional).
- **Unstabilized-camera aids (optional)** - global motion compensation
  (whole-frame phase correlation: camera jolts never appear as object
  motion; no IMU, no lens calibration), DNN multi-peak arbitration, and
  DNN embedding re-acquisition across the full search window. Opt-in via
  `ENABLE_GMC` / `ENABLE_DNN_ARBITRATION` / `ENABLE_DNN_REACQUISITION`;
  all default off. See
  [CVTRACKER_API.md](CVTRACKER_API.md#13c-vision-only-robustness-aids).

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

Optional predictive filtering: set `params.ekfEnabled = true` (or
`setParam(VTrackerParam::ENABLE_EKF, 1.0f)`) to route position/velocity
through a CTRV Extended Kalman Filter with a Mahalanobis outlier gate.
Default off. See [CVTRACKER_API.md](CVTRACKER_API.md#13a-ekf-predictive-filtering-optional)
for the motion model, tunable noise constants, and integration details.

Optional DNN appearance verifier: provide an embedding extractor and enable
the flag - the tracker then verifies its lock against a template bank of
the object's appearance (drift / distractor rejection):

```cpp
tracker.setFeatureExtractor(fe);   // TensorRT (Jetson) or custom
tracker.setParam(VTrackerParam::ENABLE_DNN_VERIFIER, 1.0f);
// telemetry: results.dnnSimilarity in [0..1]
```

Build the TensorRT backend with `-DCVTRACKER_WITH_TENSORRT=ON` (see
`TrtFeatureExtractor.h`); without it the core library stays zero-dependency
and a built-in `TinyPatchExtractor` fallback exercises the same pipeline.
See [CVTRACKER_API.md](CVTRACKER_API.md#13b-dnn-appearance-verifier-optional).

`getImage()` returns internal matrices: 0 - object reference image,
1 - object mask, 2 - correlation surface.

## Performance

1080p input, 256x256 search window, x86 CPU (single core): ~10 ms/frame.
Reduce the search window to 128x128 for ~4x faster processing on small
Jetson modules, or enable `MULTIPLE_THREADS` / build with CUDA.

## Layout

```
include/cvtracker/   Frame.h, VTracker.h (interface + params/commands),
                     CvTracker.h, CvTrackerVersion.h, FeatureExtractor.h,
                     TrtFeatureExtractor.h
src/                 implementation (FFT, correlation core, state machine,
                     template bank)
src/Ekf.{h,cpp}      optional CTRV Extended Kalman Filter + Mahalanobis gate
src/cuda/            optional cuFFT backend
src/trt/             optional TensorRT embedding extractor
test/                synthetic-sequence verification test
```
