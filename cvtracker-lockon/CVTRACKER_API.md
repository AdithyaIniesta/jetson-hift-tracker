# CvTracker API Reference

**Version:** 1.0.0  
**Language:** C++17  
**Namespaces:** `cr::vtracker` (tracker), `cr::video` (frames)

CvTracker is a self-contained, zero-external-dependency C++17 correlation-based video object tracker. It implements a MOSSE-style adaptive correlation filter with automatic loss detection, re-capture, and optional CUDA/cuFFT acceleration for NVIDIA Jetson platforms.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Building and Installing](#2-building-and-installing)
3. [Integrating into Your Project](#3-integrating-into-your-project)
4. [Core Classes](#4-core-classes)
   - [CvTracker](#41-cvtracker)
   - [VTracker (Abstract Interface)](#42-vtracker-abstract-interface)
   - [VTrackerParams](#43-vtrackerparams)
   - [VTrackerParam Enum](#44-vtrackerparam-enum)
   - [VTrackerCommand Enum](#45-vtrackercommand-enum)
   - [Frame](#46-frame)
   - [VTrackerParamsMask](#47-vtrackerparamsmask)
5. [Tracker State Machine](#5-tracker-state-machine)
6. [Configuration Guide](#6-configuration-guide)
7. [Usage Patterns](#7-usage-patterns)
8. [Remote Control / Serialization](#8-remote-control--serialization)
9. [Performance and Tuning](#9-performance-and-tuning)
10. [Thread Safety](#10-thread-safety)
11. [Debug and Visualization](#11-debug-and-visualization)
12. [Troubleshooting](#12-troubleshooting)
13. [Algorithm Notes](#13-algorithm-notes)
13a. [EKF Predictive Filtering (optional)](#13a-ekf-predictive-filtering-optional)
13b. [DNN Appearance Verifier (optional)](#13b-dnn-appearance-verifier-optional)
13c. [Vision-Only Robustness Aids](#13c-vision-only-robustness-aids)
14. [Platform Notes (Jetson)](#14-platform-notes-jetson)

---

## 1. Quick Start

```cpp
#include <cvtracker/CvTracker.h>
using namespace cr::vtracker;
using namespace cr::video;

int main()
{
    CvTracker tracker;

    // 1 — configure
    VTrackerParams params;
    params.rectWidth         = 72;    // expected object size
    params.rectHeight        = 72;
    params.searchWindowWidth = 256;   // area to search within
    params.searchWindowHeight= 256;
    params.frameBufferSize   = 32;
    params.maxFramesInLostMode = 90;  // ~3 s at 30 FPS
    params.lostModeOption    = 1;     // predict by velocity while lost
    tracker.initVTracker(params);

    // 2 — allocate frame (once)
    Frame frame(1920, 1080, Fourcc::NV12);
    int frameCounter = 0;

    // 3 — feed frames
    while (getFrameFromCamera(frame.data, frame.size))
    {
        frame.frameId = frameCounter++;   // REQUIRED — unique per frame
        tracker.processFrame(frame);

        tracker.getParams(params);
        printf("mode=%d  prob=%.2f  pos=(%d,%d)\n",
               params.mode, params.detectionProbability,
               params.objectX, params.objectY);
    }
}
```

To capture (lock onto) an object at pixel `(x, y)`:
```cpp
tracker.executeCommand(VTrackerCommand::CAPTURE, (float)x, (float)y, -1.0f);
```

To release the object:
```cpp
tracker.executeCommand(VTrackerCommand::RESET);
```

---

## 2. Building and Installing

### Prerequisites

| Requirement | CPU build | CUDA build |
|---|---|---|
| CMake | 3.14+ | 3.18+ |
| C++ compiler | GCC 9+ / Clang 10+ (C++17) | Same |
| CUDA toolkit | — | 11.4+ (JetPack 5.x) |
| OpenCV | — | — (examples only) |

There are **no mandatory external libraries**. The FFT is bundled.

### Build Options

```bash
# Default: static library, CPU-only, no examples, no tests
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Enable CUDA/cuFFT acceleration (Jetson)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCVTRACKER_WITH_CUDA=ON

# Enable the TensorRT DNN feature extractor (Jetson, JetPack ships TensorRT)
cmake -B build ... -DCVTRACKER_WITH_TENSORRT=ON

# Build the live camera demo (needs OpenCV)
cmake -B build ... -DCVTRACKER_BUILD_EXAMPLES=ON

# Build the self-test suite
cmake -B build ... -DCVTRACKER_BUILD_TESTS=ON

# Build as shared library (.so)
cmake -B build ... -DBUILD_SHARED_LIBS=ON

# Combine flags as needed
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVTRACKER_WITH_CUDA=ON \
  -DCVTRACKER_BUILD_EXAMPLES=ON \
  -DCVTRACKER_BUILD_TESTS=ON
```

```bash
cmake --build build -j$(nproc)
sudo cmake --install build          # installs to /usr/local by default
```

### Default Install Paths

| Artifact | Path |
|---|---|
| Public headers | `/usr/local/include/cvtracker/` |
| Static library | `/usr/local/lib/libcvtracker.a` |
| Shared library | `/usr/local/lib/libcvtracker.so` |
| CMake config | `/usr/local/lib/cmake/cvtracker/` |

### Running the Tests

```bash
cmake -B build -DCVTRACKER_BUILD_TESTS=ON
cmake --build build
./build/test_tracker
# Expected: ALL TESTS PASSED
```

---

## 3. Integrating into Your Project

### Option A — Using an Installed Package (Recommended)

After `cmake --install build`:

```cmake
# CMakeLists.txt of your project
find_package(cvtracker REQUIRED)
target_link_libraries(my_app PRIVATE cvtracker::cvtracker)
```

### Option B — Subdirectory (No Install Needed)

```cmake
add_subdirectory(path/to/cvtracker)
target_link_libraries(my_app PRIVATE cvtracker)
```

### Option C — Plain Makefile

```makefile
CXXFLAGS = -std=c++17 -O3 -I/usr/local/include
LDFLAGS  = -L/usr/local/lib -lcvtracker -lpthread

# Add for CUDA build:
LDFLAGS += -lcufft -lcudart
```

### Minimum Include

```cpp
#include <cvtracker/CvTracker.h>   // brings in VTracker.h and Frame.h automatically
```

---

## 4. Core Classes

### 4.1 `CvTracker`

**Header:** `<cvtracker/CvTracker.h>`  
**Namespace:** `cr::vtracker`  
**Inherits:** `VTracker`

The concrete tracker implementation. All methods are thread-safe (internal mutex).

```cpp
class CvTracker : public VTracker
{
public:
    CvTracker();
    ~CvTracker() override;

    static std::string getVersion();

    bool  initVTracker(VTrackerParams& params)          override;
    bool  setParam    (VTrackerParam id, float value)   override;
    float getParam    (VTrackerParam id)                override;
    void  getParams   (VTrackerParams& params)          override;
    bool  executeCommand(VTrackerCommand id,
                         float arg1 = 0.f,
                         float arg2 = 0.f,
                         float arg3 = 0.f)             override;
    bool  processFrame(cr::video::Frame& frame)         override;
    void  getImage    (int type, cr::video::Frame& img) override;
};
```

#### Method Reference

---

##### `static std::string getVersion()`
Returns the library version string (e.g. `"1.0.0"`).

---

##### `bool initVTracker(VTrackerParams& params)`

Initialize the tracker. Must be called before the first `processFrame()`. May be called again to reinitialize with new settings.

| Aspect | Detail |
|---|---|
| `params` | In/out. User-settable fields are read; state fields (mode, rectX, etc.) are reset to defaults. |
| Returns | `true` on success. `false` if any parameter is out of valid range. |

**Validation rules:**
- `searchWindowWidth` / `searchWindowHeight` ≥ 16  
- `rectWidth` / `rectHeight` ≥ 4  
- `frameBufferSize` ≥ 1  
- `maxFramesInLostMode` ≥ 1  
- `lostModeOption` ∈ {0, 1, 2}

---

##### `bool setParam(VTrackerParam id, float value)`

Change a single parameter at runtime. Takes effect on the next `processFrame()` call.

| Aspect | Detail |
|---|---|
| Returns | `true` on success, `false` if `id` is unknown or `value` is out of range. |

See [VTrackerParam Enum](#44-vtrackerparam-enum) for valid IDs and ranges.

---

##### `float getParam(VTrackerParam id)`

Read the current value of a single parameter.

| Aspect | Detail |
|---|---|
| Returns | Current value, or `-1.0f` if `id` is unknown. |

---

##### `void getParams(VTrackerParams& params)`

Snapshot of all parameters **and** tracking results into `params`.

---

##### `bool executeCommand(VTrackerCommand id, float arg1, float arg2, float arg3)`

Execute an action command. See [VTrackerCommand Enum](#45-vtrackercommand-enum) for argument semantics.

| Aspect | Detail |
|---|---|
| Returns | `true` on success, `false` if `id` is unknown or preconditions are not met. |

---

##### `bool processFrame(cr::video::Frame& frame)`

Process one video frame. **Must be called for every frame** — never skip frames, as the internal ring buffer and velocity estimator depend on continuity.

| Aspect | Detail |
|---|---|
| `frame.frameId` | **Required** — must be a unique, monotonically increasing integer. |
| Returns `false` | Frame dimensions changed (tracker auto-resets), unsupported `fourcc`, or invalid dimensions. |
| Returns `true` | Normal operation. |
| Side effects | Buffers frame in ring buffer; processes all buffered-but-unprocessed frames (usually 1; more after a stop-frame capture). |
| Performance | ~10 ms/frame for 1080p with 256×256 window on a single x86 core. |

---

##### `void getImage(int type, cr::video::Frame& image)`

Retrieve internal diagnostic images (GRAY format, size = internal FFT window, 64–512 px).

| `type` | Content |
|---|---|
| `0` | Reference image — running-average luma of the captured object |
| `1` | Object mask — binary (0 or 255), estimated from reference statistics |
| `2` | Correlation response surface — float peaks normalized to 0–255 |

---

### 4.2 `VTracker` (Abstract Interface)

**Header:** `<cvtracker/VTracker.h>`

Defines the contract for all tracker implementations. Use pointers of this type for polymorphism. Also provides static serialization helpers (see [Remote Control](#8-remote-control--serialization)).

```cpp
class VTracker
{
public:
    virtual ~VTracker() = default;

    virtual bool  initVTracker (VTrackerParams& params)           = 0;
    virtual bool  setParam     (VTrackerParam id, float value)    = 0;
    virtual float getParam     (VTrackerParam id)                 = 0;
    virtual void  getParams    (VTrackerParams& params)           = 0;
    virtual bool  executeCommand(VTrackerCommand id,
                                 float arg1 = 0.f,
                                 float arg2 = 0.f,
                                 float arg3 = 0.f)                = 0;
    virtual bool  processFrame (cr::video::Frame& frame)          = 0;
    virtual void  getImage     (int type, cr::video::Frame& img)  = 0;

    // Serialization (non-virtual, static or virtual)
    virtual bool decodeAndExecuteCommand(uint8_t* data, int size);

    static void encodeSetParamCommand(uint8_t* data, int& size,
                                      VTrackerParam id, float value);
    static void encodeCommand        (uint8_t* data, int& size,
                                      VTrackerCommand id,
                                      float arg1 = 0.f,
                                      float arg2 = 0.f,
                                      float arg3 = 0.f);
    static int  decodeCommand        (uint8_t* data, int size,
                                      VTrackerParam& paramId,
                                      VTrackerCommand& commandId,
                                      float& v1, float& v2, float& v3);
};
```

---

### 4.3 `VTrackerParams`

**Header:** `<cvtracker/VTracker.h>`

Data struct for configuration and tracking results. Fields are divided into two categories:

#### State / Results (written by tracker, read by user)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `int` | 0 | Tracker state: 0=FREE 1=TRACKING 2=LOST 3=INERTIAL 4=STATIC |
| `rectX` | `int` | 0 | Tracking rectangle center X (px) |
| `rectY` | `int` | 0 | Tracking rectangle center Y (px) |
| `objectX` | `int` | 0 | Estimated object center X (px) |
| `objectY` | `int` | 0 | Estimated object center Y (px) |
| `objectWidth` | `int` | 72 | Estimated object width (px) |
| `objectHeight` | `int` | 72 | Estimated object height (px) |
| `velX` | `float` | 0 | Horizontal velocity (px/frame) |
| `velY` | `float` | 0 | Vertical velocity (px/frame) |
| `detectionProbability` | `float` | 0 | Detection confidence [0..1] |
| `lostModeFrameCounter` | `int` | 0 | Frames spent in LOST mode so far |
| `frameCounter` | `int` | 0 | Total frames processed since last capture |
| `frameWidth` | `int` | 0 | Input frame width (px) |
| `frameHeight` | `int` | 0 | Input frame height (px) |
| `searchWindowX` | `int` | 0 | Search window center X for next frame |
| `searchWindowY` | `int` | 0 | Search window center Y for next frame |
| `processedFrameId` | `int` | 0 | ID of the last processed frame |
| `frameId` | `int` | 0 | ID of the most recent buffered frame |
| `processingTimeMks` | `int` | 0 | Processing time of last frame (µs) |

#### User-Configurable (set before `initVTracker` or via `setParam`)

| Field | Type | Default | Valid Range | Meaning |
|---|---|---|---|---|
| `rectWidth` | `int` | 72 | ≥ 4 | Tracking rectangle width (px) |
| `rectHeight` | `int` | 72 | ≥ 4 | Tracking rectangle height (px) |
| `searchWindowWidth` | `int` | 256 | ≥ 16 | Search area width (px) |
| `searchWindowHeight` | `int` | 256 | ≥ 16 | Search area height (px) |
| `lostModeOption` | `int` | 0 | 0–2 | Behavior when lost: 0=freeze 1=predict by velocity 2=predict + reset at edge |
| `frameBufferSize` | `int` | 128 | ≥ 1 | Ring buffer depth (enables stop-frame capture) |
| `maxFramesInLostMode` | `int` | 128 | ≥ 1 | Auto-reset after this many LOST frames |
| `rectAutoSize` | `bool` | false | — | Continuously auto-resize rect to match object |
| `rectAutoPosition` | `bool` | false | — | Continuously center rect on object |
| `multipleThreads` | `bool` | false | — | Multi-thread FFT (CPU backend) |
| `numChannels` | `int` | 2 | — | Reserved |
| `type` | `int` | 0 | 0 or 1 | Camera type: 0=daylight 1=thermal |
| `custom1` | `float` | 0 | — | Loss threshold (≤0 uses default 0.1) |
| `custom2` | `float` | 0 | — | Re-capture threshold (≤0 uses default 0.4) |
| `custom3` | `float` | 0 | — | Pattern learning rate (≤0 uses default 0.075) |
| `ekfEnabled` | `bool` | false | — | Enable CTRV EKF predictive filtering (see [§13a](#13a-ekf-predictive-filtering-optional)) |
| `dnnVerifierEnabled` | `bool` | false | — | Enable DNN appearance verifier (see [§13b](#13b-dnn-appearance-verifier-optional)) |
| `dnnVerifyInterval` | `int` | 6 | ≥ 1 | Frames between verifications in TRACKING mode |
| `dnnVetoThreshold` | `float` | 0.45 | 0–1 | Similarity below = appearance mismatch |
| `dnnAcceptThreshold` | `float` | 0.60 | 0–1 | Similarity required for re-capture / bank add |
| `dnnSimilarity` | `float` | 0 | — | **Read-only result**: last verifier similarity [0..1] |
| `gmcEnabled` | `bool` | false | — | Global Motion Compensation (see [§13c](#13c-vision-only-robustness-aids)) |
| `gmcShiftX/Y` | `float` | 0 | — | **Read-only result**: last measured scene shift, frame px |
| `dnnArbitration` | `bool` | false | — | Multi-peak identity arbitration in TRACKING (§13c) |
| `dnnReacquisition` | `bool` | false | — | Embedding grid re-acquisition in LOST (§13c) |

#### Serialization

```cpp
bool encode(uint8_t* data, int bufferSize, int& size,
            VTrackerParamsMask* mask = nullptr);
bool decode(uint8_t* data, int dataSize);
```

Encode packs selected fields into a binary buffer; decode restores them. Useful for telemetry links. See [VTrackerParamsMask](#47-vtrackerparamsmask).

---

### 4.4 `VTrackerParam` Enum

Used with `setParam()` / `getParam()`.

```cpp
enum class VTrackerParam
{
    SEARCH_WINDOW_WIDTH  = 1,
    SEARCH_WINDOW_HEIGHT,
    RECT_WIDTH,
    RECT_HEIGHT,
    LOST_MODE_OPTION,
    FRAME_BUFFER_SIZE,
    MAX_FRAMES_IN_LOST_MODE,
    RECT_AUTO_SIZE,
    RECT_AUTO_POSITION,
    MULTIPLE_THREADS,
    NUM_CHANNELS,
    TYPE,
    CUSTOM_1,             // loss threshold
    CUSTOM_2,             // re-capture threshold
    CUSTOM_3,             // learning rate
    ENABLE_EKF,           // 0 = off (default), 1 = CTRV EKF filtering
    ENABLE_DNN_VERIFIER,  // 0 = off (default), 1 = appearance verification
    DNN_VERIFY_INTERVAL,  // frames between verifications (default 6)
    DNN_VETO_THRESHOLD,   // similarity below = mismatch (default 0.45)
    DNN_ACCEPT_THRESHOLD, // similarity for re-capture/bank add (default 0.60)
    ENABLE_GMC,           // 0 = off (default), 1 = global motion compensation
    ENABLE_DNN_ARBITRATION,   // 0 = off (default), 1 = multi-peak identity pick
    ENABLE_DNN_REACQUISITION, // 0 = off (default), 1 = embedding LOST search
};
```

---

### 4.5 `VTrackerCommand` Enum

Used with `executeCommand()`.

```cpp
enum class VTrackerCommand
{
    CAPTURE = 1,
    CAPTURE_PERCENTS,
    RESET,
    SET_INERTIAL_MODE,
    SET_LOST_MODE,
    SET_STATIC_MODE,
    ADJUST_RECT_SIZE,
    ADJUST_RECT_POSITION,
    MOVE_RECT,
    SET_RECT_POSITION,
    SET_RECT_POSITION_PERCENTS,
    MOVE_SEARCH_WINDOW,
    SET_SEARCH_WINDOW_POSITION,
    SET_SEARCH_WINDOW_POSITION_PERCENTS,
    CHANGE_RECT_SIZE,
};
```

#### Command Details

| Command | `arg1` | `arg2` | `arg3` | Precondition | Effect |
|---|---|---|---|---|---|
| `CAPTURE` | X (px) | Y (px) | frameId (-1 = latest) | Any mode | Capture object at (X,Y); switch to TRACKING. If `arg3 ≥ 0`, uses that buffered frame (stop-frame capture). |
| `CAPTURE_PERCENTS` | X (%) | Y (%) | — | Any mode | Same as CAPTURE but coordinates as % of frame dimensions. |
| `RESET` | — | — | — | Any | Return to FREE; clear object. |
| `SET_INERTIAL_MODE` | — | — | — | TRACKING / LOST / STATIC | Coast by velocity; freeze pattern. |
| `SET_LOST_MODE` | — | — | — | TRACKING / INERTIAL / STATIC | Force search / re-capture attempt. |
| `SET_STATIC_MODE` | — | — | — | TRACKING / LOST / INERTIAL | Freeze rect position and size. |
| `ADJUST_RECT_SIZE` | — | — | — | Any | One-shot: snap rect to estimated object size; request re-train. |
| `ADJUST_RECT_POSITION` | — | — | — | Any | One-shot: snap rect center to smoothed object offset. |
| `MOVE_RECT` | dX (px) | dY (px) | — | Any | Move tracking rect by offset (clamped to frame). |
| `SET_RECT_POSITION` | X (px) | Y (px) | — | mode == FREE | Set rect position (FREE mode only). |
| `SET_RECT_POSITION_PERCENTS` | X (%) | Y (%) | — | mode == FREE | Same, as percentage of frame. |
| `MOVE_SEARCH_WINDOW` | dX (px) | dY (px) | — | Any | Relative offset for next frame's search window. |
| `SET_SEARCH_WINDOW_POSITION` | X (px) | Y (px) | — | Any | Absolute position for next frame's search window. |
| `SET_SEARCH_WINDOW_POSITION_PERCENTS` | X (%) | Y (%) | — | `frameWidth > 0` | Same, as percentage. |
| `CHANGE_RECT_SIZE` | dW (px) | dH (px) | — | Any | Grow/shrink rect (minimum 4×4 enforced). |

---

### 4.6 `Frame`

**Header:** `<cvtracker/Frame.h>` (included transitively by `CvTracker.h`)  
**Namespace:** `cr::video`

Container for one raw video frame.

```cpp
class Frame
{
public:
    int      width   {0};
    int      height  {0};
    Fourcc   fourcc  {Fourcc::GRAY};
    uint32_t size    {0};       // buffer size in bytes
    int32_t  frameId {-1};      // unique frame ID — user must set
    int32_t  sourceId{-1};      // reserved
    uint8_t* data    {nullptr}; // owned pixel buffer

    Frame();
    Frame(int w, int h, Fourcc f);       // allocates buffer
    Frame(const Frame&);                  // deep copy
    Frame& operator=(const Frame&);       // deep copy
    ~Frame();                             // frees buffer

    bool init   (int w, int h, Fourcc f); // (re)allocate
    void release();                        // free buffer

    static uint32_t dataSize(int w, int h, Fourcc f);
};
```

#### Supported Pixel Formats

```cpp
enum class Fourcc : uint32_t
{
    GRAY,   // 8-bit mono — 1 byte/pixel
    YUV24,  // packed YUV — 3 bytes/pixel
    YUYV,   // packed 4:2:2 — 2 bytes/pixel (Y0 U0 Y1 V0)
    UYVY,   // packed 4:2:2 — 2 bytes/pixel (U0 Y0 V0 Y1)
    NV12,   // semi-planar Y + interleaved UV 4:2:0 — 1.5 bytes/pixel avg
    NV21,   // semi-planar Y + interleaved VU 4:2:0 — 1.5 bytes/pixel avg
    YU12,   // planar I420 — 1.5 bytes/pixel avg
    YV12,   // planar YV12 — 1.5 bytes/pixel avg
};
```

The tracker extracts the luma (Y) plane internally; all listed formats are accepted.

#### Frame Setup

```cpp
// Allocate once
cr::video::Frame frame(1920, 1080, cr::video::Fourcc::NV12);

// Per-frame from camera DMA buffer (zero-copy alternative: point data at DMA buffer)
std::memcpy(frame.data, cameraBuffer, frame.size);
frame.frameId = frameCounter++;   // REQUIRED
tracker.processFrame(frame);
```

---

### 4.7 `VTrackerParamsMask`

Optional bitmask for selective parameter encoding. All fields default to `true` (include all).

```cpp
typedef struct VTrackerParamsMask
{
    bool mode{true};
    bool rectX{true};
    bool rectY{true};
    bool rectWidth{true};
    bool rectHeight{true};
    bool objectX{true};
    bool objectY{true};
    bool objectWidth{true};
    bool objectHeight{true};
    bool lostModeFrameCounter{true};
    bool frameCounter{true};
    bool frameWidth{true};
    bool frameHeight{true};
    bool searchWindowWidth{true};
    bool searchWindowHeight{true};
    bool searchWindowX{true};
    bool searchWindowY{true};
    bool lostModeOption{true};
    bool frameBufferSize{true};
    bool maxFramesInLostMode{true};
    bool processedFrameId{true};
    bool frameId{true};
    bool velX{true};
    bool velY{true};
    bool detectionProbability{true};
    bool rectAutoSize{true};
    bool rectAutoPosition{true};
    bool multipleThreads{true};
    bool numChannels{true};
    bool type{true};
    bool processingTimeMks{true};
    bool custom1{true};
    bool custom2{true};
    bool custom3{true};
    bool ekfEnabled{true};
    bool dnnVerifierEnabled{true};
    bool dnnVerifyInterval{true};
    bool dnnVetoThreshold{true};
    bool dnnAcceptThreshold{true};
    bool dnnSimilarity{true};
    bool gmcEnabled{true};
    bool gmcShiftX{true};
    bool gmcShiftY{true};
    bool dnnArbitration{true};
    bool dnnReacquisition{true};
} VTrackerParamsMask;
```

**Usage:**
```cpp
VTrackerParamsMask mask;       // defaults: all true
mask.frameCounter = false;     // skip this field
mask.processingTimeMks = false;

uint8_t buf[512];
int size = 0;
params.encode(buf, sizeof(buf), size, &mask);
```

---

## 5. Tracker State Machine

```
         RESET / init
              │
              ▼
    ┌─────────────────┐
    │      FREE       │  ◄─── RESET from any state
    └────────┬────────┘
             │ CAPTURE
             ▼
    ┌─────────────────┐
    │    TRACKING     │ ◄─────────────────────────────┐
    └──┬──────────┬───┘                               │
       │ loss     │ SET_INERTIAL_MODE / SET_LOST_MODE  │ re-capture
       ▼          │                                   │
    ┌──────────┐  │    ┌──────────────┐               │
    │   LOST   │  └───►│  INERTIAL   │               │
    └──┬───────┘       └──────┬───────┘               │
       │ auto re-capture      │ SET_STATIC_MODE        │
       │ (prob > CUSTOM_2)    ▼                        │
       └──────────────► ┌──────────┐                  │
                        │  STATIC  │──────────────────►┘
                        └──────────┘ SET_LOST_MODE /
                                     SET_INERTIAL_MODE
```

### Mode Values

| Value | Name | Behavior |
|---|---|---|
| 0 | FREE | Idle. Awaiting CAPTURE command. |
| 1 | TRACKING | Object locked. Correlates, detects, updates pattern each frame. |
| 2 | LOST | Object not found. Searches in search window. Auto re-captures when `detectionProbability > CUSTOM_2`. Auto-resets after `maxFramesInLostMode` frames. |
| 3 | INERTIAL | Coasting by last known velocity. No pattern update. |
| 4 | STATIC | Rect frozen. No movement, no pattern update. |

---

## 6. Configuration Guide

### Sizing the Tracking Rectangle

Set `rectWidth` / `rectHeight` to closely match the **expected object size** in the image:
- Too large: includes background, degrades correlation.
- Too small: misses object features, loses stability.
- Rule of thumb: cover 80–100% of the object bounding box.

### Sizing the Search Window

`searchWindowWidth` / `searchWindowHeight` determines how far from the last position the tracker searches:
- Must be larger than `rectWidth` / `rectHeight`.
- Typical: 2–4× the rect size.
- Larger = more robust to fast motion; slower.
- 256×256 is a good starting point for most scenarios.

### Frame Buffer Size

`frameBufferSize` controls how many past frames are stored:
- Enables **stop-frame capture**: send CAPTURE with a past `frameId` to handle latency.
- Higher value = more memory. For 1080p luma-only: ~2 MB per frame.
- If stop-frame is not needed, set to 1–4.

### Lost Mode Behavior (`lostModeOption`)

| Value | Behavior when lost |
|---|---|
| 0 | Freeze rect at last known position. |
| 1 | Predict position by extrapolating last velocity. Stop at frame edge. |
| 2 | Predict by velocity. Reset to FREE if position reaches frame edge. |

### Custom Tuning Parameters

These can be set to 0 or negative to use the internal defaults:

| Parameter | Enum | Default | Effect |
|---|---|---|---|
| Loss threshold | `CUSTOM_1` | 0.1 | `detectionProbability` below this triggers LOST mode. Lower = more sensitive to loss. |
| Re-capture threshold | `CUSTOM_2` | 0.4 | Probability required to auto re-lock from LOST mode. |
| Learning rate | `CUSTOM_3` | 0.075 | How quickly the filter adapts to appearance changes. Higher = faster adaptation, lower = more stable. |

**Tuning advice:**
- Objects with partial occlusion: lower `CUSTOM_1` (0.05) to tolerate lower probability.
- Fast appearance change (thermal targets): raise `CUSTOM_3` (0.1–0.15).
- Drifting onto background: lower `CUSTOM_3` (0.03–0.05).
- Fails to re-capture after occlusion: lower `CUSTOM_2` (0.3).

---

## 7. Usage Patterns

### 7.1 Basic Tracking Loop

```cpp
#include <cvtracker/CvTracker.h>
using namespace cr::vtracker;
using namespace cr::video;

CvTracker tracker;

VTrackerParams params;
params.rectWidth  = 72;
params.rectHeight = 72;
params.searchWindowWidth  = 256;
params.searchWindowHeight = 256;
params.maxFramesInLostMode = 90;
params.lostModeOption = 1;
tracker.initVTracker(params);

Frame frame(1920, 1080, Fourcc::NV12);
int id = 0;

while (camera.grab(frame.data)) {
    frame.frameId = id++;
    tracker.processFrame(frame);
    tracker.getParams(params);
    // params.mode, params.objectX, params.objectY, params.detectionProbability
}
```

### 7.2 Capture on Mouse Click

```cpp
void onMouseClick(int x, int y)
{
    tracker.executeCommand(VTrackerCommand::CAPTURE,
                           (float)x, (float)y, -1.0f);
    // arg3 = -1 means "use the most recent buffered frame"
}
```

To capture using percentage coordinates (useful for resolution-independent UIs):
```cpp
// x_pct, y_pct in [0..100]
tracker.executeCommand(VTrackerCommand::CAPTURE_PERCENTS,
                       x_pct, y_pct);
```

### 7.3 Stop-Frame Capture (Compensating for Display Latency)

When the user sees a delayed video feed (e.g. display latency of N frames), capture from the frame that was actually shown:

```cpp
// frameIdShownToUser = frame ID that was on screen when user clicked
tracker.executeCommand(VTrackerCommand::CAPTURE,
                       (float)clickX, (float)clickY,
                       (float)frameIdShownToUser);
// Tracker will find that frame in the ring buffer, capture from it,
// then catch up by processing all buffered frames up to present.
```

Ensure `frameBufferSize` is at least as large as the maximum expected latency in frames.

### 7.4 Dynamic Parameter Adjustment

```cpp
// Reduce learning rate for a stable target
tracker.setParam(VTrackerParam::CUSTOM_3, 0.03f);

// Lower loss threshold (detect loss sooner)
tracker.setParam(VTrackerParam::CUSTOM_1, 0.05f);

// Enable continuous auto-sizing
tracker.setParam(VTrackerParam::RECT_AUTO_SIZE, 1.0f);

// Grow rect manually
tracker.executeCommand(VTrackerCommand::CHANGE_RECT_SIZE, 10.0f, 10.0f);

// Move search window
tracker.executeCommand(VTrackerCommand::MOVE_SEARCH_WINDOW, 50.0f, 0.0f);
```

### 7.5 Switching Modes

```cpp
// Temporarily freeze the tracker position
tracker.executeCommand(VTrackerCommand::SET_STATIC_MODE);

// Resume tracking (via LOST mode search)
tracker.executeCommand(VTrackerCommand::SET_LOST_MODE);

// Coast by velocity (e.g. object temporarily hidden)
tracker.executeCommand(VTrackerCommand::SET_INERTIAL_MODE);

// Snap rect to best estimated object size
tracker.executeCommand(VTrackerCommand::ADJUST_RECT_SIZE);

// Snap rect position to estimated object center
tracker.executeCommand(VTrackerCommand::ADJUST_RECT_POSITION);
```

### 7.6 Reading Processing Time

```cpp
tracker.getParams(params);
printf("Frame took %d µs (%.1f ms)\n",
       params.processingTimeMks,
       params.processingTimeMks / 1000.0f);
```

---

## 8. Remote Control / Serialization

CvTracker supports encoding commands and parameters into compact binary buffers for transmission over serial, UDP, or any byte-oriented link.

### Encoding (Transmitter Side)

```cpp
uint8_t buf[32];
int size = 0;

// Encode an action command
VTracker::encodeCommand(buf, size,
                        VTrackerCommand::CAPTURE,
                        320.0f, 240.0f, -1.0f);
// size is now 19 bytes
sendBytes(buf, size);

// Encode a parameter change
VTracker::encodeSetParamCommand(buf, size,
                                VTrackerParam::CUSTOM_1, 0.05f);
// size is now 11 bytes
sendBytes(buf, size);
```

### Decoding and Executing (Receiver Side)

```cpp
// Atomic: decode + execute in one call
uint8_t buf[32];
int size = receiveBytes(buf);
bool ok = tracker.decodeAndExecuteCommand(buf, size);

// Or manually decode
VTrackerParam  paramId;
VTrackerCommand cmdId;
float v1, v2, v3;
int result = VTracker::decodeCommand(buf, size, paramId, cmdId, v1, v2, v3);
// result: 0 = action command, 1 = SET_PARAM command, -1 = error
```

### Sending Telemetry Back

```cpp
VTrackerParams telem;
tracker.getParams(telem);

uint8_t teleBuf[512];
int teleSize = 0;
telem.encode(teleBuf, sizeof(teleBuf), teleSize);
sendBytes(teleBuf, teleSize);

// ---- Ground side ----
VTrackerParams recv;
recv.decode(teleBuf, teleSize);
printf("Remote mode=%d prob=%.2f\n", recv.mode, recv.detectionProbability);
```

### Buffer Size Requirements

| Message type | Buffer size |
|---|---|
| SET_PARAM command | 11 bytes |
| Action command | 19 bytes |
| Full params (no mask) | ≤ 512 bytes |

---

## 9. Performance and Tuning

### Measured Performance

| Platform | Resolution | Window | Threads | Time/frame |
|---|---|---|---|---|
| x86 (single core) | 1920×1080 | 256×256 | Off | ~10 ms |
| Jetson Xavier NX | 1920×1080 | 256×256 | Off | 10–25 ms |
| Jetson Xavier NX | 1920×1080 | 128×128 | Off | 2.5–6 ms |
| Jetson Xavier NX | 1920×1080 | 256×256 | CUDA | 2–3 ms |
| Jetson Xavier NX | 1920×1080 | 256×256 | CPU multi | 5–10 ms |
| Jetson Nano | 1920×1080 | 256×256 | Off | 30–40 ms |
| Jetson Nano | 1280×720 | 128×128 | Off | ~8 ms |

### Memory Usage (Approximate, 1080p, 256×256 Window)

| Component | Size |
|---|---|
| Frame buffer (32 frames, luma only) | ~64 MB |
| FFT workspace | ~4 MB |
| Filter matrices A/B | ~2 MB |
| **Total** | **~70 MB** |

**Reduce memory:**
- Smaller buffer: `frameBufferSize = 4` → ~8 MB buffer
- Smaller window: 128×128 → halves FFT memory
- Lower resolution input: 720p → ~25% savings

### Throughput Recommendations by Platform

| Platform | Recommended Window | Recommended Buffer | Notes |
|---|---|---|---|
| Jetson Orin | 512×512 | 64 | Has cuFFT, use CUDA build |
| Jetson Xavier NX | 256×256 | 32 | CUDA build preferred |
| Jetson Xavier AGX | 256×256 | 32 | CUDA build preferred |
| Jetson TX2 | 128×128 | 8 | CPU build fine, 6-core |
| Jetson Nano | 128×128 | 4 | CPU only, tight timing |
| x86 desktop/server | 256–512 | 64–128 | multipleThreads=true |
| Raspberry Pi 4 | 128×128 | 4 | C++17 required |

### System-Level Performance Tips (Linux/Jetson)

```bash
# Pin tracker thread to isolated CPU cores (reduces cache thrashing)
taskset -c 4,5 ./my_tracker_app

# Enable real-time scheduling
sudo chrt -f 80 ./my_tracker_app

# Jetson: maximum performance mode (lock clocks)
sudo nvpmodel -m 0      # full power envelope
sudo jetson_clocks      # lock all clocks at max
```

---

## 10. Thread Safety

All public `CvTracker` methods are thread-safe (protected by an internal mutex).

### Recommended Threading Model

```cpp
// Thread A — video capture (single producer)
void captureThread()
{
    while (running) {
        frame.frameId = atomicCounter++;
        fillFrameFromCamera(frame);
        tracker.processFrame(frame);   // call from ONE thread only
    }
}

// Thread B — UI / control (any thread)
void onUserClick(int x, int y)
{
    tracker.executeCommand(VTrackerCommand::CAPTURE,
                           (float)x, (float)y, -1.0f);
}

// Thread C — display (any thread)
void displayThread()
{
    VTrackerParams r;
    while (running) {
        tracker.getParams(r);
        drawOverlay(r.objectX, r.objectY, r.objectWidth, r.objectHeight, r.mode);
        sleepMs(16);
    }
}

// Thread D — telemetry (any thread)
void telemetryThread()
{
    uint8_t buf[32];
    int size = 0;
    while (running) {
        size = recvBytes(buf);
        tracker.decodeAndExecuteCommand(buf, size);  // atomic
    }
}
```

**Key rule:** `processFrame()` should only be called from a **single** thread. All other methods may be called from any thread concurrently.

---

## 11. Debug and Visualization

### Retrieve Internal Images

```cpp
cr::video::Frame refImg, maskImg, surfImg;

tracker.getImage(0, refImg);   // reference template
tracker.getImage(1, maskImg);  // binary object mask
tracker.getImage(2, surfImg);  // correlation response surface

// All images: GRAY format, size = internal FFT window (64–512 px)
// Display or save with any image library
```

### Interpret `detectionProbability`

The probability is computed from:
- **Peak term** — height of the correlation peak (~1 on object, ~0 on background)
- **PSR term** — peak-to-sidelobe ratio (sharpness of peak)

```
probability = clamp(peak × 1.6, 0, 1) × clamp((PSR − 2) / 10, 0, 1)
```

Values to watch:
- > 0.4 — strong lock, tracking confidently
- 0.1–0.4 — weak detection, may drift
- < 0.1 (CUSTOM_1 default) — triggers LOST mode

### Monitor Processing Time

```cpp
tracker.getParams(params);
if (params.processingTimeMks > 20000)  // > 20 ms
    fprintf(stderr, "Tracker too slow: %d µs\n", params.processingTimeMks);
```

---

## 12. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `processFrame()` returns `false` | Wrong `fourcc` or frame size changed | Verify format; tracker auto-resets on size change — this is normal |
| Tracker resets immediately after capture | Rect too large, captures too much background | Reduce `rectWidth`/`rectHeight` to match object tightly |
| Loses low-contrast objects immediately | Default loss threshold too high | Lower `CUSTOM_1` to 0.05 |
| Rect drifts onto background over time | Learning rate too high | Lower `CUSTOM_3` to 0.03 |
| Never re-captures after occlusion | Re-capture threshold too high | Lower `CUSTOM_2` to 0.3 |
| Slow performance on Jetson Nano | Search window too large | Reduce to 128×128; build with `-DCVTRACKER_WITH_CUDA=ON` if Jetson |
| Build error: cuFFT not found | CUDA not installed or wrong version | Ensure JetPack 5.x / CUDA 11.4+; check `nvcc --version` |
| Object size estimate wrong | `rectAutoSize` off | Enable `RECT_AUTO_SIZE` or use `ADJUST_RECT_SIZE` command |
| High latency between click and capture | Frame buffer too small | Increase `frameBufferSize` to cover display latency in frames |
| Memory usage too high | Frame buffer too large | Lower `frameBufferSize` |

---

## 13. Algorithm Notes

### Filter Training (on CAPTURE)

1. Crop the search window at the capture point.
2. Apply log-transform + zero-mean + unit-norm normalization (lighting invariant).
3. Generate a Gaussian response label centered on the object.
4. Train with the base sample plus 8 perturbations (rotated/scaled at ×0.5 weight).
5. Compute optimal correlation filter in frequency domain (MOSSE formulation).

### Detection (each TRACKING frame)

1. Extract and preprocess search window at predicted position.
2. Compute cross-correlation in frequency domain (two FFTs + pointwise multiply + IFFT).
3. Locate peak with subpixel refinement (parabolic fit).
4. Compute PSR and combine into `detectionProbability`.
5. Update filter using exponential moving average at `CUSTOM_3` learning rate.

### Scale Adaptation

Every 4 frames in TRACKING mode:
- Evaluate three scales: ×0.98, ×1.0, ×1.02.
- Select scale with highest correlation peak (biased toward ×1.0).
- Accumulated scale clamped to [0.2×, 5.0×] the original capture size.

### Object Size Estimation

- Derived from moment analysis of the binary object mask.
- Updated only on high-confidence frames (`prob ≥ CUSTOM_2`).
- Smoothed with EMA (coefficient 0.85–0.9) to suppress jitter.

---

## 13a. EKF Predictive Filtering (optional)

An optional Extended Kalman Filter layers a physical motion model on top of
the correlation tracker. It is **off by default**; enable it with
`params.ekfEnabled = true` before `initVTracker()`, or at runtime via
`setParam(VTrackerParam::ENABLE_EKF, 1.0f)`. When off, the tracker behaves
exactly as the raw correlation tracker (byte-for-byte unchanged).

### Motion model — CTRV

State vector (5D): `[px, py, v, ψ, ω]`
- `px, py` — position (pixels)
- `v` — speed (pixels/frame)
- `ψ` — heading (rad)
- `ω` — turn rate (rad/frame)

Constant-turn-rate-and-velocity: the object is assumed to move at roughly
constant speed along a roughly constant-curvature arc. This is genuinely
nonlinear (justifying an EKF rather than a linear KF) and models curved
ground-object motion well. A straight-line Taylor limit is used when
`|ω|` is near zero to avoid a division singularity.

### Per-frame flow when enabled

1. **Predict** — advance the state one frame via the CTRV transition and its
   analytic Jacobian; grow the covariance by the process noise.
2. **Detect** — the correlation core produces the measured position and
   `detectionProbability` (appearance is the sole gatekeeper for "is the
   object present").
3. **Gate + Update** — fuse the measurement. Measurement noise is scaled by
   `1 / max(prob, 0.05)` (weak detections move the state less) and further
   inflated if the measurement fails the Mahalanobis gate (below).
4. **Output** — `objectX/Y`, `velX/velY` are taken from the filtered state,
   not the raw correlation peak. The appearance filter is still retrained at
   the EKF-corrected position, at a rate proportional to `prob`.

During **LOST** and **INERTIAL** modes the EKF predict step drives the
extrapolation, so the tracker coasts along a curved, physically plausible
path instead of a straight velocity line.

### Mahalanobis outlier gate

Each measurement's squared Mahalanobis distance from the prediction,
`d² = yᵀ S⁻¹ y` (innovation `y`, innovation covariance `S`), is compared to a
chi-square(2) gate. If `d² > MAHALANOBIS_GATE`, the measurement is treated as
a physics-violating outlier — for example the correlation peak sliding along
a straight edge (the aperture problem) — and its noise is inflated by
`d²/gate` (capped), so the correction is damped and the filter trusts its
motion prediction. This is the primary defense against fast edge-drift in
airborne ground-object tracking.

### Tunable constants

These are compile-time constants (not yet runtime params). Adjust and rebuild.

| Constant | File | Default | Units | Effect |
|---|---|---|---|---|
| `EKF_SIGMA_A` | src/CvTracker.cpp | 1.5 | px/frame² | Linear accel noise. ↑ follows sudden motion, ↓ smoother. |
| `EKF_SIGMA_ALPHA` | src/CvTracker.cpp | 0.15 | rad/frame² | Angular accel noise. ↑ tolerates sharp turns. |
| `EKF_R_BASE` | src/CvTracker.cpp | 4.0 | px² | Base measurement variance (÷ prob). ↓ snaps to measurement. |
| `MAHALANOBIS_GATE` | src/Ekf.cpp | 9.21 | — | χ²(2) gate (99%). ↓ stricter physics/edge rejection. |
| `GATE_MAX_INFLATION` | src/Ekf.cpp | 50.0 | — | Max damping applied to a gated outlier measurement. |
| `INIT_POS_VAR` / `INIT_SPEED_VAR` / `INIT_ANGLE_VAR` / `INIT_OMEGA_VAR` | src/Ekf.cpp | 4 / 25 / π² / 0.25 | var | Initial state covariance at capture. Rarely tuned. |

**Suggested presets by platform motion:**

| Scenario | SIGMA_A | SIGMA_ALPHA | R_BASE | GATE |
|---|---|---|---|---|
| Static camera, smooth target | 0.5 | 0.05 | 6.0 | 9.21 |
| Static camera, general | 1.5 | 0.15 | 4.0 | 9.21 |
| Handheld / vehicle camera | 6.0 | 0.6 | 2.5 | 9.21 |
| Airborne / gimbal camera | 12.0 | 1.0 | 2.0 | 9.21 |

> **Note on airborne use:** apparent on-screen motion is dominated by
> aircraft ego-motion, not object motion. The CTRV model treats these as high
> accelerations, so raise the process-noise constants (airborne preset). For
> best results, feed ego-motion compensation (IMU/gimbal telemetry or optical
> flow) upstream before `processFrame()` if available.

### Enabling

```cpp
// At init:
VTrackerParams p;
p.ekfEnabled = true;
tracker.initVTracker(p);

// Or at runtime (A/B toggling on the same stream):
tracker.setParam(VTrackerParam::ENABLE_EKF, 1.0f);  // on
tracker.setParam(VTrackerParam::ENABLE_EKF, 0.0f);  // off
```

The filtered results appear in the usual fields — no new API to read:
`params.objectX/objectY`, `params.velX/velY`.

---

## 13b. DNN Appearance Verifier (optional)

An optional embedding-based verifier layers appearance memory on top of the
correlation tracker. It is **off by default** and requires two things:

1. An extractor: `tracker.setFeatureExtractor(fe);`
2. The flag: `setParam(VTrackerParam::ENABLE_DNN_VERIFIER, 1.0f)` (or
   `params.dnnVerifierEnabled = true` before `initVTracker()`).

Without either one the tracker behaves exactly as before. The core library
stays dependency-free — inference backends plug in from outside.

### What it solves

The correlation filter has a single, slowly-decaying template. During
partial occlusion or slow drift it can gradually learn a distractor and
then confidently track the wrong object. The verifier holds an independent,
non-decaying **template bank** of embeddings of the captured object and:

- **Drift rejection (TRACKING):** every `DNN_VERIFY_INTERVAL` frames, the
  tracked patch is embedded and compared (max cosine similarity) against
  the bank. Below `DNN_VETO_THRESHOLD`: pattern learning is blocked (the
  filter cannot learn the wrong object); after 3 consecutive mismatches the
  tracker is forced to LOST mode.
- **Distractor rejection (LOST):** a re-capture candidate must both exceed
  the correlation re-capture threshold *and* match the bank at
  `DNN_ACCEPT_THRESHOLD`. Look-alikes that fool the correlation filter are
  rejected and the search continues.
- **Richer memory:** high-confidence, temporally spaced embeddings are
  added to the bank (capacity 12, ≥30 frames apart; the capture seed is
  never evicted), so the object is remembered across scale, orientation,
  and lighting states — the object only has to match *one* remembered
  appearance.

### The FeatureExtractor interface

```cpp
#include <cvtracker/FeatureExtractor.h>

class FeatureExtractor
{
public:
    virtual int inputSize() const = 0;   // square patch side, px
    virtual bool extract(const float* patch,             // gray [0..255]
                         std::vector<float>& embedding) = 0;
};
```

The tracker crops the tracking rectangle (×1.25 context), resamples it to
`inputSize()²` grayscale, and calls `extract()` under its internal lock —
implementations should complete in a few milliseconds. Any embedding
dimensionality works; cosine similarity is used throughout.

Two implementations ship with the library:

| Implementation | Header | Dependency | Purpose |
|---|---|---|---|
| `TinyPatchExtractor` | FeatureExtractor.h | none | Normalized downsampled patch (= NCC matching). Pipeline smoke-testing and a weak fallback. |
| TensorRT extractor | TrtFeatureExtractor.h | TensorRT + CUDA (`-DCVTRACKER_WITH_TENSORRT=ON`) | Real DNN embeddings on Jetson. |

### TensorRT usage (Jetson)

```cpp
#include <cvtracker/TrtFeatureExtractor.h>

TrtExtractorConfig cfg;
cfg.onnxPath  = "/home/user/models/embedder.onnx";
cfg.inputSize = 128;      // model input 1x3x128x128 (static shape!)
cfg.channels  = 3;        // gray replicated to 3ch for ImageNet backbones
// cfg.mean / cfg.std default to ImageNet statistics
// cfg.fp16 = true (default) - use FP16 on all Jetson modules

auto fe = createTrtFeatureExtractor(cfg);
if (fe)                    // nullptr if TRT not compiled in / model missing
{
    tracker.setFeatureExtractor(fe);
    tracker.setParam(VTrackerParam::ENABLE_DNN_VERIFIER, 1.0f);
}
```

- The first run parses the ONNX and builds the engine (minutes on Jetson);
  it is cached (`onnxPath + ".engine"` by default) for instant reload.
  Engine files are GPU/TensorRT-version specific — delete after JetPack
  upgrades.
- Model requirements: static input `1 x C x S x S` (C = 1 or 3), a single
  output tensor of any length. An ImageNet-pretrained MobileNetV2 backbone
  (global-pooled features, ~1280-dim) exported to ONNX works well as a
  starting point; a re-ID-trained embedding on your own footage is better.
- Budget: ~1–3 ms per verification on Xavier NX (FP16). At the default
  6-frame cadence this is negligible against the tracker's frame cost.

### Runtime parameters

| Param | Default | Effect |
|---|---|---|
| `ENABLE_DNN_VERIFIER` | 0 | Master switch (also needs an extractor set) |
| `DNN_VERIFY_INTERVAL` | 6 | Frames between verifications. Lower = faster drift detection, more inference load |
| `DNN_VETO_THRESHOLD` | 0.45 | Similarity below this = mismatch. Raise for stricter identity enforcement |
| `DNN_ACCEPT_THRESHOLD` | 0.60 | Required for LOST re-capture and bank adds. Raise to reject more look-alikes (risk: slower legit re-capture) |

Telemetry: `params.dnnSimilarity` reports the last measured similarity —
log it alongside `detectionProbability` when tuning thresholds. The
compile-time constants (veto streak = 3, bank capacity = 12, add spacing =
30 frames, crop context = 1.25×) live at the top of `src/CvTracker.cpp`.

### Interaction with the EKF

The verifier and the EKF are independent and compose cleanly: the EKF
constrains *where* the object can plausibly be (motion), the verifier
constrains *what* it must look like (appearance). Enabled together, a
re-capture candidate must pass the correlation threshold, the Mahalanobis
gate, and embedding verification.

---

## 13c. Vision-Only Robustness Aids

Three opt-in features (all default OFF, all runtime-toggleable) designed
for unstabilized cameras — e.g. a boresight-mounted camera on a fixed-wing
aircraft — with **no external sensors and no lens calibration** required.
Existing API calls are unchanged; each feature is a flag.

### Global Motion Compensation — `ENABLE_GMC`

Whole-frame **phase correlation** between consecutive frames measures the
camera-induced scene shift each frame (self-contained, ~1–2 ms on a
256×256 grid using the built-in FFT; lens/FOV agnostic because it measures
pixels, not degrees; blur-tolerant because both frames blur equally). The
measured shift is added to every stored screen position (tracking
position, sticky search override, EKF state) **before** any mode logic
runs — a violent camera jolt therefore never appears as object motion.

- Applied in TRACKING, LOST and INERTIAL. STATIC keeps literal
  screen-freeze semantics; FREE has nothing to shift.
- The measurement is rejected (shift = 0 used) on: the first frame,
  non-contiguous `frameId`s, frame-size changes, weak correlation peaks
  (featureless scenes), or implausibly large shifts.
- Telemetry: `params.gmcShiftX/Y` report the applied shift each frame —
  log them to validate against known camera motion.
- Limitation: translation only; aircraft **roll** (image rotation) is not
  measured and is absorbed by the correlation filter's rotation
  tolerance.

```cpp
tracker.setParam(VTrackerParam::ENABLE_GMC, 1.0f);
```

### Multi-Peak Identity Arbitration — `ENABLE_DNN_ARBITRATION`

When the correlation response surface has several competing peaks
(clutter, a distractor entering the window), the brightest is not
necessarily the object. With arbitration on, up to 2 secondary peaks
(≥ 50% of the primary) are embedded and compared against the template
bank; a non-primary candidate wins only **decisively** (similarity ≥
`DNN_ACCEPT_THRESHOLD` and ≥ primary + 0.05). Lock-on becomes "the
brightest match *that is my object*". Requires the DNN verifier to be
active; costs 2–3 extra inferences only on ambiguous frames.

### Embedding Re-Acquisition — `ENABLE_DNN_REACQUISITION`

LOST-mode re-detection normally sees only ~2.5× the rect around the
search center (the feature-window "sight bubble"). With re-acquisition
on, a cycling grid of crops spanning the **whole search window** is
embedded (4 crops/frame; full sweep of a 256-window/72-rect geometry in
~9 frames) and checked against the bank. Because grid crops are generally
off-center on the object, the sweep uses a **relaxed cue threshold**
(half the accept bar) to trigger one confirming correlation detect; that
detect centers the object, and the normal re-capture gates (probability
+ embedding at the **full** accept threshold, now on a centered crop)
make the final decision — *identity cues, correlation centers, identity
confirms.* Requires the DNN verifier; costs ~4 inferences per LOST frame
only. A false cue costs one extra detect and cannot cause a wrong lock.

### Also closed on this branch

Two Known Limitations of the `verifier-stable` build are fixed here
(always on, not flag-gated): the **EKF re-capture deadlock** (the filter
is now re-seeded at the re-detection instead of gated against it) and the
**verifier crop scale** defect (embedding crops now follow
`baseRect × scale`, so similarity no longer decays with zoom/altitude
change).

---

## 14. Platform Notes (Jetson)

### JetPack / CUDA Setup

```bash
# Verify CUDA is available
nvcc --version
ls /usr/local/cuda/

# Install JetPack (if not already):
sudo apt update && sudo apt install nvidia-jetpack

# Check GPU architecture (used by CMake automatically)
cat /proc/device-tree/compatible | tr '\0' '\n'
```

### CUDA Build

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVTRACKER_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=72   # Xavier=72, Orin=87, TX2=62, Nano/TX1=53

cmake --build build -j$(nproc)
```

### Supported GPU Architectures

| Jetson Model | SM Architecture | CMAKE_CUDA_ARCHITECTURES |
|---|---|---|
| Nano / TX1 | SM 5.3 | 53 |
| TX2 | SM 6.2 | 62 |
| Xavier NX / AGX Xavier | SM 7.2 | 72 |
| Orin NX / AGX Orin | SM 8.7 | 87 |

### Performance Mode Setup

```bash
# Check current mode
sudo nvpmodel -q

# Set to max performance (15W on Xavier NX)
sudo nvpmodel -m 0

# Lock all clocks at maximum
sudo jetson_clocks

# Verify
tegrastats
```

### INTEGRATION_XAVIER_NX.md

The repository includes `INTEGRATION_XAVIER_NX.md` with step-by-step instructions specific to the Jetson Xavier NX platform, including kernel configuration, camera driver notes, and system service setup.

---

## Quick Reference Card

| Task | Code |
|---|---|
| Create tracker | `CvTracker tracker;` |
| Initialize | `tracker.initVTracker(params);` |
| Process frame | `tracker.processFrame(frame);` |
| Get results | `tracker.getParams(params);` |
| Capture at pixel | `tracker.executeCommand(CAPTURE, x, y, -1);` |
| Capture at % | `tracker.executeCommand(CAPTURE_PERCENTS, xPct, yPct);` |
| Stop-frame capture | `tracker.executeCommand(CAPTURE, x, y, savedFrameId);` |
| Release object | `tracker.executeCommand(RESET);` |
| Freeze position | `tracker.executeCommand(SET_STATIC_MODE);` |
| Coast by velocity | `tracker.executeCommand(SET_INERTIAL_MODE);` |
| Force re-search | `tracker.executeCommand(SET_LOST_MODE);` |
| Change rect size | `tracker.executeCommand(CHANGE_RECT_SIZE, dW, dH);` |
| Set loss threshold | `tracker.setParam(CUSTOM_1, 0.05f);` |
| Set learning rate | `tracker.setParam(CUSTOM_3, 0.05f);` |
| Get probability | `params.detectionProbability` |
| Get object pos | `params.objectX`, `params.objectY` |
| Get mode | `params.mode` (0=FREE 1=TRACKING 2=LOST 3=INERTIAL 4=STATIC) |
| Get processing time | `params.processingTimeMks` (µs) |
| Encode command (net) | `VTracker::encodeCommand(buf, size, CMD, a1, a2, a3);` |
| Decode+execute (net) | `tracker.decodeAndExecuteCommand(buf, size);` |
| Get debug image | `tracker.getImage(0/1/2, img);` |
| Library version | `CvTracker::getVersion();` |
| Enable EKF filtering | `tracker.setParam(ENABLE_EKF, 1.0f);` |
| Set DNN extractor | `tracker.setFeatureExtractor(fe);` |
| Enable DNN verifier | `tracker.setParam(ENABLE_DNN_VERIFIER, 1.0f);` |
| Get verifier similarity | `params.dnnSimilarity` |
| TensorRT extractor | `createTrtFeatureExtractor(cfg);` (nullptr if unavailable) |
| Enable motion compensation | `tracker.setParam(ENABLE_GMC, 1.0f);` |
| Get measured scene shift | `params.gmcShiftX`, `params.gmcShiftY` |
| Enable peak arbitration | `tracker.setParam(ENABLE_DNN_ARBITRATION, 1.0f);` |
| Enable embedding re-acquisition | `tracker.setParam(ENABLE_DNN_REACQUISITION, 1.0f);` |
