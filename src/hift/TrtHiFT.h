// TrtHiFT — TensorRT wrapper for the HiFT two-branch Siamese tracker.
//
// Two engines (built/cached from ONNX by tools/export_hift.py):
//   template:  z[1,3,127,127]           -> zf0[1,384,10,10], zf1[1,384,8,8], zf2[1,256,6,6]
//   track:     x[1,3,287,287] + zf0..2  -> loc[1,4,11,11], cls1[1,2,11,11], cls2[1,1,11,11]
//
// The template branch runs once per CAPTURE; its three zf feature maps are kept on the GPU
// and bound directly as track-branch inputs, so per-frame tracking only pays the track
// branch + one H2D copy of the search crop. Measured ~6.8 ms / 147 fps (FP16, Orin NX).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cr {
namespace hift {

// ── HiFT constants (from hift/experiments/config.yaml) ───────────────────────
constexpr int   HIFT_EXEMPLAR   = 127;   // template crop size
constexpr int   HIFT_SEARCH     = 287;   // search crop size
constexpr int   HIFT_OUTPUT     = 11;    // loc/cls map is OUTPUT x OUTPUT
constexpr int   HIFT_ANCHOR_STRIDE = 16; // generate_anchor stride (cfg.ANCHOR.STRIDE)
constexpr float HIFT_CONTEXT    = 0.5f;  // context amount for crop sizing
constexpr float HIFT_PENALTY_K  = 0.08f;
constexpr float HIFT_WINDOW_INF = 0.42f;
constexpr float HIFT_LR         = 0.30f; // box size smoothing rate
constexpr float HIFT_W2         = 1.0f;  // cls1 weight
constexpr float HIFT_W3         = 1.0f;  // cls2 weight
constexpr int   HIFT_DECODE_SCALE = HIFT_SEARCH / 2;  // 143; loc tanh-decode scale
constexpr int   HIFT_ANCHOR_OFF   = (HIFT_EXEMPLAR - 1) / 2;  // 63

struct TrtHiFTConfig {
    std::string templateOnnx = "models/hift_template.onnx";
    std::string trackOnnx     = "models/hift_track.onnx";
    std::string templateEngine;   // default: templateOnnx + ".engine"
    std::string trackEngine;      // default: trackOnnx + ".engine"
    bool   fp16        = true;
    int    workspaceMb = 1024;
};

// Output maps of one track() call (raw, NCHW, row-major host copies).
struct HiFTTrackOut {
    std::vector<float> loc;   // 1 x 4 x 11 x 11
    std::vector<float> cls1;  // 1 x 2 x 11 x 11
    std::vector<float> cls2;  // 1 x 1 x 11 x 11
};

class TrtHiFT {
public:
    TrtHiFT();
    ~TrtHiFT();

    // Build/load both engines. Returns false on failure (caller falls back).
    bool initialise(const TrtHiFTConfig& cfg);
    bool ready() const;

    // Run the template branch on a 3x127x127 CHW float crop (ImageNet range is
    // NOT applied by HiFT — the crop is raw BGR mean-subtracted per get_subwindow;
    // pass exactly what the model expects). Keeps zf on the GPU for track().
    bool setTemplate(const float* z_chw);

    // Run the track branch on a 3x287x287 CHW float crop; fills `out`. Requires a
    // prior successful setTemplate().
    bool track(const float* x_chw, HiFTTrackOut& out);

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace hift
}  // namespace cr
