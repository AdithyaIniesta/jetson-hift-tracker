// HiFTTracker — a VTracker implementation backed by the HiFT Siamese tracker.
//
// Drop-in replacement for CvTracker: same VTracker interface (CAPTURE/RESET/mode
// machine, processFrame per frame, getParams for the GUI/control/telemetry), but
// the object search + box regression come from HiFT (via TrtHiFT) instead of the
// correlation filter + DINOv2 verifier.
//
// Port of hift/pysot/tracker/hift_tracker.py: get_subwindow crop, template on
// CAPTURE, generate_anchor box decode, scale/ratio penalty + cosine window +
// argmax, lr size smoothing. HiFT self-verifies (cls score), so there is no
// separate appearance verifier in the per-frame path.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cvtracker/FeatureExtractor.h>
#include <cvtracker/VTracker.h>

#include "hift/TrtHiFT.h"

namespace cr {
namespace hift {

struct HiFTTrackerConfig
{
    TrtHiFTConfig trt;
    // A track is considered lost when the PSR-gated confidence drops below this.
    float lossThreshold = 0.25f;
    // PSR (peak sharpness) at which confidence saturates. Higher = stricter loss
    // detection. Tunable at runtime via TRACKER_HIFT_PSR.
    float psrRef = 4.0f;
    // Cosine-window influence: higher biases the peak pick toward the last
    // position (damps flip-flop between competing peaks). Tunable via
    // TRACKER_HIFT_WINDOW.
    float windowInf = 0.42f;

    // ── DINOv2 appearance verifier (hybrid) ──────────────────────────────
    // HiFT localizes; the verifier guards identity — rejects distractor jumps
    // and gates re-acquisition. Active only when a FeatureExtractor is set.
    // Calibrated on-device (grayscale DINOv2): true target ~0.90-0.99, so the
    // old 0.45 never fired. 0.75 rejects distractors; true-target dips below it
    // are rare and only cost a harmless 1-frame rollback.
    float simThreshold = 0.75f;  // cosine sim below this = appearance mismatch
    int   verifyEvery  = 3;      // verify every N tracking frames
    int   maxVetoStreak = 3;     // consecutive vetoes -> LOST
    // HiFT template refresh, gated on a HiFT-confident AND verifier-confirmed
    // frame (so it can never adapt onto a distractor).
    bool  templateRefresh = true;
    int   refreshEvery    = 45;  // frames between refreshes
    float refreshMinConf  = 0.5f;

    // ── Active re-detection scan (LOST mode) ─────────────────────────────
    // When LOST, probe candidate locations across the frame: run HiFT there and
    // confirm with DINOv2. This is what re-finds the target after occlusion —
    // the verifier searches, not just vetoes. Needs an attached extractor.
    bool  redetect = true;
    int   probesPerFrame = 2;         // HiFT+DINOv2 probes per LOST frame
    float reacquireThreshold = 0.88f; // strong match to re-lock (true target ~0.9+)
    int   redetectMaxFrames = 300;    // stay in LOST (scanning) this long
};

class HiFTTracker : public cr::vtracker::VTracker
{
public:
    explicit HiFTTracker(const HiFTTrackerConfig& cfg = {});
    ~HiFTTracker() override;

    static std::string getVersion();

    bool initVTracker(cr::vtracker::VTrackerParams& params) override;
    bool setParam(cr::vtracker::VTrackerParam id, float value) override;
    float getParam(cr::vtracker::VTrackerParam id) override;
    void getParams(cr::vtracker::VTrackerParams& params) override;
    bool executeCommand(cr::vtracker::VTrackerCommand id, float arg1 = 0.0f,
                        float arg2 = 0.0f, float arg3 = 0.0f) override;
    bool processFrame(cr::video::Frame& frame) override;
    void getImage(int type, cr::video::Frame& image) override;
    bool decodeAndExecuteCommand(uint8_t* data, int size) override;

    // Cross-camera handoff: export/import the HiFT template crop so the peer
    // camera can seed its own template branch with the same appearance. Buffer
    // is 3*127*127 floats (CHW, BGR, 0..255) + a small header. Stage C wires
    // these into the handoff path.
    bool exportTemplate(std::vector<unsigned char>& out) const;
    bool importTemplate(const std::vector<unsigned char>& in);

    // ── CvTracker API-compat shims (so HiFTTracker is a drop-in for the
    //    pipeline's concrete-type call sites) ─────────────────────────────
    // Attach the DINOv2 appearance verifier. When set, HiFTTracker seeds a
    // reference embedding on CAPTURE and vetoes distractor jumps (see .cpp).
    // Pass nullptr to disable. Clears the reference bank.
    void setFeatureExtractor(
        std::shared_ptr<cr::vtracker::FeatureExtractor> extractor);
    // Handoff bank transfer maps to the HiFT template crop.
    bool exportTemplateBank(std::vector<unsigned char>& out) const
    {
        return exportTemplate(out);
    }
    bool importTemplateBank(const std::vector<unsigned char>& in)
    {
        return importTemplate(in);
    }
    // Epipolar peak biasing is a correlation-filter concept; HiFT localizes
    // via its own regression. Accepted as a no-op for now (Stage C+ may add
    // an epipolar prior on the score map).
    void setEpipolarConstraint(float /*a*/, float /*b*/, float /*c*/,
                               float /*halfWidthPx*/, int /*ttlFrames*/) {}

    bool ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace hift
}  // namespace cr
