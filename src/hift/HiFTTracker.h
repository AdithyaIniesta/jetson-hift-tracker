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

#include "VTracker.h"
#include "hift/TrtHiFT.h"

namespace cr {
namespace hift {

struct HiFTTrackerConfig
{
    TrtHiFTConfig trt;
    // A track is considered lost when the fused cls score drops below this.
    float lossThreshold = 0.25f;
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

    bool ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace hift
}  // namespace cr
