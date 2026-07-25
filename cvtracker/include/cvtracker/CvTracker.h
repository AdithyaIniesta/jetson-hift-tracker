#pragma once
#include <memory>
#include <string>
#include <vector>
#include "FeatureExtractor.h"
#include "VTracker.h"

namespace cr {
namespace vtracker {

/// Correlation video tracker.
///
/// Self-contained C++ implementation of the VTracker interface:
/// - correlation surface object search (MOSSE-style adaptive filter, FFT);
/// - FREE / TRACKING / LOST / INERTIAL / STATIC operating modes with
///   automatic loss detection and automatic object re-capture;
/// - frame buffer for STOP-FRAME function and communication channel time
///   delay compensation (capture on a past frame by frame ID, automatic
///   catch-up processing);
/// - invariant to global lighting changes (log / normalized features);
/// - tolerant to object orientation changes (perturbation training and
///   confidence weighted adaptive pattern update);
/// - occlusion handling (pattern update suppressed at low confidence,
///   trajectory prediction in LOST mode, automatic re-capture);
/// - hardware and camera agnostic: plain C++17, no external dependencies,
///   frames are passed in by the user in any supported RAW pixel format
///   (GRAY, YUV24, YUYV, UYVY, NV12, NV21, YU12, YV12).
class CvTracker : public VTracker
{
public:
    CvTracker();
    ~CvTracker() override;

    /// Get library version in "Major.Minor.Patch" format.
    static std::string getVersion();

    bool initVTracker(VTrackerParams& params) override;
    bool setParam(VTrackerParam id, float value) override;
    float getParam(VTrackerParam id) override;
    void getParams(VTrackerParams& params) override;
    bool executeCommand(VTrackerCommand id,
                        float arg1 = 0.0f,
                        float arg2 = 0.0f,
                        float arg3 = 0.0f) override;
    bool processFrame(cr::video::Frame& frame) override;
    void getImage(int type, cr::video::Frame& image) override;

    /// Set (or replace) the appearance embedding extractor used by the
    /// optional DNN verifier. Pass nullptr to remove. Thread-safe; clears
    /// the template bank. The verifier is active only when an extractor is
    /// set AND VTrackerParam::ENABLE_DNN_VERIFIER is 1. See
    /// FeatureExtractor.h (interface, built-in TinyPatchExtractor) and
    /// TrtFeatureExtractor.h (TensorRT models on Jetson).
    void setFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor);

    /// Export the DNN verifier template bank as an opaque, versioned
    /// byte buffer. Used for cross-tracker template transfer at dual-
    /// camera handoff. Format identical to the lockon engine's export
    /// (magic 'CBK1' + version 1) so a bank from either engine can be
    /// consumed by the other. Returns false if the bank is empty.
    /// Thread-safe: acquires the tracker mutex.
    bool exportTemplateBank(std::vector<unsigned char>& out) const;

    /// Replace this tracker's DNN verifier bank from a buffer produced
    /// by exportTemplateBank(). Returns false on magic/version/size
    /// mismatch; the bank is left untouched on failure so a bad buffer
    /// never leaves it half-loaded. Thread-safe.
    bool importTemplateBank(const std::vector<unsigned char>& in);

    /// Constrain the correlation peak selection to a band around an
    /// image-space line ax + by + c = 0 for the next ttlFrames calls
    /// to processFrame. Used at cross-view handoff to bias the peer
    /// tracker toward peaks that lie on the epipolar line derived from
    /// the source rect via the fundamental matrix. TTL counts down
    /// automatically. Passing ttlFrames <= 0 or a degenerate line
    /// (a == b == 0) disables the constraint immediately.
    ///
    /// halfWidthPx is the maximum perpendicular distance a peak may
    /// lie from the line and still be preferred over an off-line one.
    /// Thread-safe.
    void setEpipolarConstraint(float a, float b, float c,
                               float halfWidthPx, int ttlFrames);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace vtracker
} // namespace cr
