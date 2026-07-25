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
    /// byte buffer. Used for cross-tracker template transfer (e.g. dual-
    /// camera handoff: seed the peer tracker with the appearance model
    /// learned by the primary rather than starting from a single frame).
    ///
    /// Returns true on success; `out` receives the serialised bank.
    /// Returns false if the bank is empty or the DNN verifier has never
    /// captured a template. Buffer layout is defined by TemplateBank
    /// serialize() — see that method's docstring for the on-wire format
    /// and versioning contract.
    ///
    /// Thread-safety: caller must ensure no processFrame / executeCommand
    /// call runs concurrently on this instance.
    bool exportTemplateBank(std::vector<unsigned char>& out) const;

    /// Replace the DNN verifier template bank from a buffer previously
    /// produced by exportTemplateBank() (on this build, this instance,
    /// or another). Silently accepts buffers with a matching magic /
    /// version; rejects any that fail integrity checks so a malformed
    /// buffer never leaves the bank half-loaded.
    ///
    /// Returns true on successful import. The imported bank's
    /// `lastAddFrame` is preserved from the buffer, so subsequent
    /// maybeAdd() spacing behaves as if the transferred history had
    /// been learned locally.
    ///
    /// Thread-safety: as above — call under the caller's tracker mutex.
    bool importTemplateBank(const std::vector<unsigned char>& in);

    /// Constrain correlation peak selection to a band around an image-
    /// space line. Intended for the peer tracker right after a stereo
    /// cross-view seed: the target must lie on the epipolar line
    /// corresponding to the source rect, so peaks off that line are
    /// almost certainly distractors and can be rejected in favour of
    /// on-line secondary peaks.
    ///
    /// Line is ax + by + c = 0 in the tracker's image coordinate space
    /// (same units as VTrackerParams::rectX / rectY — post-downsample
    /// if the app is running downsampled). halfWidthPx is the maximum
    /// perpendicular distance a peak may lie from the line and still be
    /// preferred. ttlFrames is the number of subsequent processFrame
    /// calls during which the constraint remains active; it counts down
    /// automatically and expires (ttlFrames <= 0) into unconstrained
    /// operation. Call again to refresh.
    ///
    /// Passing ttlFrames = 0 (or a degenerate line with a==b==0)
    /// deactivates the constraint immediately.
    ///
    /// Thread-safety: acquires the tracker mutex.
    void setEpipolarConstraint(float a, float b, float c,
                               float halfWidthPx, int ttlFrames);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace vtracker
} // namespace cr
