#pragma once
#include <complex>
#include <cstdint>
#include <vector>
#include "Fft.h"

namespace cr {
namespace vtracker {

/// Global Motion Compensation (GMC) measurement: whole-frame phase
/// correlation between consecutive frames.
///
/// Camera motion (an unstabilized airborne platform, a violent jolt) moves
/// the ENTIRE image coherently, while object motion moves only the object.
/// Phase correlation of two consecutive downsampled luma frames finds that
/// dominant global shift directly in pixels - no lens calibration, no IMU,
/// no external sensors. The tracker adds the measured shift to its search
/// position each frame, so the correlation matcher only ever sees
/// object-relative motion.
///
/// Properties:
/// - FOV / focal-length agnostic (measures pixels, not degrees);
/// - blur tolerant (both frames blur equally during a jolt; their mutual
///   shift remains observable);
/// - the tracked object cannot bias it (it occupies a tiny fraction of the
///   frame; the scene-wide motion dominates);
/// - translation only. Image rotation (aircraft roll) is NOT measured -
///   residual roll is absorbed by the correlation filter's rotation
///   tolerance and adaptation.
///
/// Cost: one box-downsample + two 2D FFTs (forward of the current grid,
/// inverse of the cross-power spectrum) per frame on a GRID x GRID buffer.
class GlobalMotion
{
public:
    /// The FFT backend is borrowed (not owned); must outlive this object.
    void init(FftBackend* fft);

    /// Forget the previous frame (call on init / frame-size change).
    void reset();

    /// Feed the next frame's luma and measure the global shift from the
    /// PREVIOUSLY fed frame to this one, in full-resolution frame pixels.
    ///
    /// Returns true and fills (dx, dy, confidence) when a reliable
    /// measurement exists. Returns false (shift unusable, state still
    /// updated) when: this is the first frame, frameId is not contiguous
    /// with the previous call (stop-frame catch-up restart, dropped
    /// frames), the frame size changed, the correlation peak is too weak
    /// (featureless scene), or the shift is implausibly large.
    ///
    /// Sign convention: dx/dy is the displacement OF THE SCENE CONTENT on
    /// screen, i.e. cur(x) == prev(x - dx). Add it to any stored screen
    /// position to keep that position attached to the scene.
    bool measure(const uint8_t* luma, int w, int h, int frameId,
                 float& dx, float& dy, float& confidence);

private:
    using cf = std::complex<float>;

    static constexpr int GRID = 256;            // analysis grid (power of 2)
    static constexpr float MIN_CONFIDENCE = 0.03f;
    static constexpr float MAX_SHIFT_FRAC = 0.35f; // of GRID, per axis

    /// Box-downsample luma to GRID x GRID, zero-mean, Hann-window into out.
    void downsample(const uint8_t* luma, int w, int h, std::vector<cf>& out);

    FftBackend* m_fft{nullptr};
    std::vector<cf> m_prev;   // FFT of previous downsampled frame
    std::vector<cf> m_cur;    // FFT of current downsampled frame
    std::vector<cf> m_buf;    // cross-power spectrum / response
    std::vector<float> m_hann1d;
    bool m_hasPrev{false};
    int m_prevId{-1000000};
    int m_prevW{0};
    int m_prevH{0};
};

} // namespace vtracker
} // namespace cr
