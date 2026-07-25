#pragma once
#include <complex>
#include <cstdint>
#include <vector>
#include "Fft.h"

namespace cr {
namespace vtracker {

/// Adaptive correlation filter core (MOSSE-style, FFT based).
///
/// Robustness measures:
/// - log + zero-mean/unit-norm preprocessing -> invariance to global
///   illumination changes (gain and offset);
/// - capture-time training on rotated / scaled perturbations of the
///   reference -> tolerance to object orientation changes;
/// - adaptive pattern update weighted by detection confidence -> the
///   pattern follows gradual appearance changes but is not corrupted
///   during occlusions (update is suppressed at low confidence).
class CorrelationCore
{
public:
    struct Result
    {
        /// Peak displacement from window center, pixels (subpixel).
        float dx{0.0f};
        float dy{0.0f};
        /// Peak value of correlation surface.
        float peak{0.0f};
        /// Peak-to-sidelobe ratio.
        float psr{0.0f};
        /// Detection probability [0..1].
        float prob{0.0f};
    };

    /// Initialize for given window size (power of two) and FFT backend.
    void init(int w, int h, FftBackend* fft);

    /// Set object (tracking rectangle) size in window pixels. Defines the
    /// width of the gaussian response label.
    void setObjectSize(float rectW, float rectH);

    /// Capture object: train filter on window (raw luma, w*h floats, object
    /// centered) and on a set of rotated/scaled perturbations.
    void capture(const std::vector<float>& window);

    /// Correlate window with the filter. Does not modify the filter.
    Result detect(const std::vector<float>& window);

    /// Update filter and reference image with window (object centered).
    void update(const std::vector<float>& window, float rate);

    bool initialized() const { return m_initialized; }
    int width() const { return m_w; }
    int height() const { return m_h; }

    /// Running average of aligned raw windows (reference image of object).
    const std::vector<float>& referenceImage() const { return m_ref; }
    /// Last correlation surface (real part, normalized response).
    const std::vector<float>& responseSurface() const { return m_resp; }

    /// Estimate object mask from reference image: pixels that differ from
    /// background (window border statistics). mask: w*h bytes (0/255).
    void objectMask(std::vector<uint8_t>& mask) const;

private:
    using cf = std::complex<float>;

    void preprocess(const float* in, std::vector<cf>& out) const;
    void makeLabel(float sigmaX, float sigmaY);
    void trainAccumulate(const std::vector<float>& window, float weight);
    static void affineSample(const std::vector<float>& src, int w, int h,
                             float angleDeg, float scale,
                             std::vector<float>& dst);

    int m_w{0};
    int m_h{0};
    FftBackend* m_fft{nullptr};
    bool m_initialized{false};
    float m_lambda{0.01f};

    std::vector<float> m_hann;   // cosine window
    std::vector<cf> m_G;         // FFT of gaussian label
    std::vector<cf> m_A;         // numerator accumulator
    std::vector<cf> m_B;         // denominator accumulator
    std::vector<cf> m_buf;       // work buffer
    std::vector<float> m_ref;    // reference image (running average)
    std::vector<float> m_resp;   // last response surface
};

} // namespace vtracker
} // namespace cr
