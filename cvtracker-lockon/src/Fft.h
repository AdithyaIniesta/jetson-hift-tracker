#pragma once
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace cr {
namespace vtracker {

/// Abstract 2D FFT backend (in-place, complex float, power-of-two sizes).
class FftBackend
{
public:
    virtual ~FftBackend() = default;
    /// In-place 2D FFT. data layout: row-major, h rows of w complex values.
    /// Inverse transform is normalized by 1/(w*h).
    virtual void fft2d(std::complex<float>* data, int w, int h,
                       bool inverse) = 0;
    /// Enable/disable multithreading (if supported).
    virtual void setMultiThread(bool enable) { (void)enable; }
};

/// CPU radix-2 FFT backend (self-contained, no dependencies).
class CpuFft : public FftBackend
{
public:
    void fft2d(std::complex<float>* data, int w, int h, bool inverse) override;
    void setMultiThread(bool enable) override { m_multiThread = enable; }

private:
    struct Plan
    {
        std::vector<int> bitrev;
        std::vector<std::complex<float>> twiddle;    // forward
        std::vector<std::complex<float>> twiddleInv; // inverse
    };
    Plan& plan(int n);
    void fft1d(std::complex<float>* a, int n, bool inverse, const Plan& p);
    void fftRows(std::complex<float>* data, int w, int h, bool inverse);

    std::map<int, Plan> m_plans;
    bool m_multiThread{false};
};

/// Create the best available FFT backend (CUDA if compiled in and a device
/// is present, otherwise CPU).
std::unique_ptr<FftBackend> createFftBackend();

} // namespace vtracker
} // namespace cr
