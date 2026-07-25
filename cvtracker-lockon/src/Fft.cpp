#include "Fft.h"
#include <cmath>
#include <future>
#include <thread>

namespace cr {
namespace vtracker {

CpuFft::Plan& CpuFft::plan(int n)
{
    auto it = m_plans.find(n);
    if (it != m_plans.end())
        return it->second;

    Plan p;
    p.bitrev.resize(n);
    int log2n = 0;
    while ((1 << log2n) < n)
        ++log2n;
    for (int i = 0; i < n; ++i)
    {
        int r = 0;
        for (int b = 0; b < log2n; ++b)
            if (i & (1 << b))
                r |= 1 << (log2n - 1 - b);
        p.bitrev[i] = r;
    }
    p.twiddle.resize(n / 2);
    p.twiddleInv.resize(n / 2);
    for (int i = 0; i < n / 2; ++i)
    {
        const double a = -2.0 * M_PI * i / n;
        p.twiddle[i] = std::complex<float>((float)std::cos(a), (float)std::sin(a));
        p.twiddleInv[i] = std::conj(p.twiddle[i]);
    }
    return m_plans.emplace(n, std::move(p)).first->second;
}

void CpuFft::fft1d(std::complex<float>* a, int n, bool inverse, const Plan& p)
{
    for (int i = 0; i < n; ++i)
    {
        const int j = p.bitrev[i];
        if (j > i)
            std::swap(a[i], a[j]);
    }
    const auto& tw = inverse ? p.twiddleInv : p.twiddle;
    for (int len = 2; len <= n; len <<= 1)
    {
        const int half = len >> 1;
        const int step = n / len;
        for (int i = 0; i < n; i += len)
        {
            for (int k = 0; k < half; ++k)
            {
                const std::complex<float> w = tw[k * step];
                const std::complex<float> u = a[i + k];
                const std::complex<float> v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
            }
        }
    }
}

void CpuFft::fftRows(std::complex<float>* data, int w, int h, bool inverse)
{
    Plan& p = plan(w);
    const unsigned hw = std::thread::hardware_concurrency();
    if (m_multiThread && hw > 1 && h >= 8)
    {
        const int nThreads = (int)std::min<unsigned>(hw, 8);
        std::vector<std::future<void>> jobs;
        const int rowsPerJob = (h + nThreads - 1) / nThreads;
        for (int t = 0; t < nThreads; ++t)
        {
            const int r0 = t * rowsPerJob;
            const int r1 = std::min(h, r0 + rowsPerJob);
            if (r0 >= r1)
                break;
            jobs.push_back(std::async(std::launch::async, [=, &p]() {
                for (int r = r0; r < r1; ++r)
                    fft1d(data + (size_t)r * w, w, inverse, p);
            }));
        }
        for (auto& j : jobs)
            j.wait();
    }
    else
    {
        for (int r = 0; r < h; ++r)
            fft1d(data + (size_t)r * w, w, inverse, p);
    }
}

void CpuFft::fft2d(std::complex<float>* data, int w, int h, bool inverse)
{
    // Rows.
    fftRows(data, w, h, inverse);

    // Columns: gather column to contiguous buffer, transform, scatter back.
    Plan& p = plan(h);
    std::vector<std::complex<float>> col((size_t)h);
    for (int c = 0; c < w; ++c)
    {
        for (int r = 0; r < h; ++r)
            col[r] = data[(size_t)r * w + c];
        fft1d(col.data(), h, inverse, p);
        for (int r = 0; r < h; ++r)
            data[(size_t)r * w + c] = col[r];
    }

    if (inverse)
    {
        const float norm = 1.0f / ((float)w * (float)h);
        const size_t n = (size_t)w * h;
        for (size_t i = 0; i < n; ++i)
            data[i] *= norm;
    }
}

#ifndef CVTRACKER_WITH_CUDA
std::unique_ptr<FftBackend> createFftBackend()
{
    return std::make_unique<CpuFft>();
}
#endif

} // namespace vtracker
} // namespace cr
