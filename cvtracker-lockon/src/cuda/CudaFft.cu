#include "../Fft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <map>
#include <utility>

namespace cr {
namespace vtracker {

/// cuFFT backend. Used automatically when the library is built with
/// CVTRACKER_WITH_CUDA and a CUDA device is present (any Jetson under
/// JetPack). Falls back to CpuFft otherwise.
class CudaFft : public FftBackend
{
public:
    ~CudaFft() override
    {
        for (auto& kv : m_plans)
            cufftDestroy(kv.second);
        if (m_dev != nullptr)
            cudaFree(m_dev);
    }

    static bool available()
    {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }

    void fft2d(std::complex<float>* data, int w, int h, bool inverse) override
    {
        const size_t bytes = sizeof(cufftComplex) * (size_t)w * h;
        if (bytes > m_devBytes)
        {
            if (m_dev != nullptr)
                cudaFree(m_dev);
            if (cudaMalloc(&m_dev, bytes) != cudaSuccess)
            {
                m_dev = nullptr;
                m_devBytes = 0;
                m_cpuFallback.fft2d(data, w, h, inverse);
                return;
            }
            m_devBytes = bytes;
        }

        cufftHandle plan = getPlan(w, h);
        if (plan == 0)
        {
            m_cpuFallback.fft2d(data, w, h, inverse);
            return;
        }

        cudaMemcpy(m_dev, data, bytes, cudaMemcpyHostToDevice);
        cufftExecC2C(plan, (cufftComplex*)m_dev, (cufftComplex*)m_dev,
                     inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
        cudaMemcpy(data, m_dev, bytes, cudaMemcpyDeviceToHost);

        if (inverse)
        {
            const float norm = 1.0f / ((float)w * (float)h);
            const size_t n = (size_t)w * h;
            for (size_t i = 0; i < n; ++i)
                data[i] *= norm;
        }
    }

private:
    cufftHandle getPlan(int w, int h)
    {
        const auto key = std::make_pair(w, h);
        auto it = m_plans.find(key);
        if (it != m_plans.end())
            return it->second;
        cufftHandle plan = 0;
        // cuFFT plan dimensions: (rows, cols) = (h, w).
        if (cufftPlan2d(&plan, h, w, CUFFT_C2C) != CUFFT_SUCCESS)
            return 0;
        m_plans[key] = plan;
        return plan;
    }

    std::map<std::pair<int, int>, cufftHandle> m_plans;
    void* m_dev{nullptr};
    size_t m_devBytes{0};
    CpuFft m_cpuFallback;
};

std::unique_ptr<FftBackend> createFftBackend()
{
    if (CudaFft::available())
    {
        std::fprintf(stderr, "[cvtracker] FFT backend: CUDA (cuFFT)\n");
        return std::make_unique<CudaFft>();
    }
    std::fprintf(stderr,
                 "[cvtracker] FFT backend: CPU (no CUDA device found)\n");
    return std::make_unique<CpuFft>();
}

} // namespace vtracker
} // namespace cr
