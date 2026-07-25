#pragma once
#include <cmath>
#include <vector>

namespace cr {
namespace vtracker {

/// Interface for appearance embedding extractors used by the optional DNN
/// verifier layer of CvTracker (see CvTracker::setFeatureExtractor() and
/// VTrackerParam::ENABLE_DNN_VERIFIER).
///
/// The tracker crops the tracked object region from the current frame,
/// resamples it to inputSize() x inputSize() grayscale, and calls extract().
/// The returned embedding is stored in a template bank and compared with
/// cosine similarity to verify that the correlation tracker is still
/// following the captured object (drift / distractor rejection) and to
/// confirm LOST-mode re-captures.
///
/// Implementations must be usable from the tracker's processing thread;
/// extract() is called under the tracker's internal lock, so it should be
/// fast (a few milliseconds). Typical implementations: a TensorRT model on
/// Jetson (see TrtFeatureExtractor.h) or the built-in TinyPatchExtractor.
class FeatureExtractor
{
public:
    virtual ~FeatureExtractor() = default;

    /// Square input patch side length, pixels. The tracker resamples the
    /// object crop to this size before calling extract().
    virtual int inputSize() const = 0;

    /// Compute an embedding from a grayscale patch (inputSize()*inputSize()
    /// floats, row-major, pixel range [0..255]). The embedding is
    /// L2-normalized by the caller; any dimensionality is accepted as long
    /// as it is consistent between calls. Return false on failure (the
    /// verifier then skips this frame - tracking is unaffected).
    virtual bool extract(const float* patch, std::vector<float>& embedding) = 0;
};

/// Dependency-free fallback extractor: the embedding is simply the
/// zero-mean / unit-norm downsampled patch itself (cosine similarity then
/// equals normalized cross-correlation of the patches). Much weaker than a
/// trained DNN embedding, but exercises the full verifier pipeline on any
/// platform and provides basic appearance verification without an
/// inference runtime. Also used by the test suite.
class TinyPatchExtractor : public FeatureExtractor
{
public:
    explicit TinyPatchExtractor(int size = 32) : m_size(size < 8 ? 8 : size) {}

    int inputSize() const override { return m_size; }

    bool extract(const float* patch, std::vector<float>& embedding) override
    {
        const int n = m_size * m_size;
        embedding.assign(patch, patch + n);
        double mean = 0.0;
        for (int i = 0; i < n; ++i)
            mean += embedding[i];
        mean /= (double)n;
        double var = 0.0;
        for (int i = 0; i < n; ++i)
        {
            embedding[i] -= (float)mean;
            var += (double)embedding[i] * embedding[i];
        }
        const float inv = 1.0f / ((float)std::sqrt(var) + 1e-6f);
        for (int i = 0; i < n; ++i)
            embedding[i] *= inv;
        return true;
    }

private:
    int m_size;
};

} // namespace vtracker
} // namespace cr
