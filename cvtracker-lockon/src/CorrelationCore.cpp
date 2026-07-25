#include "CorrelationCore.h"
#include <algorithm>
#include <cmath>

namespace cr {
namespace vtracker {

void CorrelationCore::init(int w, int h, FftBackend* fft)
{
    m_w = w;
    m_h = h;
    m_fft = fft;
    m_initialized = false;

    const size_t n = (size_t)w * h;
    m_hann.assign(n, 0.0f);
    m_G.assign(n, cf(0.0f, 0.0f));
    m_A.assign(n, cf(0.0f, 0.0f));
    m_B.assign(n, cf(0.0f, 0.0f));
    m_buf.assign(n, cf(0.0f, 0.0f));
    m_ref.assign(n, 0.0f);
    m_resp.assign(n, 0.0f);
    setObjectSize(72.0f, 72.0f);
}

void CorrelationCore::setObjectSize(float rectW, float rectH)
{
    // Label sigma proportional to object size (KCF-style: ~ size / 10).
    const float sx = std::max(1.5f, rectW / 10.0f);
    const float sy = std::max(1.5f, rectH / 10.0f);
    makeLabel(sx, sy);

    // Feature window: cosine window restricted to ~2.5x the object size.
    // The filter stays object-centric (background contributes little), so
    // the response collapses when the object is occluded - this drives
    // loss detection - and the adaptive pattern does not learn background.
    const float exw = std::min((float)m_w, std::max(32.0f, 2.5f * rectW));
    const float exh = std::min((float)m_h, std::max(32.0f, 2.5f * rectH));
    const int cx = m_w / 2;
    const int cy = m_h / 2;
    for (int y = 0; y < m_h; ++y)
    {
        const float dy = (float)(y - cy) / (exh * 0.5f);
        const float wy = std::fabs(dy) >= 1.0f
                             ? 0.0f
                             : 0.5f + 0.5f * std::cos((float)M_PI * dy);
        for (int x = 0; x < m_w; ++x)
        {
            const float dx = (float)(x - cx) / (exw * 0.5f);
            const float wx = std::fabs(dx) >= 1.0f
                                 ? 0.0f
                                 : 0.5f + 0.5f * std::cos((float)M_PI * dx);
            m_hann[(size_t)y * m_w + x] = wx * wy;
        }
    }
}

void CorrelationCore::makeLabel(float sigmaX, float sigmaY)
{
    const int cx = m_w / 2;
    const int cy = m_h / 2;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            const float dx = (float)(x - cx) / sigmaX;
            const float dy = (float)(y - cy) / sigmaY;
            m_G[(size_t)y * m_w + x] =
                cf(std::exp(-0.5f * (dx * dx + dy * dy)), 0.0f);
        }
    }
    m_fft->fft2d(m_G.data(), m_w, m_h, false);
}

void CorrelationCore::preprocess(const float* in, std::vector<cf>& out) const
{
    const size_t n = (size_t)m_w * m_h;
    out.resize(n);

    // Log transform (compresses dynamic range, helps with lighting and
    // shadows), then zero-mean / unit-norm, then cosine window.
    double mean = 0.0;
    static thread_local std::vector<float> tmp;
    tmp.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        tmp[i] = std::log(1.0f + in[i]);
        mean += tmp[i];
    }
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        tmp[i] -= (float)mean;
        var += (double)tmp[i] * tmp[i];
    }
    const float norm = 1.0f / (float)(std::sqrt(var / (double)n) + 1e-5);
    for (size_t i = 0; i < n; ++i)
        out[i] = cf(tmp[i] * norm * m_hann[i], 0.0f);
}

void CorrelationCore::affineSample(const std::vector<float>& src, int w, int h,
                                   float angleDeg, float scale,
                                   std::vector<float>& dst)
{
    dst.resize((size_t)w * h);
    const float a = angleDeg * (float)M_PI / 180.0f;
    const float ca = std::cos(a) / scale;
    const float sa = std::sin(a) / scale;
    const float cx = w * 0.5f;
    const float cy = h * 0.5f;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const float rx = (float)x - cx;
            const float ry = (float)y - cy;
            float sxf = ca * rx - sa * ry + cx;
            float syf = sa * rx + ca * ry + cy;
            sxf = std::min(std::max(sxf, 0.0f), (float)w - 1.001f);
            syf = std::min(std::max(syf, 0.0f), (float)h - 1.001f);
            const int x0 = (int)sxf;
            const int y0 = (int)syf;
            const float fx = sxf - x0;
            const float fy = syf - y0;
            const float* p0 = &src[(size_t)y0 * w + x0];
            const float v = p0[0] * (1 - fx) * (1 - fy) + p0[1] * fx * (1 - fy) +
                            p0[w] * (1 - fx) * fy + p0[w + 1] * fx * fy;
            dst[(size_t)y * w + x] = v;
        }
    }
}

void CorrelationCore::trainAccumulate(const std::vector<float>& window,
                                      float weight)
{
    preprocess(window.data(), m_buf);
    m_fft->fft2d(m_buf.data(), m_w, m_h, false);
    const size_t n = (size_t)m_w * m_h;
    for (size_t i = 0; i < n; ++i)
    {
        const cf f = m_buf[i];
        const cf fc = std::conj(f);
        m_A[i] += weight * m_G[i] * fc;
        m_B[i] += weight * f * fc;
    }
}

void CorrelationCore::capture(const std::vector<float>& window)
{
    const size_t n = (size_t)m_w * m_h;
    std::fill(m_A.begin(), m_A.end(), cf(0.0f, 0.0f));
    std::fill(m_B.begin(), m_B.end(), cf(0.0f, 0.0f));

    // Base sample.
    trainAccumulate(window, 1.0f);

    // Rotated / scaled perturbations: tolerance to orientation and small
    // scale changes right after capture.
    static const float rotations[] = {-8.0f, -4.0f, 4.0f, 8.0f};
    static const float scales[] = {0.95f, 1.05f};
    std::vector<float> warped;
    for (float r : rotations)
    {
        affineSample(window, m_w, m_h, r, 1.0f, warped);
        trainAccumulate(warped, 0.5f);
    }
    for (float s : scales)
    {
        affineSample(window, m_w, m_h, 0.0f, s, warped);
        trainAccumulate(warped, 0.5f);
    }

    for (size_t i = 0; i < n; ++i)
        m_ref[i] = window[i];

    m_initialized = true;
}

CorrelationCore::Result CorrelationCore::detect(const std::vector<float>& window)
{
    Result res;
    if (!m_initialized)
        return res;

    preprocess(window.data(), m_buf);
    m_fft->fft2d(m_buf.data(), m_w, m_h, false);

    const size_t n = (size_t)m_w * m_h;
    for (size_t i = 0; i < n; ++i)
    {
        const cf hConj = m_A[i] / (m_B[i] + cf(m_lambda, 0.0f));
        m_buf[i] = m_buf[i] * hConj;
    }
    m_fft->fft2d(m_buf.data(), m_w, m_h, true);

    // Response surface, peak search.
    float peak = -1e30f;
    int px = 0, py = 0;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            const float v = m_buf[(size_t)y * m_w + x].real();
            m_resp[(size_t)y * m_w + x] = v;
            if (v > peak)
            {
                peak = v;
                px = x;
                py = y;
            }
        }
    }

    // PSR: exclude 11x11 region around the peak.
    double mean = 0.0;
    int cnt = 0;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            if (std::abs(x - px) <= 5 && std::abs(y - py) <= 5)
                continue;
            mean += m_resp[(size_t)y * m_w + x];
            ++cnt;
        }
    }
    mean /= (double)std::max(cnt, 1);
    double var = 0.0;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            if (std::abs(x - px) <= 5 && std::abs(y - py) <= 5)
                continue;
            const double d = m_resp[(size_t)y * m_w + x] - mean;
            var += d * d;
        }
    }
    const float sd = (float)std::sqrt(var / (double)std::max(cnt, 1));
    res.psr = (peak - (float)mean) / (sd + 1e-6f);
    res.peak = peak;

    // Detection probability: combination of response peak value (the
    // trained filter answers ~1.0 on the object, near 0 on background or
    // occluders) and peak-to-sidelobe ratio (peak sharpness).
    const float peakTerm = std::min(std::max(peak * 1.6f, 0.0f), 1.0f);
    const float psrTerm =
        std::min(std::max((res.psr - 2.0f) / 10.0f, 0.0f), 1.0f);
    res.prob = peakTerm * psrTerm;

    // Subpixel peak refinement (parabola fit).
    float sx = (float)px, sy = (float)py;
    if (px > 0 && px < m_w - 1)
    {
        const float l = m_resp[(size_t)py * m_w + px - 1];
        const float r = m_resp[(size_t)py * m_w + px + 1];
        const float d = l - 2.0f * peak + r;
        if (std::fabs(d) > 1e-9f)
            sx += 0.5f * (l - r) / d;
    }
    if (py > 0 && py < m_h - 1)
    {
        const float t = m_resp[(size_t)(py - 1) * m_w + px];
        const float b = m_resp[(size_t)(py + 1) * m_w + px];
        const float d = t - 2.0f * peak + b;
        if (std::fabs(d) > 1e-9f)
            sy += 0.5f * (t - b) / d;
    }

    res.dx = sx - (float)(m_w / 2);
    res.dy = sy - (float)(m_h / 2);
    return res;
}

void CorrelationCore::update(const std::vector<float>& window, float rate)
{
    if (!m_initialized || rate <= 0.0f)
        return;

    preprocess(window.data(), m_buf);
    m_fft->fft2d(m_buf.data(), m_w, m_h, false);
    const size_t n = (size_t)m_w * m_h;
    const float keep = 1.0f - rate;
    for (size_t i = 0; i < n; ++i)
    {
        const cf f = m_buf[i];
        const cf fc = std::conj(f);
        m_A[i] = keep * m_A[i] + rate * m_G[i] * fc;
        m_B[i] = keep * m_B[i] + rate * f * fc;
    }
    for (size_t i = 0; i < n; ++i)
        m_ref[i] = keep * m_ref[i] + rate * window[i];
}

void CorrelationCore::objectMask(std::vector<uint8_t>& mask) const
{
    const size_t n = (size_t)m_w * m_h;
    mask.assign(n, 0);
    if (!m_initialized)
        return;

    // Background statistics from window border band (10% of size).
    const int bx = std::max(2, m_w / 10);
    const int by = std::max(2, m_h / 10);
    double mean = 0.0;
    int cnt = 0;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            if (x >= bx && x < m_w - bx && y >= by && y < m_h - by)
                continue;
            mean += m_ref[(size_t)y * m_w + x];
            ++cnt;
        }
    }
    mean /= (double)std::max(cnt, 1);
    double var = 0.0;
    for (int y = 0; y < m_h; ++y)
    {
        for (int x = 0; x < m_w; ++x)
        {
            if (x >= bx && x < m_w - bx && y >= by && y < m_h - by)
                continue;
            const double d = m_ref[(size_t)y * m_w + x] - mean;
            var += d * d;
        }
    }
    const float sd = (float)std::sqrt(var / (double)std::max(cnt, 1));
    const float thr = 2.0f * std::max(sd, 4.0f);

    for (int y = by; y < m_h - by; ++y)
        for (int x = bx; x < m_w - bx; ++x)
            if (std::fabs(m_ref[(size_t)y * m_w + x] - (float)mean) > thr)
                mask[(size_t)y * m_w + x] = 255;
}

} // namespace vtracker
} // namespace cr
