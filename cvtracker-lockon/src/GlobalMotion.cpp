#include "GlobalMotion.h"
#include <algorithm>
#include <cmath>

namespace cr {
namespace vtracker {

void GlobalMotion::init(FftBackend* fft)
{
    m_fft = fft;
    const size_t n = (size_t)GRID * GRID;
    m_prev.assign(n, cf(0.0f, 0.0f));
    m_cur.assign(n, cf(0.0f, 0.0f));
    m_buf.assign(n, cf(0.0f, 0.0f));
    m_hann1d.resize(GRID);
    for (int i = 0; i < GRID; ++i)
        m_hann1d[i] =
            0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (GRID - 1));
    reset();
}

void GlobalMotion::reset()
{
    m_hasPrev = false;
    m_prevId = -1000000;
    m_prevW = 0;
    m_prevH = 0;
}

void GlobalMotion::downsample(const uint8_t* luma, int w, int h,
                              std::vector<cf>& out)
{
    // Box average each grid cell (also acts as an anti-alias filter).
    static thread_local std::vector<float> cells;
    cells.resize((size_t)GRID * GRID);
    double total = 0.0;
    for (int gy = 0; gy < GRID; ++gy)
    {
        const int y0 = (int)((int64_t)gy * h / GRID);
        const int y1 = std::max(y0 + 1, (int)((int64_t)(gy + 1) * h / GRID));
        for (int gx = 0; gx < GRID; ++gx)
        {
            const int x0 = (int)((int64_t)gx * w / GRID);
            const int x1 =
                std::max(x0 + 1, (int)((int64_t)(gx + 1) * w / GRID));
            int sum = 0;
            for (int y = y0; y < y1; ++y)
            {
                const uint8_t* row = luma + (size_t)y * w;
                for (int x = x0; x < x1; ++x)
                    sum += row[x];
            }
            const float v = (float)sum / (float)((x1 - x0) * (y1 - y0));
            cells[(size_t)gy * GRID + gx] = v;
            total += v;
        }
    }
    // Zero-mean (removes global brightness), Hann window (removes FFT
    // wrap-around edge artifacts).
    const float mean = (float)(total / ((double)GRID * GRID));
    for (int gy = 0; gy < GRID; ++gy)
    {
        const float wy = m_hann1d[gy];
        for (int gx = 0; gx < GRID; ++gx)
        {
            const size_t i = (size_t)gy * GRID + gx;
            out[i] = cf((cells[i] - mean) * wy * m_hann1d[gx], 0.0f);
        }
    }
}

bool GlobalMotion::measure(const uint8_t* luma, int w, int h, int frameId,
                           float& dx, float& dy, float& confidence)
{
    dx = 0.0f;
    dy = 0.0f;
    confidence = 0.0f;
    if (m_fft == nullptr || w <= 0 || h <= 0)
        return false;

    downsample(luma, w, h, m_cur);
    m_fft->fft2d(m_cur.data(), GRID, GRID, false);

    const bool contiguous =
        m_hasPrev && (frameId - m_prevId) == 1 && w == m_prevW && h == m_prevH;

    bool valid = false;
    if (contiguous)
    {
        // Cross-power spectrum, magnitude-normalized ("phase only"):
        //   R(k) = Fcur(k) * conj(Fprev(k)) / |Fcur(k) * conj(Fprev(k))|
        // For cur(x) = prev(x - d) this equals e^{-i 2 pi k d / N}, whose
        // inverse FFT (with this backend's e^{+i} inverse convention) is a
        // delta at +d. The peak location IS the scene shift.
        const size_t n = (size_t)GRID * GRID;
        for (size_t i = 0; i < n; ++i)
        {
            const cf r = m_cur[i] * std::conj(m_prev[i]);
            const float mag = std::abs(r);
            m_buf[i] = mag > 1e-12f ? r / mag : cf(0.0f, 0.0f);
        }
        m_fft->fft2d(m_buf.data(), GRID, GRID, true);

        // Peak over the (real) response; indices wrap: i > GRID/2 means a
        // negative shift of (i - GRID).
        float peak = -1e30f;
        int px = 0, py = 0;
        for (int y = 0; y < GRID; ++y)
        {
            for (int x = 0; x < GRID; ++x)
            {
                const float v = m_buf[(size_t)y * GRID + x].real();
                if (v > peak)
                {
                    peak = v;
                    px = x;
                    py = y;
                }
            }
        }

        // Sub-pixel refinement with wrapped neighbors.
        auto at = [&](int x, int y) -> float {
            x = (x + GRID) % GRID;
            y = (y + GRID) % GRID;
            return m_buf[(size_t)y * GRID + x].real();
        };
        float sx = (float)px, sy = (float)py;
        {
            const float l = at(px - 1, py), r = at(px + 1, py);
            const float d = l - 2.0f * peak + r;
            if (std::fabs(d) > 1e-9f)
                sx += 0.5f * (l - r) / d;
        }
        {
            const float t = at(px, py - 1), b = at(px, py + 1);
            const float d = t - 2.0f * peak + b;
            if (std::fabs(d) > 1e-9f)
                sy += 0.5f * (t - b) / d;
        }
        // Unwrap to signed shifts.
        if (sx > GRID / 2)
            sx -= GRID;
        if (sy > GRID / 2)
            sy -= GRID;

        confidence = std::max(0.0f, peak);
        const float lim = MAX_SHIFT_FRAC * GRID;
        if (confidence >= MIN_CONFIDENCE && std::fabs(sx) <= lim &&
            std::fabs(sy) <= lim)
        {
            // Grid pixels -> full-resolution frame pixels.
            dx = sx * ((float)w / (float)GRID);
            dy = sy * ((float)h / (float)GRID);
            valid = true;
        }
    }

    // Current becomes previous (swap avoids reallocation).
    std::swap(m_prev, m_cur);
    m_hasPrev = true;
    m_prevId = frameId;
    m_prevW = w;
    m_prevH = h;
    return valid;
}

} // namespace vtracker
} // namespace cr
