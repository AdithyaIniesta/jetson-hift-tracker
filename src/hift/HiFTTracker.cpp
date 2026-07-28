#include "hift/HiFTTracker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <utility>

#include <cvtracker/Frame.h>

namespace cr {
namespace hift {

using cr::vtracker::VTrackerCommand;
using cr::vtracker::VTrackerParam;
using cr::vtracker::VTrackerParams;

namespace {

constexpr int MODE_FREE = 0;
constexpr int MODE_TRACKING = 1;
constexpr int MODE_LOST = 2;
constexpr int MODE_INERTIAL = 3;
constexpr int MODE_STATIC = 4;

// Box size may not stray beyond this band around the captured size (stops the
// HiFT regressor drifting the box larger when it locks onto clutter).
// NB: not SIZE_MIN/SIZE_MAX — SIZE_MAX is a <stdint.h> macro.
constexpr float BOX_SIZE_MIN = 0.4f;
constexpr float BOX_SIZE_MAX = 2.5f;

// atanh with the HiFT clamp (dcon in hift_tracker.py), then loc decode scale.
inline float dcon(float x)
{
    if (x <= -1.0f) x = -0.99f;
    if (x >= 1.0f) x = 0.99f;
    return 0.5f * (std::log(1.0f + x) - std::log(1.0f - x));
}

// BT.601 YUV (0..255) -> BGR (0..255), clamped.
inline void yuvToBgr(float Y, float U, float V, float& b, float& g, float& r)
{
    const float u = U - 128.0f;
    const float v = V - 128.0f;
    r = Y + 1.402f * v;
    g = Y - 0.344136f * u - 0.714136f * v;
    b = Y + 1.772f * u;
    r = std::min(255.0f, std::max(0.0f, r));
    g = std::min(255.0f, std::max(0.0f, g));
    b = std::min(255.0f, std::max(0.0f, b));
}

// Sample one source pixel as YUV given the frame's pixel format. (x,y) assumed
// in-bounds. Returns Y/U/V in 0..255.
void pixelYuv(const cr::video::Frame& f, int x, int y, float& Y, float& U,
              float& V)
{
    const uint8_t* d = f.data;
    const int w = f.width;
    const int h = f.height;
    using cr::video::Fourcc;
    switch (f.fourcc)
    {
    case Fourcc::GRAY:
        Y = d[y * w + x];
        U = V = 128.0f;
        return;
    case Fourcc::YUV24:
    {
        const int i = (y * w + x) * 3;
        Y = d[i];
        U = d[i + 1];
        V = d[i + 2];
        return;
    }
    case Fourcc::YUYV:  // [Y0 U Y1 V]
    {
        const int pair = x & ~1;
        const int base = (y * w + pair) * 2;
        Y = (x & 1) ? d[base + 2] : d[base];
        U = d[base + 1];
        V = d[base + 3];
        return;
    }
    case Fourcc::UYVY:  // [U Y0 V Y1]
    {
        const int pair = x & ~1;
        const int base = (y * w + pair) * 2;
        Y = (x & 1) ? d[base + 3] : d[base + 1];
        U = d[base];
        V = d[base + 2];
        return;
    }
    case Fourcc::NV12:  // Y plane + interleaved UV (2x2 subsampled)
    {
        Y = d[y * w + x];
        const int uv = w * h + (y / 2) * w + (x / 2) * 2;
        U = d[uv];
        V = d[uv + 1];
        return;
    }
    case Fourcc::NV21:  // Y plane + interleaved VU
    {
        Y = d[y * w + x];
        const int uv = w * h + (y / 2) * w + (x / 2) * 2;
        V = d[uv];
        U = d[uv + 1];
        return;
    }
    case Fourcc::YU12:  // I420: Y, then U plane, then V plane
    {
        const int cw = w / 2;
        const int ch = h / 2;
        Y = d[y * w + x];
        const int c = (y / 2) * cw + (x / 2);
        U = d[w * h + c];
        V = d[w * h + cw * ch + c];
        return;
    }
    case Fourcc::YV12:  // Y, then V plane, then U plane
    {
        const int cw = w / 2;
        const int ch = h / 2;
        Y = d[y * w + x];
        const int c = (y / 2) * cw + (x / 2);
        V = d[w * h + c];
        U = d[w * h + cw * ch + c];
        return;
    }
    default:
        Y = d[y * w + x];
        U = V = 128.0f;
        return;
    }
}

// Debug: write a CHW-BGR float crop (side S, values 0..255) to a PPM (P6, RGB)
// so the operator can eyeball whether the color/format handed to HiFT is right.
// Enabled only when the env var TRACKER_HIFT_DUMP is set. Also logs the per-
// channel mean (a quick sanity check on range: expect ~0..255, not 0..1).
void dumpCropPpm(const char* path, const std::vector<float>& chw, int S)
{
    const int plane = S * S;
    double mb = 0, mg = 0, mr = 0;
    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        std::fprintf(stderr, "[hift/dump] cannot open %s\n", path);
        return;
    }
    f << "P6\n" << S << " " << S << "\n255\n";
    std::vector<unsigned char> row(3 * plane);
    for (int j = 0; j < plane; ++j)
    {
        const float b = chw[j];
        const float g = chw[plane + j];
        const float r = chw[2 * plane + j];
        mb += b;
        mg += g;
        mr += r;
        auto cl = [](float v) {
            return (unsigned char)std::min(255.0f, std::max(0.0f, v));
        };
        row[3 * j + 0] = cl(r);  // PPM is RGB
        row[3 * j + 1] = cl(g);
        row[3 * j + 2] = cl(b);
    }
    f.write((const char*)row.data(), (std::streamsize)row.size());
    std::fprintf(stderr,
                 "[hift/dump] wrote %s (%dx%d) mean BGR = %.1f %.1f %.1f\n",
                 path, S, S, mb / plane, mg / plane, mr / plane);
}

}  // namespace

struct HiFTTracker::Impl
{
    HiFTTrackerConfig cfg;
    TrtHiFT engine;
    std::mutex mtx;
    bool engineReady{false};

    VTrackerParams params;
    int mode{MODE_FREE};

    // Track state (float, in full-frame pixels).
    float cx{0.0f}, cy{0.0f};   // center
    float bw{72.0f}, bh{72.0f}; // box size
    float bw0{72.0f}, bh0{72.0f}; // captured size (size-clamp reference)
    float velX{0.0f}, velY{0.0f};
    float channelB{128.0f}, channelG{128.0f}, channelR{128.0f};
    float scaleaa{0.0f};        // s_z captured for large-object clamp
    int lostCounter{0};
    bool haveTemplate{false};
    bool dumpSearchPending{false};

    int frameW{0}, frameH{0};

    // Pending capture requested via executeCommand before the next frame.
    bool pendingCapture{false};
    float capX{-1.0f}, capY{-1.0f};

    // Precomputed anchor grid centers (relative to crop center) + cosine window.
    std::vector<float> gridX, gridY, window;

    // Reusable crop buffers (CHW BGR float).
    std::vector<float> zBuf, xBuf;
    // Stored template crop for handoff export.
    std::vector<float> templateCrop;

    // ── DINOv2 verifier state ────────────────────────────────────────────
    std::shared_ptr<cr::vtracker::FeatureExtractor> extractor;
    std::vector<float> bank;      // reference embedding (unit norm)
    std::vector<float> grayBuf;   // reusable gray verifier crop (fallback)
    std::vector<float> colorBuf;  // reusable RGB verifier crop (CHW, 0..255)
    std::vector<float> embBuf;    // reusable embedding
    int vetoStreak{0};
    int sinceVerify{0};
    int sinceRefresh{0};
    bool debug{false};  // TRACKER_HIFT_DEBUG: log verifier similarities
    // Re-detection probe grid (frame-space centers, swept across LOST frames).
    std::vector<std::pair<float, float>> probeCenters;
    int probeIndex{0};
    int probeGridW{0}, probeGridH{0};

    Impl()
    {
        const int N = HIFT_OUTPUT;
        gridX.resize(N * N);
        gridY.resize(N * N);
        window.resize(N * N);
        std::vector<float> hann(N);
        for (int i = 0; i < N; ++i)
            hann[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * i /
                                              (N - 1)));
        for (int row = 0; row < N; ++row)
            for (int col = 0; col < N; ++col)
            {
                const int j = row * N + col;
                gridX[j] = HIFT_ANCHOR_STRIDE * col + HIFT_ANCHOR_OFF -
                           HIFT_DECODE_SCALE;
                gridY[j] = HIFT_ANCHOR_STRIDE * row + HIFT_ANCHOR_OFF -
                           HIFT_DECODE_SCALE;
                window[j] = hann[row] * hann[col];
            }
        zBuf.resize(3 * HIFT_EXEMPLAR * HIFT_EXEMPLAR);
        xBuf.resize(3 * HIFT_SEARCH * HIFT_SEARCH);
    }

    // Build a context-padded, resized CHW BGR crop of `frame` centered at
    // (px,py), source box side originalSz, output side modelSz. Fills `out`
    // (size 3*modelSz*modelSz). Out-of-bounds samples use the channel average.
    void buildCrop(const cr::video::Frame& frame, float px, float py,
                   int originalSz, int modelSz, std::vector<float>& out)
    {
        const float c = (originalSz + 1) / 2.0f;
        const float xmin = std::floor(px - c + 0.5f);
        const float ymin = std::floor(py - c + 0.5f);
        const int plane = modelSz * modelSz;
        float* B = out.data();
        float* G = out.data() + plane;
        float* R = out.data() + 2 * plane;
        const float sc = (float)originalSz / (float)modelSz;
        // BGR of one integer source pixel (channel average if out of bounds).
        auto bgrAt = [&](int x, int y, float& b, float& g, float& r) {
            if (x < 0 || y < 0 || x >= frame.width || y >= frame.height)
            {
                b = channelB;
                g = channelG;
                r = channelR;
                return;
            }
            float Y, U, V;
            pixelYuv(frame, x, y, Y, U, V);
            yuvToBgr(Y, U, V, b, g, r);
        };
        for (int oy = 0; oy < modelSz; ++oy)
        {
            const float fy = ymin + (oy + 0.5f) * sc - 0.5f;
            const int y0 = (int)std::floor(fy);
            const float ay = fy - y0;
            for (int ox = 0; ox < modelSz; ++ox)
            {
                const float fx = xmin + (ox + 0.5f) * sc - 0.5f;
                const int x0 = (int)std::floor(fx);
                const float ax = fx - x0;
                // Bilinear blend of the 4 neighbours (matches cv2.resize, the
                // interpolation the ONNX was traced against; nearest-neighbour
                // adds sub-pixel jitter frame-to-frame).
                float b00, g00, r00, b10, g10, r10, b01, g01, r01, b11, g11, r11;
                bgrAt(x0, y0, b00, g00, r00);
                bgrAt(x0 + 1, y0, b10, g10, r10);
                bgrAt(x0, y0 + 1, b01, g01, r01);
                bgrAt(x0 + 1, y0 + 1, b11, g11, r11);
                const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay);
                const float w01 = (1 - ax) * ay, w11 = ax * ay;
                const int j = oy * modelSz + ox;
                B[j] = w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11;
                G[j] = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11;
                R[j] = w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11;
            }
        }
    }

    // Whole-frame channel average (BGR) — HiFT's avg_chans for pad fill.
    void computeChannelAverage(const cr::video::Frame& frame)
    {
        // Subsample for speed (every 4th pixel each axis).
        double sb = 0, sg = 0, sr = 0;
        long n = 0;
        for (int y = 0; y < frame.height; y += 4)
            for (int x = 0; x < frame.width; x += 4)
            {
                float Y, U, V, b, g, r;
                pixelYuv(frame, x, y, Y, U, V);
                yuvToBgr(Y, U, V, b, g, r);
                sb += b;
                sg += g;
                sr += r;
                ++n;
            }
        if (n > 0)
        {
            channelB = (float)(sb / n);
            channelG = (float)(sg / n);
            channelR = (float)(sr / n);
        }
    }

    // ── Appearance verifier helpers ─────────────────────────────────────
    // Grayscale (luminance) crop of the box region, resized to outSz x outSz,
    // for the DINOv2 extractor (which resamples/normalizes internally).
    void buildGrayCrop(const cr::video::Frame& frame, float pcx, float pcy,
                       float boxW, float boxH, int outSz, std::vector<float>& out)
    {
        out.resize((size_t)outSz * outSz);
        const float x0 = pcx - boxW * 0.5f;
        const float y0 = pcy - boxH * 0.5f;
        const float scx = boxW / (float)outSz;
        const float scy = boxH / (float)outSz;
        for (int oy = 0; oy < outSz; ++oy)
        {
            int sy = (int)std::lround(y0 + (oy + 0.5f) * scy);
            sy = std::max(0, std::min(sy, frame.height - 1));
            for (int ox = 0; ox < outSz; ++ox)
            {
                int sx = (int)std::lround(x0 + (ox + 0.5f) * scx);
                sx = std::max(0, std::min(sx, frame.width - 1));
                float Y, U, V;
                pixelYuv(frame, sx, sy, Y, U, V);
                out[(size_t)oy * outSz + ox] = Y;  // luminance
            }
        }
    }

    // RGB (colour) crop of the box region → CHW, 0..255, for the ImageNet/
    // DINOv2 extractor. Colour is a strong target discriminator; grayscale
    // throws it away.
    void buildColorCrop(const cr::video::Frame& frame, float pcx, float pcy,
                        float boxW, float boxH, int outSz, std::vector<float>& out)
    {
        out.resize((size_t)outSz * outSz * 3);
        const int plane = outSz * outSz;
        float* R = out.data();
        float* G = out.data() + plane;
        float* B = out.data() + 2 * plane;
        const float x0 = pcx - boxW * 0.5f;
        const float y0 = pcy - boxH * 0.5f;
        const float scx = boxW / (float)outSz;
        const float scy = boxH / (float)outSz;
        for (int oy = 0; oy < outSz; ++oy)
        {
            int sy = (int)std::lround(y0 + (oy + 0.5f) * scy);
            sy = std::max(0, std::min(sy, frame.height - 1));
            for (int ox = 0; ox < outSz; ++ox)
            {
                int sx = (int)std::lround(x0 + (ox + 0.5f) * scx);
                sx = std::max(0, std::min(sx, frame.width - 1));
                float Y, U, V, b, g, r;
                pixelYuv(frame, sx, sy, Y, U, V);
                yuvToBgr(Y, U, V, b, g, r);
                const int j = oy * outSz + ox;
                R[j] = r;
                G[j] = g;
                B[j] = b;
            }
        }
    }

    bool computeEmbedding(const cr::video::Frame& frame, float pcx, float pcy,
                          float boxW, float boxH, std::vector<float>& emb)
    {
        if (!extractor)
            return false;
        const int S = extractor->inputSize();
        // Prefer the colour path; fall back to grayscale if unsupported.
        buildColorCrop(frame, pcx, pcy, boxW, boxH, S, colorBuf);
        if (extractor->extractColor(colorBuf.data(), emb))
            return true;
        buildGrayCrop(frame, pcx, pcy, boxW, boxH, S, grayBuf);
        return extractor->extract(grayBuf.data(), emb);
    }

    static void normalize(std::vector<float>& v)
    {
        double s = 0.0;
        for (float x : v)
            s += (double)x * x;
        const float inv = 1.0f / ((float)std::sqrt(s) + 1e-6f);
        for (float& x : v)
            x *= inv;
    }

    static float cosine(const std::vector<float>& a, const std::vector<float>& b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9));
    }

    // Re-run HiFT's template branch on the current box (adaptation). Caller
    // must gate this on a confident + verifier-confirmed frame.
    void refreshTemplate(const cr::video::Frame& frame)
    {
        const float wz = bw + HIFT_CONTEXT * (bw + bh);
        const float hz = bh + HIFT_CONTEXT * (bw + bh);
        const float sz = std::round(std::sqrt(wz * hz));
        buildCrop(frame, cx, cy, (int)sz, HIFT_EXEMPLAR, zBuf);
        if (engine.setTemplate(zBuf.data()))
            templateCrop = zBuf;
    }

    // Coarse grid of probe centers covering the frame (built once per size).
    void buildProbeGrid(int w, int h)
    {
        if (probeGridW == w && probeGridH == h && !probeCenters.empty())
            return;
        probeGridW = w;
        probeGridH = h;
        probeCenters.clear();
        const float f[4] = {0.2f, 0.4f, 0.6f, 0.8f};
        for (float ry : f)
            for (float rx : f)
                probeCenters.emplace_back(rx * w, ry * h);
    }

    // Active re-detection: probe a few grid centers this frame, run HiFT at each
    // and confirm with DINOv2. Returns true (and re-acquires) on a confirmed
    // match. Leaves the box at last-good on failure.
    bool tryRedetect(const cr::video::Frame& frame)
    {
        if (!extractor || bank.empty())
            return false;
        buildProbeGrid(frame.width, frame.height);
        if (probeCenters.empty())
            return false;
        const float lcx = cx, lcy = cy, lbw = bw, lbh = bh;
        for (int k = 0; k < cfg.probesPerFrame; ++k)
        {
            const auto& pc = probeCenters[probeIndex % probeCenters.size()];
            ++probeIndex;
            cx = pc.first;
            cy = pc.second;
            bw = lbw;
            bh = lbh;
            const float conf = trackStep(frame);  // HiFT localizes near probe
            if (conf >= cfg.lossThreshold &&
                computeEmbedding(frame, cx, cy, bw, bh, embBuf))
            {
                const float sim = cosine(embBuf, bank);
                if (debug)
                    std::fprintf(stderr,
                                 "[hift] redetect probe(%.0f,%.0f) conf=%.3f "
                                 "sim=%.3f (reacq>=%.2f)\n",
                                 cx, cy, conf, sim, cfg.reacquireThreshold);
                if (sim >= cfg.reacquireThreshold)
                {
                    refreshTemplate(frame);  // re-seed HiFT at the found target
                    vetoStreak = 0;
                    lostCounter = 0;
                    sinceVerify = 0;
                    sinceRefresh = 0;
                    return true;
                }
            }
            cx = lcx;  // reject this probe, restore last-good
            cy = lcy;
            bw = lbw;
            bh = lbh;
            velX = velY = 0.0f;
        }
        return false;
    }

    void doCapture(const cr::video::Frame& frame)
    {
        float px = capX, py = capY;
        if (px < 0) px = frameW / 2.0f;
        if (py < 0) py = frameH / 2.0f;
        cx = px;
        cy = py;
        bw = (float)params.rectWidth;
        bh = (float)params.rectHeight;
        bw0 = bw;   // remember capture size for the size clamp
        bh0 = bh;

        computeChannelAverage(frame);

        const float wz = bw + HIFT_CONTEXT * (bw + bh);
        const float hz = bh + HIFT_CONTEXT * (bw + bh);
        const float sz = std::round(std::sqrt(wz * hz));
        scaleaa = sz;

        buildCrop(frame, cx, cy, (int)sz, HIFT_EXEMPLAR, zBuf);
        if (std::getenv("TRACKER_HIFT_DUMP"))
        {
            dumpCropPpm("hift_template.ppm", zBuf, HIFT_EXEMPLAR);
            dumpSearchPending = true;
        }
        if (engine.setTemplate(zBuf.data()))
        {
            templateCrop = zBuf;
            haveTemplate = true;
            mode = MODE_TRACKING;
            lostCounter = 0;
            velX = velY = 0.0f;
            // Seed the appearance verifier with the captured target.
            vetoStreak = 0;
            sinceVerify = 0;
            sinceRefresh = 0;
            bank.clear();
            if (extractor && computeEmbedding(frame, cx, cy, bw, bh, embBuf))
            {
                bank = embBuf;
                normalize(bank);
            }
        }
        else
        {
            mode = MODE_FREE;
            haveTemplate = false;
        }
    }

    // Run one HiFT track step on the current frame. Returns fused best score.
    float trackStep(const cr::video::Frame& frame)
    {
        const int N = HIFT_OUTPUT;
        // s_z from current size, clamped for very large objects (HiFT quirk).
        float wz = bw + HIFT_CONTEXT * (bw + bh);
        float hz = bh + HIFT_CONTEXT * (bw + bh);
        float sz = std::sqrt(wz * hz);
        if (bw * bh > 0.5f * frame.width * frame.height)
            sz = scaleaa;
        const float scaleZ = HIFT_EXEMPLAR / sz;
        const float sx = sz * ((float)HIFT_SEARCH / (float)HIFT_EXEMPLAR);

        buildCrop(frame, cx, cy, (int)std::round(sx), HIFT_SEARCH, xBuf);
        if (dumpSearchPending)
        {
            dumpSearchPending = false;
            dumpCropPpm("hift_search.ppm", xBuf, HIFT_SEARCH);
        }

        HiFTTrackOut o;
        if (!engine.track(xBuf.data(), o))
            return 0.0f;

        // Decode loc[4,N,N] -> per-location box (x,y offset from crop center,
        // w,h) exactly as generate_anchor().
        const int P = N * N;
        const float* loc = o.loc.data();  // channels: 0..3
        // cls1[2,N,N] softmax over channel -> foreground prob; cls2[1,N,N].
        const float* cls1 = o.cls1.data();
        const float* cls2 = o.cls2.data();

        float bestP = -1e30f;
        int best = 0;
        float bestScore = 0.0f, bestPenalty = 0.0f, bestFg = 0.0f;
        float bbx = 0, bby = 0, bbw = 0, bbh = 0;
        // Peak-quality (PSR) accumulators over the foreground map — HiFT's cls
        // score stays high even when the target is gone (it locks onto clutter),
        // so a sharp-single-peak test is what actually detects loss.
        float maxFg = 0.0f, sumFg = 0.0f, sumFg2 = 0.0f;

        const float refW = bw * scaleZ;
        const float refH = bh * scaleZ;
        auto szf = [](float w, float h) {
            const float pad = (w + h) * 0.5f;
            return std::sqrt((w + pad) * (h + pad));
        };
        auto change = [](float r) { return std::max(r, 1.0f / (r + 1e-5f)); };
        const float refSz = szf(refW, refH);
        const float refRatio = bw / (bh + 1e-5f);

        for (int j = 0; j < P; ++j)
        {
            const float s0 = dcon(loc[0 * P + j]) * HIFT_DECODE_SCALE;
            const float s1 = dcon(loc[1 * P + j]) * HIFT_DECODE_SCALE;
            const float s2 = dcon(loc[2 * P + j]) * HIFT_DECODE_SCALE;
            const float s3 = dcon(loc[3 * P + j]) * HIFT_DECODE_SCALE;
            const float w = s0 + s1;
            const float h = s2 + s3;
            const float bx = gridX[j] - s0 + w / 2.0f;
            const float by = gridY[j] - s2 + h / 2.0f;

            // cls1 softmax(fg): channels laid out [2,N,N], index c*P+j.
            const float a0 = cls1[0 * P + j];
            const float a1 = cls1[1 * P + j];
            const float m = std::max(a0, a1);
            const float e0 = std::exp(a0 - m), e1 = std::exp(a1 - m);
            const float fg = e1 / (e0 + e1);              // bounded 0..1
            sumFg += fg;
            sumFg2 += fg * fg;
            if (fg > maxFg)
                maxFg = fg;
            const float score1 = fg * HIFT_W2;
            const float score2 = cls2[j] * HIFT_W3;       // cls2 is a raw logit
            const float score = (score1 + score2) * 0.5f;

            const float sc = change(szf(w, h) / (refSz + 1e-9f));
            const float rc =
                change(refRatio / ((w / (h + 1e-5f)) + 1e-9f));
            const float penalty = std::exp(-(rc * sc - 1.0f) * HIFT_PENALTY_K);
            float pscore = penalty * score;
            pscore = pscore * (1.0f - cfg.windowInf) +
                     window[j] * cfg.windowInf;

            if (pscore > bestP)
            {
                bestP = pscore;
                best = j;
                bestScore = score;
                bestFg = fg;
                bestPenalty = penalty;
                bbx = bx;
                bby = by;
                bbw = w;
                bbh = h;
            }
        }
        (void)best;

        // ── Peak-to-sidelobe ratio (PSR): is there ONE sharp peak (real target)
        //    or a diffuse response (target gone → locked on clutter)? Exclude the
        //    peak itself from the sidelobe stats. ─────────────────────────────
        const float mean = (sumFg - maxFg) / (float)(P - 1);
        float var = (sumFg2 - maxFg * maxFg) / (float)(P - 1) - mean * mean;
        if (var < 1e-6f)
            var = 1e-6f;
        const float psr = (maxFg - mean) / std::sqrt(var);
        // Confidence = foreground strength gated by peak sharpness. Both are high
        // only when the target is genuinely present. This is what makes the box
        // turn "LOST" (blue) and the reported probability drop when it leaves FOV.
        const float psrConf = std::min(1.0f, psr / cfg.psrRef);
        const float conf = bestFg * psrConf;

        // Not confident → the target is likely out of view / occluded. FREEZE the
        // box: do NOT chase the clutter peak (that is what wandered + enlarged the
        // box). Position/size held at last-known; the mode machine goes LOST.
        if (conf < cfg.lossThreshold)
        {
            velX = velY = 0.0f;
            return conf;
        }

        // Back to full-frame pixels.
        const float invSz = 1.0f / scaleZ;
        const float ox = bbx * invSz;
        const float oy = bby * invSz;
        const float ow = bbw * invSz;
        const float oh = bbh * invSz;

        // Learning rate must stay a fraction: cls2 is an unbounded logit, so the
        // raw penalty*score*LR can exceed 1 and make the size update extrapolate
        // (box blow-up / jitter). Clamp to [0,1].
        const float lr =
            std::min(1.0f, std::max(0.0f, bestPenalty * bestScore * HIFT_LR));
        const float newCx = ox + cx;
        const float newCy = oy + cy;
        float newW = bw * (1.0f - lr) + ow * lr;
        float newH = bh * (1.0f - lr) + oh * lr;

        // Hard size clamp relative to the captured size — HiFT's regressor drifts
        // the box larger over time; never let it stray far from what was captured.
        newW = std::max(BOX_SIZE_MIN * bw0, std::min(newW, BOX_SIZE_MAX * bw0));
        newH = std::max(BOX_SIZE_MIN * bh0, std::min(newH, BOX_SIZE_MAX * bh0));
        // Frame clip.
        const float ncx = std::max(0.0f, std::min(newCx, (float)frame.width));
        const float ncy = std::max(0.0f, std::min(newCy, (float)frame.height));
        newW = std::max(10.0f, std::min(newW, (float)frame.width));
        newH = std::max(10.0f, std::min(newH, (float)frame.height));

        velX = ncx - cx;
        velY = ncy - cy;
        cx = ncx;
        cy = ncy;
        bw = newW;
        bh = newH;
        return conf;
    }
};

HiFTTracker::HiFTTracker(const HiFTTrackerConfig& cfg) : d_(new Impl)
{
    d_->cfg = cfg;
    // Loss threshold is compared against the foreground probability (0..1) and
    // can be tuned on the Jetson without a rebuild: TRACKER_HIFT_LOSS=0.15
    if (const char* e = std::getenv("TRACKER_HIFT_LOSS"))
    {
        const float v = (float)atof(e);
        if (v > 0.0f && v < 1.0f)
            d_->cfg.lossThreshold = v;
    }
    // TRACKER_HIFT_PSR tunes loss sensitivity: higher = stricter (goes LOST
    // sooner when the response gets diffuse).
    if (const char* e = std::getenv("TRACKER_HIFT_PSR"))
    {
        const float v = (float)atof(e);
        if (v > 0.5f && v < 50.0f)
            d_->cfg.psrRef = v;
    }
    // TRACKER_HIFT_WINDOW: stronger center bias (0..0.95) to damp peak flip-flop.
    if (const char* e = std::getenv("TRACKER_HIFT_WINDOW"))
    {
        const float v = (float)atof(e);
        if (v >= 0.0f && v < 0.95f)
            d_->cfg.windowInf = v;
    }
    // TRACKER_HIFT_SIM: verifier cosine-similarity veto threshold (0..1).
    if (const char* e = std::getenv("TRACKER_HIFT_SIM"))
    {
        const float v = (float)atof(e);
        if (v > 0.0f && v < 1.0f)
            d_->cfg.simThreshold = v;
    }
    // TRACKER_HIFT_REACQ: re-detection acceptance threshold (stricter than SIM).
    if (const char* e = std::getenv("TRACKER_HIFT_REACQ"))
    {
        const float v = (float)atof(e);
        if (v > 0.0f && v < 1.0f)
            d_->cfg.reacquireThreshold = v;
    }
    d_->debug = std::getenv("TRACKER_HIFT_DEBUG") != nullptr;
    d_->engineReady = d_->engine.initialise(d_->cfg.trt);
    if (!d_->engineReady)
        std::fprintf(stderr, "[hift] engine init FAILED — tracker inert\n");
}

HiFTTracker::~HiFTTracker() = default;

std::string HiFTTracker::getVersion() { return "hift-1.0.0"; }
bool HiFTTracker::ready() const { return d_->engineReady; }

void HiFTTracker::setFeatureExtractor(
    std::shared_ptr<cr::vtracker::FeatureExtractor> extractor)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    d_->extractor = std::move(extractor);
    d_->bank.clear();      // reference is re-seeded on the next CAPTURE
    d_->vetoStreak = 0;
    std::fprintf(stderr, "[hift] appearance verifier %s\n",
                 d_->extractor ? "attached" : "detached");
}

bool HiFTTracker::initVTracker(VTrackerParams& params)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    d_->params = params;
    d_->mode = MODE_FREE;
    d_->haveTemplate = false;
    return d_->engineReady;
}

bool HiFTTracker::setParam(VTrackerParam id, float value)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    switch (id)
    {
    case VTrackerParam::RECT_WIDTH:
        d_->params.rectWidth = (int)value;
        return true;
    case VTrackerParam::RECT_HEIGHT:
        d_->params.rectHeight = (int)value;
        return true;
    case VTrackerParam::SEARCH_WINDOW_WIDTH:
        d_->params.searchWindowWidth = (int)value;
        return true;
    case VTrackerParam::SEARCH_WINDOW_HEIGHT:
        d_->params.searchWindowHeight = (int)value;
        return true;
    case VTrackerParam::LOST_MODE_OPTION:
        d_->params.lostModeOption = (int)value;
        return true;
    case VTrackerParam::MAX_FRAMES_IN_LOST_MODE:
        d_->params.maxFramesInLostMode = (int)value;
        return true;
    case VTrackerParam::FRAME_BUFFER_SIZE:
        d_->params.frameBufferSize = (int)value;
        return true;
    default:
        return false;
    }
}

float HiFTTracker::getParam(VTrackerParam id)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    switch (id)
    {
    case VTrackerParam::RECT_WIDTH: return (float)d_->params.rectWidth;
    case VTrackerParam::RECT_HEIGHT: return (float)d_->params.rectHeight;
    case VTrackerParam::SEARCH_WINDOW_WIDTH:
        return (float)d_->params.searchWindowWidth;
    case VTrackerParam::SEARCH_WINDOW_HEIGHT:
        return (float)d_->params.searchWindowHeight;
    case VTrackerParam::LOST_MODE_OPTION:
        return (float)d_->params.lostModeOption;
    case VTrackerParam::MAX_FRAMES_IN_LOST_MODE:
        return (float)d_->params.maxFramesInLostMode;
#ifdef TRACKER_V2
    // The pipeline's external EKF (tracker.cpp) reads these to smooth the
    // tracker's per-frame position. HiFT's raw argmax jitters frame-to-frame,
    // so we DO want that smoothing — return valid noise params (base defaults),
    // not the -1 fallthrough that fed the EKF negative variances and made the
    // "smoothed" output wobble.
    case VTrackerParam::ENABLE_EKF: return 1.0f;
    case VTrackerParam::EKF_SIGMA_A: return 1.5f;
    case VTrackerParam::EKF_SIGMA_ALPHA: return 0.15f;
    case VTrackerParam::EKF_R_BASE: return 4.0f;
#endif
    default: return -1.0f;
    }
}

void HiFTTracker::getParams(VTrackerParams& params)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    d_->params.mode = d_->mode;
    d_->params.rectX = (int)std::lround(d_->cx);
    d_->params.rectY = (int)std::lround(d_->cy);
    d_->params.rectWidth = (int)std::lround(d_->bw);
    d_->params.rectHeight = (int)std::lround(d_->bh);
    d_->params.objectX = d_->params.rectX;
    d_->params.objectY = d_->params.rectY;
    d_->params.objectWidth = d_->params.rectWidth;
    d_->params.objectHeight = d_->params.rectHeight;
    d_->params.velX = d_->velX;
    d_->params.velY = d_->velY;
    d_->params.lostModeFrameCounter = d_->lostCounter;
    d_->params.frameWidth = d_->frameW;
    d_->params.frameHeight = d_->frameH;
    d_->params.searchWindowX = d_->params.rectX;
    d_->params.searchWindowY = d_->params.rectY;
    params = d_->params;
}

bool HiFTTracker::executeCommand(VTrackerCommand id, float arg1, float arg2,
                                 float arg3)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    (void)arg3;
    switch (id)
    {
    case VTrackerCommand::CAPTURE:
        d_->pendingCapture = true;
        d_->capX = arg1;
        d_->capY = arg2;
        return true;
    case VTrackerCommand::CAPTURE_PERCENTS:
        d_->pendingCapture = true;
        d_->capX = (d_->frameW > 0) ? arg1 * 0.01f * d_->frameW : -1.0f;
        d_->capY = (d_->frameH > 0) ? arg2 * 0.01f * d_->frameH : -1.0f;
        return true;
    case VTrackerCommand::RESET:
        d_->mode = MODE_FREE;
        d_->haveTemplate = false;
        d_->pendingCapture = false;
        d_->lostCounter = 0;
        d_->velX = d_->velY = 0.0f;
        return true;
    case VTrackerCommand::SET_INERTIAL_MODE:
        d_->mode = MODE_INERTIAL;
        return true;
    case VTrackerCommand::SET_LOST_MODE:
        d_->mode = MODE_LOST;
        return true;
    case VTrackerCommand::SET_STATIC_MODE:
        d_->mode = MODE_STATIC;
        return true;
    case VTrackerCommand::SET_RECT_POSITION:
        d_->cx = arg1;
        d_->cy = arg2;
        return true;
    case VTrackerCommand::MOVE_RECT:
        d_->cx += arg1;
        d_->cy += arg2;
        return true;
    case VTrackerCommand::CHANGE_RECT_SIZE:
        d_->params.rectWidth =
            std::max(10, d_->params.rectWidth + (int)arg1);
        d_->params.rectHeight =
            std::max(10, d_->params.rectHeight + (int)arg2);
        return true;
    default:
        return false;
    }
}

bool HiFTTracker::processFrame(cr::video::Frame& frame)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    if (!d_->engineReady || frame.data == nullptr)
        return false;
    d_->frameW = frame.width;
    d_->frameH = frame.height;

    if (d_->pendingCapture)
    {
        d_->pendingCapture = false;
        d_->doCapture(frame);
        return true;
    }

    if (d_->mode == MODE_TRACKING || d_->mode == MODE_LOST)
    {
        if (!d_->haveTemplate)
            return true;

        // Remember last-good box so the verifier can reject a distractor jump.
        const float pcx = d_->cx, pcy = d_->cy, pbw = d_->bw, pbh = d_->bh;

        float conf = d_->trackStep(frame);
        bool ok = conf >= d_->cfg.lossThreshold;  // HiFT thinks it's on target

        // ── Appearance verification (DINOv2) ─────────────────────────────
        // HiFT's cls can't tell the target from a look-alike; the verifier can.
        if (ok && d_->extractor && !d_->bank.empty())
        {
            if (++d_->sinceVerify >= d_->cfg.verifyEvery)
            {
                d_->sinceVerify = 0;
                if (d_->computeEmbedding(frame, d_->cx, d_->cy, d_->bw, d_->bh,
                                         d_->embBuf))
                {
                    const float sim = Impl::cosine(d_->embBuf, d_->bank);
                    if (d_->debug)
                        std::fprintf(stderr,
                                     "[hift] verify sim=%.3f conf=%.3f "
                                     "(veto<%.2f)\n",
                                     sim, conf, d_->cfg.simThreshold);
                    if (sim < d_->cfg.simThreshold)
                    {
                        // Distractor: reject the jump, roll back to last-good.
                        d_->cx = pcx;
                        d_->cy = pcy;
                        d_->bw = pbw;
                        d_->bh = pbh;
                        d_->velX = d_->velY = 0.0f;
                        ++d_->vetoStreak;
                        ok = false;
                        conf = std::min(conf, d_->cfg.lossThreshold * 0.9f);
                    }
                    else
                    {
                        d_->vetoStreak = 0;
                        // EMA-refresh the reference so slow appearance change is
                        // tracked (only ever on a verified frame).
                        for (size_t i = 0; i < d_->bank.size(); ++i)
                            d_->bank[i] =
                                0.9f * d_->bank[i] + 0.1f * d_->embBuf[i];
                        Impl::normalize(d_->bank);
                        // HiFT template refresh — confident AND verified only.
                        if (d_->cfg.templateRefresh &&
                            conf >= d_->cfg.refreshMinConf &&
                            ++d_->sinceRefresh >= d_->cfg.refreshEvery)
                        {
                            d_->sinceRefresh = 0;
                            d_->refreshTemplate(frame);
                        }
                    }
                }
            }
        }

        d_->params.detectionProbability = conf;

        if (ok && d_->vetoStreak < d_->cfg.maxVetoStreak)
        {
            d_->mode = MODE_TRACKING;
            d_->lostCounter = 0;
        }
        else
        {
            d_->mode = MODE_LOST;
            ++d_->lostCounter;

            // Active re-detection: scan the frame with HiFT+DINOv2 to re-find the
            // target (occlusion / re-entry), not just re-search the last spot.
            const bool verifierOn = d_->extractor && !d_->bank.empty();
            bool reacquired = false;
            if (d_->cfg.redetect && verifierOn)
                reacquired = d_->tryRedetect(frame);

            if (reacquired)
            {
                d_->mode = MODE_TRACKING;
                d_->params.detectionProbability = d_->cfg.reacquireThreshold;
            }
            else
            {
                if (d_->params.lostModeOption != 0)
                {
                    d_->cx += d_->velX;
                    d_->cy += d_->velY;
                }
                // Keep scanning longer when we CAN re-detect; otherwise fall
                // back to the user's LOST budget.
                const int budget = verifierOn ? d_->cfg.redetectMaxFrames
                                              : d_->params.maxFramesInLostMode;
                if (d_->lostCounter > budget)
                {
                    d_->mode = MODE_FREE;
                    d_->haveTemplate = false;
                    d_->lostCounter = 0;
                    d_->vetoStreak = 0;
                }
            }
        }
    }
    return true;
}

void HiFTTracker::getImage(int /*type*/, cr::video::Frame& image)
{
    // Provide a small GRAY preview of the template crop (green channel proxy).
    std::lock_guard<std::mutex> lk(d_->mtx);
    const int S = 128;
    if (image.data == nullptr || image.width != S || image.height != S ||
        image.fourcc != cr::video::Fourcc::GRAY)
        image.init(S, S, cr::video::Fourcc::GRAY);
    if (!d_->haveTemplate || d_->templateCrop.empty())
    {
        std::memset(image.data, 0, (size_t)S * S);
        return;
    }
    const int E = HIFT_EXEMPLAR;
    const float* Gp = d_->templateCrop.data() + E * E;  // green plane
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
        {
            const int sx = x * E / S;
            const int sy = y * E / S;
            float v = Gp[sy * E + sx];
            v = std::min(255.0f, std::max(0.0f, v));
            image.data[y * S + x] = (uint8_t)v;
        }
}

bool HiFTTracker::decodeAndExecuteCommand(uint8_t* data, int size)
{
    VTrackerParam pid;
    VTrackerCommand cid;
    float v1, v2, v3;
    const int kind = cr::vtracker::VTracker::decodeCommand(data, size, pid, cid,
                                                           v1, v2, v3);
    if (kind == 0)
        return executeCommand(cid, v1, v2, v3);
    if (kind == 1)
        return setParam(pid, v1);
    return false;
}

// ── Handoff template transfer ────────────────────────────────────────────────
namespace {
constexpr uint32_t HIFT_TPL_MAGIC = 0x48544631;  // 'HTF1'
}

bool HiFTTracker::exportTemplate(std::vector<unsigned char>& out) const
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    if (!d_->haveTemplate || d_->templateCrop.empty())
        return false;
    const size_t bytes = d_->templateCrop.size() * sizeof(float);
    out.resize(sizeof(uint32_t) + bytes);
    std::memcpy(out.data(), &HIFT_TPL_MAGIC, sizeof(uint32_t));
    std::memcpy(out.data() + sizeof(uint32_t), d_->templateCrop.data(), bytes);
    return true;
}

bool HiFTTracker::importTemplate(const std::vector<unsigned char>& in)
{
    std::lock_guard<std::mutex> lk(d_->mtx);
    const size_t need = 3 * HIFT_EXEMPLAR * HIFT_EXEMPLAR;
    if (in.size() != sizeof(uint32_t) + need * sizeof(float))
        return false;
    uint32_t magic = 0;
    std::memcpy(&magic, in.data(), sizeof(uint32_t));
    if (magic != HIFT_TPL_MAGIC)
        return false;
    d_->templateCrop.resize(need);
    std::memcpy(d_->templateCrop.data(), in.data() + sizeof(uint32_t),
                need * sizeof(float));
    if (!d_->engine.setTemplate(d_->templateCrop.data()))
        return false;
    d_->haveTemplate = true;
    d_->mode = MODE_TRACKING;
    d_->lostCounter = 0;
    return true;
}

}  // namespace hift
}  // namespace cr
