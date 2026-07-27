#include "hift/HiFTTracker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

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
constexpr float SIZE_MIN = 0.4f;
constexpr float SIZE_MAX = 2.5f;

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
            pscore = pscore * (1.0f - HIFT_WINDOW_INF) +
                     window[j] * HIFT_WINDOW_INF;

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
        newW = std::max(SIZE_MIN * bw0, std::min(newW, SIZE_MAX * bw0));
        newH = std::max(SIZE_MIN * bh0, std::min(newH, SIZE_MAX * bh0));
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
    d_->engineReady = d_->engine.initialise(d_->cfg.trt);
    if (!d_->engineReady)
        std::fprintf(stderr, "[hift] engine init FAILED — tracker inert\n");
}

HiFTTracker::~HiFTTracker() = default;

std::string HiFTTracker::getVersion() { return "hift-1.0.0"; }
bool HiFTTracker::ready() const { return d_->engineReady; }

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
        const float score = d_->trackStep(frame);
        d_->params.detectionProbability = score;
        if (score >= d_->cfg.lossThreshold)
        {
            d_->mode = MODE_TRACKING;
            d_->lostCounter = 0;
        }
        else
        {
            d_->mode = MODE_LOST;
            ++d_->lostCounter;
            // Inertial coast while lost (lostModeOption 1/2).
            if (d_->params.lostModeOption != 0)
            {
                d_->cx += d_->velX;
                d_->cy += d_->velY;
            }
            if (d_->lostCounter > d_->params.maxFramesInLostMode)
            {
                d_->mode = MODE_FREE;
                d_->haveTemplate = false;
                d_->lostCounter = 0;
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
