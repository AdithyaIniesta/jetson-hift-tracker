#include "cvtracker/CvTracker.h"
#include "cvtracker/CvTrackerVersion.h"
#include "CorrelationCore.h"
#include "Fft.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace cr {
namespace vtracker {

namespace {

constexpr int MODE_FREE = 0;
constexpr int MODE_TRACKING = 1;
constexpr int MODE_LOST = 2;
constexpr int MODE_INERTIAL = 3;
constexpr int MODE_STATIC = 4;

constexpr float DEFAULT_LOSS_THRESHOLD = 0.1f;
constexpr float DEFAULT_RECAPTURE_THRESHOLD = 0.4f;
constexpr float DEFAULT_LEARNING_RATE = 0.075f;

int nextPow2(int v)
{
    int p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

} // namespace

/// Buffered video frame (luma only).
struct BufFrame
{
    int id{-1};
    int width{0};
    int height{0};
    std::vector<uint8_t> luma;
};

struct CvTracker::Impl
{
    std::mutex mutex;
    VTrackerParams params;
    bool initialized{false};

    std::unique_ptr<FftBackend> fft;
    CorrelationCore core;

    // Frame buffer (ring).
    std::deque<BufFrame> buffer;
    int autoFrameId{0};
    long long lastProcessedId{-1000000000LL};

    // Processing window geometry.
    int fftW{256};
    int fftH{256};
    float stepX{1.0f}; // frame pixels per window pixel (horizontal)
    float stepY{1.0f};

    // Tracking state.
    float posX{0.0f}; // tracking rectangle center, subpixel
    float posY{0.0f};
    float scale{1.0f};       // current object scale relative to capture
    float baseRectW{72.0f};  // rectangle size at capture
    float baseRectH{72.0f};
    int scaleProbeCounter{0};
    bool searchWindowOverride{false};
    float overrideX{0.0f};
    float overrideY{0.0f};
    bool adjustSizeOnce{false};
    bool adjustPositionOnce{false};
    bool pendingRetrain{false};
    // Smoothed object estimate (EMA). Offsets are relative to the tracking
    // rectangle center, sizes in frame pixels.
    float objOffX{0.0f};
    float objOffY{0.0f};
    float objWf{72.0f};
    float objHf{72.0f};

    std::vector<float> window; // work buffer

    // ---------------------------------------------------------------------

    bool extractLuma(const cr::video::Frame& frame, BufFrame& out) const
    {
        const int w = frame.width;
        const int h = frame.height;
        if (w <= 0 || h <= 0 || frame.data == nullptr)
            return false;
        if (frame.size < cr::video::Frame::dataSize(w, h, frame.fourcc) ||
            cr::video::Frame::dataSize(w, h, frame.fourcc) == 0)
            return false;

        out.width = w;
        out.height = h;
        out.luma.resize((size_t)w * h);
        const uint8_t* src = frame.data;
        uint8_t* dst = out.luma.data();
        const size_t n = (size_t)w * h;

        switch (frame.fourcc)
        {
        case cr::video::Fourcc::GRAY:
        case cr::video::Fourcc::NV12:
        case cr::video::Fourcc::NV21:
        case cr::video::Fourcc::YU12:
        case cr::video::Fourcc::YV12:
            // Y plane first.
            std::memcpy(dst, src, n);
            return true;
        case cr::video::Fourcc::YUV24:
            for (size_t i = 0; i < n; ++i)
                dst[i] = src[i * 3];
            return true;
        case cr::video::Fourcc::YUYV:
            for (size_t i = 0; i < n; ++i)
                dst[i] = src[i * 2];
            return true;
        case cr::video::Fourcc::UYVY:
            for (size_t i = 0; i < n; ++i)
                dst[i] = src[i * 2 + 1];
            return true;
        default:
            return false;
        }
    }

    /// Extract search window centered at (cx, cy), covering
    /// (searchWindowWidth * scl) x (searchWindowHeight * scl) frame pixels,
    /// resampled (bilinear) to fftW x fftH. Out-of-frame samples are clamped
    /// to the frame edge.
    void extractWindow(const BufFrame& f, float cx, float cy, float scl,
                       std::vector<float>& out) const
    {
        out.resize((size_t)fftW * fftH);
        const float sx = stepX * scl;
        const float sy = stepY * scl;
        const float x0 = cx - sx * (fftW / 2);
        const float y0 = cy - sy * (fftH / 2);
        for (int y = 0; y < fftH; ++y)
        {
            float fy = y0 + sy * y;
            fy = std::min(std::max(fy, 0.0f), (float)f.height - 1.001f);
            const int iy = (int)fy;
            const float wy = fy - iy;
            const uint8_t* row0 = &f.luma[(size_t)iy * f.width];
            const uint8_t* row1 = &f.luma[(size_t)std::min(iy + 1, f.height - 1) * f.width];
            for (int x = 0; x < fftW; ++x)
            {
                float fx = x0 + sx * x;
                fx = std::min(std::max(fx, 0.0f), (float)f.width - 1.001f);
                const int ix = (int)fx;
                const float wx = fx - ix;
                const int ix1 = std::min(ix + 1, f.width - 1);
                const float v =
                    row0[ix] * (1 - wx) * (1 - wy) + row0[ix1] * wx * (1 - wy) +
                    row1[ix] * (1 - wx) * wy + row1[ix1] * wx * wy;
                out[(size_t)y * fftW + x] = v;
            }
        }
    }

    void setupGeometry()
    {
        fftW = std::min(std::max(nextPow2(params.searchWindowWidth), 64), 512);
        fftH = std::min(std::max(nextPow2(params.searchWindowHeight), 64), 512);
        stepX = (float)params.searchWindowWidth / (float)fftW;
        stepY = (float)params.searchWindowHeight / (float)fftH;
        core.init(fftW, fftH, fft.get());
    }

    float lossThreshold() const
    {
        return params.custom1 > 0.0f ? params.custom1 : DEFAULT_LOSS_THRESHOLD;
    }
    float recaptureThreshold() const
    {
        return params.custom2 > 0.0f ? params.custom2
                                     : DEFAULT_RECAPTURE_THRESHOLD;
    }
    float learningRate() const
    {
        return params.custom3 > 0.0f ? params.custom3 : DEFAULT_LEARNING_RATE;
    }

    BufFrame* findFrame(int id)
    {
        for (auto& f : buffer)
            if (f.id == id)
                return &f;
        return nullptr;
    }

    void resetToFree()
    {
        pendingRetrain = false;
        params.mode = MODE_FREE;
        params.lostModeFrameCounter = 0;
        params.frameCounter = 0;
        params.velX = 0.0f;
        params.velY = 0.0f;
        params.detectionProbability = 0.0f;
        scale = 1.0f;
    }

    bool captureOnFrame(const BufFrame& f, float cx, float cy)
    {
        if (f.width <= 0)
            return false;
        cx = std::min(std::max(cx, 0.0f), (float)f.width - 1.0f);
        cy = std::min(std::max(cy, 0.0f), (float)f.height - 1.0f);

        setupGeometry();
        scale = 1.0f;
        baseRectW = (float)params.rectWidth;
        baseRectH = (float)params.rectHeight;
        core.setObjectSize(baseRectW / stepX, baseRectH / stepY);

        extractWindow(f, cx, cy, 1.0f, window);
        core.capture(window);

        posX = cx;
        posY = cy;
        params.rectX = (int)std::lround(cx);
        params.rectY = (int)std::lround(cy);
        params.objectX = params.rectX;
        params.objectY = params.rectY;
        params.objectWidth = params.rectWidth;
        params.objectHeight = params.rectHeight;
        objOffX = 0.0f;
        objOffY = 0.0f;
        objWf = (float)params.rectWidth;
        objHf = (float)params.rectHeight;
        params.velX = 0.0f;
        params.velY = 0.0f;
        params.frameCounter = 0;
        params.lostModeFrameCounter = 0;
        params.detectionProbability = 1.0f;
        params.mode = MODE_TRACKING;
        scaleProbeCounter = 0;
        pendingRetrain = false;
        return true;
    }

    bool centerTouchesEdge() const
    {
        return params.rectX <= 0 || params.rectY <= 0 ||
               params.rectX >= params.frameWidth - 1 ||
               params.rectY >= params.frameHeight - 1;
    }

    void clampPos()
    {
        posX = std::min(std::max(posX, 0.0f), (float)params.frameWidth - 1.0f);
        posY = std::min(std::max(posY, 0.0f), (float)params.frameHeight - 1.0f);
        params.rectX = (int)std::lround(posX);
        params.rectY = (int)std::lround(posY);
    }

    /// Estimate object position / size from the reference image mask.
    /// Moment based (robust to single noisy mask pixels), updated only on
    /// confident frames, smoothed with EMA - the estimate moves slowly and
    /// monotonically instead of jumping with per-frame mask noise.
    void estimateObject(float prob)
    {
        std::vector<uint8_t> mask;
        core.objectMask(mask);

        // Restrict to a neighborhood of the rectangle around center.
        const float rw = baseRectW * scale / stepX; // rect in window pixels
        const float rh = baseRectH * scale / stepY;
        const int cx = fftW / 2;
        const int cy = fftH / 2;
        const int hx = std::min(fftW / 2 - 1, (int)(0.9f * rw));
        const int hy = std::min(fftH / 2 - 1, (int)(0.9f * rh));

        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0;
        long cnt = 0;
        for (int y = cy - hy; y <= cy + hy; ++y)
        {
            for (int x = cx - hx; x <= cx + hx; ++x)
            {
                if (mask[(size_t)y * fftW + x] == 0)
                    continue;
                sx += x;
                sy += y;
                sxx += (double)x * x;
                syy += (double)y * y;
                ++cnt;
            }
        }

        // Update the smoothed estimate only with enough mask evidence on a
        // confident frame; otherwise hold the previous (stable) estimate.
        if (cnt >= 30 && prob >= recaptureThreshold())
        {
            const float mx = (float)(sx / cnt);
            const float my = (float)(sy / cnt);
            const float stdX =
                (float)std::sqrt(std::max(0.0, sxx / cnt - (sx / cnt) * (sx / cnt)));
            const float stdY =
                (float)std::sqrt(std::max(0.0, syy / cnt - (sy / cnt) * (sy / cnt)));
            // Extent of a uniform distribution = sqrt(12) * std.
            const float wEst = 3.46f * stdX * stepX * scale;
            const float hEst = 3.46f * stdY * stepY * scale;
            const float oxOff = (mx - cx) * stepX * scale;
            const float oyOff = (my - cy) * stepY * scale;
            const float wMax = 2.0f * baseRectW * scale;
            const float hMax = 2.0f * baseRectH * scale;

            objOffX = 0.85f * objOffX + 0.15f * oxOff;
            objOffY = 0.85f * objOffY + 0.15f * oyOff;
            objWf = 0.9f * objWf +
                    0.1f * std::min(std::max(wEst, 8.0f), wMax);
            objHf = 0.9f * objHf +
                    0.1f * std::min(std::max(hEst, 8.0f), hMax);
        }

        params.objectX = (int)std::lround(posX + objOffX);
        params.objectY = (int)std::lround(posY + objOffY);
        params.objectWidth = (int)std::lround(objWf);
        params.objectHeight = (int)std::lround(objHf);
    }

    void applyAutoAdjust()
    {
        if (adjustSizeOnce)
        {
            // One-shot: snap to the (smoothed) object estimate and request
            // a consistent re-train of the filter at the new size.
            params.rectWidth = std::max(16, (int)std::lround(objWf));
            params.rectHeight = std::max(16, (int)std::lround(objHf));
            pendingRetrain = true;
            adjustSizeOnce = false;
        }
        else if (params.rectAutoSize)
        {
            // Continuous: deadband of 15% suppresses estimate noise, then
            // approach slowly (5% per frame) - no erratic jumps.
            if (std::fabs(objWf - (float)params.rectWidth) >
                0.15f * (float)params.rectWidth)
                params.rectWidth = std::max(
                    16, (int)std::lround(0.95f * params.rectWidth +
                                         0.05f * objWf));
            if (std::fabs(objHf - (float)params.rectHeight) >
                0.15f * (float)params.rectHeight)
                params.rectHeight = std::max(
                    16, (int)std::lround(0.95f * params.rectHeight +
                                         0.05f * objHf));
        }

        if (adjustPositionOnce)
        {
            posX = posX + objOffX;
            posY = posY + objOffY;
            objOffX = 0.0f;
            objOffY = 0.0f;
            clampPos();
            adjustPositionOnce = false;
        }
        else if (params.rectAutoPosition)
        {
            posX += 0.2f * objOffX;
            posY += 0.2f * objOffY;
            objOffX *= 0.8f;
            objOffY *= 0.8f;
            clampPos();
        }

    }

    /// Search window center used for processing of this frame.
    void searchCenter(float& cx, float& cy)
    {
        if (searchWindowOverride)
        {
            cx = overrideX;
            cy = overrideY;
            searchWindowOverride = false;
        }
        else
        {
            cx = posX;
            cy = posY;
        }
        params.searchWindowX = (int)std::lround(cx);
        params.searchWindowY = (int)std::lround(cy);
    }

    void processBufferedFrame(const BufFrame& f)
    {
        params.frameWidth = f.width;
        params.frameHeight = f.height;

        switch (params.mode)
        {
        case MODE_FREE:
        case MODE_STATIC:
            break;
        case MODE_INERTIAL:
        {
            posX += params.velX;
            posY += params.velY;
            clampPos();
            if (centerTouchesEdge())
                resetToFree();
            break;
        }
        case MODE_TRACKING:
            trackingStep(f);
            break;
        case MODE_LOST:
            lostStep(f);
            break;
        default:
            break;
        }
        params.processedFrameId = f.id;
    }

    void trackingStep(const BufFrame& f)
    {
        float cx, cy;
        searchCenter(cx, cy);

        extractWindow(f, cx, cy, scale, window);
        CorrelationCore::Result r = core.detect(window);
        params.detectionProbability = r.prob;
        params.frameCounter++;

        if (r.prob < lossThreshold())
        {
            // Object lost: keep last good position / velocity, predict.
            params.mode = MODE_LOST;
            params.lostModeFrameCounter = 0;
            return;
        }

        // New position.
        const float nx = cx + r.dx * stepX * scale;
        const float ny = cy + r.dy * stepY * scale;
        const float dxF = nx - posX;
        const float dyF = ny - posY;
        posX = nx;
        posY = ny;

        // Velocity (exponential smoothing, pixels per frame).
        params.velX = 0.6f * params.velX + 0.4f * dxF;
        params.velY = 0.6f * params.velY + 0.4f * dyF;

        clampPos();
        if (centerTouchesEdge() && params.lostModeOption == 2)
        {
            resetToFree();
            return;
        }

        // Scale probe: every 4th frame test slightly smaller / larger scale.
        if (++scaleProbeCounter >= 4)
        {
            scaleProbeCounter = 0;
            static const float probes[2] = {0.98f, 1.02f};
            float bestScale = 1.0f;
            float bestPeak = r.peak * 1.02f; // bias to current scale
            for (float s : probes)
            {
                extractWindow(f, posX, posY, scale * s, window);
                CorrelationCore::Result rs = core.detect(window);
                if (rs.peak > bestPeak)
                {
                    bestPeak = rs.peak;
                    bestScale = s;
                }
            }
            if (bestScale != 1.0f)
                scale = std::min(std::max(scale * bestScale, 0.2f), 5.0f);
        }

        // Update pattern at the new (re-centered) position. Update rate is
        // weighted by detection confidence: no update during occlusions.
        extractWindow(f, posX, posY, scale, window);
        float rate = 0.0f;
        if (r.prob >= recaptureThreshold())
            rate = learningRate();
        else
            rate = learningRate() * r.prob;
        core.update(window, rate);

        estimateObject(r.prob);
        applyAutoAdjust();

        if (pendingRetrain && r.prob >= 0.6f)
        {
            pendingRetrain = false;
            baseRectW = (float)params.rectWidth / scale;
            baseRectH = (float)params.rectHeight / scale;
            core.setObjectSize(baseRectW / stepX, baseRectH / stepY);
            extractWindow(f, posX, posY, scale, window);
            core.capture(window);
        }
    }

    void lostStep(const BufFrame& f)
    {
        params.lostModeFrameCounter++;

        // Update coordinates according to LOST mode option.
        if (params.lostModeOption == 1 || params.lostModeOption == 2)
        {
            posX += params.velX;
            posY += params.velY;
            const bool touched =
                posX <= 0.0f || posY <= 0.0f ||
                posX >= (float)params.frameWidth - 1.0f ||
                posY >= (float)params.frameHeight - 1.0f;
            clampPos();
            if (touched && params.lostModeOption == 2)
            {
                resetToFree();
                return;
            }
        }

        // Try to re-detect object in the search window.
        float cx, cy;
        searchCenter(cx, cy);
        extractWindow(f, cx, cy, scale, window);
        CorrelationCore::Result r = core.detect(window);
        params.detectionProbability = r.prob;

        if (r.prob > recaptureThreshold())
        {
            // Automatic re-capture.
            posX = cx + r.dx * stepX * scale;
            posY = cy + r.dy * stepY * scale;
            clampPos();
            params.mode = MODE_TRACKING;
            params.lostModeFrameCounter = 0;
            return;
        }

        if (params.lostModeFrameCounter >= params.maxFramesInLostMode)
            resetToFree();
    }
};

// ---------------------------------------------------------------------------

CvTracker::CvTracker() : m_impl(new Impl)
{
    m_impl->fft = createFftBackend();
    m_impl->setupGeometry();
}

CvTracker::~CvTracker() = default;

std::string CvTracker::getVersion()
{
    return CVTRACKER_VERSION;
}

bool CvTracker::initVTracker(VTrackerParams& params)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (params.searchWindowWidth < 16 || params.searchWindowHeight < 16 ||
        params.rectWidth < 4 || params.rectHeight < 4 ||
        params.frameBufferSize < 1 || params.maxFramesInLostMode < 1 ||
        params.lostModeOption < 0 || params.lostModeOption > 2)
        return false;

    // Copy user-settable fields, reset state fields.
    VTrackerParams& p = m_impl->params;
    p = params;
    p.mode = MODE_FREE;
    p.lostModeFrameCounter = 0;
    p.frameCounter = 0;
    p.detectionProbability = 0.0f;
    p.velX = 0.0f;
    p.velY = 0.0f;
    p.processingTimeMks = 0;

    m_impl->fft->setMultiThread(p.multipleThreads);
    m_impl->setupGeometry();
    m_impl->buffer.clear();
    m_impl->initialized = true;
    return true;
}

bool CvTracker::setParam(VTrackerParam id, float value)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    VTrackerParams& p = m_impl->params;
    switch (id)
    {
    case VTrackerParam::SEARCH_WINDOW_WIDTH:
        if (value < 16.0f)
            return false;
        p.searchWindowWidth = (int)value;
        return true;
    case VTrackerParam::SEARCH_WINDOW_HEIGHT:
        if (value < 16.0f)
            return false;
        p.searchWindowHeight = (int)value;
        return true;
    case VTrackerParam::RECT_WIDTH:
        if (value < 4.0f)
            return false;
        p.rectWidth = (int)value;
        return true;
    case VTrackerParam::RECT_HEIGHT:
        if (value < 4.0f)
            return false;
        p.rectHeight = (int)value;
        return true;
    case VTrackerParam::LOST_MODE_OPTION:
        if (value < 0.0f || value > 2.0f)
            return false;
        p.lostModeOption = (int)value;
        return true;
    case VTrackerParam::FRAME_BUFFER_SIZE:
        if (value < 1.0f)
            return false;
        p.frameBufferSize = (int)value;
        while ((int)m_impl->buffer.size() > p.frameBufferSize)
            m_impl->buffer.pop_front();
        return true;
    case VTrackerParam::MAX_FRAMES_IN_LOST_MODE:
        if (value < 1.0f)
            return false;
        p.maxFramesInLostMode = (int)value;
        return true;
    case VTrackerParam::RECT_AUTO_SIZE:
        p.rectAutoSize = value != 0.0f;
        if (!p.rectAutoSize)
            m_impl->adjustSizeOnce = false;
        return true;
    case VTrackerParam::RECT_AUTO_POSITION:
        p.rectAutoPosition = value != 0.0f;
        if (!p.rectAutoPosition)
            m_impl->adjustPositionOnce = false;
        return true;
    case VTrackerParam::MULTIPLE_THREADS:
        p.multipleThreads = value != 0.0f;
        m_impl->fft->setMultiThread(p.multipleThreads);
        return true;
    case VTrackerParam::NUM_CHANNELS:
        p.numChannels = (int)value;
        return true;
    case VTrackerParam::TYPE:
        p.type = (int)value;
        return true;
    case VTrackerParam::CUSTOM_1:
        p.custom1 = value;
        return true;
    case VTrackerParam::CUSTOM_2:
        p.custom2 = value;
        return true;
    case VTrackerParam::CUSTOM_3:
        p.custom3 = value;
        return true;
    default:
        return false;
    }
}

float CvTracker::getParam(VTrackerParam id)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const VTrackerParams& p = m_impl->params;
    switch (id)
    {
    case VTrackerParam::SEARCH_WINDOW_WIDTH: return (float)p.searchWindowWidth;
    case VTrackerParam::SEARCH_WINDOW_HEIGHT: return (float)p.searchWindowHeight;
    case VTrackerParam::RECT_WIDTH: return (float)p.rectWidth;
    case VTrackerParam::RECT_HEIGHT: return (float)p.rectHeight;
    case VTrackerParam::LOST_MODE_OPTION: return (float)p.lostModeOption;
    case VTrackerParam::FRAME_BUFFER_SIZE: return (float)p.frameBufferSize;
    case VTrackerParam::MAX_FRAMES_IN_LOST_MODE: return (float)p.maxFramesInLostMode;
    case VTrackerParam::RECT_AUTO_SIZE: return p.rectAutoSize ? 1.0f : 0.0f;
    case VTrackerParam::RECT_AUTO_POSITION: return p.rectAutoPosition ? 1.0f : 0.0f;
    case VTrackerParam::MULTIPLE_THREADS: return p.multipleThreads ? 1.0f : 0.0f;
    case VTrackerParam::NUM_CHANNELS: return (float)p.numChannels;
    case VTrackerParam::TYPE: return (float)p.type;
    case VTrackerParam::CUSTOM_1: return p.custom1;
    case VTrackerParam::CUSTOM_2: return p.custom2;
    case VTrackerParam::CUSTOM_3: return p.custom3;
    default: return -1.0f;
    }
}

void CvTracker::getParams(VTrackerParams& params)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    params = m_impl->params;
}

bool CvTracker::executeCommand(VTrackerCommand id, float arg1, float arg2,
                               float arg3)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    Impl& d = *m_impl;
    VTrackerParams& p = d.params;

    switch (id)
    {
    case VTrackerCommand::CAPTURE:
    {
        if (d.buffer.empty())
            return false;
        BufFrame* f = nullptr;
        if (arg3 >= 0.0f)
            f = d.findFrame((int)arg3);
        else
            f = &d.buffer.back();
        if (f == nullptr)
            return false;
        const float cx = arg1 >= 0.0f ? arg1 : (float)f->width * 0.5f;
        const float cy = arg2 >= 0.0f ? arg2 : (float)f->height * 0.5f;
        if (!d.captureOnFrame(*f, cx, cy))
            return false;
        // Mark catch-up start: buffered frames after this one will be
        // processed on the next processFrame() call.
        d.lastProcessedId = (long long)f->id;
        p.processedFrameId = f->id;
        return true;
    }
    case VTrackerCommand::CAPTURE_PERCENTS:
    {
        if (d.buffer.empty())
            return false;
        BufFrame& f = d.buffer.back();
        const float cx = arg1 * 0.01f * (float)f.width;
        const float cy = arg2 * 0.01f * (float)f.height;
        if (!d.captureOnFrame(f, cx, cy))
            return false;
        d.lastProcessedId = (long long)f.id;
        p.processedFrameId = f.id;
        return true;
    }
    case VTrackerCommand::RESET:
        d.resetToFree();
        return true;
    case VTrackerCommand::SET_INERTIAL_MODE:
        if (p.mode == MODE_TRACKING || p.mode == MODE_LOST ||
            p.mode == MODE_STATIC)
        {
            p.mode = MODE_INERTIAL;
            return true;
        }
        return false;
    case VTrackerCommand::SET_LOST_MODE:
        if (p.mode == MODE_TRACKING || p.mode == MODE_INERTIAL ||
            p.mode == MODE_STATIC)
        {
            p.mode = MODE_LOST;
            p.lostModeFrameCounter = 0;
            return true;
        }
        return false;
    case VTrackerCommand::SET_STATIC_MODE:
        if (p.mode == MODE_TRACKING || p.mode == MODE_LOST ||
            p.mode == MODE_INERTIAL)
        {
            p.mode = MODE_STATIC;
            return true;
        }
        return false;
    case VTrackerCommand::ADJUST_RECT_SIZE:
        d.adjustSizeOnce = true;
        return true;
    case VTrackerCommand::ADJUST_RECT_POSITION:
        d.adjustPositionOnce = true;
        return true;
    case VTrackerCommand::MOVE_RECT:
        d.posX += arg1;
        d.posY += arg2;
        d.clampPos();
        return true;
    case VTrackerCommand::SET_RECT_POSITION:
        if (p.mode != MODE_FREE)
            return false;
        d.posX = arg1;
        d.posY = arg2;
        d.clampPos();
        return true;
    case VTrackerCommand::SET_RECT_POSITION_PERCENTS:
        if (p.mode != MODE_FREE || p.frameWidth <= 0)
            return false;
        d.posX = arg1 * 0.01f * (float)p.frameWidth;
        d.posY = arg2 * 0.01f * (float)p.frameHeight;
        d.clampPos();
        return true;
    case VTrackerCommand::MOVE_SEARCH_WINDOW:
        d.searchWindowOverride = true;
        d.overrideX = (float)p.searchWindowX + arg1;
        d.overrideY = (float)p.searchWindowY + arg2;
        return true;
    case VTrackerCommand::SET_SEARCH_WINDOW_POSITION:
        d.searchWindowOverride = true;
        d.overrideX = arg1;
        d.overrideY = arg2;
        return true;
    case VTrackerCommand::SET_SEARCH_WINDOW_POSITION_PERCENTS:
        if (p.frameWidth <= 0)
            return false;
        d.searchWindowOverride = true;
        d.overrideX = arg1 * 0.01f * (float)p.frameWidth;
        d.overrideY = arg2 * 0.01f * (float)p.frameHeight;
        return true;
    case VTrackerCommand::CHANGE_RECT_SIZE:
        p.rectWidth = std::max(4, p.rectWidth + (int)arg1);
        p.rectHeight = std::max(4, p.rectHeight + (int)arg2);
        return true;
    default:
        return false;
    }
}

bool CvTracker::processFrame(cr::video::Frame& frame)
{
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    Impl& d = *m_impl;
    VTrackerParams& p = d.params;

    // Frame size change: reset tracker state and frame buffer.
    if (p.frameWidth > 0 &&
        (frame.width != p.frameWidth || frame.height != p.frameHeight))
    {
        d.resetToFree();
        d.buffer.clear();
    }

    // Add frame to the ring buffer.
    BufFrame bf;
    if (!d.extractLuma(frame, bf))
        return false;
    bf.id = frame.frameId >= 0 ? frame.frameId : d.autoFrameId;
    d.autoFrameId = bf.id + 1;
    d.buffer.push_back(std::move(bf));
    while ((int)d.buffer.size() > std::max(1, p.frameBufferSize))
        d.buffer.pop_front();
    p.frameId = d.buffer.back().id;
    p.frameWidth = frame.width;
    p.frameHeight = frame.height;

    // Process all not yet processed frames in the buffer (normally one;
    // multiple after a capture on a past frame - catch-up processing).
    auto it = d.buffer.end();
    for (auto i = d.buffer.begin(); i != d.buffer.end(); ++i)
    {
        if ((long long)i->id == d.lastProcessedId)
        {
            it = i;
            break;
        }
    }
    if (it == d.buffer.end())
    {
        // Last processed frame is not in the buffer: process newest frame.
        d.processBufferedFrame(d.buffer.back());
    }
    else
    {
        for (++it; it != d.buffer.end(); ++it)
            d.processBufferedFrame(*it);
    }
    d.lastProcessedId = (long long)d.buffer.back().id;
    p.processedFrameId = d.buffer.back().id;

    p.processingTimeMks = (int)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
    return true;
}

void CvTracker::getImage(int type, cr::video::Frame& image)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    Impl& d = *m_impl;
    if (image.width != d.fftW || image.height != d.fftH ||
        image.fourcc != cr::video::Fourcc::GRAY || image.data == nullptr)
        image.init(d.fftW, d.fftH, cr::video::Fourcc::GRAY);

    const size_t n = (size_t)d.fftW * d.fftH;
    switch (type)
    {
    case 0: // Reference image of the object.
    {
        const std::vector<float>& ref = d.core.referenceImage();
        for (size_t i = 0; i < n; ++i)
            image.data[i] =
                (uint8_t)std::min(std::max(ref[i], 0.0f), 255.0f);
        break;
    }
    case 1: // Object mask image.
    {
        std::vector<uint8_t> mask;
        d.core.objectMask(mask);
        std::memcpy(image.data, mask.data(), n);
        break;
    }
    case 2: // Correlation surface.
    {
        const std::vector<float>& resp = d.core.responseSurface();
        float mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < n; ++i)
        {
            mn = std::min(mn, resp[i]);
            mx = std::max(mx, resp[i]);
        }
        const float k = mx > mn ? 255.0f / (mx - mn) : 0.0f;
        for (size_t i = 0; i < n; ++i)
            image.data[i] = (uint8_t)((resp[i] - mn) * k);
        break;
    }
    default:
        std::memset(image.data, 0, n);
        break;
    }
}

} // namespace vtracker
} // namespace cr
