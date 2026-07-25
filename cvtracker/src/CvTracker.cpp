#include "cvtracker/CvTracker.h"
#include "cvtracker/CvTrackerVersion.h"
#include "CorrelationCore.h"
#include "Ekf.h"
#include "Fft.h"
#include "TemplateBank.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <utility>
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

// EKF process / measurement noise. Tuned for pixel-scale tracking at ~30 FPS.
// sigma_a: linear acceleration std-dev (px/frame^2).
// sigma_alpha: angular acceleration std-dev (rad/frame^2).
// r_base: position measurement variance (px^2), divided by detection
// probability at update time.
constexpr float EKF_SIGMA_A     = 1.5f;
constexpr float EKF_SIGMA_ALPHA = 0.15f;
constexpr float EKF_R_BASE      = 4.0f;

// LOST-mode scan: off-center probe detections per frame (in addition to
// the primary detect at the search window center). Higher = faster full
// window sweep, more per-frame cost while lost.
constexpr int LOST_PROBES_PER_FRAME = 2;

// Re-capture acceptance discipline. Correlation filters discriminate
// same-class objects poorly, so a full-window scan with honest
// (attenuation-compensated) scores will also score look-alike distractors
// well. Two gates keep false re-locks out:
// - Physics reach: plausible displacement from the (predicted) loss
//   position grows with time lost; candidates beyond it are ignored (the
//   radius keeps growing, so a genuinely far re-appearance is accepted
//   later). reach = REACH_BASE_FACTOR * rect + (speed + REACH_GROWTH) * t.
// - Confirmation: the same location must clear all gates on
//   RECAPTURE_CONFIRM_FRAMES consecutive frames before the lock commits
//   (single-frame correlation flukes never re-lock).
constexpr int RECAPTURE_CONFIRM_FRAMES = 2;
constexpr float RECAPTURE_CONFIRM_RADIUS = 0.75f; // x max(rectW, rectH)
constexpr float REACH_BASE_FACTOR = 1.25f;        // x max(rectW, rectH)
constexpr float REACH_GROWTH_PX_PER_FRAME = 2.0f;

// DNN verifier constants.
// Consecutive failed verifications before the tracker is forced to LOST
// mode (single mismatches only block pattern learning).
constexpr int DNN_VETO_STREAK_TO_LOST = 3;
// Template bank: how many appearance embeddings to remember, and the
// minimum spacing (frames) between adds - diversity over redundancy.
constexpr int DNN_BANK_CAPACITY = 12;
constexpr int DNN_BANK_MIN_SPACING = 30;
// Context factor for the embedding crop: the patch covers the tracking
// rectangle scaled by this factor (a little background context helps
// discrimination).
constexpr float DNN_CROP_CONTEXT = 1.25f;

// DNN scale-pick: on each verify frame also embed the crop at these box-size
// multipliers and nudge the running scale toward the best appearance match
// (damped). Sizes the box from appearance, no learned regression head.
constexpr float SCALE_PICK_LO   = 0.90f;
constexpr float SCALE_PICK_HI   = 1.10f;
constexpr float SCALE_PICK_DAMP = 0.30f;   // move 30% toward the best size

// Secondary-peak search parameters. Used ONLY by the epipolar-band
// constraint fallback (lockon uses them additionally for DNN multi-peak
// arbitration; that feature is not ported here). A secondary peak
// must clear this fraction of the primary peak's magnitude to be
// considered, and we surface at most this many candidates.
constexpr float ARB_SECOND_PEAK_RATIO = 0.5f;
constexpr int   ARB_MAX_CANDIDATES    = 3;

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
    mutable std::mutex mutex;   // allow locking from const methods
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

    // Optional EKF predictive filter (CTRV motion model). Active only when
    // params.ekfEnabled is true. Outputs (filtered pos + velocity) replace
    // the raw correlation peak in objectX/Y and velX/velY, and drive
    // extrapolation during LOST/INERTIAL modes.
    Ekf ekf;

    // Optional DNN embedding verifier. Active only when an extractor is
    // set and params.dnnVerifierEnabled is true. Verifies at a fixed
    // cadence that the tracked patch still matches the captured object
    // (template bank, cosine similarity); sustained mismatch blocks
    // pattern learning and forces LOST mode, and LOST-mode re-captures
    // must pass verification before being accepted (distractor rejection).
    std::shared_ptr<FeatureExtractor> extractor;
    TemplateBank bank;
    int verifyCountdown{0};
    int vetoStreak{0};

    // Epipolar band constraint (set by setEpipolarConstraint). While
    // epiTtl > 0 the correlation peak selection prefers peaks whose
    // perpendicular distance to the line ax + by + c = 0 is <= epiHalfW.
    // epiTtl decrements once per processFrame call; hits 0 → inactive.
    // Same shape as the lockon engine's version so the app-side wiring
    // is identical across engines.
    float epiA{0.0f}, epiB{0.0f}, epiC{0.0f}, epiHalfW{0.0f};
    int   epiTtl{0};
    int probeIndex{0}; // LOST-mode scan grid cursor
    // Tentative re-capture candidate (multi-frame confirmation).
    float candX{0.0f};
    float candY{0.0f};
    int candStreak{0};
    int candMiss{0};
    std::vector<std::pair<float, float>> probeList; // nearest-first offsets
    float probeGeomSightX{-1.0f}; // geometry cache keys for probeList
    float probeGeomSightY{-1.0f};
    float probeGeomSwX{-1.0f};
    float probeGeomSwY{-1.0f};
    std::vector<float> patchBuf;  // embedding crop work buffer
    std::vector<float> embBuf;    // embedding work buffer

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

    /// Crop a wFrame x hFrame region centered at (cx, cy) from the frame,
    /// bilinearly resampled to outSize x outSize (for the embedding
    /// extractor). Out-of-frame samples clamp to the frame edge.
    void extractPatch(const BufFrame& f, float cx, float cy, float wFrame,
                      float hFrame, int outSize, std::vector<float>& out) const
    {
        out.resize((size_t)outSize * outSize);
        const float sx = wFrame / (float)outSize;
        const float sy = hFrame / (float)outSize;
        const float x0 = cx - wFrame * 0.5f;
        const float y0 = cy - hFrame * 0.5f;
        for (int y = 0; y < outSize; ++y)
        {
            float fy = y0 + sy * ((float)y + 0.5f);
            fy = std::min(std::max(fy, 0.0f), (float)f.height - 1.001f);
            const int iy = (int)fy;
            const float wy = fy - iy;
            const uint8_t* row0 = &f.luma[(size_t)iy * f.width];
            const uint8_t* row1 =
                &f.luma[(size_t)std::min(iy + 1, f.height - 1) * f.width];
            for (int x = 0; x < outSize; ++x)
            {
                float fx = x0 + sx * ((float)x + 0.5f);
                fx = std::min(std::max(fx, 0.0f), (float)f.width - 1.001f);
                const int ix = (int)fx;
                const float wx = fx - ix;
                const int ix1 = std::min(ix + 1, f.width - 1);
                out[(size_t)y * outSize + x] =
                    row0[ix] * (1 - wx) * (1 - wy) + row0[ix1] * wx * (1 - wy) +
                    row1[ix] * (1 - wx) * wy + row1[ix1] * wx * wy;
            }
        }
    }

    /// Crop the object region at (cx, cy) and run the embedding extractor.
    /// Returns false when no extractor is set or inference fails.
    bool computeEmbedding(const BufFrame& f, float cx, float cy,
                          std::vector<float>& emb, float scaleMul = 1.0f)
    {
        if (!extractor)
            return false;
        const int s = extractor->inputSize();
        if (s < 8)
            return false;
        // The object's true extent in frame pixels is baseRect * scale
        // (rectWidth stays at its capture value when rectAutoSize is off,
        // while the scale probe tracks the real size). Cropping without
        // the scale factor would frame the object inconsistently between
        // the bank seed and later verifications - degrading similarity as
        // altitude/zoom changes, for no real appearance change.
        const float w = DNN_CROP_CONTEXT * baseRectW * scale * scaleMul;
        const float h = DNN_CROP_CONTEXT * baseRectH * scale * scaleMul;
        extractPatch(f, cx, cy, w, h, s, patchBuf);
        return extractor->extract(patchBuf.data(), emb);
    }

    void resetVerifier()
    {
        bank.clear();
        vetoStreak = 0;
        verifyCountdown = 0;
        params.dnnSimilarity = 0.0f;
    }

    /// Find up to maxOut secondary local maxima of the last correlation
    /// response surface: at least an object-size apart from the primary
    /// peak and from each other, and at least ARB_SECOND_PEAK_RATIO of
    /// the primary value. Outputs subpixel window offsets (like
    /// Result.dx/dy). Returns the number found.
    ///
    /// Ported from the lockon engine so the epipolar-band constraint
    /// has secondary candidates to fall back on when the primary peak
    /// is off the line. The lockon-only features that use this same
    /// helper (DNN arbitration, DNN re-acquisition) are deliberately
    /// NOT ported — this build stays a pure correlation tracker plus
    /// the handoff enrichment path.
    int findSecondaryPeaks(float primDx, float primDy, float primPeak,
                           int maxOut, float outDx[], float outDy[])
    {
        const std::vector<float>& resp = core.responseSurface();
        const int sepX = std::max(5, (int)(0.5f * baseRectW / stepX));
        const int sepY = std::max(5, (int)(0.5f * baseRectH / stepY));
        const float minVal = ARB_SECOND_PEAK_RATIO * primPeak;
        int takenX[1 + ARB_MAX_CANDIDATES];
        int takenY[1 + ARB_MAX_CANDIDATES];
        takenX[0] = (int)std::lround(primDx) + fftW / 2;
        takenY[0] = (int)std::lround(primDy) + fftH / 2;
        int nTaken = 1;
        int nOut = 0;

        while (nOut < maxOut)
        {
            float best = minVal;
            int bx = -1, by = -1;
            for (int y = 1; y < fftH - 1; ++y)
            {
                for (int x = 1; x < fftW - 1; ++x)
                {
                    const float v = resp[(size_t)y * fftW + x];
                    if (v <= best)
                        continue;
                    bool near = false;
                    for (int t = 0; t < nTaken; ++t)
                    {
                        if (std::abs(x - takenX[t]) <= sepX &&
                            std::abs(y - takenY[t]) <= sepY)
                        {
                            near = true;
                            break;
                        }
                    }
                    if (near)
                        continue;
                    best = v;
                    bx = x;
                    by = y;
                }
            }
            if (bx < 0)
                break;
            float sx = (float)bx, sy = (float)by;
            {
                const float l = resp[(size_t)by * fftW + bx - 1];
                const float rr = resp[(size_t)by * fftW + bx + 1];
                const float d = l - 2.0f * best + rr;
                if (std::fabs(d) > 1e-9f)
                    sx += 0.5f * (l - rr) / d;
            }
            {
                const float t = resp[(size_t)(by - 1) * fftW + bx];
                const float b = resp[(size_t)(by + 1) * fftW + bx];
                const float d = t - 2.0f * best + b;
                if (std::fabs(d) > 1e-9f)
                    sy += 0.5f * (t - b) / d;
            }
            outDx[nOut] = sx - (float)(fftW / 2);
            outDy[nOut] = sy - (float)(fftH / 2);
            ++nOut;
            takenX[nTaken] = bx;
            takenY[nTaken] = by;
            ++nTaken;
        }
        return nOut;
    }

    void clearRecaptureCandidate()
    {
        candStreak = 0;
        candMiss = 0;
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
        ekf.reset();
        resetVerifier();
        clearRecaptureCandidate();
        searchWindowOverride = false;
        probeIndex = 0;
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
        searchWindowOverride = false;
        probeIndex = 0;
        clearRecaptureCandidate();
        if (params.ekfEnabled)
            ekf.initialize(posX, posY);
        else
            ekf.reset();

        // Seed the verifier template bank with the captured appearance.
        resetVerifier();
        if (params.dnnVerifierEnabled && computeEmbedding(f, cx, cy, embBuf))
        {
            bank.seed(embBuf, params.frameCounter);
            params.dnnSimilarity = 1.0f;
        }
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

        // Restrict to a neighborhood of the rectangle around center. The
        // window is extracted AT the current scale, so the object always
        // occupies baseRect/step window pixels regardless of scale -
        // multiplying by scale here would wrongly grow the neighborhood
        // and pull background into the moment statistics.
        const float rw = baseRectW / stepX; // rect in window pixels
        const float rh = baseRectH / stepY;
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

    /// Search window center used for processing of this frame. The manual
    /// override (SET_SEARCH_WINDOW_POSITION / MOVE_SEARCH_WINDOW) is
    /// one-shot in TRACKING mode (the tracker re-centers every frame
    /// anyway) but STICKY in LOST mode: it persists until re-capture,
    /// reset, or a new command, so an operator can hold the search on a
    /// spot while the tracker scans for the object there.
    void searchCenter(float& cx, float& cy)
    {
        if (searchWindowOverride)
        {
            cx = overrideX;
            cy = overrideY;
            if (params.mode != MODE_LOST)
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

    /// LOST-mode probe grid.
    ///
    /// The correlation features are attenuated by a cosine window spanning
    /// only ~2.5x the object size around the extraction center, so a
    /// single detect() "sees" just the middle of a large search window.
    /// To honor searchWindowWidth/Height mathematically, LOST mode probes
    /// additional off-center positions each frame, cycling over a grid of
    /// centers that:
    /// - reach the WINDOW EDGE (a probe's sight spilling past the edge is
    ///   harmless; stopping short leaves an undetectable edge band), and
    /// - are spaced at 0.5x the sight extent, so the worst-case cosine
    ///   attenuation between adjacent centers stays ~0.5 - which the
    ///   attenuation-compensated probability (CorrelationCore) then
    ///   corrects back to a near-centered score.
    /// Probes are visited nearest-first (the object usually re-appears
    /// close to the loss point), cycling.
    void rebuildProbeListIfNeeded()
    {
        const float exw = std::min(
            (float)fftW, std::max(32.0f, 2.5f * baseRectW / stepX));
        const float exh = std::min(
            (float)fftH, std::max(32.0f, 2.5f * baseRectH / stepY));
        const float sightX = exw * stepX * scale;
        const float sightY = exh * stepY * scale;
        const float swX = (float)params.searchWindowWidth * scale;
        const float swY = (float)params.searchWindowHeight * scale;
        if (sightX == probeGeomSightX && sightY == probeGeomSightY &&
            swX == probeGeomSwX && swY == probeGeomSwY)
            return; // geometry unchanged
        probeGeomSightX = sightX;
        probeGeomSightY = sightY;
        probeGeomSwX = swX;
        probeGeomSwY = swY;
        probeList.clear();
        probeIndex = 0;

        // No probing needed when one detect already sees the whole window.
        if (sightX >= swX - 1.0f && sightY >= swY - 1.0f)
            return;

        const float maxOffX = 0.5f * swX;
        const float maxOffY = 0.5f * swY;
        const int nx = std::max(
            1, 1 + (int)std::ceil(2.0f * maxOffX / (0.5f * sightX)));
        const int ny = std::max(
            1, 1 + (int)std::ceil(2.0f * maxOffY / (0.5f * sightY)));
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int ix = 0; ix < nx; ++ix)
            {
                const float ox =
                    nx > 1 ? -maxOffX + (2.0f * maxOffX * ix) / (nx - 1)
                           : 0.0f;
                const float oy =
                    ny > 1 ? -maxOffY + (2.0f * maxOffY * iy) / (ny - 1)
                           : 0.0f;
                // The central cell is covered by the primary detect.
                if (std::fabs(ox) < 1.0f && std::fabs(oy) < 1.0f)
                    continue;
                probeList.emplace_back(ox, oy);
            }
        }
        // Nearest-first visit order; cap the cycle length.
        std::sort(probeList.begin(), probeList.end(),
                  [](const std::pair<float, float>& a,
                     const std::pair<float, float>& b) {
                      return a.first * a.first + a.second * a.second <
                             b.first * b.first + b.second * b.second;
                  });
        if (probeList.size() > 48)
            probeList.resize(48);
    }

    bool nextLostProbe(float cx, float cy, float& px, float& py)
    {
        rebuildProbeListIfNeeded();
        if (probeList.empty())
            return false;
        const auto& off = probeList[(size_t)(probeIndex++) % probeList.size()];
        px = std::min(std::max(cx + off.first, 0.0f),
                      (float)params.frameWidth - 1.0f);
        py = std::min(std::max(cy + off.second, 0.0f),
                      (float)params.frameHeight - 1.0f);
        return true;
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
            if (params.ekfEnabled && ekf.initialized())
            {
                ekf.predict(1.0f, EKF_SIGMA_A, EKF_SIGMA_ALPHA);
                posX = ekf.px();
                posY = ekf.py();
                params.velX = ekf.vx();
                params.velY = ekf.vy();
            }
            else
            {
                posX += params.velX;
                posY += params.velY;
            }
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
        // EKF predict (advance state by one frame before consuming the
        // measurement). Skipped when EKF is disabled or uninitialized.
        if (params.ekfEnabled)
        {
            if (!ekf.initialized())
                ekf.initialize(posX, posY);
            ekf.predict(1.0f, EKF_SIGMA_A, EKF_SIGMA_ALPHA);
        }

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
            clearRecaptureCandidate();
            return;
        }

        // Epipolar band gating (dual-camera handoff): while active, if
        // the primary correlation peak lies off the epipolar line for
        // this peer, look at secondary peaks and pick the one closest
        // to the line as long as it's inside the half-width. Falls
        // back to the primary if no secondary qualifies. TTL decrement
        // runs unconditionally so the constraint expires even when we
        // don't use it. Same behaviour as the lockon engine.
        if (epiTtl > 0)
        {
            --epiTtl;
            const float norm2 = epiA * epiA + epiB * epiB;
            if (norm2 > 1e-12f && epiHalfW > 0.0f)
            {
                const float invNorm = 1.0f / std::sqrt(norm2);
                auto lineDist = [&](float px, float py) {
                    return std::fabs(epiA * px + epiB * py + epiC) * invNorm;
                };
                const float primX = cx + r.dx * stepX * scale;
                const float primY = cy + r.dy * stepY * scale;
                const float primDist = lineDist(primX, primY);
                if (primDist > epiHalfW)
                {
                    float sdx[ARB_MAX_CANDIDATES];
                    float sdy[ARB_MAX_CANDIDATES];
                    const int nSec = findSecondaryPeaks(
                        r.dx, r.dy, r.peak, ARB_MAX_CANDIDATES - 1, sdx, sdy);
                    float bestDist = primDist;
                    int   bestIdx  = -1;
                    for (int i = 0; i < nSec; ++i)
                    {
                        const float sxp = cx + sdx[i] * stepX * scale;
                        const float syp = cy + sdy[i] * stepY * scale;
                        const float d   = lineDist(sxp, syp);
                        if (d <= epiHalfW && d < bestDist)
                        {
                            bestDist = d;
                            bestIdx  = i;
                        }
                    }
                    if (bestIdx >= 0)
                    {
                        r.dx = sdx[bestIdx];
                        r.dy = sdy[bestIdx];
                    }
                    // else: keep primary; drift will correct next frame.
                }
            }
        }

        // Raw correlation peak position.
        const float nx = cx + r.dx * stepX * scale;
        const float ny = cy + r.dy * stepY * scale;
        const float dxF = nx - posX;
        const float dyF = ny - posY;

        if (params.ekfEnabled)
        {
            // EKF update with the measured position; use the filtered
            // estimate as the canonical position and velocity.
            ekf.update(nx, ny, r.prob, EKF_R_BASE);
            posX = ekf.px();
            posY = ekf.py();
            params.velX = ekf.vx();
            params.velY = ekf.vy();
        }
        else
        {
            posX = nx;
            posY = ny;
            // Legacy velocity (exponential smoothing, px/frame).
            params.velX = 0.6f * params.velX + 0.4f * dxF;
            params.velY = 0.6f * params.velY + 0.4f * dyF;
        }

        clampPos();
        if (centerTouchesEdge() && params.lostModeOption == 2)
        {
            resetToFree();
            return;
        }

        // DNN embedding verification (optional): confirm at a fixed cadence
        // that the tracked patch still matches the captured object.
        if (params.dnnVerifierEnabled && extractor)
        {
            if (--verifyCountdown <= 0)
            {
                verifyCountdown = std::max(1, params.dnnVerifyInterval);
                if (computeEmbedding(f, posX, posY, embBuf))
                {
                    if (bank.empty())
                        bank.seed(embBuf, params.frameCounter);
                    const float sim = bank.similarity(embBuf);
                    if (sim >= -1.0f)
                    {
                        params.dnnSimilarity = std::max(0.0f, sim);
                        if (sim < params.dnnVetoThreshold)
                            vetoStreak++;
                        else
                            vetoStreak = 0;
                        if (sim >= params.dnnAcceptThreshold &&
                            r.prob >= recaptureThreshold())
                            bank.maybeAdd(embBuf, params.frameCounter);

                        // DNN scale-pick: nudge box size toward the box scale
                        // DINOv2 matches best. Reuse embBuf (1.0x, sim above)
                        // + two extra embeds (0.9x, 1.1x); move the running
                        // scale a damped fraction toward the best match. Sizes
                        // the box from appearance, no regression head.
                        if (!bank.empty())
                        {
                            std::vector<float> pickBuf;
                            float bestMul = 1.0f, bestSim = sim;
                            for (float mul : {SCALE_PICK_LO, SCALE_PICK_HI})
                            {
                                if (computeEmbedding(f, posX, posY, pickBuf, mul))
                                {
                                    const float s2 = bank.similarity(pickBuf);
                                    if (s2 > bestSim) { bestSim = s2; bestMul = mul; }
                                }
                            }
                            if (bestMul != 1.0f)
                                scale = std::min(std::max(
                                    scale * (1.0f + SCALE_PICK_DAMP * (bestMul - 1.0f)),
                                    0.2f), 5.0f);
                        }
                    }
                }
            }
            if (vetoStreak >= DNN_VETO_STREAK_TO_LOST)
            {
                // Sustained appearance mismatch: whatever the correlation
                // filter is following no longer looks like the captured
                // object (drift onto background / distractor). Force LOST
                // so the pattern is frozen and re-detection takes over.
                vetoStreak = 0;
                params.mode = MODE_LOST;
                params.lostModeFrameCounter = 0;
                clearRecaptureCandidate();
                return;
            }
        }

        // Scale probe: every 4th frame test slightly smaller / larger scale.
        // Skipped when the DNN scale-pick is active (its appearance-based
        // sizing above supersedes this correlation probe, so they don't fight).
        const bool dnnScalePick = params.dnnVerifierEnabled && extractor;
        if (!dnnScalePick && ++scaleProbeCounter >= 4)
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
        // A pending verifier mismatch (vetoStreak > 0) blocks the update
        // entirely so a drifting pattern cannot learn the wrong object.
        extractWindow(f, posX, posY, scale, window);
        float rate = 0.0f;
        if (r.prob >= recaptureThreshold())
            rate = learningRate();
        else
            rate = learningRate() * r.prob;
        if (vetoStreak > 0)
            rate = 0.0f;
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

        // Update coordinates according to LOST mode option. The EKF (if
        // enabled) carries the prediction forward through curved motion;
        // otherwise fall back to the legacy straight-line velocity model.
        if (params.lostModeOption == 1 || params.lostModeOption == 2)
        {
            if (params.ekfEnabled && ekf.initialized())
            {
                ekf.predict(1.0f, EKF_SIGMA_A, EKF_SIGMA_ALPHA);
                posX = ekf.px();
                posY = ekf.py();
                params.velX = ekf.vx();
                params.velY = ekf.vy();
            }
            else
            {
                posX += params.velX;
                posY += params.velY;
            }
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

        // Try to re-detect the object. Primary detect at the search window
        // center, plus cycling off-center probes so the WHOLE configured
        // search window is covered within a few frames (the feature
        // support of a single detect is only ~2.5x the object size - see
        // rebuildProbeListIfNeeded). detectBest() correlates each window
        // against BOTH the adaptive filter and the capture-time snapshot:
        // the adaptive filter may have decayed toward the occluder during
        // the confidence drop before the loss, while the snapshot still
        // matches the object as captured.
        float cx, cy;
        searchCenter(cx, cy);
        const float cx0 = cx, cy0 = cy; // probe offsets anchor here
        extractWindow(f, cx, cy, scale, window);
        CorrelationCore::Result r = core.detectBest(window);
        // Pending-candidate detect: examine the tentative re-capture
        // location directly so confirmation does not depend on the probe
        // cycle returning there.
        if (candStreak > 0)
        {
            extractWindow(f, candX, candY, scale, window);
            CorrelationCore::Result rc = core.detectBest(window);
            if (rc.prob > r.prob)
            {
                r = rc;
                cx = candX;
                cy = candY;
            }
        }
        for (int p = 0; p < LOST_PROBES_PER_FRAME; ++p)
        {
            float probeX, probeY;
            if (!nextLostProbe(cx0, cy0, probeX, probeY))
                break;
            extractWindow(f, probeX, probeY, scale, window);
            CorrelationCore::Result rp = core.detectBest(window);
            if (rp.prob > r.prob)
            {
                r = rp;
                cx = probeX;
                cy = probeY;
            }
        }
        params.detectionProbability = r.prob;

        if (r.prob > recaptureThreshold())
        {
            const float nx = cx + r.dx * stepX * scale;
            const float ny = cy + r.dy * stepY * scale;

            // Physics reach gate: a ground object cannot teleport - the
            // plausible displacement from the (predicted) loss position
            // grows with time lost. Candidates beyond the reachable
            // radius are almost certainly distractors; ignore them. The
            // radius keeps growing, so a genuinely far re-appearance is
            // still accepted later. Skipped when the operator explicitly
            // repositioned the search window (sticky override = intent).
            if (!searchWindowOverride)
            {
                const float speed = std::sqrt(params.velX * params.velX +
                                              params.velY * params.velY);
                const float rectMax = (float)std::max(params.rectWidth,
                                                      params.rectHeight);
                const float reach =
                    REACH_BASE_FACTOR * rectMax +
                    (speed + REACH_GROWTH_PX_PER_FRAME) *
                        (float)params.lostModeFrameCounter;
                const float ddx = nx - posX;
                const float ddy = ny - posY;
                if (ddx * ddx + ddy * ddy > reach * reach)
                {
                    if (params.lostModeFrameCounter >=
                        params.maxFramesInLostMode)
                        resetToFree();
                    return;
                }
            }

            // DNN verification of the re-capture candidate (optional):
            // the correlation filter is confident, but the candidate must
            // also LOOK like the captured object. This rejects locking
            // onto similar-looking distractors after occlusion.
            if (params.dnnVerifierEnabled && extractor && !bank.empty())
            {
                if (computeEmbedding(f, nx, ny, embBuf))
                {
                    const float sim = bank.similarity(embBuf);
                    if (sim >= -1.0f)
                    {
                        params.dnnSimilarity = std::max(0.0f, sim);
                        if (sim < params.dnnAcceptThreshold)
                        {
                            // Candidate rejected - stay LOST this frame.
                            if (params.lostModeFrameCounter >=
                                params.maxFramesInLostMode)
                                resetToFree();
                            return;
                        }
                    }
                }
            }

            // Multi-frame confirmation: the SAME location must clear all
            // gates on RECAPTURE_CONFIRM_FRAMES consecutive frames before
            // the lock commits. Single-frame correlation flukes (noise, a
            // passing object clipping the window) never re-lock.
            const float confirmR =
                RECAPTURE_CONFIRM_RADIUS *
                (float)std::max(params.rectWidth, params.rectHeight);
            if (candStreak > 0 &&
                (nx - candX) * (nx - candX) + (ny - candY) * (ny - candY) <=
                    confirmR * confirmR)
                ++candStreak;
            else
                candStreak = 1;
            candX = nx;
            candY = ny;
            candMiss = 0;
            if (candStreak < RECAPTURE_CONFIRM_FRAMES)
            {
                if (params.lostModeFrameCounter >= params.maxFramesInLostMode)
                    resetToFree();
                return;
            }
            clearRecaptureCandidate();

            // Automatic re-capture. Re-acquisition starts a NEW track
            // segment: the motion state is stale after LOST (and with
            // lostModeOption 0 the covariance is still tight around the
            // loss point), so an incremental EKF update would be crushed
            // by the Mahalanobis gate - the filter would hold the old
            // position and the tracker would flicker TRACKING->LOST.
            // Re-seed the filter at the measurement instead; velocity
            // re-converges within a few frames.
            if (params.ekfEnabled)
            {
                ekf.initialize(nx, ny);
                params.velX = 0.0f;
                params.velY = 0.0f;
            }
            // If the snapshot filter won the re-detection, the adaptive
            // filter had decayed away from the object - resume TRACKING
            // with the filter that actually matched.
            if (r.snapshot)
                core.restoreSnapshot();
            posX = nx;
            posY = ny;
            clampPos();
            params.mode = MODE_TRACKING;
            params.lostModeFrameCounter = 0;
            vetoStreak = 0;
            searchWindowOverride = false; // release sticky LOST override
            probeIndex = 0;
            return;
        }

        // No acceptable detection this frame: age out a pending candidate
        // that stops re-appearing (2 consecutive misses).
        if (candStreak > 0 && ++candMiss > 2)
            clearRecaptureCandidate();

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
        params.lostModeOption < 0 || params.lostModeOption > 2 ||
        params.dnnVerifyInterval < 1 ||
        params.dnnVetoThreshold < 0.0f || params.dnnVetoThreshold > 1.0f ||
        params.dnnAcceptThreshold < 0.0f || params.dnnAcceptThreshold > 1.0f)
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
    p.dnnSimilarity = 0.0f;

    m_impl->fft->setMultiThread(p.multipleThreads);
    m_impl->setupGeometry();
    m_impl->buffer.clear();
    m_impl->ekf.reset();
    m_impl->bank.configure(DNN_BANK_CAPACITY, DNN_BANK_MIN_SPACING);
    m_impl->resetVerifier();
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
    
        // --- ADDED CASES FOR EKF SETTERS ---
    case VTrackerParam::EKF_SIGMA_A:
        p.ekfSigmaA = value;
        return true;
    case VTrackerParam::EKF_SIGMA_ALPHA:
        p.ekfSigmaAlpha = value;
        return true;
    case VTrackerParam::EKF_R_BASE:
        p.ekfRBase = value;
        return true;
    // -------------------------------------
    
    case VTrackerParam::ENABLE_EKF:
        p.ekfEnabled = value != 0.0f;
        if (!p.ekfEnabled)
            m_impl->ekf.reset();
        else if (p.mode == MODE_TRACKING || p.mode == MODE_LOST ||
                 p.mode == MODE_INERTIAL)
            m_impl->ekf.initialize(m_impl->posX, m_impl->posY);
        return true;
    case VTrackerParam::ENABLE_DNN_VERIFIER:
        p.dnnVerifierEnabled = value != 0.0f;
        if (!p.dnnVerifierEnabled)
            m_impl->resetVerifier();
        return true;
    
    case VTrackerParam::DNN_VERIFY_INTERVAL:
        if (value < 1.0f)
            return false;
        p.dnnVerifyInterval = (int)value;
        return true;
    case VTrackerParam::DNN_VETO_THRESHOLD:
        if (value < 0.0f || value > 1.0f)
            return false;
        p.dnnVetoThreshold = value;
        return true;
    case VTrackerParam::DNN_ACCEPT_THRESHOLD:
        if (value < 0.0f || value > 1.0f)
            return false;
        p.dnnAcceptThreshold = value;
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
    // --- ADDED CASES FOR EKF VARIABLES ---
    case VTrackerParam::EKF_SIGMA_A:     return p.ekfSigmaA;
    case VTrackerParam::EKF_SIGMA_ALPHA: return p.ekfSigmaAlpha;
    case VTrackerParam::EKF_R_BASE:      return p.ekfRBase;
    // -------------------------------------
    case VTrackerParam::ENABLE_EKF: return p.ekfEnabled ? 1.0f : 0.0f;
    case VTrackerParam::ENABLE_DNN_VERIFIER: return p.dnnVerifierEnabled ? 1.0f : 0.0f;
    case VTrackerParam::DNN_VERIFY_INTERVAL: return (float)p.dnnVerifyInterval;
    case VTrackerParam::DNN_VETO_THRESHOLD: return p.dnnVetoThreshold;
    case VTrackerParam::DNN_ACCEPT_THRESHOLD: return p.dnnAcceptThreshold;
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
            d.clearRecaptureCandidate();
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

void CvTracker::setFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->extractor = std::move(extractor);
    m_impl->resetVerifier();
}

bool CvTracker::exportTemplateBank(std::vector<unsigned char>& out) const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->bank.serialize(out);
}

bool CvTracker::importTemplateBank(const std::vector<unsigned char>& in)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->bank.deserialize(in);
}

void CvTracker::setEpipolarConstraint(float a, float b, float c,
                                      float halfWidthPx, int ttlFrames)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (ttlFrames <= 0 || (a == 0.0f && b == 0.0f))
    {
        m_impl->epiTtl = 0;
        return;
    }
    m_impl->epiA = a;
    m_impl->epiB = b;
    m_impl->epiC = c;
    m_impl->epiHalfW = halfWidthPx;
    m_impl->epiTtl = ttlFrames;
}

} // namespace vtracker
} // namespace cr
