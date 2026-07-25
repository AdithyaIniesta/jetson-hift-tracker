// Synthetic-sequence verification of CvTracker:
// - tracking under motion, rotation (orientation change) and global
//   lighting changes;
// - loss detection and automatic re-capture after full occlusion;
// - stop-frame capture on a past frame (catch-up processing);
// - params / command serialization round trip.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cvtracker/CvTracker.h"

using namespace cr::vtracker;
using namespace cr::video;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);        \
            ++g_failures;                                                      \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            std::printf("PASS: %s\n", msg);                                    \
        }                                                                      \
    } while (0)

constexpr int W = 640;
constexpr int H = 480;
constexpr int OBJ = 56; // object size

// Deterministic pseudo-random background texture.
uint8_t bgTexture(int x, int y)
{
    uint32_t v = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    v = (v ^ (v >> 13)) * 1274126177u;
    return (uint8_t)(90 + ((v >> 16) % 50)); // mid-gray noise
}

// Object texture (checker + radial gradient), sampled in object-local
// coordinates u, v in [-1, 1], rotated by angle.
uint8_t objTexture(float u, float v, float angle)
{
    const float ca = std::cos(angle), sa = std::sin(angle);
    const float ru = ca * u - sa * v;
    const float rv = sa * u + ca * v;
    const int cu = (int)std::floor((ru + 1.0f) * 3.0f);
    const int cv = (int)std::floor((rv + 1.0f) * 3.0f);
    const float r = std::sqrt(u * u + v * v);
    const float base = ((cu + cv) & 1) ? 230.0f : 30.0f;
    return (uint8_t)std::max(0.0f, std::min(255.0f, base * (1.2f - 0.5f * r)));
}

void renderFrame(Frame& frame, float objX, float objY, float angle,
                 float gain, float offset, bool occluded)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float px = bgTexture(x, y);
            const float u = (x - objX) / (OBJ * 0.5f);
            const float v = (y - objY) / (OBJ * 0.5f);
            if (!occluded && std::fabs(u) <= 1.0f && std::fabs(v) <= 1.0f)
                px = objTexture(u, v, angle);
            px = px * gain + offset;
            frame.data[(size_t)y * W + x] =
                (uint8_t)std::max(0.0f, std::min(255.0f, px));
        }
    }
}

float objPathX(int i) { return 320.0f + 180.0f * std::sin(0.012f * i); }
float objPathY(int i) { return 240.0f + 60.0f * std::sin(0.05f * i); }

} // namespace

int main()
{
    std::printf("CvTracker version: %s\n", CvTracker::getVersion().c_str());

    // ---------------- Serialization round trip ----------------
    {
        VTrackerParams a;
        a.rectX = 123;
        a.rectY = 456;
        a.velX = 1.25f;
        a.detectionProbability = 0.5f;
        a.rectAutoSize = true;
        a.custom2 = 0.33f;
        uint8_t buf[512];
        int size = 0;
        CHECK(a.encode(buf, sizeof(buf), size), "params encode");
        VTrackerParams b;
        CHECK(b.decode(buf, size), "params decode");
        CHECK(b.rectX == 123 && b.rectY == 456 && b.velX == 1.25f &&
                  b.rectAutoSize && b.custom2 == 0.33f,
              "params round trip values");

        uint8_t cbuf[32];
        int csize = 0;
        VTracker::encodeCommand(cbuf, csize, VTrackerCommand::CAPTURE, 11.0f,
                                22.0f, 33.0f);
        VTrackerParam pid{};
        VTrackerCommand cid{};
        float v1, v2, v3;
        CHECK(VTracker::decodeCommand(cbuf, csize, pid, cid, v1, v2, v3) == 0 &&
                  cid == VTrackerCommand::CAPTURE && v1 == 11.0f &&
                  v3 == 33.0f,
              "command round trip");
        VTracker::encodeSetParamCommand(cbuf, csize,
                                        VTrackerParam::RECT_WIDTH, 64.0f);
        CHECK(VTracker::decodeCommand(cbuf, csize, pid, cid, v1, v2, v3) == 1 &&
                  pid == VTrackerParam::RECT_WIDTH && v1 == 64.0f,
              "set param command round trip");
    }

    // ---------------- Tracking scenario ----------------
    CvTracker tracker;
    VTrackerParams params;
    params.rectWidth = OBJ + 8;
    params.rectHeight = OBJ + 8;
    params.searchWindowWidth = 256;
    params.searchWindowHeight = 256;
    params.frameBufferSize = 64;
    params.maxFramesInLostMode = 256;
    params.lostModeOption = 1; // predict by velocity in LOST mode
    params.multipleThreads = true;
    CHECK(tracker.initVTracker(params), "initVTracker");

    Frame frame(W, H, Fourcc::GRAY);
    VTrackerParams r;

    auto feed = [&](int i, bool occluded, float gain, float offset) {
        renderFrame(frame, objPathX(i), objPathY(i), 0.01f * i, gain, offset,
                    occluded);
        frame.frameId = i;
        if (!tracker.processFrame(frame))
        {
            std::printf("FAIL: processFrame returned false at %d\n", i);
            ++g_failures;
        }
        tracker.getParams(r);
    };

    // Warm up a few frames in FREE mode, then capture.
    for (int i = 0; i < 3; ++i)
        feed(i, false, 1.0f, 0.0f);
    CHECK(r.mode == 0, "FREE mode before capture");
    CHECK(tracker.executeCommand(VTrackerCommand::CAPTURE, objPathX(2),
                                 objPathY(2), -1.0f),
          "CAPTURE command");
    tracker.getParams(r);
    CHECK(r.mode == 1, "TRACKING mode after capture");

    // Phase A: motion + rotation + lighting ramp.
    float maxErrA = 0.0f;
    for (int i = 3; i < 120; ++i)
    {
        const float gain = 1.0f + 0.4f * std::sin(0.05f * i);  // 0.6 .. 1.4
        const float offset = 20.0f * std::sin(0.11f * i);      // -20 .. 20
        feed(i, false, gain, offset);
        const float ex = std::fabs(r.rectX - objPathX(i));
        const float ey = std::fabs(r.rectY - objPathY(i));
        maxErrA = std::max(maxErrA, std::max(ex, ey));
    }
    std::printf("Phase A (motion+rotation+lighting): mode=%d prob=%.2f "
                "maxErr=%.1f px, vel=(%.2f, %.2f)\n",
                r.mode, r.detectionProbability, maxErrA, r.velX, r.velY);
    CHECK(r.mode == 1, "still TRACKING after phase A");
    CHECK(maxErrA < 12.0f, "phase A tracking error < 12 px");
    CHECK(r.detectionProbability > 0.4f, "high confidence at end of phase A");

    // Phase B: full occlusion for 25 frames.
    int lostSeen = 0;
    for (int i = 120; i < 145; ++i)
    {
        feed(i, true, 1.0f, 0.0f);
        if (r.mode == 2)
            ++lostSeen;
    }
    std::printf("Phase B (occlusion): mode=%d prob=%.2f lostFrames=%d\n",
                r.mode, r.detectionProbability, lostSeen);
    CHECK(lostSeen > 10, "LOST mode detected during occlusion");

    // Phase C: object re-appears, expect automatic re-capture.
    int recapturedAt = -1;
    float endErr = 1e9f;
    for (int i = 145; i < 220; ++i)
    {
        feed(i, false, 1.0f, 0.0f);
        if (r.mode == 1 && recapturedAt < 0)
            recapturedAt = i;
        if (i == 219)
            endErr = std::max(std::fabs(r.rectX - objPathX(i)),
                              std::fabs(r.rectY - objPathY(i)));
    }
    std::printf("Phase C (re-capture): mode=%d recapturedAt=%d endErr=%.1f\n",
                r.mode, recapturedAt, endErr);
    CHECK(recapturedAt >= 0, "automatic re-capture after occlusion");
    CHECK(r.mode == 1, "TRACKING at end of phase C");
    CHECK(endErr < 12.0f, "post-re-capture tracking error < 12 px");

    // ---------------- RESET and stop-frame capture ----------------
    CHECK(tracker.executeCommand(VTrackerCommand::RESET), "RESET command");
    tracker.getParams(r);
    CHECK(r.mode == 0, "FREE after RESET");

    // Feed 10 more frames, then capture on a frame 8 frames in the past
    // (stop-frame / delay compensation), then continue.
    for (int i = 220; i < 230; ++i)
        feed(i, false, 1.0f, 0.0f);
    const int pastId = 222;
    CHECK(tracker.executeCommand(VTrackerCommand::CAPTURE, objPathX(pastId),
                                 objPathY(pastId), (float)pastId),
          "CAPTURE on past frame (stop-frame)");
    tracker.getParams(r);
    CHECK(r.mode == 1 && r.processedFrameId == pastId,
          "captured on buffered past frame");
    // Next frame triggers catch-up processing of buffered frames.
    feed(230, false, 1.0f, 0.0f);
    CHECK(r.processedFrameId == 230, "catch-up processing reached newest frame");
    float cuErr = std::max(std::fabs(r.rectX - objPathX(230)),
                           std::fabs(r.rectY - objPathY(230)));
    std::printf("Stop-frame catch-up: mode=%d err=%.1f px prob=%.2f\n", r.mode,
                cuErr, r.detectionProbability);
    CHECK(r.mode == 1 && cuErr < 12.0f, "accurate after catch-up");

    // ---------------- rect auto-size behavior ----------------
    {
        // Auto-size OFF: rectangle size must never change by itself.
        tracker.setParam(VTrackerParam::RECT_AUTO_SIZE, 0.0f);
        tracker.getParams(r);
        const int w0 = r.rectWidth, h0 = r.rectHeight;
        bool sizeFrozen = true;
        for (int i = 231; i < 290; ++i)
        {
            feed(i, false, 1.0f + 0.2f * std::sin(0.1f * i), 0.0f);
            if (r.rectWidth != w0 || r.rectHeight != h0)
                sizeFrozen = false;
        }
        CHECK(sizeFrozen, "rect size frozen with RECT_AUTO_SIZE off");

        // Auto-size ON: size may change, but only smoothly (<= 6% / frame).
        tracker.setParam(VTrackerParam::RECT_AUTO_SIZE, 1.0f);
        bool smooth = true;
        int prevW = r.rectWidth, prevH = r.rectHeight;
        for (int i = 290; i < 360; ++i)
        {
            feed(i, false, 1.0f, 0.0f);
            if (std::abs(r.rectWidth - prevW) > std::max(2, prevW * 6 / 100) ||
                std::abs(r.rectHeight - prevH) > std::max(2, prevH * 6 / 100))
                smooth = false;
            prevW = r.rectWidth;
            prevH = r.rectHeight;
        }
        std::printf("Auto-size on: rect %dx%d (object %dx%d, obj est %dx%d)\n",
                    r.rectWidth, r.rectHeight, OBJ, OBJ, r.objectWidth,
                    r.objectHeight);
        CHECK(smooth, "auto-size changes are rate-limited");
        CHECK(r.mode == 1, "still TRACKING during auto-size");

        // Toggle OFF again: must freeze immediately at current size.
        tracker.setParam(VTrackerParam::RECT_AUTO_SIZE, 0.0f);
        tracker.getParams(r);
        const int wf = r.rectWidth, hf = r.rectHeight;
        bool frozenAgain = true;
        for (int i = 360; i < 400; ++i)
        {
            feed(i, false, 1.0f, 0.0f);
            if (r.rectWidth != wf || r.rectHeight != hf)
                frozenAgain = false;
        }
        CHECK(frozenAgain, "rect size frozen again after toggling off");
    }

    // ---------------- getImage sanity ----------------
    Frame img;
    tracker.getImage(0, img);
    CHECK(img.width == 256 && img.height == 256 &&
              img.fourcc == Fourcc::GRAY,
          "getImage reference image format");
    tracker.getImage(2, img);
    CHECK(img.data != nullptr, "getImage correlation surface");

    // ---------------- YUYV input (camera agnosticism) ----------------
    {
        CvTracker t2;
        VTrackerParams p2;
        p2.rectWidth = OBJ;
        p2.rectHeight = OBJ;
        CHECK(t2.initVTracker(p2), "init second tracker");
        Frame gray(W, H, Fourcc::GRAY);
        Frame yuyv(W, H, Fourcc::YUYV);
        for (int i = 0; i < 40; ++i)
        {
            renderFrame(gray, objPathX(i), objPathY(i), 0.0f, 1.0f, 0.0f,
                        false);
            for (size_t px = 0; px < (size_t)W * H; ++px)
            {
                yuyv.data[px * 2] = gray.data[px];
                yuyv.data[px * 2 + 1] = 128;
            }
            yuyv.frameId = i;
            t2.processFrame(yuyv);
            if (i == 2)
                t2.executeCommand(VTrackerCommand::CAPTURE, objPathX(2),
                                  objPathY(2), -1.0f);
        }
        VTrackerParams r2;
        t2.getParams(r2);
        const float err = std::max(std::fabs(r2.rectX - objPathX(39)),
                                   std::fabs(r2.rectY - objPathY(39)));
        std::printf("YUYV input: mode=%d err=%.1f px\n", r2.mode, err);
        CHECK(r2.mode == 1 && err < 12.0f, "tracking on YUYV input");
    }

    std::printf("\n%s (%d failure(s))\n",
                g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
