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

// objX/objY are WORLD coordinates; camX/camY simulate camera motion (the
// whole scene, object included, shifts on screen by -cam). Default 0 keeps
// legacy world == screen behavior for existing tests.
void renderFrame(Frame& frame, float objX, float objY, float angle,
                 float gain, float offset, bool occluded, int camX = 0,
                 int camY = 0)
{
    const float sxObj = objX - (float)camX; // object position on SCREEN
    const float syObj = objY - (float)camY;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float px = bgTexture(x + camX, y + camY);
            const float u = (x - sxObj) / (OBJ * 0.5f);
            const float v = (y - syObj) / (OBJ * 0.5f);
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
        a.dnnVerifierEnabled = true;
        a.dnnVerifyInterval = 4;
        a.dnnVetoThreshold = 0.37f;
        a.dnnSimilarity = 0.91f;
        uint8_t buf[512];
        int size = 0;
        CHECK(a.encode(buf, sizeof(buf), size), "params encode");
        VTrackerParams b;
        CHECK(b.decode(buf, size), "params decode");
        CHECK(b.rectX == 123 && b.rectY == 456 && b.velX == 1.25f &&
                  b.rectAutoSize && b.custom2 == 0.33f,
              "params round trip values");
        CHECK(b.dnnVerifierEnabled && b.dnnVerifyInterval == 4 &&
                  b.dnnVetoThreshold == 0.37f && b.dnnSimilarity == 0.91f,
              "dnn verifier params round trip");
        a.gmcEnabled = true;
        a.gmcShiftX = -3.5f;
        a.dnnArbitration = true;
        a.dnnReacquisition = true;
        CHECK(a.encode(buf, sizeof(buf), size), "params encode v2");
        VTrackerParams c;
        CHECK(c.decode(buf, size) && c.gmcEnabled && c.gmcShiftX == -3.5f &&
                  c.dnnArbitration && c.dnnReacquisition,
              "gmc/arbitration/reacquisition params round trip");

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

    // ---------------- DNN verifier plumbing (TinyPatchExtractor) ----------
    {
        CvTracker t3;
        VTrackerParams p3;
        p3.rectWidth = OBJ;
        p3.rectHeight = OBJ;
        p3.dnnVerifierEnabled = true;
        p3.dnnVerifyInterval = 4;
        CHECK(t3.initVTracker(p3), "init verifier tracker");
        t3.setFeatureExtractor(std::make_shared<TinyPatchExtractor>());
        CHECK(t3.getParam(VTrackerParam::ENABLE_DNN_VERIFIER) == 1.0f,
              "ENABLE_DNN_VERIFIER readable");
        CHECK(t3.setParam(VTrackerParam::DNN_ACCEPT_THRESHOLD, 0.6f),
              "set DNN_ACCEPT_THRESHOLD");
        CHECK(!t3.setParam(VTrackerParam::DNN_VETO_THRESHOLD, 1.5f),
              "invalid DNN_VETO_THRESHOLD rejected");

        Frame f3(W, H, Fourcc::GRAY);
        VTrackerParams r3;
        for (int i = 0; i < 60; ++i)
        {
            renderFrame(f3, objPathX(i), objPathY(i), 0.0f, 1.0f, 0.0f,
                        false);
            f3.frameId = i;
            t3.processFrame(f3);
            if (i == 2)
                t3.executeCommand(VTrackerCommand::CAPTURE, objPathX(2),
                                  objPathY(2), -1.0f);
        }
        t3.getParams(r3);
        std::printf("DNN verifier: mode=%d prob=%.2f sim=%.2f\n", r3.mode,
                    r3.detectionProbability, r3.dnnSimilarity);
        CHECK(r3.mode == 1, "TRACKING with verifier enabled");
        CHECK(r3.dnnSimilarity > 0.6f,
              "verifier similarity high on consistent appearance");
    }

    // ---------------- GMC: survive a violent camera jolt -------------------
    // A boresight camera jolt shifts the WHOLE scene in one frame. Without
    // GMC a (60, 25) px jump exceeds what the matcher can absorb and the
    // lock is permanently gone (screen position never returns). With GMC
    // the phase-correlation measurement moves the search position with the
    // scene and tracking must survive.
    {
        CvTracker t6;
        VTrackerParams p6;
        p6.rectWidth = OBJ;
        p6.rectHeight = OBJ;
        p6.searchWindowWidth = 256;
        p6.searchWindowHeight = 256;
        p6.gmcEnabled = true;
        CHECK(t6.initVTracker(p6), "init t6 (GMC jolt)");

        Frame f6(W, H, Fourcc::GRAY);
        VTrackerParams r6;
        const float wx = 320.0f, wy = 240.0f; // object WORLD position
        int camX = 0, camY = 0;
        auto feed6 = [&](int i) {
            renderFrame(f6, wx, wy, 0.0f, 1.0f, 0.0f, false, camX, camY);
            f6.frameId = i;
            t6.processFrame(f6);
            t6.getParams(r6);
        };
        for (int i = 0; i < 3; ++i)
            feed6(i);
        t6.executeCommand(VTrackerCommand::CAPTURE, wx, wy, -1.0f);
        for (int i = 3; i < 40; ++i)
            feed6(i);
        CHECK(r6.mode == 1, "t6 tracking before jolt");

        camX = 60; // violent single-frame camera jolt
        camY = 25;
        feed6(40);
        // Scene content moved by (-60, -25) on screen; GMC must measure it
        // (sign check: gmcShift is the shift applied to stored positions).
        std::printf("GMC jolt frame: shift=(%.1f, %.1f) expected (-60, -25)\n",
                    r6.gmcShiftX, r6.gmcShiftY);
        CHECK(std::fabs(r6.gmcShiftX + 60.0f) < 8.0f &&
                  std::fabs(r6.gmcShiftY + 25.0f) < 8.0f,
              "GMC measured the jolt (magnitude and sign)");

        for (int i = 41; i < 90; ++i)
            feed6(i);
        const float sx = wx - camX, sy = wy - camY; // screen position now
        std::printf("GMC after jolt: mode=%d rect=(%d,%d) target=(%.0f,%.0f)\n",
                    r6.mode, r6.rectX, r6.rectY, sx, sy);
        CHECK(r6.mode == 1 && std::fabs(r6.rectX - sx) < 12.0f &&
                  std::fabs(r6.rectY - sy) < 12.0f,
              "tracking survived the camera jolt with GMC");
    }

    // ---------------- Embedding re-acquisition (sight-bubble escape) -------
    // Object re-appears 64 px from the frozen LOST search center: inside
    // the search window, but far enough that the correlation response is
    // attenuated to ~zero (feature window support). The embedding grid
    // sweep must find it and cue the confirming detect.
    {
        CvTracker t7;
        VTrackerParams p7;
        p7.rectWidth = OBJ;
        p7.rectHeight = OBJ;
        p7.searchWindowWidth = 256;
        p7.searchWindowHeight = 256;
        p7.maxFramesInLostMode = 300;
        p7.lostModeOption = 0; // freeze at loss position
        p7.custom2 = 0.35f;
        p7.dnnVerifierEnabled = true;
        p7.dnnReacquisition = true;
        CHECK(t7.initVTracker(p7), "init t7 (embedding re-acquisition)");
        t7.setFeatureExtractor(std::make_shared<TinyPatchExtractor>());
        t7.setParam(VTrackerParam::DNN_ACCEPT_THRESHOLD, 0.40f);

        Frame f7(W, H, Fourcc::GRAY);
        VTrackerParams r7;
        const float bx = 260.0f, by = 240.0f;
        const float rx = bx + 64.0f; // re-appear: on the reacq grid ring,
                                     // outside effective correlation reach
        auto feed7 = [&](int i, float ox, bool occ) {
            renderFrame(f7, ox, by, 0.0f, 1.0f, 0.0f, occ);
            f7.frameId = i;
            t7.processFrame(f7);
            t7.getParams(r7);
        };
        int i7 = 0;
        for (; i7 < 3; ++i7)
            feed7(i7, bx, false);
        t7.executeCommand(VTrackerCommand::CAPTURE, bx, by, -1.0f);
        for (; i7 < 30; ++i7)
            feed7(i7, bx, false);
        CHECK(r7.mode == 1, "t7 tracking before occlusion");
        for (; i7 < 50; ++i7)
            feed7(i7, bx, true);
        CHECK(r7.mode == 2, "t7 LOST during occlusion");
        int recapAt = -1;
        for (; i7 < 200; ++i7)
        {
            feed7(i7, rx, false);
            if (r7.mode == 1 && recapAt < 0)
                recapAt = i7;
        }
        std::printf("Embedding re-acquisition: mode=%d recapAt=%d "
                    "rect=(%d,%d) target=(%.0f,%.0f)\n",
                    r7.mode, recapAt, r7.rectX, r7.rectY, rx, by);
        CHECK(recapAt >= 0, "re-acquired outside the correlation sight bubble");
        CHECK(r7.mode == 1 && std::fabs(r7.rectX - rx) < 12.0f &&
                  std::fabs(r7.rectY - by) < 12.0f,
              "accurate lock after embedding re-acquisition");
    }

    // ---------------- Multi-peak arbitration plumbing ----------------------
    // Non-regression: arbitration enabled on a clean single-object scene
    // must not disturb normal tracking.
    {
        CvTracker t8;
        VTrackerParams p8;
        p8.rectWidth = OBJ;
        p8.rectHeight = OBJ;
        p8.dnnVerifierEnabled = true;
        p8.dnnArbitration = true;
        CHECK(t8.initVTracker(p8), "init t8 (arbitration plumbing)");
        t8.setFeatureExtractor(std::make_shared<TinyPatchExtractor>());
        Frame f8(W, H, Fourcc::GRAY);
        VTrackerParams r8;
        for (int i = 0; i < 60; ++i)
        {
            renderFrame(f8, objPathX(i), objPathY(i), 0.0f, 1.0f, 0.0f,
                        false);
            f8.frameId = i;
            t8.processFrame(f8);
            if (i == 2)
                t8.executeCommand(VTrackerCommand::CAPTURE, objPathX(2),
                                  objPathY(2), -1.0f);
        }
        t8.getParams(r8);
        const float err8 = std::max(std::fabs(r8.rectX - objPathX(59)),
                                    std::fabs(r8.rectY - objPathY(59)));
        std::printf("Arbitration plumbing: mode=%d err=%.1f px\n", r8.mode,
                    err8);
        CHECK(r8.mode == 1 && err8 < 12.0f,
              "tracking unchanged with arbitration enabled");
    }

    std::printf("\n%s (%d failure(s))\n",
                g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
