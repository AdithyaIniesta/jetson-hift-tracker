#include "streaming.h"

#include <gst/app/gstappsrc.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>

#include "../control/control.h"
#include "../telemetry/telemetry.h"

std::string makeRecordingSessionName() {
  std::time_t now = std::time(nullptr);
  std::tm tm = *std::localtime(&now);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H-%M-%S");
  return oss.str();
}

std::string makeRecordingPipelineDesc(const std::string &outputFile, int W,
                                      int H, int fps) {
  char buf[1024];
  snprintf(buf, sizeof(buf),
           "appsrc name=recordsrc is-live=true format=3 ! "
           "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! "
           "videoconvert ! video/x-raw,format=NV12 ! nvvidconv ! "
           "video/x-raw(memory:NVMM),format=NV12 ! "
           "nvv4l2h264enc ! h264parse ! matroskamux ! "
           "filesink location=%s",
           W, H, fps, outputFile.c_str());
  return std::string(buf);
}

// ── Helper: push one frame from a ring to a GstAppSrc ────────
// Returns true if a frame was pushed, false if no frame available.
static bool push_raw_frame(DualRingBuffer &ring, int &readIdx, GstElement *src,
                           int frame_sz) {
  if (!src)
    return false;

  DualCaptureSlot captured;
  {
    std::unique_lock<std::mutex> lk(ring.ring_mutex);
    ring.ring_condition.wait_for(lk, std::chrono::milliseconds(33), [&] {
      return ring.write_index.load() > readIdx || !g_running;
    });
    if (!g_running)
      return false;
    if (ring.write_index.load() <= readIdx)
      return false;
    if (ring.write_index.load() - readIdx > DUAL_RING_SIZE)
      readIdx = ring.write_index.load() - DUAL_RING_SIZE;
    DualCaptureSlot &slot = ring.slots[readIdx % DUAL_RING_SIZE];
    if (!slot.contains_valid_frame) {
      readIdx++;
      return false;
    }
    captured = slot;
    slot.contains_valid_frame = false;
    readIdx++;
  }

  GstBuffer *buf = gst_buffer_new_allocate(nullptr, frame_sz, nullptr);
  GstMapInfo map;
  gst_buffer_map(buf, &map, GST_MAP_WRITE);
  memcpy(map.data, captured.bgr_frame_data.data(), frame_sz);
  gst_buffer_unmap(buf, &map);
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(src), buf);
  if (ret != GST_FLOW_OK)
    fprintf(stderr, "[RAW] push failed: %d\n", ret);
  return true;
}

// ── rawStreamWorker ──────────────────────────────────────────
// One instance per camera. Each worker owns one ring buffer and
// one appsrc exclusively — no starvation between cameras.
// WHY: old single-thread design blocked up to 33ms on one ring
// mutex, starving the other camera at 60fps (16ms/frame).
// Now each thread blocks only on its own condition variable.
void rawStreamWorker(DualRingBuffer &ring, GstElement *src, int camera_id,
                     int W, int H) {
  (void)ring; (void)src; (void)W; (void)H;
  // Dual-lock: outputThread is the sole producer for each camera's
  // stream src — it pushes annotated frames on every mode (FREE too,
  // just without a rect). The old raw-fills-pre-CAPTURE-gap path is
  // gone. Thread kept for wiring symmetry; just idles until shutdown.
  printf("[RAW-%s] Stream thread started (dual-lock: idle).\n",
         camera_id == 1 ? "L" : "R");
  while (g_running)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  printf("[RAW-%s] Stream thread exiting.\n", camera_id == 1 ? "L" : "R");
}

// ── outputThread ─────────────────────────────────────────────
// Drains resultQueue from trackerThread.
// Draws overlay on every frame regardless of mode.
// Pushes annotated frame to active src.
// Sends UART angle offsets and UDP telemetry every frame.
void outputThread(int camera_id,
                  cr::vtracker::CvTracker &self_tracker,
                  std::mutex &self_mtx,
                  cr::vtracker::CvTracker &peer_tracker,
                  std::mutex &peer_mtx,
                  std::vector<ResultSlot> &resultQueue,
                  std::mutex &resultMtx,
                  std::condition_variable &resultCv,
                  GstElement *stream_src,
                  GstElement *annotated_src,
                  int W, int H, int fps, bool recEnabled) {
  const char *tag = camera_id == 1 ? "L" : "R";
  const int peer_id = (camera_id == 1) ? 2 : 1;
  printf("[OUTPUT-%s] Thread started.\n", tag);
  // Reference-parameter workaround: we accept peer_tracker/peer_mtx so
  // the compiler validates the wiring, but the dual-lock cross-check
  // below only needs to READ this camera's own state and CAPTURE on
  // the peer. All access to peer_tracker goes through peer_mtx.
  (void)peer_tracker;

  while (g_running) {
    ResultSlot result;

    // ── Wait for result from trackerThread ────────────────────────
    {
      std::unique_lock<std::mutex> lk(resultMtx);
      resultCv.wait_for(lk, std::chrono::milliseconds(100),
                        [&] { return !resultQueue.empty() || !g_running; });
      if (!g_running && resultQueue.empty())
        break;
      if (resultQueue.empty())
        continue;

      // WHY: prevent unbounded queue growth — drop oldest frames
      // if output thread falls behind tracker.
      while (resultQueue.size() > 10)
        resultQueue.erase(resultQueue.begin());

      result = std::move(resultQueue.front());
      resultQueue.erase(resultQueue.begin());
    }

    VTrackerParams &p = result.params;
    int frame_sz = W * H * 3;

    // Publish this camera's rect centre for handoff / diagnostics.
    if (camera_id == 1) {
      g_last_rect_x.store(p.rectX, std::memory_order_relaxed);
      g_last_rect_y.store(p.rectY, std::memory_order_relaxed);
    }

    // ── Dual-lock cross-view seed ──────────────────────────────────
    // Runs on every frame of THIS camera. When self is TRACKING with
    // good confidence AND the peer tracker is FREE/LOST, project the
    // self rect via the handoff homography and CAPTURE on the peer.
    // Both trackers then run independently until one loses again.
    //
    // Two-second cooldown prevents thrashing when both cameras briefly
    // see the target from different quality angles. Peer state is
    // read under peer_mtx so we don't race with peer_tracker's own
    // processFrame.
    // Diag: log why we're NOT seeding when self would otherwise want to.
    // Rate-limited to one line per second per camera so we don't spam.
    if (p.mode == 1 && p.detectionProbability > 0.5f && !g_handoff.ready()) {
      thread_local auto s_last_diag = std::chrono::steady_clock::now() - std::chrono::hours(1);
      auto now_diag = std::chrono::steady_clock::now();
      if (now_diag - s_last_diag > std::chrono::seconds(1)) {
        printf(LOG_YELLOW "[HANDOFF-%s] not ready — stereo_calib.json failed to load\n"
               LOG_RESET, tag);
        fflush(stdout);
        s_last_diag = now_diag;
      }
    }
    // Track the most recent frame at which this source had a confident
    // fix (mode == TRACKING and det above the seed threshold). Used
    // below to allow a "grace" seed while the source is briefly LOST
    // but still remembers where the target was.
    thread_local int64_t s_last_good_frame = -1;
    constexpr float SEED_DET_MIN = 0.4f;
    constexpr int   RECENTLY_GOOD_FRAMES = 10;
    if (p.mode == 1 && p.detectionProbability > SEED_DET_MIN) {
      s_last_good_frame = (int64_t)result.frameId;
    }
    const bool recently_good = (s_last_good_frame >= 0) &&
        ((int64_t)result.frameId - s_last_good_frame <= RECENTLY_GOOD_FRAMES);

    // Seed while the source is actively tracking OR while it's LOST
    // but had a confident fix in the last N frames — the tracker's
    // rectX/rectY is still meaningful during LOST (frozen last-good
    // position), and this is exactly the FOV-edge scenario where the
    // peer most needs a hand-off.
    const bool src_worth_seeding =
        g_handoff.ready() &&
        ((p.mode == 1 && p.detectionProbability > SEED_DET_MIN) ||
         (p.mode == 2 && recently_good));

    // Per-camera cooldown: was `static` before, shared between L→R and
    // R→L seed paths, so one direction's cooldown blocked the other.
    // thread_local restores per-direction pacing.
    thread_local std::chrono::steady_clock::time_point s_last_seed =
        std::chrono::steady_clock::now() - std::chrono::hours(1);
    const auto now = std::chrono::steady_clock::now();
    const auto cooldown = std::chrono::seconds(2);

    if (src_worth_seeding) {
      if (now - s_last_seed > cooldown) {
        // Peek at the peer's current mode without blocking long.
        int peer_mode = -1;
        {
          std::lock_guard<std::mutex> lk(peer_mtx);
          cr::vtracker::VTrackerParams pp;
          peer_tracker.getParams(pp);
          peer_mode = pp.mode;
        }
        // Seed only when peer isn't already tracking.
        const bool peer_needs_seed =
            (peer_mode == 0 || peer_mode == 2);   // FREE or LOST
        // Homography seed path requires a positive depth. In DISTANCE-FREE
        // mode (depth <= 0) the epipolar-line + DINOv2 search path (added in
        // a later stage) takes over; until then, skip seeding so we don't
        // spam projection failures.
        if (peer_needs_seed && g_handoff.homographyReady()) {
          double dst_u = 0.0, dst_v = 0.0;
          const bool proj_ok = (camera_id == 1)
              ? g_handoff.projectLtoR((double)p.rectX, (double)p.rectY,
                                       dst_u, dst_v)
              : g_handoff.projectRtoL((double)p.rectX, (double)p.rectY,
                                       dst_u, dst_v);
          if (!proj_ok) {
            thread_local auto s_last_proj_err =
                std::chrono::steady_clock::now() - std::chrono::hours(1);
            if (now - s_last_proj_err > std::chrono::seconds(1)) {
              printf(LOG_YELLOW "[HANDOFF-%s] projection failed for self(%d,%d)\n"
                     LOG_RESET, tag, p.rectX, p.rectY);
              fflush(stdout);
              s_last_proj_err = now;
            }
          } else if (dst_u < 0 || dst_u >= W || dst_v < 0 || dst_v >= H) {
            thread_local auto s_last_oob =
                std::chrono::steady_clock::now() - std::chrono::hours(1);
            if (now - s_last_oob > std::chrono::seconds(1)) {
              printf(LOG_YELLOW "[HANDOFF-%s] projected pixel (%.1f,%.1f) "
                     "outside peer frame %dx%d — target not yet in overlap\n"
                     LOG_RESET, tag, dst_u, dst_v, W, H);
              fflush(stdout);
              s_last_oob = now;
            }
          }
          if (proj_ok && dst_u >= 0 && dst_u < W &&
              dst_v >= 0 && dst_v < H) {
            printf(LOG_CYAN "[HANDOFF]" LOG_RESET
                   " %s→%s  self(%d, %d) → peer(%.1f, %.1f)\n",
                   camera_id == 1 ? "BORE" : "DEPR",
                   peer_id == 1 ? "BORE" : "DEPR",
                   p.rectX, p.rectY, dst_u, dst_v);
            // Focal-ratio-scaled destination size (matches what the
            // tracker's handoff overlay does for size).
            double fx_self = (camera_id == 1) ? g_handoff.fxL() : g_handoff.fxR();
            double fy_self = (camera_id == 1) ? g_handoff.fyL() : g_handoff.fyR();
            double fx_peer = (camera_id == 1) ? g_handoff.fxR() : g_handoff.fxL();
            double fy_peer = (camera_id == 1) ? g_handoff.fyR() : g_handoff.fyL();
            double dst_w = (fx_self > 0) ? p.rectWidth  * fx_peer / fx_self
                                          : (double)p.rectWidth;
            double dst_h = (fy_self > 0) ? p.rectHeight * fy_peer / fy_self
                                          : (double)p.rectHeight;
            if (recEnabled && g_rec) {
              g_rec->logHandoff(result.frameId, result.pts,
                                camera_id, peer_id,
                                p.rectX, p.rectY,
                                dst_u, dst_v, dst_w, dst_h);
            }
            // Compute the epipolar line the target is constrained to
            // in the peer's image (exact — doesn't depend on the plane-
            // depth assumption). We hand it to the peer alongside the
            // seed CAPTURE so its correlation peak selection prefers
            // on-line candidates for a short window, rejecting the
            // brightest-but-wrong distractors the peer's own filter
            // hasn't yet learned to veto.
            double epi_a = 0.0, epi_b = 0.0, epi_c = 0.0;
            const bool epi_ok = (camera_id == 1)
                ? g_handoff.epipolarLineInR((double)p.rectX, (double)p.rectY,
                                            epi_a, epi_b, epi_c)
                : g_handoff.epipolarLineInL((double)p.rectX, (double)p.rectY,
                                            epi_a, epi_b, epi_c);

            // Export self's DNN template bank OUTSIDE the peer lock
            // (uses self's own mutex internally). We snapshot appearance
            // history now so the peer starts life with a mature bank —
            // seed sample + high-confidence follow-ups the source has
            // accumulated across scales / lighting — instead of a
            // single-frame template drawn from one downsampled crop.
            //
            // cuda_library and lockon both expose bank export and the
            // epipolar constraint (cuda_library got the same public
            // API port as lockon). The v1 baseline (constant_robotics)
            // has no DNN and no bank, so gate the whole enrichment
            // path on TRACKER_V2 — true for both V2 engines, false for
            // v1 — and let v1 keep the geometric seed only.
            std::vector<unsigned char> bank_bytes;
            bool bank_ok = false;
#ifdef TRACKER_V2
            bank_ok = self_tracker.exportTemplateBank(bank_bytes);
#endif

            {
              std::lock_guard<std::mutex> lk(peer_mtx);
              // Reset first so the homography seed replaces any prior
              // template on the peer, rather than being biased by it.
              peer_tracker.executeCommand(
                  cr::vtracker::VTrackerCommand::RESET);
              peer_tracker.executeCommand(
                  cr::vtracker::VTrackerCommand::CAPTURE,
                  (float)dst_u, (float)dst_v, -1.0f);
              // Import bank AFTER CAPTURE so it isn't cleared by the
              // capture path's bank.clear() + seed() sequence. The
              // homography seed rect is what the peer's correlation
              // filter locks to; the imported bank is what the DNN
              // verifier / arbitrator consults on subsequent frames.
#ifdef TRACKER_V2
              if (bank_ok) {
                if (peer_tracker.importTemplateBank(bank_bytes)) {
                  printf(LOG_CYAN "[HANDOFF]" LOG_RESET
                         " %s→%s  transferred DNN bank (%zu bytes)\n",
                         camera_id == 1 ? "BORE" : "DEPR",
                         peer_id == 1 ? "BORE" : "DEPR",
                         bank_bytes.size());
                } else {
                  printf(LOG_YELLOW "[HANDOFF]" LOG_RESET
                         " bank import rejected — magic/version/size\n");
                }
              }
              if (epi_ok) {
                // Half-width: generous enough to absorb calibration
                // slack (~5% of the smaller image dimension). TTL:
                // ~1 second at 60 fps — long enough to cover the
                // brief window before the peer's own filter is
                // confident, short enough to not fight it later.
                const float half_w = (float)std::min(W, H) * 0.05f;
                peer_tracker.setEpipolarConstraint(
                    (float)epi_a, (float)epi_b, (float)epi_c,
                    half_w, 60);
                printf(LOG_CYAN "[HANDOFF]" LOG_RESET
                       " %s→%s  epipolar band a=%.3f b=%.3f c=%.1f  "
                       "±%.0fpx / 60f\n",
                       camera_id == 1 ? "BORE" : "DEPR",
                       peer_id == 1 ? "BORE" : "DEPR",
                       epi_a, epi_b, epi_c, half_w);
              }
#else
              (void)bank_ok;
              (void)epi_ok;
              (void)epi_a; (void)epi_b; (void)epi_c;
#endif
            }
            s_last_seed = now;
          }
        }
      }
    }

    // ── UART / UDP telemetry — only the PRIMARY camera writes ────
    // Single wire to the flight controller means one tracker's angles
    // may fly at a time. g_selected_camera stores the current primary;
    // both trackers run, but only the primary's angles reach the FC.
    //
    // Auto-primary-flip: if THIS camera is TRACKING with good det AND
    // the current primary is us-not-tracking OR primary is peer-and-
    // peer-not-tracking, claim the primary role. This makes handoff
    // fully automatic — boresight loses at frame edge, depression is
    // already TRACKING (auto-seeded via homography earlier), depression
    // silently becomes primary and the FC keeps getting angles.
    if (p.mode == 1 && p.detectionProbability > 0.5f) {
      int cur_primary = g_selected_camera.load();
      if (cur_primary != camera_id) {
        int peer_mode = -1;
        {
          std::lock_guard<std::mutex> lk(peer_mtx);
          cr::vtracker::VTrackerParams pp;
          peer_tracker.getParams(pp);
          peer_mode = pp.mode;
        }
        if (cur_primary == peer_id && peer_mode != 1) {
          if (g_selected_camera.compare_exchange_strong(cur_primary, camera_id))
            printf(LOG_YELLOW "[PRIMARY-FLIP]" LOG_RESET
                   " %s took primary from %s (peer mode=%d)\n",
                   tag, peer_id == 1 ? "L" : "R", peer_mode);
        } else if (cur_primary == 0) {
          int expected0 = 0;
          if (g_selected_camera.compare_exchange_strong(expected0, camera_id))
            printf(LOG_YELLOW "[PRIMARY-CLAIM]" LOG_RESET
                   " %s claimed unassigned primary\n", tag);
        }
      }
    }

    // ── Publish this camera's live state for fusion ──────────────
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    // Only publish "active" detection when actually TRACKING; LOST /
    // INERTIAL frames have stale rects and would poison the fusion
    // weight. Publish det=0 for those so the fuser treats us as absent.
    const float pub_det = (p.mode == 1) ? p.detectionProbability : 0.0f;
    if (camera_id == 1) {
      g_state_L_ang_x.store(result.angleOffX, std::memory_order_relaxed);
      g_state_L_ang_y.store(result.angleOffY, std::memory_order_relaxed);
      g_state_L_pix_x.store(result.pixOffX,   std::memory_order_relaxed);
      g_state_L_pix_y.store(result.pixOffY,   std::memory_order_relaxed);
      g_state_L_det.store(pub_det,            std::memory_order_relaxed);
      g_state_L_stamp_ms.store(now_ms,        std::memory_order_relaxed);
    } else {
      g_state_R_ang_x.store(result.angleOffX, std::memory_order_relaxed);
      g_state_R_ang_y.store(result.angleOffY, std::memory_order_relaxed);
      g_state_R_pix_x.store(result.pixOffX,   std::memory_order_relaxed);
      g_state_R_pix_y.store(result.pixOffY,   std::memory_order_relaxed);
      g_state_R_det.store(pub_det,            std::memory_order_relaxed);
      g_state_R_stamp_ms.store(now_ms,        std::memory_order_relaxed);
    }

    // ── UART: single fused stream, driven only by camera 1 ───────
    // Camera 1's output thread owns the UART wire. It reads both
    // cameras' latest state and produces one angle stream:
    //   both TRACKING within TTL   → det-weighted average
    //   only one TRACKING          → that camera's angles verbatim
    //   neither TRACKING           → send self mode/pixels so the FC
    //                                still sees a LOST/FREE beat
    // This removes the discontinuity at handoff moments — the FC no
    // longer sees an angle jump when primary flips L↔R.
    if (camera_id == 1) {
      float det_L = g_state_L_det.load(std::memory_order_relaxed);
      float det_R = g_state_R_det.load(std::memory_order_relaxed);
      int64_t st_L = g_state_L_stamp_ms.load(std::memory_order_relaxed);
      int64_t st_R = g_state_R_stamp_ms.load(std::memory_order_relaxed);
      const bool fresh_L = (st_L > 0) && (now_ms - st_L) <= FUSION_TTL_MS;
      const bool fresh_R = (st_R > 0) && (now_ms - st_R) <= FUSION_TTL_MS;
      const bool active_L = fresh_L && det_L > 0.0f;
      const bool active_R = fresh_R && det_R > 0.0f;

      float fused_ax = result.angleOffX;
      float fused_ay = result.angleOffY;
      int   fused_px = result.pixOffX;
      int   fused_py = result.pixOffY;
      float fused_det = p.detectionProbability;
      int   fused_mode = p.mode;

      if (active_L && active_R) {
        float wL = det_L, wR = det_R;
        float wsum = wL + wR;
        fused_ax = (wL * g_state_L_ang_x.load(std::memory_order_relaxed) +
                    wR * g_state_R_ang_x.load(std::memory_order_relaxed)) / wsum;
        fused_ay = (wL * g_state_L_ang_y.load(std::memory_order_relaxed) +
                    wR * g_state_R_ang_y.load(std::memory_order_relaxed)) / wsum;
        fused_px = (int)((wL * g_state_L_pix_x.load(std::memory_order_relaxed) +
                          wR * g_state_R_pix_x.load(std::memory_order_relaxed)) / wsum);
        fused_py = (int)((wL * g_state_L_pix_y.load(std::memory_order_relaxed) +
                          wR * g_state_R_pix_y.load(std::memory_order_relaxed)) / wsum);
        fused_det = std::max(det_L, det_R);
        fused_mode = 1;   // TRACKING
      } else if (active_R && !active_L) {
        fused_ax = g_state_R_ang_x.load(std::memory_order_relaxed);
        fused_ay = g_state_R_ang_y.load(std::memory_order_relaxed);
        fused_px = g_state_R_pix_x.load(std::memory_order_relaxed);
        fused_py = g_state_R_pix_y.load(std::memory_order_relaxed);
        fused_det = det_R;
        fused_mode = 1;
      }
      // active_L only → self values already set above.

      if (fused_mode >= 1 && fused_mode <= 3) {
        uartSendAngle(g_uartFd, fused_mode, fused_ax, fused_ay,
                      fused_det, fused_px, fused_py,
                      g_target_confirmed.load());
      }
    }

    // ── UDP telemetry: each camera → its OWN GUI panel, always ───
    // Removes the old is_primary gate so S1 and S2 panels both show
    // live numbers even when the other camera is currently primary.
    const int my_telem_port = (camera_id == 2) ? g_rightCtrlPort
                                                : g_leftCtrlPort;
    {
      std::lock_guard<std::mutex> lk(self_mtx);
      sendTelemetry(p, result.frameId, result.angleOffX, result.angleOffY,
                    result.pixOffX, result.pixOffY, W, H, my_telem_port);
    }
    // Kept for the recording-side check below.
    const int primary_cam = g_selected_camera.load();
    const bool is_primary = (primary_cam == camera_id);

    // ── Flight recorder — each camera thread logs INDEPENDENTLY ──
    // Every event carries THIS camera's id so replay can reconstruct
    // both trackers' state per frame. Static s_prevMode is per-thread
    // (thread_local) so mode-change events reflect only this camera.
    if (recEnabled && g_rec) {
      thread_local int s_prevMode = 0;
      if (p.mode != s_prevMode) {
        g_rec->logModeChange(result.frameId, result.pts, camera_id,
                             s_prevMode, p.mode);
        s_prevMode = p.mode;
      }
      const bool engaged = (p.mode >= 1 && p.mode <= 3);
      if (engaged || result.frameId % 30 == 0) {
        g_rec->logAngle(result.frameId, result.pts, camera_id,
                        result.angleOffX, result.angleOffY,
                        p.detectionProbability,
                        p.rectX, p.rectY, p.rectWidth, p.rectHeight);
      }
    }

    // Raw recording is now handled directly in the capture threads
    // (dual_capture.cpp) — each camera pushes into its own raw_left.mkv /
    // raw_right.mkv from boot, independent of tracker state. Nothing to
    // do here for the raw stream. Annotated push (below) stays.

    // ── Draw overlay ──────────────────────────────────────────────
    cv::Mat bgr(H, W, CV_8UC3, result.bgrData.data());
    cv::Scalar col = modeColor(p.mode);

    // Tracking rect
    cv::Rect tRect(p.rectX - p.rectWidth / 2, p.rectY - p.rectHeight / 2,
                   p.rectWidth, p.rectHeight);
    cv::rectangle(bgr, tRect, col, 2);

    // Crosshair at frame center
    int cx = W / 2, cy = H / 2;
    cv::Scalar crossColor(0, 0, 255);  // red (BGR)
    const int crossThick = 2;
    cv::line(bgr, {cx - 20, cy}, {cx - 6, cy}, crossColor, crossThick, cv::LINE_AA);
    cv::line(bgr, {cx + 6, cy}, {cx + 20, cy}, crossColor, crossThick, cv::LINE_AA);
    cv::line(bgr, {cx, cy - 20}, {cx, cy - 6}, crossColor, crossThick, cv::LINE_AA);
    cv::line(bgr, {cx, cy + 6}, {cx, cy + 20}, crossColor, crossThick, cv::LINE_AA);

    // Object rect when TRACKING
    if (p.mode == 1) {
      cv::Rect oRect(p.objectX - p.objectWidth / 2,
                     p.objectY - p.objectHeight / 2, p.objectWidth,
                     p.objectHeight);
      cv::rectangle(bgr, oRect, cv::Scalar(255, 255, 255), 1);
    }

    // Rect center marker
    cv::line(bgr, {p.rectX - 10, p.rectY}, {p.rectX + 10, p.rectY}, col, 1);
    cv::line(bgr, {p.rectX, p.rectY - 10}, {p.rectX, p.rectY + 10}, col, 1);

    // Status text — used for the on-frame overlay AND, once per second,
    // for the console. Previously this printf ran every frame with \r
    // and got scrambled by other stdout output (NvMMLite, RAW push,
    // CTRL messages). Newline + 1 Hz rate limit makes it clean.
    char txt[192];
    snprintf(txt, sizeof(txt), "%s  det:%.2f  %dus  ang:(%.1f,%.1f)",
             modeStr(p.mode), p.detectionProbability, p.processingTimeMks,
             result.angleOffX, result.angleOffY);
    {
      thread_local auto s_last = std::chrono::steady_clock::now();
      auto now = std::chrono::steady_clock::now();
      if (now - s_last >= std::chrono::seconds(1)) {
        // Color by tracker mode: FREE gray, TRACKING green, LOST red,
        // INERTIAL yellow, STATIC dim.
        const char *col;
        switch (p.mode) {
          case 1:  col = LOG_GREEN;   break;  // TRACKING
          case 2:  col = LOG_RED;     break;  // LOST
          case 3:  col = LOG_YELLOW;  break;  // INERTIAL
          case 4:  col = LOG_DIM;     break;  // STATIC
          default: col = LOG_GRAY;    break;  // FREE
        }
        printf("%s[%s] %s%s\n", col, tag, txt, LOG_RESET);
        fflush(stdout);
        s_last = now;
      }
    }
    // Outdoor-readable HUD: larger, thicker, high-contrast yellow with
    // a black outline so it survives sky/grass/foliage backgrounds.
    {
      const cv::Point org(12, 34);
      const double scale = 0.75;
      const int thick = 2;
      cv::putText(bgr, txt, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                  cv::Scalar(0, 0, 0), thick + 3, cv::LINE_AA);   // outline
      cv::putText(bgr, txt, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                  cv::Scalar(0, 255, 255), thick, cv::LINE_AA);   // yellow
    }

    // ── Push annotated frame to THIS camera's own stream src ─────
    // Both cameras always push their annotated frames — the GUI now
    // shows two boxes when both trackers have TRACKING mode (overlap
    // region), one when only one has lock, none when both are LOST.
    if (stream_src) {
      GstBuffer *outBuf = gst_buffer_new_allocate(nullptr, frame_sz, nullptr);
      GstMapInfo outMap;
      gst_buffer_map(outBuf, &outMap, GST_MAP_WRITE);
      memcpy(outMap.data, bgr.data, frame_sz);
      gst_buffer_unmap(outBuf, &outMap);
      if (gst_app_src_push_buffer(GST_APP_SRC(stream_src), outBuf) !=
          GST_FLOW_OK)
        fprintf(stderr, "[OUTPUT-%s] stream push failed.\n", tag);
    }

    // ── Annotated recording push ─────────────────────────────────
    // Each camera writes its own annotated file (annotated_left.mkv /
    // annotated_right.mkv). Both fill unconditionally so replay of a
    // handoff shows the overlay on the source AND destination view.
    if (recEnabled && annotated_src) {
      GstBuffer *annBuf = gst_buffer_new_allocate(nullptr, frame_sz, nullptr);
      GstMapInfo annMap;
      gst_buffer_map(annBuf, &annMap, GST_MAP_WRITE);
      memcpy(annMap.data, bgr.data, frame_sz);
      gst_buffer_unmap(annBuf, &annMap);
      GST_BUFFER_PTS(annBuf) = result.pts;
      GST_BUFFER_DURATION(annBuf) = result.frameDur;
      if (gst_app_src_push_buffer(GST_APP_SRC(annotated_src), annBuf) !=
          GST_FLOW_OK) {
        fprintf(stderr, "[REC-%s] annotated push failed.\n", tag);
      }
    }
  }

  printf("[OUTPUT-%s] Thread exiting.\n", tag);
}
