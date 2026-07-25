#include "tracker.h"

// Correctly resolving based on TRACKER_V2 flag
#ifdef TRACKER_V2
#include <cvtracker/CvTracker.h>
#else
#include "CvTracker.h"
#endif

// The lockon engine ships its own cr::vtracker::Ekf and uses it internally.
// Including the pipeline's identically-named EKF here (and linking its .cpp)
// would collide at link time, so the lockon build skips the pipeline EKF
// entirely and relies on the engine's internally-smoothed output.
#ifndef TRACKER_LOCKON
#include <ekf/Ekf.h>
#endif

#include <chrono>
#include <thread>

// Per-camera tracker thread. Refactored from the old single-selected-
// camera version to always process the assigned camera's ring buffer.
// Handoff is now handled by the output side: this thread never touches
// g_selected_camera and never waits for a "selection" — it starts
// running as soon as frames arrive and stays running until g_running
// goes false.
void trackerThread(int camera_id,
                   cr::vtracker::CvTracker &tracker,
                   std::mutex &tracker_mtx,
                   DualRingBuffer &ring,
                   std::vector<ResultSlot> &resultQueue,
                   std::mutex &resultMtx,
                   std::condition_variable &resultCv,
                   int W, int H, int fps,
                   float degPerPixelX, float degPerPixelY,
                   float centerX, float centerY) {
  const char *tag = camera_id == 1 ? "L" : "R";
  printf("[TRACKER-%s] Thread started.\n", tag);

#ifndef TRACKER_LOCKON
  cr::vtracker::Ekf ekf;
#endif
  int s_prev_mode = 0;

  int readIdx = ring.write_index.load();

  while (g_running) {
    // ── Read next frame from THIS camera's ring ───────────────────
    DualCaptureSlot captured;
    {
      std::unique_lock<std::mutex> lk(ring.ring_mutex);
      ring.ring_condition.wait_for(lk, std::chrono::milliseconds(100), [&] {
        return ring.write_index.load() > readIdx || !g_running;
      });
      if (!g_running)
        break;
      if (ring.write_index.load() <= readIdx)
        continue;

      // Skip ahead if we fell behind by more than ring size.
      if (ring.write_index.load() - readIdx > DUAL_RING_SIZE)
        readIdx = ring.write_index.load() - DUAL_RING_SIZE;

      DualCaptureSlot &slot = ring.slots[readIdx % DUAL_RING_SIZE];
      if (!slot.contains_valid_frame) {
        readIdx++;
        continue;
      }
      captured = slot;
      slot.contains_valid_frame = false;
      readIdx++;
    }

    // ── Convert BGR → YUV24 for tracker ──────────────────────────
    cr::video::Frame tkFrame(W, H, cr::video::Fourcc::YUV24);
    cv::Mat yuvMat(H, W, CV_8UC3, tkFrame.data);
    cv::Mat bgrMat(H, W, CV_8UC3, captured.bgr_frame_data.data());
    cv::cvtColor(bgrMat, yuvMat, cv::COLOR_BGR2YUV);
    tkFrame.frameId = captured.frame_sequence_id;

    // ── Process frame under this camera's mutex ──────────────────
    {
      std::lock_guard<std::mutex> lk(tracker_mtx);
      tracker.processFrame(tkFrame);
    }

    // ── Get tracker result ────────────────────────────────────────
    cr::vtracker::VTrackerParams p;
    tracker.getParams(p);

    // Publish this camera's mode. g_tracker_mode is a legacy global
    // used by rawStreamWorker to decide when outputThread owns the
    // RTP src. For dual-lock we OR the two modes so RTP is skipped
    // when EITHER tracker owns the stream.
    if (camera_id == 1) {
      g_tracker_mode_L.store(p.mode);
      g_tracker_mode.store(p.mode);   // legacy mirror of L
    } else {
      g_tracker_mode_R.store(p.mode);
    }

    // EKF read/step (baseline v2 only). Lockon runs its own internally.
#if defined(TRACKER_V2) && !defined(TRACKER_LOCKON)
    bool native_ekf_active =
        (tracker.getParam(cr::vtracker::VTrackerParam::ENABLE_EKF) != 0.0f);

    if (native_ekf_active) {
      float sigma_a =
          tracker.getParam(cr::vtracker::VTrackerParam::EKF_SIGMA_A);
      float sigma_alpha =
          tracker.getParam(cr::vtracker::VTrackerParam::EKF_SIGMA_ALPHA);
      float r_base =
          tracker.getParam(cr::vtracker::VTrackerParam::EKF_R_BASE);

      if (p.mode == 1) {
        if (!ekf.initialized() || s_prev_mode != 1)
          ekf.initialize(p.rectX, p.rectY);
        ekf.predict(1.0f, sigma_a, sigma_alpha);
        ekf.update(p.rectX, p.rectY, p.detectionProbability, r_base);
      } else if (ekf.initialized()) {
        ekf.predict(1.0f, sigma_a, sigma_alpha);
      }
    }
#else
    const bool native_ekf_active = false;
#ifndef TRACKER_LOCKON
    (void)ekf;
#endif
#endif
    s_prev_mode = p.mode;

#ifndef TRACKER_LOCKON
    float filtered_x =
        (native_ekf_active && ekf.initialized()) ? ekf.px() : (float)p.rectX;
    float filtered_y =
        (native_ekf_active && ekf.initialized()) ? ekf.py() : (float)p.rectY;
#else
    float filtered_x = (float)p.rectX;
    float filtered_y = (float)p.rectY;
#endif

    int pixOffX = (int)filtered_x - (int)centerX;
    int pixOffY = (int)filtered_y - (int)centerY;
    float angleOffX = (float)pixOffX * degPerPixelX;
    float angleOffY = (float)pixOffY * degPerPixelY;

    // ── Push result to this camera's output queue ────────────────
    {
      std::lock_guard<std::mutex> lk(resultMtx);
      ResultSlot result;
      result.bgrData = std::move(captured.bgr_frame_data);
      result.params = p;
      result.frameId = captured.frame_sequence_id;
      result.pts = captured.pipeline_timestamp_ns;
      result.frameDur = gst_util_uint64_scale_int(1, GST_SECOND, fps);
      result.angleOffX = angleOffX;
      result.angleOffY = angleOffY;
      result.pixOffX = pixOffX;
      result.pixOffY = pixOffY;
      resultQueue.push_back(std::move(result));
    }
    resultCv.notify_one();
  }

  printf("[TRACKER-%s] Thread exiting.\n", tag);
}
