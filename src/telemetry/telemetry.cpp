#include "telemetry.h"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <cstdlib>
#include <cstring>

void sendTelemetry(const cr::vtracker::VTrackerParams &p, int frameId,
                   float angleX, float angleY, int pixOffX, int pixOffY,
                   int frameW, int frameH, int telemPort) {
  if (g_telemSock < 0)
    return;

  TelemetryPacket t{};
  t.magic = TELEMETRY_MAGIC;
  t.frame_id = (uint32_t)frameId;
  t.mode = (uint32_t)p.mode;
  t.det_prob = p.detectionProbability;
  t.proc_time_us = (uint32_t)p.processingTimeMks;
  t.rect_x = p.rectX;
  t.rect_y = p.rectY;
  t.rect_width = p.rectWidth;
  t.rect_height = p.rectHeight;
  t.object_x = p.objectX;
  t.object_y = p.objectY;
  t.object_width = p.objectWidth;
  t.object_height = p.objectHeight;
  t.angle_x_deg = angleX;
  t.angle_y_deg = angleY;
  t.pixel_offset_x = (int32_t)pixOffX;
  t.pixel_offset_y = (int32_t)pixOffY;
  t.search_win_width = p.searchWindowWidth;
  t.search_win_height = p.searchWindowHeight;
  t.lost_mode_option =
      g_tracker.getParam(cr::vtracker::VTrackerParam::LOST_MODE_OPTION);
  t.frame_buffer_size = (int32_t)g_tracker.getParam(
      cr::vtracker::VTrackerParam::FRAME_BUFFER_SIZE);
  t.max_frames_lost = (int32_t)g_tracker.getParam(
      cr::vtracker::VTrackerParam::MAX_FRAMES_IN_LOST_MODE);
  t.rect_auto_size =
      g_tracker.getParam(cr::vtracker::VTrackerParam::RECT_AUTO_SIZE);
  t.rect_auto_position =
      g_tracker.getParam(cr::vtracker::VTrackerParam::RECT_AUTO_POSITION);
  t.multiple_threads = (int32_t)g_tracker.getParam(
      cr::vtracker::VTrackerParam::MULTIPLE_THREADS);
  t.num_channels =
      (int32_t)g_tracker.getParam(cr::vtracker::VTrackerParam::NUM_CHANNELS);
  t.tracker_type =
      (int32_t)g_tracker.getParam(cr::vtracker::VTrackerParam::TYPE);

  t.custom_1 = g_tracker.getParam(cr::vtracker::VTrackerParam::CUSTOM_1);
  t.custom_2 = g_tracker.getParam(cr::vtracker::VTrackerParam::CUSTOM_2);
  t.custom_3 = g_tracker.getParam(cr::vtracker::VTrackerParam::CUSTOM_3);

#if defined(TRACKER_V2) && !defined(TRACKER_LOCKON)
  // Baseline cvtracker: EKF sigma knobs + DNN verifier all present.
  t.ekf_sigma_a = g_tracker.getParam(cr::vtracker::VTrackerParam::EKF_SIGMA_A);
  t.ekf_sigma_alpha =
      g_tracker.getParam(cr::vtracker::VTrackerParam::EKF_SIGMA_ALPHA);
  t.ekf_r_base = g_tracker.getParam(cr::vtracker::VTrackerParam::EKF_R_BASE);
  t.dnn_verify_interval = (int32_t)g_tracker.getParam(
      cr::vtracker::VTrackerParam::DNN_VERIFY_INTERVAL);
  t.dnn_veto_threshold =
      g_tracker.getParam(cr::vtracker::VTrackerParam::DNN_VETO_THRESHOLD);
  t.dnn_accept_threshold =
      g_tracker.getParam(cr::vtracker::VTrackerParam::DNN_ACCEPT_THRESHOLD);
  t.dnn_similarity = p.dnnSimilarity;
#elif defined(TRACKER_LOCKON)
  // Lockon engine: no EKF sigma knobs (internal EKF), but the DNN verifier
  // params still exist (at shifted enum ids — the enum names resolve
  // correctly regardless of numeric value).
  t.ekf_sigma_a = 0.0f;
  t.ekf_sigma_alpha = 0.0f;
  t.ekf_r_base = 0.0f;
  t.dnn_verify_interval = (int32_t)g_tracker.getParam(
      cr::vtracker::VTrackerParam::DNN_VERIFY_INTERVAL);
  t.dnn_veto_threshold =
      g_tracker.getParam(cr::vtracker::VTrackerParam::DNN_VETO_THRESHOLD);
  t.dnn_accept_threshold =
      g_tracker.getParam(cr::vtracker::VTrackerParam::DNN_ACCEPT_THRESHOLD);
  t.dnn_similarity = p.dnnSimilarity;
#else
  // v1 (constant_robotics_lib) has no EKF sigma params or DNN verifier
  t.ekf_sigma_a = 0.0f;
  t.ekf_sigma_alpha = 0.0f;
  t.ekf_r_base = 0.0f;
  t.dnn_verify_interval = 0;
  t.dnn_veto_threshold = 0.0f;
  t.dnn_accept_threshold = 0.0f;
  t.dnn_similarity = 0.0f;
#endif

  t.frame_width = (int32_t)frameW;
  t.frame_height = (int32_t)frameH;
  // Stamp the tracker engine into the (previously unused) reserved field so
  // the ground station can auto-detect which VTrackerParam enum layout is
  // live and remap param ids 16-23 accordingly. Tagged 0xE00000NN to
  // distinguish an engine-aware binary from a legacy one (which sent 0).
  //   byte 0 = engine: 0 = constant_robotics_lib (v1), 1 = cuda_library
  //            (v2), 2 = lockon.
  //   bit  8 = operator target-confirmation flag (echoes g_target_confirmed
  //            so the ground station button reflects real pipeline state).
#if defined(TRACKER_LOCKON)
  t.reserved = 0xE0000002u;
#elif defined(TRACKER_V2)
  t.reserved = 0xE0000001u;
#else
  t.reserved = 0xE0000000u;
#endif
  if (g_target_confirmed.load())
    t.reserved |= 0x00000100u;

  // WHY: outputThread is a single thread driving the shared g_tracker
  // for whichever camera is currently locked (g_selected_camera). The
  // startup value g_telemDest.sin_port = leftCtrlPort was correct only
  // when we tracked on cam 1; when cam 2 was captured, telemetry still
  // flew to port 5001 and the GUI's S2 panel stayed dark. Route per
  // packet: the locked camera decides which port receives it, so S1
  // sees its own numbers and S2 sees its own.
  //
  // g_selected_camera == 0 → not locked yet, fall back to left port
  //   so the GUI keeps its historical default.
  // g_selected_camera == 1 → left/S1 port  (typically 5001)
  // g_selected_camera == 2 → right/S2 port (typically 5003)
  // Dual-lock fused telemetry: caller passes the port for THIS camera
  // (non-zero) so each output thread routes to its own S1/S2 GUI panel
  // regardless of primary. Legacy behaviour (telemPort == 0) still
  // routes via g_selected_camera for callers that haven't migrated.
  sockaddr_in dest = g_telemDest;
  int port;
  if (telemPort > 0) {
    port = telemPort;
  } else {
    int cam = g_selected_camera.load();
    port = (cam == 2) ? g_rightCtrlPort : g_leftCtrlPort;
  }
  // Benchmark / localhost override: main.cpp set g_telemDest.sin_port
  // from TRACKER_TELEM_PORT_OVERRIDE if the env var was set. Honour
  // that here — otherwise this code path unconditionally routes back
  // to ctrl_port, which on localhost is bound by the tracker itself
  // and causes it to receive its own telemetry with wrong magic.
  const char *env = std::getenv("TRACKER_TELEM_PORT_OVERRIDE");
  if (env && env[0] != '\0') {
    int p = atoi(env);
    if (p > 0 && p < 65536) port = p;
  }
  dest.sin_port = htons(port);

  // (Removed the routing-change log. It made sense when only the
  // primary camera sent telemetry — the port changed once per manual
  // handoff, worth logging. With dual-lock the two output threads
  // both send every frame to their own port, so this line fired at
  // 120 Hz and drowned the rest of the log.)

  sendto(g_telemSock, &t, sizeof(t), 0,
         reinterpret_cast<const sockaddr *>(&dest), sizeof(dest));

}