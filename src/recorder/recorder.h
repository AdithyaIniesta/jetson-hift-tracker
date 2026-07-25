/**
 * @file recorder.h
 * @brief Flight recorder — writes meta.json and events.jsonl
 *
 * meta.json    : written once at startup, updated once at shutdown.
 *                Contains static mission context (build, cameras, tracker
 *                params, network) — enough to reproduce conditions after
 *                a crash, or to replay the session offline.
 *
 * events.jsonl : one JSON line per event, flushed immediately. Every
 *                event carries frame, t_ns and camera_id so a replay tool
 *                can index by (camera, frame) deterministically.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

// ── Forward declaration — avoid pulling in CvTracker.h here ──────────────
struct TrackerConfig {
  std::string version;
  int rect_width;
  int rect_height;
  int search_window_width;
  int search_window_height;
  int lost_mode_option;
  int num_channels;
  int multiple_threads;
  int type;
  int frame_buffer_size;
  float custom_1;
  float custom_2;
};

// ── Shutdown reasons ─────────────────────────────────────────────────────
enum class ShutdownReason {
  CLEAN,
  KILLED,
  CRASH,
};

// ── SessionInfo ─────────────────────────────────────────────────────────
// Everything meta.json needs, gathered in one place so main.cpp can
// populate it at boot and controlThread can consume it when it decides
// to create the FlightRecorder (on first CAPTURE).
struct SessionInfo {
  std::string sessionPath;

  // Frame geometry
  int W = 0;
  int H = 0;
  int fps = 0;
  float hfov = 0.0f;
  float vfov = 0.0f;

  // Cameras
  int camModel = 0;
  bool dualEnabled = false;
  std::string leftDev;
  std::string rightDev;
  int selectedCameraId = 0; // 0 = not yet locked, 1 = left, 2 = right

  // Network
  std::string clientIp;
  int leftVideoPort = 0;
  int leftCtrlPort = 0;
  int rightVideoPort = 0;
  int rightCtrlPort = 0;
  std::string uartDev;

  // Build / reproducibility — filled from CMake-injected macros
  std::string gitCommit;
  std::string buildTime;
  std::string trackerEngine;
  std::string argvLine; // raw argv[] joined by spaces
};

// ── Main class ───────────────────────────────────────────────────────────
class FlightRecorder {
public:
  explicit FlightRecorder(const SessionInfo &info);
  ~FlightRecorder();

  // Called once after tracker is configured (in controlThread, on CAPTURE).
  void writeMetaJson(const TrackerConfig &cfg);

  // ── Event loggers — each flushes immediately to disk ─────────────
  void logStart(int frame, uint64_t pts_ns, int camera_id);
  void logModeChange(int frame, uint64_t pts_ns, int camera_id, int prevMode,
                     int newMode);
  void logAngle(int frame, uint64_t pts_ns, int camera_id, float ax, float ay,
                float det_prob,
                int rect_x = 0, int rect_y = 0,
                int rect_w = 0, int rect_h = 0);
  // Full command payload — cmdType + 3 float args. Every field of
  // CmdPacket that carries operator intent lands in events.jsonl.
  void logCmd(int frame, uint64_t pts_ns, int camera_id, uint32_t cmdType,
              float arg1, float arg2, float arg3);
  void logShutdown(int frame, uint64_t pts_ns, ShutdownReason reason,
                   int totalFrames, double durationSec);
  // Automatic cross-view handoff: source camera's seed CAPTURE fired
  // on the peer. Logs the geometric evidence needed to evaluate the
  // handoff independently of tracker performance. Peer's rect w/h are
  // the focal-ratio-scaled sizes; if scaling was skipped (identical
  // intrinsics), pass 0 for both.
  void logHandoff(int frame, uint64_t pts_ns,
                  int from_cam, int to_cam,
                  int src_x, int src_y,
                  double dst_x, double dst_y,
                  double dst_w, double dst_h);

private:
  void writeEvent(const char *json);
  // Remap the tracker's global frame counter to a session-local counter
  // that starts at 1 on the very first event. This makes indices in
  // events.jsonl match the frame indices ffmpeg produces when extracting
  // raw_left.mkv / raw_right.mkv, so no offset math is needed downstream.
  int  remap(int raw_frame);

  const char *modeName(int mode) const;
  const char *camName(int model) const;
  const char *shutdownName(ShutdownReason r) const;
  const char *cmdName(uint32_t cmdType) const;

  // -1 until the first event arrives, then set so that
  //   session_frame = raw_frame - m_frame_offset
  // makes the first event's frame_id equal to 1.
  int m_frame_offset = -1;

  SessionInfo m_info;
  std::string m_metaPath;
  std::string m_eventsPath;
  FILE *m_eventsFile = nullptr;
};
