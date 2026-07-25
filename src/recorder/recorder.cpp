/**
 * @file recorder.cpp
 * @brief Flight recorder implementation — meta.json + events.jsonl
 */

#include "recorder.h"

#include <unistd.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// meta.json schema version. Bump when field shape changes so replay
// tools can tell old recordings from new.
static constexpr int META_SCHEMA_VERSION = 2;

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

FlightRecorder::FlightRecorder(const SessionInfo &info) : m_info(info) {
  m_metaPath = m_info.sessionPath + "/meta.json";
  m_eventsPath = m_info.sessionPath + "/events.jsonl";

  m_eventsFile = fopen(m_eventsPath.c_str(), "a");
  if (!m_eventsFile) {
    fprintf(stderr, "[REC] Cannot open events file: %s\n",
            m_eventsPath.c_str());
  }
}

FlightRecorder::~FlightRecorder() {
  if (m_eventsFile) {
    fclose(m_eventsFile);
    m_eventsFile = nullptr;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// meta.json — written once at startup, patched at shutdown / on lock
// ════════════════════════════════════════════════════════════════════════════

// Escape backslashes and double-quotes so free-form strings (argv, paths)
// survive being embedded in a JSON string literal. Also strips control
// chars — anything <0x20 becomes a space.
static std::string jsonEscape(const std::string &in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    if (c == '\\' || c == '"') { out += '\\'; out += c; }
    else if ((unsigned char)c < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

void FlightRecorder::writeMetaJson(const TrackerConfig &cfg) {
  float dpx = (m_info.W > 0) ? m_info.hfov / (float)m_info.W : 0.0f;
  float dpy = (m_info.H > 0) ? m_info.vfov / (float)m_info.H : 0.0f;

  std::time_t now = std::time(nullptr);
  char dt[32];
  std::strftime(dt, sizeof(dt), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

  const std::string argvEsc = jsonEscape(m_info.argvLine);
  const std::string sessionPathEsc = jsonEscape(m_info.sessionPath);
  const std::string leftDevEsc = jsonEscape(m_info.leftDev);
  const std::string rightDevEsc = jsonEscape(m_info.rightDev);
  const std::string clientIpEsc = jsonEscape(m_info.clientIp);
  const std::string uartDevEsc = jsonEscape(m_info.uartDev);
  const std::string gitCommitEsc = jsonEscape(m_info.gitCommit);
  const std::string buildTimeEsc = jsonEscape(m_info.buildTime);
  const std::string engineEsc = jsonEscape(m_info.trackerEngine);
  const std::string versionEsc = jsonEscape(cfg.version);

  FILE *f = fopen(m_metaPath.c_str(), "w");
  if (!f) {
    fprintf(stderr, "[REC] Cannot open meta.json: %s\n", m_metaPath.c_str());
    return;
  }

  const char *session_id = m_info.sessionPath.c_str() +
                           m_info.sessionPath.rfind('/') + 1;

  // Files block — names track what main.cpp actually writes on disk.
  // Dual mode records both cameras continuously (raw_left.mkv +
  // raw_right.mkv). Single-camera modes keep the historical raw.mkv
  // name so downstream tooling is unaffected. Annotated is single
  // regardless — overlay is only drawn on the tracked camera.
  const char *raw_mkv       = m_info.dualEnabled ? ""              : "raw.mkv";
  const char *raw_mkv_left  = m_info.dualEnabled ? "raw_left.mkv"  : "";
  const char *raw_mkv_right = m_info.dualEnabled ? "raw_right.mkv" : "";
  const char *annotated_mkv = "annotated.mkv";

  fprintf(f,
          "{\n"
          "  \"schema_version\": %d,\n"

          // ── Build / reproducibility ─────────────────────────────
          "  \"build\": {\n"
          "    \"git_commit\":     \"%s\",\n"
          "    \"build_time\":     \"%s\",\n"
          "    \"tracker_engine\": \"%s\",\n"
          "    \"argv\":           \"%s\"\n"
          "  },\n"

          // ── Session ────────────────────────────────────────────────
          "  \"session\": {\n"
          "    \"session_id\":      \"%s\",\n"
          "    \"scenario_id\":     \"\",\n"
          "    \"run_number\":      1,\n"
          "    \"date_time_utc\":   \"%s\",\n"
          "    \"pilot\":           \"\",\n"
          "    \"observer\":        \"\",\n"
          "    \"notes\":           \"\"\n"
          "  },\n"

          // ── Files ──────────────────────────────────────────────────
          "  \"files\": {\n"
          "    \"raw_mkv\":         \"%s\",\n"
          "    \"raw_mkv_left\":    \"%s\",\n"
          "    \"raw_mkv_right\":   \"%s\",\n"
          "    \"annotated_mkv\":   \"%s\",\n"
          "    \"events_jsonl\":    \"events.jsonl\",\n"
          "    \"session_path\":    \"%s\"\n"
          "  },\n"

          // ── Cameras ────────────────────────────────────────────────
          // Both devices always recorded so replay can tell which was
          // available regardless of which one the operator locked.
          "  \"cameras\": {\n"
          "    \"dual_enabled\":       %s,\n"
          "    \"selected_camera_id\": %d,\n"
          "    \"model\":              \"%s\",\n"
          "    \"left_device\":        \"%s\",\n"
          "    \"right_device\":       \"%s\",\n"
          "    \"resolution_w\":       %d,\n"
          "    \"resolution_h\":       %d,\n"
          "    \"fps\":                %d,\n"
          "    \"hfov_deg\":           %.1f,\n"
          "    \"vfov_deg\":           %.1f,\n"
          "    \"deg_per_pixel_x\":    %.6f,\n"
          "    \"deg_per_pixel_y\":    %.6f\n"
          "  },\n"

          // ── Tracker ────────────────────────────────────────────────
          "  \"tracker\": {\n"
          "    \"version\":               \"%s\",\n"
          "    \"rect_width\":            %d,\n"
          "    \"rect_height\":           %d,\n"
          "    \"search_window_width\":   %d,\n"
          "    \"search_window_height\":  %d,\n"
          "    \"lost_mode_option\":      %d,\n"
          "    \"num_channels\":          %d,\n"
          "    \"multiple_threads\":      %d,\n"
          "    \"type\":                  %d,\n"
          "    \"frame_buffer_size\":     %d,\n"
          "    \"custom_1\":              %.4f,\n"
          "    \"custom_2\":              %.4f\n"
          "  },\n"

          // ── Network ────────────────────────────────────────────────
          "  \"network\": {\n"
          "    \"jetson_ip\":           \"%s\",\n"
          "    \"left_video_port\":     %d,\n"
          "    \"left_ctrl_port\":      %d,\n"
          "    \"right_video_port\":    %d,\n"
          "    \"right_ctrl_port\":     %d,\n"
          "    \"uart_device\":         \"%s\"\n"
          "  },\n"

          // ── Site / weather / sun / airframe / gimbal / ground_truth
          "  \"site\": {\n"
          "    \"name\": \"\", \"lat\": 0.0, \"lon\": 0.0, \"terrain\": \"\"\n"
          "  },\n"
          "  \"weather\": {\n"
          "    \"sky\": \"\", \"visibility_km\": 0, \"wind_speed_kmh\": 0,\n"
          "    \"wind_dir_deg\": 0, \"temp_c\": 0, \"humidity_pct\": 0\n"
          "  },\n"
          "  \"sun\":       { \"azimuth_deg\": 0, \"elevation_deg\": 0 },\n"
          "  \"airframe\":  { \"aircraft_id\": \"\", \"payload_sn\": \"\", "
          "\"camera_mount\": \"\" },\n"
          "  \"gimbal\":    { \"mode\": \"\", \"pitch_deg\": 0, "
          "\"yaw_bias_deg\": 0 },\n"
          "  \"ground_truth\": { \"method\": \"\", \"notes\": \"\" },\n"

          // ── Results — patched at shutdown ──────────────────────────
          "  \"results\": {\n"
          "    \"total_frames\":        0,\n"
          "    \"duration_seconds\":    0.0,\n"
          "    \"pass_count_planned\":  0,\n"
          "    \"pass_count_actual\":   0,\n"
          "    \"shutdown_reason\":     \"UNKNOWN\"\n"
          "  }\n"
          "}\n",

          META_SCHEMA_VERSION,
          // build
          gitCommitEsc.c_str(), buildTimeEsc.c_str(), engineEsc.c_str(),
          argvEsc.c_str(),
          // session
          session_id, dt,
          // files
          raw_mkv, raw_mkv_left, raw_mkv_right, annotated_mkv,
          sessionPathEsc.c_str(),
          // cameras
          m_info.dualEnabled ? "true" : "false", m_info.selectedCameraId,
          camName(m_info.camModel), leftDevEsc.c_str(), rightDevEsc.c_str(),
          m_info.W, m_info.H, m_info.fps, m_info.hfov, m_info.vfov, dpx, dpy,
          // tracker
          versionEsc.c_str(), cfg.rect_width, cfg.rect_height,
          cfg.search_window_width, cfg.search_window_height,
          cfg.lost_mode_option, cfg.num_channels, cfg.multiple_threads,
          cfg.type, cfg.frame_buffer_size, cfg.custom_1, cfg.custom_2,
          // network
          clientIpEsc.c_str(), m_info.leftVideoPort, m_info.leftCtrlPort,
          m_info.rightVideoPort, m_info.rightCtrlPort, uartDevEsc.c_str());

  fclose(f);
}

// ════════════════════════════════════════════════════════════════════════════
// events.jsonl — one line per event, flushed immediately
// ════════════════════════════════════════════════════════════════════════════

void FlightRecorder::writeEvent(const char *json) {
  if (!m_eventsFile) return;
  fprintf(m_eventsFile, "%s\n", json);
  fflush(m_eventsFile);
  fsync(fileno(m_eventsFile));
}

// Remap the tracker's global g_frameId to a 1-based session-local index,
// anchored on the first event this recorder ever sees.
int FlightRecorder::remap(int raw_frame) {
  if (m_frame_offset < 0)
    m_frame_offset = raw_frame - 1;   // first event -> session frame 1
  int f = raw_frame - m_frame_offset;
  return f > 0 ? f : 1;
}

// ── START ─────────────────────────────────────────────────────────────────
void FlightRecorder::logStart(int frame, uint64_t pts_ns, int camera_id) {
  int f = remap(frame);
  char buf[256];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"START\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"cam\":%d}",
           f, pts_ns, camera_id);
  writeEvent(buf);
}

// ── MODE CHANGE ───────────────────────────────────────────────────────────
void FlightRecorder::logModeChange(int frame, uint64_t pts_ns, int camera_id,
                                   int prevMode, int newMode) {
  int f = remap(frame);
  char buf[256];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"MODE_CHANGE\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"cam\":%d,"
           "\"from\":\"%s\","
           "\"to\":\"%s\"}",
           f, pts_ns, camera_id, modeName(prevMode), modeName(newMode));
  writeEvent(buf);
}

// ── ANGLE SAMPLE ──────────────────────────────────────────────────────────
void FlightRecorder::logAngle(int frame, uint64_t pts_ns, int camera_id,
                              float ax, float ay, float det_prob,
                              int rect_x, int rect_y,
                              int rect_w, int rect_h) {
  int f = remap(frame);
  char buf[320];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"ANGLE\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"cam\":%d,"
           "\"ax\":%.2f,"
           "\"ay\":%.2f,"
           "\"det\":%.3f,"
           "\"rect_x\":%d,"
           "\"rect_y\":%d,"
           "\"rect_w\":%d,"
           "\"rect_h\":%d}",
           f, pts_ns, camera_id, ax, ay, det_prob,
           rect_x, rect_y, rect_w, rect_h);
  writeEvent(buf);
}

// ── COMMAND RECEIVED ──────────────────────────────────────────────────────
// Now includes cmd args so replay can reproduce operator intent verbatim.
// CMD_CAPTURE / CMD_SET_RECT_POS carry (x, y, w) — the click coord and
// rect size — which is what "the template" is.
void FlightRecorder::logCmd(int frame, uint64_t pts_ns, int camera_id,
                            uint32_t cmdType, float arg1, float arg2,
                            float arg3) {
  int f = remap(frame);
  char buf[320];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"CMD\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"cam\":%d,"
           "\"type\":\"%s\","
           "\"code\":%u,"
           "\"arg1\":%.4f,"
           "\"arg2\":%.4f,"
           "\"arg3\":%.4f}",
           f, pts_ns, camera_id, cmdName(cmdType), cmdType, arg1, arg2,
           arg3);
  writeEvent(buf);
}

// ── HANDOFF ───────────────────────────────────────────────────────────────
// Emitted whenever the app-side output thread fires a seed CAPTURE on
// the peer camera. Geometric ground truth for evaluating the automatic
// handoff without touching tracker output.
void FlightRecorder::logHandoff(int frame, uint64_t pts_ns,
                                int from_cam, int to_cam,
                                int src_x, int src_y,
                                double dst_x, double dst_y,
                                double dst_w, double dst_h) {
  int session_frame = remap(frame);
  const char *dir = (from_cam == 1) ? "L2R" : "R2L";
  char buf[384];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"HANDOFF\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"from_cam\":%d,"
           "\"to_cam\":%d,"
           "\"src_x\":%d,"
           "\"src_y\":%d,"
           "\"dst_x\":%.2f,"
           "\"dst_y\":%.2f,"
           "\"dst_w\":%.2f,"
           "\"dst_h\":%.2f,"
           "\"direction\":\"%s\"}",
           session_frame, pts_ns, from_cam, to_cam,
           src_x, src_y, dst_x, dst_y, dst_w, dst_h, dir);
  writeEvent(buf);
}

// ── SHUTDOWN ──────────────────────────────────────────────────────────────
void FlightRecorder::logShutdown(int frame, uint64_t pts_ns,
                                 ShutdownReason reason, int totalFrames,
                                 double durationSec) {
  int session_frame = remap(frame);
  char buf[256];
  snprintf(buf, sizeof(buf),
           "{\"event\":\"SHUTDOWN\","
           "\"frame\":%d,"
           "\"t_ns\":%" PRIu64 ","
           "\"reason\":\"%s\","
           "\"total_frames\":%d,"
           "\"duration_sec\":%.3f}",
           session_frame, pts_ns, shutdownName(reason), totalFrames,
           durationSec);
  writeEvent(buf);

  // Patch results block in meta.json
  FILE *f = fopen(m_metaPath.c_str(), "r");
  if (!f) return;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char *content = (char *)malloc(sz + 1);
  if (!content) { fclose(f); return; }
  (void)fread(content, 1, sz, f);
  content[sz] = '\0';
  fclose(f);

  const char *results_key = "\"results\": {";
  char *pos = strstr(content, results_key);
  if (pos) {
    *pos = '\0';
    FILE *fw = fopen(m_metaPath.c_str(), "w");
    if (fw) {
      fprintf(fw, "%s", content);
      fprintf(fw,
              "  \"results\": {\n"
              "    \"total_frames\":       %d,\n"
              "    \"duration_seconds\":   %.3f,\n"
              "    \"pass_count_planned\": 0,\n"
              "    \"pass_count_actual\":  0,\n"
              "    \"shutdown_reason\":    \"%s\"\n"
              "  }\n"
              "}\n",
              totalFrames, durationSec, shutdownName(reason));
      fclose(fw);
    }
  }
  free(content);
}

// ════════════════════════════════════════════════════════════════════════════
// Private helpers
// ════════════════════════════════════════════════════════════════════════════

const char *FlightRecorder::modeName(int mode) const {
  switch (mode) {
  case 0: return "FREE";
  case 1: return "TRACKING";
  case 2: return "LOST";
  case 3: return "INERTIAL";
  case 4: return "STATIC";
  default: return "UNKNOWN";
  }
}

const char *FlightRecorder::camName(int model) const {
  switch (model) {
  case 1: return "Arducam AR0234";
  case 2: return "e-CAM24_CUNX AR0234";
  default: return "UNKNOWN";
  }
}

const char *FlightRecorder::shutdownName(ShutdownReason r) const {
  switch (r) {
  case ShutdownReason::CLEAN: return "CLEAN";
  case ShutdownReason::KILLED: return "KILLED";
  case ShutdownReason::CRASH: return "CRASH";
  default: return "UNKNOWN";
  }
}

const char *FlightRecorder::cmdName(uint32_t cmdType) const {
  switch (cmdType) {
  case 1: return "CAPTURE";
  case 2: return "RESET";
  case 3: return "CHANGE_SIZE";
  case 4: return "SET_RECT_POS";
  case 5: return "SET_PARAM";
  case 6: return "SAVE_PARAMS";
  case 7: return "HANDOFF";
  case 8: return "SET_CAMERA_PARAM";
  case 9: return "GET_PARAMS";
  case 10: return "CONFIRM_TARGET";
  default: return "UNKNOWN";
  }
}
