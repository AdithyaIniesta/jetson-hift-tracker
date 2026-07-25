#pragma once
#include <arpa/inet.h>
#include <gst/gst.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef TRACKER_V2
    #include <cvtracker/CvTracker.h>
    #include <cvtracker/VTracker.h> // <--- USE THIS PATH
#else
    #include "CvTracker.h"
    #include "VTracker.h"
#endif

// ── Active tracker type ──────────────────────────────────────────────────────
// This fork replaces the correlation filter with HiFT. When built with
// -DTRACKER_HIFT the global trackers are HiFTTracker; otherwise they stay
// CvTracker. All shared signatures use cr::vtracker::AppTracker so the swap is
// a single typedef. (HiFTTracker still relies on the cvtracker submodule for
// the VTracker/Frame/FeatureExtractor base symbols.)
#ifdef TRACKER_HIFT
    #include "hift/HiFTTracker.h"
    namespace cr { namespace vtracker { using AppTracker = cr::hift::HiFTTracker; } }
#else
    namespace cr { namespace vtracker { using AppTracker = CvTracker; } }
#endif

#include "dual/dual_capture.h"
#include "dual/pipeline.h"
#include "dual/tracker_state.h"
#include "handoff/handoff.h"
#include "recorder/recorder.h"

// =====================================================================
// ANSI color helpers for terminal log output
// Used inline in printf format strings, e.g.:
//   printf(LOG_GREEN "[MAIN] pipeline ready" LOG_RESET "\n");
// Safe even when stdout is not a TTY — most modern terminals strip
// them, and the escape sequences are harmless bytes to file redirects.
// =====================================================================
#define LOG_RESET   "\033[0m"
#define LOG_DIM     "\033[2m"
#define LOG_BOLD    "\033[1m"
#define LOG_RED     "\033[0;31m"
#define LOG_GREEN   "\033[0;32m"
#define LOG_YELLOW  "\033[1;33m"
#define LOG_BLUE    "\033[0;34m"
#define LOG_MAGENTA "\033[0;35m"
#define LOG_CYAN    "\033[0;36m"
#define LOG_GRAY    "\033[0;37m"

// =====================================================================
// Camera optical parameters (from lens datasheet)
// HFOV and VFOV in degrees � used to convert pixel offset to angle offset
// =====================================================================
static constexpr float HFOV_ARDUCAM = 68.0f;
static constexpr float VFOV_ARDUCAM = 54.0f;
static constexpr float HFOV_ECON = 104.6f;
static constexpr float VFOV_ECON = 61.6f;

// =====================================================================
// Highest VTrackerParam id the app persists / accepts over UDP.
// Both engines expose exactly 23 params (1-23). The baseline cvtracker
// maps 16-18 to EKF_SIGMA_A/ALPHA/R_BASE; the lockon engine drops those
// (internal EKF) and instead maps 21-23 to ENABLE_GMC /
// ENABLE_DNN_ARBITRATION / ENABLE_DNN_REACQUISITION. Same count, so the
// UDP/config range is identical — but a config file is NOT portable
// between engines (see the engine-specific CONFIG_FILE in main.cpp).
// =====================================================================
static constexpr int MAX_TRACKER_PARAM_ID = 23;

// Packets Structures
#pragma pack(push, 1)
struct CmdPacket {
  uint32_t magic;
  uint32_t type;
  float arg1;
  float arg2;
  float arg3;
};

struct AckPacket {
  uint32_t magic;
  uint32_t ack_type;
  uint32_t param_id;
  float value;
  uint32_t success;
  uint32_t reserved;
};

struct UartAnglePacket {
  uint8_t header[2];
  uint8_t mode;
  float angle_x_deg;
  float angle_y_deg;
  float det_prob;
  int16_t pixel_x;
  int16_t pixel_y;
  // Operator target-confirmation flag: 1 = operator confirmed this is the
  // correct object, 0 = not (yet) confirmed. Appended before the checksum
  // so all existing field offsets are unchanged; the XOR checksum covers it.
  uint8_t confirmed;
  uint8_t checksum;
};

struct TelemetryPacket {
  uint32_t magic;
  uint32_t frame_id;
  uint32_t mode;
  float det_prob;
  uint32_t proc_time_us;

  int32_t rect_x;
  int32_t rect_y;
  int32_t rect_width;
  int32_t rect_height;

  int32_t object_x;
  int32_t object_y;
  int32_t object_width;
  int32_t object_height;

  float angle_x_deg;
  float angle_y_deg;
  int32_t pixel_offset_x;
  int32_t pixel_offset_y;

  int32_t search_win_width;
  int32_t search_win_height;

  float lost_mode_option;

  int32_t frame_buffer_size;
  int32_t max_frames_lost;

  float rect_auto_size;
  float rect_auto_position;

  int32_t multiple_threads;
  int32_t num_channels;
  int32_t tracker_type;

  // Parameters 13, 14, 15
  float custom_1;
  float custom_2;
  float custom_3;

  // Parameters 16, 17, 18 (EKF Variables)
  float ekf_sigma_a;
  float ekf_sigma_alpha;
  float ekf_r_base;

  // Parameter 19, 20, 21 (DNN Settings)
  int32_t dnn_verify_interval;
  float dnn_veto_threshold;
  float dnn_accept_threshold;
  float dnn_similarity; // (Live matching score metric)

  // Video resolution context
  int32_t frame_width;
  int32_t frame_height;

  uint32_t reserved;
};

#pragma pack(pop)

static constexpr uint32_t CMD_MAGIC = 0x54524B43;
static constexpr uint32_t ACK_MAGIC = 0x41434B00;
static constexpr uint32_t TELEMETRY_MAGIC = 0x544C4D54;

enum CmdType : uint32_t {
  CMD_CAPTURE = 1,
  CMD_RESET = 2,
  CMD_CHANGE_SIZE = 3,
  CMD_SET_RECT_POS = 4,
  CMD_SET_PARAM = 5,
  CMD_SAVE_PARAMS = 6,
  CMD_HANDOFF = 7,
  CMD_SET_CAMERA_PARAM = 8,
  CMD_GET_PARAMS = 9, // <--- Add this here
  // Operator target-confirmation. arg1 != 0 asserts "this IS the right
  // object"; arg1 == 0 clears it. Reuses CmdPacket (no layout change).
  CMD_CONFIRM_TARGET = 10,
  // Manual handoff trigger. arg1 = source camera id (1 = LEFT, 2 =
  // RIGHT); arg2 = source-camera pixel u; arg3 = source-camera pixel v.
  // If arg2 or arg3 is negative, use the source camera's current
  // tracker rect centre. Handoff computes the destination pixel using
  // g_handoff and fires a synthetic CMD_CAPTURE on the destination
  // camera's tracker.
  CMD_HANDOFF_MANUAL = 11,
  // Query current V4L2 camera control values. No args. The control thread
  // reads all camera params from its device and replies with one ack per
  // param (id = simple camera-param id 1..7, value = current value), same
  // ack pattern as CMD_GET_PARAMS. Lets the GUI sync its sliders to the
  // camera's real state on connect.
  CMD_GET_CAMERA_PARAM = 12,
};

static constexpr int RING_SIZE = 4;

struct CaptureSlot {
  std::vector<uint8_t> bgrData; // raw BGR pixels from camera
  int frameId;                  // frame sequence number
  GstClockTime pts;             // pipeline clock timestamp
  bool ready;                   // true when slot contains valid data
};

// Result slot — carries tracker output from tracker thread to output thread
struct ResultSlot {
  std::vector<uint8_t> bgrData; // BGR frame for overlay drawing
  cr::vtracker::VTrackerParams
      params;            // tracker result (rectX, rectY, mode, etc.)
  int frameId;           // frame sequence number
  GstClockTime pts;      // pipeline clock timestamp
  GstClockTime frameDur; // frame duration for MKV muxer
  float angleOffX;       // pre-computed angle offset X (degrees)
  float angleOffY;       // pre-computed angle offset Y (degrees)
  int pixOffX;           // pre-computed pixel offset X
  int pixOffY;           // pre-computed pixel offset Y
};

// Global Extern Declarations
extern std::atomic<bool> g_running;

// ── Dual-lock trackers ──────────────────────────────────────────────
// One tracker instance PER CAMERA. Both process their own camera's
// frames continuously — no g_selected_camera gating on the frame path.
// Handoff is now "seed the other tracker via CAPTURE at a projected
// pixel when it's LOST/FREE and this one is TRACKING." Both mutexes
// protect their respective tracker from concurrent access by the
// control thread (CAPTURE / RESET / SET_PARAM) and the per-camera
// tracker thread (processFrame).
//
// SET_PARAM applied at runtime is broadcast to BOTH trackers so tuning
// stays symmetric. CAPTURE / RESET / CONFIRM_TARGET target the tracker
// matching the command's camera_id.
//
// `g_tracker` remains as an alias for g_tracker_L so pre-refactor call
// sites (of which there are many, in telemetry.cpp and control.cpp
// query paths) still compile — those sites happen to want the same
// tuning value that lives on both instances. Any new code that has
// a concept of "which camera" MUST pick the correct _L / _R instance
// explicitly.
extern cr::vtracker::AppTracker g_tracker_L;   // boresight
extern cr::vtracker::AppTracker g_tracker_R;   // depression
extern std::mutex g_mtx_L;
extern std::mutex g_mtx_R;
#define g_tracker g_tracker_L
#define g_mtx     g_mtx_L
extern std::atomic<int> g_frameId;
extern int g_uartFd;
extern int g_ctrlSock;
extern sockaddr_in g_gsAddr;
extern bool g_gsAddrValid;
extern int g_telemSock;
extern sockaddr_in g_telemDest;
extern char g_clientIpStr[64];
extern std::string recordingSessionPath;
extern bool recEnabled;
extern int g_tracker_W, g_tracker_H, g_fps, g_camModel;
extern float g_hfov, g_vfov;
extern std::string g_leftDev, g_rightDev, g_clientIp, g_uartDev;
extern int g_leftVideoPort, g_leftCtrlPort, g_rightVideoPort, g_rightCtrlPort;
extern std::string recordingFilePath;
extern bool recordingEnabled;
// Per-camera raw record pipelines. In dual mode both are non-null and
// each camera streams into its own raw_left.mkv / raw_right.mkv from
// boot, independent of which camera the operator eventually locks —
// so a replay tool can reproduce a handoff. In single-camera mode
// only the enabled side is non-null and writes to raw.mkv.
extern GstElement *g_recordPipeL;
extern GstElement *g_recordSrcL;
extern GstElement *g_recordPipeR;
extern GstElement *g_recordSrcR;
extern GstElement *g_annotatedPipeL;
extern GstElement *g_annotatedSrcL;
extern GstElement *g_annotatedPipeR;
extern GstElement *g_annotatedSrcR;
extern std::string recordingPipelineDesc;
extern std::mutex g_gsMtx;
extern std::atomic<int> g_selected_camera;
extern std::atomic<int> g_tracker_mode; // 0=FREE 1=TRACKING 2=LOST (legacy: mirror of L)
extern std::atomic<int> g_tracker_mode_L;  // per-camera mode published
extern std::atomic<int> g_tracker_mode_R;  // by each tracker thread

// ── Per-camera live state for UART fusion ────────────────────────
// Each output thread publishes its latest {angleX/Y, pixOffX/Y, det}
// plus a monotonic timestamp. Camera 1's output thread reads BOTH
// pairs and produces a single fused UART angle stream. Freshness is
// gated by FUSION_TTL_MS — a peer sample older than this is treated
// as absent (that camera briefly stalled or went FREE).
extern std::atomic<float> g_state_L_ang_x;
extern std::atomic<float> g_state_L_ang_y;
extern std::atomic<int>   g_state_L_pix_x;
extern std::atomic<int>   g_state_L_pix_y;
extern std::atomic<float> g_state_L_det;
extern std::atomic<int64_t> g_state_L_stamp_ms;
extern std::atomic<float> g_state_R_ang_x;
extern std::atomic<float> g_state_R_ang_y;
extern std::atomic<int>   g_state_R_pix_x;
extern std::atomic<int>   g_state_R_pix_y;
extern std::atomic<float> g_state_R_det;
extern std::atomic<int64_t> g_state_R_stamp_ms;
static constexpr int64_t FUSION_TTL_MS = 100;

extern std::atomic<bool>
    g_handoff_requested; // true = unlock camera for handoff
// Operator target-confirmation flag (CMD_CONFIRM_TARGET). 1 = confirmed.
// Auto-cleared on CAPTURE / RESET / HANDOFF so each new target is
// unconfirmed until the operator re-asserts it. Surfaced in the UART
// packet and echoed in telemetry (reserved bit 8).
extern std::atomic<int> g_target_confirmed;

extern FlightRecorder *g_rec;

// ── Handoff ──────────────────────────────────────────────────────────
// Cross-view target projection module — computes the pixel in one
// camera corresponding to the tracked target pixel in the other, given
// stereo calibration + target-plane depth.
//
// Populated at boot from stereo_calib.json in the working directory (or
// overridden via argv). When not initialised (no calibration file /
// invalid depth), g_handoff.ready() returns false and control.cpp
// gracefully rejects handoff commands with a clear error.
//
// g_target_depth_mm is prompted at run_jp{5,6}.sh startup; default 2000
// (2 m indoor). Airframe replaces this with per-frame barometer AGL.
extern handoff::HandoffModel g_handoff;
extern float g_target_depth_mm;

// Last rect published by outputThread — read by control.cpp when the
// operator triggers a handoff without specifying an explicit source
// pixel. Encoded as int32 so we can use std::atomic without needing
// a mutex; float precision isn't required for a pixel coordinate.
extern std::atomic<int32_t> g_last_rect_x;
extern std::atomic<int32_t> g_last_rect_y;
// Populated in main.cpp at boot with build/reproducibility + camera info.
// controlThread copies from this when it constructs g_rec on first CAPTURE.
extern SessionInfo g_sessionInfo;
extern std::string CONFIG_FILE;

static inline const char *modeStr(int m) {
  switch (m) {
  case 0:
    return "FREE";
  case 1:
    return "TRACKING";
  case 2:
    return "LOST";
  case 3:
    return "INERTIAL";
  case 4:
    return "STATIC";
  default:
    return "?";
  }
}

static inline cv::Scalar modeColor(int m) {
  switch (m) {
  case 0:
    return cv::Scalar(255, 255, 255);
  case 1:
    return cv::Scalar(0, 0, 255);
  case 2:
    return cv::Scalar(255, 0, 0);
  case 3:
    return cv::Scalar(0, 255, 255);
  case 4:
    return cv::Scalar(128, 128, 128);
  default:
    return cv::Scalar(255, 255, 255);
  }
}