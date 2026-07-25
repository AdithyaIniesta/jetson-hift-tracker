// ============================================================
// main.cpp — JetsonTracker entry point
//
// Architecture: producer-consumer pipeline
//
//   Left camera → left_ring → activeTrackerThread → resultQueue → outputThread
//   → stream / UART
//
// Modules:
//   capture/    — GStreamer appsink → DualRingBuffer
//   tracker/    — DualRingBuffer → VTracker → ResultSlot
//   streaming/  — ResultSlot → appsrc → H264/RTP
//   control/    — UDP command listener → tracker commands
//   telemetry/  — UDP telemetry sender → ground station
// ============================================================

// ── Module headers ───────────────────────────────────────────
#include "capture/capture.h"
#include "common/globals.h"
#include "control/control.h"
#include "streaming/streaming.h"
#include "telemetry/telemetry.h"
#include "tracker/tracker.h"
#ifdef TRACKER_V2
#include <cvtracker/TrtFeatureExtractor.h>
#endif

// ── System / GStreamer headers ───────────────────────────────
#include <arpa/inet.h>
#include <fcntl.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <thread>

// ── Global definitions ───────────────────────────────────────
// Dual-lock: one tracker instance per camera. g_tracker / g_mtx are
// macro aliases for g_tracker_L / g_mtx_L (see common/globals.h), so
// legacy call sites that read tuning still compile; SET_PARAM in
// control.cpp broadcasts to BOTH trackers.
cr::vtracker::AppTracker g_tracker_L;
cr::vtracker::AppTracker g_tracker_R;
std::mutex g_mtx_L;
std::mutex g_mtx_R;
std::atomic<int> g_frameId{0};
std::atomic<int> g_selected_camera{0};
std::atomic<int> g_tracker_mode{0};
std::atomic<int> g_tracker_mode_L{0};
std::atomic<int> g_tracker_mode_R{0};
std::atomic<float> g_state_L_ang_x{0.0f};
std::atomic<float> g_state_L_ang_y{0.0f};
std::atomic<int>   g_state_L_pix_x{0};
std::atomic<int>   g_state_L_pix_y{0};
std::atomic<float> g_state_L_det{0.0f};
std::atomic<int64_t> g_state_L_stamp_ms{0};
std::atomic<float> g_state_R_ang_x{0.0f};
std::atomic<float> g_state_R_ang_y{0.0f};
std::atomic<int>   g_state_R_pix_x{0};
std::atomic<int>   g_state_R_pix_y{0};
std::atomic<float> g_state_R_det{0.0f};
std::atomic<int64_t> g_state_R_stamp_ms{0};
std::atomic<bool> g_handoff_requested{false};
std::atomic<int> g_target_confirmed{0};
std::atomic<bool> g_running{true};
int g_uartFd = -1;
int g_ctrlSock = -1;
sockaddr_in g_gsAddr{};
bool g_gsAddrValid = false;
std::mutex g_gsMtx;
int g_telemSock = -1;
sockaddr_in g_telemDest{};
char g_clientIpStr[64] = "192.168.0.100";
std::string recordingSessionPath;
bool recEnabled = false;
int g_tracker_W = 0, g_tracker_H = 0, g_fps = 0, g_camModel = 0;
float g_hfov = 0, g_vfov = 0;
std::string g_leftDev, g_rightDev, g_clientIp, g_uartDev;
int g_leftVideoPort = 0, g_leftCtrlPort = 0, g_rightVideoPort = 0,
    g_rightCtrlPort = 0;
std::string recordingFilePath;
bool recordingEnabled = false;
GstElement *g_recordPipeL = nullptr;
GstElement *g_recordSrcL = nullptr;
GstElement *g_recordPipeR = nullptr;
GstElement *g_recordSrcR = nullptr;
GstElement *g_annotatedPipeL = nullptr;
GstElement *g_annotatedSrcL = nullptr;
GstElement *g_annotatedPipeR = nullptr;
GstElement *g_annotatedSrcR = nullptr;
std::string recordingPipelineDesc;
FlightRecorder *g_rec = nullptr;
SessionInfo g_sessionInfo;
handoff::HandoffModel g_handoff;
float g_target_depth_mm = 0.0f;
std::atomic<int32_t> g_last_rect_x{0};
std::atomic<int32_t> g_last_rect_y{0};
std::string CONFIG_FILE;

// Fallback definitions in case CMake didn't inject build info (e.g. an
// out-of-tree build without git). Never override a real value.
#ifndef JT_GIT_COMMIT
#define JT_GIT_COMMIT "unknown"
#endif
#ifndef JT_BUILD_TIME
#define JT_BUILD_TIME __DATE__ " " __TIME__
#endif
#ifndef JT_TRACKER_ENGINE
#define JT_TRACKER_ENGINE "unknown"
#endif

// Two-strike SIGINT handler: first signal requests a clean shutdown by
// flipping g_running; second signal (operator hitting Ctrl+C again
// because the first appeared to hang) hard-exits. Necessary because
// blocking recvfrom / gst_pull_sample calls inside worker threads
// don't observe the flag until they either time out or receive data.
static void sigHandler(int) {
  static std::atomic<int> hits{0};
  g_running = false;
  if (hits.fetch_add(1) >= 1) {
    fprintf(stderr, "\n[MAIN] Second SIGINT — forcing exit.\n");
    _exit(2);
  }
}

int main(int argc, char *argv[]) {
  // ── argv layout ─────────────────────────────────────────────
  // [1]  clientIp
  // [2]  leftVideoPort    [3]  leftCtrlPort
  // [4]  rightVideoPort   [5]  rightCtrlPort
  // [6]  W               [7]  H
  // [8]  uartDev
  // [9]  leftDev          [10] rightDev
  // [11] streamFormat     [12] fps
  // [13] camModel
  // [14] recBasePath      [15] recEnabled
  // [16] downsample      [17] hfov_deg      [18] vfov_deg
  const char *clientIp = (argc > 1) ? argv[1] : "192.168.0.100";
  int leftVideoPort = (argc > 2) ? atoi(argv[2]) : 5000;
  int leftCtrlPort = (argc > 3) ? atoi(argv[3]) : 5001;
  int rightVideoPort = (argc > 4) ? atoi(argv[4]) : 5002;
  int rightCtrlPort = (argc > 5) ? atoi(argv[5]) : 5003;
  int W = (argc > 6) ? atoi(argv[6]) : 1280;
  int H = (argc > 7) ? atoi(argv[7]) : 720;
  const char *uartDev = (argc > 8) ? argv[8] : "/dev/ttyTHS0";
  const char *leftDev = (argc > 9) ? argv[9] : "";
  const char *rightDev = (argc > 10) ? argv[10] : "";
  const char *streamFormat = (argc > 11) ? argv[11] : "UYVY";
  int fps = (argc > 12) ? atoi(argv[12]) : 60;
  int camModel = (argc > 13) ? atoi(argv[13]) : 2;
  const char *recBasePath = (argc > 14) ? argv[14] : "/home/nvidia/recordings";
  recEnabled = (argc > 15) ? atoi(argv[15]) != 0 : false;
  bool downsample_enabled = (argc > 16) ? atoi(argv[16]) != 0 : false;
  // Optional lens overrides. <=0 or absent => fall back to camModel default.
  float hfov_override = (argc > 17) ? (float)atof(argv[17]) : 0.0f;
  float vfov_override = (argc > 18) ? (float)atof(argv[18]) : 0.0f;
  // argv[19] — target-plane depth in millimetres for handoff. Optional;
  // when absent or non-positive, handoff is disabled and CMD_HANDOFF_
  // MANUAL is rejected with a clear log. Indoor default set by run
  // scripts is 2000 mm (2 m altitude, ground target).
  g_target_depth_mm = (argc > 19) ? (float)atof(argv[19]) : 0.0f;
  // ── Camera mode ──────────────────────────────────────────────
  // WHY: port mapping fixed by convention.
  // Left  → leftVideoPort:5000   leftCtrlPort:5001
  // Right → rightVideoPort:5002  rightCtrlPort:5003
  bool left_enabled = (leftDev[0] != '\0');
  bool right_enabled = (rightDev[0] != '\0');
  bool dual_enabled = left_enabled && right_enabled;

  // ── Tracker resolution ───────────────────────────────────────
  int tracker_W = downsample_enabled ? 640 : W;
  int tracker_H = downsample_enabled ? 480 : H;
  g_tracker_W = tracker_W;
  g_tracker_H = tracker_H;

  // ── Recording session paths ──────────────────────────────────
  // WHY: recording pipelines must use tracker_W/tracker_H, not W/H —
  // the buffers pushed into g_recordSrc are downsampled frames when
  // downsample_enabled is true. Using W/H here caused a caps mismatch
  // (green screen) because appsrc caps declared one size while actual
  // buffer data was a different size.
  std::string sessionName = makeRecordingSessionName();
  if (recEnabled) {
    recordingSessionPath = std::string(recBasePath) + "/" + sessionName;
    mkdir(recordingSessionPath.c_str(), 0755);
  }
  // Per-camera raw filenames: in dual mode we record both feeds so
  // replay can reproduce a handoff. Single-camera mode keeps the
  // historical raw.mkv name so downstream tooling is unaffected.
  std::string rawLeftFilePath, rawRightFilePath;
  if (dual_enabled) {
    rawLeftFilePath  = recordingSessionPath + "/raw_left.mkv";
    rawRightFilePath = recordingSessionPath + "/raw_right.mkv";
  } else if (left_enabled) {
    rawLeftFilePath  = recordingSessionPath + "/raw.mkv";
  } else if (right_enabled) {
    rawRightFilePath = recordingSessionPath + "/raw.mkv";
  }
  recordingFilePath = dual_enabled ? rawLeftFilePath
                                   : (left_enabled ? rawLeftFilePath
                                                   : rawRightFilePath);
  // Dual mode: one annotated file per camera so replay of a handoff
  // has the overlay on BOTH sides. Single mode: unchanged name.
  std::string annotatedLeftFilePath, annotatedRightFilePath;
  if (dual_enabled) {
    annotatedLeftFilePath  = recordingSessionPath + "/annotated_left.mkv";
    annotatedRightFilePath = recordingSessionPath + "/annotated_right.mkv";
  } else if (left_enabled) {
    annotatedLeftFilePath  = recordingSessionPath + "/annotated.mkv";
  } else if (right_enabled) {
    annotatedRightFilePath = recordingSessionPath + "/annotated.mkv";
  }

  // ── Startup banner ───────────────────────────────────────────
  printf("\n==========================================\n");
  printf("  JetsonTracker\n");
  printf("==========================================\n");
  printf("  Camera mode   : %s\n",
         dual_enabled ? "dual" : left_enabled ? "left only" : "right only");
  printf("  Left camera   : %s\n", left_enabled ? leftDev : "disabled");
  printf("  Right camera  : %s\n", right_enabled ? rightDev : "disabled");
  printf("  Left stream   : %s:%d  ctrl:%d\n", clientIp, leftVideoPort,
         leftCtrlPort);
  printf("  Right stream  : %s:%d  ctrl:%d\n", clientIp, rightVideoPort,
         rightCtrlPort);
  printf("  Resolution    : %dx%d  format:%s  fps:%d\n", W, H, streamFormat,
         fps);
  printf("  Tracker res   : %dx%d%s\n", tracker_W, tracker_H,
         downsample_enabled ? " (downsampled)" : "");
  // EKF and DNN switches only exist in the cvtracker (v2) enum; the
  // Constant Robotics VTrackerParam stops at CUSTOM_3. Guard so the
  // constant_robotics build compiles the same source cleanly.
#ifdef TRACKER_V2
  printf("  EKF            : %s\n",
         g_tracker.getParam(cr::vtracker::VTrackerParam::ENABLE_EKF) != 0.0f
             ? "enabled"
             : "disabled");
  printf("  DNN Verifier   : %s\n",
         g_tracker.getParam(cr::vtracker::VTrackerParam::ENABLE_DNN_VERIFIER) !=
                 0.0f
             ? "enabled"
             : "disabled");
#else
  printf("  EKF            : n/a (constant_robotics build)\n");
  printf("  DNN Verifier   : n/a (constant_robotics build)\n");
#endif
  printf("  UART          : %s\n", uartDev);
  printf("==========================================\n\n");

  // ── Telemetry UDP socket ──────────────────────────────────────
  // outputThread sends TelemetryPacket to ground station every frame.
  g_telemSock = socket(AF_INET, SOCK_DGRAM, 0);
  if (g_telemSock >= 0) {
    memset(&g_telemDest, 0, sizeof(g_telemDest));
    g_telemDest.sin_family = AF_INET;
    // Optional benchmark override: on localhost, the CTRL port (5001)
    // is already bound by the harness's telemetry listener, so a naked
    // tracker would collide when it tries to bind CTRL. Setting
    // TRACKER_TELEM_PORT_OVERRIDE lets the harness route telemetry to
    // a *different* port (e.g. 15001) while the tracker keeps CTRL on
    // 5001. Env var unset → production behaviour, telemetry goes to
    // ctrl_port as before.
    int telem_port = leftCtrlPort;
    if (const char *e = std::getenv("TRACKER_TELEM_PORT_OVERRIDE")) {
      int p = atoi(e);
      if (p > 0 && p < 65536) telem_port = p;
    }
    g_telemDest.sin_port = htons(telem_port);
    if (inet_pton(AF_INET, clientIp, &g_telemDest.sin_addr) <= 0) {
      fprintf(stderr, "[TELEM] Invalid client IP: %s\n", clientIp);
      close(g_telemSock);
      g_telemSock = -1;

    } else {
      printf("[TELEM] Will send telemetry to %s:%d\n", clientIp, leftCtrlPort);
    }
  }

  // ── Signals and GStreamer ────────────────────────────────────
  signal(SIGINT, sigHandler);
  signal(SIGTERM, sigHandler);
  gst_init(&argc, &argv);

  // ── Camera readiness check ───────────────────────────────────
  // Check left if available, otherwise right. Skipped entirely in
  // file-source override mode (benchmark replay) — the device path
  // there is just a label, not a real v4l2 node.
  const bool file_src_active =
      (std::getenv("TRACKER_FILE_SRC_LEFT") != nullptr) ||
      (std::getenv("TRACKER_FILE_SRC_RIGHT") != nullptr) ||
      (std::getenv("TRACKER_FILE_SRC") != nullptr);
  if (!file_src_active) {
    const char *check_dev = left_enabled ? leftDev : rightDev;
    int fd = -1;
    for (int i = 0; i < 10; i++) {
      fd = open(check_dev, O_RDONLY | O_NONBLOCK);
      if (fd >= 0) {
        close(fd);
        break;
      }
      fprintf(stderr, "[WARN] %s not ready (attempt %d/10)\n", check_dev,
              i + 1);
      sleep(1);
    }
    if (fd < 0) {
      fprintf(stderr, "[ERROR] %s never became accessible.\n", check_dev);
      return 1;
    }
  } else {
    fprintf(stdout,
            "[MAIN] file-source mode: skipping camera readiness check\n");
  }

  // ── Capture pipelines ────────────────────────────────────────
  // build_capture_pipeline() from dual/pipeline.h.
  // Left pipeline built if left_enabled.
  // Right pipeline built if right_enabled.
  GstElement *capPipe = nullptr;
  GstElement *sinkL = nullptr;
  GstElement *capPipeR = nullptr;
  GstElement *sinkR = nullptr;

  auto make_cap_cfg = [&](const char *dev) {
    CameraConfig cfg;
    cfg.video_device_path = dev;
    cfg.pixel_format = streamFormat;
    cfg.capture_width_pixels = W;
    cfg.capture_height_pixels = H;
    // WHY: output size is what the appsink actually delivers.
    // When downsample_enabled, this triggers a real videoscale
    // element in the capture pipeline so ring buffers, tracker,
    // stream, and recording all see consistent 640x480 frame data
    // instead of a mismatched label-only flag.
    cfg.output_width_pixels = tracker_W;
    cfg.output_height_pixels = tracker_H;
    cfg.frames_per_second = fps;
#ifdef JETSON_XAVIER
    cfg.requires_xavier_nvvidconv = true;
#else
    cfg.requires_xavier_nvvidconv = false;
#endif
    return cfg;
  };

  if (left_enabled) {
    CameraConfig cfg = make_cap_cfg(leftDev);
    capPipe = build_capture_pipeline(cfg);
    sinkL = capPipe ? get_pipeline_element(capPipe, "sink") : nullptr;
    if (!capPipe || !sinkL) {
      fprintf(stderr, "[ERROR] Left capture pipeline failed.\n");
      return 1;
    }
    printf(LOG_GREEN "[MAIN]" LOG_RESET " Left capture pipeline ready.\n");
  }

  if (right_enabled) {
    CameraConfig cfg = make_cap_cfg(rightDev);
    capPipeR = build_capture_pipeline(cfg);
    sinkR = capPipeR ? get_pipeline_element(capPipeR, "sink") : nullptr;
    if (!capPipeR || !sinkR) {
      fprintf(stderr, "[ERROR] Right capture pipeline failed.\n");
      return 1;
    }
    printf(LOG_GREEN "[MAIN]" LOG_RESET " Right capture pipeline ready.\n");
  }

  // ── Stream pipelines ─────────────────────────────────────────
  // build_stream_pipeline() from dual/pipeline.h.
  // Left  → leftVideoPort:5000
  // Right → rightVideoPort:5002
  GstElement *strPipe = nullptr;
  GstElement *srcL = nullptr;
  GstElement *strPipeR = nullptr;
  GstElement *srcR = nullptr;

  auto make_str_cfg = [&](int port) {
    StreamConfig cfg;
    cfg.destination_ip = clientIp;
    cfg.destination_port = port;
    cfg.stream_width_pixels = tracker_W;
    cfg.stream_height_pixels = tracker_H;
    cfg.frames_per_second = fps;
    return cfg;
  };

  if (left_enabled) {
    StreamConfig cfg = make_str_cfg(leftVideoPort);
    strPipe = build_stream_pipeline(cfg);
    srcL = strPipe ? get_pipeline_element(strPipe, "streamsrc") : nullptr;
    if (!strPipe || !srcL) {
      fprintf(stderr, "[ERROR] Left stream pipeline failed.\n");
      return 1;
    }
    printf(LOG_GREEN "[MAIN]" LOG_RESET " Left stream pipeline ready → "
                     LOG_CYAN "%s:%d" LOG_RESET "\n",
           clientIp, leftVideoPort);
  }

  if (right_enabled) {
    StreamConfig cfg = make_str_cfg(rightVideoPort);
    strPipeR = build_stream_pipeline(cfg);
    srcR = strPipeR ? get_pipeline_element(strPipeR, "streamsrc") : nullptr;
    if (!strPipeR || !srcR) {
      fprintf(stderr, "[ERROR] Right stream pipeline failed.\n");
      return 1;
    }
    printf(LOG_GREEN "[MAIN]" LOG_RESET " Right stream pipeline ready → "
                     LOG_CYAN "%s:%d" LOG_RESET "\n",
           clientIp, rightVideoPort);
  }

  // ── Start pipelines ──────────────────────────────────────────
  if (capPipe)
    gst_element_set_state(capPipe, GST_STATE_PLAYING);
  if (strPipe)
    gst_element_set_state(strPipe, GST_STATE_PLAYING);
  if (capPipeR)
    gst_element_set_state(capPipeR, GST_STATE_PLAYING);
  if (strPipeR)
    gst_element_set_state(strPipeR, GST_STATE_PLAYING);

  if (capPipe &&
      gst_element_get_state(capPipe, nullptr, nullptr, 5 * GST_SECOND) ==
          GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "[ERROR] Left capture failed to reach PLAYING.\n");
    return 1;
  }
  if (strPipe &&
      gst_element_get_state(strPipe, nullptr, nullptr, 5 * GST_SECOND) ==
          GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "[ERROR] Left stream failed to reach PLAYING.\n");
    return 1;
  }
  if (capPipeR &&
      gst_element_get_state(capPipeR, nullptr, nullptr, 5 * GST_SECOND) ==
          GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "[ERROR] Right capture failed to reach PLAYING.\n");
    return 1;
  }
  if (strPipeR &&
      gst_element_get_state(strPipeR, nullptr, nullptr, 5 * GST_SECOND) ==
          GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "[ERROR] Right stream failed to reach PLAYING.\n");
    return 1;
  }

  // ── Control threads ──────────────────────────────────────────
  // control/control.h — one thread per camera port.
  // First CAPTURE locks g_selected_camera for the session.
  // Left  ctrl listens on leftCtrlPort,  camera_id=1
  // Right ctrl listens on rightCtrlPort, camera_id=2
  std::thread ctrlLeft(controlThread, leftCtrlPort, 1, leftDev);
  std::thread ctrlRight(controlThread, rightCtrlPort, 2, rightDev);
  ctrlLeft.detach();
  ctrlRight.detach();

  // ── UART ─────────────────────────────────────────────────────
  // Angle offsets sent over UART every frame while tracking.
  g_uartFd = uartOpen(uartDev, 115200);
  if (g_uartFd < 0)
    printf("[WARN] UART not available — angle output disabled.\n");

  // ── Tracker configuration ────────────────────────────────────
  // CvTracker — set hardcoded defaults then override with saved config.
  // loadParams() reads from CONFIG_FILE set from HOME env.
  const char *homeEnv = getenv("HOME");
  // Engine-specific config file. Param ids 16-23 mean different things in
  // the baseline vs lockon engines, so a shared file would cross-contaminate
  // an A/B run (a cuda-saved EKF sigma would land on lockon's GMC enable,
  // etc.). Keep a separate file per engine.
#ifdef TRACKER_LOCKON
  CONFIG_FILE =
      std::string(homeEnv ? homeEnv : "/tmp") + "/tracker_params_lockon.cfg";
#else
  CONFIG_FILE = std::string(homeEnv ? homeEnv : "/tmp") + "/tracker_params.cfg";
#endif
  float CAMERA_HFOV_DEG = (camModel == 2) ? HFOV_ECON : HFOV_ARDUCAM;
  float CAMERA_VFOV_DEG = (camModel == 2) ? VFOV_ECON : VFOV_ARDUCAM;
  if (hfov_override > 0.0f) CAMERA_HFOV_DEG = hfov_override;
  if (vfov_override > 0.0f) CAMERA_VFOV_DEG = vfov_override;
  // FOV maps onto the tracker's pixel grid, which is 640x480 when
  // downsampling is enabled and W x H otherwise. Using full-sensor W/H here
  // under-reports the angle by the downsample ratio.
  const float degPerPixelX = CAMERA_HFOV_DEG / (float)tracker_W;
  const float degPerPixelY = CAMERA_VFOV_DEG / (float)tracker_H;
  g_hfov = CAMERA_HFOV_DEG;
  g_vfov = CAMERA_VFOV_DEG;
  g_fps = fps;
  g_camModel = camModel;
  g_leftDev = leftDev;
  g_rightDev = rightDev;
  g_clientIp = clientIp;
  g_uartDev = uartDev;
  g_leftVideoPort = leftVideoPort;
  g_leftCtrlPort = leftCtrlPort;
  g_rightVideoPort = rightVideoPort;
  g_rightCtrlPort = rightCtrlPort;

  // ── Populate SessionInfo (used by FlightRecorder on first CAPTURE) ──
  // WHY: gather all reproducibility fields at boot from one place. The
  // controlThread later copies this, patches selectedCameraId, and hands
  // it to the FlightRecorder constructor.
  g_sessionInfo.sessionPath = recordingSessionPath;
  g_sessionInfo.W = tracker_W;
  g_sessionInfo.H = tracker_H;
  g_sessionInfo.fps = fps;
  g_sessionInfo.hfov = CAMERA_HFOV_DEG;
  g_sessionInfo.vfov = CAMERA_VFOV_DEG;
  g_sessionInfo.camModel = camModel;
  g_sessionInfo.dualEnabled = dual_enabled;
  g_sessionInfo.leftDev = leftDev;
  g_sessionInfo.rightDev = rightDev;
  g_sessionInfo.clientIp = clientIp;
  g_sessionInfo.leftVideoPort = leftVideoPort;
  g_sessionInfo.leftCtrlPort = leftCtrlPort;
  g_sessionInfo.rightVideoPort = rightVideoPort;
  g_sessionInfo.rightCtrlPort = rightCtrlPort;
  g_sessionInfo.uartDev = uartDev;
  g_sessionInfo.gitCommit = JT_GIT_COMMIT;
  g_sessionInfo.buildTime = JT_BUILD_TIME;
  g_sessionInfo.trackerEngine = JT_TRACKER_ENGINE;
  {
    std::string argvLine;
    for (int i = 0; i < argc; i++) {
      if (i) argvLine += ' ';
      argvLine += argv[i];
    }
    g_sessionInfo.argvLine = argvLine;
  }
  g_tracker.setParam(VTrackerParam::RECT_WIDTH, 72);
  g_tracker.setParam(VTrackerParam::RECT_HEIGHT, 72);
  g_tracker.setParam(VTrackerParam::SEARCH_WINDOW_WIDTH, 256);
  g_tracker.setParam(VTrackerParam::SEARCH_WINDOW_HEIGHT, 256);
  g_tracker.setParam(VTrackerParam::LOST_MODE_OPTION, 0);
  g_tracker.setParam(VTrackerParam::NUM_CHANNELS, 2);
  g_tracker.setParam(VTrackerParam::MULTIPLE_THREADS, 1);
  g_tracker.setParam(VTrackerParam::TYPE, 0);
  g_tracker.setParam(VTrackerParam::FRAME_BUFFER_SIZE, 1);
  g_tracker.setParam(VTrackerParam::CUSTOM_1, 0.1f);   // loss threshold
  g_tracker.setParam(VTrackerParam::CUSTOM_2, 0.4f);   // recapture threshold
  g_tracker.setParam(VTrackerParam::CUSTOM_3, 0.075f); // learning rate

  // ── EKF defaults (cvtracker/v2 only — enum not present in v1) ─────
  // The three EKF_SIGMA_*/R_BASE noise knobs exist only in the baseline
  // cvtracker engine, where the pipeline runs its own external EKF. The
  // lockon engine dropped them (its EKF is internal) and exposes only
  // ENABLE_EKF, so guard the sigma writes out for TRACKER_LOCKON.
#ifdef TRACKER_V2
#ifndef TRACKER_LOCKON
  g_tracker.setParam(cr::vtracker::VTrackerParam::EKF_SIGMA_A, 1.5f);
  g_tracker.setParam(cr::vtracker::VTrackerParam::EKF_SIGMA_ALPHA, 0.15f);
  g_tracker.setParam(cr::vtracker::VTrackerParam::EKF_R_BASE, 4.0f);
#endif
  g_tracker.setParam(cr::vtracker::VTrackerParam::ENABLE_EKF,
                     1.0f); // Default EKF to ON
#endif
  // ──────────────────────────────────────────────────────────────────

#ifdef TRACKER_V2
  // DNN verifier defaults (persisted; GUI can override via SET_PARAM)
  g_tracker.setParam(VTrackerParam::DNN_VERIFY_INTERVAL, 5.0f);
  g_tracker.setParam(VTrackerParam::DNN_VETO_THRESHOLD, 0.35f);
  g_tracker.setParam(VTrackerParam::DNN_ACCEPT_THRESHOLD, 0.75f);
#endif

  // 2. NOW call loadParams EXACTLY ONCE to cleanly overwrite defaults with disk
  // values
  printf("[BOOT] Initializing parameter setup... Loading from: %s\n",
         CONFIG_FILE.c_str());
  if (loadParams()) {
    printf(
        "[BOOT] Successfully loaded your custom configuration file values!\n");
  } else {
    printf("[BOOT] No config file found or load failed. Using system fallback "
           "defaults.\n");
  }

  // ── Broadcast tuning to BOTH trackers ────────────────────────
  // All setParam calls above hit g_tracker_L (via the g_tracker macro
  // alias). Copy every tunable ID onto g_tracker_R so the two trackers
  // stay symmetric. SET_PARAM at runtime does the same broadcast in
  // control.cpp.
  for (int id = 1; id <= MAX_TRACKER_PARAM_ID; ++id) {
    const float v =
        g_tracker_L.getParam(static_cast<cr::vtracker::VTrackerParam>(id));
    g_tracker_R.setParam(static_cast<cr::vtracker::VTrackerParam>(id), v);
  }

#ifdef TRACKER_V2
  // Read the native state that was just populated by loadParams()
  bool native_dnn_requested =
      (g_tracker.getParam(cr::vtracker::VTrackerParam::ENABLE_DNN_VERIFIER) !=
       0.0f);
  bool dnn_ok = false;

  if (native_dnn_requested) {
    // Verifier model selectable at runtime via TRACKER_VERIFIER:
    //   dino     (default) - DINOv2-small CLS token (self-supervised, strong
    //                        instance retrieval, 224px/384-D; best in field.
    //                        Needs models/dinov2_small.onnx, see
    //                        tools/export_dinov2.py. HEAVY ViT — watch latency.)
    //   embedder           - trained re-id embedder (discriminative, 128px)
    //   resnet             - frozen ResNet18 block2 (scale-invariant, 128px;
    //                        needs models/resnet18_block2.onnx, see
    //                        tools/export_resnet18_block2.py)
    //   none               - skip the DNN verifier (built-in fallback used)
    const char *verEnv = std::getenv("TRACKER_VERIFIER");
    std::string verifier = verEnv ? verEnv : "dino";   // dino = best in field

    if (verifier == "none") {
      printf("[MAIN] DNN verifier disabled via TRACKER_VERIFIER=none.\n");
    } else {
      cr::vtracker::TrtExtractorConfig dnnCfg;
      dnnCfg.channels = 3;
      if (verifier == "resnet") {
        dnnCfg.onnxPath = "models/resnet18_block2.onnx";
        dnnCfg.inputSize = 128;
      } else if (verifier == "dino") {
        dnnCfg.onnxPath = "models/dinov2_small.onnx";
        dnnCfg.inputSize = 224;   // DINOv2 patch-14 grid needs 224
      } else {
        dnnCfg.onnxPath = "models/embedder_legacy.onnx";
        dnnCfg.inputSize = 128;
      }
      auto dnnExtractor = cr::vtracker::createTrtFeatureExtractor(dnnCfg);
      if (dnnExtractor) {
        g_tracker_L.setFeatureExtractor(dnnExtractor);
        g_tracker_R.setFeatureExtractor(dnnExtractor);
        dnn_ok = true;
        printf("[MAIN] DNN verifier initialized: %s (%s)\n", verifier.c_str(),
               dnnCfg.onnxPath.c_str());
      } else {
        printf("[MAIN] DNN verifier init failed (%s) -- falling back to "
               "disabled.\n",
               dnnCfg.onnxPath.c_str());
      }
    }
  } else {
    printf("[MAIN] DNN verifier is disabled by configuration choice.\n");
  }

  // Force sync the tracker parameter state with reality in case the model file
  // was missing
  g_tracker_L.setParam(cr::vtracker::VTrackerParam::ENABLE_DNN_VERIFIER,
                       dnn_ok ? 1.0f : 0.0f);
  g_tracker_R.setParam(cr::vtracker::VTrackerParam::ENABLE_DNN_VERIFIER,
                       dnn_ok ? 1.0f : 0.0f);
#endif

  // ── Handoff ──────────────────────────────────────────────────
  // The stereo calibration is resolution-specific. If a file named
  // stereo_calib_${W}x${H}.json exists for the tracker's operating
  // resolution, use it; otherwise fall back to the generic
  // stereo_calib.json. No K-scaling fudge — the loaded file is used
  // verbatim, so a mismatched resolution would produce wrong seeds.
  // Recalibrate at the tracker resolution instead of scaling on the
  // fly.
  // Always initialise when calibration is available. depth > 0 gives the
  // full homography + epipolar handoff; depth <= 0 is DISTANCE-FREE mode
  // (epipolar line + DINOv2 search, no homography) — initialise() handles
  // both and only skips the plane homography when depth <= 0.
  {
    char sized_path[64];
    snprintf(sized_path, sizeof(sized_path),
             "stereo_calib_%dx%d.json", tracker_W, tracker_H);
    struct stat st;
    const bool sized_exists = (stat(sized_path, &st) == 0);
    const char *calib_path = sized_exists ? sized_path : "stereo_calib.json";
    if (sized_exists) {
      printf("[HANDOFF] Using resolution-specific calibration: %s\n",
             calib_path);
    } else {
      printf("[HANDOFF] %s not found, using stereo_calib.json\n",
             sized_path);
    }

    handoff::StereoCalib calib{};
    if (handoff::loadStereoCalib(calib_path, calib)) {
      if (!g_handoff.initialise(calib, g_target_depth_mm)) {
        fprintf(stderr, "[HANDOFF] initialise() failed — handoff disabled\n");
      }
    } else {
      fprintf(stderr, "[HANDOFF] %s not loaded — handoff disabled\n",
              calib_path);
    }
  }

  printf(LOG_BOLD LOG_GREEN "[MAIN]" LOG_RESET " "
         LOG_BOLD "Pipelines running." LOG_RESET " Ctrl+C to stop.\n\n");
  // ── Recording pipelines ──────────────────────────────────────
  // Raw recording: original active-camera frames, no overlay.
  // Annotated recording: frames with bounding box and status overlay.
  // WHY: built only when recEnabled — keeps capture-only/no-record
  // runs free of any recording-related GStreamer overhead.
  if (recEnabled) {
    GError *recErr = nullptr;

    // One raw record pipeline per enabled camera. In dual mode both
    // fire continuously from boot, feeding raw_left.mkv / raw_right.mkv.
    // In single-camera mode only one pipeline is built and writes to
    // raw.mkv (the historical filename).
    auto launch_raw = [&](const std::string &path, GstElement **pipe,
                          GstElement **src, const char *tag) -> bool {
      if (path.empty()) return true;
      std::string desc =
          makeRecordingPipelineDesc(path, tracker_W, tracker_H, fps);
      *pipe = gst_parse_launch(desc.c_str(), &recErr);
      if (recErr || !*pipe) {
        fprintf(stderr, "[ERROR] %s raw record pipeline failed.\n", tag);
        return false;
      }
      *src = gst_bin_get_by_name(GST_BIN(*pipe), "recordsrc");
      gst_element_set_state(*pipe, GST_STATE_READY);
      gst_element_set_state(*pipe, GST_STATE_PLAYING);
      return true;
    };

    if (left_enabled  && !launch_raw(rawLeftFilePath,  &g_recordPipeL,
                                     &g_recordSrcL, "Left"))  return 1;
    if (right_enabled && !launch_raw(rawRightFilePath, &g_recordPipeR,
                                     &g_recordSrcR, "Right")) return 1;

    auto launch_annotated = [&](const std::string &path, GstElement **pipe,
                                GstElement **src, const char *tag) -> bool {
      if (path.empty()) return true;
      std::string desc =
          makeRecordingPipelineDesc(path, tracker_W, tracker_H, fps);
      GError *e = nullptr;
      *pipe = gst_parse_launch(desc.c_str(), &e);
      if (e || !*pipe) {
        fprintf(stderr, "[ERROR] Annotated %s pipeline failed.\n", tag);
        if (e) g_error_free(e);
        return false;
      }
      *src = gst_bin_get_by_name(GST_BIN(*pipe), "recordsrc");
      gst_element_set_state(*pipe, GST_STATE_READY);
      gst_element_set_state(*pipe, GST_STATE_PLAYING);
      return true;
    };
    if (left_enabled  && !launch_annotated(annotatedLeftFilePath,
                                           &g_annotatedPipeL,
                                           &g_annotatedSrcL,  "Left"))  return 1;
    if (right_enabled && !launch_annotated(annotatedRightFilePath,
                                           &g_annotatedPipeR,
                                           &g_annotatedSrcR, "Right")) return 1;

    printf("[REC] Session: %s\n", recordingSessionPath.c_str());
    if (dual_enabled)
      printf("[REC] Raw+annotated: {raw,annotated}_{left,right}.mkv\n");
  }
  // ── Ring buffers ─────────────────────────────────────────────
  // left_ring  — filled by tCapture,  read by tRawStream → srcL
  // right_ring — filled by tCaptureR, read by tRawStream → srcR
  DualRingBuffer left_ring;
  DualRingBuffer right_ring;
  int frameSz = tracker_W * tracker_H * 3;
  for (auto &slot : left_ring.slots) {
    slot.bgr_frame_data.resize(frameSz);
    slot.contains_valid_frame = false;
  }
  for (auto &slot : right_ring.slots) {
    slot.bgr_frame_data.resize(frameSz);
    slot.contains_valid_frame = false;
  }

  // ── Result queue ─────────────────────────────────────────────
  // Connects trackerThread → outputThread.
  // Per-camera result queues — one for each output thread to drain.
  std::vector<ResultSlot> resultQueueL, resultQueueR;
  std::mutex resultMtxL, resultMtxR;
  std::condition_variable resultCvL, resultCvR;

  // ── Thread launch ────────────────────────────────────────────
  // tCapture   — left camera  → left_ring
  // tCaptureR  — right camera → right_ring
  // tRawStreamL — left_ring  → srcL  (one worker per camera)
  // tRawStreamR — right_ring → srcR (no starvation between cameras)
  std::thread tCapture([&]() {
    if (sinkL)
      capture_thread_left(sinkL, left_ring, g_frameId, fps,
                          recEnabled, g_recordPipeL, g_recordSrcL);
  });

  std::thread tCaptureR([&]() {
    if (sinkR)
      capture_thread_right(sinkR, right_ring, g_frameId, fps,
                           recEnabled, g_recordPipeR, g_recordSrcR);
  });

  std::thread tRawStreamL(rawStreamWorker, std::ref(left_ring), srcL, 1,
                          tracker_W, tracker_H);
  std::thread tRawStreamR(rawStreamWorker, std::ref(right_ring), srcR, 2,
                          tracker_W, tracker_H);

  float tracker_center_x = tracker_W / 2.0f;
  float tracker_center_y = tracker_H / 2.0f;

  // Two tracker threads — one per camera. Each drives its own tracker
  // instance and pushes to its own result queue. No g_selected_camera
  // gating on the frame path.
  std::thread tTrackerL(trackerThread, 1, std::ref(g_tracker_L),
                        std::ref(g_mtx_L), std::ref(left_ring),
                        std::ref(resultQueueL), std::ref(resultMtxL),
                        std::ref(resultCvL), tracker_W, tracker_H, fps,
                        degPerPixelX, degPerPixelY, tracker_center_x,
                        tracker_center_y);
  std::thread tTrackerR(trackerThread, 2, std::ref(g_tracker_R),
                        std::ref(g_mtx_R), std::ref(right_ring),
                        std::ref(resultQueueR), std::ref(resultMtxR),
                        std::ref(resultCvR), tracker_W, tracker_H, fps,
                        degPerPixelX, degPerPixelY, tracker_center_x,
                        tracker_center_y);

  // Two output threads — one per camera. Peer refs are swapped so
  // each can cross-check the other and fire the handoff CAPTURE
  // seed. Only the primary (g_selected_camera) writes UART/UDP.
  std::thread tOutputL(outputThread, 1,
                        std::ref(g_tracker_L), std::ref(g_mtx_L),
                        std::ref(g_tracker_R), std::ref(g_mtx_R),
                        std::ref(resultQueueL), std::ref(resultMtxL),
                        std::ref(resultCvL),
                        srcL, g_annotatedSrcL,
                        tracker_W, tracker_H, fps, recEnabled);
  std::thread tOutputR(outputThread, 2,
                        std::ref(g_tracker_R), std::ref(g_mtx_R),
                        std::ref(g_tracker_L), std::ref(g_mtx_L),
                        std::ref(resultQueueR), std::ref(resultMtxR),
                        std::ref(resultCvR),
                        srcR, g_annotatedSrcR,
                        tracker_W, tracker_H, fps, recEnabled);

  // ── Wait ─────────────────────────────────────────────────────
  tCapture.join();
  tCaptureR.join();
  tRawStreamL.join();
  tRawStreamR.join();
  tTrackerL.join();
  tTrackerR.join();
  tOutputL.join();
  tOutputR.join();

  // ── Shutdown ─────────────────────────────────────────────────
  printf("\n[MAIN] Shutting down... total frames: %d\n", g_frameId.load());
  // ── Flight recorder shutdown ─────────────────────────────────
  // WHY: logShutdown writes the final events.jsonl entry with total
  // frame count and session duration — only if recorder was created
  // (i.e. operator actually captured a target at some point).
  if (g_rec) {
    double durationSec = g_fps > 0 ? (double)g_frameId.load() / g_fps : 0.0;
    g_rec->logShutdown(g_frameId.load(), 0, ShutdownReason::CLEAN,
                       g_frameId.load(), durationSec);
    delete g_rec;
    g_rec = nullptr;
  }
  // WHY: wait for NULL state to actually complete before unref/exit.
  // Without this, the process can exit while GStreamer is still
  // releasing the v4l2 camera device, leaving it locked/busy for
  // the next run — symptoms: camera connects intermittently or
  // fails to enumerate on subsequent launches.
  if (capPipe) {
    gst_element_set_state(capPipe, GST_STATE_NULL);
    gst_element_get_state(capPipe, nullptr, nullptr, 2 * GST_SECOND);
  }
  if (strPipe) {
    gst_element_set_state(strPipe, GST_STATE_NULL);
    gst_element_get_state(strPipe, nullptr, nullptr, 2 * GST_SECOND);
  }
  if (capPipeR) {
    gst_element_set_state(capPipeR, GST_STATE_NULL);
    gst_element_get_state(capPipeR, nullptr, nullptr, 2 * GST_SECOND);
  }
  if (strPipeR) {
    gst_element_set_state(strPipeR, GST_STATE_NULL);
    gst_element_get_state(strPipeR, nullptr, nullptr, 2 * GST_SECOND);
  }

  if (sinkL)
    gst_object_unref(sinkL);
  if (srcL)
    gst_object_unref(srcL);
  if (sinkR)
    gst_object_unref(sinkR);
  if (srcR)
    gst_object_unref(srcR);

  if (capPipe)
    gst_object_unref(capPipe);
  if (strPipe)
    gst_object_unref(strPipe);
  if (capPipeR)
    gst_object_unref(capPipeR);
  if (strPipeR)
    gst_object_unref(strPipeR);

  // ── Socket and fd cleanup ────────────────────────────────────
  if (g_telemSock >= 0)
    close(g_telemSock);
  if (g_uartFd >= 0)
    close(g_uartFd);
  if (g_ctrlSock >= 0)
    close(g_ctrlSock);

#ifdef TRACKER_V2
  // Release the TensorRT extractor explicitly, before g_tracker's own
  // destructor runs at static-destruction time. Without this, the
  // engine/context get torn down in an order CUDA doesn't guarantee is
  // safe, causing the segfault-on-exit seen after "Shutting down".
  g_tracker_L.setFeatureExtractor(nullptr);
  g_tracker_R.setFeatureExtractor(nullptr);
#endif
  printf("[MAIN] Done.\n");
  return 0;
}
