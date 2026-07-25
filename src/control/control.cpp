#include "control.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <fstream>

bool saveParams() {
  std::string tmpFile = CONFIG_FILE + ".tmp";
  std::ofstream f(tmpFile);

  if (!f.is_open()) {
    fprintf(stderr, "[CFG] Cannot open %s for writing: %s\n", tmpFile.c_str(),
            strerror(errno));
    return false;
  }

  for (int id = 1; id <= MAX_TRACKER_PARAM_ID; id++) {
    float val = g_tracker.getParam(static_cast<cr::vtracker::VTrackerParam>(id));
    f << id << "=" << val << "\n";
  }
  f.close(); // flush + close before atomic rename

  if (!f) {
    fprintf(stderr, "[CFG] Write to %s failed: %s\n", tmpFile.c_str(),
            strerror(errno));
    unlink(tmpFile.c_str());
    return false;
  }

  if (rename(tmpFile.c_str(), CONFIG_FILE.c_str()) != 0) {
    fprintf(stderr, "[CFG] rename(%s → %s) failed: %s\n", tmpFile.c_str(),
            CONFIG_FILE.c_str(), strerror(errno));
    unlink(tmpFile.c_str());
    return false;
  }

  // Byte count of the written cfg so operators can eyeball the tail and
  // instantly tell whether persist actually landed on disk (23 params
  // takes ~130 bytes; anything drastically smaller means truncation).
  struct stat st{};
  if (stat(CONFIG_FILE.c_str(), &st) == 0) {
    printf(LOG_CYAN "[CFG]" LOG_RESET " Persisted %d params to %s "
           LOG_DIM "(%lld bytes)" LOG_RESET "\n",
           MAX_TRACKER_PARAM_ID, CONFIG_FILE.c_str(), (long long)st.st_size);
  } else {
    printf("[CFG] Persisted %d params to %s (size unknown: %s)\n",
           MAX_TRACKER_PARAM_ID, CONFIG_FILE.c_str(), strerror(errno));
  }
  return true;
}

bool loadParams() {
  std::ifstream f(CONFIG_FILE);
  if (!f.is_open()) {
    printf("[CFG] No config file found at %s. Using system defaults.\n",
           CONFIG_FILE.c_str());
    saveParams(); // Generate a standard default file containing up to 23 params
                  // right away
    return false;
  }
  std::string line;
  int loaded = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    try {
      int id = std::stoi(line.substr(0, eq));
      float val = std::stof(line.substr(eq + 1));

      // Accept and pass all known parameters directly into the core tracker engine
      if (id >= 1 && id <= MAX_TRACKER_PARAM_ID) {
        g_tracker.setParam(static_cast<cr::vtracker::VTrackerParam>(id), val);
        loaded++;
      }
    } catch (...) {
      // Catch formatting anomalies silently
    }
  }
  printf("[CFG] Loaded %d params from %s\n", loaded, CONFIG_FILE.c_str());
  return loaded > 0;
}

int uartOpen(const char *device, int baudrate) {
  int fd = open(device, O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0)
    return -1;
  struct termios tty {};
  if (tcgetattr(fd, &tty) != 0) {
    close(fd);
    return -1;
  }
  speed_t baud;
  switch (baudrate) {
  case 9600:
    baud = B9600;
    break;
  case 19200:
    baud = B19200;
    break;
  case 38400:
    baud = B38400;
    break;
  case 57600:
    baud = B57600;
    break;
  case 115200:
    baud = B115200;
    break;
  case 230400:
    baud = B230400;
    break;
  case 460800:
    baud = B460800;
    break;
  default:
    baud = B115200;
    break;
  }
  cfsetospeed(&tty, baud);
  cfsetispeed(&tty, baud);
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
  tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tty.c_oflag &= ~OPOST;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  tcflush(fd, TCIFLUSH);
  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    close(fd);
    return -1;
  }
  // Open with O_NONBLOCK so open() never blocks waiting on carrier, but then
  // clear it so writes are BLOCKING. A non-blocking write can transmit only
  // part of a 21-byte packet (or return EAGAIN and send none) when the TX
  // buffer is briefly full; the dropped tail desyncs the receiver's framing
  // and every following checksum fails — the intermittent, load/device-
  // dependent corruption we chased. Blocking writes (as in the proven
  // reference implementation) guarantee each packet leaves whole.
  int fl = fcntl(fd, F_GETFL, 0);
  if (fl != -1)
    fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
  return fd;
}

void uartSendAngle(int fd, int mode, float angle_x, float angle_y,
                   float det_prob, int pixel_x, int pixel_y, int confirmed) {
  if (fd < 0)
    return;
  UartAnglePacket pkt{};
  pkt.header[0] = 0xAA;
  pkt.header[1] = 0x55;
  pkt.mode = (uint8_t)mode;
  pkt.angle_x_deg = angle_x;
  pkt.angle_y_deg = angle_y;
  pkt.det_prob = det_prob;
  pkt.pixel_x = (int16_t)pixel_x;
  pkt.pixel_y = (int16_t)pixel_y;
  pkt.confirmed = confirmed ? 1 : 0;
  uint8_t cs = 0;
  const uint8_t *raw = reinterpret_cast<const uint8_t *>(&pkt);
  // Checksum covers every byte up to (not including) the checksum field,
  // so the new confirmed byte is automatically included.
  for (size_t i = 0; i < offsetof(UartAnglePacket, checksum); i++)
    cs ^= raw[i];
  pkt.checksum = cs;
  // Write the whole packet. Even on a blocking fd a write can be cut short by
  // a signal (EINTR) or a partial completion, so loop until all bytes are out
  // — a half-sent frame would corrupt the receiver's framing.
  const uint8_t *out = reinterpret_cast<const uint8_t *>(&pkt);
  size_t remaining = sizeof(pkt);
  while (remaining > 0) {
    ssize_t n = write(fd, out, remaining);
    if (n > 0) {
      out += n;
      remaining -= (size_t)n;
    } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
      continue; // interrupted or (defensively) would-block: retry the rest
    } else {
      break; // real error — drop this packet rather than spin forever
    }
  }
}

void sendAck(uint32_t cmdType, uint32_t paramId, float value, bool success) {
  std::lock_guard<std::mutex> lk(g_gsMtx);
  if (!g_gsAddrValid || g_ctrlSock < 0)
    return;
  AckPacket ack{};
  ack.magic = ACK_MAGIC;
  ack.ack_type = cmdType;
  ack.param_id = paramId;
  ack.value = value;
  ack.success = success ? 1 : 0;
  ack.reserved = 0;
  sendto(g_ctrlSock, &ack, sizeof(ack), 0,
         reinterpret_cast<sockaddr *>(&g_gsAddr), sizeof(g_gsAddr));
}

// ── Camera parameter ID lookup table ──────────────────────────
// WHY: raw V4L2_CID_* values are large hex constants — sending them
// over UDP as-is is clunky and error-prone from the GUI side.
// Map simple sequential IDs (1-7) to actual V4L2 control IDs instead.
struct CameraParamEntry {
  int simple_id;
  uint32_t v4l2_id;
  const char *name;
};

static const CameraParamEntry kCameraParams[] = {
    {1, V4L2_CID_BRIGHTNESS, "brightness"},
    {2, V4L2_CID_CONTRAST, "contrast"},
    {3, V4L2_CID_SATURATION, "saturation"},
    {4, V4L2_CID_GAMMA, "gamma"},
    {5, V4L2_CID_GAIN, "gain"},
    {6, V4L2_CID_SHARPNESS, "sharpness"},
    {7, V4L2_CID_EXPOSURE_ABSOLUTE, "exposure"},
};

static const uint32_t kNumCameraParams =
    sizeof(kCameraParams) / sizeof(kCameraParams[0]);

// ── Camera parameter control (V4L2 ioctl, no subprocess) ─────
// WHY: direct ioctl is faster and safer than spawning v4l2-ctl as
// a subprocess — no fork/exec overhead, no shell injection risk,
// immediate error codes.
bool setCameraControl(const char *device, uint32_t ctrl_id, int32_t value) {
  int fd = open(device, O_RDWR);
  if (fd < 0) {
    fprintf(stderr, "[CAMCTL] Cannot open %s: %s\n", device, strerror(errno));
    return false;
  }

  // EXPOSURE_ABSOLUTE is gated by the driver until EXPOSURE_AUTO is switched
  // to manual — otherwise VIDIOC_S_CTRL returns EPIPE ("Broken pipe"), which
  // is what we saw on both cameras in the field-test log. Best-effort switch
  // to manual first; ignore its result so cameras that don't expose the auto
  // control (or are already manual) still get the absolute value applied.
  if (ctrl_id == V4L2_CID_EXPOSURE_ABSOLUTE) {
    struct v4l2_control auto_ctrl {};
    auto_ctrl.id = V4L2_CID_EXPOSURE_AUTO;
    auto_ctrl.value = V4L2_EXPOSURE_MANUAL; // = 1
    (void)ioctl(fd, VIDIOC_S_CTRL, &auto_ctrl);
  }

  struct v4l2_control ctrl {};
  ctrl.id = ctrl_id;
  ctrl.value = value;

  bool ok = (ioctl(fd, VIDIOC_S_CTRL, &ctrl) == 0);
  if (!ok)
    fprintf(stderr, "[CAMCTL] Failed to set ctrl 0x%08x on %s: %s\n", ctrl_id,
            device, strerror(errno));
  close(fd);
  return ok;
}

// Read a V4L2 control. Returns true on success (value in out_value); false
// on open/ioctl failure. Uses an out-param, not a sentinel, because some
// controls (e.g. brightness) have legitimately negative values.
bool getCameraControl(const char *device, uint32_t ctrl_id,
                      int32_t &out_value) {
  int fd = open(device, O_RDWR);
  if (fd < 0) {
    fprintf(stderr, "[CAMCTL] Cannot open %s: %s\n", device, strerror(errno));
    return false;
  }

  struct v4l2_control ctrl {};
  ctrl.id = ctrl_id;
  if (ioctl(fd, VIDIOC_G_CTRL, &ctrl) != 0) {
    fprintf(stderr, "[CAMCTL] Failed to get ctrl 0x%08x on %s: %s\n", ctrl_id,
            device, strerror(errno));
    close(fd);
    return false;
  }
  close(fd);
  out_value = ctrl.value;
  return true;
}

// =====================================================================
// Control listener thread
// =====================================================================
void controlThread(int port, int camera_id, const char *device_path) {
  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    perror("[CTRL] socket");
    return;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    perror("[CTRL] bind");
    close(sock);
    return;
  }

  // Store socket fd for ACK responses
  {
    std::lock_guard<std::mutex> lk(g_gsMtx);
    g_ctrlSock = sock;
  }

  timeval tv{};
  tv.tv_usec = 200000;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  printf("[CTRL] Listening for commands on UDP port %d\n", port);

  CmdPacket pkt{};
  sockaddr_in senderAddr{};
  socklen_t senderLen = sizeof(senderAddr);

  while (g_running) {
    senderLen = sizeof(senderAddr);
    ssize_t n = recvfrom(sock, &pkt, sizeof(pkt), 0,
                         reinterpret_cast<sockaddr *>(&senderAddr), &senderLen);
    if (n < 0)
      continue;  // timeout or EAGAIN
    if (n != (ssize_t)sizeof(pkt)) {
      printf("[CTRL-%s] rx malformed: %zd bytes (expected %zu)\n",
             camera_id == 1 ? "L" : "R", n, sizeof(pkt));
      fflush(stdout);
      continue;
    }
    if (pkt.magic != CMD_MAGIC) {
      printf("[CTRL-%s] rx wrong magic: 0x%08x (expected 0x%08x)\n",
             camera_id == 1 ? "L" : "R", pkt.magic, CMD_MAGIC);
      fflush(stdout);
      continue;
    }

    // Remember ground station address for ACK responses
    {
      std::lock_guard<std::mutex> lk(g_gsMtx);
      g_gsAddr = senderAddr;
      g_gsAddrValid = true;
    }

    // No top-level tracker lock: g_mtx aliases g_mtx_L, so grabbing it
    // here would deadlock with the per-tracker lock the handlers take
    // below (CAPTURE on camera 1 double-locks g_mtx_L; CAPTURE on
    // camera 2 starves the L tracker). Each case that mutates a
    // tracker now takes only that tracker's mutex.

    // Log the raw command into events.jsonl before dispatching. Recorded
    // fields are frame + camera_id + cmdType + full arg payload — this
    // is what a future replay tool needs to reproduce operator intent
    // (in particular, CAPTURE / SET_RECT_POS click coordinates).
    // pts_ns=0 here; frame is authoritative for replay indexing.
    if (recEnabled && g_rec) {
      g_rec->logCmd(g_frameId.load(), 0, camera_id, pkt.type, pkt.arg1,
                    pkt.arg2, pkt.arg3);
    }

    switch (pkt.type) {
    case CMD_CAPTURE: {
      printf("[CTRL-%s] CAPTURE at (%.0f, %.0f)\n", camera_id == 1 ? "L" : "R",
             pkt.arg1, pkt.arg2);
      fflush(stdout);
      int expected = 0;
      if (g_selected_camera.compare_exchange_strong(expected, camera_id)) {
        printf("[CTRL] Camera %s locked for this session.\n",
               camera_id == 1 ? "LEFT" : "RIGHT");

        // ── Flight recorder — created once, on first CAPTURE ────
        // WHY: metadata must reflect the camera actually tracked,
        // which is only known once the operator locks a camera.
        if (recEnabled && !g_rec) {
          SessionInfo info = g_sessionInfo;
          info.selectedCameraId = camera_id;
          g_rec = new FlightRecorder(info);

          TrackerConfig tcfg;
          tcfg.version = CvTracker::getVersion();
          tcfg.rect_width = (int)g_tracker.getParam(VTrackerParam::RECT_WIDTH);
          tcfg.rect_height =
              (int)g_tracker.getParam(VTrackerParam::RECT_HEIGHT);
          tcfg.search_window_width =
              (int)g_tracker.getParam(VTrackerParam::SEARCH_WINDOW_WIDTH);
          tcfg.search_window_height =
              (int)g_tracker.getParam(VTrackerParam::SEARCH_WINDOW_HEIGHT);
          tcfg.lost_mode_option =
              (int)g_tracker.getParam(VTrackerParam::LOST_MODE_OPTION);
          tcfg.num_channels =
              (int)g_tracker.getParam(VTrackerParam::NUM_CHANNELS);
          tcfg.multiple_threads =
              (int)g_tracker.getParam(VTrackerParam::MULTIPLE_THREADS);
          tcfg.type = (int)g_tracker.getParam(VTrackerParam::TYPE);
          tcfg.frame_buffer_size =
              (int)g_tracker.getParam(VTrackerParam::FRAME_BUFFER_SIZE);
          tcfg.custom_1 = g_tracker.getParam(VTrackerParam::CUSTOM_1);
          tcfg.custom_2 = g_tracker.getParam(VTrackerParam::CUSTOM_2);
          g_rec->writeMetaJson(tcfg);
          g_rec->logStart(g_frameId.load(), 0, camera_id);
          // The CAPTURE command that just created the recorder was skipped
          // by the early logCmd gate above (g_rec was null then). Log it
          // now so replay sees the click that started everything.
          g_rec->logCmd(g_frameId.load(), 0, camera_id, pkt.type, pkt.arg1,
                        pkt.arg2, pkt.arg3);
          printf("[REC] Flight recorder started (camera %s).\n",
                 camera_id == 2 ? "RIGHT" : "LEFT");
        }

        // Dual-lock: dispatch CAPTURE to the tracker matching camera_id.
        // g_selected_camera still tracks "primary" for UART/UDP routing.
        auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
        auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
        std::lock_guard<std::mutex> lk(mx);
        // Wipe any prior template/DNN bank/EKF state before seeding
        // the new ROI. Prevents the previous target from "winning" if
        // it re-enters the frame after re-CAPTURE on a new object.
        tk.executeCommand(VTrackerCommand::RESET);
        tk.executeCommand(VTrackerCommand::CAPTURE, pkt.arg1, pkt.arg2,
                          pkt.arg3);
        g_target_confirmed.store(0);
      } else if (g_selected_camera.load() == camera_id) {
        // Re-capture on the same (primary) camera.
        printf("[CTRL-%s] Re-capture at (%.0f, %.0f)\n",
               camera_id == 1 ? "L" : "R", pkt.arg1, pkt.arg2);
        auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
        auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
        {
          std::lock_guard<std::mutex> lk(mx);
          tk.executeCommand(VTrackerCommand::RESET);
          tk.executeCommand(VTrackerCommand::CAPTURE, pkt.arg1, pkt.arg2,
                            pkt.arg3);
        }
        // Re-capturing the primary means "track a NEW object", so drop the
        // peer's lock too. RESET clears its correlation template, EKF AND the
        // DNN bank that was transferred at the earlier handoff
        // (resetToFree -> resetVerifier -> bank.clear()), returning it to
        // FREE. The auto-handoff (streaming.cpp) only seeds a FREE/LOST peer,
        // so this lets the new selection re-drive the peer once it enters the
        // overlap. Without it the peer keeps tracking the previously handed
        // ROI (both its template and the old bank still match it). Separate
        // lock scope so we never hold both tracker mutexes at once.
        auto &peer = (camera_id == 2) ? g_tracker_L : g_tracker_R;
        auto &pmx  = (camera_id == 2) ? g_mtx_L    : g_mtx_R;
        {
          std::lock_guard<std::mutex> plk(pmx);
          peer.executeCommand(VTrackerCommand::RESET);
        }
        printf("[CTRL] Primary re-capture -> RESET peer camera %s so it "
               "re-acquires the NEW target on next handoff\n",
               camera_id == 2 ? "LEFT/BORE" : "RIGHT/DEPR");
        fflush(stdout);
        g_target_confirmed.store(0);
      } else {
        // Operator captured on the secondary camera — switch primary to it
        // and seed the NEW ROI. A new selection means "track this object",
        // so we also drop the OLD primary's (now peer's) lock: otherwise it
        // keeps tracking the previous target (its template + DNN bank still
        // match it), which shows up as the peer camera "locking onto the
        // previous ROI". The auto-handoff (streaming.cpp) re-seeds the now-
        // FREE peer with the new target once it enters the overlap. This
        // supersedes the old dual-lock "keep both independent" behaviour.
        printf("[CTRL-%s] Secondary-camera CAPTURE — switching primary "
               "from %s to %s\n",
               camera_id == 1 ? "L" : "R",
               g_selected_camera.load() == 1 ? "LEFT" : "RIGHT",
               camera_id == 1 ? "LEFT" : "RIGHT");
        auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
        auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
        {
          std::lock_guard<std::mutex> lk(mx);
          tk.executeCommand(VTrackerCommand::RESET);
          tk.executeCommand(VTrackerCommand::CAPTURE, pkt.arg1, pkt.arg2,
                            pkt.arg3);
        }
        // Reset the old primary (now the peer) in its own lock scope so we
        // never hold both tracker mutexes at once.
        auto &peer = (camera_id == 2) ? g_tracker_L : g_tracker_R;
        auto &pmx  = (camera_id == 2) ? g_mtx_L    : g_mtx_R;
        {
          std::lock_guard<std::mutex> plk(pmx);
          peer.executeCommand(VTrackerCommand::RESET);
        }
        printf("[CTRL] Secondary CAPTURE -> RESET peer camera %s so it "
               "re-acquires the NEW target on next handoff\n",
               camera_id == 2 ? "LEFT/BORE" : "RIGHT/DEPR");
        fflush(stdout);
        g_selected_camera.store(camera_id);
        g_target_confirmed.store(0);
      }
      break;
    }
    case CMD_RESET: {
      // Dual-lock: reset only the sender's tracker so the peer keeps
      // its lock if it has one.
      printf("[CTRL-%s] RESET\n", camera_id == 1 ? "L" : "R");
      auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
      auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
      std::lock_guard<std::mutex> lk(mx);
      tk.executeCommand(VTrackerCommand::RESET);
      g_target_confirmed.store(0);
      break;
    }
    case CMD_CHANGE_SIZE: {
      auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
      auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
      std::lock_guard<std::mutex> lk(mx);
      tk.executeCommand(VTrackerCommand::CHANGE_RECT_SIZE, pkt.arg1, pkt.arg2);
      break;
    }
    case CMD_SET_RECT_POS: {
      auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
      auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
      std::lock_guard<std::mutex> lk(mx);
      tk.executeCommand(VTrackerCommand::SET_RECT_POSITION, pkt.arg1, pkt.arg2);
      break;
    }

    case CMD_SET_PARAM: {
      // WHY: g_tracker is a single shared instance across BOTH control
      // threads (see main.cpp — one CvTracker, two controlThreads).
      // Gating SET_PARAM by camera lock made the "off-camera" port
      // silently swallow every parameter change with no ACK, no log —
      // that was the biggest cause of "Jetson isn't receiving values"
      // after any CAPTURE. Params are tracker-wide, not per-camera,
      // so either port may tune them. CAPTURE / RESET stay gated
      // above because those are per-camera operations.
      int paramId = static_cast<int>(pkt.arg1);
      float value = pkt.arg2;
      bool ok = false;

      // Gate RECT_WIDTH / RECT_HEIGHT by camera lock: a drag on the
      // non-selected canvas would otherwise resize the shared tracker's
      // rectangle and visibly move the box on the LOCKED camera. All
      // other params stay tracker-wide so tuning still works from
      // either port (see rationale in the comment above).
      constexpr int RECT_WIDTH_ID  =
          static_cast<int>(VTrackerParam::RECT_WIDTH);
      constexpr int RECT_HEIGHT_ID =
          static_cast<int>(VTrackerParam::RECT_HEIGHT);
      const bool is_rect_param =
          (paramId == RECT_WIDTH_ID || paramId == RECT_HEIGHT_ID);
      const int  locked = g_selected_camera.load();
      const bool camera_allowed =
          (!is_rect_param) || (locked == 0) || (locked == camera_id);

      if (paramId >= 1 && paramId <= MAX_TRACKER_PARAM_ID && camera_allowed) {
        // Dual-lock: broadcast tuning to BOTH trackers so they stay
        // symmetric. Each lock is taken independently — the two
        // mutexes are separate.
        bool okL, okR;
        {
          std::lock_guard<std::mutex> lkL(g_mtx_L);
          okL = g_tracker_L.setParam(
              static_cast<VTrackerParam>(paramId), value);
        }
        {
          std::lock_guard<std::mutex> lkR(g_mtx_R);
          okR = g_tracker_R.setParam(
              static_cast<VTrackerParam>(paramId), value);
        }
        ok = okL && okR;
        // Also fake the legacy single-tracker path so the following
        // code (which reads back g_tracker.getParam) sees the value.
        (void)okL; (void)okR;
      } else if (is_rect_param && !camera_allowed) {
        printf("[CTRL-%s] SET_PARAM id=%d ignored — rect params locked to %s\n",
               camera_id == 1 ? "L" : "R", paramId,
               locked == 1 ? "LEFT" : "RIGHT");
      }

      printf(LOG_YELLOW "[CTRL-%s]" LOG_RESET
             " SET_PARAM id=%d value=%.4f -> %s%s" LOG_RESET "\n",
             camera_id == 1 ? "L" : "R", paramId, value,
             ok ? LOG_GREEN : LOG_RED, ok ? "OK" : "FAIL");
      sendAck(CMD_SET_PARAM, paramId, value, ok);
      break;
    }

    case CMD_GET_PARAMS: {
      // No log — the GUI polls GET_PARAMS continuously to keep its param
      // panel in sync, so printing here just floods the console.
      for (int id = 1; id <= MAX_TRACKER_PARAM_ID; id++) {
        float val = g_tracker.getParam(static_cast<VTrackerParam>(id));
        sendAck(CMD_GET_PARAMS, id, val, true);
        usleep(1000); 
      }
      break;
    }
    
    case CMD_SAVE_PARAMS: {
      // WHY: previous version ACKed true unconditionally, so a disk
      // failure looked identical to a successful persist in the GUI.
      // Report the real result so the operator sees it fail.
      printf(LOG_YELLOW "[CTRL-%s]" LOG_RESET " SAVE_PARAMS\n",
             camera_id == 1 ? "L" : "R");
      bool ok = saveParams();
      sendAck(CMD_SAVE_PARAMS, 0, 0.0f, ok);
      break;
    }

    case CMD_SET_CAMERA_PARAM: {
      // WHY: pkt.arg1 = simple_id (1-7), pkt.arg2 = value.
      // Looked up against kCameraParams to get the real V4L2_CID_*.
      int simple_id = (int)pkt.arg1;
      int32_t value = (int32_t)pkt.arg2;
      bool ok = false;
      for (uint32_t i = 0; i < kNumCameraParams; i++) {
        if (kCameraParams[i].simple_id == simple_id) {
          ok = setCameraControl(device_path, kCameraParams[i].v4l2_id, value);
          printf("[CTRL-%s] SET_CAMERA_PARAM %s=%d ? %s\n",
                 camera_id == 1 ? "L" : "R", kCameraParams[i].name, value,
                 ok ? "OK" : "FAILED");
          break;
        }
      }
      sendAck(pkt.type, simple_id, (float)value, ok);
      break;
    }

    case CMD_GET_CAMERA_PARAM: {
      // Read every V4L2 control from this camera's device and reply with one
      // ack per param (id = simple id 1..7, value = current value) so the GUI
      // can sync its sliders to the real camera state. Same ack pattern as
      // CMD_GET_PARAMS.
      for (uint32_t i = 0; i < kNumCameraParams; i++) {
        int32_t value = 0;
        bool ok = getCameraControl(device_path, kCameraParams[i].v4l2_id,
                                   value);
        sendAck(CMD_GET_CAMERA_PARAM, kCameraParams[i].simple_id, (float)value,
                ok);
        usleep(1000);
      }
      break;
    }

    case CMD_HANDOFF:
      printf("[CTRL-%s] HANDOFF requested — unlocking camera.\n",
             camera_id == 1 ? "L" : "R");
      g_selected_camera.store(0);
      g_handoff_requested.store(true);
      // Dual-lock: reset only the sender's tracker; peer keeps its lock.
      {
        auto &tk = (camera_id == 2) ? g_tracker_R : g_tracker_L;
        auto &mx = (camera_id == 2) ? g_mtx_R    : g_mtx_L;
        std::lock_guard<std::mutex> lk(mx);
        tk.executeCommand(VTrackerCommand::RESET, 0, 0, 0);
      }
      g_target_confirmed.store(0);
      sendAck(pkt.type, 0, 0, true);
      break;

    case CMD_HANDOFF_MANUAL: {
      // arg1 = source camera id (1=L, 2=R), arg2/arg3 = source pixel
      // (negative → use current tracker rect centre). Handoff computes
      // the destination-camera pixel via the plane-induced homography
      // and locks the destination tracker onto it.
      const int source_cam = (int)pkt.arg1;
      if (!g_handoff.ready()) {
        printf(LOG_RED "[HANDOFF]" LOG_RESET " model not initialised — "
                       "rejected. Check stereo_calib.json + target depth.\n");
        sendAck(pkt.type, 0, 0, false);
        break;
      }
      if (source_cam != 1 && source_cam != 2) {
        printf(LOG_RED "[HANDOFF]" LOG_RESET
               " invalid source camera id %d — expected 1 or 2\n", source_cam);
        sendAck(pkt.type, 0, 0, false);
        break;
      }

      double src_u = (double)pkt.arg2;
      double src_v = (double)pkt.arg3;
      if (src_u < 0.0 || src_v < 0.0) {
        // Fall back to the source camera's current tracker rect centre.
        // Only meaningful when the source camera is currently locked —
        // otherwise rectX/rectY reflect stale state.
        int locked = g_selected_camera.load();
        if (locked != source_cam) {
          printf(LOG_RED "[HANDOFF]" LOG_RESET " source cam %d has no live "
                         "tracker rect (currently locked=%d) and no "
                         "explicit pixel supplied\n", source_cam, locked);
          sendAck(pkt.type, 0, 0, false);
          break;
        }
        // g_last_rect_x/y are updated every output frame by streaming.cpp.
        src_u = (double)g_last_rect_x.load(std::memory_order_relaxed);
        src_v = (double)g_last_rect_y.load(std::memory_order_relaxed);
      }

      double dst_u = 0.0, dst_v = 0.0;
      const bool ok = (source_cam == 1)
          ? g_handoff.projectLtoR(src_u, src_v, dst_u, dst_v)
          : g_handoff.projectRtoL(src_u, src_v, dst_u, dst_v);
      if (!ok) {
        printf(LOG_RED "[HANDOFF]" LOG_RESET
               " projection failed (behind camera / bad geometry)\n");
        sendAck(pkt.type, 0, 0, false);
        break;
      }

      const int dst_cam = (source_cam == 1) ? 2 : 1;
      printf(LOG_CYAN "[HANDOFF]" LOG_RESET
             " %s (%.1f, %.1f) → %s (%.1f, %.1f)  depth=%.1fmm\n",
             source_cam == 1 ? "LEFT" : "RIGHT",
             src_u, src_v,
             dst_cam == 1 ? "LEFT" : "RIGHT",
             dst_u, dst_v,
             g_handoff.planeDepthMm());

      // Dual-lock: CAPTURE on the DESTINATION tracker; source keeps
      // running independently. Primary switches to destination so
      // UART/UDP follow the operator's explicit intent.
      auto &dst_tk = (dst_cam == 2) ? g_tracker_R : g_tracker_L;
      auto &dst_mx = (dst_cam == 2) ? g_mtx_R    : g_mtx_L;
      {
        std::lock_guard<std::mutex> lk(dst_mx);
        dst_tk.executeCommand(VTrackerCommand::CAPTURE,
                              (float)dst_u, (float)dst_v, -1.0f);
      }
      g_selected_camera.store(dst_cam);
      g_target_confirmed.store(0);
      sendAck(pkt.type, (uint32_t)dst_cam, (float)dst_u, true);
      break;
    }

    case CMD_CONFIRM_TARGET: {
      // Operator asserts (arg1 != 0) or retracts (arg1 == 0) that the
      // currently-tracked object is the correct one. Tracker-wide flag
      // (single shared tracker), so either control port may set it.
      int confirmed = (pkt.arg1 != 0.0f) ? 1 : 0;
      g_target_confirmed.store(confirmed);
      printf(LOG_YELLOW "[CTRL-%s]" LOG_RESET " CONFIRM_TARGET -> %s%s" LOG_RESET
             "\n", camera_id == 1 ? "L" : "R",
             confirmed ? LOG_GREEN : LOG_DIM,
             confirmed ? "CONFIRMED" : "unconfirmed");
      sendAck(CMD_CONFIRM_TARGET, 0, (float)confirmed, true);
      break;
    }

    default:
      printf("[CTRL] Unknown type %u\n", pkt.type);
      break;
    }
  }

  close(sock);
  printf("[CTRL] Thread exiting.\n");
}
