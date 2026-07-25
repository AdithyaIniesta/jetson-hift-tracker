// ============================================================
// pipeline.cpp
// GStreamer pipeline construction for dual camera capture
// and UDP stream output.
// WHY: isolating pipeline string construction here means
// main.cpp never needs to know about GStreamer syntax —
// it just calls build_capture_pipeline() and gets a pipeline.
// ============================================================

#include "pipeline.h"

#include "../common/globals.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ------------------------------------------------------------
// build_capture_pipeline_string
// Constructs the GStreamer pipeline description string for
// one camera based on its pixel format and board type.
// WHY: pipeline string varies by format (UYVY/YUYV/MJPG) and
// by board (Xavier needs extra nvvidconv, Orin does not).
// ------------------------------------------------------------
// ------------------------------------------------------------
// Optional benchmark / replay hook.
// When the environment variable TRACKER_FILE_SRC_LEFT (or _RIGHT) is
// set to a printf-style path (e.g. "/data/uav123/bike1/%06d.jpg"),
// this side's capture pipeline reads JPGs from disk via multifilesrc
// instead of the real v4l2 camera. Everything downstream (jpegdec,
// videoconvert, optional videoscale, appsink) is identical to the
// MJPG live path, so the tracker's frame-processing sees byte-
// identical bytes as it would from the camera.
//
// Env vars are OPT-IN — with both unset the pipeline builder produces
// exactly the same v4l2src pipeline as before. Production behaviour
// unchanged. The v4l2loopback benchmark path becomes optional.
// ------------------------------------------------------------
static std::string
build_capture_pipeline_string(const CameraConfig &camera_config) {
  char pipeline_string_buffer[768];

  // Extract config for cleaner code
  const std::string &fmt = camera_config.pixel_format;
  const std::string &device = camera_config.video_device_path;
  int width = camera_config.capture_width_pixels;
  int height = camera_config.capture_height_pixels;
  int out_width = camera_config.output_width_pixels;
  int out_height = camera_config.output_height_pixels;
  int fps = camera_config.frames_per_second;
  bool xavier = camera_config.requires_xavier_nvvidconv;

  // WHY: only insert videoscale + caps when output size differs from
  // capture size. Avoids unnecessary CPU cost when not downsampling.
  bool needs_scale = (out_width != width || out_height != height);
  char scale_stage[128] = "";
  if (needs_scale) {
    snprintf(scale_stage, sizeof(scale_stage),
             "videoscale ! video/x-raw,width=%d,height=%d ! ", out_width,
             out_height);
  }

  // Optional file-source override for benchmarks (opt-in, per side).
  //   TRACKER_FILE_SRC_LEFT  → replaces the left  camera's source
  //   TRACKER_FILE_SRC_RIGHT → replaces the right camera's source
  // The path is a printf-style pattern that multifilesrc understands
  // (e.g. ".../bike1/%06d.jpg"). Everything downstream is untouched.
  const char *file_src_env = nullptr;
  if (device == "/dev/video0" || device.find("boresight") != std::string::npos) {
    file_src_env = std::getenv("TRACKER_FILE_SRC_LEFT");
  } else if (device == "/dev/video2" || device.find("depression") != std::string::npos) {
    file_src_env = std::getenv("TRACKER_FILE_SRC_RIGHT");
  }
  // Fallback: whichever side the device is, honour a generic
  // TRACKER_FILE_SRC for single-camera benchmarks.
  if (!file_src_env)
    file_src_env = std::getenv("TRACKER_FILE_SRC");

  if (file_src_env && file_src_env[0] != '\0') {
    // multifilesrc's default start-index=0 makes it look for
    // 000000.jpg. UAV123 and most datasets start at 000001.jpg, so the
    // pipeline never gets a first buffer and fails to reach PLAYING.
    // Honour TRACKER_FILE_SRC_START_INDEX (default 1).
    const char *idx_env = std::getenv("TRACKER_FILE_SRC_START_INDEX");
    int start_index = 1;
    if (idx_env && idx_env[0] != '\0') {
        start_index = atoi(idx_env);
        if (start_index < 0) start_index = 1;
    }
    fprintf(stdout,
            "[PIPELINE] file-source override: %s (start-index=%d, "
            "was device=%s)\n",
            file_src_env, start_index, device.c_str());
    // sync=false on the appsink: buffers from multifilesrc have no
    // real timestamps, and sync=true would stall the pipeline waiting
    // for a wall-clock match that never comes.
    snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
             "multifilesrc location=%s start-index=%d loop=false "
             "caps=image/jpeg,framerate=%d/1 ! "
             "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
             "%s"
             "appsink name=sink emit-signals=false sync=false "
             "max-buffers=2 drop=true",
             file_src_env, start_index, fps, scale_stage);
    return std::string(pipeline_string_buffer);
  }

  if (fmt == "UYVY") {
    if (xavier) {
      // WHY: Xavier NX requires nvvidconv to move frames
      // into NVMM memory before videoconvert can produce BGR.
      snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
               "v4l2src device=%s ! "
               "video/x-raw,format=UYVY,width=%d,height=%d,framerate=%d/1 ! "
               "nvvidconv ! video/x-raw,format=BGRx ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "%s"
               "appsink name=sink emit-signals=false sync=false "
               "max-buffers=2 drop=true",
               device.c_str(), width, height, fps, scale_stage);
    } else {
      // WHY: Orin NX handles direct videoconvert without nvvidconv.
      snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
               "v4l2src device=%s ! "
               "video/x-raw,format=UYVY,width=%d,height=%d,framerate=%d/1 ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "%s"
               "appsink name=sink emit-signals=false sync=false "
               "max-buffers=2 drop=true",
               device.c_str(), width, height, fps, scale_stage);
    }

  } else if (fmt == "YUYV" || fmt == "YUY2") {
    if (xavier) {
      snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
               "v4l2src device=%s ! "
               "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! "
               "nvvidconv ! video/x-raw,format=BGRx ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "%s"
               "appsink name=sink emit-signals=false sync=false "
               "max-buffers=2 drop=true",
               device.c_str(), width, height, fps, scale_stage);
    } else {
      snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
               "v4l2src device=%s ! "
               "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "%s"
               "appsink name=sink emit-signals=false sync=false "
               "max-buffers=2 drop=true",
               device.c_str(), width, height, fps, scale_stage);
    }

  } else {
    // WHY: MJPG path — jpegdec decompresses before videoconvert.
    snprintf(pipeline_string_buffer, sizeof(pipeline_string_buffer),
             "v4l2src device=%s ! "
             "image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
             "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
             "%s"
             "appsink name=sink emit-signals=false sync=false "
             "max-buffers=2 drop=true",
             device.c_str(), width, height, fps, scale_stage);
  }

  return std::string(pipeline_string_buffer);
}

// ------------------------------------------------------------
// build_capture_pipeline
// WHY: public entry point — builds the string then launches
// the GStreamer pipeline. Returns nullptr on failure.
// ------------------------------------------------------------
GstElement *build_capture_pipeline(const CameraConfig &camera_config) {
  std::string pipeline_str = build_capture_pipeline_string(camera_config);
  GError *err = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
  if (err) {
    fprintf(stderr, "[PIPELINE] Capture pipeline error: %s\n", err->message);
    g_error_free(err);
    return nullptr;
  }
  return pipeline;
}

// ------------------------------------------------------------
// build_stream_pipeline
// Builds the full streaming pipeline: appsrc → encoding → RTP → UDP
// ------------------------------------------------------------
// Takes StreamConfig and returns a ready GstElement* pipeline
GstElement *build_stream_pipeline(const StreamConfig &stream_config) {
  char pipeline_string_buffer[512];

  // Build the complete pipeline string
  snprintf(
      pipeline_string_buffer, sizeof(pipeline_string_buffer),
      "appsrc name=streamsrc is-live=true format=3 do-timestamp=true ! "
      "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! "
      "videoconvert ! "
      "video/x-raw,format=NV12 ! "
      "nvvidconv ! "

      "video/x-raw(memory:NVMM),format=NV12 ! "
      "queue max-size-buffers=2 leaky=downstream ! "
      "nvv4l2h264enc insert-sps-pps=true iframeinterval=30 idrinterval=30 ! "
      "h264parse config-interval=1 ! "
      "rtph264pay mtu=1400 config-interval=1 pt=96 ! "
      "udpsink host=%s port=%d sync=false",
      stream_config.stream_width_pixels, stream_config.stream_height_pixels,
      stream_config.frames_per_second, stream_config.destination_ip.c_str(),
      stream_config.destination_port);

  // Print a compact summary instead of the whole 300-char gst-launch
  // string. Set TRACKER_VERBOSE=1 to see the raw pipeline for debugging.
  if (getenv("TRACKER_VERBOSE"))
    printf(LOG_DIM "[PIPELINE] Stream: %s" LOG_RESET "\n",
           pipeline_string_buffer);
  else
    printf(LOG_CYAN "[PIPELINE]" LOG_RESET " Stream → " LOG_GREEN
                    "udp://%s:%d" LOG_RESET " (H264 %dx%d @ %dfps)\n",
           stream_config.destination_ip.c_str(),
           stream_config.destination_port,
           stream_config.stream_width_pixels,
           stream_config.stream_height_pixels,
           stream_config.frames_per_second);

  GError *pipeline_error = nullptr;

  // Parse the string and create the actual pipeline
  GstElement *stream_pipeline =
      gst_parse_launch(pipeline_string_buffer, &pipeline_error);

  if (pipeline_error != nullptr) {
    fprintf(stderr, "[PIPELINE] Stream parse error: %s\n",
            pipeline_error->message);
    g_error_free(pipeline_error); // Clean up error object
    return nullptr;
  }

  // Success - return the streaming pipeline
  return stream_pipeline;
}

// ------------------------------------------------------------
// get_pipeline_element
// Helper function to extract a named element from a pipeline bin.
// ------------------------------------------------------------
// Returns nullptr if the element is not found (with error message)
GstElement *get_pipeline_element(GstElement *pipeline,
                                 const std::string &element_name) {
  // Search for the element by name inside the pipeline
  GstElement *found_element =
      gst_bin_get_by_name(GST_BIN(pipeline), element_name.c_str());

  // Optional but very useful: warn the user if the element doesn't exist
  if (found_element == nullptr) {
    fprintf(stderr, "[PIPELINE] Element '%s' not found in pipeline.\n",
            element_name.c_str());
  }

  // Return the found element (caller must unref it when done)
  return found_element;
}