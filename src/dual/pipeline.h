#pragma once
// ============================================================
// pipeline.h
// Declares GStreamer pipeline builder functions for dual
// camera capture and stream pipelines.
// WHY: separating pipeline string construction from main.cpp
// keeps main.cpp focused on thread orchestration and makes
// pipeline strings testable in isolation.
// ============================================================

#include <gst/gst.h>

#include <string>

// ------------------------------------------------------------
// CameraConfig
// All parameters needed to build one camera's capture pipeline.
// WHY: grouping into a struct avoids long argument lists and
// makes it easy to add new parameters without changing every
// call site.
// ------------------------------------------------------------
struct CameraConfig {
  // V4L2 device path e.g. "/dev/video0"
  std::string video_device_path;

  // Pixel format: "UYVY", "YUYV", or "MJPG"
  std::string pixel_format;

  // Capture resolution in pixels.
  int capture_width_pixels = 1280;
  int capture_height_pixels = 720;

  // Output resolution in pixels — what the appsink actually delivers.
  // WHY: when downsampling is enabled, this differs from
  // capture_width/height_pixels — a videoscale element is inserted
  // to resize frames before they reach the appsink, so the ring
  // buffer, tracker, and output stages all see consistent
  // already-downsampled frame data instead of a label-only flag.
  // Defaults match capture size (no scaling) when left untouched.
  int output_width_pixels = 1280;
  int output_height_pixels = 720;

  // Camera framerate in frames per second.
  int frames_per_second = 60;

  // True if this build targets Jetson Xavier NX.
  // WHY: Xavier NX requires an extra nvvidconv step before
  // videoconvert — Orin NX does not need this.
  bool requires_xavier_nvvidconv = false;
};

// ------------------------------------------------------------
// StreamConfig
// All parameters needed to build one UDP stream pipeline.
// ------------------------------------------------------------
struct StreamConfig {
  // Destination IP address for the RTP/UDP stream.
  std::string destination_ip;

  // Destination UDP port for the RTP/UDP stream.
  int destination_port = 5000;

  // Stream resolution in pixels.
  int stream_width_pixels = 1280;
  int stream_height_pixels = 720;

  // Stream framerate in frames per second.
  int frames_per_second = 60;
};

// ------------------------------------------------------------
// Pipeline builder functions.
// Each returns a fully constructed GstElement* pipeline ready
// to be set to GST_STATE_PLAYING.
// Returns nullptr on failure — caller must handle this.
// ------------------------------------------------------------

// Builds the capture pipeline: v4l2src → (optional converters) → appsink
// Used by capture threads to get raw frames from the camera.
GstElement *build_capture_pipeline(const CameraConfig &camera_config);

// Builds the streaming pipeline: appsrc → videoconvert → x264enc →
// rtph264pay → udpsink for network streaming.
GstElement *build_stream_pipeline(const StreamConfig &stream_config);

// WHY: extract the named element from a pipeline bin.
// Returns nullptr if the element name is not found.
// Useful for getting pointers to 'appsink' or 'appsrc' after pipeline creation.
GstElement *get_pipeline_element(GstElement *pipeline,
                                 const std::string &element_name);