// ============================================================
// dual_capture.cpp
// Implementation of dual camera capture threads.
// Each thread pulls frames from its own GstAppSink and writes
// into its own DualRingBuffer independently.
// WHY: two independent capture threads means a slow tracker
// never causes either camera to drop frames — both cameras
// always pull at full framerate regardless of tracker load.
// ============================================================

#include "dual_capture.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <cstdio>
#include <cstring>

#include "tracker_state.h"

// ------------------------------------------------------------
// g_running is defined in main.cpp — extern reference so
// both capture threads respect the global shutdown signal.
// ------------------------------------------------------------
extern std::atomic<bool> g_running;

// ------------------------------------------------------------
// compute_frame_timestamp
// WHY: when recording is active we use the pipeline clock so
// MKV timestamps are accurate. When recording is disabled we
// fall back to frame-count arithmetic which is simpler but
// accumulates drift over long sessions.
// ------------------------------------------------------------
// Returns a relative timestamp (starting from 0) for the current frame
static GstClockTime
compute_frame_timestamp(int local_frame_index, int frames_per_second,
                        bool recording_enabled, GstElement *recording_pipeline,
                        GstClockTime &first_frame_timestamp_ns) {
  GstClockTime current_timestamp_ns;

  if (recording_enabled && recording_pipeline != nullptr) {
    // WHY: pipeline clock gives wall-clock accurate timestamps
    // for MKV muxer — frame counting drifts at high framerates.
    GstClock *pipeline_clock = gst_element_get_clock(recording_pipeline);

    GstClockTime pipeline_base_time =
        gst_element_get_base_time(recording_pipeline);

    if (pipeline_clock != nullptr) {
      // Get current time from the pipeline's clock and subtract base time
      // to get a monotonic timestamp aligned with the recording pipeline
      current_timestamp_ns =
          gst_clock_get_time(pipeline_clock) - pipeline_base_time;

      // Always unref GStreamer objects to prevent memory leaks
      gst_object_unref(pipeline_clock);
    } else {
      // Fallback if clock is not yet available (rare early case)
      current_timestamp_ns =
          (GstClockTime)local_frame_index *
          gst_util_uint64_scale_int(1, GST_SECOND, frames_per_second);
    }
  } else {
    // WHY: no recording pipeline — use simple frame counting
    // This is lighter but can slowly drift over very long sessions
    current_timestamp_ns =
        (GstClockTime)local_frame_index *
        gst_util_uint64_scale_int(1, GST_SECOND, frames_per_second);
  }

  // Anchor timestamp to zero at the first frame of this session
  // This makes all timestamps relative (0, 16.6ms, 33.3ms, ...)
  if (first_frame_timestamp_ns == GST_CLOCK_TIME_NONE) {
    first_frame_timestamp_ns = current_timestamp_ns;
  }

  // Return timestamp relative to the first frame
  return current_timestamp_ns - first_frame_timestamp_ns;
}

// ------------------------------------------------------------
// run_capture_loop
// Shared implementation used by both capture_thread_left and
// capture_thread_right. The camera_label is used for logging.
// WHY: avoids duplicating ~80 lines of identical logic for
// each camera — only the ring buffer and label differ.
// ------------------------------------------------------------
// Main capture loop: pulls frames from GStreamer appsink → timestamps them
// → writes into ring buffer → signals tracker thread.
static void run_capture_loop(const char *camera_label,
                             GstElement *camera_appsink,
                             DualRingBuffer &ring_buffer,
                             std::atomic<int> &global_frame_id,
                             int frames_per_second, bool recording_enabled,
                             GstElement *recording_pipeline,
                             GstElement *record_src) {
  int local_frame_index = 0;
  GstClockTime first_frame_timestamp_ns = GST_CLOCK_TIME_NONE;

  // Pull timeout = 2 frame periods.
  // WHY: short enough to check g_running frequently,
  // long enough to avoid busy-waiting between frames.
  const GstClockTime pull_timeout_ns = 2 * GST_SECOND / frames_per_second;

  printf("[CAPTURE-%s] Thread started.\n", camera_label);

  while (g_running) {
    // Try to pull a frame from the appsink with timeout
    GstSample *incoming_sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(camera_appsink), pull_timeout_ns);

    if (incoming_sample == nullptr) {
      // WHY: timeout expired — no frame arrived. Check
      // g_running and loop again. This is the clean
      // shutdown polling point for this thread.
      if (!g_running)
        break;
      continue;
    }

    GstBuffer *frame_buffer = gst_sample_get_buffer(incoming_sample);

    GstMapInfo frame_map;

    if (!gst_buffer_map(frame_buffer, &frame_map, GST_MAP_READ)) {
      // WHY: map failure is rare but must be handled —
      // unref the sample to avoid GStreamer memory leak.
      gst_sample_unref(incoming_sample);
      continue;
    }

    // Compute accurate timestamp (pipeline clock when recording)
    GstClockTime frame_timestamp_ns = compute_frame_timestamp(
        local_frame_index, frames_per_second, recording_enabled,
        recording_pipeline, first_frame_timestamp_ns);

    // --------------------------------------------------
    // Write frame into the next ring buffer slot.
    // WHY: if the tracker is slow and has not read the
    // previous slot, we overwrite it — one stale frame
    // lost is better than blocking the camera thread.
    // --------------------------------------------------
    {
      std::lock_guard<std::mutex> ring_lock(ring_buffer.ring_mutex);

      int slot_index = ring_buffer.write_index.load() % DUAL_RING_SIZE;
      DualCaptureSlot &target_slot = ring_buffer.slots[slot_index];

      // Copy raw BGR pixel data
      target_slot.bgr_frame_data.assign(frame_map.data,
                                        frame_map.data + frame_map.size);

      target_slot.frame_sequence_id = global_frame_id++;
      target_slot.pipeline_timestamp_ns = frame_timestamp_ns;
      target_slot.contains_valid_frame = true;

      ring_buffer.write_index++;
    }

    // Wake up trackerThread
    ring_buffer.ring_condition.notify_one();

    // ─── Per-camera raw record push ─────────────────────────────
    // Independent of tracker/streaming — records every frame from
    // boot so replay has continuous video for both cameras (not
    // just the one the operator eventually locked).
    if (record_src != nullptr) {
      GstBuffer *rec_buf = gst_buffer_new_allocate(nullptr, frame_map.size,
                                                   nullptr);
      GstMapInfo rec_map;
      gst_buffer_map(rec_buf, &rec_map, GST_MAP_WRITE);
      memcpy(rec_map.data, frame_map.data, frame_map.size);
      gst_buffer_unmap(rec_buf, &rec_map);
      GST_BUFFER_PTS(rec_buf) = frame_timestamp_ns;
      GST_BUFFER_DURATION(rec_buf) = gst_util_uint64_scale_int(
          1, GST_SECOND, frames_per_second);
      GstFlowReturn ret =
          gst_app_src_push_buffer(GST_APP_SRC(record_src), rec_buf);
      if (ret != GST_FLOW_OK) {
        fprintf(stderr, "[REC-%s] push failed: %d\n", camera_label, (int)ret);
      }
    }

    // Clean up resources
    gst_buffer_unmap(frame_buffer, &frame_map);
    gst_sample_unref(incoming_sample);

    local_frame_index++;
  }

  printf("[CAPTURE-%s] Thread exiting.\n", camera_label);
}

// ------------------------------------------------------------
// Public entry points — thin wrappers around run_capture_loop
// so that main.cpp can launch them as std::thread targets.
// ------------------------------------------------------------

// Thread function for the Left camera
// Simply forwards to the shared implementation with camera label "L"
void capture_thread_left(GstElement *camera_appsink,
                         DualRingBuffer &ring_buffer,
                         std::atomic<int> &global_frame_id,
                         int frames_per_second, bool recording_enabled,
                         GstElement *recording_pipeline,
                         GstElement *record_src) {
  run_capture_loop("L", camera_appsink, ring_buffer, global_frame_id,
                   frames_per_second, recording_enabled, recording_pipeline,
                   record_src);
}

// Thread function for the Right camera
// Simply forwards to the shared implementation with camera label "R"
void capture_thread_right(GstElement *camera_appsink,
                          DualRingBuffer &ring_buffer,
                          std::atomic<int> &global_frame_id,
                          int frames_per_second, bool recording_enabled,
                          GstElement *recording_pipeline,
                          GstElement *record_src) {
  run_capture_loop("R", camera_appsink, ring_buffer, global_frame_id,
                   frames_per_second, recording_enabled, recording_pipeline,
                   record_src);
}