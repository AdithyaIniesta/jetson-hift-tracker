// ============================================================
// dual_capture.h
// Declares the ring buffer structures and capture thread
// functions for two simultaneous cameras.
// Each camera has its own independent ring buffer so that
// a slow tracker on one camera never blocks the other.
// ============================================================

// ------------------------------------------------------------
// Standard C++ library headers

#include <array> // Fixed-size container (e.g. for camera states or small buffers)
#include <atomic> // Atomic variables for thread-safe flags (like g_active_tracking_camera)
#include <condition_variable> // For thread synchronization and waiting (signaling between threads)
#include <cstdint> // Fixed-width integer types (uint8_t, uint32_t, etc.) for clarity and portability
#include <mutex> // Mutexes for protecting shared data in multi-threaded environment
#include <vector> // Dynamic array (likely used for image buffers, frames, or point lists)

// GStreamer multimedia framework
#include <gst/gst.h>

// ------------------------------------------------------------
// Ring buffer sizing
// WHY: 4 slots gives the tracker up to 4 frame periods of
// slack (~66ms at 60fps) before any frame is overwritten.
// Enough headroom for CUDA FFT spikes on hard frames.
// ------------------------------------------------------------
static constexpr int DUAL_RING_SIZE = 4;

// ------------------------------------------------------------
// DualCaptureSlot
// One slot in the ring buffer — holds a single raw BGR frame
// plus metadata needed by the tracker and recorder.
// WHY: pre-allocated at startup to avoid heap allocation
// in the capture hot path (every 16.6ms at 60fps).
// ------------------------------------------------------------
struct DualCaptureSlot {
  // Raw BGR pixel data for this frame (1280x720x3 bytes typically)
  // Stored as std::vector so it owns its memory
  std::vector<uint8_t> bgr_frame_data;

  // Monotonically increasing frame sequence number
  // WHY: used by outputThread to detect dropped frames
  // and maintain correct ordering
  int frame_sequence_id = 0;

  // Pipeline clock timestamp anchored to first frame (nanoseconds)
  // WHY: used to stamp MKV recording buffers correctly
  // and maintain accurate timing in recordings
  GstClockTime pipeline_timestamp_ns = 0;

  // True when this slot contains valid unread frame data
  // WHY: allows trackerThread to skip slots that were
  // overwritten before it could read them (ring buffer protection)
  bool contains_valid_frame = false;
};

// ------------------------------------------------------------
// DualRingBuffer
// A fixed-size circular buffer connecting one captureThread
// to the shared trackerThread.
// ------------------------------------------------------------
// This is a classic producer-consumer ring buffer (lock-free where possible)
// designed for high-speed frame passing between capture and tracking threads.
struct DualRingBuffer {
  // Fixed array of pre-allocated slots (size defined by DUAL_RING_SIZE)
  // Each slot holds one complete frame + metadata
  std::array<DualCaptureSlot, DUAL_RING_SIZE> slots;

  // Write index — incremented by captureThread only
  // WHY: atomic so trackerThread can safely read it without locking
  // (lock-free progress for the hot path)
  std::atomic<int> write_index{0};

  // Mutex to protect shared state when full synchronization is needed
  // (used together with condition variable)
  std::mutex ring_mutex;

  // Condition variable for signaling between threads
  // Allows trackerThread to efficiently wait when no new frames are available
  std::condition_variable ring_condition;
};

// ------------------------------------------------------------
// Capture thread entry points — one per camera.
// Each runs independently, writing into its own ring buffer.
//
// Parameters:
//   camera_appsink     : GStreamer appsink element for this camera
//   ring_buffer        : this camera's dedicated ring buffer
//   global_frame_id    : shared atomic frame counter (g_frameId)
//   frames_per_second  : camera framerate (used for PTS calculation)
//   recording_enabled  : whether MKV recording is active
//   recording_pipeline : GStreamer recording pipeline (for clock)
// ------------------------------------------------------------
// record_src : per-camera raw record appsrc (or nullptr). When non-null,
//              every captured frame is copied straight into it — recording
//              runs from boot regardless of tracker state, so both cameras
//              end up on disk in dual mode.
void capture_thread_left(GstElement *camera_appsink,
                         DualRingBuffer &ring_buffer,
                         std::atomic<int> &global_frame_id,
                         int frames_per_second, bool recording_enabled,
                         GstElement *recording_pipeline,
                         GstElement *record_src);

void capture_thread_right(GstElement *camera_appsink,
                          DualRingBuffer &ring_buffer,
                          std::atomic<int> &global_frame_id,
                          int frames_per_second, bool recording_enabled,
                          GstElement *recording_pipeline,
                          GstElement *record_src);
