#include "capture.h"

#include <gst/app/gstappsink.h>

void captureThread(GstElement *sink, std::array<CaptureSlot, RING_SIZE> &ring,
                   std::atomic<int> &writeIdx, std::mutex &ringMtx,
                   std::condition_variable &ringCv, int fps, bool recEnabled,
                   GstElement *recordPipe, int W, int H,
                   bool downsample_enabled) {
  int localFrameId = 0;
  static GstClockTime s_firstPts = GST_CLOCK_TIME_NONE;
  while (g_running) {
    GstSample *sample =
        gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND / fps);
    if (!sample) {
      if (!g_running)
        break;
      continue;
    }
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      continue;
    }
    GstClockTime now = (GstClockTime)localFrameId *
                       gst_util_uint64_scale_int(1, GST_SECOND, fps);
    if (recEnabled && recordPipe) {
      GstClock *clk = gst_element_get_clock(recordPipe);
      GstClockTime base = gst_element_get_base_time(recordPipe);
      if (clk) {
        now = gst_clock_get_time(clk) - base;
        gst_object_unref(clk);
      }
    }
    if (s_firstPts == GST_CLOCK_TIME_NONE)
      s_firstPts = now;
    GstClockTime pts = now - s_firstPts;

    const uint8_t *frame_data_ptr = map.data;
    size_t frame_data_size = map.size;
    cv::Mat downsampled_frame;
    if (downsample_enabled) {
      cv::Mat full_frame(H, W, CV_8UC3, map.data);
      cv::resize(full_frame, downsampled_frame, cv::Size(640, 480), 0, 0,
                 cv::INTER_AREA);
      frame_data_ptr = downsampled_frame.data;
      frame_data_size = downsampled_frame.total() * 3;
    }

    {
      std::lock_guard<std::mutex> lk(ringMtx);
      int idx = writeIdx.load() % RING_SIZE;
      CaptureSlot &slot = ring[idx];
      slot.bgrData.assign(frame_data_ptr, frame_data_ptr + frame_data_size);
      slot.frameId = g_frameId++;
      slot.pts = pts;
      slot.ready = true;
      writeIdx++;
    }
    ringCv.notify_one();
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);
    localFrameId++;
  }
}