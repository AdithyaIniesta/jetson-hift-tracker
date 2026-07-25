#pragma once
#include "common/globals.h"

void captureThread(GstElement *sink, std::array<CaptureSlot, RING_SIZE> &ring,
                   std::atomic<int> &writeIdx, std::mutex &ringMtx,
                   std::condition_variable &ringCv, int fps, bool recEnabled,
                   GstElement *recordPipe, int W, int H,
                   bool downsample_enabled);