#pragma once
#include "common/globals.h"

// Pull submodule header paths cleanly so everything including this knows
// VTrackerParam
#ifdef TRACKER_V2
#include <cvtracker/VTracker.h>
#else
#include "VTracker.h"
#endif

// ── trackerThread (per camera) ────────────────────────────────
// One instance per camera. Each thread reads exclusively from its
// own ring buffer and drives its own tracker instance. There is no
// g_selected_camera gating on the frame path — both cameras always
// track when frames arrive.
//
// Dual-lock: both trackers process their camera continuously. When
// one is FREE/LOST and the other is TRACKING with high confidence,
// streaming.cpp's outputThread projects the tracked position through
// the handoff homography and fires CAPTURE on the LOST one. Neither
// tracker "gives up" while the other has a good lock.
//
// camera_id : 1 = LEFT (boresight), 2 = RIGHT (depression)
// tracker   : the g_tracker_L or g_tracker_R instance to drive
// tracker_mtx : matching g_mtx_L / g_mtx_R
// ring      : the ring buffer for THIS camera (left_ring or right_ring)
// resultQueue etc. : per-camera output pipeline
void trackerThread(int camera_id,
                   cr::vtracker::CvTracker &tracker,
                   std::mutex &tracker_mtx,
                   DualRingBuffer &ring,
                   std::vector<ResultSlot> &resultQueue,
                   std::mutex &resultMtx,
                   std::condition_variable &resultCv,
                   int W, int H, int fps,
                   float degPerPixelX, float degPerPixelY,
                   float centerX, float centerY);
