#pragma once
#include "common/globals.h"

// ── Recording helpers ────────────────────────────────────────
std::string makeRecordingSessionName();
std::string makeRecordingPipelineDesc(const std::string &outputFile, int W,
                                      int H, int fps);

// ── rawStreamWorker ──────────────────────────────────────────
// One instance per camera. Each worker owns one ring buffer and
// one appsrc exclusively — no starvation between cameras.
// Skips push when outputThread owns the src during TRACKING.
void rawStreamWorker(DualRingBuffer &ring, GstElement *src, int camera_id,
                     int W, int H);

// ── outputThread (per camera) ────────────────────────────────
// One instance per camera. Drains its own tracker's result queue,
// draws overlay, pushes the annotated frame to that camera's RTP
// src, sends telemetry, drives the auto-handoff check against the
// OTHER camera's tracker.
//
// camera_id     : 1 = LEFT, 2 = RIGHT
// self_tracker  : g_tracker_L or g_tracker_R matching this camera
// self_mtx      : matching per-camera tracker mutex
// peer_tracker  : the OTHER camera's tracker instance (for handoff)
// peer_mtx      : matching per-camera tracker mutex
// stream_src    : this camera's H264 RTP appsrc (srcL or srcR)
// annotated_src : this camera's annotated recording appsrc (may be
//                 null if this camera isn't the recording target)
void outputThread(int camera_id,
                  cr::vtracker::AppTracker &self_tracker,
                  std::mutex &self_mtx,
                  cr::vtracker::AppTracker &peer_tracker,
                  std::mutex &peer_mtx,
                  std::vector<ResultSlot> &resultQueue,
                  std::mutex &resultMtx,
                  std::condition_variable &resultCv,
                  GstElement *stream_src,
                  GstElement *annotated_src,
                  int W, int H, int fps, bool recEnabled);
