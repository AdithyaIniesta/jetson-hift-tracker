// ============================================================
// tracker_state.cpp
// Implementation of the active-camera state machine.
// All state transitions go through these functions so that
// the logic is in one place and easy to reason about.
// ============================================================

#include "tracker_state.h"

#include <cstdio>

// ------------------------------------------------------------
// g_active_tracking_camera
// WHY: initialised to NONE so that on startup no camera is
// tracking — operator must send an explicit CAPTURE command
// to designate which camera owns the tracker.
// ------------------------------------------------------------
std::atomic<ActiveCamera> g_active_tracking_camera{ActiveCamera::NONE};

// ------------------------------------------------------------
void activate_left_camera() {
  g_active_tracking_camera.store(ActiveCamera::LEFT);
  printf("[STATE] Active camera → LEFT\n");
}

void activate_right_camera() {
  g_active_tracking_camera.store(ActiveCamera::RIGHT);
  printf("[STATE] Active camera → RIGHT\n");
}

// ------------------------------------------------------------
// Function: Deactivates all cameras and puts the tracking system into idle
// state
void deactivate_all_cameras() {
  // WHY: On RESET command we want to immediately stop tracking
  // This ensures the trackerThread sees NONE on its next loop
  // and goes idle until the next CAPTURE command arrives
  g_active_tracking_camera.store(ActiveCamera::NONE);

  // Print status message for logging and debugging
  // Clearly indicates the system is now in idle mode
  printf("[STATE] Active camera → NONE (idle)\n");
}

// ------------------------------------------------------------
// Function: Checks whether a specific camera is currently the active tracking
// camera
bool is_camera_active(ActiveCamera camera_to_check) {
  // Load the current value of the atomic variable (thread-safe read)
  // and compare it with the camera we want to check
  return g_active_tracking_camera.load() == camera_to_check;
}
