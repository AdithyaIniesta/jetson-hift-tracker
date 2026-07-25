#pragma once

#include <atomic>

// ============================================================
// tracker_state.h
// Defines the active camera selection state shared across
// captureThread, trackerThread, and outputThread.
//
// WHY: a single atomic variable drives all dual-camera
// behaviour — which ring buffer trackerThread reads from,
// which stream gets the overlay, which angle goes to UART.
// Using std::atomic avoids a mutex for this hot-path read.
// ============================================================

// ------------------------------------------------------------
// ActiveCamera — which camera is currently tracking
// ------------------------------------------------------------
// NONE  : system idle, no tracker running, waiting for CAPTURE
// LEFT  : Camera L is tracking, Camera R streams only
// RIGHT : Camera R is tracking, Camera L streams only
enum class ActiveCamera { NONE = 0, LEFT = 1, RIGHT = 2 };

// ------------------------------------------------------------
// g_active_tracking_camera
// WHY: atomic so captureThread, trackerThread, outputThread,
// and controlThread can all read/write without a mutex.
// Only controlThread writes — all others read.
// ------------------------------------------------------------
extern std::atomic<ActiveCamera> g_active_tracking_camera;

// ------------------------------------------------------------
// State transition functions — called from controlThread
// when a CAPTURE or RESET command arrives.
// ------------------------------------------------------------

// WHY: called when CAPTURE arrives with camera_id = 0 (LEFT).
// Resets tracker and switches active camera to LEFT.
void activate_left_camera();

// WHY: called when CAPTURE arrives with camera_id = 1 (RIGHT).
// Resets tracker and switches active camera to RIGHT.
void activate_right_camera();

// WHY: called on RESET command — stops tracking on both cameras.
void deactivate_all_cameras();

// Returns true if the given camera is currently the active tracker.
bool is_camera_active(ActiveCamera camera_to_check);
