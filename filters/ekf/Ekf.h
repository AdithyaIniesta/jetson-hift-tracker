#pragma once
#include <array>

namespace cr {
namespace vtracker {

/// Extended Kalman Filter with a constant-turn-rate-and-velocity (CTRV)
/// motion model. State: x = [px, py, v, psi, omega]^T where (px,py) are
/// position in pixels, v is speed in pixels/frame, psi is heading angle
/// (rad), omega is turn rate (rad/frame). Measurement: z = [px, py] from
/// the correlation tracker (linear in state). Used as an optional output
/// filter on top of CvTracker for smoother position / velocity and better
/// LOST-mode extrapolation through curved motion.
class Ekf
{
public:
    /// (Re)initialize state to a measured position. Velocity, heading, and
    /// turn rate are zeroed; covariance is set to a large initial value.
    void initialize(float px, float py);

    /// Reset to uninitialized state (next initialize/update will reseed).
    void reset();

    /// Has initialize() been called since the last reset?
    bool initialized() const { return m_initialized; }

    /// Run the EKF predict step over dt frames (typically dt = 1).
    /// Updates state mean + covariance using the CTRV transition and its
    /// analytic Jacobian. Process noise is parameterized by sigma_a
    /// (linear accel std-dev, px/frame^2) and sigma_alpha (angular accel
    /// std-dev, rad/frame^2).
    void predict(float dt, float sigma_a, float sigma_alpha);

    /// Run the EKF update step with a position measurement. The
    /// measurement noise is scaled by 1/max(prob, 0.05) so that
    /// low-confidence detections pull the state less.
    void update(float zx, float zy, float prob, float r_base);

    /// Filtered state accessors.
    float px()    const { return m_x[0]; }
    float py()    const { return m_x[1]; }
    float speed() const { return m_x[2]; }
    float psi()   const { return m_x[3]; }
    float omega() const { return m_x[4]; }

    /// Cartesian velocity components (pixels/frame), derived from speed
    /// and heading. Convenient for callers that want vx/vy.
    float vx() const;
    float vy() const;

private:
    bool m_initialized{false};
    std::array<float, 5> m_x{};       // state mean
    std::array<float, 25> m_P{};      // 5x5 covariance, row-major
};

} // namespace vtracker
} // namespace cr
