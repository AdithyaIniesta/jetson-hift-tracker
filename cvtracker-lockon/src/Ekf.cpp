#include "Ekf.h"
#include <cmath>

namespace cr {
namespace vtracker {

namespace {

constexpr float OMEGA_EPS = 1e-4f;   // below this, use straight-line limit
constexpr float INIT_POS_VAR    = 4.0f;        // (pixels)^2
constexpr float INIT_SPEED_VAR  = 25.0f;       // (px/frame)^2
constexpr float INIT_ANGLE_VAR  = 9.8696f;     // pi^2 (full ignorance)
constexpr float INIT_OMEGA_VAR  = 0.25f;       // (rad/frame)^2

// Mahalanobis gate for measurement outlier rejection. The squared
// Mahalanobis distance of the innovation follows a chi-square distribution
// with 2 DOF (x,y). 9.21 = 99% confidence, 13.8 = 99.9%. A measurement
// beyond the gate is physically implausible given the motion model (e.g.
// the correlation peak slid along a straight edge) - instead of accepting
// the jump, we damp the correction so the filter trusts its prediction.
// Raise the gate to accept more aggressive maneuvers; lower it for stricter
// physics enforcement (more edge-drift rejection).
constexpr float MAHALANOBIS_GATE = 9.21f;
// Cap on how hard a gated (outlier) measurement is damped: R is inflated by
// up to this factor. Prevents a single bad frame from yanking the state
// while still letting sustained real motion pull the filter over several
// frames.
constexpr float GATE_MAX_INFLATION = 50.0f;

inline float& M(std::array<float, 25>& A, int r, int c) { return A[r * 5 + c]; }
inline float  M(const std::array<float, 25>& A, int r, int c) { return A[r * 5 + c]; }

/// Wrap angle to (-pi, pi].
inline float wrapAngle(float a)
{
    constexpr float pi = 3.14159265358979323846f;
    while (a >  pi) a -= 2.0f * pi;
    while (a <= -pi) a += 2.0f * pi;
    return a;
}

} // namespace

void Ekf::reset()
{
    m_initialized = false;
    m_x.fill(0.0f);
    m_P.fill(0.0f);
}

void Ekf::initialize(float px, float py)
{
    m_x = {px, py, 0.0f, 0.0f, 0.0f};
    m_P.fill(0.0f);
    M(m_P, 0, 0) = INIT_POS_VAR;
    M(m_P, 1, 1) = INIT_POS_VAR;
    M(m_P, 2, 2) = INIT_SPEED_VAR;
    M(m_P, 3, 3) = INIT_ANGLE_VAR;
    M(m_P, 4, 4) = INIT_OMEGA_VAR;
    m_initialized = true;
}

float Ekf::vx() const { return m_x[2] * std::cos(m_x[3]); }
float Ekf::vy() const { return m_x[2] * std::sin(m_x[3]); }

void Ekf::translate(float dx, float dy)
{
    if (!m_initialized)
        return;
    m_x[0] += dx;
    m_x[1] += dy;
}

void Ekf::predict(float dt, float sigma_a, float sigma_alpha)
{
    if (!m_initialized || dt <= 0.0f)
        return;

    const float px    = m_x[0];
    const float py    = m_x[1];
    const float v     = m_x[2];
    const float psi   = m_x[3];
    const float omega = m_x[4];

    const float psi_n = psi + omega * dt;
    const float c_psi = std::cos(psi);
    const float s_psi = std::sin(psi);
    const float c_psn = std::cos(psi_n);
    const float s_psn = std::sin(psi_n);

    // ---- state propagation (CTRV) ------------------------------------------
    std::array<float, 5> x_pred{};
    if (std::fabs(omega) > OMEGA_EPS)
    {
        const float vw = v / omega;
        x_pred[0] = px + vw * (s_psn - s_psi);
        x_pred[1] = py + vw * (-c_psn + c_psi);
    }
    else
    {
        x_pred[0] = px + v * c_psi * dt;
        x_pred[1] = py + v * s_psi * dt;
    }
    x_pred[2] = v;
    x_pred[3] = wrapAngle(psi_n);
    x_pred[4] = omega;

    // ---- Jacobian F = df/dx (5x5) ------------------------------------------
    std::array<float, 25> F{};
    // identity diagonal
    M(F, 0, 0) = 1.0f;
    M(F, 1, 1) = 1.0f;
    M(F, 2, 2) = 1.0f;
    M(F, 3, 3) = 1.0f;
    M(F, 4, 4) = 1.0f;
    M(F, 3, 4) = dt;

    if (std::fabs(omega) > OMEGA_EPS)
    {
        const float inv_w = 1.0f / omega;
        const float inv_w2 = inv_w * inv_w;
        // d px / d v, psi, omega
        M(F, 0, 2) = (s_psn - s_psi) * inv_w;
        M(F, 0, 3) = v * inv_w * (c_psn - c_psi);
        M(F, 0, 4) = -v * inv_w2 * (s_psn - s_psi) + v * dt * inv_w * c_psn;
        // d py / d v, psi, omega
        M(F, 1, 2) = (c_psi - c_psn) * inv_w;
        M(F, 1, 3) = v * inv_w * (s_psn - s_psi);
        M(F, 1, 4) = v * inv_w2 * (c_psn - c_psi) + v * dt * inv_w * s_psn;
    }
    else
    {
        // Straight-line limit (Taylor at omega = 0).
        M(F, 0, 2) = c_psi * dt;
        M(F, 0, 3) = -v * s_psi * dt;
        M(F, 0, 4) = -0.5f * v * s_psi * dt * dt;
        M(F, 1, 2) = s_psi * dt;
        M(F, 1, 3) = v * c_psi * dt;
        M(F, 1, 4) = 0.5f * v * c_psi * dt * dt;
    }

    // ---- P' = F P F^T + Q --------------------------------------------------
    // FP = F * P  (5x5)
    std::array<float, 25> FP{};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 5; ++k)
                s += M(F, i, k) * M(m_P, k, j);
            M(FP, i, j) = s;
        }
    // P' = FP * F^T
    std::array<float, 25> P_pred{};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 5; ++k)
                s += M(FP, i, k) * M(F, j, k);  // F^T at (k,j) = F at (j,k)
            M(P_pred, i, j) = s;
        }

    // Process noise Q = G * diag(sigma_a^2, sigma_alpha^2) * G^T, where
    // G (5x2) maps [linear_accel, angular_accel] to state increments over dt:
    //   px increment ~ 0.5 dt^2 cos(psi) * a
    //   py increment ~ 0.5 dt^2 sin(psi) * a
    //   v  increment ~ dt * a
    //   psi increment ~ 0.5 dt^2 * alpha
    //   omega increment ~ dt * alpha
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    const float va = sigma_a * sigma_a;
    const float vw_n = sigma_alpha * sigma_alpha;
    const float cc = c_psi * c_psi;
    const float ss = s_psi * s_psi;
    const float cs = c_psi * s_psi;

    // Additive Q (only nonzero entries listed).
    M(P_pred, 0, 0) += 0.25f * dt4 * cc * va;
    M(P_pred, 0, 1) += 0.25f * dt4 * cs * va;
    M(P_pred, 0, 2) += 0.5f  * dt3 * c_psi * va;
    M(P_pred, 1, 0) += 0.25f * dt4 * cs * va;
    M(P_pred, 1, 1) += 0.25f * dt4 * ss * va;
    M(P_pred, 1, 2) += 0.5f  * dt3 * s_psi * va;
    M(P_pred, 2, 0) += 0.5f  * dt3 * c_psi * va;
    M(P_pred, 2, 1) += 0.5f  * dt3 * s_psi * va;
    M(P_pred, 2, 2) +=        dt2 * va;
    M(P_pred, 3, 3) += 0.25f * dt4 * vw_n;
    M(P_pred, 3, 4) += 0.5f  * dt3 * vw_n;
    M(P_pred, 4, 3) += 0.5f  * dt3 * vw_n;
    M(P_pred, 4, 4) +=        dt2 * vw_n;

    m_x = x_pred;
    m_P = P_pred;
}

void Ekf::update(float zx, float zy, float prob, float r_base)
{
    if (!m_initialized)
    {
        initialize(zx, zy);
        return;
    }

    // Confidence-weighted measurement variance: weak detections trust the
    // prediction more. Floor prob at 0.05 to avoid division blowup.
    const float p = prob < 0.05f ? 0.05f : prob;
    float r = r_base / p;

    // Innovation y = z - H x  (H selects px, py)
    const float yx = zx - m_x[0];
    const float yy = zy - m_x[1];

    // Innovation covariance S = H P H^T + R with the nominal R, used for the
    // physics (Mahalanobis) gate below.
    {
        const float S00 = M(m_P, 0, 0) + r;
        const float S01 = M(m_P, 0, 1);
        const float S10 = M(m_P, 1, 0);
        const float S11 = M(m_P, 1, 1) + r;
        const float det = S00 * S11 - S01 * S10;
        if (std::fabs(det) > 1e-12f)
        {
            const float invDet = 1.0f / det;
            const float d2 = yx * (S11 * invDet * yx - S01 * invDet * yy) +
                             yy * (-S10 * invDet * yx + S00 * invDet * yy);
            if (d2 > MAHALANOBIS_GATE)
            {
                // Outlier: the measured jump conflicts with the motion model
                // (e.g. correlation slid along a straight edge). Inflate R so
                // the correction is heavily damped - the filter coasts on its
                // physics prediction rather than chasing the bad measurement.
                float infl = d2 / MAHALANOBIS_GATE;
                if (infl > GATE_MAX_INFLATION)
                    infl = GATE_MAX_INFLATION;
                r *= infl;
            }
        }
    }

    // S = H P H^T + R  (2x2), recomputed with the (possibly inflated) R.
    const float S00 = M(m_P, 0, 0) + r;
    const float S01 = M(m_P, 0, 1);
    const float S10 = M(m_P, 1, 0);
    const float S11 = M(m_P, 1, 1) + r;
    const float det = S00 * S11 - S01 * S10;
    if (std::fabs(det) < 1e-12f)
        return;  // singular - skip this update
    const float invDet = 1.0f / det;
    const float Si00 =  S11 * invDet;
    const float Si01 = -S01 * invDet;
    const float Si10 = -S10 * invDet;
    const float Si11 =  S00 * invDet;

    // K = P H^T S^-1  (5x2). P H^T = first two columns of P.
    std::array<float, 10> K{}; // row-major 5x2
    for (int i = 0; i < 5; ++i)
    {
        const float ph0 = M(m_P, i, 0);
        const float ph1 = M(m_P, i, 1);
        K[i * 2 + 0] = ph0 * Si00 + ph1 * Si10;
        K[i * 2 + 1] = ph0 * Si01 + ph1 * Si11;
    }

    // x = x + K y
    for (int i = 0; i < 5; ++i)
        m_x[i] += K[i * 2 + 0] * yx + K[i * 2 + 1] * yy;
    m_x[3] = wrapAngle(m_x[3]);

    // P = (I - K H) P. K H is 5x5 but only columns 0,1 are nonzero, so
    //   (K H)[i][j] = K[i][j]   if j in {0,1} else 0.
    // (I - K H) P = P - K H P. (K H P)[i][j] = K[i][0]*P[0][j] + K[i][1]*P[1][j].
    std::array<float, 25> P_new{};
    for (int i = 0; i < 5; ++i)
    {
        const float k0 = K[i * 2 + 0];
        const float k1 = K[i * 2 + 1];
        for (int j = 0; j < 5; ++j)
        {
            M(P_new, i, j) =
                M(m_P, i, j) - (k0 * M(m_P, 0, j) + k1 * M(m_P, 1, j));
        }
    }
    // Symmetrize to suppress numerical drift.
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
        {
            const float a = 0.5f * (M(P_new, i, j) + M(P_new, j, i));
            M(P_new, i, j) = a;
            M(P_new, j, i) = a;
        }
    m_P = P_new;
}

} // namespace vtracker
} // namespace cr
