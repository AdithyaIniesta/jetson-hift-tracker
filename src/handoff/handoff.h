/**
 * @file handoff.h
 * @brief Cross-view target handoff between boresight and depression cameras.
 *
 * Given stereo calibration (K_L, K_R, R, t) and a per-frame target depth
 * (constant indoor / barometer AGL on the airframe), this module computes
 * the pixel in one camera that corresponds to the tracked target pixel in
 * the other camera. Enables:
 *
 *   1. Symmetric handoff — target visible in either camera can be handed
 *      to the other automatically when it approaches the shared-FOV
 *      region, or when the receiving camera is currently LOST.
 *
 *   2. Dual-lock sanity check — both cameras track simultaneously; the
 *      module cross-projects each camera's rect to the other and flags
 *      drift or disagreement between them.
 *
 * Depth model: single ground plane at `plane_depth_mm` in the boresight
 * camera's optical-axis Z. Indoor: fixed value from run script prompt.
 * Airframe: replace with per-frame barometer AGL (separate change).
 *
 * Both directions are precomputed once at boot as 3x3 homographies:
 *   H_L2R : (u_L, v_L) → (u_R, v_R)
 *   H_R2L : (u_R, v_R) → (u_L, v_L)
 *
 * A single float multiply-add per pixel per direction — negligible cost
 * even at 60 fps on both cameras.
 */

#pragma once

#include <array>
#include <string>

// Ideally we'd take cv::Mat directly, but we don't want handoff.h to drag
// OpenCV into every include chain. Use POD arrays and let the .cpp convert.

namespace handoff {

struct Intrinsics {
    // 3x3 K matrix in row-major order:
    //   fx  0 cx
    //    0 fy cy
    //    0  0  1
    std::array<double, 9> K{};
    // 5-element distortion coefficient vector (k1, k2, p1, p2, k3).
    std::array<double, 5> dist{};
};

struct Extrinsics {
    // 3x3 rotation from LEFT to RIGHT camera frame, row-major.
    std::array<double, 9> R{};
    // 3-vector translation LEFT->RIGHT in millimetres.
    std::array<double, 3> t_mm{};
};

struct StereoCalib {
    Intrinsics left;
    Intrinsics right;
    Extrinsics ext;
    // Image size the calibration was solved at (needed to sanity-check
    // whether the loaded K matches the current tracker operating res).
    int img_w = 0;
    int img_h = 0;
};

/**
 * Load `stereo_calib.json` written by the Windows stereo calibrator.
 * Accepts the placeholder file too (K_L / K_R / R / t_mm fields).
 * Returns false if the file is missing or malformed; `out` is unchanged
 * on failure so the caller can decide to error or fall back to defaults.
 */
bool loadStereoCalib(const std::string& path, StereoCalib& out);

/**
 * Clip the infinite epipolar line a*x + b*y + c = 0 to the image rectangle
 * [0,w-1] x [0,h-1], returning the two boundary intersection points
 * (x0,y0)-(x1,y1). Used by DISTANCE-FREE handoff to enumerate seed
 * candidates along the line. Returns false if the line misses the image.
 */
bool clipLineToImage(double a, double b, double c, int w, int h,
                     double& x0, double& y0, double& x1, double& y1);

/**
 * Cross-view handoff computer. Constructed once per session with the
 * loaded calibration and current target depth. Both homographies are
 * precomputed at construction; per-pixel projection is a 3x3 matmul
 * plus a homogeneous divide.
 *
 * Thread-safety: after construction the object is read-only — any thread
 * may call projectLtoR / projectRtoL concurrently.
 */
class HandoffModel {
public:
    HandoffModel();
    ~HandoffModel();

    /**
     * Initialise with a loaded stereo calibration and the current
     * target depth. `plane_depth_mm` is the distance from the boresight
     * camera's optical origin to the ground plane along the camera's
     * forward axis (indoor: measured; airframe: barometer AGL).
     *
     * The calibration must be captured at the same resolution the
     * tracker operates at. If the tracker runs downsampled (e.g.
     * 640x480), load a stereo calibration produced from a session at
     * that resolution — main.cpp picks the file by tracker size.
     *
     * Returns false if the calibration is inconsistent (zero focal
     * length, degenerate rotation, etc.).
     */
    bool initialise(const StereoCalib& calib, double plane_depth_mm);

    /** True after successful initialise() (epipolar geometry available). */
    bool ready() const;

    /**
     * True only when a positive depth was given, so the plane-induced
     * homography (projectLtoR/RtoL) is usable. False in DISTANCE-FREE mode
     * (depth <= 0): epipolar line still works, homography does not.
     */
    bool homographyReady() const;

    /** The depth passed to initialise(), for logging / re-init. */
    double planeDepthMm() const;

    /**
     * Project a pixel from LEFT (boresight) to RIGHT (depression).
     * Inputs and outputs are in the same coord space the tracker uses
     * for CMD_CAPTURE (post-downsample, so 640x480 by default).
     *
     * `out_u/v` are written only on success. Returns false if the
     * projected point lands behind either camera (rare but possible
     * for extreme geometries) or if the model isn't initialised.
     */
    bool projectLtoR(double in_u, double in_v,
                     double& out_u, double& out_v) const;

    /** Reverse direction. */
    bool projectRtoL(double in_u, double in_v,
                     double& out_u, double& out_v) const;

    /**
     * Update depth without full re-initialise. Useful when barometer
     * reading changes on the airframe. Recomputes both homographies.
     */
    void setDepth(double plane_depth_mm);

    /**
     * Focal-length accessors for the loaded intrinsics. Used by the
     * shadow-projection overlay to scale a rect from one camera into
     * the pixel size it would occupy in the other view at the same
     * physical depth.
     */
    double fxL() const;
    double fyL() const;
    double fxR() const;
    double fyR() const;

    /**
     * Epipolar line in the RIGHT (peer) image for a point observed in
     * the LEFT image. Line coefficients (a, b, c) satisfy
     * a·u_R + b·v_R + c = 0. Sign convention normalised so that
     * sqrt(a² + b²) = 1 — the perpendicular distance from any pixel
     * (u, v) to the line is simply |a·u + b·v + c|.
     *
     * Uses the fundamental matrix F derived from the loaded stereo
     * calibration: F = K_R^{-T} [t]_x R K_L^{-1}.
     * The plane-depth assumption used by the homography does NOT apply
     * here — the epipolar constraint is exact for a rigid stereo pair.
     *
     * Returns false if the calibration isn't loaded or the projected
     * line is degenerate (both a and b zero).
     */
    bool epipolarLineInR(double in_u_L, double in_v_L,
                         double& a, double& b, double& c) const;

    /** Reverse direction: line in LEFT image for a point in RIGHT. */
    bool epipolarLineInL(double in_u_R, double in_v_R,
                         double& a, double& b, double& c) const;

private:
    struct Impl;
    Impl* d_;
};

}  // namespace handoff
