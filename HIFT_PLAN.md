# jetson-hift-tracker

Production fork of **jetson-tracking-perception @ handoff (7170763)** that replaces the
correlation-filter tracker with **HiFT** (Hierarchical Feature Transformer, ICCV 2021 —
`github.com/vision4robotics/HiFT`). Everything around the tracker (capture, streaming,
handoff geometry, EKF, recorder, control, telemetry, angle calc, camera params) is kept.

## What changes vs the base repo
- **Remove:** the correlation core (FFT filter), scale probe, auto-size, DINOv2 verifier.
- **Replace:** a new `HiFTTracker` implementing the existing `VTracker` interface — HiFT
  does localization **and** box regression in one model.
- **Keep:** the whole pipeline wrapper — mode machine (FREE/TRACKING/LOST/INERTIAL/STATIC),
  EKF, handoff (homography + epipolar + bank), recorder, control, UART/UDP telemetry.
- **DINOv2:** optional — dropped for per-frame (HiFT self-verifies); may return only for
  re-detection after loss and cross-camera handoff identity.

## HiFT interface (from pysot/models/model_builder.py)
- `template(z)`: backbone(z=127x127 crop) -> template features `zf`. Run once on CAPTURE.
- `track(x)`:   backbone(x=255x255 search) + grader(xf, zf) -> `loc, cls1, cls2`. Per frame.
- Post-process `loc`/`cls` -> box (see `getcentercuda`).

## Deployment shape (Jetson / TensorRT)
Export two ONNX graphs → two TensorRT engines:
1. **template branch**: z[1,3,127,127] -> zf   (run on CAPTURE)
2. **track branch**:   x[1,3,255,255] + zf -> loc, cls1, cls2   (run per frame)
Box post-processing (getcentercuda) ported to C++.

## Phases + decision gates
- **Phase 0 — HiFT standalone (Python):** run pretrained HiFT on UAV123 + our recordings;
  benchmark **single-cam FPS on the Orin NX**. GATE 1: quality good AND ~90 fps single-cam?
- **Phase 1 — export:** `tools/export_hift.py` → template + track ONNX (static shapes).
- **Phase 2 — Jetson TRT:** build engines, measure **dual-cam FPS**. GATE 2: 45 fps dual?
- **Phase 3 — C++ `HiFTTracker`:** implement behind `VTracker`, reuse mode/EKF/handoff/
  recorder. A/B vs base. GATE 3: beats CF+DINOv2?

## Risks
- TensorRT export of the transformer head (dynamic ops) — the main unknown.
- Dual-cam 45 fps budget (two deep trackers).
- HiFT static template (no update) — may need a template-refresh strategy for long tracks.
- Domain: HiFT trained on aerial data; our targets may need fine-tune (a training task).

## Status
- [x] Fork created (base 7170763), build/ + old git stripped, fresh `main`.
- [x] Phase 0/2 gate: trtexec track branch 147 fps / 6.8 ms (dual-cam ~73 fps > 45). PASS.
- [x] Phase 1: ONNX export (tools/export_hift.py → hift_template.onnx + hift_track.onnx).
- [x] Phase 3 Stage A: `TrtHiFT` engine wrapper (src/hift/TrtHiFT.{h,cpp}) — two engines,
      shared zf GPU buffers, setTemplate()/track(). TRT 8.5 enqueueV2 path.
- [x] Phase 3 Stage B: `HiFTTracker` (VTracker impl, src/hift/HiFTTracker.{h,cpp}) —
      get_subwindow crop (YUV→BGR sampler for all fourccs), generate_anchor box decode,
      scale/ratio penalty + cosine window + argmax, lr smoothing, FREE/TRACKING/LOST mode
      machine, exportTemplate/importTemplate for handoff. Raw BGR CHW 0..255 (no norm).
- [x] Phase 3 Stage C: integrate — `AppTracker` typedef swap in common/globals.h (gated by
      -DTRACKER_HIFT), all shared signatures (main/tracker/streaming/control) on AppTracker,
      HiFTTracker CvTracker-compat shims (setFeatureExtractor no-op, template-bank ->
      HiFT template, epipolar no-op), src/CMakeLists ENABLE_HIFT (TRT/CUDA + defines),
      build.sh option [6]. NEXT: build on Jetson (option 6) + iterate compile errors + GUI test.

## Single-camera stability fixes (2026-07-27)
Audit vs hift_tracker.py: decode/penalty/window/argmax/crop-centering all faithful;
cls1 correctly softmaxed from raw logits (track() returns raw grader output, no double
softmax). Fixed three robustness issues in the mode-machine glue:
- **lr clamped to [0,1]** — cls2 is an unbounded BCE logit, so raw penalty*score*LR could
  exceed 1 and make the box-size update extrapolate (blow-up/jitter).
- **Confidence = bounded fg probability** — LOST detection now uses the softmax foreground
  prob (0..1), not the logit-inflated fused score, so lossThreshold is meaningful.
- **Bilinear crop** — matches cv2.resize the ONNX was traced with (nearest added jitter).
- `TRACKER_HIFT_LOSS` env tunes the threshold without a rebuild.
Root-cause test still pending on Jetson: `TRACKER_HIFT_DUMP=1` to verify the YUV→BGR crop.

## Target-loss handling (2026-07-27) — in-FOV good, out-of-FOV was unstable
HiFT has no "target absent" concept: when the ROI leaves FOV it locks onto clutter with
high cls score (false confidence), never goes LOST, and the box drifts larger. Fixes:
- **PSR peak-quality gate** — confidence = fg · min(1, PSR/psrRef). PSR (peak-to-sidelobe on
  the fg map) is low when the response is diffuse (target gone), so confidence drops and the
  box turns LOST. Tunable: `TRACKER_HIFT_PSR` (higher = stricter, LOST sooner).
- **Freeze on low confidence** — below lossThreshold, hold the box (don't chase the clutter
  peak that caused wander + enlargement); mode machine goes LOST.
- **Hard size clamp** — box constrained to [0.4, 2.5]× the captured size, so the regressor
  can't drift it large regardless.
NOTE: PSR default (psrRef=4.0) may need tuning on-device; the size clamp works regardless.
Still likely wants a DINOv2 re-detection/identity check for robust re-acquire after loss.

## Build & run (Jetson)
1. Export ONNX (once):  python3 tools/export_hift.py --snapshot models/first.pth --config hift/experiments/config.yaml
2. Build:  ./build.sh  → choose [6] hift   (first run builds TRT engines from ONNX; minutes)
3. Run the cuda_library run-script but point it at the hift binary (JetsonTracker_hift_orin),
   optionally TRACKER_VERIFIER=none. Then test capture + handoff from the GUI.

### Ported HiFT constants (hift/experiments/config.yaml → src/hift/TrtHiFT.h)
EXEMPLAR 127, SEARCH 287, OUTPUT 11×11, ANCHOR_STRIDE 16, CONTEXT 0.5,
PENALTY_K 0.08, WINDOW_INFLUENCE 0.42, LR 0.30, w2/w3 1.0, decode_scale 143, anchor_off 63.

### Engine I/O (record for box post-processing)
template: z[1,3,127,127] → zf0[1,384,10,10], zf1[1,384,8,8], zf2[1,256,6,6]
track:    x[1,3,287,287] + zf0..2 → loc[1,4,11,11], cls1[1,2,11,11], cls2[1,1,11,11]
