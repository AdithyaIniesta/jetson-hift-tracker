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
- [ ] Phase 0: pretrained HiFT standalone benchmark (needs general_model.pth).
- [ ] Phase 1: ONNX export (tools/export_hift.py scaffold in place).
- [ ] Phase 2/3: TRT engines + C++ HiFTTracker.
