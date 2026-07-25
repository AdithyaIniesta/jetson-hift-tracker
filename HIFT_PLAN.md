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
- [ ] Phase 3 Stage B: `HiFTTracker` (VTracker impl) — get_subwindow crop, generate_anchor
      box decode, penalty/window/argmax, lr smoothing, mode machine.
- [ ] Phase 3 Stage C: integrate — main.cpp select, CMakeLists, build.sh option, handoff
      HiFT-template transfer.

### Ported HiFT constants (hift/experiments/config.yaml → src/hift/TrtHiFT.h)
EXEMPLAR 127, SEARCH 287, OUTPUT 11×11, ANCHOR_STRIDE 16, CONTEXT 0.5,
PENALTY_K 0.08, WINDOW_INFLUENCE 0.42, LR 0.30, w2/w3 1.0, decode_scale 143, anchor_off 63.

### Engine I/O (record for box post-processing)
template: z[1,3,127,127] → zf0[1,384,10,10], zf1[1,384,8,8], zf2[1,256,6,6]
track:    x[1,3,287,287] + zf0..2 → loc[1,4,11,11], cls1[1,2,11,11], cls2[1,1,11,11]
