#!/usr/bin/env python3
"""Export HiFT to two static-shape ONNX graphs for TensorRT deployment.

HiFT is a Siamese tracker (pysot/models/model_builder.py):
  * template(z): backbone(z) -> zf   (run once on CAPTURE)
  * track(x):    backbone(x) + grader(xf, zf) -> loc, cls1, cls2   (per frame)

For the Jetson we export two graphs so the template branch runs once and only the track
branch runs per frame:
  1) template_branch:  z[1,3,EXEMPLAR,EXEMPLAR] -> zf   (list/tensor of template features)
  2) track_branch:     x[1,3,SEARCH,SEARCH] + zf -> loc, cls1, cls2

Box post-processing (getcentercuda) is NOT exported — port it to C++ (HiFTTracker).

Prereqs (run on a machine with the HiFT repo + weights):
  * HiFT repo cloned (default: ../HiFT relative to this fork, or pass --hift-dir)
  * pretrained general_model.pth in HiFT/tools/snapshot/  (from the HiFT README links)
  * pip install torch onnx onnxslim onnxruntime  (matching the Jetson torch if exporting there)

Usage:
  python3 tools/export_hift.py --hift-dir ../HiFT \
      --snapshot ../HiFT/tools/snapshot/general_model.pth \
      --config   ../HiFT/experiments/HiFT/config.yaml

STATUS: scaffold. The two wrapper modules below assume the standard pysot forward; verify
the exact `zf` structure (single tensor vs list) and search/exemplar sizes against the
loaded cfg before trusting the export (marked TODO).
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hift-dir", default="hift",
                    help="path to the vendored HiFT model code (default: ./hift)")
    ap.add_argument("--snapshot", required=True, help="general_model.pth")
    ap.add_argument("--config", required=True, help="HiFT experiments config.yaml")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.hift_dir))
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.utils.model_load import load_pretrain

    cfg.merge_from_file(args.config)
    exemplar = cfg.TRACK.EXEMPLAR_SIZE if hasattr(cfg.TRACK, "EXEMPLAR_SIZE") else 127
    search = cfg.TRACK.INSTANCE_SIZE if hasattr(cfg.TRACK, "INSTANCE_SIZE") else 255
    print("exemplar=%d  search=%d" % (exemplar, search))

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).eval()
    # NOTE: ModelBuilder puts submodules on CUDA in __init__. For CPU/ONNX export move
    # everything to CPU first.  TODO: confirm this is enough (some ops hard-code .cuda()).
    model = model.cpu()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- template branch: z -> zf ------------------------------------------------
    class TemplateBranch(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
        def forward(self, z):
            return self.backbone(z)   # zf  (TODO: tensor or tuple? adjust output_names)

    tmpl = TemplateBranch(model).eval()
    z = torch.randn(1, 3, exemplar, exemplar)
    torch.onnx.export(
        tmpl, z, os.path.join(args.out_dir, "hift_template.onnx"),
        input_names=["z"], output_names=["zf"], opset_version=args.opset, dynamo=False)
    print("wrote hift_template.onnx")

    # --- track branch: (x, zf) -> loc, cls1, cls2 --------------------------------
    class TrackBranch(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
            self.grader = m.grader
        def forward(self, x, zf):
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, zf)
            return loc, cls1, cls2

    trk = TrackBranch(model).eval()
    x = torch.randn(1, 3, search, search)
    with torch.no_grad():
        zf = tmpl(z)   # real-shaped zf to trace the track graph
    torch.onnx.export(
        trk, (x, zf), os.path.join(args.out_dir, "hift_track.onnx"),
        input_names=["x", "zf"], output_names=["loc", "cls1", "cls2"],
        opset_version=args.opset, dynamo=False)
    print("wrote hift_track.onnx")

    print("\nNEXT: onnxslim both for static shapes, then build TRT engines on the Jetson.")
    print("Verify zf structure + search/exemplar sizes against cfg before trusting these.")


if __name__ == "__main__":
    main()
