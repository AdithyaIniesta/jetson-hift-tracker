#!/usr/bin/env python3
"""Export HiFT to two static-shape ONNX graphs for TensorRT deployment.

HiFT is a Siamese tracker (hift/pysot/models/model_builder.py):
  * template(z): backbone(z) -> zf (single tensor)      run once on CAPTURE
  * track(x):    backbone(x) + grader(xf, zf) -> loc, cls1, cls2   per frame

We export two graphs so the template branch runs once and only the track branch runs per
frame:
  1) hift_template.onnx:  z[1,3,127,127]        -> zf
  2) hift_track.onnx:     x[1,3,S,S] + zf       -> loc, cls1, cls2      (S = search size)

Box post-processing (getcentercuda) is NOT exported — port it to C++ (HiFTTracker).

The Jetson's torch is CPU-only but HiFT hard-codes .cuda() everywhere (model builder, the
grader's utile.py, the weight loader). We monkeypatch .cuda() to a CPU no-op and load the
checkpoint with map_location='cpu', so no vendored code needs editing.

Usage (on the Jetson):
  python3 tools/export_hift.py \
      --snapshot hift/pretrained_models/first.pth \
      --config   hift/experiments/config.yaml
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

# --- make HiFT's hard-coded CUDA calls CPU no-ops (Jetson torch is CPU-only) ---
torch.Tensor.cuda = lambda self, *a, **k: self          # type: ignore
nn.Module.cuda = lambda self, *a, **k: self              # type: ignore
if not hasattr(torch.cuda, "_orig_current_device"):
    torch.cuda._orig_current_device = getattr(torch.cuda, "current_device", None)
torch.cuda.current_device = lambda: 0                    # used by load_pretrain
torch.cuda.is_available = lambda: False


def load_weights_cpu(model, path):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("  (missing keys: %d — e.g. %s)" % (len(missing), missing[:3]))
    if unexpected:
        print("  (unexpected keys: %d — e.g. %s)" % (len(unexpected), unexpected[:3]))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hift-dir", default="hift",
                    help="path to the vendored HiFT model code (default: ./hift)")
    ap.add_argument("--snapshot", required=True, help="trained weights, e.g. first.pth")
    ap.add_argument("--config", required=True, help="HiFT config.yaml")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--search", type=int, default=0,
                    help="search input size (0 = use cfg.TRAIN.SEARCH_SIZE)")
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.hift_dir))
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder

    cfg.merge_from_file(args.config)
    exemplar = cfg.TRAIN.EXEMPLAR_SIZE
    search = args.search or cfg.TRAIN.SEARCH_SIZE
    print("exemplar=%d  search=%d" % (exemplar, search))

    model = ModelBuilder().eval().cpu()
    load_weights_cpu(model, args.snapshot)
    os.makedirs(args.out_dir, exist_ok=True)

    # The AlexNet backbone returns a TUPLE of feature maps (HiFT uses hierarchical
    # features); the grader indexes x[0..2] / z[0..2]. Export handles N tensors.

    # --- template branch: z -> zf0..zf{N-1} ---
    class TemplateBranch(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
        def forward(self, z):
            return self.backbone(z)      # tuple of feature maps

    tmpl = TemplateBranch(model).eval()
    z = torch.randn(1, 3, exemplar, exemplar)
    with torch.no_grad():
        zf = tmpl(z)
    zf_list = list(zf) if isinstance(zf, (tuple, list)) else [zf]
    n = len(zf_list)
    zf_names = ["zf%d" % i for i in range(n)]
    print("zf: %d tensors, shapes: %s" % (n, [tuple(t.shape) for t in zf_list]))
    torch.onnx.export(
        tmpl, z, os.path.join(args.out_dir, "hift_template.onnx"),
        input_names=["z"], output_names=zf_names, opset_version=args.opset,
        do_constant_folding=True, dynamo=False)
    print("wrote hift_template.onnx")

    # --- track branch: (x, zf0..zf{N-1}) -> loc, cls1, cls2 ---
    class TrackBranch(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
            self.grader = m.grader
        def forward(self, x, *zfs):
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, list(zfs))
            return loc, cls1, cls2

    trk = TrackBranch(model).eval()
    x = torch.randn(1, 3, search, search)
    with torch.no_grad():
        loc, cls1, cls2 = trk(x, *zf_list)
    print("loc/cls1/cls2 shapes:", tuple(loc.shape), tuple(cls1.shape),
          tuple(cls2.shape))
    torch.onnx.export(
        trk, (x, *zf_list), os.path.join(args.out_dir, "hift_track.onnx"),
        input_names=["x"] + zf_names, output_names=["loc", "cls1", "cls2"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print("wrote hift_track.onnx")

    print("\nNEXT: onnxslim both for static shapes, then build TRT engines on the Jetson.")
    print("Record the printed shapes — the C++ HiFTTracker box post-processing depends on "
          "the loc/cls map size.")


if __name__ == "__main__":
    main()
