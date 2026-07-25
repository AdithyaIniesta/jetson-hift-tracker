#!/usr/bin/env python3
"""Export DINOv2-small (facebook/dinov2-small) to a static-shape ONNX for the tracker's
appearance verifier (TRACKER_VERIFIER=dino) — a self-supervised alternative to the
embedder / ResNet block2, known for strong one-shot instance retrieval on novel objects.

DINOv2 outputs per-patch tokens; this wraps it to emit only the CLS token
(last_hidden_state[:, 0, :]) — a single 384-D global descriptor, matching the
"one output tensor = the embedding" contract of cvtracker-lockon's FeatureExtractor.

Input is [1,3,224,224] ImageNet-normalized. NOTE: the tracker feeds the verifier a
GRAYSCALE patch replicated to 3 channels, so DINOv2 runs here without colour — it still
works but is handicapped vs its RGB training. Watch per-frame latency: this is a ViT and
much heavier than the 128px CNNs.

Run ON A MACHINE WITH INTERNET (downloads the model) + torch/transformers:
    pip install torch transformers onnx onnxslim onnxruntime
    python3 tools/export_dinov2.py --out models/dinov2_small.onnx

Two non-obvious steps, both handled:
1. Uses the legacy TorchScript exporter (dynamo=False) at opset 14 — conservative op set,
   better precedented on this project's Jetson/TensorRT than the newer dynamo/opset-18 path.
2. onnxslim constant-folds away the leftover dynamic batch dim (a shape-introspection
   artifact; input is always fixed [1,3,224,224]) so TrtFeatureExtractor gets fully static
   shapes. Verified numerically identical to PyTorch (cosine 1.0).
"""
import argparse
import subprocess
import sys

import numpy as np
import torch
from transformers import Dinov2Model


class Dinov2ClsEmbedder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]   # [batch, 384]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/dinov2_small.onnx")
    args = ap.parse_args()

    model = Dinov2Model.from_pretrained("facebook/dinov2-small").eval()
    wrapped = Dinov2ClsEmbedder(model)

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch_out = wrapped(dummy).numpy()
    print("Sanity forward output shape:", torch_out.shape)

    raw_path = args.out.replace(".onnx", "_raw.onnx")
    torch.onnx.export(
        wrapped, dummy, raw_path,
        input_names=["pixel_values"], output_names=["cls_embedding"],
        opset_version=14, do_constant_folding=True, dynamo=False,
    )
    print("Raw export:", raw_path)

    subprocess.run([sys.executable, "-m", "onnxslim", raw_path, args.out], check=True)
    print("Simplified (static shapes):", args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out)
    onnx_out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    cos = float(np.dot(torch_out.flatten(), onnx_out.flatten())
                / (np.linalg.norm(torch_out) * np.linalg.norm(onnx_out)))
    print("Verification vs PyTorch: cosine=%.6f" % cos)
    assert cos > 0.999, "Simplified ONNX diverged from PyTorch reference"
    print("OK ->", args.out)


if __name__ == "__main__":
    main()
