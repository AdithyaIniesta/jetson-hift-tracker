#!/usr/bin/env python3
"""Export ResNet18 truncated at block2 + global-average-pool to ONNX for the tracker's
appearance verifier (TrtFeatureExtractor / TRACKER_VERIFIER=resnet).

The C++ verifier feeds a 128x128 patch, grayscale replicated to 3 channels, ImageNet-
normalized (out = (pixel/255 - mean)/std). This model must therefore accept 1x3x128x128
and emit an L2-comparable embedding. We output the GAP vector after layer2 (128-D), which
the research showed is scale-invariant. The C++ side L2-normalizes the output.

Run ON THE JETSON (needs torch), then the file lands in models/:
    python3 tools/export_resnet18_block2.py --out models/resnet18_block2.onnx

Note: matches TrtExtractorConfig defaults (inputSize=128, channels=3, ImageNet mean/std).
"""
import argparse

import torch
import torch.nn as nn
import torchvision


class ResNet18Block2(nn.Module):
    """conv1..layer2 then global-average-pool -> 128-D embedding."""
    def __init__(self):
        super().__init__()
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=w)
        except Exception:
            net = torchvision.models.resnet18(pretrained=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return torch.flatten(x, 1)   # (N, 128); C++ L2-normalizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/resnet18_block2.onnx")
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    model = ResNet18Block2().eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        out = model(dummy)
    print("Output embedding dim:", out.shape[1])

    # Fixed batch=1 (no dynamic axes): the C++ verifier feeds one patch at a time,
    # and TrtFeatureExtractor builds the engine WITHOUT a dynamic optimization
    # profile (same as embedder_legacy.onnx). A dynamic batch axis makes TensorRT
    # fail with "no optimization profile has been defined".
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["embedding"],
        opset_version=args.opset,
    )
    print("Wrote", args.out, "(fixed batch=1)")


if __name__ == "__main__":
    main()
