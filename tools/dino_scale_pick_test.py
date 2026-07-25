#!/usr/bin/env python3
"""Go/no-go test for the DINOv2 scale-pick box-size idea (analysis only, no tracker code).

The scale-pick would size the box by embedding the target at a few candidate box sizes and
keeping the one DINOv2 matches best. That only works if DINOv2's cosine has a SHARP peak at
the correct size — but DINOv2 is deliberately scale-tolerant, which may flatten it. Measure
that before writing any C++.

Method: for each UAV123 ground-truth box, embed the GT-size crop as the "template", then
embed the same target center at 0.7x..1.3x box size (bigger = more background, smaller =
target cropped) and measure cosine vs the template. Average over many targets.

Verdict:
  * SHARP peak at 1.0x (cosine drops clearly by 0.8x/1.2x) -> scale-pick VIABLE, build it.
  * FLAT (cosine ~0.98 everywhere)                         -> use a scale pyramid / trained head.

Uses the SAME preprocessing as the production verifier (gray -> 3ch, 224, ImageNet norm).

Run on the Jetson (needs onnxruntime + the exported dino model + UAV123):
  python3 tools/dino_scale_pick_test.py \
      --onnx models/dinov2_small.onnx \
      --seqs boat6 uav5 person1 bike1 --per-seq 25
"""
import argparse
import glob
import os

import cv2
import numpy as np

DATA_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_BASE = "/home/nvidia/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 224
SCALES = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def crop(img, box_xywh):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box_xywh
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = min(w_img, int(round(x + w))), min(h_img, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def load_gt(seq):
    p = os.path.join(ANNO_BASE, seq + ".txt")
    if not os.path.isfile(p):
        return None
    out = []
    for line in open(p):
        line = line.strip().replace(",", " ")
        if not line:
            out.append(None); continue
        try:
            x, y, w, h = [float(v) for v in line.split()[:4]]
            out.append(None if (any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0)
                       else [x, y, w, h])
        except Exception:
            out.append(None)
    return out


class Dino:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        out = self.sess.run([self.out_name], {self.in_name: blob})[0].flatten()
        n = np.linalg.norm(out)
        return out.astype(np.float64) / n if n > 0 else out.astype(np.float64)


def scaled_box(b, s):
    x, y, w, h = b
    cx, cy = x + w / 2.0, y + h / 2.0
    return [cx - w * s / 2.0, cy - h * s / 2.0, w * s, h * s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--per-seq", type=int, default=25)
    ap.add_argument("--min-size", type=int, default=30)
    args = ap.parse_args()

    dino = Dino(args.onnx)
    sims = {s: [] for s in SCALES}
    total = 0

    for seq in args.seqs:
        seq_dir = os.path.join(DATA_BASE, seq)
        frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        gt = load_gt(seq)
        if not frames or gt is None:
            print("skip", seq); continue
        idx = [i for i, b in enumerate(gt) if b and min(b[2], b[3]) >= args.min_size]
        if not idx:
            continue
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1,
                                           min(args.per_seq, len(idx))).astype(int)]
        print("[%s] %d targets" % (seq, len(idx)))
        for i in idx:
            img = cv2.imread(frames[i])
            if img is None:
                continue
            tc = crop(img, gt[i])
            if tc is None or tc.size == 0:
                continue
            tmpl = dino.embed(tc)
            for s in SCALES:
                c = crop(img, scaled_box(gt[i], s))
                if c is not None and c.size:
                    sims[s].append(cosine(tmpl, dino.embed(c)))
            total += 1

    if total == 0:
        print("No targets."); return 1

    print("\nDINOv2 cosine vs box scale (n=%d):" % total)
    means = {}
    for s in SCALES:
        m = float(np.mean(sims[s]))
        means[s] = m
        bar = "#" * int(round((m - 0.7) / 0.3 * 40)) if m > 0.7 else ""
        print("  %.2fx  %.4f  %s" % (s, m, bar))

    drop = means[1.0] - 0.5 * (means[0.8] + means[1.2])
    print("\n  peak drop (1.0x vs mean of 0.8x/1.2x) = %.4f" % drop)
    if drop >= 0.03:
        print("  VERDICT: sharp enough → scale-pick VIABLE. Build it in C++.")
    elif drop >= 0.012:
        print("  VERDICT: marginal → may work but coarse; test a scale pyramid too.")
    else:
        print("  VERDICT: too FLAT → scale-pick unreliable. Use a scale pyramid / trained head.")
    print("\n(Grayscale input, matching production. If flat, retry with RGB — DINOv2 is "
          "RGB-trained and colour may sharpen the size signal.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
