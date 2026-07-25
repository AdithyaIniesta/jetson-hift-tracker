"""UAV123 dataset adapter.

Expected layout under `root`:

    root/
      data_seq/UAV123/<seqname>/<frame_number>.jpg
      anno/UAV123/<seqname>.txt      # per-frame (x,y,w,h), 1 line per frame

Downloaded from https://cemse.kaust.edu.sa/ivul/uav123 — the "complete UAV123
& UAV20L" archive. This adapter targets the UAV123 subset (not UAV20L). Each
annotation file is CSV: "x,y,w,h" per line, with NaN entries for occluded
frames. Frame filenames are zero-padded 6-digit indices starting at 000001.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

from .base import DatasetAdapter, Sequence


class UAV123Adapter(DatasetAdapter):
    ANNO_SUBDIR = "anno/UAV123"
    FRAMES_SUBDIR = "data_seq/UAV123"

    def _anno_dir(self) -> Path:
        return self.root / self.ANNO_SUBDIR

    def _frames_dir(self) -> Path:
        return self.root / self.FRAMES_SUBDIR

    def list_sequences(self) -> List[str]:
        anno = self._anno_dir()
        if not anno.is_dir():
            raise FileNotFoundError(
                f"expected {anno} to exist — is the dataset root correct?"
            )
        return sorted(p.stem for p in anno.glob("*.txt"))

    def load(self, name: str) -> Sequence:
        anno_file = self._anno_dir() / f"{name}.txt"
        frames_dir = self._frames_dir() / name

        if not anno_file.exists():
            raise FileNotFoundError(f"annotation {anno_file} missing")
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"frames dir {frames_dir} missing")

        # Parse annotations. Empty / NaN lines mean occlusion — keep as
        # (NaN, NaN, NaN, NaN) so the frame index still lines up with
        # the frames on disk.
        gt: List[tuple] = []
        with anno_file.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.lower() == "nan,nan,nan,nan":
                    gt.append((math.nan, math.nan, math.nan, math.nan))
                    continue
                try:
                    parts = [float(x) for x in line.split(",")]
                    if len(parts) != 4:
                        gt.append((math.nan,) * 4)
                        continue
                    gt.append(tuple(parts))
                except ValueError:
                    gt.append((math.nan,) * 4)

        # Frames — .jpg, zero-padded 6-digit indices from 000001.
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"no .jpg frames in {frames_dir}")

        # Trim to matching length — sometimes anno has one fewer / more line
        # than the frames folder. Common quirk.
        n = min(len(frames), len(gt))
        frames = frames[:n]
        gt = gt[:n]

        # First-frame ground truth is our init bbox. If it's NaN (rare but
        # possible), walk forward until we find the first labeled frame and
        # start there.
        init_idx = 0
        while init_idx < len(gt) and math.isnan(gt[init_idx][0]):
            init_idx += 1
        if init_idx >= len(gt):
            raise ValueError(f"sequence {name} has no labeled frames")
        init_bbox = gt[init_idx]

        # Peek at the first frame to get resolution — needed for the feeder.
        # Cheap enough (JPEG header only, no full decode).
        w, h = _read_jpeg_dimensions(frames[init_idx])

        # UAV123 frames are named "000001.jpg", "000002.jpg", … so
        # multifilesrc can read them in place. `start_index` is the
        # numeric index of the first frame after we skipped any NaN
        # entries at the head.
        first_num = int(frames[init_idx].stem)  # "000047" -> 47
        direct = (frames_dir, "%06d.jpg", first_num)

        return Sequence(
            name=name,
            frames=frames[init_idx:],
            init_bbox=init_bbox,
            gt_bboxes=gt[init_idx:],
            width=w,
            height=h,
            direct_source=direct,
        )


def _read_jpeg_dimensions(path: Path) -> tuple:
    """Read (width, height) from a JPEG file without decoding the whole
    image. Walks the segment markers looking for SOF0/SOF2. Robust enough
    for UAV123 which uses baseline JPEG throughout."""
    with path.open("rb") as f:
        data = f.read()
    i = 0
    # Skip SOI
    if data[i:i + 2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG")
    i += 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"malformed JPEG {path} at byte {i}")
        while i < len(data) and data[i] == 0xFF:
            i += 1
        marker = data[i]
        i += 1
        # SOFn markers hold the frame dimensions.
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            # skip length (2 bytes) + precision (1 byte)
            i += 3
            h = int.from_bytes(data[i:i + 2], "big")
            w = int.from_bytes(data[i + 2:i + 4], "big")
            return w, h
        # otherwise skip this segment
        length = int.from_bytes(data[i:i + 2], "big")
        i += length
    raise ValueError(f"no SOF marker in {path}")
