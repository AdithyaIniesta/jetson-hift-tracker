"""Abstract dataset adapter — defines the interface every benchmark must
expose so the harness can iterate its sequences uniformly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Sequence:
    """One benchmark sequence: an ordered list of frame paths + init bbox +
    per-frame ground-truth bboxes.

    bbox layout throughout the harness: (x, y, w, h) with (x, y) as the
    TOP-LEFT corner in pixel coordinates. Matches OpenCV convention and
    almost every SOT dataset's raw label format.

    `direct_source`: for datasets whose raw filenames already match
    multifilesrc's `%0Nd.ext` printf pattern (UAV123 uses "000001.jpg"),
    the adapter sets this to (directory, pattern, start_index). The
    feeder can then read files in place without symlinking through
    a scratch directory — noticeably faster on CIFS-mounted datasets
    and avoids some symlink-following quirks.
    """

    name: str                        # sequence name, e.g. "bike1"
    frames: List[Path]               # ordered list of frame files (JPG/PNG)
    init_bbox: tuple                 # (x, y, w, h) at frames[0]
    gt_bboxes: List[tuple]           # per-frame (x, y, w, h); NaNs on missing
    width: int                       # frame width in pixels
    height: int                      # frame height in pixels
    direct_source: Optional[tuple] = None  # (dir: Path, pattern: str, start: int)

    def __len__(self) -> int:
        return len(self.frames)


class DatasetAdapter:
    """Base class. Subclass per benchmark."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root {self.root} not found")

    def list_sequences(self) -> List[str]:
        """All sequence names in the dataset. Deterministic order."""
        raise NotImplementedError

    def load(self, name: str) -> Sequence:
        """Load one sequence by name."""
        raise NotImplementedError

    def iter(self, names: Iterable[str] | None = None) -> Iterable[Sequence]:
        """Iterate sequences. `names=None` means all."""
        for n in (names or self.list_sequences()):
            yield self.load(n)
