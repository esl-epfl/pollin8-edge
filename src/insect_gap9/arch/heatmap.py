"""Centroid-heatmap geometry (pure NumPy; no torch). Unit-tested.

A centroid detector predicts, per class, a low-resolution grid where each cell carries the
probability that an object centre falls in it. These functions convert between YOLO boxes
and that grid target, and decode a predicted grid back into centroids for scoring with the
existing centre-matched F1 / counting metrics (insect_gap9.monitor_metrics).
"""
from __future__ import annotations
import numpy as np


def boxes_to_heatmap(boxes, nc: int, grid: int) -> np.ndarray:
    """boxes: list of (cls, xc, yc) normalised in [0,1]. Returns (nc, grid, grid) {0,1}
    target: the cell containing each object centre is set to 1 (FOMO-style hard target)."""
    t = np.zeros((nc, grid, grid), dtype=np.float32)
    for c, xc, yc in boxes:
        gx = min(int(xc * grid), grid - 1)
        gy = min(int(yc * grid), grid - 1)
        t[int(c), gy, gx] = 1.0
    return t


def heatmap_to_centroids(hm: np.ndarray, conf: float = 0.5, peak: bool = True):
    """hm: (nc, grid, grid) probabilities. Returns list of (cls, xc, yc, score) at cell
    centres (normalised). If peak, keep only 3x3 local maxima (deduplicates blobs);
    otherwise every cell above `conf`."""
    nc, gh, gw = hm.shape
    out = []
    for c in range(nc):
        m = hm[c]
        for gy in range(gh):
            for gx in range(gw):
                v = m[gy, gx]
                if v < conf:
                    continue
                if peak:
                    y0, y1 = max(gy - 1, 0), min(gy + 2, gh)
                    x0, x1 = max(gx - 1, 0), min(gx + 2, gw)
                    if v < m[y0:y1, x0:x1].max():
                        continue   # not a local maximum
                out.append((c, (gx + 0.5) / gw, (gy + 0.5) / gh, float(v)))
    return out


def grid_for(imgsz: int, stride: int) -> int:
    """Heatmap side length for a square input at a given output stride."""
    return max(imgsz // stride, 1)
