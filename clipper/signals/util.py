from __future__ import annotations
import numpy as np


def rolling_z(x: np.ndarray, window: int, min_std: float) -> np.ndarray:
    """z-score of each sample against a centered rolling mean/std (window in samples)."""
    n = len(x)
    if n == 0:
        return x.copy()
    half = max(1, window // 2)
    c1 = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - half), np.minimum(n, idx + half + 1)
    cnt = hi - lo
    mean = (c1[hi] - c1[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mean ** 2, 0.0)
    return (x - mean) / np.maximum(np.sqrt(var), min_std)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start_idx, end_idx_exclusive)] of consecutive True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out
