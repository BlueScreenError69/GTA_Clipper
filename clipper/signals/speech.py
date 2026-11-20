"""Speech-rate bursts: words/sec (smoothed) well above the local baseline."""
from __future__ import annotations
import json

import numpy as np

from . import Signal, TimedScore
from .util import rolling_z, runs


def words_per_second(words: list[dict], n_bins: int, smooth_s: int) -> np.ndarray:
    counts = np.zeros(n_bins, dtype=np.float64)
    for w in words:
        i = int((w["start"] + w["end"]) / 2)
        if 0 <= i < n_bins:
            counts[i] += 1
    k = max(1, smooth_s)
    return np.convolve(counts, np.ones(k) / k, mode="same")


class SpeechRate(Signal):
    name = "speech_rate"

    def compute(self, job) -> list[TimedScore]:
        c = self.cfg("speech_rate")
        dur = job.read_json("meta.json")["duration_s"]
        words = json.loads((job.dir / "transcript.json").read_text())["words"]
        rate = words_per_second(words, int(np.ceil(dur)), int(c.get("smooth_s", 5)))
        z = rolling_z(rate, int(c.get("window_s", 120)), float(c.get("min_std", 0.5)))
        mask = (z >= float(c.get("z_threshold", 2.0))) & (rate >= float(c.get("min_rate", 2.0)))
        return [TimedScore(a, b, min(1.0, float(z[a:b].max()) / 4.0),
                           {"z": round(float(z[a:b].max()), 2), "wps": round(float(rate[a:b].max()), 2)})
                for a, b in runs(mask)]
