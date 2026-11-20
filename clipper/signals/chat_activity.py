"""Chat-activity bursts from chat.json (message timestamps only, relative to the recording start).
Writes nothing and returns [] when a job has no chat data (e.g. VOD jobs)."""
from __future__ import annotations
import json

import numpy as np

from . import Signal, TimedScore
from .util import rolling_z, runs


def rate_per_second(times: list[float], n: int, smooth_s: int) -> np.ndarray:
    c = np.zeros(n)
    for t in times:
        i = int(t)
        if 0 <= i < n:
            c[i] += 1
    k = max(1, smooth_s)
    return np.convolve(c, np.ones(k) / k, mode="same")


class ChatActivity(Signal):
    name = "chat_activity"

    def enabled(self) -> bool:
        return bool(self.settings.chat_activity)

    def compute(self, job) -> list[TimedScore]:
        p = job.dir / "chat.json"
        if not p.exists():
            return []
        c = self.cfg("chat_activity")
        n = int(np.ceil(job.read_json("meta.json")["duration_s"]))
        rate = rate_per_second(json.loads(p.read_text())["times"], n, int(c.get("smooth_s", 10)))
        z = rolling_z(rate, int(c.get("window_s", 120)), float(c.get("min_std", 0.3)))
        mask = (z >= float(c.get("z_threshold", 2.0))) & (rate >= float(c.get("min_rate", 0.3)))
        return [TimedScore(a, b, min(1.0, float(z[a:b].max()) / 4.0),
                           {"z": round(float(z[a:b].max()), 2), "msgs_per_s": round(float(rate[a:b].max()), 2)})
                for a, b in runs(mask)]
