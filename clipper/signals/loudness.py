"""Loudness peaks: per-second RMS (dBFS) -> rolling z-score. A clue, not proof: explosions,
sirens and music also spike, so candidates later flag high loudness with little speech."""
from __future__ import annotations
import wave

import numpy as np

from . import Signal, TimedScore
from .util import rolling_z, runs


def per_second_db(path, block_s: int = 60) -> np.ndarray:
    """RMS in dBFS per 1 s bin, streamed from a 16 kHz mono 16-bit WAV."""
    out = []
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        while True:
            raw = w.readframes(sr * block_s)
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(a) // sr
            if n:
                sec = a[: n * sr].reshape(n, sr)
                out.append(20 * np.log10(np.sqrt((sec ** 2).mean(axis=1)) + 1e-9))
            rest = a[n * sr:]
            if len(rest) >= sr // 2:
                out.append(np.array([20 * np.log10(np.sqrt((rest ** 2).mean()) + 1e-9)], dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class Loudness(Signal):
    name = "loudness"

    def compute(self, job) -> list[TimedScore]:
        c = self.cfg("loudness")
        db = per_second_db(job.dir / "audio.wav")
        z = rolling_z(db, int(c.get("window_s", 120)), float(c.get("min_std_db", 2.0)))
        thr = float(c.get("z_threshold", 2.0))
        res = []
        for a, b in runs(z >= thr):
            peak = float(z[a:b].max())
            res.append(TimedScore(a, b, min(1.0, peak / 4.0),
                                  {"z": round(peak, 2), "db": round(float(db[a:b].max()), 1)}))
        return res
