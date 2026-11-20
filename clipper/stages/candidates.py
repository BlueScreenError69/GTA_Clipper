"""candidates stage: combine signals -> moments -> clip windows -> top N compact file.
Pure local compute. candidates_compact.json is the only file Claude reads for scoring."""
from __future__ import annotations
import bisect
import json

import numpy as np

from ..manifest import atomic_write
from ..signals.util import runs
from .common import Job

DEFAULT = {
    "weights": {"loudness": 0.3, "speech_rate": 0.3, "keywords": 0.4, "chat_activity": 0.3},
    "min_moment": 0.15, "merge_gap_s": 10, "lead_s": 12, "payoff_s": 15,
    "max_overlap": 0.25, "snap_s": 2.0, "pause_s": 0.3,
    "music_words": ["music", "song", "radio", "playlist", "track", "lyrics"], "music_min_wps": 0.8,
}


# --- pure helpers ---------------------------------------------------------------------
def combined_timeline(sig: dict, n: int, weights: dict) -> tuple[np.ndarray, dict]:
    """Weighted per-second score in 0..1 (normalised by weights of signals that ran)
    plus each signal's own per-second array."""
    per = {}
    for name, ranges in sig.items():
        a = np.zeros(n)
        for r in ranges:
            lo, hi = int(r["start"]), max(int(r["start"]) + 1, int(np.ceil(r["end"])))
            a[max(0, lo):min(n, hi)] = np.maximum(a[max(0, lo):min(n, hi)], r["score"])
        per[name] = a
    tot = sum(weights.get(k, 0.2) for k in per) or 1.0
    comb = sum(weights.get(k, 0.2) * a for k, a in per.items()) / tot if per else np.zeros(n)
    return np.asarray(comb), per


def find_moments(comb: np.ndarray, per: dict, min_moment: float, merge_gap: int) -> list[dict]:
    rs = runs(comb >= min_moment)
    merged: list[list[int]] = []
    for a, b in rs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        peak = a + int(np.argmax(comb[a:b]))
        firing = [k for k, arr in per.items() if arr[a:b].max() > 0]
        out.append({"start": a, "end": b, "peak": peak, "peak_score": float(comb[peak]),
                    "signals": firing,
                    "score": float(min(1.0, comb[peak] + 0.1 * (len(firing) - 1)))})
    return out


def snap_start(starts, ends, t, shift, pause):
    """Nearest word start (preferring ones after a pause) within +-shift of t."""
    lo, hi = bisect.bisect_left(starts, t - shift), bisect.bisect_right(starts, t + shift)
    idx = range(lo, hi)
    paused = [i for i in idx if i == 0 or starts[i] - ends[i - 1] >= pause]
    pool = paused or list(idx)
    return starts[min(pool, key=lambda i: abs(starts[i] - t))] if pool else t


def snap_end(starts, ends, t, shift, pause):
    """Nearest word end (preferring ones before a pause) within +-shift of t."""
    order = sorted(range(len(ends)), key=lambda i: ends[i]) if ends != sorted(ends) else range(len(ends))
    e = [ends[i] for i in order]
    lo, hi = bisect.bisect_left(e, t - shift), bisect.bisect_right(e, t + shift)
    idx = [order[i] for i in range(lo, hi)]
    paused = [i for i in idx if i == len(starts) - 1 or starts[i + 1] - ends[i] >= pause]
    pool = paused or idx
    return ends[min(pool, key=lambda i: abs(ends[i] - t))] if pool else t


def build_window(m: dict, dur: float, words_se: tuple, cfg: dict, cmin: int, cmax: int):
    starts, ends = words_se
    start = m["peak"] - cfg["lead_s"]
    end = max(m["peak"] + cfg["payoff_s"], min(m["end"] + 3, m["peak"] + 25))
    start, end = max(0.0, start), min(dur, end)
    if starts:
        start = snap_start(starts, ends, start, cfg["snap_s"], cfg["pause_s"])
        end = snap_end(starts, ends, end, cfg["snap_s"], cfg["pause_s"])
    if end - start > cmax:                       # trim lead-in first, keep the payoff
        start = max(0.0, end - cmax)
        if starts:
            start = snap_start(starts, ends, start, 1.0, cfg["pause_s"])
        if end - start > cmax:
            start = end - cmax
    if end - start < cmin:                       # grow: payoff side first, then lead side
        end = min(dur, start + cmin)
        start = max(0.0, end - cmin)
    if end - start < min(cmin, 5):
        return None
    return round(start, 2), round(end, 2)


def overlap_frac(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def select(cands: list[dict], max_overlap: float, n: int) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["heuristic_score"]):
        if all(overlap_frac((c["start_s"], c["end_s"]), (k["start_s"], k["end_s"])) <= max_overlap
               for k in kept):
            kept.append(c)
        if len(kept) >= n:
            break
    return kept


def excerpt(words: list[dict], start: float, end: float, peak: float, max_words: int) -> str:
    inside = [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]
    if len(inside) > max_words:
        ci = min(range(len(inside)), key=lambda i: abs(inside[i]["start"] - peak))
        lo = max(0, min(ci - max_words // 2, len(inside) - max_words))
        inside = inside[lo:lo + max_words]
    return " ".join(w["w"] for w in inside)


# --- stage ----------------------------------------------------------------------------
def run(job: Job) -> list[str]:
    s = job.settings
    log = job.logger("candidates")
    cfg = {**DEFAULT, **((s.yaml.get("candidates")) or {})}
    meta = job.read_json("meta.json")
    rights = job.read_json("rights.json")
    sigs = job.read_json("signals.json")["signals"]
    tr = job.read_json("transcript.json")
    words = sorted(tr["words"], key=lambda w: w["start"])
    dur = float(meta["duration_s"])
    n = int(np.ceil(dur))
    starts, ends = [w["start"] for w in words], [w["end"] for w in words]

    comb, per = combined_timeline(sigs, n, cfg["weights"])
    moments = find_moments(comb, per, cfg["min_moment"], int(cfg["merge_gap_s"]))
    log.info("%d moments", len(moments))

    cands = []
    for m in moments:
        win = build_window(m, dur, (starts, ends), cfg, s.clip_min, s.clip_max)
        if not win:
            continue
        a, b = win
        c_words = [w for w in words if a <= (w["start"] + w["end"]) / 2 < b]
        wps = len(c_words) / max(1.0, b - a)
        ls = [r for r in sigs.get("loudness", []) if r["end"] > a and r["start"] < b]
        kws = sorted({r["meta"].get("keyword") for r in sigs.get("keywords", [])
                      if r["end"] > a and r["start"] < b and r["meta"].get("keyword")})
        text = excerpt(words, a, b, m["peak"], s.excerpt_max_words)
        flags = []
        if s.music_check and (
                (ls and wps < cfg["music_min_wps"]) or
                any(mw in text.lower().split() for mw in cfg["music_words"])):
            flags.append("possible_music")
        avg_p = float(np.mean([w["p"] for w in c_words])) if c_words else 1.0
        if "caption_check" in tr.get("flags", []) or avg_p < 0.5:
            flags.append("caption_check")
        cands.append({
            "start_s": a, "end_s": b, "peak_s": m["peak"],
            "heuristic_score": round(m["score"], 3),
            "signals": {"loudness_z": max((r["meta"].get("z", 0) for r in ls), default=0),
                        "speech_burst": any(r["end"] > a and r["start"] < b
                                            for r in sigs.get("speech_rate", [])),
                        "keywords": kws},
            "flags": flags, "excerpt": text})

    top = select(cands, float(cfg["max_overlap"]), s.max_llm_candidates)
    for i, c in enumerate(sorted(top, key=lambda c: c["start_s"]), 1):
        c["id"] = f"c{i:02d}"
    top.sort(key=lambda c: -c["heuristic_score"])
    out = {"job_id": job.job_id,
           "source": {"creator": rights.get("creator", ""), "title": meta.get("title", ""),
                      "duration_s": dur},
           "candidates": [{k: c[k] for k in ("id", "start_s", "end_s", "heuristic_score",
                                             "signals", "flags", "excerpt")} for c in top]}
    atomic_write(job.dir / "candidates_compact.json", json.dumps(out, indent=1))
    log.info("kept %d of %d windows", len(top), len(cands))
    return ["candidates_compact.json"]


def verify(job: Job) -> bool:
    p = job.dir / "candidates_compact.json"
    try:
        return p.exists() and "candidates" in json.loads(p.read_text())
    except ValueError:
        return False
