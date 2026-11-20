"""Reaction keywords/phrases from config/settings.yaml (editable, Hindi/Hinglish included)."""
from __future__ import annotations
import json
import re

from . import Signal, TimedScore

_clean = re.compile(r"[^\w' ]+", re.UNICODE)


def norm(w: str) -> str:
    return _clean.sub("", w.lower()).strip()


def find_phrases(words: list[dict], phrases: list[str]) -> list[tuple[str, float, float]]:
    """[(phrase, start, end)] for every occurrence of each phrase (word-aligned)."""
    toks = [norm(w["w"]) for w in words]
    hits = []
    for ph in phrases:
        parts = [norm(p) for p in ph.split()]
        n = len(parts)
        if not n or not all(parts):
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == parts:
                hits.append((ph, words[i]["start"], words[i + n - 1]["end"]))
    return hits


class Keywords(Signal):
    name = "keywords"

    def compute(self, job) -> list[TimedScore]:
        c = self.cfg("keywords")
        pad = float(c.get("pad_s", 1.5))
        words = json.loads((job.dir / "transcript.json").read_text())["words"]
        hits = find_phrases(words, [str(p) for p in c.get("list", [])])
        return sorted((TimedScore(max(0.0, s - pad), e + pad, 1.0, {"keyword": ph})
                       for ph, s, e in hits), key=lambda t: t.start)
