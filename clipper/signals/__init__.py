"""Signal plugin registry. A signal turns a job's audio/transcript into scored time ranges.
Adding one = write a Signal subclass + append it to SIGNALS. No other stage changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class TimedScore:
    start: float
    end: float
    score: float                    # 0..1, higher = more interesting
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"], d["score"] = round(self.start, 2), round(self.end, 2), round(self.score, 3)
        return d


class Signal:
    name = "base"

    def __init__(self, settings):
        self.settings = settings

    def enabled(self) -> bool:
        return True

    def cfg(self, key: str) -> dict:
        return (self.settings.yaml.get("signals") or {}).get(key) or {}

    def compute(self, job) -> list[TimedScore]:
        raise NotImplementedError


from .loudness import Loudness            # noqa: E402
from .speech import SpeechRate            # noqa: E402
from .keywords import Keywords            # noqa: E402
from .chat_activity import ChatActivity   # noqa: E402

SIGNALS = [Loudness, SpeechRate, Keywords, ChatActivity]


def active(settings) -> list[Signal]:
    return [s for s in (cls(settings) for cls in SIGNALS) if s.enabled()]
