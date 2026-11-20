"""signals stage: run every enabled plugin, write signals.json (pure local compute, no LLM)."""
from __future__ import annotations
import json

from .. import signals as registry
from ..manifest import atomic_write
from .common import Job, StageError


def run(job: Job) -> list[str]:
    log = job.logger("signals")
    meta = job.read_json("meta.json")
    out = {"duration_s": meta["duration_s"], "signals": {}}
    for sig in registry.active(job.settings):
        try:
            res = sig.compute(job)
        except NotImplementedError as e:
            raise StageError(f"{e}. Disable its flag in .env or build it first.")
        out["signals"][sig.name] = [r.to_dict() for r in res]
        log.info("%s: %d ranges", sig.name, len(res))
    atomic_write(job.dir / "signals.json", json.dumps(out))
    return ["signals.json"]


def verify(job: Job) -> bool:
    p = job.dir / "signals.json"
    try:
        return p.exists() and "signals" in json.loads(p.read_text())
    except ValueError:
        return False
