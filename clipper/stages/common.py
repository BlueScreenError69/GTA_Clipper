"""Shared job context + stage runner (status, logging, skip-if-done)."""
from __future__ import annotations
import json
import logging
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path

from ..manifest import Manifest


class StageError(RuntimeError):
    """Expected, user-facing failure (short message, no traceback in output)."""


class Waiting(Exception):
    """Stage cannot proceed until someone supplies input (e.g. Claude writes scores.json).
    Not a failure: the stage stays 'pending' and `resume` continues from it."""


@dataclass
class Job:
    settings: object
    job_id: str
    dir: Path
    manifest: Manifest

    @classmethod
    def open(cls, settings, job_id: str) -> "Job":
        d = settings.jobs_dir / job_id
        (d / "logs").mkdir(parents=True, exist_ok=True)
        return cls(settings, job_id, d, Manifest(d))

    def read_json(self, name: str) -> dict:
        return json.loads((self.dir / name).read_text())

    def logger(self, stage: str) -> logging.Logger:
        lg = logging.getLogger(f"clipper.{self.job_id}.{stage}")
        if not lg.handlers:
            h = logging.FileHandler(self.dir / "logs" / f"{stage}.log")
            h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
            lg.propagate = False
        return lg


def probe_duration(settings, path: Path) -> float:
    """ffprobe duration in seconds; raises StageError if no audio/video stream is readable."""
    r = subprocess.run([settings.ffprobe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise StageError(f"ffprobe could not read {path.name}")


def run_stage(job: Job, name: str, fn, verify=None) -> str:
    """Run fn(job) -> list of output names. Skips when done and verify(job) passes.
    Returns 'skipped' | 'done'. Raises StageError / other on failure after marking failed."""
    st = job.manifest.get(name)
    if st["status"] == "done" and (verify is None or verify(job)):
        return "skipped"
    log = job.logger(name)
    job.manifest.mark(name, "running")
    try:
        outputs = fn(job) or []
    except Waiting as w:
        job.manifest.mark(name, "pending", error=f"waiting: {w}")
        log.info("waiting: %s", w)
        raise
    except Exception as e:
        log.error("failed: %s\n%s", e, traceback.format_exc())
        job.manifest.mark(name, "failed", error=str(e))
        raise
    job.manifest.mark(name, "done", outputs=outputs)
    log.info("done: %s", outputs)
    return "done"
