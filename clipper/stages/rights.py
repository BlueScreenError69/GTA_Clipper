"""Rights gate. A URL is never fetched until rights.json exists with a stated basis."""
from __future__ import annotations
import json
import time

from ..manifest import atomic_write
from .common import Job, StageError


def write_rights(job: Job, url: str, basis: str, credit: str = "", platforms: str = "") -> list[str]:
    basis = (basis or "").strip()
    if len(basis) < 5:
        raise StageError(
            "rights not confirmed: pass --basis \"<why you may reuse this footage>\" "
            "(e.g. 'my own channel' or 'streamer X clipping program, link')")
    rec = {
        "url": url,
        "creator": "",  # filled from video metadata by the probe stage
        "date_collected": time.strftime("%Y-%m-%d"),
        "permission_basis": basis,
        "program_rules": {"required_credit": credit, "allowed_platforms": platforms},
    }
    atomic_write(job.dir / "rights.json", json.dumps(rec, indent=2))
    return ["rights.json"]


def verify(job: Job) -> bool:
    return (job.dir / "rights.json").exists()
