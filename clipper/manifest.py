"""Per-job stage manifest with atomic writes. Resume = first stage not 'done'."""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path

STAGES = ["rights", "probe", "fetch_audio", "transcribe", "signals",
          "candidates", "score", "fetch_sections", "render", "package"]
STATUSES = ("pending", "running", "done", "failed")


def atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def job_id_from_url(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError("not a recognizable YouTube URL")
    return m.group(1)


class Manifest:
    def __init__(self, job_dir: Path):
        self.dir = job_dir
        self.path = job_dir / "manifest.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"stages": {s: {"status": "pending"} for s in STAGES}}

    def save(self) -> None:
        atomic_write(self.path, json.dumps(self.data, indent=2))

    def get(self, stage: str) -> dict:
        return self.data["stages"][stage]

    def mark(self, stage: str, status: str, outputs: list[str] | None = None,
             error: str | None = None) -> None:
        assert status in STATUSES
        st = self.data["stages"][stage]
        now = time.time()
        if status == "running":
            st.update(status=status, started=now)
            st.pop("error", None)
        else:
            st.update(status=status, ended=now)
        if outputs is not None:
            st["outputs"] = outputs
        if error:
            st["error"] = error[:300]
        self.save()

    def next_stage(self) -> str | None:
        for s in STAGES:
            if self.get(s)["status"] != "done":
                return s
        return None

    def summary(self) -> list[str]:
        out = []
        for s in STAGES:
            st = self.get(s)
            line = f"{s:<15}{st['status']}"
            if st.get("error"):
                line += f"  ({st['error'][:80]})"
            out.append(line)
        return out
