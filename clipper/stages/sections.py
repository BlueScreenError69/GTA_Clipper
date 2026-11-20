"""Pick clips to render and download only those video sections (yt-dlp --download-sections).
Fallback when sections fail for a non-access reason: download the full video once and cut locally."""
from __future__ import annotations
import json
import subprocess

import yt_dlp

from .. import diskguard
from ..manifest import atomic_write
from . import fetch as F
from .common import Job, StageError, probe_duration

DEFAULT_PAD = 3.0


# --- selection ------------------------------------------------------------------------
def choose_ids(job: Job, ids: list[str] | None = None, top: int | None = None) -> list[str]:
    scored = sorted(job.read_json("scores.json")["scored"], key=lambda e: -e["overall_score"])
    have = {e["id"] for e in scored}
    if ids:
        bad = [i for i in ids if i not in have]
        if bad:
            raise StageError(f"not scored (or rejected): {', '.join(bad)}")
        return list(dict.fromkeys(ids))
    if not top or top < 1:
        raise StageError("pass --ids c01,c02 or --top N (nothing is rendered without approval)")
    return [e["id"] for e in scored[:top]]


LAYOUTS = ("auto", "blur_fit", "center_crop")


def save_selection(job: Job, ids: list[str], layout: str = "auto") -> None:
    if layout not in LAYOUTS:
        raise StageError(f"layout must be one of {', '.join(LAYOUTS)}")
    atomic_write(job.dir / "render_selection.json", json.dumps({"ids": ids, "layout": layout}))


def forced_layout(job: Job) -> str | None:
    """'blur_fit' / 'center_crop' when the user forced one, else None (auto-detect)."""
    p = job.dir / "render_selection.json"
    lay = json.loads(p.read_text()).get("layout", "auto") if p.exists() else "auto"
    return None if lay == "auto" else lay


def selected_ids(job: Job) -> list[str]:
    p = job.dir / "render_selection.json"
    if not p.exists():
        raise StageError("no render selection; use: render <job> --ids ... | --top N")
    return json.loads(p.read_text())["ids"]


def clip_window(job: Job, cid: str) -> tuple[float, float]:
    """Candidate window, overridden by scores.json suggested_trim when present."""
    c = next(c for c in job.read_json("candidates_compact.json")["candidates"] if c["id"] == cid)
    start, end = c["start_s"], c["end_s"]
    e = next(e for e in job.read_json("scores.json")["scored"] if e["id"] == cid)
    t = e.get("suggested_trim") or {}
    s2 = t.get("start_s") if t.get("start_s") is not None else start
    e2 = t.get("end_s") if t.get("end_s") is not None else end
    if 0 <= s2 < e2:
        start, end = s2, e2
    return float(start), float(end)


# --- yt-dlp wrappers (monkeypatched in tests) -------------------------------------------
def _opts(ffmpeg: str, outtmpl: str) -> dict:
    return {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b", "merge_output_format": "mp4",
            "outtmpl": outtmpl, "overwrites": True, "ffmpeg_location": ffmpeg}


def download_section(url: str, outtmpl: str, start: float, end: float, ffmpeg: str) -> None:
    o = _opts(ffmpeg, outtmpl)
    o["download_ranges"] = yt_dlp.utils.download_range_func(None, [(start, end)])
    o["force_keyframes_at_cuts"] = True            # accurate cut points
    with yt_dlp.YoutubeDL(o) as y:
        y.download([url])


def download_full(url: str, outtmpl: str, ffmpeg: str) -> None:
    with yt_dlp.YoutubeDL(_opts(ffmpeg, outtmpl)) as y:
        y.download([url])


def cut_local(s, src, out, start: float, dur: float) -> None:
    r = subprocess.run([s.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start}", "-i",
                        str(src), "-t", f"{dur}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-c:a", "aac", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise StageError(f"local cut failed: {r.stderr[-120:]}")


def _find(d, stem):
    c = [p for p in d.glob(f"{stem}.*") if p.suffix in (".mp4", ".mkv", ".webm") and ".f" not in p.stem[len(stem):]]
    return c[0] if c else None


# --- stage ----------------------------------------------------------------------------
def fetch_sections(job: Job) -> list[str]:
    s = job.settings
    log = job.logger("fetch_sections")
    ids = selected_ids(job)
    meta = job.read_json("meta.json")
    url = job.read_json("rights.json")["url"]
    pad = float((s.yaml.get("render") or {}).get("section_pad_s", DEFAULT_PAD))
    sec = job.dir / "sections"
    sec.mkdir(exist_ok=True)
    wins = {i: clip_window(job, i) for i in ids}
    todo = [i for i in ids if not _ok(job, i)]
    if not todo:
        return [f"sections/{i}.mp4" for i in ids]
    local = meta.get("local_file")            # live recordings: cut from the saved file, no network
    diskguard.check(s, diskguard.estimate_clip_bytes(sum(wins[i][1] - wins[i][0] + 2 * pad for i in todo)),
                    "fetch_sections")
    full = None
    for i in todo:
        a, b = wins[i]
        s0, s1 = max(0.0, a - pad), min(meta["duration_s"], b + pad)
        for old in sec.glob(f"{i}.*"):
            old.unlink()
        out = sec / f"{i}.mp4"
        if local:
            cut_local(s, job.dir / local, out, s0, s1 - s0)
            atomic_write(sec / f"{i}.json", json.dumps({"section_start": s0, "offset": a - s0,
                                                        "duration": b - a}))
            continue
        try:
            F.with_retries(lambda: download_section(url, str(sec / f"{i}.%(ext)s"), s0, s1, s.ffmpeg), log)
            got = _find(sec, i)
            if not got:
                raise StageError("section download produced no file")
            if got != out:
                got.rename(out)
        except F.AccessError:
            raise
        except StageError as e:
            log.warning("section %s failed (%s); falling back to full-video download", i, e)
            full = full or _download_full_once(job, url, meta, log)
            cut_local(s, full, out, s0, s1 - s0)
        d = probe_duration(s, out)
        if abs(d - (s1 - s0)) > 2.5:
            out.unlink(missing_ok=True)
            raise StageError(f"{i}: section is {d:.1f}s, expected {s1 - s0:.1f}s")
        atomic_write(sec / f"{i}.json", json.dumps({"section_start": s0, "offset": a - s0,
                                                    "duration": b - a}))
        log.info("%s: section %.1f-%.1f ok", i, s0, s1)
    if full and not s.keep_temp_files:
        full.unlink(missing_ok=True)
    return [f"sections/{i}.mp4" for i in ids]


def _download_full_once(job: Job, url: str, meta: dict, log):
    s = job.settings
    diskguard.check(s, int(meta["duration_s"] * 1.5 * 1024 ** 2), "fetch_sections(full video)")
    for old in job.dir.glob("video_full.*"):
        old.unlink()
    F.with_retries(lambda: download_full(url, str(job.dir / "video_full.%(ext)s"), s.ffmpeg), log)
    f = _find(job.dir, "video_full")
    if not f:
        raise StageError("full-video fallback produced no file")
    return f


def _ok(job: Job, cid: str) -> bool:
    return (job.dir / "sections" / f"{cid}.mp4").exists() and (job.dir / "sections" / f"{cid}.json").exists()


def verify(job: Job) -> bool:
    try:
        from . import render as R
        f = forced_layout(job)
        return all(_ok(job, i) or (R.clip_ok(job, i) and R.layout_matches(job, i, f)) for i in selected_ids(job))
    except (StageError, OSError):
        return False
