"""probe (YouTube metadata) and fetch_audio (audio-only download -> 16 kHz mono WAV)."""
from __future__ import annotations
import json
import re
import subprocess
import time

import yt_dlp

from .. import diskguard
from ..manifest import atomic_write
from .common import Job, StageError, probe_duration

ACCESS_PATTERNS = re.compile(
    r"sign in|private video|members[- ]only|membership|age[- ]restrict|confirm your age|"
    r"not available in your country|geo|blocked it|premium|drm|login required|"
    r"video unavailable|has been removed|copyright", re.I)
NETWORK_PATTERNS = re.compile(
    r"timed out|timeout|connection|temporar|HTTP Error 5\d\d|reset by peer|unreachable|"
    r"incomplete read|ssl", re.I)
RETRIES = 3


class AccessError(StageError):
    pass


def classify(msg: str) -> str:
    """'access' (never retry) | 'network' (retry) | 'other'."""
    if ACCESS_PATTERNS.search(msg):
        return "access"
    if NETWORK_PATTERNS.search(msg):
        return "network"
    return "other"


# --- thin yt-dlp wrappers (monkeypatched in tests) ---------------------------------
def extract_info(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as y:
        return y.extract_info(url, download=False)


def download_audio(url: str, outtmpl: str) -> None:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
            "format": "bestaudio/best", "outtmpl": outtmpl, "overwrites": True}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])


def with_retries(fn, log, sleep=time.sleep):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # yt_dlp.DownloadError / OSError
            kind = classify(str(e))
            if kind == "access":
                raise AccessError(
                    f"access denied or unavailable ({str(e)[:120]}). Not retrying and not bypassing it; "
                    "provide an authorized local file instead.")
            if kind != "network" or attempt == RETRIES:
                raise StageError(f"download failed: {str(e)[:160]}")
            log.warning("network error (attempt %d/%d): %s", attempt, RETRIES, e)
            sleep(2 ** attempt)


# --- stages -------------------------------------------------------------------------
def probe(job: Job) -> list[str]:
    rights = job.read_json("rights.json")  # gate: raises if missing
    log = job.logger("probe")
    info = with_retries(lambda: extract_info(rights["url"]), log)
    status = info.get("live_status")
    if status in ("is_live", "is_upcoming"):
        raise StageError(f"stream is {status}; wait until it ends and the VOD is available")
    if info.get("availability") in ("needs_auth", "subscriber_only", "premium_only", "private"):
        raise AccessError(f"video availability is '{info['availability']}'; provide an authorized local file")
    dur = info.get("duration")
    if not dur or dur <= 0:
        raise StageError("video has no duration (still processing?)")
    meta = {"id": info.get("id"), "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader") or "",
            "duration_s": float(dur), "live_status": status,
            "upload_date": info.get("upload_date")}
    atomic_write(job.dir / "meta.json", json.dumps(meta, indent=2))
    rights["creator"] = meta["channel"]
    atomic_write(job.dir / "rights.json", json.dumps(rights, indent=2))
    return ["meta.json"]


def fetch_audio(job: Job) -> list[str]:
    s = job.settings
    meta = job.read_json("meta.json")
    rights = job.read_json("rights.json")
    log = job.logger("fetch_audio")
    diskguard.check(s, diskguard.estimate_audio_bytes(meta["duration_s"]), "fetch_audio")

    for old in job.dir.glob("audio_src.*"):
        old.unlink()
    with_retries(lambda: download_audio(rights["url"], str(job.dir / "audio_src.%(ext)s")), log)
    srcs = [p for p in job.dir.glob("audio_src.*") if not p.name.endswith((".part", ".ytdl"))]
    if not srcs:
        raise StageError("download produced no audio file")
    src = srcs[0]

    tmp = job.dir / "audio.tmp.wav"
    r = subprocess.run([s.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise StageError(f"ffmpeg audio conversion failed: {r.stderr[-150:]}")
    got = probe_duration(s, tmp)
    if abs(got - meta["duration_s"]) > 5:
        tmp.unlink(missing_ok=True)
        raise StageError(f"audio duration {got:.1f}s != video {meta['duration_s']:.1f}s")
    tmp.replace(job.dir / "audio.wav")
    log.info("audio.wav %.1fs", got)
    if not s.keep_temp_files:
        src.unlink(missing_ok=True)
    return ["audio.wav"]


def verify_audio(job: Job) -> bool:
    w = job.dir / "audio.wav"
    if not w.exists() or not (job.dir / "meta.json").exists():
        return False
    try:
        return abs(probe_duration(job.settings, w) - job.read_json("meta.json")["duration_s"]) <= 5
    except StageError:
        return False
