"""Record a live YouTube stream window around its busiest chat moment.
YouTube keeps only ~15 min of live HLS, so history can't be fetched; we watch chat, then record
the rolling window that contains the peak. Only message timestamps are kept (no names/text)."""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

PEAK_WAIT_S = 420          # wait this long after a peak so the window can centre on it
MARGIN_S = 60              # never start within this of the playlist start (segments expire)
FORMAT = "bv*[height<=720]+ba/b[height<=720]"


# --- chat ----------------------------------------------------------------------------
def parse_chat_lines(lines, seen: set) -> list[float]:
    """Epoch seconds of new text messages (dedupes by message id; text/authors are discarded)."""
    out = []
    for line in lines:
        try:
            j = json.loads(line)
        except ValueError:
            continue
        for a in (j.get("replayChatItemAction") or {}).get("actions", []):
            r = (a.get("addChatItemAction") or {}).get("item", {}).get("liveChatTextMessageRenderer")
            if not r or r.get("id") in seen:
                continue
            seen.add(r.get("id"))
            try:
                out.append(int(r["timestampUsec"]) / 1e6)
            except (KeyError, ValueError):
                pass
    return out


class ChatMonitor:
    def __init__(self, url: str, work: Path):
        self.url, self.work = url, work
        self.times: list[float] = []
        self.seen: set = set()
        self.pos = 0
        self.proc = None

    def start(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--skip-download", "--write-subs",
             "--sub-langs", "live_chat", "-o", "chat.%(ext)s", self.url],
            cwd=self.work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _file(self):
        for n in ("chat.live_chat.json.part", "chat.live_chat.json"):
            if (self.work / n).exists():
                return self.work / n
        return None

    def poll(self) -> int:
        f = self._file()
        if not f:
            return 0
        with open(f, "rb") as fh:
            fh.seek(self.pos)
            data = fh.read()
        cut = data.rfind(b"\n")
        if cut < 0:
            return 0
        self.pos += cut + 1
        new = parse_chat_lines(data[:cut].decode("utf-8", "ignore").splitlines(), self.seen)
        self.times.extend(new)
        return len(new)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.poll()
        for p in self.work.glob("chat.live_chat.json*"):      # raw files hold names/text: remove
            p.unlink(missing_ok=True)


def busiest(times: list[float], win: float = 60.0, step: float = 15.0, min_count: int = 5):
    """(center_epoch, count) of the densest `win`-second span, or None if too quiet."""
    if not times:
        return None
    ts = sorted(times)
    best, t = None, ts[0]
    import bisect
    while t <= ts[-1]:
        n = bisect.bisect_left(ts, t + win) - bisect.bisect_left(ts, t)
        if best is None or n > best[1]:
            best = (t + win / 2, n)
        t += step
    return best if best and best[1] >= min_count else None


def choose_window(peak: float | None, edge: float, pl_start: float, width: float = 840.0,
                  margin: float = MARGIN_S) -> tuple[float, float]:
    """[start, end] epoch window of up to `width` s centred on the peak, inside the playlist."""
    lo, hi = pl_start + margin, edge
    width = min(width, hi - lo)
    start = (peak - width / 2) if peak else hi - width
    start = max(lo, min(start, hi - width))
    return start, start + width


# --- HLS window -------------------------------------------------------------------------
def hls_info(url: str) -> dict:
    """Current live HLS playlist: epoch start/edge plus stream metadata."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as y:
        info = y.extract_info(url, download=False)
    f = next(f for f in info["formats"] if f.get("protocol") == "m3u8_native" and f.get("acodec") not in (None,))
    txt = urllib.request.urlopen(f["url"], timeout=30).read().decode()
    pdt = re.findall(r"#EXT-X-PROGRAM-DATE-TIME:(\S+)", txt)
    durs = [float(x) for x in re.findall(r"#EXTINF:([\d.]+)", txt)]
    if not pdt or not durs:
        raise RuntimeError("live playlist has no timing info")
    start = datetime.fromisoformat(pdt[0].replace("Z", "+00:00")).timestamp()
    return {"start": start, "edge": start + sum(durs), "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader") or "", "id": info.get("id"),
            "release_ts": info.get("release_timestamp") or info.get("timestamp")}


def record_window(url: str, out_dir: Path, start_epoch: float, dur: float, ffmpeg: str) -> float:
    """Download [start_epoch, start_epoch+dur] (real-time speed). Returns the actual start epoch."""
    h = hls_info(url)                                   # fresh playlist right before the download
    rel0 = max(MARGIN_S / 2, start_epoch - h["start"])
    rel1 = min(rel0 + dur, h["edge"] - h["start"])
    o = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "format": FORMAT,
         "merge_output_format": "mp4", "outtmpl": str(out_dir / "rec.%(ext)s"), "overwrites": True,
         "ffmpeg_location": ffmpeg, "force_keyframes_at_cuts": True,
         "download_ranges": download_range_func(None, [(rel0, rel1)])}
    with yt_dlp.YoutubeDL(o) as y:
        y.download([url])
    if not (out_dir / "rec.mp4").exists():
        raise RuntimeError("recording produced no file")
    return h["start"] + rel0


def now() -> float:
    return time.time()
