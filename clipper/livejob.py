"""`clipper live`: watch chat, record the window around the busiest moment, analyse it."""
from __future__ import annotations
import json
import subprocess
import time

from . import diskguard, live
from .manifest import atomic_write, job_id_from_url
from .stages import (candidates as cand_stage, fetch, rights as rights_stage, score as score_stage,
                     signals as signals_stage, transcribe as transcribe_stage)
from .stages.common import Job, StageError, probe_duration, run_stage


def build_job(s, job: Job, url: str, h: dict, rec_start: float, chat_times: list[float]) -> None:
    """Turn rec.mp4 + chat timestamps into a normal analysable job (stages up to fetch_audio done)."""
    rec = job.dir / "rec.mp4"
    dur = probe_duration(s, rec)
    atomic_write(job.dir / "meta.json", json.dumps({
        "id": h["id"], "title": h["title"], "channel": h["channel"], "duration_s": dur,
        "live_status": "live_recording", "local_file": "rec.mp4",
        "stream_offset_s": (rec_start - h["release_ts"]) if h.get("release_ts") else 0,
        "recorded_epoch_start": rec_start}))
    rights = job.read_json("rights.json")
    rights["creator"] = h["channel"]
    atomic_write(job.dir / "rights.json", json.dumps(rights, indent=2))
    rel = sorted(t - rec_start for t in chat_times if 0 <= t - rec_start <= dur)
    atomic_write(job.dir / "chat.json", json.dumps({"times": [round(t, 2) for t in rel]}))
    tmp = job.dir / "audio.tmp.wav"
    r = subprocess.run([s.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(rec), "-vn",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise StageError(f"audio extraction failed: {r.stderr[-120:]}")
    tmp.replace(job.dir / "audio.wav")
    job.manifest.mark("probe", "done", outputs=["meta.json"])
    job.manifest.mark("fetch_audio", "done", outputs=["audio.wav"])


def analyze(job: Job) -> None:
    for name, fn, ver in (("transcribe", transcribe_stage.transcribe, transcribe_stage.verify),
                          ("signals", signals_stage.run, signals_stage.verify),
                          ("candidates", cand_stage.run, cand_stage.verify),
                          ("score", score_stage.run, score_stage.verify)):
        print(f"{name}: {run_stage(job, name, fn, ver)}", flush=True)


def run_live(s, url: str, basis: str, credit: str, watch_min: float, window_min: float,
             poll_s: float = 15.0) -> Job:
    vid = job_id_from_url(url)
    job = Job.open(s, f"L{vid}_{time.strftime('%m%d%H%M')}")
    run_stage(job, "rights", lambda j: rights_stage.write_rights(j, url, basis, credit), rights_stage.verify)
    diskguard.check(s, int(1.5 * diskguard.GB), "live recording")
    live.hls_info(url)            # fail early if the stream is not live / has no HLS window
    print(f"job {job.job_id}: watching chat for up to {watch_min:g} min", flush=True)
    mon = live.ChatMonitor(url, job.dir)
    mon.start()
    try:
        t0 = live.now()
        deadline = t0 + watch_min * 60
        peak, last_log = None, 0.0
        while live.now() < deadline:
            time.sleep(poll_s)
            mon.poll()
            p = live.busiest(mon.times)
            if p and (peak is None or p[1] > peak[1]):
                peak = p
            if live.now() - last_log > 60:
                last_log = live.now()
                print(f"  chat msgs seen {len(mon.times)}, peak so far "
                      f"{peak[1] if peak else 0} msgs/min", flush=True)
            if peak and live.now() >= peak[0] + live.PEAK_WAIT_S:
                break
        h = live.hls_info(url)
        start, end = live.choose_window(peak[0] if peak else None, h["edge"], h["start"],
                                        window_min * 60)
        print(f"recording {int(end - start)} s (real-time) around chat peak "
              f"{peak[1] if peak else 'n/a'} msgs/min ...", flush=True)
        rec_start = live.record_window(url, job.dir, start, end - start, s.ffmpeg)
    finally:
        mon.stop()
    build_job(s, job, url, h, rec_start, mon.times)
    print("recorded; analysing", flush=True)
    analyze(job)
    return job
