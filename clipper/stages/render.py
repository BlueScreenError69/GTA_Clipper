"""render: cut each selected clip to 9:16, burn word-by-word captions (libass), loudnorm, verify."""
from __future__ import annotations
import json
import shutil
import subprocess

from .. import captions, diskguard, layout
from ..manifest import atomic_write
from . import sections as SEC
from .common import Job, StageError


def clip_filename(job_id: str, cid: str, start: float, end: float) -> str:
    return f"{job_id}_{cid}_{int(round(start))}-{int(round(end))}.mp4"


def probe_media(settings, path) -> dict:
    r = subprocess.run([settings.ffprobe, "-v", "error", "-show_entries",
                        "stream=codec_type,width,height:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        st = j.get("streams", [])
        v = next((x for x in st if x["codec_type"] == "video"), None)
        return {"duration": float(j["format"]["duration"]), "has_video": v is not None,
                "has_audio": any(x["codec_type"] == "audio" for x in st),
                "w": v and v.get("width"), "h": v and v.get("height")}
    except (ValueError, KeyError):
        return {"duration": 0.0, "has_video": False, "has_audio": False, "w": None, "h": None}


def verify_clip(settings, path, want_dur: float) -> str | None:
    """None if the clip is good, else the reason it is not."""
    m = probe_media(settings, path)
    if not m["has_video"] or not m["has_audio"]:
        return "missing video or audio stream"
    if (m["w"], m["h"]) != (settings.width, settings.height):
        return f"resolution {m['w']}x{m['h']} != {settings.width}x{settings.height}"
    if abs(m["duration"] - want_dur) > 0.5:
        return f"duration {m['duration']:.2f}s != requested {want_dur:.2f}s"
    return None


def clip_path(job: Job, cid: str):
    a, b = SEC.clip_window(job, cid)
    return job.dir / "clips" / clip_filename(job.job_id, cid, a, b), b - a


def layout_matches(job: Job, cid: str, forced: str | None) -> bool:
    """Existing clip is acceptable unless a layout is forced and it was rendered with another."""
    if not forced:
        return True
    p = job.dir / "render_report.json"
    try:
        return json.loads(p.read_text()).get(cid, {}).get("layout") == forced
    except (OSError, ValueError):
        return False


def clip_ok(job: Job, cid: str) -> bool:
    try:
        p, d = clip_path(job, cid)
        return p.exists() and verify_clip(job.settings, p, d) is None
    except (StopIteration, OSError, KeyError):
        return False


def render_clip(job: Job, cid: str, words: list[dict], log) -> dict:
    s = job.settings
    rcfg = s.yaml.get("render") or {}
    ccfg = s.yaml.get("captions") or {}
    out, dur = clip_path(job, cid)
    forced = SEC.forced_layout(job)
    if out.exists() and verify_clip(s, out, dur) is None and layout_matches(job, cid, forced):
        return {"status": "ok", "file": out.name, "skipped": True}
    start, end = SEC.clip_window(job, cid)
    sec = job.dir / "sections" / f"{cid}.mp4"
    info = json.loads((job.dir / "sections" / f"{cid}.json").read_text())
    diskguard.check(s, diskguard.estimate_clip_bytes(dur), f"render {cid}")

    if forced:
        lay = {"kind": forced, "rect": None, "review": False, "src": None, "info": "forced by user"}
    else:
        lay = layout.choose_layout_at(sec, info["offset"], dur)
    style = captions.pick_style(ccfg.get("styles"), cid)
    top_frac = float(rcfg.get("top_height_frac", 0.34))
    y = None
    if lay["kind"] == "facecam_top":
        y = max(style["y_frac"], layout.top_height(s.height, top_frac) / s.height + 0.07)
    work = job.dir / "render_tmp" / cid
    work.mkdir(parents=True, exist_ok=True)
    atomic_write(work / "captions.ass", captions.build_ass(
        words, start, end, style, s.width, s.height, y, int(ccfg.get("words_per_line", 3))))
    tmp = out.parent / (out.stem + ".part.mp4")
    out.parent.mkdir(exist_ok=True)
    cmd = [s.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{info['offset']:.3f}",
           "-i", str(sec), "-t", f"{dur:.3f}",
           "-filter_complex", layout.build_filter(lay, s.width, s.height, top_frac),
           "-map", "[v]", "-map", "0:a:0", "-af", f"loudnorm={rcfg.get('loudnorm', 'I=-16:TP=-1.5:LRA=11')}",
           "-c:v", "libx264", "-preset", "fast", "-crf", str(rcfg.get("crf", 20)), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    base = {"layout": lay["kind"], "layout_review": lay["review"], "style": style["name"],
            "layout_info": lay["info"]}
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return {**base, "status": "failed", "reason": f"ffmpeg: {r.stderr.strip()[-160:]}"}
    bad = verify_clip(s, tmp, dur)
    if bad:
        tmp.unlink(missing_ok=True)
        return {**base, "status": "failed", "reason": bad}
    tmp.replace(out)
    log.info("%s -> %s (%s)", cid, out.name, lay["kind"])
    return {**base, "status": "ok", "file": out.name}


def run(job: Job) -> list[str]:
    s = job.settings
    log = job.logger("render")
    ids = SEC.selected_ids(job)
    words = json.loads((job.dir / "transcript.json").read_text())["words"]
    report_p = job.dir / "render_report.json"
    report = json.loads(report_p.read_text()) if report_p.exists() else {}
    for cid in ids:
        report[cid] = render_clip(job, cid, words, log)
        if report[cid]["status"] == "ok" and not s.keep_temp_files:
            for p in (job.dir / "sections").glob(f"{cid}.*"):
                p.unlink(missing_ok=True)
            shutil.rmtree(job.dir / "render_tmp" / cid, ignore_errors=True)
    atomic_write(report_p, json.dumps(report, indent=1))
    bad = {i: report[i] for i in ids if report[i]["status"] != "ok"}
    if len(bad) == len(ids):
        raise StageError("every clip failed: " + "; ".join(f"{i}: {v['reason']}" for i, v in bad.items())[:300])
    if bad:
        log.warning("failed clips: %s", bad)
        raise StageError("some clips failed verification: " + "; ".join(
            f"{i}: {v['reason']}" for i, v in bad.items())[:300])
    return [report[i]["file"] for i in ids]


def verify(job: Job) -> bool:
    try:
        f = SEC.forced_layout(job)
        return all(clip_ok(job, i) and layout_matches(job, i, f) for i in SEC.selected_ids(job))
    except StageError:
        return False
