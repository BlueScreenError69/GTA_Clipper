"""package: per-clip review folder + review/summary.md. Ends the pipeline; never uploads."""
from __future__ import annotations
import json
import os
import subprocess

from .. import diskguard
from ..manifest import atomic_write
from . import render as R
from . import sections as SEC
from .common import Job, StageError
from .score import fmt_ts

NOTE = ("Review every clip before posting. Source permission does not guarantee platform "
        "reused-content or copyright treatment. Nothing here is uploaded or scheduled.")
FLAG_HELP = {
    "possible_music": "check for copyrighted music",
    "caption_check": "check captions (mixed-language or low-confidence speech)",
    "layout_review": "check 9:16 layout/crop",
    "spoiler_check": "story spoiler: confirm the hook gives nothing away",
    "spoiler_unchecked": "spoilers not reviewed (heuristic scoring)",
    "credit_unconfirmed": "no required credit in program rules: confirm what is needed",
    "context_heavy": "needs stream context to make sense",
    "llm_failed": "local model failed; heuristic score used",
}


def make_caption(hook: str, credit: str, hashtags: list[str]) -> str:
    return "\n".join(p for p in (hook.strip(), "", credit.strip(), " ".join(hashtags)) if p is not None).strip() + "\n"


def clip_flags(score: dict, report: dict, credit_ok: bool) -> list[str]:
    f = set(score.get("flags", []))
    if report.get("layout_review"):
        f.add("layout_review")
    if score.get("spoiler_level") == "major":
        f.add("spoiler_check")
    if not credit_ok:
        f.add("credit_unconfirmed")
    return sorted(f)


def md(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def build_summary(job_id: str, meta: dict, rights: dict, mode: str, counts: dict, rows: list[dict],
                  folder: str, max_rows: int = 18) -> str:
    L = [f"# Review: {md(meta.get('title') or job_id)} ({job_id})",
         f"Source: {rights['url']} | Creator: {md(rights.get('creator') or 'unknown')} | "
         f"Length: {fmt_ts(meta['duration_s'])}",
         f"Rights basis: {md(rights['permission_basis'])} | Credit: {md(rows[0]['credit']) if rows else '-'}"
         f" | Platforms: {md(rights['program_rules'].get('allowed_platforms') or 'not specified')}",
         f"Candidates {counts['candidates']} / scored {counts['scored']} / rendered {len(rows)} "
         f"(scorer: {mode}). Folder: {folder}", "",
         "| ID | Source time | Score | Hook | Reason | Flags |", "|---|---|---|---|---|---|"]
    shown = sorted(rows, key=lambda r: -r["score"])[:max_rows]
    for r in shown:
        L.append(f"| {r['id']} | {fmt_ts(r['start_s'])}-{fmt_ts(r['end_s'])} | {r['score']} | "
                 f"{md(r['hook'])[:60]} | {md(r['reason'])[:70]} | {', '.join(r['flags']) or '-'} |")
    if len(rows) > len(shown):
        L.append(f"| ... | {len(rows) - len(shown)} more clips, see review/<id>/clip.json | | | | |")
    by_flag: dict[str, list[str]] = {}
    for r in rows:
        for f in r["flags"]:
            by_flag.setdefault(f, []).append(r["id"])
    L.append("")
    if by_flag:
        L.append("Needs human review:")
        for f, ids in sorted(by_flag.items()):
            L.append(f"- {f} ({', '.join(ids)}): {FLAG_HELP.get(f, f)}")
    else:
        L.append("No automatic review flags (still watch every clip).")
    L += ["", f"Next: watch each clip, edit hooks/captions in review/<id>/caption.txt. {NOTE}"]
    return "\n".join(L) + "\n"


def _thumb(s, clip, dst, dur) -> None:
    tmp = dst.with_name(dst.stem + ".tmp.jpg")
    t = max(0.5, min(dur * 0.3, dur - 0.5))
    r = subprocess.run([s.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}",
                        "-i", str(clip), "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise StageError(f"thumbnail failed: {r.stderr.strip()[-120:]}")
    tmp.replace(dst)


def run(job: Job) -> list[str]:
    s = job.settings
    log = job.logger("package")
    pcfg = s.yaml.get("package") or {}
    ids = SEC.selected_ids(job)
    bad = [i for i in ids if not R.clip_ok(job, i)]
    if bad:
        raise StageError(f"clips missing or failed verification: {', '.join(bad)}; run render first")
    diskguard.check(s, 5 * 1024 ** 2 * len(ids), "package")

    meta, rights = job.read_json("meta.json"), job.read_json("rights.json")
    scores = {e["id"]: e for e in job.read_json("scores.json")["scored"]}
    cands = job.read_json("candidates_compact.json")["candidates"]
    cand = {c["id"]: c for c in cands}
    words = job.read_json("transcript.json")["words"]
    report = job.read_json("render_report.json")
    mode = job.read_json("scores.json").get("mode", s.scorer_mode)
    required = (rights.get("program_rules") or {}).get("required_credit", "").strip()
    credit = required or str(pcfg.get("default_credit", "Clip from {creator}")).format(
        creator=rights.get("creator") or "the creator")
    tags = list(pcfg.get("hashtags", ["#GTA6", "#Shorts"]))

    off = float(meta.get("stream_offset_s") or 0)   # live recordings: window start within the stream
    rows, outs = [], []
    for cid in ids:
        start, end = SEC.clip_window(job, cid)
        clip, dur = R.clip_path(job, cid)
        d = job.dir / "review" / cid
        d.mkdir(parents=True, exist_ok=True)
        sc = scores[cid]
        flags = clip_flags(sc, report.get(cid, {}), bool(required))
        text = " ".join(w["w"] for w in words if start <= (w["start"] + w["end"]) / 2 < end)
        _thumb(s, clip, d / "thumb.jpg", dur)
        atomic_write(d / "transcript.txt", text + "\n")
        atomic_write(d / "caption.txt", make_caption(sc.get("hook_title") or "", credit, tags))
        link = d / "clip.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(clip, d), link)
        info = {"id": cid, "clip_file": str(clip), "source_url": rights["url"], "source_start_s": start + off,
                "source_end_s": end + off, "source_timestamps": f"{fmt_ts(start + off)}-{fmt_ts(end + off)}",
                "overall_score": sc["overall_score"], "score_breakdown": sc.get("score_breakdown"),
                "penalties": sc.get("penalties", 0), "event_type": sc.get("event_type"),
                "hook_title": sc.get("hook_title"), "spoiler_level": sc.get("spoiler_level"),
                "one_line_reason": sc.get("one_line_reason"), "flags": flags,
                "signals": cand[cid]["signals"], "layout": report.get(cid, {}).get("layout"),
                "caption_style": report.get(cid, {}).get("style"), "credit": credit,
                "rights": {"creator": rights.get("creator"), "basis": rights["permission_basis"],
                           "program_rules": rights.get("program_rules")}}
        atomic_write(d / "clip.json", json.dumps(info, indent=1))
        rows.append({"id": cid, "start_s": start + off, "end_s": end + off, "score": sc["overall_score"],
                     "hook": sc.get("hook_title") or "", "reason": sc.get("one_line_reason") or "",
                     "flags": flags, "credit": credit})
        outs.append(f"review/{cid}/clip.json")
        log.info("%s packaged (%s)", cid, ",".join(flags) or "no flags")
    summary = build_summary(job.job_id, meta, rights, mode,
                            {"candidates": len(cands), "scored": len(scores)}, rows,
                            str(job.dir / "review"), int(pcfg.get("max_table_rows", 18)))
    atomic_write(job.dir / "review" / "summary.md", summary)
    return outs + ["review/summary.md"]


def verify(job: Job) -> bool:
    try:
        d = job.dir / "review"
        return (d / "summary.md").exists() and all(
            (d / i / "clip.json").exists() and (d / i / "thumb.jpg").exists() for i in SEC.selected_ids(job))
    except (StageError, OSError):
        return False
