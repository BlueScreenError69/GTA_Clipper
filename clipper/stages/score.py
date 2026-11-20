"""score stage. SCORER_MODE:
  heuristic - no LLM; derived from signals only (honest, no invented breakdown)
  local     - Ollama (zero Claude tokens); JSON parsed, retried once, falls back per candidate
  hybrid    - Claude scores candidates_compact.json and writes scores.json; the stage validates it
Output schema: references/scoring.md. overall_score = sum(breakdown) - penalties, clamped 0..100.
"""
from __future__ import annotations
import json
import re
import urllib.error
import urllib.request

from ..manifest import atomic_write
from .common import Job, StageError, Waiting

MAX_PTS = {"hook": 20, "setup_payoff": 20, "self_contained": 20, "emotion_or_gameplay": 15,
           "clarity": 10, "novelty": 10, "pacing": 5}
EVENTS = {"reaction", "discovery", "funny_fail", "chase", "crash", "combat", "story", "other"}
SPOILERS = {"none", "minor", "major"}
FLAGS_OK = {"possible_music", "caption_check", "context_heavy", "spoiler_unchecked", "llm_failed"}

RUBRIC = """You score short-video clip candidates from a GTA 6 stream. Judge ONLY from the excerpt and signals; \
if the excerpt shows no payoff, do not assume one. Rubric (max points): hook 20 (grabs attention in first 1-2s), \
setup_payoff 20, self_contained 20 (no prior stream context needed), emotion_or_gameplay 15, clarity 10, \
novelty 10, pacing 5. penalties 0-25 (needs context 10-25, no payoff 10-20). spoiler_level: major = reveals \
story events/endings/deaths/mission outcomes. Hook title must be accurate, no clickbait lies, no spoilers. \
Add flag possible_music if little speech during high loudness; caption_check for mixed-language/low-confidence \
speech. Reply with ONE JSON object only: {"event_type": one of reaction|discovery|funny_fail|chase|crash|combat|story|other, \
"score_breakdown": {hook, setup_payoff, self_contained, emotion_or_gameplay, clarity, novelty, pacing}, \
"penalties": int, "hook_title": str, "spoiler_level": none|minor|major, "flags": [str], \
"one_line_reason": str, "reject": null or short reason string}."""


# --- shared helpers -------------------------------------------------------------------
def overall(breakdown: dict, penalties: float) -> int:
    return int(max(0, min(100, round(sum(breakdown.values()) - penalties))))


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def normalize_entry(raw: dict, cand: dict) -> dict:
    """Clamp an LLM answer into the schema; overall_score is always recomputed so it adds up."""
    bd_raw = raw.get("score_breakdown") or {}
    bd = {k: int(max(0, min(mx, round(float(bd_raw.get(k, 0) or 0))))) for k, mx in MAX_PTS.items()}
    pen = int(max(0, min(25, round(float(raw.get("penalties", 0) or 0)))))
    flags = sorted(set(cand.get("flags", [])) | {f for f in (raw.get("flags") or []) if f in FLAGS_OK})
    return {
        "id": cand["id"],
        "event_type": raw.get("event_type") if raw.get("event_type") in EVENTS else "other",
        "score_breakdown": bd, "penalties": pen, "overall_score": overall(bd, pen),
        "hook_title": str(raw.get("hook_title") or "").strip()[:100],
        "spoiler_level": raw.get("spoiler_level") if raw.get("spoiler_level") in SPOILERS else "minor",
        "flags": flags, "suggested_trim": {"start_s": None, "end_s": None},
        "one_line_reason": str(raw.get("one_line_reason") or "").strip()[:200],
    }


def heuristic_entry(cand: dict, extra_flags: tuple = ()) -> dict:
    sig = cand["signals"]
    bits = []
    if sig.get("loudness_z"):
        bits.append(f"loud peak (z={sig['loudness_z']})")
    if sig.get("speech_burst"):
        bits.append("speech burst")
    if sig.get("keywords"):
        bits.append("keywords: " + ", ".join(sig["keywords"]))
    kw = sig.get("keywords") or []
    title = f'Reaction: "{kw[0]}"' if kw else f"GTA 6 moment at {fmt_ts(cand['start_s'])}"
    return {
        "id": cand["id"], "event_type": "other", "score_breakdown": None, "penalties": 0,
        "overall_score": int(round(cand["heuristic_score"] * 100)),
        "hook_title": title, "spoiler_level": "minor",   # unknown without a reader; assume cautious
        "flags": sorted(set(cand.get("flags", [])) | {"spoiler_unchecked"} | set(extra_flags)),
        "suggested_trim": {"start_s": None, "end_s": None},
        "one_line_reason": ("heuristic only: " + "; ".join(bits)) if bits else "heuristic only",
    }


def validate_scores(data: dict, cands: dict) -> list[str]:
    """Problems with a scores.json (empty list = valid). Heuristic entries (no breakdown) pass."""
    errs = []
    ids = {c["id"] for c in cands["candidates"]}
    if data.get("job_id") != cands.get("job_id"):
        errs.append("job_id does not match candidates_compact.json")
    seen = set()
    for e in data.get("scored", []):
        i = e.get("id")
        if i not in ids:
            errs.append(f"{i}: unknown candidate id")
            continue
        if i in seen:
            errs.append(f"{i}: scored twice")
        seen.add(i)
        bd = e.get("score_breakdown")
        if bd is None:
            continue
        for k, mx in MAX_PTS.items():
            if not isinstance(bd.get(k), (int, float)) or not 0 <= bd[k] <= mx:
                errs.append(f"{i}: {k} must be 0..{mx}")
        if not errs or not any(x.startswith(f"{i}:") for x in errs):
            want = overall(bd, e.get("penalties", 0))
            if e.get("overall_score") != want:
                errs.append(f"{i}: overall_score {e.get('overall_score')} != sum-penalties {want}")
        if e.get("spoiler_level") not in SPOILERS:
            errs.append(f"{i}: spoiler_level must be none|minor|major")
        if e.get("event_type") not in EVENTS:
            errs.append(f"{i}: bad event_type")
        if not e.get("hook_title"):
            errs.append(f"{i}: missing hook_title")
    missing = ids - seen - {r.get("id") for r in data.get("rejected", [])}
    if missing:
        errs.append("not scored or rejected: " + ", ".join(sorted(missing)))
    return errs


def _write(job: Job, scored: list, rejected: list, mode: str) -> None:
    scored.sort(key=lambda e: -e["overall_score"])
    atomic_write(job.dir / "scores.json", json.dumps(
        {"job_id": job.job_id, "mode": mode, "scored": scored, "rejected": rejected}, indent=1))


# --- Ollama ---------------------------------------------------------------------------
def ollama_chat(host: str, model: str, messages: list) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False, "format": "json",
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]
    except urllib.error.URLError as e:
        raise StageError(f"cannot reach Ollama at {host} ({e.reason}). Install/start it or change SCORER_MODE.")


def parse_json_obj(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def _prompt(cand: dict, src: dict) -> list:
    payload = {k: cand[k] for k in ("id", "start_s", "end_s", "signals", "excerpt")}
    payload["duration_s"] = round(cand["end_s"] - cand["start_s"], 1)
    payload["source"] = {"creator": src.get("creator"), "title": src.get("title")}
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": json.dumps(payload)}]


def score_local(job: Job, cands: dict) -> tuple[list, list]:
    s = job.settings
    if not s.ollama_model:
        raise StageError("SCORER_MODE=local needs OLLAMA_MODEL in .env (and Ollama running)")
    log = job.logger("score")
    scored, rejected = [], []
    for c in cands["candidates"]:
        raw = None
        for attempt in (1, 2):                      # retry once on malformed output
            try:
                raw = parse_json_obj(ollama_chat(s.ollama_host, s.ollama_model, _prompt(c, cands["source"])))
                break
            except (ValueError, KeyError) as e:
                log.warning("%s attempt %d malformed: %s", c["id"], attempt, e)
        if raw is None:
            scored.append(heuristic_entry(c, ("llm_failed",)))
        elif raw.get("reject"):
            rejected.append({"id": c["id"], "reason": str(raw["reject"])[:120]})
        else:
            scored.append(normalize_entry(raw, c))
    return scored, rejected


# --- stage ----------------------------------------------------------------------------
def run(job: Job) -> list[str]:
    mode = job.settings.scorer_mode
    cands = job.read_json("candidates_compact.json")
    if mode == "hybrid":
        p = job.dir / "scores.json"
        if not p.exists():
            raise Waiting(
                f"hybrid mode: Claude must score {job.dir / 'candidates_compact.json'} per "
                f"references/scoring.md and write {p}, then run: resume {job.job_id}")
        try:
            data = json.loads(p.read_text())
        except ValueError:
            raise StageError("scores.json is not valid JSON; rewrite it")
        errs = validate_scores(data, cands)
        if errs:
            raise StageError("scores.json invalid: " + " | ".join(errs[:4]) + (" ..." if len(errs) > 4 else ""))
        return ["scores.json"]
    if mode == "local":
        scored, rejected = score_local(job, cands)
    else:
        scored, rejected = [heuristic_entry(c) for c in cands["candidates"]], []
    _write(job, scored, rejected, mode)
    return ["scores.json"]


def verify(job: Job) -> bool:
    p, c = job.dir / "scores.json", job.dir / "candidates_compact.json"
    try:
        return p.exists() and not validate_scores(json.loads(p.read_text()), json.loads(c.read_text()))
    except (ValueError, OSError):
        return False
