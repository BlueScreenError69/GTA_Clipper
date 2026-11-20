"""Chunked faster-whisper transcription with per-chunk cache and word timestamps.

Chunk i decodes [i*C - O, (i+1)*C + O] (O = overlap) for context, but *owns* only words whose
midpoint lies in [i*C, (i+1)*C). Each word is therefore kept exactly once at chunk seams.
"""
from __future__ import annotations
import json
import os
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from .. import diskguard
from ..manifest import atomic_write
from .common import Job, StageError

OVERLAP_S = 2.0
LOW_CONF = 0.5
MODEL_DL_ESTIMATE = {"tiny": 0.1, "base": 0.2, "small": 0.6, "medium": 1.6, "large-v3": 3.2}


# --- pure helpers (unit-tested) -------------------------------------------------------
def plan_chunks(duration_s: float, chunk_s: int, overlap_s: float = OVERLAP_S) -> list[dict]:
    n = max(1, int(np.ceil(duration_s / chunk_s)))
    out = []
    for i in range(n):
        own_start, own_end = i * chunk_s, min((i + 1) * chunk_s, duration_s)
        out.append({"i": i, "own_start": own_start, "own_end": own_end,
                    "dec_start": max(0.0, own_start - overlap_s),
                    "dec_end": min(duration_s, own_end + overlap_s)})
    return out


def own_words(words: list[dict], own_start: float, own_end: float, last: bool = False) -> list[dict]:
    """Keep words whose midpoint is in [own_start, own_end) (last chunk includes the end)."""
    keep = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if own_start <= mid < own_end or (last and mid >= own_start):
            keep.append(w)
    return keep


def merge_chunks(chunks: list[dict]) -> list[dict]:
    words: list[dict] = []
    for c in sorted(chunks, key=lambda c: c["i"]):
        words.extend(c["words"])
    words.sort(key=lambda w: w["start"])
    return words


def read_wav_slice(path: Path, start_s: float, end_s: float) -> np.ndarray:
    """float32 mono samples for [start_s, end_s) from a 16 kHz mono pcm_s16le WAV (no full load)."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise StageError("audio.wav must be 16 kHz mono 16-bit")
        w.setpos(int(start_s * 16000))
        n = max(0, int((end_s - start_s) * 16000))
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# --- model wrapper (monkeypatched in tests) --------------------------------------------
def load_model(s):
    from faster_whisper import WhisperModel  # heavy import, only when a chunk is missing
    root = os.environ.get("HF_HOME", "")
    return WhisperModel(s.whisper_model, device=s.whisper_device,
                        compute_type=s.whisper_compute_type,
                        download_root=str(Path(root) / "whisper") if root else None)


def _model_cached(s) -> bool:
    root = Path(os.environ.get("HF_HOME", s.data_dir / "models")) / "whisper"
    return root.exists() and any(root.glob(f"*faster-whisper-{s.whisper_model}*"))


def decode_chunk(model, audio: np.ndarray, offset: float, language: str | None) -> dict:
    segs, info = model.transcribe(audio, language=language, word_timestamps=True,
                                  vad_filter=True, condition_on_previous_text=False,
                                  beam_size=5)
    words = []
    for seg in segs:
        for w in (seg.words or []):
            words.append({"w": w.word.strip(), "start": round(w.start + offset, 3),
                          "end": round(w.end + offset, 3),
                          "p": round(float(w.probability), 3)})
    return {"language": info.language, "lang_prob": round(float(info.language_probability), 3),
            "words": [w for w in words if w["w"]]}


# --- stage ------------------------------------------------------------------------------
def transcribe(job: Job) -> list[str]:
    s = job.settings
    log = job.logger("transcribe")
    meta = job.read_json("meta.json")
    wav = job.dir / "audio.wav"
    if not wav.exists():
        raise StageError("audio.wav missing; run fetch_audio first")
    lang = None if s.transcribe_language in ("", "auto") else s.transcribe_language
    plan = plan_chunks(meta["duration_s"], s.chunk_seconds)
    cache = job.dir / "transcript_chunks"
    cache.mkdir(exist_ok=True)

    def cpath(i): return cache / f"chunk_{i:04d}.json"

    todo = [c for c in plan if not cpath(c["i"]).exists()]
    if todo:
        diskguard.check(s, int((0 if _model_cached(s) else MODEL_DL_ESTIMATE.get(s.whisper_model, 1.0)) * diskguard.GB)
                        + 20 * 1024 ** 2, "transcribe")
        model = load_model(s)
        for c in todo:
            audio = read_wav_slice(wav, c["dec_start"], c["dec_end"])
            res = decode_chunk(model, audio, c["dec_start"], lang)
            res["words"] = own_words(res["words"], c["own_start"], c["own_end"],
                                     last=c["i"] == len(plan) - 1)
            res["i"] = c["i"]
            atomic_write(cpath(c["i"]), json.dumps(res))
            log.info("chunk %d/%d: %d words, lang=%s", c["i"] + 1, len(plan),
                     len(res["words"]), res["language"])

    chunks = [json.loads(cpath(c["i"]).read_text()) for c in plan]
    words = merge_chunks(chunks)
    langs = Counter(c["language"] for c in chunks if c["words"])
    language = langs.most_common(1)[0][0] if langs else (lang or "unknown")
    avg_p = float(np.mean([w["p"] for w in words])) if words else 0.0
    flags = []
    if language == "hi" or len(langs) > 1:
        flags.append("caption_check")  # Hindi/Hinglish or mixed language
    if words and avg_p < LOW_CONF:
        flags.append("caption_check")
    out = {"language": language, "languages_seen": dict(langs), "word_count": len(words),
           "avg_word_prob": round(avg_p, 3), "flags": sorted(set(flags)), "words": words}
    atomic_write(job.dir / "transcript.json", json.dumps(out))
    return ["transcript.json"]


def verify(job: Job) -> bool:
    p = job.dir / "transcript.json"
    if not p.exists():
        return False
    try:
        return "words" in json.loads(p.read_text())
    except ValueError:
        return False
