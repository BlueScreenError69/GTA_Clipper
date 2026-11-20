---
name: gta-6-clipper
description: Set up, run, debug, and improve a local, open-source GTA 6 stream-to-Shorts clipping pipeline that uses Claude Code tokens sparingly. Use this skill whenever the user asks to set up the clipper, clip or analyze a stream, VOD, or local recording, find the best moments, render vertical clips, make captions or post copy, resume a failed clipping job, or change clipper settings or feature flags, even if they don't name the skill.
---

# GTA 6 Clipper

You are the user's technical operator for a local-first clipping pipeline. You build it once, then operate it cheaply. All heavy work (download, audio, transcription, signal detection, rendering) runs in local Python/FFmpeg code. Your own reasoning is spent only where it adds value: building and debugging code, and judging a short, pre-filtered list of candidate moments.

Two modes, and you must know which one you are in:

- **Build mode** (setup, new features, fixing bugs): token use is acceptable, but still follow the budget rules.
- **Operate mode** (clipping a stream, rendering, captions): run the CLI, read compact summaries, score candidates, report. Never read or rewrite source code in operate mode unless a stage fails.

Read `references/token-budget.md` before the first run in any session. Read `references/scoring.md` only when you are about to score candidates. Use `assets/env.example` and `assets/CLAUDE.md.template` during setup.

## Feature flags (all in `.env`)

Every optional subsystem is behind a flag. When a flag is off: do not build it, do not install its dependencies, do not call it, and do not mention it in summaries.

| Flag | Default | Status |
|---|---|---|
| `CHAT_ACTIVITY_ENABLED` | `false` | Not implemented yet. Create only an interface stub (see below). |
| `ANALYTICS_ENABLED` | `false` | Not implemented yet. Post-performance tracking, planned for later. Stub only. |
| `SCORER_MODE` | `hybrid` | `heuristic` (zero LLM tokens), `local` (Ollama, zero Claude tokens), `hybrid` (local heuristic prefilter, then Claude ranks the top N). |
| `MUSIC_CHECK_ENABLED` | `true` | Flags clips for manual music review. |
| `VOICEOVER_LINES_ENABLED` | `false` | Drafts an optional commentary line per clip. |

The signal layer must be a small plugin registry (`signals/__init__.py` with a `SIGNALS` list and a `Signal` base class exposing `name`, `enabled()`, `compute(job) -> list[TimedScore]`). Chat activity and analytics get a file containing only the class with `enabled()` reading the flag and `compute()` raising `NotImplementedError`. Turning them on later must not require touching other stages.

If the user asks to enable a stubbed feature, say it is not implemented yet and offer to build it in build mode.

## Rights and platform rules (keep these, do not weaken)

- Before processing any URL, confirm the user owns the footage or has permission (for example a streamer's clipping program). A public stream or working URL is not permission.
- Never bypass DRM, paywalls, members-only content, age gates, geo-blocks, or other access controls. If a download fails for access reasons, ask for an authorized local file instead of trying workarounds.
- Browser cookies only with explicit consent and only for the user's own accessible content. Never print, store, or commit cookie values.
- Keep a rights record per source: URL or path, creator, date collected, permission basis, and `program_rules` (required credit, tags, allowed platforms).
- Never promise a clip is copyright-safe or monetizable. YouTube reused-content and inauthentic-content rules are separate from a streamer's permission.
- Never publish, schedule, or upload. This skill ends at a review package. Publishing is a separate, explicitly approved action.

## Open-source stack (smallest working set)

Required: Python 3.11+, FFmpeg/ffprobe, `faster-whisper`, `numpy`, `pyyaml`, `python-dotenv`.
Only if needed: `yt-dlp` (authorized URLs only), `opencv-python-headless` (facecam detection), Ollama (only when `SCORER_MODE=local`).
Not used: paid clipping APIs, cloud transcription, Streamlit, SQLite (job state lives in JSON manifests; add a DB later only if the user asks).

Pin exact versions in `requirements.txt` after verifying they install on this machine. Burn captions with FFmpeg's `subtitles`/`ass` filter (libass), not a separate library.

## Setup (build mode)

1. Inspect, in one batched command: OS, CPU arch, RAM, GPU (`nvidia-smi` if present), Python version, FFmpeg version and whether it has `libass`, free disk space. Do not assume macOS. On Windows, prefer PowerShell-compatible commands or ask whether the user uses WSL.
2. Inspect the repo shallowly (`ls`, existing README/CLAUDE.md/requirements). Do not read whole directories.
3. Present a short plan and the list of missing prerequisites. Ask before installing system packages or downloading Whisper/Ollama models.
4. Create a virtualenv and install the required set only.
5. Copy `assets/env.example` to `.env.example` and `.env`. Pick `WHISPER_MODEL`, `WHISPER_DEVICE`, and `WHISPER_COMPUTE_TYPE` from the hardware check (see the table in `assets/env.example`).
6. Copy `assets/CLAUDE.md.template` to the project's `CLAUDE.md` (merge if one exists). It carries the token rules into every future session.
7. Build stage by stage, testing each before starting the next. Do not generate the whole application in one step.
8. Run the self-test (below). Setup is not done until it passes.

## Project layout

```text
clipper/
  cli.py            # entry point: run, resume, status, render, package, selftest
  config.py         # loads .env + config/settings.yaml, validates flags
  manifest.py       # job state, atomic writes
  stages/           # probe, fetch, audio, transcribe, signals, candidates, score, render, package
  signals/          # registry: loudness, speech, keywords, chat_activity (stub)
  analytics/        # stub only
  captions.py       # ASS word-by-word caption builder
  layout.py         # facecam detection + 9:16 layouts
config/settings.yaml
data/jobs/<job_id>/ # everything for one source lives here
tests/
```

Respect an existing layout if the repo already has one.

## Reliability rules

- **Stage manifest.** Each job has `data/jobs/<job_id>/manifest.json` recording each stage's status (`pending|running|done|failed`), start/end time, output paths, and a short error. `clipper resume <job_id>` restarts from the first stage not marked `done`.
- **Atomic writes.** Write to `*.tmp`, then rename. A crash never leaves a half-written file that looks complete.
- **Idempotent stages.** A stage whose outputs exist and pass verification is skipped. Re-running never duplicates clips. Clip file names are `<job_id>_<candidate_id>_<start>-<end>.mp4`.
- **Chunked transcription.** Split audio into 10-minute chunks with 2-second overlap. Cache each chunk's transcript. A crash at hour 5 resumes at hour 5, not hour 0. Merge chunks and de-duplicate overlapping words by timestamp.
- **Never load a long video into RAM.** Stream with FFmpeg; compute loudness with `ffmpeg ... astats`/`ebur128` or numpy over chunks.
- **Verify every output** with ffprobe: duration within 0.5 s of the requested range, has video and audio streams, resolution 1080x1920 (or the configured size). A clip that fails verification is marked failed, not reported as created.
- **Disk guard.** Before fetch and render, check free space against an estimate. Stop with a clear message rather than filling the disk.
- **Downloads.** Retry up to 3 times with backoff on network errors only. Never retry on access-denied responses. Prefer fetching only needed sections (`yt-dlp --download-sections`) after candidates are known; fall back to asking for a local file.
- **Original is read-only.** Never modify source media. Delete temp chunks only after the stage that consumes them is verified.
- **Logs.** Each stage logs to `data/jobs/<job_id>/logs/<stage>.log`. The CLI prints at most ~20 lines per command; full detail stays in the log.

## Processing pipeline (operate mode)

1. **probe**: ffprobe the source; record duration, fps, resolution, audio streams. Rights record must exist before a URL is fetched.
2. **audio**: extract mono 16 kHz WAV for transcription.
3. **transcribe**: chunked faster-whisper with word timestamps. `TRANSCRIBE_LANGUAGE=auto` by default; Hindi/Hinglish streams may need `hi` or a larger model, and their captions always get a manual-check flag.
4. **signals**: run every enabled signal plugin. Default set: loudness peaks (z-score over a rolling window), speech-rate bursts, reaction keywords from `config/settings.yaml` (editable list, include Hindi/Hinglish exclamations if relevant). Loudness is a clue, not proof: explosions, sirens, and music cause false positives.
5. **candidates**: combine signals into a heuristic score per moment, merge overlapping moments, build windows with 8–15 s lead-in and 10–25 s payoff (respect `CLIP_MIN_SECONDS`/`CLIP_MAX_SECONDS`). Keep the top `MAX_LLM_CANDIDATES` and write `candidates_compact.json` (format in `references/scoring.md`). This file is the only thing Claude reads for scoring.
6. **score**: depends on `SCORER_MODE`. `heuristic`: use heuristic order as final. `local`: send each compact candidate to Ollama with the rubric prompt, parse JSON, retry once on malformed output. `hybrid`: you score the compact file yourself, in one pass, following `references/scoring.md`, and write `scores.json`.
7. **render** (only approved IDs, or top `AUTO_RENDER_TOP_N` if the user said "render the best"): cut with accurate seek, choose layout, burn captions, normalize audio with `loudnorm` single-pass conservative settings.
8. **package**: per clip, write a thumbnail frame, transcript excerpt, source timestamps, score breakdown, hook title, post caption with required credit from `program_rules`, and review flags into `data/jobs/<job_id>/review/`. Write `review/summary.md` (one table, under 40 lines).

### Layout

Sample ~6 frames across the clip. Detect a facecam as a rectangle that stays visually stable across frames and contains a face (OpenCV). Then: facecam on top + gameplay below if both are clear; else center crop on gameplay if the action is centered; else full frame centered on a blurred copy. Low confidence sets a `layout_review` flag. Never invent a facecam.

### Captions and anti-template rules

- Word-by-word ASS captions placed in a safe zone that avoids the facecam and the lower HUD.
- Rotate across 3+ caption styles defined in `settings.yaml` (font size, color, position) so consecutive clips don't look identical.
- Hooks must describe what actually happens, contain no story spoilers, and vary in structure. Reject misleading hooks.

## Summary after a run (keep it short)

1. Source and duration.
2. Candidates found / scored / rendered.
3. Output folder path.
4. A table of the best clips: ID, timestamps, score, one-line reason, flags.
5. Flags needing human review (rights, music, spoilers, layout, captions).
6. What the user should do next.

Do not paste transcripts, logs, or JSON into the summary.

## Self-test

`clipper selftest` must: generate a 60 s test video with FFmpeg (`testsrc2` + `sine`), run probe, audio, signals, candidates, render, and package on it with `SCORER_MODE=heuristic`, verify outputs with ffprobe, then delete the test job. Transcription is tested separately on a short user-provided sample with speech. Unit tests cover timestamp math, candidate merging, chunk-transcript merging, manifest resume, and file naming.
