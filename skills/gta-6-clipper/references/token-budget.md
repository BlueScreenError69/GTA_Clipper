# Token budget rules

The user runs this on a Claude subscription with usage limits. Every file you read, every command output you see, and every file you write costs tokens. Follow these rules in both modes.

## Never read these into context

- Media files, WAVs, model files, anything binary.
- Full transcripts (`transcript*.json`, `*.srt`, `*.ass`). Use `clipper` CLI summaries or `head -c 2000` if you must inspect format.
- Full logs. Use `tail -n 30 <log>` and only on failure.
- `requirements` lock output, `pip freeze`, `pip install` progress, virtualenv folders, `__pycache__`, `.git`.
- Whole directories. Use `ls` or `find -maxdepth 2`, never recursive cat.

## Command output

- Run installs and long jobs with quiet flags (`pip install -q`, `ffmpeg -hide_banner -loglevel error`, `yt-dlp -q --no-progress`).
- Pipe anything that might be long through `| tail -n 20`.
- Batch related checks into one shell command instead of many small calls.

## Long jobs (transcription, rendering, downloads)

- Start them in the background with output to the stage log: `nohup python -m clipper.cli run ... > data/jobs/<id>/logs/run.out 2>&1 &`.
- Check progress with `python -m clipper.cli status <job_id>` (prints ≤ 10 lines). Don't poll more often than every few minutes; tell the user roughly how long the stage will take instead of watching it.
- A 4–8 hour stream should cost only a handful of status checks.

## Editing code

- Edit with targeted replacements. Don't rewrite a whole file to change a few lines.
- Don't re-read a file you just wrote or edited; you already know its content.
- When a stage fails: read the last 30 lines of that stage's log, read only the function named in the traceback, fix, then `resume`. Don't re-run earlier stages.

## Scoring (hybrid mode)

- Read only `candidates_compact.json`. It is capped by `MAX_LLM_CANDIDATES` (default 12) and `EXCERPT_MAX_WORDS` (default 120).
- Score all candidates in one pass and write `scores.json` once. No second pass unless the user asks.
- If the user wants more clips, raise `MAX_LLM_CANDIDATES` for that job rather than reading the full transcript.
- If limits are tight, suggest `SCORER_MODE=local` (Ollama) or `heuristic`, which use no Claude tokens for scoring.

## Session hygiene (tell the user once, during setup)

- Start a fresh Claude Code session per stream (`/clear`). Old context from previous jobs costs tokens on every message.
- Use `/compact` if a build-mode session gets long.
- If the plan offers a choice of models, a lighter model is enough for operate mode; save the strongest model for building and debugging.
- Keep `CLAUDE.md` short. It is loaded every session.
