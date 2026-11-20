# Project rules (GTA 6 Clipper)

Use the gta-6-clipper skill for all clipping work.

Token rules (subscription limits apply):
- Never read media, full transcripts, full logs, venvs, or .git. Use `python -m clipper.cli status <job>` and `tail -n 30` on failure only.
- Run long jobs in the background; check status every few minutes at most.
- For scoring, read only `candidates_compact.json`; write `scores.json` in one pass.
- Edit code with targeted replacements; don't re-read files you just wrote.
- Operate mode (clipping a stream): don't read or change source code unless a stage fails.
- Respect feature flags in `.env`; don't build or mention disabled features.

Safety: confirm rights before any URL; never bypass access controls; never publish or upload.
