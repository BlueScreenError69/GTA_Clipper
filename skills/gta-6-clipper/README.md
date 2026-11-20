# GTA 6 Clipper — Claude Code Skill (v2)

A Claude Code skill that builds and operates a local, open-source stream-to-Shorts clipping pipeline while using as few Claude tokens as possible.

## What changed from v1

- Chat activity and analytics are off by default behind `.env` flags, with stubs so they can be added later without touching other stages.
- Scoring modes: `heuristic` (no LLM), `local` (Ollama), `hybrid` (Claude ranks only a short, pre-filtered list).
- Token budget rules for Claude Code, also copied into your project's `CLAUDE.md`.
- Reliability: stage manifest with resume, chunked transcription cache, atomic writes, ffprobe verification, disk guard, self-test.
- Score breakdown that adds up, spoiler flag, clipping-program credit rules, rotating caption styles.

## Install

Personal (all projects):
- macOS/Linux: `~/.claude/skills/gta-6-clipper/SKILL.md`
- Windows: `C:\Users\<you>\.claude\skills\gta-6-clipper\SKILL.md`

Project only: `your-project/.claude/skills/gta-6-clipper/SKILL.md`

Keep the `references/` and `assets/` folders next to `SKILL.md`.

## Use

```text
Set up the GTA 6 clipper.
```

Then, in a fresh session per stream:

```text
Clip this authorized local recording: /path/to/stream.mp4 — show me the top candidates before rendering.
```

## Important

This is instructions for Claude Code, not a prebuilt app. Setup builds the Python project on your machine, so do it before launch day. Only process footage you own or are authorized to reuse, and review every clip before posting.
