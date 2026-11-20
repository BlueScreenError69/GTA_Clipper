# Scoring candidates

## Input: `candidates_compact.json`

Written by the candidates stage. Keep it small; this is everything you see.

```json
{
  "job_id": "abc123",
  "source": {"creator": "", "title": "", "duration_s": 0},
  "candidates": [
    {
      "id": "c07",
      "start_s": 5123.4,
      "end_s": 5161.0,
      "heuristic_score": 0.82,
      "signals": {"loudness_z": 3.1, "speech_burst": true, "keywords": ["no way"]},
      "excerpt": "transcript text for the window, at most EXCERPT_MAX_WORDS words"
    }
  ]
}
```

## Output: `scores.json`

```json
{
  "job_id": "abc123",
  "scored": [
    {
      "id": "c07",
      "event_type": "reaction|discovery|funny_fail|chase|crash|combat|story|other",
      "score_breakdown": {
        "hook": 0, "setup_payoff": 0, "self_contained": 0,
        "emotion_or_gameplay": 0, "clarity": 0, "novelty": 0, "pacing": 0
      },
      "penalties": 0,
      "overall_score": 0,
      "hook_title": "",
      "spoiler_level": "none|minor|major",
      "flags": ["possible_music", "caption_check", "context_heavy"],
      "suggested_trim": {"start_s": null, "end_s": null},
      "one_line_reason": ""
    }
  ],
  "rejected": [{"id": "c03", "reason": "no payoff"}]
}
```

`overall_score` = sum of `score_breakdown` minus `penalties`, clamped to 0–100. It must add up.

## Rubric (max points)

- `hook` (20): something grabs attention in the first 1–2 seconds.
- `setup_payoff` (20): a clear setup and a payoff inside the window.
- `self_contained` (20): understandable with no prior stream context.
- `emotion_or_gameplay` (15): strong reaction or genuinely interesting gameplay.
- `clarity` (10): speech is intelligible from the excerpt; signals don't suggest noise only.
- `novelty` (10): not a repeat of another selected candidate.
- `pacing` (5): fits the length without dead air.

## Penalties

- Needs substantial context: 10–25.
- No payoff or incomplete: 10–20.
- Duplicate of a higher-scoring candidate: reject.
- Misleading or fabricated hook needed to make it work: reject.

## Rules

- Judge only from the excerpt and signals. If the excerpt doesn't show a payoff, don't assume one; lower the score or flag it.
- `spoiler_level` major = reveals story events, endings, character deaths, or mission outcomes. Hooks for any spoiler clip must not reveal the spoiler.
- Add `possible_music` when the excerpt has little speech during high loudness, or mentions radio or songs.
- Add `caption_check` for mixed-language or low-confidence speech.
- Return fewer good clips rather than padding. Never invent moments to hit a count.
- Hooks: accurate, no clickbait lies, and vary the structure across clips (question, statement, "POV:", reaction quote, and so on).
