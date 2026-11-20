import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipper import config
from clipper.stages import score as SC
from clipper.stages.common import Job, StageError, Waiting, run_stage
from clipper.manifest import Manifest, atomic_write

S = config.load()
CANDS = {"job_id": "abcDEF12345", "source": {"creator": "Chan", "title": "T", "duration_s": 600},
         "candidates": [
             {"id": "c01", "start_s": 108, "end_s": 140, "heuristic_score": 0.8, "flags": ["possible_music"],
              "signals": {"loudness_z": 4.2, "speech_burst": True, "keywords": ["no way"]}, "excerpt": "no way he did"},
             {"id": "c02", "start_s": 300, "end_s": 330, "heuristic_score": 0.4, "flags": [],
              "signals": {"loudness_z": 0, "speech_burst": False, "keywords": []}, "excerpt": "hmm"}]}
GOOD = {"event_type": "reaction",
        "score_breakdown": {"hook": 18, "setup_payoff": 15, "self_contained": 16, "emotion_or_gameplay": 12,
                            "clarity": 8, "novelty": 7, "pacing": 4},
        "penalties": 5, "hook_title": "He did WHAT?", "spoiler_level": "none", "flags": ["caption_check"],
        "one_line_reason": "clear reaction", "reject": None}


def make_job(td, mode):
    s = mock.Mock(wraps=S)
    s.jobs_dir = s.data_dir = Path(td)
    s.scorer_mode, s.ollama_model, s.ollama_host = mode, "m", "http://x"
    job = Job.open(s, "abcDEF12345")
    atomic_write(job.dir / "candidates_compact.json", json.dumps(CANDS))
    return job


def scores(job):
    return json.loads((job.dir / "scores.json").read_text())


class Helpers(unittest.TestCase):
    def test_overall_clamped(self):
        self.assertEqual(SC.overall({"a": 90, "b": 30}, 5), 100)
        self.assertEqual(SC.overall({"a": 3}, 25), 0)

    def test_normalize_clamps_and_adds_up(self):
        raw = {**GOOD, "score_breakdown": {"hook": 99, "pacing": -3}, "event_type": "weird",
               "spoiler_level": "huh", "flags": ["bogus", "context_heavy"]}
        e = SC.normalize_entry(raw, CANDS["candidates"][0])
        self.assertEqual(e["score_breakdown"]["hook"], 20)
        self.assertEqual(e["score_breakdown"]["pacing"], 0)
        self.assertEqual(e["overall_score"], SC.overall(e["score_breakdown"], e["penalties"]))
        self.assertEqual(e["event_type"], "other")
        self.assertEqual(e["spoiler_level"], "minor")
        self.assertEqual(e["flags"], ["context_heavy", "possible_music"])   # candidate flag kept, bogus dropped

    def test_parse_json_obj_fenced(self):
        self.assertEqual(SC.parse_json_obj('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            SC.parse_json_obj("nope")


class Heuristic(unittest.TestCase):
    def test_mode(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "heuristic")
            self.assertEqual(run_stage(job, "score", SC.run, SC.verify), "done")
            d = scores(job)
            self.assertEqual([e["id"] for e in d["scored"]], ["c01", "c02"])
            self.assertEqual(d["scored"][0]["overall_score"], 80)
            self.assertIn("spoiler_unchecked", d["scored"][0]["flags"])
            self.assertIn("possible_music", d["scored"][0]["flags"])
            self.assertEqual(d["scored"][0]["hook_title"], 'Reaction: "no way"')
            self.assertEqual(d["scored"][1]["hook_title"], "GTA 6 moment at 0:05:00")
            self.assertEqual(run_stage(job, "score", SC.run, SC.verify), "skipped")


class Local(unittest.TestCase):
    def test_scores_and_reject_and_retry(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "local")
            replies = iter([json.dumps(GOOD), "garbage", json.dumps({**GOOD, "reject": "no payoff"})])
            with mock.patch.object(SC, "ollama_chat", side_effect=lambda *a: next(replies)) as m:
                SC.run(job)
            self.assertEqual(m.call_count, 3)                       # c02 retried once, then parsed
            d = scores(job)
            self.assertEqual([e["id"] for e in d["scored"]], ["c01"])
            self.assertEqual(d["rejected"], [{"id": "c02", "reason": "no payoff"}])
            self.assertEqual(d["scored"][0]["overall_score"], 75)   # 18+15+16+12+8+7+4 - 5 = 75
            self.assertEqual(SC.validate_scores(d, CANDS), [])

    def test_malformed_twice_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "local")
            with mock.patch.object(SC, "ollama_chat", return_value="not json"):
                SC.run(job)
            d = scores(job)
            self.assertEqual(len(d["scored"]), 2)
            self.assertTrue(all("llm_failed" in e["flags"] for e in d["scored"]))

    def test_needs_model(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "local")
            job.settings.ollama_model = ""
            with self.assertRaises(StageError):
                SC.run(job)


class Hybrid(unittest.TestCase):
    def entry(self, cid, **kw):
        e = SC.normalize_entry(GOOD, {"id": cid, "flags": []})
        e.update(kw)
        return e

    def write(self, job, scored, rejected=()):
        atomic_write(job.dir / "scores.json", json.dumps(
            {"job_id": "abcDEF12345", "scored": scored, "rejected": list(rejected)}))

    def test_waits_then_accepts(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "hybrid")
            with self.assertRaises(Waiting):
                run_stage(job, "score", SC.run, SC.verify)
            self.assertEqual(Manifest(job.dir).get("score")["status"], "pending")
            self.write(job, [self.entry("c01"), self.entry("c02")])
            self.assertEqual(run_stage(job, "score", SC.run, SC.verify), "done")

    def test_rejects_bad_sum_unknown_id_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, "hybrid")
            self.write(job, [self.entry("c01", overall_score=99), self.entry("c02")])
            with self.assertRaises(StageError) as cm:
                SC.run(job)
            self.assertIn("overall_score", str(cm.exception))
            self.write(job, [self.entry("c09"), self.entry("c01")])
            with self.assertRaises(StageError) as cm:
                SC.run(job)
            self.assertIn("unknown", str(cm.exception))
            self.assertIn("c02", str(cm.exception))                 # unscored candidate named
            self.write(job, [self.entry("c01")], [{"id": "c02", "reason": "no payoff"}])
            self.assertEqual(SC.run(job), ["scores.json"])          # rejected counts as handled


if __name__ == "__main__":
    unittest.main()
