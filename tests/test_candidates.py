import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from clipper import config
from clipper.stages import candidates as C
from clipper.stages.common import Job, run_stage
from clipper.manifest import atomic_write

S = config.load()
CFG = {**C.DEFAULT}


def ts(a, b, score=0.8, **meta):
    return {"start": a, "end": b, "score": score, "meta": meta}


class Pure(unittest.TestCase):
    def test_moments_merge_and_peak(self):
        sig = {"loudness": [ts(100, 103, 1.0, z=4)], "keywords": [ts(104, 106, 1.0, keyword="x")],
               "speech_rate": [ts(300, 305, 0.5)]}
        comb, per = C.combined_timeline(sig, 400, CFG["weights"])
        self.assertTrue(0 <= comb.max() <= 1)
        ms = C.find_moments(comb, per, 0.15, 10)
        self.assertEqual(len(ms), 2)                       # 100-103 and 104-106 merge (gap 1s)
        first = ms[0]
        self.assertEqual(set(first["signals"]), {"loudness", "keywords"})
        self.assertGreater(first["score"], ms[1]["score"])

    def test_window_within_limits_and_bounds(self):
        m = {"peak": 5, "end": 8}
        a, b = C.build_window(m, 400, ([], []), CFG, 15, 59)         # near start: no negative time
        self.assertEqual(a, 0.0)
        self.assertTrue(15 <= b - a <= 59)
        a, b = C.build_window({"peak": 395, "end": 398}, 400, ([], []), CFG, 15, 59)
        self.assertEqual(b, 400)
        self.assertGreaterEqual(b - a, 15)
        a, b = C.build_window({"peak": 200, "end": 400}, 1000, ([], []), CFG, 15, 59)
        self.assertLessEqual(b - a, 59)

    def test_lead_and_payoff_ranges(self):
        a, b = C.build_window({"peak": 200, "end": 201}, 1000, ([], []), CFG, 15, 59)
        self.assertTrue(8 <= 200 - a <= 15)
        self.assertTrue(10 <= b - 200 <= 25)

    def test_snap_to_pause_boundaries(self):
        # words: 0-1 "a", gap, 1.6-2.0 "b", 2.1-2.5 "c" ... a pause sits before 'b' and after 'a'
        starts, ends = [0.0, 1.6, 2.1, 2.7], [1.0, 2.0, 2.5, 3.1]
        self.assertEqual(C.snap_start(starts, ends, 1.9, 2.0, 0.3), 1.6)    # after pause, not mid-run
        self.assertEqual(C.snap_end(starts, ends, 1.1, 2.0, 0.3), 1.0)      # before the pause

    def test_select_suppresses_overlap_and_caps(self):
        cs = [{"start_s": 0, "end_s": 30, "heuristic_score": 0.9},
              {"start_s": 10, "end_s": 40, "heuristic_score": 0.8},    # 2/3 overlap -> drop
              {"start_s": 100, "end_s": 130, "heuristic_score": 0.7},
              {"start_s": 200, "end_s": 230, "heuristic_score": 0.6}]
        k = C.select(cs, 0.25, 2)
        self.assertEqual([c["start_s"] for c in k], [0, 100])

    def test_excerpt_caps_words_around_peak(self):
        ws = [{"w": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(100)]
        t = C.excerpt(ws, 0, 100, 50, 10).split()
        self.assertEqual(len(t), 10)
        self.assertIn("w50", t)


def make_job(td, music=True):
    s = mock.Mock(wraps=S)
    s.jobs_dir = s.data_dir = Path(td)
    s.yaml, s.clip_min, s.clip_max = S.yaml, 15, 59
    s.max_llm_candidates, s.excerpt_max_words, s.music_check = 2, 120, music
    job = Job.open(s, "abcDEF12345")
    words = [{"w": f"w{t}", "start": float(t), "end": t + 0.4, "p": 0.9} for t in range(0, 600, 1)
             if not (300 <= t < 330)]                    # 300-330 near-silent (music-like)
    words.append({"w": "no", "start": 120.0, "end": 120.2, "p": 1})
    sigs = {"loudness": [ts(120, 122, 1.0, z=4.2), ts(310, 315, 1.0, z=4.0), ts(500, 503, 0.9, z=3.6),
                         ts(50, 52, 0.7, z=3)],
            "keywords": [ts(118, 123, 1.0, keyword="no way")],
            "speech_rate": [ts(121, 124, 0.8, z=3)]}
    atomic_write(job.dir / "meta.json", json.dumps({"duration_s": 600.0, "title": "T"}))
    atomic_write(job.dir / "rights.json", json.dumps({"creator": "Chan"}))
    atomic_write(job.dir / "signals.json", json.dumps({"duration_s": 600, "signals": sigs}))
    atomic_write(job.dir / "transcript.json", json.dumps({"words": words, "flags": []}))
    return job


class StageTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            self.assertEqual(run_stage(job, "candidates", C.run, C.verify), "done")
            d = json.loads((job.dir / "candidates_compact.json").read_text())
            cs = d["candidates"]
            self.assertEqual(len(cs), 2)                                   # capped at MAX_LLM
            self.assertEqual(d["source"]["creator"], "Chan")
            self.assertGreaterEqual(cs[0]["heuristic_score"], cs[1]["heuristic_score"])
            best = cs[0]
            self.assertTrue(best["start_s"] <= 120 <= best["end_s"])   # contains the peak
            self.assertIn("no way", best["signals"]["keywords"])
            self.assertTrue(best["signals"]["speech_burst"])
            for c in cs:
                self.assertTrue(15 <= c["end_s"] - c["start_s"] <= 59)
                self.assertLessEqual(len(c["excerpt"].split()), 120)
            self.assertEqual(run_stage(job, "candidates", C.run, C.verify), "skipped")

    def test_music_flag_loud_without_speech(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            job.settings.max_llm_candidates = 5
            C.run(job)
            cs = json.loads((job.dir / "candidates_compact.json").read_text())["candidates"]
            quiet = [c for c in cs if c["start_s"] < 315 < c["end_s"] + 20 and c["start_s"] > 290]
            self.assertTrue(quiet and "possible_music" in quiet[0]["flags"])
            loud_speech = [c for c in cs if c["start_s"] < 120 < c["end_s"]][0]
            self.assertNotIn("possible_music", loud_speech["flags"])

    def test_music_check_off(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, music=False)
            job.settings.max_llm_candidates = 5
            C.run(job)
            cs = json.loads((job.dir / "candidates_compact.json").read_text())["candidates"]
            self.assertTrue(all("possible_music" not in c["flags"] for c in cs))

    def test_empty_signals_ok(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            atomic_write(job.dir / "signals.json", json.dumps({"duration_s": 600, "signals": {}}))
            C.run(job)
            self.assertEqual(json.loads((job.dir / "candidates_compact.json").read_text())["candidates"], [])


if __name__ == "__main__":
    unittest.main()
