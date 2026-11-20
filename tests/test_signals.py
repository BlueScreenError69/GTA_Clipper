import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from clipper import config, signals as reg
from clipper.signals.util import rolling_z, runs
from clipper.signals import keywords as K, speech as SP
from clipper.signals.loudness import per_second_db
from clipper.stages import signals as stage
from clipper.stages.common import Job, StageError, run_stage
from clipper.manifest import atomic_write

S = config.load()
DUR = 300


def make_job(td, loud=(100, 103), chat=False):
    s = mock.Mock(wraps=S)
    s.jobs_dir = s.data_dir = Path(td)
    s.yaml, s.chat_activity, s.analytics = S.yaml, chat, False
    job = Job.open(s, "abcDEF12345")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.02, DUR * 16000)
    a[loud[0] * 16000: loud[1] * 16000] *= 15                      # loud burst
    with wave.open(str(job.dir / "audio.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    words = [{"w": "word", "start": float(t), "end": t + 0.3, "p": 0.9} for t in range(0, DUR, 2)]  # 0.5 wps
    words += [{"w": "x", "start": 200 + i * 0.25, "end": 200 + i * 0.25 + 0.2, "p": 0.9} for i in range(40)]  # 4 wps burst
    words += [{"w": "No", "start": 50.0, "end": 50.2, "p": 1}, {"w": "way!", "start": 50.3, "end": 50.6, "p": 1},
              {"w": "Kya", "start": 120.0, "end": 120.2, "p": 1}, {"w": "baat", "start": 120.3, "end": 120.5, "p": 1}]
    words.sort(key=lambda w: w["start"])
    atomic_write(job.dir / "transcript.json", json.dumps({"words": words}))
    atomic_write(job.dir / "meta.json", json.dumps({"duration_s": float(DUR)}))
    return job


class Util(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(runs(np.array([0, 1, 1, 0, 1], bool)), [(1, 3), (4, 5)])

    def test_rolling_z_flags_spike_not_flat(self):
        x = np.zeros(300); x[150] = 10
        z = rolling_z(x, 120, 1.0)
        self.assertGreater(z[150], 5)
        self.assertLess(abs(z[10]), 0.5)
        self.assertTrue(np.all(np.abs(rolling_z(np.ones(50), 20, 1.0)) < 1e-6))   # flat -> no fake peaks


class Keyword(unittest.TestCase):
    def test_phrase_and_punctuation(self):
        ws = [{"w": "No", "start": 1, "end": 1.2}, {"w": "WAY!", "start": 1.3, "end": 1.6},
              {"w": "way", "start": 5, "end": 5.2}]
        h = K.find_phrases(ws, ["no way", "yaar"])
        self.assertEqual(h, [("no way", 1, 1.6)])


class Plugins(unittest.TestCase):
    def test_all_three_find_their_event(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            sigs = {s.name: s.compute(job) for s in reg.active(job.settings)}
            self.assertEqual(set(sigs), {"loudness", "speech_rate", "keywords"})   # chat stub off
            self.assertTrue(any(t.start <= 101 and t.end >= 102 for t in sigs["loudness"]))
            self.assertTrue(any(t.start <= 205 <= t.end for t in sigs["speech_rate"]))
            kw = {t.meta["keyword"] for t in sigs["keywords"]}
            self.assertIn("no way", kw); self.assertIn("kya baat", kw)
            self.assertLess(len(sigs["loudness"]), 6)      # quiet noise not flagged everywhere
            self.assertTrue(all(0 <= t.score <= 1 for ts in sigs.values() for t in ts))

    def test_per_second_db_length(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            self.assertEqual(len(per_second_db(job.dir / "audio.wav")), DUR)

    def test_stage_writes_and_skips(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            self.assertEqual(run_stage(job, "signals", stage.run, stage.verify), "done")
            d = json.loads((job.dir / "signals.json").read_text())
            self.assertIn("loudness", d["signals"])
            self.assertEqual(run_stage(job, "signals", stage.run, stage.verify), "skipped")

    def test_chat_enabled_without_chat_data_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, chat=True)
            stage.run(job)
            d = json.loads((job.dir / "signals.json").read_text())
            self.assertEqual(d["signals"]["chat_activity"], [])


if __name__ == "__main__":
    unittest.main()
