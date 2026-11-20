import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from clipper import config
from clipper.stages import transcribe as T
from clipper.stages.common import Job, run_stage
from clipper.manifest import atomic_write

S = config.load()


class Pure(unittest.TestCase):
    def test_plan(self):
        p = T.plan_chunks(1300, 600)
        self.assertEqual(len(p), 3)
        self.assertEqual((p[1]["own_start"], p[1]["own_end"]), (600, 1200))
        self.assertEqual((p[1]["dec_start"], p[1]["dec_end"]), (598, 1202))
        self.assertEqual(p[2]["dec_end"], 1300)
        self.assertEqual(p[0]["dec_start"], 0)
        self.assertEqual(len(T.plan_chunks(5, 600)), 1)

    def test_ownership_exactly_once_at_seam(self):
        # a word straddling the 600s seam appears in both decodes; only one chunk may keep it
        word = {"w": "boom", "start": 599.8, "end": 600.6, "p": 1}   # mid 600.2 -> chunk 1
        early = {"w": "hi", "start": 599.0, "end": 599.4, "p": 1}     # mid 599.2 -> chunk 0
        a = T.own_words([early, word], 0, 600)
        b = T.own_words([early, word], 600, 1200)
        self.assertEqual([w["w"] for w in a], ["hi"])
        self.assertEqual([w["w"] for w in b], ["boom"])

    def test_last_chunk_keeps_tail(self):
        w = {"w": "end", "start": 99.9, "end": 100.4, "p": 1}
        self.assertEqual(len(T.own_words([w], 0, 100, last=False)), 0)
        self.assertEqual(len(T.own_words([w], 0, 100, last=True)), 1)

    def test_merge_sorted(self):
        m = T.merge_chunks([{"i": 1, "words": [{"start": 5}]}, {"i": 0, "words": [{"start": 1}, {"start": 2}]}])
        self.assertEqual([w["start"] for w in m], [1, 2, 5])


class FakeModel:
    """Emits one word per second of the decoded audio, text = absolute-second index."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kw):
        self.calls += 1
        n = int(len(audio) / 16000)
        words = [NS(word=f" w{i}", start=i + 0.1, end=i + 0.9, probability=0.9) for i in range(n)]
        return iter([NS(words=words)]), NS(language="en", language_probability=0.99)


def make_job(td, secs):
    s = mock.Mock(wraps=S)
    s.jobs_dir = s.data_dir = Path(td)
    s.max_data_gb, s.min_free_disk_gb, s.keep_temp_files = 50, 0, False
    s.transcribe_language, s.chunk_seconds = "auto", 10
    s.whisper_model = "tiny"
    job = Job.open(s, "abcDEF12345")
    subprocess.run([S.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=duration={secs}", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(job.dir / "audio.wav")], check=True)
    atomic_write(job.dir / "meta.json", json.dumps({"duration_s": float(secs)}))
    return job


class StageTests(unittest.TestCase):
    def test_chunked_exactly_once_and_cache_resume(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, 25)           # chunks: 0-10, 10-20, 20-25
            fm = FakeModel()
            with mock.patch.object(T, "load_model", return_value=fm):
                self.assertEqual(run_stage(job, "transcribe", T.transcribe, T.verify), "done")
            self.assertEqual(fm.calls, 3)
            t = json.loads((job.dir / "transcript.json").read_text())
            starts = [w["start"] for w in t["words"]]
            self.assertEqual(starts, sorted(starts))
            # no duplicate word at any seam: word centers are unique within 0.5s
            mids = [(w["start"] + w["end"]) / 2 for w in t["words"]]
            self.assertTrue(all(b - a > 0.5 for a, b in zip(mids, mids[1:])))
            self.assertEqual(t["language"], "en")
            self.assertEqual(t["flags"], [])
            self.assertGreaterEqual(t["word_count"], 20)

    def test_crash_resumes_from_cached_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, 25)
            fm = FakeModel()
            real = fm.transcribe
            def flaky(audio, **kw):
                if fm.calls == 2:
                    fm.calls += 1
                    raise RuntimeError("crash in chunk 3")
                return real(audio, **kw)
            fm.transcribe = flaky
            with mock.patch.object(T, "load_model", return_value=fm):
                with self.assertRaises(RuntimeError):
                    run_stage(job, "transcribe", T.transcribe, T.verify)
            self.assertEqual(len(list((job.dir / "transcript_chunks").glob("chunk_*.json"))), 2)
            fm2 = FakeModel()
            with mock.patch.object(T, "load_model", return_value=fm2):
                run_stage(job, "transcribe", T.transcribe, T.verify)
            self.assertEqual(fm2.calls, 1)   # only the missing chunk is decoded

    def test_hindi_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, 8)
            fm = FakeModel()
            orig = fm.transcribe
            def hi(audio, **kw):
                segs, info = orig(audio, **kw)
                info.language = "hi"
                return segs, info
            fm.transcribe = hi
            with mock.patch.object(T, "load_model", return_value=fm):
                T.transcribe(job)
            self.assertIn("caption_check", json.loads((job.dir / "transcript.json").read_text())["flags"])

    def test_read_wav_slice_length(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td, 5)
            self.assertEqual(len(T.read_wav_slice(job.dir / "audio.wav", 1, 3)), 32000)


if __name__ == "__main__":
    unittest.main()
