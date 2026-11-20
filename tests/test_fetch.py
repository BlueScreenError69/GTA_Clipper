import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipper import config
from clipper.stages import fetch, rights
from clipper.stages.common import Job, StageError, run_stage
from clipper.manifest import Manifest

S = config.load()


def make_job(td):
    s = mock.Mock(wraps=S)
    s.jobs_dir = Path(td)
    s.data_dir = Path(td)
    s.max_data_gb, s.min_free_disk_gb, s.keep_temp_files = 50, 0, False
    s.ffmpeg, s.ffprobe = S.ffmpeg, S.ffprobe
    return Job.open(s, "abcDEF12345")


def fake_download(secs):
    def dl(url, outtmpl):
        out = outtmpl.replace("%(ext)s", "m4a")
        subprocess.run([S.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"sine=frequency=300:duration={secs}", "-c:a", "aac", out], check=True)
    return dl


class Classify(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(fetch.classify("Sign in to confirm your age"), "access")
        self.assertEqual(fetch.classify("This video is private video"), "access")
        self.assertEqual(fetch.classify("HTTP Error 503"), "network")
        self.assertEqual(fetch.classify("read timed out"), "network")
        self.assertEqual(fetch.classify("weird"), "other")


class Retries(unittest.TestCase):
    def test_access_never_retried(self):
        calls = []
        def f():
            calls.append(1)
            raise Exception("Sign in to confirm your age")
        with self.assertRaises(fetch.AccessError):
            fetch.with_retries(f, mock.Mock(), sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_network_retried_then_ok(self):
        n = {"c": 0}
        def f():
            n["c"] += 1
            if n["c"] < 3:
                raise Exception("connection reset by peer")
            return "ok"
        self.assertEqual(fetch.with_retries(f, mock.Mock(), sleep=lambda s: None), "ok")
        self.assertEqual(n["c"], 3)

    def test_network_gives_up_after_3(self):
        n = {"c": 0}
        def f():
            n["c"] += 1
            raise Exception("timed out")
        with self.assertRaises(StageError):
            fetch.with_retries(f, mock.Mock(), sleep=lambda s: None)
        self.assertEqual(n["c"], 3)


class Stages(unittest.TestCase):
    def test_rights_required(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            with self.assertRaises(StageError):
                rights.write_rights(job, "u", "")
            with self.assertRaises(FileNotFoundError):  # probe without rights.json
                fetch.probe(job)

    def test_probe_rejects_live_and_members(self):
        for info in ({"live_status": "is_live", "duration": 10},
                     {"availability": "subscriber_only", "duration": 10}):
            with tempfile.TemporaryDirectory() as td:
                job = make_job(td)
                rights.write_rights(job, "https://youtu.be/abcDEF12345", "my own channel")
                with mock.patch.object(fetch, "extract_info", return_value=info):
                    with self.assertRaises(StageError):
                        fetch.probe(job)

    def test_full_fetch_and_skip_and_verify(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            rights.write_rights(job, "https://youtu.be/abcDEF12345", "my own channel", credit="via X")
            info = {"id": "abcDEF12345", "title": "t", "channel": "Chan", "duration": 6,
                    "live_status": "not_live"}
            with mock.patch.object(fetch, "extract_info", return_value=info), \
                 mock.patch.object(fetch, "download_audio", fake_download(6)) as dl:
                self.assertEqual(run_stage(job, "probe", fetch.probe), "done")
                self.assertEqual(run_stage(job, "fetch_audio", fetch.fetch_audio, fetch.verify_audio), "done")
                self.assertTrue((job.dir / "audio.wav").exists())
                self.assertFalse(list(job.dir.glob("audio_src.*")))  # temp removed
                self.assertEqual(job.read_json("rights.json")["creator"], "Chan")
                self.assertEqual(run_stage(job, "fetch_audio", fetch.fetch_audio, fetch.verify_audio), "skipped")
            self.assertEqual(Manifest(job.dir).get("fetch_audio")["status"], "done")

    def test_duration_mismatch_fails_and_leaves_no_wav(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            rights.write_rights(job, "https://youtu.be/abcDEF12345", "my own channel")
            info = {"id": "x", "duration": 60, "live_status": "not_live"}
            with mock.patch.object(fetch, "extract_info", return_value=info), \
                 mock.patch.object(fetch, "download_audio", fake_download(6)):
                run_stage(job, "probe", fetch.probe)
                with self.assertRaises(StageError):
                    run_stage(job, "fetch_audio", fetch.fetch_audio)
            self.assertFalse((job.dir / "audio.wav").exists())
            self.assertEqual(job.manifest.get("fetch_audio")["status"], "failed")

    def test_disk_guard_blocks_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            job.settings.max_data_gb = 0.0001
            rights.write_rights(job, "https://youtu.be/abcDEF12345", "my own channel")
            info = {"id": "x", "duration": 36000, "live_status": "not_live"}
            with mock.patch.object(fetch, "extract_info", return_value=info), \
                 mock.patch.object(fetch, "download_audio") as dl:
                run_stage(job, "probe", fetch.probe)
                with self.assertRaises(Exception):
                    run_stage(job, "fetch_audio", fetch.fetch_audio)
                dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
