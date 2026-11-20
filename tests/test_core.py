import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clipper import diskguard
from clipper.manifest import Manifest, job_id_from_url, atomic_write, STAGES


def fake_settings(d, cap_gb=50, floor_gb=0):
    return SimpleNamespace(data_dir=Path(d), max_data_gb=cap_gb, min_free_disk_gb=floor_gb,
                           jobs_dir=Path(d) / "jobs")


class DiskGuard(unittest.TestCase):
    def test_dir_size(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a").mkdir()
            (Path(d) / "a" / "f").write_bytes(b"x" * 1000)
            (Path(d) / "g").write_bytes(b"y" * 500)
            self.assertEqual(diskguard.dir_size(Path(d)), 1500)

    def test_blocks_over_cap(self):
        with tempfile.TemporaryDirectory() as d:
            s = fake_settings(d, cap_gb=1)
            diskguard.check(s, 10 * 1024 ** 2)
            with self.assertRaises(diskguard.DiskGuardError):
                diskguard.check(s, 2 * diskguard.GB)

    def test_counts_existing_usage(self):
        with tempfile.TemporaryDirectory() as d:
            s = fake_settings(d, cap_gb=1)
            (Path(d) / "big").write_bytes(b"0" * (600 * 1024 ** 2))
            with self.assertRaises(diskguard.DiskGuardError):
                diskguard.check(s, 500 * 1024 ** 2)

    def test_free_space_floor(self):
        with tempfile.TemporaryDirectory() as d:
            s = fake_settings(d, cap_gb=10 ** 6, floor_gb=10 ** 6)
            with self.assertRaises(diskguard.DiskGuardError):
                diskguard.check(s, 0)

    def test_estimates_positive_and_scale(self):
        self.assertGreater(diskguard.estimate_audio_bytes(3600), diskguard.estimate_audio_bytes(60))
        self.assertGreater(diskguard.estimate_clip_bytes(60), 0)


class ManifestTests(unittest.TestCase):
    def test_resume_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(Path(d))
            self.assertEqual(m.next_stage(), STAGES[0])
            for s in STAGES[:3]:
                m.mark(s, "done")
            m.mark(STAGES[3], "failed", error="boom")
            m2 = Manifest(Path(d))
            self.assertEqual(m2.next_stage(), STAGES[3])
            self.assertEqual(m2.get(STAGES[3])["error"], "boom")
            self.assertFalse(list(Path(d).glob("*.tmp")))

    def test_atomic_write_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            atomic_write(p, "1")
            atomic_write(p, "2")
            self.assertEqual(p.read_text(), "2")

    def test_job_id(self):
        self.assertEqual(job_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5"), "dQw4w9WgXcQ")
        self.assertEqual(job_id_from_url("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        with self.assertRaises(ValueError):
            job_id_from_url("https://example.com")


if __name__ == "__main__":
    unittest.main()
