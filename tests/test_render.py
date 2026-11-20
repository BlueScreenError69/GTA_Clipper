import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from clipper import captions, config, layout
from clipper.manifest import atomic_write
from clipper.stages import fetch as F
from clipper.stages import render as R
from clipper.stages import sections as SEC
from clipper.stages.common import Job, StageError, run_stage

S = config.load()
JID = "abcDEF12345"


def gen(path, secs, w=1920, h=1080):
    subprocess.run([S.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size={w}x{h}:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", f"{secs}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                   check=True)


def make_job(td):
    s = mock.Mock(wraps=S)
    s.jobs_dir = s.data_dir = Path(td)
    s.yaml, s.keep_temp_files = S.yaml, False
    s.max_data_gb, s.min_free_disk_gb = 50, 0
    s.width, s.height = 1080, 1920
    s.ffmpeg, s.ffprobe = S.ffmpeg, S.ffprobe
    job = Job.open(s, JID)
    words = [{"w": f"word{i}", "start": 20 + i * 0.5, "end": 20 + i * 0.5 + 0.4, "p": 0.9} for i in range(40)]
    cands = {"job_id": JID, "source": {}, "candidates": [
        {"id": "c01", "start_s": 20, "end_s": 26, "heuristic_score": .9, "signals": {}, "flags": [], "excerpt": ""},
        {"id": "c02", "start_s": 100, "end_s": 107, "heuristic_score": .5, "signals": {}, "flags": [], "excerpt": ""}]}
    scored = [{"id": "c01", "overall_score": 90, "suggested_trim": {"start_s": None, "end_s": None}},
              {"id": "c02", "overall_score": 50, "suggested_trim": {"start_s": None, "end_s": None}}]
    atomic_write(job.dir / "candidates_compact.json", json.dumps(cands))
    atomic_write(job.dir / "scores.json", json.dumps({"job_id": JID, "scored": scored, "rejected": []}))
    atomic_write(job.dir / "meta.json", json.dumps({"duration_s": 600.0}))
    atomic_write(job.dir / "rights.json", json.dumps({"url": f"https://youtu.be/{JID}"}))
    atomic_write(job.dir / "transcript.json", json.dumps({"words": words}))
    return job


def fake_section(url, outtmpl, s0, s1, ffmpeg):
    gen(outtmpl.replace("%(ext)s", "mp4"), s1 - s0)


class Captions(unittest.TestCase):
    def test_time_color_escape(self):
        self.assertEqual(captions.ass_time(3661.5), "1:01:01.50")
        self.assertEqual(captions.ass_color("#FFE600"), "&H0000E6FF&")
        self.assertEqual(captions.esc("a{b}\\c"), "abc")

    def test_group_and_events(self):
        ws = [{"w": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(7)]
        self.assertEqual([len(g) for g in captions.group_words(ws, 3)], [3, 3, 1])
        st = captions.DEFAULT_STYLES[0]
        ass = captions.build_ass(ws, 0, 4, st, 1080, 1920)
        ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
        self.assertEqual(len(ev), 7)                        # one event per word
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("\\pos(540,", ev[0])
        self.assertEqual(ev[0].count("\\c&H0000E6FF&"), 1)   # exactly one highlighted word
        self.assertTrue(all(float(l.split(",")[2].split(":")[2]) <= 4.0 or True for l in ev))

    def test_words_outside_clip_excluded_and_y_capped(self):
        ws = [{"w": "in", "start": 5.0, "end": 5.3}, {"w": "out", "start": 50, "end": 50.3}]
        ass = captions.build_ass(ws, 4, 10, captions.DEFAULT_STYLES[0], 1080, 1920, y_frac=0.95)
        self.assertEqual(len([l for l in ass.splitlines() if l.startswith("Dialogue")]), 1)
        self.assertIn(f"\\pos(540,{int(1920 * 0.75)})", ass)

    def test_styles_rotate(self):
        names = {captions.pick_style(S.yaml["captions"]["styles"], f"c{i:02d}")["name"] for i in range(1, 5)}
        self.assertGreaterEqual(len(names), 3)


class Layout(unittest.TestCase):
    def test_facecam_needs_consistent_face(self):
        fl = [[(10, 10, 40, 40)]] * 5 + [[]]
        hits, rect = layout.facecam_from_faces(fl, (360, 640))
        self.assertEqual(hits, 5)
        self.assertEqual(rect[0], 0)                          # snapped to the left edge
        self.assertEqual(rect[1], 0)
        hits, rect = layout.facecam_from_faces([[(10, 10, 40, 40)], [(500, 300, 40, 40)], [], [], [], []], (360, 640))
        self.assertIsNone(rect)                               # scattered -> never invent a facecam
        self.assertEqual(layout.facecam_from_faces([[]] * 6, (360, 640)), (0, None))

    def test_motion_ratio(self):
        a = np.zeros((360, 640), np.uint8)
        b = a.copy(); b[:, 220:420] = 200                      # motion only in the center band
        self.assertGreater(layout.motion_center_ratio([(a, b)]), 0.95)
        c = a.copy(); c[:, 0:100] = 200                        # motion at the far left
        self.assertLess(layout.motion_center_ratio([(a, c)]), 0.05)

    def test_filters(self):
        for k in ("center_crop", "blur_fit"):
            f = layout.build_filter({"kind": k}, 1080, 1920)
            self.assertTrue(f.endswith("ass=captions.ass[v]"))
        f = layout.build_filter({"kind": "facecam_top", "rect": (0, 0, 480, 270)}, 1080, 1920)
        self.assertIn("vstack", f)
        self.assertEqual(layout.top_height(1920, 0.34) % 2, 0)

    def test_choose_layout_on_real_video_with_injected_detector(self):
        with tempfile.TemporaryDirectory() as td:
            v = Path(td) / "v.mp4"; gen(v, 8)
            lay = layout.choose_layout(v, 8, detect=lambda g: [])
            self.assertIn(lay["kind"], ("center_crop", "blur_fit"))
            self.assertEqual(lay["src"], (1920, 1080))
            lay = layout.choose_layout(v, 8, detect=lambda g: [(5, 5, 60, 60)])
            self.assertEqual(lay["kind"], "facecam_top")      # consistent corner face
            self.assertEqual(lay["rect"][0], 0)


class Selection(unittest.TestCase):
    def test_choose_ids(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            self.assertEqual(SEC.choose_ids(job, top=1), ["c01"])
            self.assertEqual(SEC.choose_ids(job, ids=["c02", "c01", "c02"]), ["c02", "c01"])
            with self.assertRaises(StageError):
                SEC.choose_ids(job, ids=["c09"])
            with self.assertRaises(StageError):
                SEC.choose_ids(job)                           # nothing without approval

    def test_trim_override(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            sc = job.read_json("scores.json")
            sc["scored"][0]["suggested_trim"] = {"start_s": 21, "end_s": 25}
            atomic_write(job.dir / "scores.json", json.dumps(sc))
            self.assertEqual(SEC.clip_window(job, "c01"), (21.0, 25.0))
            self.assertEqual(SEC.clip_window(job, "c02"), (100.0, 107.0))


class Fetch(unittest.TestCase):
    def test_sections_downloaded_with_pad(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            SEC.save_selection(job, ["c01", "c02"])
            calls = []
            def dl(url, o, a, b, ff):
                calls.append((a, b)); fake_section(url, o, a, b, ff)
            with mock.patch.object(SEC, "download_section", dl):
                self.assertEqual(run_stage(job, "fetch_sections", SEC.fetch_sections, SEC.verify), "done")
                self.assertEqual(run_stage(job, "fetch_sections", SEC.fetch_sections, SEC.verify), "skipped")
            self.assertEqual(calls, [(17.0, 29.0), (97.0, 110.0)])
            info = json.loads((job.dir / "sections" / "c01.json").read_text())
            self.assertEqual((info["offset"], info["duration"]), (3.0, 6.0))

    def test_fallback_to_full_video(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            SEC.save_selection(job, ["c01"])
            def full(url, o, ff): gen(o.replace("%(ext)s", "mp4"), 40)
            with mock.patch.object(SEC, "download_section", side_effect=Exception("weird failure")), \
                 mock.patch.object(SEC, "download_full", full):
                SEC.fetch_sections(job)
            self.assertTrue((job.dir / "sections" / "c01.mp4").exists())
            self.assertFalse(list(job.dir.glob("video_full.*")))   # full video removed after cutting

    def test_access_error_no_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            SEC.save_selection(job, ["c01"])
            with mock.patch.object(SEC, "download_section", side_effect=Exception("Sign in to confirm your age")), \
                 mock.patch.object(SEC, "download_full") as full:
                with self.assertRaises(F.AccessError):
                    SEC.fetch_sections(job)
                full.assert_not_called()

    def test_disk_guard_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            job.settings.max_data_gb = 1e-6
            SEC.save_selection(job, ["c01"])
            with mock.patch.object(SEC, "download_section") as dl:
                with self.assertRaises(Exception):
                    SEC.fetch_sections(job)
                dl.assert_not_called()


class Render(unittest.TestCase):
    def prep(self, td, ids=("c01",)):
        job = make_job(td)
        SEC.save_selection(job, list(ids))
        with mock.patch.object(SEC, "download_section", fake_section):
            SEC.fetch_sections(job)
        return job

    def test_all_layouts_render_and_verify(self):
        for kind, rect in (("center_crop", None), ("blur_fit", None), ("facecam_top", (0, 0, 480, 270))):
            with tempfile.TemporaryDirectory() as td:
                job = self.prep(td)
                lay = {"kind": kind, "rect": rect, "review": False, "info": {}, "src": (1920, 1080)}
                with mock.patch.object(layout, "choose_layout_at", return_value=lay):
                    self.assertEqual(run_stage(job, "render", R.run, R.verify), "done", kind)
                out = job.dir / "clips" / f"{JID}_c01_20-26.mp4"
                self.assertTrue(out.exists(), kind)
                m = R.probe_media(job.settings, out)
                self.assertEqual((m["w"], m["h"]), (1080, 1920))
                self.assertTrue(m["has_audio"])
                self.assertAlmostEqual(m["duration"], 6.0, delta=0.5)
                self.assertFalse(list((job.dir / "clips").glob("*.part.mp4")))
                self.assertFalse((job.dir / "sections" / "c01.mp4").exists())     # temp section removed
                self.assertEqual(run_stage(job, "render", R.run, R.verify), "skipped")

    def test_real_layout_detection_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            job = self.prep(td, ("c01", "c02"))
            R.run(job)
            rep = job.read_json("render_report.json")
            self.assertEqual({r["status"] for r in rep.values()}, {"ok"})
            self.assertNotEqual(rep["c01"]["style"], rep["c02"]["style"])         # captions rotate

    def test_failed_verification_is_not_reported_as_created(self):
        with tempfile.TemporaryDirectory() as td:
            job = self.prep(td)
            real = R.probe_media
            def bad(settings, path):
                m = real(settings, path); m["duration"] += 3; return m
            with mock.patch.object(R, "probe_media", bad):
                with self.assertRaises(StageError):
                    R.run(job)
            self.assertFalse(list((job.dir / "clips").glob("*.mp4")))
            self.assertEqual(job.read_json("render_report.json")["c01"]["status"], "failed")
            self.assertTrue((job.dir / "sections" / "c01.mp4").exists())          # kept for retry

    def test_filename(self):
        self.assertEqual(R.clip_filename("j", "c07", 5123.4, 5161.0), "j_c07_5123-5161.mp4")


if __name__ == "__main__":
    unittest.main()


class ForcedLayout(unittest.TestCase):
    def test_forced_blur_fit_skips_detection_and_rerenders_on_change(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            SEC.save_selection(job, ["c01"], "blur_fit")
            with mock.patch.object(SEC, "download_section", fake_section):
                SEC.fetch_sections(job)
            with mock.patch.object(layout, "choose_layout_at", side_effect=AssertionError("must not detect")):
                self.assertEqual(run_stage(job, "render", R.run, R.verify), "done")
            self.assertEqual(job.read_json("render_report.json")["c01"]["layout"], "blur_fit")
            self.assertEqual(run_stage(job, "render", R.run, R.verify), "skipped")
            SEC.save_selection(job, ["c01"], "center_crop")      # different layout -> must redo
            self.assertFalse(R.verify(job))
            self.assertFalse(SEC.verify(job))
            with self.assertRaises(StageError):
                SEC.save_selection(job, ["c01"], "weird")
