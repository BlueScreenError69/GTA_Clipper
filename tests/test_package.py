import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipper import config
from clipper.manifest import atomic_write
from clipper.stages import package as P
from clipper.stages import sections as SEC
from clipper.stages.common import StageError, run_stage
from test_render import JID, S, make_job


def make_clip(job, cid="c01", start=20, end=26):
    out = job.dir / "clips" / f"{JID}_{cid}_{start}-{end}.mp4"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([S.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1080x1920:rate=30", "-f", "lavfi", "-i", "sine", "-t", str(end - start),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)


def prep(td, credit="via StreamerX"):
    job = make_job(td)
    job.settings.scorer_mode = "heuristic"
    SEC.save_selection(job, ["c01"])
    make_clip(job)
    atomic_write(job.dir / "rights.json", json.dumps({
        "url": f"https://youtu.be/{JID}", "creator": "StreamerX", "permission_basis": "clipping program",
        "program_rules": {"required_credit": credit, "allowed_platforms": "YouTube Shorts"}}))
    atomic_write(job.dir / "meta.json", json.dumps({"duration_s": 3700.0, "title": "Big | stream"}))
    sc = job.read_json("scores.json")
    sc["mode"] = "hybrid"
    sc["scored"][0].update(hook_title="He did WHAT?", one_line_reason="clear reaction", event_type="reaction",
                           spoiler_level="none", flags=["possible_music"], penalties=0,
                           score_breakdown={"hook": 18})
    atomic_write(job.dir / "scores.json", json.dumps(sc))
    atomic_write(job.dir / "render_report.json", json.dumps(
        {"c01": {"status": "ok", "layout": "center_crop", "layout_review": True, "style": "yellow_pop"}}))
    return job


class Pure(unittest.TestCase):
    def test_flags(self):
        f = P.clip_flags({"flags": ["caption_check"], "spoiler_level": "major"}, {"layout_review": True}, False)
        self.assertEqual(f, ["caption_check", "credit_unconfirmed", "layout_review", "spoiler_check"])
        self.assertEqual(P.clip_flags({"flags": []}, {}, True), [])

    def test_caption_has_credit_and_hook(self):
        c = P.make_caption("He did WHAT?", "via X", ["#GTA6", "#Shorts"])
        self.assertIn("He did WHAT?", c)
        self.assertIn("via X", c)
        self.assertTrue(c.strip().endswith("#GTA6 #Shorts"))

    def test_summary_under_40_lines_even_with_many_clips(self):
        rows = [{"id": f"c{i:02d}", "start_s": i * 100, "end_s": i * 100 + 30, "score": 90 - i,
                 "hook": "h|x", "reason": "r", "flags": ["possible_music", "layout_review"], "credit": "c"}
                for i in range(40)]
        rights = {"url": "u", "creator": "C", "permission_basis": "b", "program_rules": {}}
        md = P.build_summary("j", {"title": "t", "duration_s": 3600}, rights, "heuristic",
                             {"candidates": 40, "scored": 40}, rows, "/f")
        self.assertLess(len(md.splitlines()), 40)
        self.assertIn("22 more clips", md)
        self.assertNotIn("h|x", md)                              # pipes escaped for the table
        self.assertIn("possible_music (", md)                    # flags grouped one line each


class Stage(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            job = prep(td)
            self.assertEqual(run_stage(job, "package", P.run, P.verify), "done")
            d = job.dir / "review" / "c01"
            info = json.loads((d / "clip.json").read_text())
            self.assertEqual(info["source_timestamps"], "0:00:20-0:00:26")
            self.assertEqual(info["flags"], ["layout_review", "possible_music"])
            self.assertEqual(info["credit"], "via StreamerX")
            self.assertEqual(info["caption_style"], "yellow_pop")
            self.assertTrue((d / "thumb.jpg").stat().st_size > 1000)
            self.assertTrue((d / "clip.mp4").is_symlink() and (d / "clip.mp4").exists())
            cap = (d / "caption.txt").read_text()
            self.assertIn("He did WHAT?", cap); self.assertIn("via StreamerX", cap)
            tr = (d / "transcript.txt").read_text().split()
            self.assertEqual(tr[0], "word0"); self.assertEqual(len(tr), 12)    # only words inside 20-26s
            md = (job.dir / "review" / "summary.md").read_text()
            self.assertIn("Big / stream", md)
            self.assertIn("0:00:20-0:00:26", md)
            self.assertIn("scorer: hybrid", md)
            self.assertIn("Nothing here is uploaded", md)
            self.assertLess(len(md.splitlines()), 40)
            self.assertEqual(run_stage(job, "package", P.run, P.verify), "skipped")

    def test_missing_credit_flagged_with_default(self):
        with tempfile.TemporaryDirectory() as td:
            job = prep(td, credit="")
            P.run(job)
            info = json.loads((job.dir / "review" / "c01" / "clip.json").read_text())
            self.assertEqual(info["credit"], "Clip from StreamerX")
            self.assertIn("credit_unconfirmed", info["flags"])

    def test_refuses_without_verified_clip(self):
        with tempfile.TemporaryDirectory() as td:
            job = prep(td)
            for p in (job.dir / "clips").glob("*.mp4"):
                p.unlink()
            with self.assertRaises(StageError):
                P.run(job)
            self.assertFalse((job.dir / "review" / "summary.md").exists())

    def test_rerun_after_selection_change(self):
        with tempfile.TemporaryDirectory() as td:
            job = prep(td)
            P.run(job)
            SEC.save_selection(job, ["c01", "c02"])
            self.assertFalse(P.verify(job))                      # c02 not packaged -> stage must re-run


if __name__ == "__main__":
    unittest.main()
