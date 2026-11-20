import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipper import config, live
from clipper.manifest import atomic_write
from clipper.signals.chat_activity import ChatActivity, rate_per_second
from clipper.stages import package as P
from clipper.stages import sections as SEC
from clipper.stages.common import Job
from test_render import JID, S, gen, make_job
from test_package import make_clip, prep


def chat_line(mid, usec, kind="liveChatTextMessageRenderer"):
    return json.dumps({"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": {
        kind: {"id": mid, "timestampUsec": str(usec), "authorName": {"simpleText": "someone"},
               "message": {"runs": [{"text": "secret text"}]}}}}}]}})


class Chat(unittest.TestCase):
    def test_parse_dedupes_and_ignores_non_text(self):
        seen = set()
        lines = [chat_line("a", 1_700_000_000_000_000), chat_line("a", 1_700_000_000_000_000),
                 chat_line("b", 1_700_000_005_000_000), chat_line("c", 1, "liveChatPaidMessageRenderer"),
                 "not json"]
        out = live.parse_chat_lines(lines, seen)
        self.assertEqual(out, [1_700_000_000.0, 1_700_000_005.0])
        self.assertEqual(live.parse_chat_lines([chat_line("b", 1_700_000_005_000_000)], seen), [])

    def test_monitor_reads_incrementally_and_scrubs_raw_files(self):
        with tempfile.TemporaryDirectory() as td:
            m = live.ChatMonitor("u", Path(td))
            f = Path(td) / "chat.live_chat.json.part"
            f.write_text(chat_line("a", 1_700_000_000_000_000) + "\n" + chat_line("b", 1_700_000_001_000_000)[:30])
            self.assertEqual(m.poll(), 1)                        # partial last line is not consumed
            with open(f, "a") as fh:
                fh.write(chat_line("b", 1_700_000_001_000_000)[30:] + "\n")
            self.assertEqual(m.poll(), 1)
            self.assertEqual(len(m.times), 2)
            m.stop()
            self.assertFalse(list(Path(td).glob("chat.live_chat.json*")))   # names/text not kept

    def test_busiest_and_window(self):
        times = [1000 + i for i in range(0, 600, 20)] + [1300 + i * 2 for i in range(30)]   # dense around 1300-1360
        c, n = live.busiest(times)
        self.assertTrue(1290 <= c <= 1360 and n >= 25)
        self.assertIsNone(live.busiest([1.0, 2.0]))
        self.assertIsNone(live.busiest([]))

    def test_choose_window_clamps_to_playlist(self):
        pl, edge = 10_000.0, 10_900.0
        s, e = live.choose_window(10_450.0, edge, pl, 840)          # peak near the middle
        self.assertAlmostEqual(e - s, 840 - 0 if (edge - (pl + 60)) >= 840 else edge - (pl + 60))
        self.assertGreaterEqual(s, pl + 60)
        self.assertLessEqual(e, edge)
        s, e = live.choose_window(10_050.0, edge, pl, 600)          # peak near the start -> clamped
        self.assertEqual(s, pl + 60)
        s, e = live.choose_window(None, edge, pl, 600)              # no chat peak -> latest window
        self.assertEqual(e, edge)


class ChatSignal(unittest.TestCase):
    def job(self, td):
        job = make_job(td)
        job.settings.chat_activity, job.settings.analytics = True, False
        atomic_write(job.dir / "meta.json", json.dumps({"duration_s": 300.0}))
        quiet = [i * 5.0 for i in range(60)]                     # 0.2 msg/s baseline
        burst = [150 + i * 0.25 for i in range(120)]            # 4 msg/s at 150-180
        atomic_write(job.dir / "chat.json", json.dumps({"times": quiet + burst}))
        return job

    def test_burst_found(self):
        with tempfile.TemporaryDirectory() as td:
            job = self.job(td)
            res = ChatActivity(job.settings).compute(job)
            self.assertTrue(any(t.start <= 165 <= t.end for t in res))
            self.assertTrue(all(0 <= t.score <= 1 for t in res))

    def test_rate_per_second_length(self):
        self.assertEqual(len(rate_per_second([1.0, 2.0], 10, 3)), 10)


class LocalSource(unittest.TestCase):
    def test_sections_cut_from_local_recording_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            job = make_job(td)
            gen(job.dir / "rec.mp4", 40)
            meta = job.read_json("meta.json")
            meta.update(local_file="rec.mp4", duration_s=40.0)
            atomic_write(job.dir / "meta.json", json.dumps(meta))
            cands = job.read_json("candidates_compact.json")
            cands["candidates"] = [{**cands["candidates"][0], "start_s": 10, "end_s": 16}]
            atomic_write(job.dir / "candidates_compact.json", json.dumps(cands))
            SEC.save_selection(job, ["c01"])
            with mock.patch.object(SEC, "download_section", side_effect=AssertionError("no network")):
                SEC.fetch_sections(job)
            info = json.loads((job.dir / "sections" / "c01.json").read_text())
            self.assertEqual((info["section_start"], info["offset"]), (7.0, 3.0))
            self.assertTrue((job.dir / "sections" / "c01.mp4").exists())

    def test_package_shows_stream_time_offset(self):
        with tempfile.TemporaryDirectory() as td:
            job = prep(td)
            meta = job.read_json("meta.json")
            meta["stream_offset_s"] = 7200.0                     # window started 2h into the stream
            atomic_write(job.dir / "meta.json", json.dumps(meta))
            P.run(job)
            info = json.loads((job.dir / "review" / "c01" / "clip.json").read_text())
            self.assertEqual(info["source_timestamps"], "2:00:20-2:00:26")
            self.assertIn("2:00:20-2:00:26", (job.dir / "review" / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
