"""clipper CLI. Output is kept short on purpose (token budget)."""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config, diskguard
from .manifest import Manifest, STAGES, job_id_from_url
from .stages import rights as rights_stage, fetch, transcribe as transcribe_stage, signals as signals_stage, candidates as cand_stage, score as score_stage, sections as sec_stage, render as render_stage, package as package_stage
from .stages.common import Job, StageError, Waiting, run_stage


def cmd_disk(s, _a) -> int:
    u = diskguard.usage(s)
    print(f"data dir : {s.data_dir}")
    print(f"used     : {u['used_gb']:.2f} GB of {u['cap_gb']:g} GB cap ({u['remaining_gb']:.1f} GB left)")
    print(f"disk free: {u['free_disk_gb']:.1f} GB (floor {u['min_free_gb']:g} GB)")
    try:
        diskguard.check(s, what="disk")
        print("status   : OK")
        return 0
    except diskguard.DiskGuardError as e:
        print(f"status   : BLOCKED - {e}")
        return 2


def cmd_status(s, a) -> int:
    d = s.jobs_dir / a.job_id
    if not (d / "manifest.json").exists():
        print(f"no such job: {a.job_id}")
        return 1
    m = Manifest(d)
    print("\n".join(m.summary()))
    print(f"next: {m.next_stage() or 'complete'}")
    return 0


def _pipeline(job, url=None, basis=None, credit=""):
    steps = [
        ("rights", lambda j: rights_stage.write_rights(j, url, basis, credit), rights_stage.verify),
        ("probe", fetch.probe, lambda j: (j.dir / "meta.json").exists()),
        ("fetch_audio", fetch.fetch_audio, fetch.verify_audio),
        ("transcribe", transcribe_stage.transcribe, transcribe_stage.verify),
        ("signals", signals_stage.run, signals_stage.verify),
        ("candidates", cand_stage.run, cand_stage.verify),
        ("score", score_stage.run, score_stage.verify),
    ]
    for name, fn, ver in steps:
        if name == "rights" and url is None:  # resume: rights already recorded
            if not rights_stage.verify(job):
                print("no rights.json; start with: run <url> --basis ...")
                return 1
            continue
        try:
            res = run_stage(job, name, fn, ver)
        except Waiting as w:
            print(f"{name}: WAITING - {w}")
            return 4
        except StageError as e:
            print(f"{name}: FAILED - {e}")
            return 1
        except Exception as e:  # unexpected: details are in the stage log
            print(f"{name}: FAILED - {type(e).__name__}: {str(e)[:120]} (see logs/{name}.log)")
            return 1
        print(f"{name}: {res}")
    print(f"analysis done. Review data/jobs/{job.job_id}/candidates_compact.json and scores.json, then:")
    print(f"  render {job.job_id} --ids c01,c02   (or --top N), then: package {job.job_id}")
    return 0


def cmd_run(s, a) -> int:
    try:
        jid = job_id_from_url(a.target)
    except ValueError as e:
        print(e)
        return 1
    job = Job.open(s, jid)
    print(f"job: {jid}")
    return _pipeline(job, a.target, a.basis, a.credit)


def cmd_live(s, a) -> int:
    from . import livejob
    try:
        job = livejob.run_live(s, a.url, a.basis, a.credit, a.watch_min, a.window_min)
    except StageError as e:
        print(f"live: FAILED - {e}")
        return 1
    except Exception as e:
        print(f"live: FAILED - {type(e).__name__}: {str(e)[:160]}")
        return 1
    print(f"analysis done. job: {job.job_id}")
    print(f"  render {job.job_id} --top 3 --layout blur_fit, then: package {job.job_id}")
    return 0


def cmd_resume(s, a) -> int:
    if not (s.jobs_dir / a.target / "manifest.json").exists():
        print(f"no such job: {a.target}")
        return 1
    return _pipeline(Job.open(s, a.target))


def cmd_render(s, a) -> int:
    if not (s.jobs_dir / a.job_id / "manifest.json").exists():
        print(f"no such job: {a.job_id}")
        return 1
    job = Job.open(s, a.job_id)
    if not (job.dir / "scores.json").exists():
        print("no scores.json yet; finish the score stage first (resume)")
        return 1
    try:
        ids = sec_stage.choose_ids(job, a.ids.split(",") if a.ids else None, a.top)
        sec_stage.save_selection(job, ids, a.layout)
    except StageError as e:
        print(f"select: FAILED - {e}")
        return 1
    print(f"rendering: {', '.join(ids)}")
    for name, fn, ver in (("fetch_sections", sec_stage.fetch_sections, sec_stage.verify),
                          ("render", render_stage.run, render_stage.verify)):
        try:
            print(f"{name}: {run_stage(job, name, fn, ver)}")
        except StageError as e:
            print(f"{name}: FAILED - {e}")
            return 1
        except Exception as e:
            print(f"{name}: FAILED - {type(e).__name__}: {str(e)[:120]} (see logs/{name}.log)")
            return 1
    rep = job.read_json("render_report.json")
    for i in ids:
        r = rep[i]
        print(f"  {i}  {r['file']}  layout={r['layout']}{' (review)' if r['layout_review'] else ''}  style={r['style']}")
    print(f"clips: {job.dir / 'clips'}")
    return 0


def cmd_package(s, a) -> int:
    if not (s.jobs_dir / a.job_id / "manifest.json").exists():
        print(f"no such job: {a.job_id}")
        return 1
    job = Job.open(s, a.job_id)
    try:
        res = run_stage(job, "package", package_stage.run, package_stage.verify)
    except StageError as e:
        print(f"package: FAILED - {e}")
        return 1
    except Exception as e:
        print(f"package: FAILED - {type(e).__name__}: {str(e)[:120]} (see logs/package.log)")
        return 1
    print(f"package: {res}")
    print(f"summary: {job.dir / 'review' / 'summary.md'}")
    return 0


def _ff(s, tool, *args) -> subprocess.CompletedProcess:
    return subprocess.run([tool, "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True)


def cmd_selftest(s, _a) -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}".rstrip())

    # ffmpeg capabilities
    v = subprocess.run([s.ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True)
    check("ffmpeg has libass (ass/subtitles)", " subtitles " in v.stdout and " ass " in v.stdout)
    # disk guard
    try:
        diskguard.check(s, 0, "selftest")
        check("disk guard passes at 0 bytes", True)
    except diskguard.DiskGuardError as e:
        check("disk guard passes at 0 bytes", False, str(e))
    try:
        diskguard.check(s, int((s.max_data_gb + 1) * diskguard.GB), "selftest")
        check("disk guard blocks over-cap request", False)
    except diskguard.DiskGuardError:
        check("disk guard blocks over-cap request", True)
    # synthetic video + ffprobe verification (written under data/, inside the guard)
    diskguard.check(s, 50 * 1024 ** 2, "selftest")
    with tempfile.TemporaryDirectory(dir=s.data_dir) as td:
        out = Path(td) / "t.mp4"
        r = _ff(s, s.ffmpeg, "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440", "-t", "5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", str(out))
        check("generate 5s test video", r.returncode == 0 and out.exists(), r.stderr[:100])
        if out.exists():
            p = subprocess.run([s.ffprobe, "-v", "error", "-show_entries",
                                "stream=codec_type:format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True)
            check("ffprobe sees video+audio, ~5s",
                  "video" in p.stdout and "audio" in p.stdout and "5.0" in p.stdout)
    # manifest atomic + resume
    with tempfile.TemporaryDirectory(dir=s.data_dir) as td:
        m = Manifest(Path(td))
        m.mark("rights", "done")
        m.mark("probe", "failed", error="x")
        check("manifest resumes at first non-done stage", Manifest(Path(td)).next_stage() == "probe")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clipper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("disk", help="show storage usage vs the cap")
    sub.add_parser("selftest", help="check ffmpeg, disk guard, manifest")
    st = sub.add_parser("status")
    st.add_argument("job_id")
    r = sub.add_parser("run", help="run <youtube_url> --basis '<why you may reuse it>'")
    r.add_argument("target")
    r.add_argument("--basis", required=True, help="permission basis (rights gate)")
    r.add_argument("--credit", default="", help="required credit line from program rules")
    lv = sub.add_parser("live", help="record the live window around the busiest chat moment")
    lv.add_argument("url")
    lv.add_argument("--basis", required=True)
    lv.add_argument("--credit", default="")
    lv.add_argument("--watch-min", type=float, default=5.0, help="max minutes to watch chat first")
    lv.add_argument("--window-min", type=float, default=14.0, help="minutes to record")
    rs = sub.add_parser("resume")
    rs.add_argument("target", help="job id")
    rd = sub.add_parser("render", help="render approved clips: --ids c01,c02 or --top N")
    rd.add_argument("job_id")
    rd.add_argument("--ids")
    rd.add_argument("--top", type=int)
    rd.add_argument("--layout", default="auto", choices=("auto", "blur_fit", "center_crop"),
                    help="blur_fit = landscape video centered on a blurred portrait background")
    pk = sub.add_parser("package", help="build review/ folder + summary.md for rendered clips")
    pk.add_argument("job_id")
    a = ap.parse_args(argv)
    s = config.load()
    return {"disk": cmd_disk, "status": cmd_status, "run": cmd_run, "live": cmd_live, "render": cmd_render, "package": cmd_package, "resume": cmd_resume, "selftest": cmd_selftest}[a.cmd](s, a)


if __name__ == "__main__":
    sys.exit(main())
