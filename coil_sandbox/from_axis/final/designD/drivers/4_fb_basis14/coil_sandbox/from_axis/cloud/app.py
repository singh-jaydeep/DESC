"""Run from_axis jobs on Modal (or locally with the same wrapper), writing straight to a persistent Volume.

    modal run --detach app.py --jobs accept_ss4          # one job, detached (survives this machine sleeping)
    modal run --detach app.py --jobs screen_b14          # a group: its jobs run in PARALLEL, one container each
    modal run app.py --list                              # job names and groups (jobs.py)
    python app.py --local accept_ss4 [--root DIR]        # the same wrapper without Modal (test the plumbing)
    python harvest.py NAME ...                           # pull finished runs back and verify them

What the wrapper (`run_job`) guarantees, on Modal or locally:
  * every job writes into its own directory <root>/<name>/ and nowhere else ({OUT} in the command);
  * stdout+stderr are appended line by line to <root>/<name>/run.log; a heartbeat file is touched every 2 min;
  * on Modal the Volume is committed every 2 min and at exit, so a killed container loses at most ~2 min;
  * a job is a list of steps; a step that already left its `done` file is skipped, so a retried or re-submitted job
    continues where it stopped (the scripts themselves resume inside a step: --auto-resume, shells_done.json);
  * a failed step raises, which makes Modal retry the function (3 times) into the same directory;
  * meta.json records the git commit of the code that was shipped, the command, the GPU and every attempt.

The image holds a COPY of desc/ and coil_sandbox/ taken when `modal run` is invoked: commit first, the commit hash is
read here and stored with every run.
"""
import json
import os
import shlex
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # .../DESC
sys.path.insert(0, HERE)
try:
    from jobs import GROUPS, JOBS  # noqa: E402
except ImportError:  # inside the container app.py is imported alone; the job dict arrives as an argument
    GROUPS, JOBS = {}, {}

VOLUME = os.environ.get("FA_VOLUME", "from-axis-runs")
GPU = os.environ.get("FA_GPU", "A100-40GB")
REMOTE_REPO, REMOTE_ROOT = "/repo", "/vol"


def git_state():
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "desc", "coil_sandbox/sandbox",
                                         "coil_sandbox/single_stage.py", "coil_sandbox/from_axis"], text=True).strip()
        return dict(commit=h, dirty=[ln for ln in dirty.splitlines() if not ln.endswith(("/", ".log"))][:40])
    except Exception as e:  # noqa: BLE001
        return dict(commit=None, error=repr(e))


def run_job(job, root, repo, commit=None, volume=None):
    """Run one job (dict from jobs.py) under `root`; see the module docstring. Returns the job's summary dict."""
    out = os.path.join(root, job["name"])
    os.makedirs(out, exist_ok=True)
    meta_f = os.path.join(out, "meta.json")
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else dict(name=job["name"], attempts=[])
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    att = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, gpu=gpu, steps=[])
    meta["attempts"].append(att)
    meta["job"] = job

    def save_meta():
        json.dump(meta, open(meta_f + ".tmp", "w"), indent=1)
        os.replace(meta_f + ".tmp", meta_f)

    stop = threading.Event()

    def keeper():  # heartbeat + Volume commit
        while not stop.wait(120):
            open(os.path.join(out, "heartbeat"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            if volume is not None:
                try:
                    volume.commit()
                except Exception as e:  # noqa: BLE001  (a file mid-write: the next tick gets it)
                    print(f"[keeper] commit failed: {e!r}", flush=True)

    threading.Thread(target=keeper, daemon=True).start()
    env = dict(os.environ, PYTHONUNBUFFERED="1", OUT=out, FA_ROOT=root, **job.get("env", {}))
    if volume is None:  # local run: use this machine's device choice unless the job says otherwise
        env.setdefault("SANDBOX_DEVICE", "cpu")
    else:
        env.setdefault("SANDBOX_DEVICE", "gpu")
    rc = 0
    try:
        with open(os.path.join(out, "run.log"), "a") as lf:
            lf.write(f"\n===== attempt {len(meta['attempts'])} {att['started']} commit {commit} gpu {gpu}\n")
            lf.flush()
            for i, step in enumerate(job["steps"]):
                mark = os.path.join(out, f"step{i}.done")
                cmd = [c.replace("{OUT}", out).replace("{ROOT}", root).replace("{REPO}", repo) for c in step["cmd"]]
                rec = dict(i=i, cmd=cmd)
                att["steps"].append(rec)
                if os.path.exists(mark):
                    rec["skipped"] = True
                    lf.write(f"--- step {i} already done, skipped\n")
                    continue
                lf.write(f"--- step {i}: {' '.join(shlex.quote(c) for c in cmd)}\n")
                lf.flush()
                t = time.time()
                rc = subprocess.run(cmd, cwd=os.path.join(repo, step.get("cwd", "coil_sandbox/from_axis/ss")), env=env,
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
                rec.update(rc=rc, minutes=round((time.time() - t) / 60, 1))
                save_meta()
                if rc != 0:
                    break
                open(mark, "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            att["finished"], att["rc"] = time.strftime("%Y-%m-%d %H:%M:%S"), rc
            if rc == 0:
                open(os.path.join(out, "DONE"), "w").write(att["finished"] + "\n")
    finally:
        stop.set()
        save_meta()
        if volume is not None:
            volume.commit()
    if rc != 0:
        raise RuntimeError(f"job {job['name']} failed at step {len(att['steps']) - 1} (rc {rc}); see {out}/run.log")
    return dict(name=job["name"], out=out, minutes=sum(s.get("minutes", 0) for s in att["steps"]))


def expand(names):
    out = []
    for n in names.split(","):
        out += GROUPS[n] if n in GROUPS else [n]
    missing = [n for n in out if n not in JOBS]
    if missing:
        raise SystemExit(f"unknown job(s) {missing}; --list shows the names")
    return out


def listing():
    for g, members in GROUPS.items():
        print(f"group {g}: {', '.join(members)}")
    for n, j in JOBS.items():
        print(f"  {n:28s} {j['what']}")


try:
    import modal
except ImportError:  # local mode works without it
    modal = None

if modal is not None:
    app = modal.App("from-axis")
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    SKIP = ["**/*.log", "**/*.png", "**/*.html", "**/__pycache__/**", "**/*.pyc", "**/trace.npz", "**/cloud_runs/**"]
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "tk", "libtk8.6")  # something in the import chain loads tkinter
        .env({"MPLBACKEND": "Agg"})
        .pip_install_from_requirements(os.path.join(HERE, "requirements.txt"))
        .add_local_dir(os.path.join(REPO, "desc"), f"{REMOTE_REPO}/desc", ignore=["**/__pycache__/**"])
        .add_local_dir(os.path.join(REPO, "coil_sandbox", "sandbox"), f"{REMOTE_REPO}/coil_sandbox/sandbox",
                       ignore=["**/__pycache__/**"])
        .add_local_dir(os.path.join(REPO, "coil_sandbox", "bounds"), f"{REMOTE_REPO}/coil_sandbox/bounds")
        .add_local_file(os.path.join(REPO, "coil_sandbox", "single_stage.py"),
                        f"{REMOTE_REPO}/coil_sandbox/single_stage.py")
        .add_local_dir(os.path.join(REPO, "coil_sandbox", "from_axis"), f"{REMOTE_REPO}/coil_sandbox/from_axis",
                       ignore=SKIP)
    )

    @app.function(image=image, gpu=GPU, volumes={REMOTE_ROOT: vol}, timeout=24 * 3600, cpu=8, memory=32768,
                  retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=1.0))
    def remote(job: dict, commit: str):
        return run_job(job, REMOTE_ROOT, REMOTE_REPO, commit=commit, volume=vol)

    @app.local_entrypoint()
    def main(jobs: str = "", list: bool = False):  # noqa: A002
        if list or not jobs:
            listing()
            return
        g = git_state()
        if g.get("dirty"):
            print("WARNING: uncommitted changes are being shipped (the stored commit will not describe this code):")
            print("  " + "\n  ".join(g["dirty"]))
        names = expand(jobs)
        calls = [(n, remote.spawn(JOBS[n] | {"name": n}, g.get("commit"))) for n in names]
        for n, c in calls:
            print(f"submitted {n}: call {c.object_id}   (volume {VOLUME}:/{n}, gpu {GPU})")
        print("detached runs keep going; follow with:  modal volume get", VOLUME, "<name>/run.log -   |   python harvest.py <name>")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="local mode: the same wrapper without Modal")
    ap.add_argument("--local", required=False)
    ap.add_argument("--root", default=os.path.join(HERE, "..", "ss", "cloud_runs", "_local"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.local:
        listing()
        raise SystemExit(0)
    for n in expand(a.local):
        print(run_job(JOBS[n] | {"name": n}, os.path.abspath(a.root), REPO, commit=git_state().get("commit")))
