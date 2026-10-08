"""Execute the sweep from its config file. A loop, not a script full of decisions.

    python tools/sweep_configs.py --json sweep_grid.json     # generate the config
    python tools/run_sweep.py sweep_grid.json --dry-run      # show the plan
    python tools/run_sweep.py sweep_grid.json                # run it, serially
    python tools/run_sweep.py sweep_grid.json --rep arcB5    # one chain only
    python tools/run_sweep.py sweep_grid.json --resume       # skip finished points

Every decision lives in `sweep_configs.py` and is frozen into the JSON. This module only
orders the runs, enforces the two invariants below, and adjudicates.

TWO INVARIANTS IT ENFORCES, both from runs already lost:

1. ADJUDICATION IS SERIAL WITH THE SOLVE. Two runs died in the 09-09 session to `verify`
   running concurrently with a solve on this 15 GB machine. Each point here is
   solve-then-adjudicate, and nothing overlaps.

2. A WARM RUN DOES NOT START WITHOUT ITS DONOR'S FINAL h5. A run stopped by hand before
   `opt.optimize()` returns writes checkpoints but no final coilset, and `--start-from`
   pointed at a missing file would either fail late or -- worse -- a stale file from a
   previous attempt would silently seed the wrong geometry.

PARALLELISM. Chains are sequential WITHIN a representation, so the width is one worker
per representation, never one per point. This driver is deliberately serial: at
MemoryMax=12G per run against 15 GB total, this machine fits exactly one. To run wide,
group by `rep` and dispatch each group to its own CPU container (SWEEP_BLUEPRINT 8.3);
the dependency edges never cross representations, so the groups are independent.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def final_h5(rundir):
    """The run's final coilset, or None. Checkpoints do not count."""
    if not os.path.isdir(rundir):
        return None
    for f in sorted(os.listdir(rundir)):
        if f.endswith(".h5") and "ckpt" not in f and "checkpoint" not in f:
            return os.path.join(rundir, f)
    return None


def adjudication(rundir):
    """The run's adjudication record, or None if absent or unreadable.

    Parsed, not just stat'ed: a crash part-way through the write leaves a truncated
    JSON file, and treating that as "finished" would silently drop a sweep point.
    """
    f = os.path.join(rundir, "adjudication.json")
    if not os.path.exists(f):
        return None
    try:
        rec = json.load(open(f))
    except Exception:
        return None
    return rec if "normalized_BdotN" in rec else None


def done(run):
    """A point is finished when it has a final coilset AND a readable adjudication."""
    return (
        final_h5(run["out_dir"]) is not None
        and adjudication(run["out_dir"]) is not None
    )


def state(run):
    """`done` / `solved` / `partial` / `todo`, for --status and for resume decisions."""
    d = run["out_dir"]
    if done(run):
        return "done"
    if final_h5(d) is not None:
        return "solved"  # solve finished, adjudication did not
    if os.path.isdir(d) and glob.glob(os.path.join(d, "*.h5")):
        return "partial"  # checkpoints only -- crashed or killed mid-solve
    return "todo"


def order(runs):
    """Dependency order: every donor before its dependants, chains kept together."""
    by = {r["id"]: r for r in runs}
    out, seen = [], set()

    def visit(r):
        if r["id"] in seen:
            return
        seen.add(r["id"])
        d = r["depends_on"]
        if d is not None:
            if d not in by:
                raise SystemExit(
                    f"{r['id']} depends on {d}, which is not in this config"
                )
            visit(by[d])
        out.append(r)

    for r in runs:
        visit(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--rep", default=None, help="restrict to one representation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip finished points")
    ap.add_argument(
        "--status",
        action="store_true",
        help="report what is done / solved-but-unadjudicated / partial / "
        "to-do, and exit",
    )
    ap.add_argument("--mem", default="12G", help="cgroup cap per run")
    ap.add_argument(
        "--no-cgroup",
        action="store_true",
        help="skip systemd-run. WSL2 has NO OOM killer -- without the cap a "
        "run that grows takes the machine down instead of itself.",
    )
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    runs = cfg["runs"]
    if a.rep:
        runs = [r for r in runs if r["rep"] == a.rep]
        if not runs:
            raise SystemExit(f"no runs for rep {a.rep!r}")
    runs = order(runs)

    os.chdir(ROOT)  # every path in the config is BUNDLE-root relative

    if a.status:
        by = {}
        for r in runs:
            by.setdefault(state(r), []).append(r)
        for k, label in (
            ("done", "done"),
            ("solved", "SOLVED, NOT ADJUDICATED"),
            ("partial", "PARTIAL (checkpoints only)"),
            ("todo", "to do"),
        ):
            got = by.get(k, [])
            print(f"{label:28s} {len(got):3d}")
            if k in ("solved", "partial"):
                for r in got:
                    n = len(glob.glob(os.path.join(r["out_dir"], "ckpt_it*.h5")))
                    print(f"    {r['id']}  ({n} checkpoints)")
        if by.get("solved"):
            print("\n  --resume will adjudicate these without re-solving.")
        if by.get("partial"):
            print(
                "\n  --resume will RE-RUN these from scratch. If one died late, "
                "consider instead:"
            )
            print(
                "    python tools/promote_checkpoint.py <rundir>   # bank the last "
                "feasible checkpoint"
            )
        return
    todo = [r for r in runs if not (a.resume and done(r))]
    skipped = len(runs) - len(todo)
    mins = sum(r["minutes"] for r in todo)
    print(
        f"{len(todo)} runs to do"
        + (f" ({skipped} already finished, skipped)" if skipped else "")
        + f"  ~{mins} min = {mins/60:.1f} h sequential\n"
    )

    for i, r in enumerate(todo, 1):
        if not a.dry_run:  # a plan must not leave artefacts behind
            os.makedirs(r["out_dir"], exist_ok=True)
        log = f"runs/{r['id']}.log"

        dep = r["depends_on"]
        # In a dry run the donor legitimately does not exist yet -- the whole point is
        # to show the plan BEFORE anything is solved.
        if dep is not None and not a.dry_run:
            src = f"runs/{dep}/{dep}.h5"
            if not os.path.exists(src):
                have = final_h5(f"runs/{dep}")
                raise SystemExit(
                    f"\n{r['id']} warm-starts from {src}, which does not exist.\n"
                    + (
                        f"  {dep} did produce {have} -- the tag and filename disagree; "
                        f"fix the config.\n"
                        if have
                        else f"  {dep} has no final coilset. If it was stopped by hand it "
                        f"wrote only checkpoints,\n  and a warm start from a checkpoint "
                        f"is a different run -- do it deliberately, not by accident.\n"
                    )
                )

        cmd = list(r["argv"])
        pre = (
            []
            if a.no_cgroup
            else [
                "systemd-run",
                "--user",
                "--scope",
                "-q",
                f"-pMemoryMax={a.mem}",
                "-pMemorySwapMax=0",
            ]
        )
        full = pre + ["python", "-u", "tools/diag_ymask.py"] + cmd

        adj = [
            "python",
            "-u",
            "tools/adjudicate.py",
            r["out_dir"],
            "--length-max",
            f"{r['length_max']:.6f}",
        ]
        # VERIFY_N=400 is not converged for d_cc on the most contorted coilsets (still
        # moving 1.2% at N=800). Check it where that bites: the relaxed end.
        if r["mult"] >= 1.4:
            adj.append("--dcc-converge")

        head = (
            f"[{i}/{len(todo)}] {r['id']}  {r['rep']} xL={r['mult']} "
            f"rung {r['rung']} {r['role']}  ~{r['minutes']} min"
        )
        if a.dry_run:
            print(head)
            print("   " + " ".join(full) + f" >> {log} 2>&1")
            print("   " + " ".join(adj))
            continue

        print(head, flush=True)
        # A crash between the solve and its adjudication used to cost a full re-solve:
        # the finished coilset was already on disk and was thrown away. If the final h5
        # is there, the expensive part is done -- adjudicate it and move on.
        if a.resume and final_h5(r["out_dir"]) is not None:
            print(
                "   solve already complete (final h5 present) -- adjudicating only",
                flush=True,
            )
        else:
            t0 = time.time()
            # Append, never truncate: on a resume the previous log is the only record of
            # why the run died, and overwriting it destroys the evidence.
            with open(log, "a") as fh:
                fh.write(
                    f"\n===== attempt started {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    f" =====\n"
                )
                fh.flush()
                rc = subprocess.call(full, stdout=fh, stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 60
            if rc != 0:
                print(f"   SOLVE FAILED rc={rc} after {dt:.1f} min -- see {log}")
                print(
                    "   stopping: a chain's later rungs would warm-start from nothing."
                )
                sys.exit(rc)
            print(
                f"   solved in {dt:.1f} min (estimate was {r['minutes']})", flush=True
            )

        # serial, never alongside a solve -- see the module docstring
        with open(log, "a") as fh:
            rc = subprocess.call(adj, stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"   ADJUDICATION FAILED rc={rc} -- see {log}")
            sys.exit(rc)
        rec = json.load(open(os.path.join(r["out_dir"], "adjudication.json")))
        v = "FEASIBLE" if rec["feasible"] else "FAILS " + ",".join(rec["failed"])
        print(
            f"   bn {rec['normalized_BdotN']:.4e}   linked pairs "
            f"{rec['linking']['n_linked_pairs']}   {v}",
            flush=True,
        )


if __name__ == "__main__":
    main()
