"""Adjudicate every checkpoint of a run and promote the last cleanly feasible one.

    python tools/promote_checkpoint.py runs/sw_arcB5_L1p2            # scan + promote
    python tools/promote_checkpoint.py runs/sw_arcB5_L1p2 --scan     # scan only
    python tools/promote_checkpoint.py runs/sw_arcB5_L1p2 --keep-going

Two jobs, both learned the hard way.

**1. It rescues a run that was stopped by hand.** A run killed before `opt.optimize()`
returns writes checkpoints but no final `h5` and no JSON record. Seven planar runs were
lost that way on 2026-09-09 -- they had to be reconstructed offline, and the status board
carried a void result for a week because of it. Worse, in a continuation chain the next
rung cannot start at all: `run_sweep.py` refuses to warm-start from a missing donor.
Promoting a checkpoint to `<tag>.h5` repairs the chain.

**2. It refuses to trust the LAST iterate, which is the real trap.** Measured on
`planar_N7_cold_roa320`: `bn` keeps falling to 3.3858e-02 at it=350 and looks like the
best point of the run -- but `kappa` drifts to 1.0048 from it=200 and the run then trades
that drift for `d_cc`. **The apparent improvement past it=175 IS the constraint drift.**
It was caught only by scanning every checkpoint's margins instead of adjudicating the
last one. So this scans all of them and takes the last one that is *cleanly* feasible.

A promoted point is not a normal one. It is recorded as such in `promotion.json` and
`results.py` shows it, because it was stopped rather than converged: its `bn` is a lower
bound on what the run would have reached, and it was chosen by this rule rather than by
the solver's own termination.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def checkpoints(rundir):
    """Every `ckpt_itNNNNN.h5`, oldest first, plus the rolling `checkpoint.h5`."""
    num = []
    for f in glob.glob(os.path.join(rundir, "ckpt_it*.h5")):
        m = re.search(r"ckpt_it(\d+)\.h5$", f)
        if m:
            num.append((int(m.group(1)), f))
    num.sort()
    roll = os.path.join(rundir, "checkpoint.h5")
    if os.path.exists(roll):
        num.append((10**9, roll))   # unknown iteration, but it is the newest
    return num


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--scan", action="store_true", help="report only, promote nothing")
    ap.add_argument("--keep-going", action="store_true",
                    help="adjudicate every checkpoint even after finding a feasible one")
    ap.add_argument("--length-max", type=float, default=None, dest="length_max",
                    help="swept length bound; taken from sweep_grid.json if omitted")
    a = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    rundir = a.rundir.rstrip("/")
    tag = os.path.basename(rundir)
    cks = checkpoints(rundir)
    if not cks:
        raise SystemExit(f"{rundir}: no checkpoints. Was the run started with "
                         f"--check-every?")

    L = a.length_max
    if L is None:
        cfg = os.path.join(ROOT, "sweep_grid.json")
        if os.path.exists(cfg):
            for r in json.load(open(cfg))["runs"]:
                if r["id"] == tag:
                    L = r["length_max"]
        if L is None:
            raise SystemExit(f"{tag} not in sweep_grid.json; pass --length-max")

    final = os.path.join(rundir, f"{tag}.h5")
    if os.path.exists(final) and not a.scan:
        raise SystemExit(f"{final} already exists -- this run finished normally. "
                         f"Nothing to promote. Use --scan to inspect its checkpoints.")

    from desc.examples import get
    from desc.io import load

    import verify as V
    from adjudicate import linking_matrix
    from diag_ymask import _source_grid, PAPER

    eq = get("precise_QA")
    print(f"{tag}: {len(cks)} checkpoints, length judged against {L:.6f} m\n")
    print(f"{'iter':>7} {'bn':>12} {'length':>8} {'kappa':>8} {'kMS':>8} "
          f"{'d_cc':>8} {'d_pc':>8} {'linked':>6}  verdict")
    print("-" * 84)

    rows, best = [], None
    for it, path in cks:
        cs = load(path)
        src, _ = _source_grid(cs, PAPER["gl_m"], PAPER["uniform_src_N"])
        v = V.verify(cs, eq, src, n=V.VERIFY_N, label=tag)
        b = v["bounds"]
        lk = linking_matrix(cs)
        RT = 1e-8
        mg = dict(length=v["max_length"] / L,
                  kappa=v["max_kappa"] / b["kappa_max"],
                  kappa_MS=v["max_kappa_MS"] / b["kappa_ms_max"],
                  d_cc=v["d_cc"] / b["d_cc_min"],
                  d_pc=v["d_pc"] / b["d_pc_min"])
        failed = [k for k in ("length", "kappa", "kappa_MS") if mg[k] > 1 + RT]
        failed += [k for k in ("d_cc", "d_pc") if mg[k] < 1 - RT]
        if lk["n_linked_pairs"]:
            failed.append("linked")
        label = "it=" + (str(it) if it < 10**9 else "roll")
        print(f"{label:>7} {v['normalized_BdotN']:12.4e} {mg['length']:8.4f} "
              f"{mg['kappa']:8.4f} {mg['kappa_MS']:8.4f} {mg['d_cc']:8.4f} "
              f"{mg['d_pc']:8.4f} {lk['n_linked_pairs']:6d}  "
              f"{'FEASIBLE' if not failed else 'no: ' + ','.join(failed)}", flush=True)
        rec = dict(iteration=None if it >= 10**9 else it, file=os.path.basename(path),
                   bn=v["normalized_BdotN"], max_BdotN_over_B=v["max_BdotN_over_B"],
                   margins={k: round(x, 5) for k, x in mg.items()},
                   linked=lk["n_linked_pairs"], failed=failed)
        rows.append(rec)
        if not failed:
            best = (it, path, rec)
        elif best is not None and not a.keep_going:
            print(f"        ^ first infeasible checkpoint after a feasible one. "
                  f"Later ones are not scanned (--keep-going to force);\n"
                  f"          past this point an improving bn is usually the "
                  f"constraint drift, not progress.")
            break

    if best is None:
        print("\nNO CLEANLY FEASIBLE CHECKPOINT. Nothing to promote; this run has no "
              "bankable point.\nIf a chain depends on it, that chain stops here.")
        json.dump(dict(tag=tag, promoted=None, checkpoints=rows),
                  open(os.path.join(rundir, "promotion.json"), "w"), indent=1)
        raise SystemExit(2)

    it, path, rec = best
    print(f"\nlast cleanly feasible: {rec['file']}  bn = {rec['bn']:.4e}")
    if a.scan:
        print("(--scan: nothing written)")
        return
    shutil.copy2(path, final)
    json.dump(dict(tag=tag, promoted=rec, checkpoints=rows,
                   note=("Stopped rather than converged. `bn` is a LOWER BOUND on what "
                         "this run would have reached, and the iterate was chosen by "
                         "last-cleanly-feasible rather than by the solver's own "
                         "termination.")),
              open(os.path.join(rundir, "promotion.json"), "w"), indent=1)
    print(f"promoted -> {final}")
    print(f"\nnow adjudicate it:\n"
          f"  python tools/adjudicate.py {rundir} --length-max {L:.6f}"
          + ("  --dcc-converge" if L / 5.153350 >= 1.4 else ""))


if __name__ == "__main__":
    main()
