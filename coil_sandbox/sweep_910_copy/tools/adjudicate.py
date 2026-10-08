"""Adjudicate one solved sweep point. Run AFTER the solve, never alongside it.

    python tools/adjudicate.py runs/<tag> --length-max <swept L>
    python tools/adjudicate.py runs/<tag> --length-max 7.214690 --dcc-converge

Writes `<rundir>/adjudication.json` and prints a verdict block.

WHY THIS IS A SEPARATE STEP. Two runs died in the 09-09 session to `verify` running
concurrently with a solve on this 15 GB machine (HANDOFF_sweep_configs sec 5.4). The
sweep runner calls this serially after each solve completes.

WHAT IT ADDS OVER `verify.verify()`, all of it load-bearing:

1. THE PAIRWISE LINKING MATRIX. `verify`'s `n_linked_pairs` is the GL estimate reduced
   to `int(max|Lk| > 0.5)` -- a BOOLEAN. It reported "1" for coilsets carrying SIX
   linked pairs. Two points in this campaign were banked as improvements before anyone
   read the field, and both improvements WERE the linking (8.6% on `planar_N14_cert`,
   18.4% on an unguarded N=4->7 continuation). This reports the actual pair list.

   Grid: `LinearGrid(N=100)`. 8o measured the SIGN identical at N=50/100/150/200 and
   against an N=400 reference -- we need it only to +/-0.5, so a coarse grid is exact.
   Memory in `_compute_linking_number` is the (ncoil, ncol, nnode, nnode, 3) broadcast,
   i.e. O(N^2): 16x16 at 801 nodes is 3.9 GB, at 201 nodes ~250 MB.

2. kappa_MS. `verify.verify()` COMPUTES `max_kappa_MS` but does not put it in `checks`,
   so its `feasible` verdict ignores mean-square curvature entirely -- the constraint
   that voided `al_planar_N14` (10.9% over while satisfying the pointwise bound).

3. LENGTH ADJUDICATED AGAINST THE SWEPT BOUND. `verify` always adjudicates against the
   unbacked-off PAPER length, so every relaxed point reads `NO / length`. Correct and
   intended, and meaningless for this sweep: a x1.4 point is MEANT to exceed 5.153350.

4. d_cc GRID CONVERGENCE, on request. VERIFY_N=400 is converged for curvature
   everywhere and for d_cc everywhere EXCEPT the most contorted coilsets, where it
   still moves 1.2% going to N=800 and has not converged even there. Check it per point
   at the relaxed end rather than assuming the adjudication grid is adequate because it
   is finer than the enforcement grid (SWEEP_BLUEPRINT sec 5.4).

Every number reported here is against the PAPER bounds, never the enforced ones. The
backoffs are an implementation detail and are never reported (SWEEP_BLUEPRINT sec 5.3).
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LINK_GRID_N = 100
LINK_THRESHOLD = 0.5


def solved_h5(rundir):
    """The run's FINAL coilset, never a checkpoint.

    Checkpoints are written every `--check-every` steps and a killed run leaves one as
    its newest h5, so picking by mtime silently adjudicates an intermediate iterate.
    """
    hits = [
        f
        for f in glob.glob(os.path.join(rundir, "*.h5"))
        if "ckpt" not in os.path.basename(f) and "checkpoint" not in os.path.basename(f)
    ]
    if not hits:
        raise SystemExit(
            f"{rundir}: no final coilset (only checkpoints?). A run stopped by hand "
            f"before opt.optimize() returned writes no final h5 -- adjudicate the "
            f"checkpoint explicitly with --h5 if that is what you mean to do."
        )
    return sorted(hits)[0]


def linking_matrix(coilset, n=LINK_GRID_N):
    """Every pair's linking number, and the list of the linked ones.

    Returns the FULL upper triangle, so a caller can see six linked pairs as six.
    """
    import numpy as np

    from desc.grid import LinearGrid

    lk = np.asarray(coilset._compute_linking_number(grid=LinearGrid(N=n)))
    ncoil = lk.shape[0]
    pairs, worst = [], 0.0
    for i in range(ncoil):
        for j in range(i + 1, ncoil):
            v = float(lk[i, j])
            worst = max(worst, abs(v))
            if abs(v) > LINK_THRESHOLD:
                pairs.append([i + 1, j + 1, round(v, 4)])  # 1-based, as 8o reports them
    return {
        "linking_grid_N": n,
        "n_coils": int(ncoil),
        "n_pairs_checked": ncoil * (ncoil - 1) // 2,
        "linked_pairs": pairs,
        "n_linked_pairs": len(pairs),
        "max_abs_linking_number": worst,
    }


def dcc_convergence(coilset, ns=(400, 800)):
    """min d_cc on successive grids. A moving value means the point is NOT adjudicated."""
    import numpy as np  # noqa: F401
    from bench.bench import geometry_metrics
    from verify import _dcc_segment

    from desc.grid import LinearGrid

    out = {}
    for n in ns:
        g = LinearGrid(N=n)
        # _dcc_segment needs the point-sampled value only as a reference for its own
        # disagreement field; the segment minimum is what is adjudicated.
        pt = geometry_metrics(coilset, _EQ, g, surf_MN=(8, 8))["coil_coil_distance"]
        out[f"d_cc_N{n}"] = float(_dcc_segment(coilset, n, pt)["coil_coil_distance"])
    lo, hi = out[f"d_cc_N{ns[0]}"], out[f"d_cc_N{ns[-1]}"]
    out["d_cc_drift_rel"] = abs(hi - lo) / max(abs(lo), 1e-300)
    return out


_EQ = None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--h5", default=None, help="adjudicate this coilset, not the final")
    ap.add_argument(
        "--length-max",
        type=float,
        default=None,
        dest="length_max",
        help="the SWEPT length bound this point was solved against. "
        "Length is judged against this; every other bound against "
        "the paper value.",
    )
    ap.add_argument(
        "--dcc-converge",
        action="store_true",
        help="also check d_cc at N=400 vs N=800. Do this at the relaxed "
        "end, where VERIFY_N=400 is NOT converged.",
    )
    ap.add_argument("--verify-n", type=int, default=None, dest="verify_n")
    a = ap.parse_args()

    global _EQ
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import verify as V
    from diag_ymask import PAPER, _source_grid

    from desc.examples import get
    from desc.io import load

    _EQ = get("precise_QA")
    path = a.h5 or solved_h5(a.rundir)
    cs = load(path)
    src, src_kind = _source_grid(cs, PAPER["gl_m"], PAPER["uniform_src_N"])

    n = a.verify_n or V.VERIFY_N
    v = V.verify(cs, _EQ, src, n=n, label=os.path.basename(a.rundir.rstrip("/")))
    V.print_verify(v)

    b = v["bounds"]
    L_bound = a.length_max if a.length_max is not None else b["length_max"]
    lk = linking_matrix(cs)

    # The verdict, against PAPER bounds, with length against the SWEPT bound.
    FEAS_RTOL = 1e-8
    checks = {
        "length": v["max_length"] <= L_bound * (1 + FEAS_RTOL),
        "kappa": v["max_kappa"] <= b["kappa_max"] * (1 + FEAS_RTOL),
        "kappa_MS": v["max_kappa_MS"] <= b["kappa_ms_max"] * (1 + FEAS_RTOL),
        "d_cc": v["d_cc"] >= b["d_cc_min"] * (1 - FEAS_RTOL),
        "d_pc": v["d_pc"] >= b["d_pc_min"] * (1 - FEAS_RTOL),
        "unlinked": lk["n_linked_pairs"] == 0,
    }
    margins = {
        "length": v["max_length"] / L_bound,
        "kappa": v["max_kappa"] / b["kappa_max"],
        "kappa_MS": v["max_kappa_MS"] / b["kappa_ms_max"],
        "d_cc": v["d_cc"] / b["d_cc_min"],
        "d_pc": v["d_pc"] / b["d_pc_min"],
    }
    rec = {
        "rundir": a.rundir,
        "coilset": path,
        "source_grid": src_kind,
        "verify_grid_N": n,
        "length_bound_adjudicated": L_bound,
        "length_is_swept": a.length_max is not None,
        "normalized_BdotN": v["normalized_BdotN"],
        "max_BdotN_over_B": v["max_BdotN_over_B"],
        "bn_grid_drift": v.get("bn_grid_drift"),
        "linking": lk,
        "checks": checks,
        "margins": margins,
        "feasible": all(checks.values()),
        "failed": [k for k, x in checks.items() if not x],
    }
    if a.dcc_converge:
        rec["dcc_convergence"] = dcc_convergence(cs)

    print("\n" + "=" * 72)
    print(f"ADJUDICATION  {rec['rundir']}")
    print(
        f"  bn (source grid) {v['normalized_BdotN']:.4e}"
        f"   max|B.n|/|B| {v['max_BdotN_over_B']:.3e}"
    )
    print(
        f"  length judged against {L_bound:.6f} m"
        f"{'  (SWEPT)' if a.length_max is not None else '  (paper)'}"
    )
    print("  margins  " + " · ".join(f"{k} {x:.4f}" for k, x in margins.items()))
    # The linking line is printed on its own, always, count first. It is the field two
    # banked "improvements" turned out to be.
    print(
        f"  LINKED PAIRS {lk['n_linked_pairs']} of {lk['n_pairs_checked']}"
        f"   worst |Lk| {lk['max_abs_linking_number']:.3e}"
        f"   (matrix, N={lk['linking_grid_N']})"
    )
    if lk["linked_pairs"]:
        print(f"    {lk['linked_pairs']}")
        print(
            "    ^ HARD REJECTION. This point's bn is meaningless -- an apparent "
            "improvement\n      from a warm start IS the linking until this reads 0."
        )
    if a.dcc_converge:
        d = rec["dcc_convergence"]
        print(
            f"  d_cc  N400 {d['d_cc_N400']:.6f}  N800 {d['d_cc_N800']:.6f}"
            f"  drift {100*d['d_cc_drift_rel']:.2f}%"
            + (
                "   NOT CONVERGED -- do not quote d_cc"
                if d["d_cc_drift_rel"] > 2e-3
                else ""
            )
        )
    print(
        f"  VERDICT  {'FEASIBLE' if rec['feasible'] else 'FAILS ' + ','.join(rec['failed'])}"
    )
    print("=" * 72)

    out = os.path.join(a.rundir, "adjudication.json")
    json.dump(rec, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
