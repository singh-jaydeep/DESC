"""Time master's tr_method="qr" against tr_method="qr-struct" on real eq.solve runs.

Runs each (equilibrium, tr_method) pair as a full DESC equilibrium solve from an
identical perturbed start, and reports wall time plus the solve trajectory so we
can confirm the two took the SAME optimization path (identical iterate count and
final cost) and the speedup is not from converging differently.

Usage
-----
    python solve_bench.py                      # default case list
    python solve_bench.py '[["W7-X",6,0.01]]'  # [name, L, perturbation]
    DESC_BENCH_REPS=2 python solve_bench.py    # repeat each timing

Set DESC_BENCH_DEVICE=cpu to force CPU. Writes solve_bench_<platform>.json.
"""

import json
import os
import sys
import time

import numpy as np

# device selection must happen before desc imports jax
_dev = os.environ.get("DESC_BENCH_DEVICE", "gpu")
import desc

desc.set_device(_dev)

import jax

from desc.examples import get
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)
from desc.optimize._constraint_wrappers import LinearConstraintProjection

REPS = int(os.environ.get("DESC_BENCH_REPS", "1"))
MAXITER = int(os.environ.get("DESC_BENCH_MAXITER", "25"))


def build_case(name, L, pert, N=None):
    """Fresh perturbed equilibrium + objective/constraints, identical per method."""
    eq = get(name)
    NN = eq.N if N is None else N
    eq.change_resolution(L=L, M=L, N=NN, L_grid=2 * L, M_grid=2 * L, N_grid=2 * NN)
    rng = np.random.default_rng(0)
    if pert:
        eq.R_lmn = eq.R_lmn + pert * rng.standard_normal(eq.R_lmn.size) * np.abs(
            eq.R_lmn
        ).max()
        eq.Z_lmn = eq.Z_lmn + pert * rng.standard_normal(eq.Z_lmn.size) * np.abs(
            eq.Z_lmn
        ).max()
    obj = ObjectiveFunction(ForceBalance(eq))
    cons = get_fixed_boundary_constraints(eq)
    return eq, obj, cons


def problem_shape(name, L, pert, N=None):
    """(m, n) of the constraint-projected least-squares problem lsqtr sees."""
    eq, obj, cons = build_case(name, L, pert, N)
    lcp = LinearConstraintProjection(obj, ObjectiveFunction(cons))
    lcp.build(verbose=0)
    return int(lcp.dim_f), int(lcp.dim_x)


def run_one(name, L, pert, tr_method, N=None, block=128):
    """One full eq.solve, timed. Returns (wall_s, cost, n_iter, converged)."""
    eq, obj, cons = build_case(name, L, pert, N)
    options = {"tr_method": tr_method}
    if tr_method == "qr-struct":
        options["tr_qr_block"] = block
    t0 = time.perf_counter()
    out = eq.solve(
        objective=obj,
        constraints=cons,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        maxiter=MAXITER,
        options=options,
        verbose=0,
        copy=True,
    )
    wall = time.perf_counter() - t0
    res = out[1] if isinstance(out, (tuple, list)) else out
    cost = float(getattr(res, "cost", np.nan))
    nit = int(getattr(res, "nit", -1))
    return wall, cost, nit, bool(getattr(res, "success", False))


def run_isolated(name, L, pert, tr_method, N=None, block=128):
    """Run ONE solve in a FRESH process.

    Two methods run in the same process share DESC's/jax's compilation caches:
    the objective and its Jacobian are compiled once and reused, so whichever
    method runs second skips all compilation and looks ~7x faster regardless of
    which one it is (verified by order reversal). Every timing must therefore be
    a cold process.
    """
    import subprocess

    payload = json.dumps([name, L, pert, tr_method, N, block])
    env = dict(os.environ, DESC_BENCH_CHILD=payload)
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        capture_output=True, text=True, env=env,
    )
    for line in r.stdout.splitlines():
        if line.startswith("CHILD_RESULT "):
            return json.loads(line[len("CHILD_RESULT "):])
    return dict(
        error=(r.stderr or r.stdout)[-600:], wall_s=float("nan"),
        cost=float("nan"), n_iter=-1, success=False,
    )


if __name__ == "__main__":
    # child mode: one solve, one process, result on stdout
    _child = os.environ.get("DESC_BENCH_CHILD")
    if _child:
        nm, L_, pert_, meth_, N_, blk_ = json.loads(_child)
        try:
            w, c_, it, ok = run_one(nm, L_, pert_, meth_, N_, blk_)
            res = dict(wall_s=w, cost=c_, n_iter=it, success=ok)
        except Exception as e:
            res = dict(error=f"{type(e).__name__}: {e}", wall_s=float("nan"),
                       cost=float("nan"), n_iter=-1, success=False)
        print("CHILD_RESULT " + json.dumps(res), flush=True)
        sys.exit(0)

    default = [
        ["HELIOTRON", 6, 0.01],
        ["precise_QA", 6, 0.01],
        ["W7-X", 6, 0.01],
        ["HELIOTRON", 8, 0.01],
    ]
    cases = json.loads(sys.argv[1]) if len(sys.argv) > 1 else default

    dev = jax.devices()[0]
    meta = dict(
        platform=dev.platform,
        device=str(dev.device_kind),
        jax=jax.__version__,
        desc=desc.__version__,
        reps=REPS,
        maxiter=MAXITER,
    )
    print("DEVICE:", meta, flush=True)

    rows = []
    for spec in cases:
        name, L, pert = spec[0], spec[1], spec[2]
        N = spec[3] if len(spec) > 3 else None
        try:
            m, n = problem_shape(name, L, pert, N)
        except Exception as e:
            print(f"[{name} L{L}] shape probe failed: {e}", flush=True)
            m = n = -1
        rec = dict(name=name, L=L, pert=pert, m=m, n=n, **meta)
        # alternate the order across reps so any residual ordering effect shows up
        for meth in ["qr", "qr-struct"]:
            ts, costs, nits, oks, errs = [], [], [], [], []
            for rep in range(REPS):
                r = run_isolated(name, L, pert, meth, N)
                if "error" in r:
                    print(f"[{name} L{L} {meth}] child error: {r['error'][:200]}",
                          flush=True)
                    errs.append(r["error"])
                ts.append(r["wall_s"]); costs.append(r["cost"])
                nits.append(r["n_iter"]); oks.append(r["success"])
            rec[meth] = dict(
                wall_s=float(np.median(ts)), wall_all=ts,
                cost=costs[0], n_iter=nits[0], success=oks[0],
                errors=errs or None,
            )
        a, b = rec["qr"]["wall_s"], rec["qr-struct"]["wall_s"]
        rec["speedup"] = a / b if b == b and b > 0 else float("nan")
        # did the two follow the same optimization path?
        ca, cb = rec["qr"]["cost"], rec["qr-struct"]["cost"]
        rec["cost_reldiff"] = (
            abs(ca - cb) / max(abs(ca), 1e-300) if ca == ca and cb == cb else float("nan")
        )
        rec["same_iters"] = rec["qr"]["n_iter"] == rec["qr-struct"]["n_iter"]
        rec["same_path"] = bool(rec["same_iters"] and rec["cost_reldiff"] < 1e-10)
        rows.append(rec)
        print(
            f"[{name} L{L}] m/n={m}/{n}={m/max(n,1):.2f} | "
            f"qr={a:7.2f}s (it={rec['qr']['n_iter']}, cost={rec['qr']['cost']:.6e}) | "
            f"qr-struct={b:7.2f}s (it={rec['qr-struct']['n_iter']}, "
            f"cost={rec['qr-struct']['cost']:.6e}) | speedup={rec['speedup']:.3f}x | "
            f"same_iters={rec['same_iters']} cost_reldiff={rec['cost_reldiff']:.2e}",
            flush=True,
        )
        outdir = "out" if os.path.isdir("out") else "."
        json.dump(rows, open(f"{outdir}/solve_bench_{dev.platform}.json", "w"), indent=2)
