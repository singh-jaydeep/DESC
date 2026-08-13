"""Where does an eq.solve actually spend its time?

Directly measures, inside a real DESC equilibrium solve, the wall time spent in
  (a) the trust-region subproblem (the alpha loop we have been optimizing),
  (b) Jacobian evaluations,
  (c) objective/residual evaluations,
by wrapping each with a timer. This settles whether the alpha loop is a large
enough share of the solve for a faster factorization to matter (Amdahl), rather
than inferring the share from separate microbenchmarks.

Timers use jax.block_until_ready so async dispatch is not mistaken for speed.
"""

import json
import os
import sys
import time

import numpy as np

_dev = os.environ.get("DESC_BENCH_DEVICE", "gpu")
import desc

desc.set_device(_dev)

import jax

import desc.optimize.least_squares as LS
from desc.examples import get
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)

T = dict(subproblem=0.0, n_sub=0, jac=0.0, n_jac=0, fun=0.0, n_fun=0)


def _timed(fn, key, nkey):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        jax.block_until_ready(out)
        T[key] += time.perf_counter() - t0
        T[nkey] += 1
        return out

    return wrapper


def _build(name, L, pert, N=None):
    eq = get(name)
    NN = eq.N if N is None else N
    eq.change_resolution(L=L, M=L, N=NN, L_grid=2 * L, M_grid=2 * L, N_grid=2 * NN)
    rng = np.random.default_rng(0)
    eq.R_lmn = eq.R_lmn + pert * rng.standard_normal(eq.R_lmn.size) * np.abs(
        eq.R_lmn
    ).max()
    eq.Z_lmn = eq.Z_lmn + pert * rng.standard_normal(eq.Z_lmn.size) * np.abs(
        eq.Z_lmn
    ).max()
    obj = ObjectiveFunction(ForceBalance(eq))
    cons = get_fixed_boundary_constraints(eq)
    return eq, obj, cons


def profile(name, L, pert, tr_method="qr", maxiter=15, N=None, passes=2):
    """Profile ONE method. Must be the only method profiled in this process.

    Runs the SAME solve ``passes`` times in this process. Pass 0 pays JIT
    compilation of the objective, Jacobian and subproblem; later passes hit the
    compilation cache and are the steady state. Both are reported, because they
    answer different questions and mixing them is how the earlier version of
    this script produced a wrong Amdahl fraction:

      - pass 0 ``sub_frac`` divides the subproblem time by a wall time that is
        mostly compilation, so it UNDERSTATES the alpha loop's share.
      - a later pass run in a process that already compiled the OTHER method
        OVERSTATES it, because the shared objective/Jacobian cache makes the
        denominator collapse while the subproblem still runs.

    Only ``sub_frac`` from a warm pass in a single-method process is the
    steady-state share that Amdahl's law should be applied to.
    """
    key = (
        "trust_region_step_exact_qr_struct"
        if tr_method == "qr-struct"
        else "trust_region_step_exact_qr"
    )
    orig_sub = getattr(LS, key)
    out_passes = []
    for p in range(passes):
        for k in T:
            T[k] = 0 if isinstance(T[k], int) else 0.0
        eq, obj, cons = _build(name, L, pert, N)
        setattr(LS, key, _timed(orig_sub, "subproblem", "n_sub"))
        # NOTE: the objective's jac/compute are NOT wrapped. ObjectiveFunction is
        # a registered jax pytree, and replacing a bound method with a closure
        # puts a function in its flattened leaves, which jit rejects. Only the
        # subproblem is timed; everything else is (wall - subproblem).
        t0 = time.perf_counter()
        out = eq.solve(
            objective=obj, constraints=cons, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            maxiter=maxiter, options={"tr_method": tr_method}, verbose=0, copy=True,
        )
        wall = time.perf_counter() - t0
        setattr(LS, key, orig_sub)
        res = out[1] if isinstance(out, (tuple, list)) else out
        out_passes.append(
            dict(
                pass_i=p, wall_s=wall, subproblem_s=T["subproblem"], n_sub=T["n_sub"],
                sub_frac=T["subproblem"] / wall,
                n_iter=int(getattr(res, "nit", -1)),
                cost=float(getattr(res, "cost", np.nan)),
            )
        )
    cold, warm = out_passes[0], out_passes[-1]
    return dict(
        name=name, L=L, tr_method=tr_method, passes=out_passes,
        # cold: includes compilation in the denominator
        wall_cold_s=cold["wall_s"], sub_frac_cold=cold["sub_frac"],
        # warm: the steady-state share -- this is the Amdahl fraction
        wall_s=warm["wall_s"], subproblem_s=warm["subproblem_s"],
        n_sub=warm["n_sub"], sub_frac=warm["sub_frac"],
        n_iter=warm["n_iter"], cost=warm["cost"],
        compile_s=cold["wall_s"] - warm["wall_s"],
    )


def run_isolated(name, L, pert, tr_method, maxiter, N=None, passes=2):
    """Profile one method in a FRESH process.

    Two methods in one process share DESC's/jax's compilation caches for the
    objective and Jacobian, so whichever runs second solves a nearly
    fully-compiled problem: its wall time collapses (~10x) while its subproblem
    time does not, inflating sub_frac from a few percent to ~65%. Isolation is
    what makes sub_frac comparable across methods.
    """
    import subprocess

    payload = json.dumps([name, L, pert, tr_method, maxiter, N, passes])
    env = dict(os.environ, DESC_PROFILE_CHILD=payload)
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        capture_output=True, text=True, env=env,
    )
    for line in r.stdout.splitlines():
        if line.startswith("CHILD_RESULT "):
            return json.loads(line[len("CHILD_RESULT "):])
    return dict(error=(r.stderr or r.stdout)[-800:])


if __name__ == "__main__":
    _child = os.environ.get("DESC_PROFILE_CHILD")
    if _child:
        nm, L_, pert_, meth_, mi_, N_, ps_ = json.loads(_child)
        try:
            res = profile(nm, L_, pert_, meth_, mi_, N_, ps_)
        except Exception as e:
            res = dict(error=f"{type(e).__name__}: {e}")
        print("CHILD_RESULT " + json.dumps(res), flush=True)
        sys.exit(0)

    dev = jax.devices()[0]
    print("DEVICE:", dev.platform, dev.device_kind, flush=True)
    cases = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        ["HELIOTRON", 6, 0.01], ["W7-X", 6, 0.01]
    ]
    maxiter = int(os.environ.get("DESC_BENCH_MAXITER", "15"))
    passes = int(os.environ.get("DESC_PROFILE_PASSES", "3"))
    out = []
    for spec in cases:
        for meth in ["qr", "qr-struct"]:
            r = run_isolated(spec[0], spec[1], spec[2], meth, maxiter,
                             spec[3] if len(spec) > 3 else None, passes)
            if "error" in r:
                print(f"[{spec[0]} L{spec[1]} {meth}] child error: "
                      f"{r['error'][:300]}", flush=True)
                out.append(dict(name=spec[0], L=spec[1], tr_method=meth, **r))
            else:
                out.append(r)
                print(
                    f"[{r['name']} L{r['L']} {meth:9s}] "
                    f"cold: wall={r['wall_cold_s']:7.2f}s "
                    f"sub_frac={r['sub_frac_cold']*100:5.2f}% | "
                    f"WARM: wall={r['wall_s']:6.2f}s "
                    f"subproblem={r['subproblem_s']:5.2f}s "
                    f"({r['sub_frac']*100:5.2f}%, {r['n_sub']} calls) | "
                    f"compile={r['compile_s']:6.2f}s | it={r['n_iter']}",
                    flush=True,
                )
            outdir = "out" if os.path.isdir("out") else "."
            json.dump(out, open(f"{outdir}/profile_{dev.platform}.json", "w"), indent=2)
