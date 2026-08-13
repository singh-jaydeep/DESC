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


def profile(name, L, pert, tr_method="qr", maxiter=15, N=None):
    for k in T:
        T[k] = 0 if isinstance(T[k], int) else 0.0

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

    # wrap the subproblem for the method under test
    key = "trust_region_step_exact_qr_struct" if tr_method == "qr-struct" else (
        "trust_region_step_exact_qr"
    )
    orig_sub = getattr(LS, key)
    setattr(LS, key, _timed(orig_sub, "subproblem", "n_sub"))

    # NOTE: the objective's jac/compute are NOT wrapped. ObjectiveFunction is a
    # registered jax pytree, and replacing a bound method with a closure makes
    # its flattened leaves contain a function, which jit rejects. Only the
    # subproblem is timed; jac/fun cost is (wall - subproblem) plus overhead.

    t0 = time.perf_counter()
    out = eq.solve(
        objective=obj, constraints=cons, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        maxiter=maxiter, options={"tr_method": tr_method}, verbose=0, copy=True,
    )
    wall = time.perf_counter() - t0

    setattr(LS, key, orig_sub)

    res = out[1] if isinstance(out, (tuple, list)) else out
    return dict(
        name=name, L=L, tr_method=tr_method, wall_s=wall,
        subproblem_s=T["subproblem"], n_sub=T["n_sub"],
        jac_s=T["jac"], n_jac=T["n_jac"], fun_s=T["fun"], n_fun=T["n_fun"],
        sub_frac=T["subproblem"] / wall, jac_frac=T["jac"] / wall,
        fun_frac=T["fun"] / wall,
        n_iter=int(getattr(res, "nit", -1)),
        cost=float(getattr(res, "cost", np.nan)),
    )


if __name__ == "__main__":
    dev = jax.devices()[0]
    print("DEVICE:", dev.platform, dev.device_kind, flush=True)
    cases = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        ["HELIOTRON", 6, 0.01], ["W7-X", 6, 0.01]
    ]
    maxiter = int(os.environ.get("DESC_BENCH_MAXITER", "15"))
    out = []
    for spec in cases:
        for meth in ["qr", "qr-struct"]:
            r = profile(spec[0], spec[1], spec[2], meth, maxiter,
                        spec[3] if len(spec) > 3 else None)
            out.append(r)
            print(
                f"[{r['name']} L{r['L']} {meth:9s}] wall={r['wall_s']:7.2f}s "
                f"| subproblem={r['subproblem_s']:6.2f}s ({r['sub_frac']*100:5.2f}%, "
                f"{r['n_sub']} calls) | jac={r['jac_s']:7.2f}s "
                f"| everything else={r['wall_s']-r['subproblem_s']:7.2f}s "
                f"({(1-r['sub_frac'])*100:5.1f}%) | it={r['n_iter']}",
                flush=True,
            )
            outdir = "out" if os.path.isdir("out") else "."
            json.dump(out, open(f"{outdir}/profile_{dev.platform}.json", "w"), indent=2)
