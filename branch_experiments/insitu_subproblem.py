"""Is the microbenchmark's per-call speedup real on the arguments a solve produces?

The in-solve profile showed tr_method="qr-struct" taking MORE subproblem time
than "qr", contradicting the 1.25-1.59x per-alpha-iteration win measured on
synthetic matrices. Two candidate explanations:

  (i)  COMPILE TIME. Each subproblem is jit-compiled once and a solve makes only
       ~26-30 calls. The structured version unrolls a Python panel loop, so it
       emits far more HLO than the dense version's single qr_multiply call, and
       its compile cost may swamp any steady-state gain over so few calls.
  (ii) THE ARGUMENTS. The synthetic benchmark used random matrices with a
       log-spaced spectrum at m/n = 1.5-3. A real solve's R is the
       constraint-projected reduced factor, whose conditioning and the resulting
       alpha-iteration counts may differ.

This script separates them. It captures the REAL (p_newton, z, R, trust_radius,
alpha) tuples from an actual eq.solve, then for each captured argument set:

  - times the FIRST call to each implementation  -> compile + execute
  - times subsequent calls                       -> steady-state execute
  - counts alpha iterations (numpy replica) so per-iteration cost is comparable
  - checks the two produce the same step

Reports steady-state per-call speedup, compile cost, and the break-even number
of calls at which the structured version would start paying for itself.
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
import jax.numpy as jnp

import desc.optimize.least_squares as LS
from desc.examples import get
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)
from desc.optimize.tr_subproblems import (
    trust_region_step_exact_qr,
    trust_region_step_exact_qr_struct,
)

CAPTURED = []


def capture_args(name, L, pert, maxiter=10, N=None, max_keep=6):
    """Run a real solve, keeping the arguments passed to the subproblem."""
    CAPTURED.clear()
    orig = LS.trust_region_step_exact_qr

    def spy(p_newton, z, R, trust_radius, initial_alpha=0.0, *a, **k):
        if len(CAPTURED) < max_keep:
            CAPTURED.append(
                dict(
                    p_newton=np.asarray(p_newton), z=np.asarray(z), R=np.asarray(R),
                    trust_radius=float(trust_radius),
                    initial_alpha=float(initial_alpha),
                )
            )
        return orig(p_newton, z, R, trust_radius, initial_alpha, *a, **k)

    LS.trust_region_step_exact_qr = spy
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
    eq.solve(objective=obj, constraints=cons, ftol=1e-10, xtol=1e-10, gtol=1e-10,
             maxiter=maxiter, options={"tr_method": "qr"}, verbose=0, copy=True)
    LS.trust_region_step_exact_qr = orig
    return list(CAPTURED)


def count_alpha_iters(R, z, trust_radius, initial_alpha, rtol=0.01, max_iter=10):
    """numpy replica of the loop, to get the trip count jax's while_loop hides."""
    R = np.asarray(R, float); z = np.asarray(z, float)
    n = R.shape[1]
    a_hi = np.linalg.norm(R.T @ z) / trust_radius
    alpha = float(np.clip(initial_alpha, 0.0, a_hi))
    a_lo, phi, k = 0.0, np.inf, 0
    while abs(phi) > rtol * trust_radius and k < max_iter:
        A = np.vstack([R, np.sqrt(alpha) * np.eye(n)])
        Q, Rt = np.linalg.qr(A)
        Qtz = Q.T @ np.concatenate([z, np.zeros(n)])
        try:
            p = np.linalg.solve(Rt, -Qtz)
        except np.linalg.LinAlgError:
            break
        pn = np.linalg.norm(p); phi = pn - trust_radius
        a_hi = alpha if phi < 0 else a_hi
        a_lo = alpha if phi > 0 else a_lo
        q = np.linalg.solve(Rt.T, p); qn = np.linalg.norm(q)
        alpha = float(np.clip(alpha + (pn / qn) ** 2 * phi / trust_radius, a_lo, a_hi))
        k += 1
    return k


def time_calls(fn, args, kw, n_warm=6):
    """(first-call time incl. compile, median steady-state time)."""
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args, **kw))
    t_first = time.perf_counter() - t0
    ts = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args, **kw))
        ts.append(time.perf_counter() - t0)
    return t_first, float(np.median(ts)), out


if __name__ == "__main__":
    dev = jax.devices()[0]
    print("DEVICE:", dev.platform, dev.device_kind, flush=True)
    cases = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        ["HELIOTRON", 6, 0.01], ["precise_QA", 6, 0.01], ["W7-X", 6, 0.01]
    ]
    blocks = [int(b) for b in os.environ.get("DESC_BENCH_BLOCKS", "64,128,256").split(",")]
    rows = []
    for spec in cases:
        name, L, pert = spec[0], spec[1], spec[2]
        N = spec[3] if len(spec) > 3 else None
        caps = capture_args(name, L, pert, N=N)
        print(f"\n=== {name} L{L}: captured {len(caps)} real subproblem arg sets ===",
              flush=True)
        for ci, cap in enumerate(caps):
            R = jnp.asarray(cap["R"]); z = jnp.asarray(cap["z"])
            pn = jnp.asarray(cap["p_newton"])
            tr = cap["trust_radius"]; a0 = cap["initial_alpha"]
            n = R.shape[1]
            # only the calls that actually enter the loop are interesting
            hits = float(jnp.linalg.norm(pn)) > tr
            k = count_alpha_iters(R, z, tr, a0) if hits else 0
            cond_R = float(jnp.linalg.cond(R))
            fa, sa, outa = time_calls(trust_region_step_exact_qr, (pn, z, R, tr, a0), {})
            rec = dict(name=name, L=L, call=ci, n=int(n), m_rows_R=int(R.shape[0]),
                       trust_radius=tr, initial_alpha=a0, hits_boundary=bool(hits),
                       alpha_iters=int(k), cond_R=cond_R,
                       qr_first=fa, qr_steady=sa, struct={})
            for b in blocks:
                fb, sb, outb = time_calls(
                    trust_region_step_exact_qr_struct, (pn, z, R, tr, a0), {"block": b}
                )
                rel = float(
                    jnp.linalg.norm(outb[0] - outa[0])
                    / max(float(jnp.linalg.norm(outa[0])), 1e-300)
                )
                rec["struct"][str(b)] = dict(
                    first=fb, steady=sb, speedup=sa / sb, relerr=rel,
                    compile_extra=(fb - sb) - (fa - sa),
                    breakeven_calls=(
                        ((fb - sb) - (fa - sa)) / (sa - sb) if sb < sa else float("inf")
                    ),
                )
            best_b = max(rec["struct"], key=lambda b: rec["struct"][b]["speedup"])
            bs = rec["struct"][best_b]
            print(
                f"  call {ci}: n={n:5d} cond(R)={cond_R:8.2e} k_alpha={k:2d} "
                f"hits={hits} | qr: first={fa*1e3:8.1f} steady={sa*1e3:7.2f} ms | "
                f"struct b={best_b:>3}: first={bs['first']*1e3:8.1f} "
                f"steady={bs['steady']*1e3:7.2f} ms -> speedup={bs['speedup']:5.2f}x "
                f"relerr={bs['relerr']:.1e} breakeven={bs['breakeven_calls']:.0f} calls",
                flush=True,
            )
            rows.append(rec)
            outdir = "out" if os.path.isdir("out") else "."
            json.dump(rows, open(f"{outdir}/insitu_{dev.platform}.json", "w"), indent=2)
