"""Cost + accuracy comparison of the four alpha-loop variants.

Accounts for the real DESC call pattern: per OUTER iteration lsqtr does the
one-time prep once, then calls the subproblem once per pass of the inner
`while actual_reduction <= 0` trust-radius-shrink loop, each pass warm-started
from the previous alpha. So the quantity that matters is

    total(n_calls, k) = prep + n_calls * (loop_overhead + k * per_iteration)

which is what `total_*` below reports.
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

from bench_qr_alpha import make_problem
from variants_alpha import (
    givens_retriangularize,
    prep_cho,
    prep_svd_of_J,
    prep_svd_of_R,
    solve_triangular_regularized,
    step_cho_shift,
    step_qr_dense,
    step_svd_of_R,
)


def outer_qr(J, f):
    """desc/optimize/least_squares.py:301-306 (tall branch)."""
    from jax.scipy.linalg import qr_multiply

    Qt_fa, R = qr_multiply(J, f, mode="right")
    p_newton = solve_triangular_regularized(R, -Qt_fa)
    return Qt_fa, R, p_newton


def timeit(fn, *a, reps=5):
    out = jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def one_size(n, m, cond_exp=6.0, seed=0):
    J, f = make_problem(m, n, seed=seed, cond_exp=cond_exp)
    row = dict(n=n, m=m, cond_exp=cond_exp)

    # ---- prep: one-time work per Jacobian ---------------------------------
    t_qr, (z, R, p_newton) = timeit(outer_qr, J, f)
    row["prep_A_qr"] = t_qr

    t_cho_extra, (G, Rtz) = timeit(prep_cho, R, z)
    row["prep_B_cho"] = t_qr + t_cho_extra

    t_svdR_extra, (pnC, Vt, s, c) = timeit(prep_svd_of_R, R, z)
    row["prep_C_svdR"] = t_qr + t_svdR_extra
    row["prep_C_svd_only"] = t_svdR_extra

    t_svdJ, (pnD, VtD, sD, sufD) = timeit(prep_svd_of_J, J, f)
    row["prep_D_svdJ"] = t_svdJ

    # ---- a single subproblem call, at a trust radius that hits the boundary
    pn = float(jnp.linalg.norm(p_newton))
    tr = 0.1 * pn
    row["newton_norm"] = pn

    tA, (pA, hA, aA) = timeit(step_qr_dense, p_newton, z, R, tr, 0.0)
    tB, (pB, hB, aB) = timeit(step_cho_shift, p_newton, G, Rtz, tr, 0.0)
    tC, (pC, hC, aC) = timeit(step_svd_of_R, pnC, Vt, s, c, tr, 0.0)
    row["call_cold_A"], row["call_cold_B"], row["call_cold_C"] = tA, tB, tC

    # warm-started call (alpha from the cold solve) -- the common case
    tAw, _ = timeit(step_qr_dense, p_newton, z, R, tr, aA)
    tBw, _ = timeit(step_cho_shift, p_newton, G, Rtz, tr, aA)
    tCw, _ = timeit(step_svd_of_R, pnC, Vt, s, c, tr, aA)
    row["call_warm_A"], row["call_warm_B"], row["call_warm_C"] = tAw, tBw, tCw

    row["alpha_A"] = float(aA)
    row["alpha_B"] = float(aB)
    row["alpha_C"] = float(aC)
    row["relerr_B"] = float(jnp.linalg.norm(pB - pA) / jnp.linalg.norm(pA))
    row["relerr_C"] = float(jnp.linalg.norm(pC - pA) / jnp.linalg.norm(pA))

    def kkt(p, a):
        return float(
            jnp.linalg.norm(R.T @ (R @ p) + a * p + R.T @ z)
            / jnp.linalg.norm(R.T @ z)
        )

    row["kkt_A"], row["kkt_B"], row["kkt_C"] = kkt(pA, aA), kkt(pB, aB), kkt(pC, aC)
    return row


if __name__ == "__main__":
    rows = []
    for n in [500, 1000, 2000, 4000]:
        for m in [int(1.5 * n), n]:
            r = one_size(n, m)
            rows.append(r)
            print(
                f"n={n:5d} m={m:5d} | prep A={r['prep_A_qr']*1e3:8.1f} "
                f"C={r['prep_C_svdR']*1e3:8.1f} (svd only {r['prep_C_svd_only']*1e3:7.1f}) "
                f"D={r['prep_D_svdJ']*1e3:8.1f} | cold call A={r['call_cold_A']*1e3:8.1f} "
                f"B={r['call_cold_B']*1e3:7.1f} C={r['call_cold_C']*1e3:6.2f} | "
                f"warm A={r['call_warm_A']*1e3:8.1f} C={r['call_warm_C']*1e3:6.2f} | "
                f"relerrC={r['relerr_C']:.1e}",
                flush=True,
            )
    json.dump(rows, open("variants_timing.json", "w"), indent=2)

    # ---- the "update alpha_1 -> alpha_2" question, measured ---------------
    print("\nstructured retriangularization flop count vs n (from scratch):")
    gv = []
    for n in [40, 60, 80, 120, 160]:
        J, f = make_problem(int(1.5 * n), n, cond_exp=6.0)
        z, R, _ = outer_qr(J, f)
        _, _, fl = givens_retriangularize(
            np.asarray(R), np.asarray(z), 1e-3, count_flops=True
        )
        gv.append(dict(n=n, flops=fl, n3_over_6=n**3 / 6, dense=10 * n**3 / 3))
        print(
            f"  n={n:4d} givens={fl:10.3e}  n^3/6={n**3/6:10.3e}  "
            f"ratio={fl/(n**3/6):5.2f}  dense/givens={10*n**3/3/fl:6.2f}"
        )
    json.dump(gv, open("givens_flops.json", "w"), indent=2)
