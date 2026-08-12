"""Baseline cost profile of DESC's QR trust-region LM step (tr_method='qr').

Replicates desc/optimize/tr_subproblems.py::trust_region_step_exact_qr and the
calling pattern in desc/optimize/least_squares.py::lsqtr verbatim (jax, float64),
and times the three distinct pieces of work per outer iteration:

  (a) outer factorization  qr_multiply(J_a, f_a)              -- once per Jacobian
  (b) one alpha iteration  qr_multiply([R; sqrt(a) I], zp)    -- per LM inner step
  (c) the two triangular solves inside one alpha iteration

so we can see what fraction of the step is spent in the alpha loop.
"""

import json
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.lax import cond, while_loop
from jax.scipy.linalg import qr, qr_multiply, solve_triangular


# --- verbatim from desc/optimize/utils.py -----------------------------------
def solve_triangular_regularized(R, b, lower=False):
    dr = jnp.diag(R)
    denom = jnp.where(dr == 0, 1, dr)
    dri = jnp.where(dr == 0, 0, 1 / denom)
    Rs = R * dri[:, None]
    b = dri * b
    return solve_triangular(Rs, b, unit_diagonal=True, lower=lower)


# --- verbatim from desc/optimize/tr_subproblems.py --------------------------
@jit
def trust_region_step_exact_qr(
    p_newton, z, R, trust_radius, initial_alpha=0.0, rtol=0.01, max_iter=10
):
    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(R.T @ z) / trust_radius
        alpha_lower = 0.0
        alpha = initial_alpha
        alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
        k = 0

        n = R.shape[1]
        zp = jnp.concatenate([z, jnp.zeros(n)])

        def loop_cond(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state
            return (jnp.abs(phi) > rtol * trust_radius) & (k < max_iter)

        def loop_body(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state

            A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
            Qtz, Rtil = qr_multiply(A, zp, mode="right")

            p = solve_triangular_regularized(Rtil, -Qtz)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            alpha_upper = jnp.where(phi < 0, alpha, alpha_upper)
            alpha_lower = jnp.where(phi > 0, alpha, alpha_lower)

            q = solve_triangular_regularized(Rtil.T, p, lower=True)
            q_norm = jnp.linalg.norm(q)

            alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
            alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
            k += 1
            return p, alpha, alpha_lower, alpha_upper, phi, k

        p, alpha, *_ = while_loop(
            loop_cond,
            loop_body,
            (p_newton, alpha, alpha_lower, alpha_upper, jnp.inf, k),
        )
        p *= trust_radius / jnp.linalg.norm(p)
        return p, True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


# --- isolated pieces, jitted separately so we can time them ------------------
@jit
def outer_factorization(J_a, f_a):
    """(a) what lsqtr does once per Jacobian: QR of J plus the Newton step."""
    Qt_fa, R = qr_multiply(J_a, f_a, mode="right")
    p_newton = solve_triangular_regularized(R, -Qt_fa)
    return Qt_fa, R, p_newton


@jit
def one_alpha_iteration(R, zp, alpha):
    """(b) the body of the alpha loop: dense QR of the stacked [R; sqrt(a) I]."""
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
    Qtz, Rtil = qr_multiply(A, zp, mode="right")
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q, Rtil


@jit
def two_solves_only(Rtil, Qtz):
    """(c) just the triangular solves, for reference against the QR cost."""
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q


def timeit(fn, *args, reps=5):
    out = jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def count_alpha_iters(R, z, trust_radius, initial_alpha, rtol=0.01, max_iter=10):
    """Run the same Newton-on-phi iteration in numpy to count iterations taken."""
    n = R.shape[1]
    zp = np.concatenate([z, np.zeros(n)])
    alpha_upper = np.linalg.norm(R.T @ z) / trust_radius
    alpha_lower = 0.0
    alpha = float(np.clip(initial_alpha, alpha_lower, alpha_upper))
    phi = np.inf
    k = 0
    while abs(phi) > rtol * trust_radius and k < max_iter:
        A = np.vstack([R, np.sqrt(alpha) * np.eye(n)])
        Q, Rtil = np.linalg.qr(A)
        Qtz = Q.T @ zp
        p = np.linalg.solve(Rtil, -Qtz)
        p_norm = np.linalg.norm(p)
        phi = p_norm - trust_radius
        if phi < 0:
            alpha_upper = alpha
        if phi > 0:
            alpha_lower = alpha
        q = np.linalg.solve(Rtil.T, p)
        alpha += (p_norm / np.linalg.norm(q)) ** 2 * phi / trust_radius
        alpha = float(np.clip(alpha, alpha_lower, alpha_upper))
        k += 1
    return k, alpha


def make_problem(m, n, seed=0, cond_exp=6.0):
    """Jacobian with a DESC-like decaying spectrum (ill-conditioned, full rank)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
    V, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
    s = np.logspace(0, -cond_exp, min(m, n))
    J = (U * s) @ V.T
    f = rng.standard_normal(m)
    return jnp.asarray(J), jnp.asarray(f)


if __name__ == "__main__":
    rows = []
    for n in [500, 1000, 2000, 4000]:
        for shape in ["tall", "square-ish"]:
            m = int(1.5 * n) if shape == "tall" else n
            J, f = make_problem(m, n)
            t_outer, (Qt_fa, R, p_newton) = timeit(outer_factorization, J, f)

            zp = jnp.concatenate([Qt_fa, jnp.zeros(n)])
            alpha0 = 1.0
            t_iter, (p, q, Rtil) = timeit(one_alpha_iteration, R, zp, alpha0)
            t_solves, _ = timeit(two_solves_only, Rtil, Qt_fa)

            # how many alpha iterations a realistic step actually takes:
            # trust radius set well inside the Newton step so the boundary is hit
            pn = float(jnp.linalg.norm(p_newton))
            iters = {}
            for frac in [0.5, 0.1, 0.01]:
                tr = frac * pn
                k_cold, a_cold = count_alpha_iters(
                    np.asarray(R), np.asarray(Qt_fa), tr, 0.0
                )
                k_warm, _ = count_alpha_iters(
                    np.asarray(R), np.asarray(Qt_fa), tr, a_cold
                )
                iters[frac] = (k_cold, k_warm)

            rows.append(
                dict(
                    n=n,
                    m=m,
                    shape=shape,
                    t_outer_qr=t_outer,
                    t_alpha_iter=t_iter,
                    t_two_solves=t_solves,
                    ratio_iter_over_outer=t_iter / t_outer,
                    newton_norm=pn,
                    iters_cold={str(k): v[0] for k, v in iters.items()},
                    iters_warm={str(k): v[1] for k, v in iters.items()},
                )
            )
            print(
                f"n={n:5d} m={m:5d} {shape:11s} "
                f"outer={t_outer*1e3:8.2f}ms  alpha_iter={t_iter*1e3:8.2f}ms  "
                f"solves={t_solves*1e3:7.3f}ms  iter/outer={t_iter/t_outer:5.2f}  "
                f"k_cold={[iters[f][0] for f in [0.5,0.1,0.01]]} "
                f"k_warm={[iters[f][1] for f in [0.5,0.1,0.01]]}"
            )

    with open("qr_alpha_baseline.json", "w") as fh:
        json.dump(rows, fh, indent=2)
