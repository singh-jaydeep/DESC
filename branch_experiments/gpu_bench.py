"""GPU scaling of the four alpha-loop routes. Self-contained (runs on Modal).

Measures, per state-vector size n and aspect ratio m/n:
  prep_qr        outer QR of J_a  (what lsqtr already pays)
  prep_svdR      + SVD of R       (variant C additional prep)
  prep_svdJ      SVD of J_a       (variant D, DESC's existing tr_method="svd")
  iter_dense     one alpha iteration, dense QR of [R; sqrt(a) I]   (master)
  iter_struct    one alpha iteration, BLOCKED structured QR         (variant E2)
  iter_givens    one alpha iteration, sequential Givens (fori_loop) (variant E1)
  call_svdR      a whole subproblem call via SVD-of-R               (variant C)

The E1 measurement is the one that answers "would a Givens approach work on a
GPU?" directly: same math as E2, but n^2/2 sequential rank-1 updates instead of
n/b blocked panel updates.
"""

import functools
import json
import platform
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop
from jax.scipy.linalg import qr_multiply, solve_triangular


def solve_triangular_regularized(R, b, lower=False):
    dr = jnp.diag(R)
    denom = jnp.where(dr == 0, 1, dr)
    dri = jnp.where(dr == 0, 0, 1 / denom)
    Rs = R * dri[:, None]
    b = dri * b
    return solve_triangular(Rs, b, unit_diagonal=True, lower=lower)


# ---------------- A: master, dense QR of the stacked matrix ------------------
@jit
def iter_dense(R, zp, alpha):
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
    Qtz, Rtil = qr_multiply(A, zp, mode="right")
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q


# ---------------- E2: blocked structured (LAPACK dtpqrt-style) ---------------
def _apply_QT_wy(packed, taus, C):
    M, k = packed.shape
    V = jnp.where(
        jnp.tril(jnp.ones((M, k), bool), -1), packed, jnp.eye(M, k, dtype=packed.dtype)
    )
    live = taus != 0
    V = V * live
    taus_safe = jnp.where(live, taus, 1.0)
    T_inv = V.T @ V - jnp.diag(1.0 / taus_safe)
    with jax.default_matmul_precision("highest"):
        return C - V @ solve_triangular(T_inv, V.T @ C, lower=True)


@functools.partial(jit, static_argnames=("block",))
def iter_struct(R, z, alpha, block=128):
    n = R.shape[1]
    k_ = R.shape[0]
    M = jnp.zeros((2 * n, n + 1))
    M = M.at[:k_, :n].set(R)
    M = M.at[n:, :n].set(jnp.sqrt(alpha) * jnp.eye(n))
    M = M.at[:k_, n].set(z)
    for kb in range((n + block - 1) // block):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        idx = jnp.concatenate([jnp.arange(c0, c1), n + jnp.arange(0, c1)])
        Sub = M[idx, c0:]
        h, taus = jnp.linalg.qr(Sub[:, :bk], mode="raw")
        Sub = _apply_QT_wy(h.swapaxes(-1, -2), taus, Sub)
        M = M.at[idx, c0:].set(Sub)
    Rtil, Qtz = jnp.triu(M[:n, :n]), M[:n, n]
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q


# ---------------- E1: sequential Givens, the O(n^2)-rotation route -----------
@jit
def iter_givens(R, z, alpha):
    """Annihilate sqrt(alpha) I row by row with Givens rotations.

    n outer steps (jax fori_loop), each an inner fori_loop of up to n rotations
    -> ~n^2/2 SEQUENTIAL rank-1 updates. This is the structure that cannot be
    expressed as BLAS-3, and the point of measuring it is to show what that
    costs on a GPU relative to the blocked form.
    """
    n = R.shape[1]
    k_ = R.shape[0]
    S = jnp.zeros((n, n)).at[:k_, :].set(R)
    b = jnp.zeros(n).at[:k_].set(z)
    sa = jnp.sqrt(alpha)

    def outer(i, carry):
        S, b = carry
        w = jnp.zeros(n).at[i].set(sa)

        def inner(j, st):
            S, b, w, wb = st
            a_, b_ = S[j, j], w[j]
            r = jnp.hypot(a_, b_)
            safe = r > 0
            c_ = jnp.where(safe, a_ / jnp.where(safe, r, 1.0), 1.0)
            s_ = jnp.where(safe, b_ / jnp.where(safe, r, 1.0), 0.0)
            mask = jnp.arange(n) >= j
            Sj, wj = jnp.where(mask, S[j], 0.0), jnp.where(mask, w, 0.0)
            S = S.at[j].set(jnp.where(mask, c_ * Sj + s_ * wj, S[j]))
            w = jnp.where(mask, -s_ * Sj + c_ * wj, w)
            bj = b[j]
            b = b.at[j].set(c_ * bj + s_ * wb)
            wb = -s_ * bj + c_ * wb
            return S, b, w, wb

        S, b, w, wb = fori_loop(i, n, inner, (S, b, w, 0.0))
        return S, b

    S, b = fori_loop(0, n, outer, (S, b))
    Rtil = jnp.triu(S)
    p = solve_triangular_regularized(Rtil, -b)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q


# ---------------- C: SVD of R -----------------------------------------------
@jit
def prep_svd_of_R(R, z):
    W, s, Vt = jnp.linalg.svd(R, full_matrices=False)
    y = W.T @ z
    c = s * y
    thresh = jnp.finfo(s.dtype).eps * z.size * s[0]
    s_inv = jnp.where(s > thresh, 1 / jnp.where(s == 0, 1, s), 0.0)
    return -Vt.T @ (y * s_inv), Vt, s, c


@jit
def call_svd_of_R(Vt, s, c, trust_radius, initial_alpha):
    """Whole subproblem: the full alpha root-find plus the single final matvec."""
    from jax.lax import while_loop

    alpha_upper = jnp.linalg.norm(c) / trust_radius
    alpha = jnp.clip(initial_alpha, 0.0, alpha_upper)
    s2, c2 = s**2, c**2

    def cond(st):
        return (jnp.abs(st[3]) > 0.01 * trust_radius) & (st[4] < 10)

    def body(st):
        alpha, a_lo, a_hi, phi, k = st
        d = s2 + alpha
        d = jnp.where(d == 0, 1.0, d)
        p_norm = jnp.sqrt(jnp.sum(c2 / d**2))
        phi = p_norm - trust_radius
        a_hi = jnp.where(phi < 0, alpha, a_hi)
        a_lo = jnp.where(phi > 0, alpha, a_lo)
        q_norm = jnp.sqrt(jnp.sum(c2 / d**3))
        alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
        return jnp.clip(alpha, a_lo, a_hi), a_lo, a_hi, phi, k + 1

    alpha, *_ = while_loop(cond, body, (alpha, 0.0, alpha_upper, jnp.inf, 0))
    dn = s2 + alpha
    return -Vt.T @ (c / jnp.where(dn == 0, 1.0, dn)), alpha


@jit
def prep_qr(J, f):
    Qt_fa, R = qr_multiply(J, f, mode="right")
    return Qt_fa, R, solve_triangular_regularized(R, -Qt_fa)


@jit
def prep_svd_of_J(J, f):
    U, s, Vt = jnp.linalg.svd(J, full_matrices=False)
    uf = U.T @ f
    thresh = jnp.finfo(s.dtype).eps * f.size * s[0]
    s_inv = jnp.where(s > thresh, 1 / jnp.where(s == 0, 1, s), 0.0)
    return -Vt.T @ (uf * s_inv), Vt, s, s * uf


def make_problem(m, n, seed=0, cond_exp=6.0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
    V, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
    s = np.logspace(0, -cond_exp, min(m, n))
    return jnp.asarray((U * s) @ V.T), jnp.asarray(rng.standard_normal(m))


def timeit(fn, *a, reps=3, warmup=True):
    if warmup:
        jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


if __name__ == "__main__":
    dev = jax.devices()[0]
    meta = dict(
        platform=dev.platform,
        device=str(dev.device_kind),
        jax=jax.__version__,
        host=platform.platform(),
    )
    print("DEVICE:", meta, flush=True)

    rows = []
    # E1 (sequential Givens) is measured only up to n=1000: it is O(n^2)
    # sequential kernel launches, and the trend is unambiguous well before 4000.
    GIVENS_MAX_N = 1000
    for n in [500, 1000, 2000, 4000]:
        for ratio in [1.5, 3.0]:
            m = int(ratio * n)
            J, f = make_problem(m, n)
            r = dict(n=n, m=m, ratio=ratio, **meta)

            r["prep_qr"] = timeit(prep_qr, J, f)
            z, R, p_newton = prep_qr(J, f)
            zp = jnp.concatenate([z, jnp.zeros(n)])

            r["prep_svdR_extra"] = timeit(prep_svd_of_R, R, z)
            _, Vt, s, c = prep_svd_of_R(R, z)
            r["prep_svdJ"] = timeit(prep_svd_of_J, J, f)

            r["iter_dense"] = timeit(iter_dense, R, zp, 1.0)
            best = None
            for blk in [64, 128, 256]:
                t = timeit(iter_struct, R, z, 1.0, blk)
                r[f"iter_struct_b{blk}"] = t
                if best is None or t < best[1]:
                    best = (blk, t)
            r["iter_struct"], r["struct_block"] = best[1], best[0]

            if n <= GIVENS_MAX_N:
                r["iter_givens"] = timeit(iter_givens, R, z, 1.0, reps=1)
                # correctness of E1 against the dense reference
                pg, _ = iter_givens(R, z, 1.0)
                pd, _ = iter_dense(R, zp, 1.0)
                r["givens_relerr"] = float(
                    jnp.linalg.norm(pg - pd) / jnp.linalg.norm(pd)
                )
            ps, _ = iter_struct(R, z, 1.0, r["struct_block"])
            pd, _ = iter_dense(R, zp, 1.0)
            r["struct_relerr"] = float(jnp.linalg.norm(ps - pd) / jnp.linalg.norm(pd))

            tr = 0.1 * float(jnp.linalg.norm(p_newton))
            r["call_svdR"] = timeit(call_svd_of_R, Vt, s, c, tr, 0.0)

            rows.append(r)
            g = r.get("iter_givens")
            print(
                f"n={n:5d} m/n={ratio:4.1f} | prepQR={r['prep_qr']*1e3:8.2f} "
                f"+svdR={r['prep_svdR_extra']*1e3:8.2f} svdJ={r['prep_svdJ']*1e3:8.2f} "
                f"| dense={r['iter_dense']*1e3:8.2f} struct(b{r['struct_block']})="
                f"{r['iter_struct']*1e3:8.2f} ({r['iter_dense']/r['iter_struct']:4.2f}x) "
                f"givens={'%.1f' % (g*1e3) if g else '   skip':>9} "
                f"| svdR_call={r['call_svdR']*1e3:7.3f}",
                flush=True,
            )

    import os

    os.makedirs("out", exist_ok=True)
    json.dump(rows, open("out/gpu_results.json", "w"), indent=2)
