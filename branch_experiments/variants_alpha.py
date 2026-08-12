"""Candidate replacements for DESC's QR trust-region alpha loop.

All variants solve the same subproblem as
desc/optimize/tr_subproblems.py::trust_region_step_exact_qr,

    min_p ||J p + f||^2 + alpha ||p||^2   s.t.  ||p|| = trust_radius,

driven by the SAME safeguarded Hebden/Reinsch root-finder on
phi(alpha) = ||p(alpha)|| - trust_radius, so the alpha ITERATES are identical
across variants and only the per-iteration COST differs.

Variants
--------
A  qr_dense    : master. Per alpha: dense QR of the 2n x n matrix [R; sqrt(a) I].
B  cho_shift   : Cholesky of (R'R + a I) with G = R'R cached across alphas.
                 Flop-reduced (10x) stand-in for a structured retriangularization;
                 squares the condition number.
C  svd_of_R    : one SVD of R (n x n), then each alpha is O(n) -- no matvec at all
                 inside the loop, one matvec to form p at the end.
D  svd_of_J    : DESC's existing tr_method="svd" (SVD of the m x n J_a), for
                 reference -- it already has a cheap alpha loop.

Key identity behind C, with R = W S V' (S = diag of singular values):
    (R'R + a I)^-1 R'z  =  V diag(1/(s^2 + a)) V' (R'z)
so with c = V'(R'z) computed once,
    p(a)      = -V (c / (s^2 + a))
    ||p(a)||^2 = sum_i c_i^2 / (s_i^2 + a)^2         <- O(n), no matvec
    ||q(a)||^2 = sum_i c_i^2 / (s_i^2 + a)^3         <- O(n), the Hebden denominator
(q solves Rtil' q = p where Rtil'Rtil = R'R + aI, so ||q||^2 = p'(R'R+aI)^-1 p.)
Correct for rank-deficient and for wide J (R is k x n, k = min(m,n) < n): the
component of p in null(R) is zero for any a > 0, and V' R' z lives in range(V').
"""

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.lax import cond, while_loop
from jax.scipy.linalg import cho_factor, cho_solve, qr_multiply, solve_triangular

RTOL = 0.01
MAX_ITER = 10


def solve_triangular_regularized(R, b, lower=False):
    """Verbatim from desc/optimize/utils.py."""
    dr = jnp.diag(R)
    denom = jnp.where(dr == 0, 1, dr)
    dri = jnp.where(dr == 0, 0, 1 / denom)
    Rs = R * dri[:, None]
    b = dri * b
    return solve_triangular(Rs, b, unit_diagonal=True, lower=lower)


def chol(A):
    """Verbatim from desc/optimize/utils.py (modified Cholesky w/ shift)."""
    A = (A + A.T) / 2
    d = jnp.diag(A)
    scale = jnp.maximum(jnp.max(jnp.abs(d)), 1e-16)
    A = A + jnp.eye(A.shape[0]) * scale * 1e-12
    return cho_factor(A, lower=True)[0]


# ---------------------------------------------------------------------------
# A: master
# ---------------------------------------------------------------------------
@jit
def step_qr_dense(p_newton, z, R, trust_radius, initial_alpha=0.0):
    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(R.T @ z) / trust_radius
        alpha = jnp.clip(initial_alpha, 0.0, alpha_upper)
        n = R.shape[1]
        zp = jnp.concatenate([z, jnp.zeros(n)])

        def loop_cond(s):
            return (jnp.abs(s[4]) > RTOL * trust_radius) & (s[5] < MAX_ITER)

        def loop_body(s):
            p, alpha, a_lo, a_hi, phi, k = s
            A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
            Qtz, Rtil = qr_multiply(A, zp, mode="right")
            p = solve_triangular_regularized(Rtil, -Qtz)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            a_hi = jnp.where(phi < 0, alpha, a_hi)
            a_lo = jnp.where(phi > 0, alpha, a_lo)
            q = solve_triangular_regularized(Rtil.T, p, lower=True)
            alpha += (p_norm / jnp.linalg.norm(q)) ** 2 * phi / trust_radius
            return p, jnp.clip(alpha, a_lo, a_hi), a_lo, a_hi, phi, k + 1

        p, alpha, *_ = while_loop(
            loop_cond, loop_body, (p_newton, alpha, 0.0, alpha_upper, jnp.inf, 0)
        )
        return p * trust_radius / jnp.linalg.norm(p), True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


# ---------------------------------------------------------------------------
# B: Cholesky of the shifted Gram matrix, G = R'R cached
# ---------------------------------------------------------------------------
@jit
def prep_cho(R, z):
    return R.T @ R, R.T @ z


@jit
def step_cho_shift(p_newton, G, Rtz, trust_radius, initial_alpha=0.0):
    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(Rtz) / trust_radius
        alpha = jnp.clip(initial_alpha, 0.0, alpha_upper)
        n = G.shape[0]

        def loop_cond(s):
            return (jnp.abs(s[4]) > RTOL * trust_radius) & (s[5] < MAX_ITER)

        def loop_body(s):
            p, alpha, a_lo, a_hi, phi, k = s
            L = chol(G + alpha * jnp.eye(n))
            p = cho_solve((L, True), -Rtz)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            a_hi = jnp.where(phi < 0, alpha, a_hi)
            a_lo = jnp.where(phi > 0, alpha, a_lo)
            q = solve_triangular(L, p, lower=True)
            alpha += (p_norm / jnp.linalg.norm(q)) ** 2 * phi / trust_radius
            return p, jnp.clip(alpha, a_lo, a_hi), a_lo, a_hi, phi, k + 1

        p, alpha, *_ = while_loop(
            loop_cond, loop_body, (p_newton, alpha, 0.0, alpha_upper, jnp.inf, 0)
        )
        return p * trust_radius / jnp.linalg.norm(p), True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


# ---------------------------------------------------------------------------
# C: SVD of R -- the proposal.  Per alpha: O(n), zero matvecs.
# ---------------------------------------------------------------------------
@jit
def prep_svd_of_R(R, z):
    """One-time work: SVD of R, plus c = V'(R'z) and the truncated Newton step."""
    W, s, Vt = jnp.linalg.svd(R, full_matrices=False)
    y = W.T @ z
    c = s * y  # == V' R' z, computed without forming R'z
    # rank-truncated Newton step, using the same threshold policy as the svd branch
    thresh = jnp.finfo(s.dtype).eps * z.size * s[0]
    s_inv = jnp.where(s > thresh, 1 / jnp.where(s == 0, 1, s), 0.0)
    p_newton = -Vt.T @ (y * s_inv)
    return p_newton, Vt, s, c


@jit
def step_svd_of_R(p_newton, Vt, s, c, trust_radius, initial_alpha=0.0):
    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(c) / trust_radius  # ||R'z|| == ||c||
        alpha = jnp.clip(initial_alpha, 0.0, alpha_upper)
        s2 = s**2
        c2 = c**2

        def loop_cond(st):
            return (jnp.abs(st[3]) > RTOL * trust_radius) & (st[4] < MAX_ITER)

        def loop_body(st):
            alpha, a_lo, a_hi, phi, k = st
            # c_i == 0 exactly wherever s_i == 0, so a rank-deficient R with
            # alpha == 0 would give 0/0. Guard the denominator the same way the
            # existing svd branch does (tr_subproblems.py:208).
            d = s2 + alpha
            d = jnp.where(d == 0, 1.0, d)
            p_norm = jnp.sqrt(jnp.sum(c2 / d**2))  # O(n), no matvec
            phi = p_norm - trust_radius
            a_hi = jnp.where(phi < 0, alpha, a_hi)
            a_lo = jnp.where(phi > 0, alpha, a_lo)
            q_norm = jnp.sqrt(jnp.sum(c2 / d**3))  # O(n), no triangular solve
            alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
            return jnp.clip(alpha, a_lo, a_hi), a_lo, a_hi, phi, k + 1

        alpha, *_ = while_loop(
            loop_cond, loop_body, (alpha, 0.0, alpha_upper, jnp.inf, 0)
        )
        dn = s2 + alpha
        p = -Vt.T @ (c / jnp.where(dn == 0, 1.0, dn))  # the ONLY matvec, once
        return p * trust_radius / jnp.linalg.norm(p), True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


# ---------------------------------------------------------------------------
# D: DESC's existing svd branch (SVD of J_a), for reference
# ---------------------------------------------------------------------------
@jit
def prep_svd_of_J(J_a, f_a):
    U, s, Vt = jnp.linalg.svd(J_a, full_matrices=False)
    uf = U.T @ f_a
    thresh = jnp.finfo(s.dtype).eps * f_a.size * s[0]
    s_inv = jnp.where(s > thresh, 1 / jnp.where(s == 0, 1, s), 0.0)
    return -Vt.T @ (uf * s_inv), Vt, s, s * uf


# ---------------------------------------------------------------------------
# Reference: structured Givens retriangularization of [R; sqrt(a) I] in numpy.
# Answers "is there a cheap alpha_1 -> alpha_2 update?" empirically:
# this is the from-scratch structured cost, and it is O(n^3/6), not O(n^2).
# ---------------------------------------------------------------------------
def givens_retriangularize(R, z, alpha, count_flops=False):
    """Return (Rtil, Qtz, nflops) with Rtil'Rtil = R'R + alpha I.

    Annihilates the rows of sqrt(alpha) I bottom-up against the triangle; each
    row i needs n-i rotations, each touching O(n-k) entries -> ~n^3/6 total.
    """
    R = np.array(R, dtype=float, copy=True)
    k, n = R.shape
    S = np.zeros((n, n))
    S[:k] = R
    b = np.zeros(n)
    b[:k] = z
    sa = np.sqrt(alpha)
    flops = 0
    for i in range(n - 1, -1, -1):
        w = np.zeros(n)  # the row being annihilated: sqrt(alpha) e_i'
        w[i] = sa
        wb = 0.0
        for j in range(i, n):
            a_, b_ = S[j, j], w[j]
            r = np.hypot(a_, b_)
            if r == 0.0:
                continue
            c_, s_ = a_ / r, b_ / r
            seg = slice(j, n)
            Sj, wj = S[j, seg].copy(), w[seg].copy()
            S[j, seg] = c_ * Sj + s_ * wj
            w[seg] = -s_ * Sj + c_ * wj
            bj, wbn = b[j], wb
            b[j] = c_ * bj + s_ * wbn
            wb = -s_ * bj + c_ * wbn
            flops += 6 * (n - j) + 6
    return (S, b, flops) if count_flops else (S, b)
