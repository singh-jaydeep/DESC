"""Blocked structured retriangularization of A = [R; sqrt(alpha) I]  (variant E).

WHY THIS IS NOT THE SAME AS "GIVENS ROTATIONS"
----------------------------------------------
Two different structured algorithms get conflated:

  E1  Givens-per-entry.  Annihilate the bottom block one scalar at a time.
      ~n^3 flops (measured, see givens_flops.json), but n^2/2 SEQUENTIAL
      rank-1 rotations. Latency-bound; hostile to a GPU and to jit.

  E2  Blocked Householder, i.e. LAPACK's ``dtpqrt`` ("QR of a triangular-
      pentagonal matrix") -- the routine that exists precisely for the
      [dense; triangular] stacked structure and is used in tall-skinny and
      updating QR. Householder-per-COLUMN in panels of width b:
        - only n/b sequential steps (32 steps at n=4000, b=128)
        - each step is a small panel QR plus GEMM updates of the trailing
          panel -> BLAS-3, high arithmetic intensity, GPU-friendly
        - ~n^3/3 flops: a 10x reduction over the dense 10n^3/3, and the same
          flop count as Cholesky but with QR's conditioning (no squaring
          of kappa(J))

This module implements E2. At block k the active panel has (k+2)b rows, so
    sum_k 2*(k+2)b*b*(n-kb)  ->  n^3/3.

Structure of the sweep. With B initialized to sqrt(alpha) I, at block column
k = [c0, c1) the rows that can have nonzeros in those columns are the b rows
R[c0:c1] plus B rows 0..c1-1 (B rows >= c1 still have their single diagonal
entry at a column >= c1). Applying Q^T fills B rows 0..c1-1 in columns >= c1,
and those rows are all included at step k+1 -- so the pentagonal frontier
grows by exactly b per step and nothing is missed.

Reflectors are applied via the compact-WY / UT transform (Q = I - V T V^T)
so each application is two GEMMs rather than b rank-1 updates.
"""

import functools

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve_triangular


def _apply_QT_wy(packed, taus, C):
    """Return Q^T @ C, with Q the product of the Householder reflectors.

    ``packed`` is (M, k): reflectors strictly below the diagonal, unit diagonal
    implied. Uses the identity T^-1 + T^-T = V^T V (Joffrain & Low 2006), the
    same route DESC's vendored ``qr_multiply`` fallback takes in backend.py.
    """
    M, k = packed.shape
    V = jnp.where(
        jnp.tril(jnp.ones((M, k), bool), -1), packed, jnp.eye(M, k, dtype=packed.dtype)
    )
    # A zero tau means that reflector is the identity; zero its column so it
    # contributes nothing instead of producing 1/0.
    live = taus != 0
    V = V * live
    taus_safe = jnp.where(live, taus, 1.0)
    T_inv = V.T @ V - jnp.diag(1.0 / taus_safe)
    with jax.default_matmul_precision("highest"):
        return C - V @ solve_triangular(T_inv, V.T @ C, lower=True)


@functools.partial(jit, static_argnames=("block",))
def structured_retriangularize(R, z, alpha, block=64):
    """QR of [R; sqrt(alpha) I] exploiting the structure. Returns (Rtil, Qtz).

    Satisfies Rtil.T @ Rtil == R.T @ R + alpha * I and
    Qtz == (Q.T @ [z; 0])[:n], matching ``qr_multiply(vstack([R, sqrt(a) I]), zp)``.
    """
    n = R.shape[1]
    k_ = R.shape[0]
    # Augmented work array: [R; sqrt(alpha) I] with the RHS as a trailing column.
    top = jnp.zeros((n, n)).at[:k_, :].set(R)
    M = jnp.zeros((2 * n, n + 1))
    M = M.at[:n, :n].set(top)
    M = M.at[n:, :n].set(jnp.sqrt(alpha) * jnp.eye(n))
    M = M.at[:k_, n].set(z)

    nblocks = (n + block - 1) // block
    for kb in range(nblocks):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        # b rows of the R part, plus the grown pentagonal frontier of the B part
        idx = jnp.concatenate(
            [jnp.arange(c0, c1), n + jnp.arange(0, c1)]
        )
        Sub = M[idx, c0:]
        h, taus = jnp.linalg.qr(Sub[:, :bk], mode="raw")
        packed = h.swapaxes(-1, -2)
        Sub = _apply_QT_wy(packed, taus, Sub)
        M = M.at[idx, c0:].set(Sub)

    return jnp.triu(M[:n, :n]), M[:n, n]


@functools.partial(jit, static_argnames=("block",))
def structured_alpha_iteration(R, z, alpha, block=64):
    """One alpha iteration using the structured factorization (variant E2)."""
    from variants_alpha import solve_triangular_regularized

    Rtil, Qtz = structured_retriangularize(R, z, alpha, block=block)
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p, q, Rtil
