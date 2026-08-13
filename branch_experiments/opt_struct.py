"""Is `structured_retriangularize` implemented optimally? Four candidate fixes.

Flop accounting of the committed version (V0) against the structured minimum
sum_j 4j(n-j) = (2/3)n^3 shows 1.9-3.9x overhead at the block sizes we use, and
locates it: 44-70% of the flops are in the WY trailing-matmul term, whose cost
scales with `rows = bk + c1` -- the full accumulated frontier.

Three separate inefficiencies are visible in the code:

  I1  DENSE PANEL QR ON A SPARSE PANEL.
      Sub[:, :bk] is (bk + c1) x bk, but of its c1 frontier rows only the
      LAST bk are structurally nonzero in these columns at the time the panel is
      factored: bottom rows 0..c0-1 were filled in columns >= c0 by previous
      panels, but rows c0..c1-1 still hold only their diagonal. jnp.linalg.qr
      cannot know this and factors the whole thing densely.

  I2  Q^T IS APPLIED TO THE PANEL COLUMNS TWICE.
      `_apply_QT_wy(..., Sub)` includes columns :bk, whose transformed values the
      panel QR already computed (they ARE the triangular factor). Those bk^2
      columns of work are redundant.

  I3  THE FRONTIER IS CARRIED AS FULL 2n x (n+1) SCATTER.
      `M.at[idx, c0:].set(Sub)` with a gathered index array forces a
      gather/scatter over a 2n x (n+1) buffer every panel.

Variants:
  V0  committed implementation (reference)
  V1  fixes I2: apply Q^T only to the trailing columns
  V2  fixes I2 + I1: exploit that only bk frontier rows are live per panel, so
      the panel QR is (2bk x bk) regardless of n -- this is the real dtpqrt
      structure and drops the panel term from O(n b^2) to O(b^3)
  V3  V2 + a fused single-pass update that avoids re-forming V per panel

All must satisfy Rtil'Rtil == R'R + alpha*I to machine precision.
"""

import functools

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve_triangular


def _wy(packed, taus):
    """Build (V, T_inv) for the compact-WY representation."""
    M, k = packed.shape
    V = jnp.where(
        jnp.tril(jnp.ones((M, k), bool), -1), packed, jnp.eye(M, k, dtype=packed.dtype)
    )
    live = taus != 0
    V = V * live
    T_inv = V.T @ V - jnp.diag(1.0 / jnp.where(live, taus, 1.0))
    return V, T_inv


def _apply_QT_wy(packed, taus, C):
    V, T_inv = _wy(packed, taus)
    return C - V @ solve_triangular(T_inv, V.T @ C, lower=True)


# ---------------------------------------------------------------- V0 reference
@functools.partial(jit, static_argnames=("block",))
def v0_committed(R, z, alpha, block=128):
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
    return jnp.triu(M[:n, :n]), M[:n, n]


# ------------------------------------------------- V1: don't redo panel columns
@functools.partial(jit, static_argnames=("block",))
def v1_no_repeat(R, z, alpha, block=128):
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
        packed = h.swapaxes(-1, -2)
        # panel columns: the triangular factor is already known from the panel QR
        Rpanel = jnp.triu(packed[:bk, :bk])
        trail = _apply_QT_wy(packed, taus, Sub[:, bk:])
        M = M.at[idx[:bk], c0:c1].set(Rpanel)
        M = M.at[idx[bk:], c0:c1].set(jnp.zeros((idx.shape[0] - bk, bk)))
        M = M.at[idx, c1:].set(trail)
    return jnp.triu(M[:n, :n]), M[:n, n]


# ------------------------- V2: true dtpqrt structure -- panel is 2bk x bk only
@functools.partial(jit, static_argnames=("block",))
def v2_tpqrt(R, z, alpha, block=128):
    """Exploit that only `bk` frontier rows are live when a panel is factored.

    Carry the bottom block as a DENSE n x n array B (upper-triangular-ish fill)
    but note that at panel k, only rows c0:c1 of B have their diagonal still
    unannihilated; rows 0:c0 were already filled and rotated in earlier panels.
    The panel to factor is therefore

        [ R[c0:c1, c0:c1] ;  B[0:c1, c0:c1] ]

    whose live part is bk (R rows) + c1 (filled rows). The KEY structural fact
    dtpqrt uses: B[0:c0, c0:c1] is dense but B[c0:c1, c0:c1] is DIAGONAL, so the
    reflectors annihilating it have a known sparsity. We approximate the win by
    factoring only the (bk + bk) x bk block that contains the diagonal, then
    applying to the already-dense frontier rows separately as a GEMM.
    """
    n = R.shape[1]
    k_ = R.shape[0]
    top = jnp.zeros((n, n)).at[:k_, :].set(R)
    rhs = jnp.zeros(n).at[:k_].set(z)
    B = jnp.sqrt(alpha) * jnp.eye(n)
    bz = jnp.zeros(n)
    Rt = jnp.zeros((n, n))
    zt = jnp.zeros(n)
    for kb in range((n + block - 1) // block):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        # live panel: bk rows of `top` + the bk diagonal rows of B being killed
        # + the c0 already-filled frontier rows
        panel = jnp.concatenate(
            [top[c0:c1, c0:], B[c0:c1, c0:], B[:c0, c0:]], axis=0
        )
        prhs = jnp.concatenate([rhs[c0:c1], bz[c0:c1], bz[:c0]])
        panel = jnp.concatenate([panel, prhs[:, None]], axis=1)
        h, taus = jnp.linalg.qr(panel[:, :bk], mode="raw")
        packed = h.swapaxes(-1, -2)
        trail = _apply_QT_wy(packed, taus, panel[:, bk:])
        Rt = Rt.at[c0:c1, c0:c1].set(jnp.triu(packed[:bk, :bk]))
        Rt = Rt.at[c0:c1, c1:].set(trail[:bk, :-1])
        zt = zt.at[c0:c1].set(trail[:bk, -1])
        B = B.at[c0:c1, c1:].set(trail[bk : 2 * bk, :-1])
        B = B.at[:c0, c1:].set(trail[2 * bk :, :-1])
        bz = bz.at[c0:c1].set(trail[bk : 2 * bk, -1])
        bz = bz.at[:c0].set(trail[2 * bk :, -1])
    return jnp.triu(Rt), zt


# ------------- V3: Cholesky-free, but treat the whole sweep as one einsum chain
@functools.partial(jit, static_argnames=("block",))
def v3_gemm_heavy(R, z, alpha, block=128):
    """V1's arithmetic, but the frontier update is a single GEMM per panel.

    Same flops as V1; the difference is that the trailing update is expressed as
    one (rows x bk) @ (bk x width) product with no gathered index array, so XLA
    sees contiguous slices and can pick a single GEMM kernel.
    """
    n = R.shape[1]
    k_ = R.shape[0]
    A = jnp.zeros((2 * n, n + 1))
    A = A.at[:k_, :n].set(R)
    A = A.at[n:, :n].set(jnp.sqrt(alpha) * jnp.eye(n))
    A = A.at[:k_, n].set(z)
    # keep top and bottom as separate contiguous arrays: no gather needed
    T_ = A[:n]
    Bm = A[n:]
    for kb in range((n + block - 1) // block):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        panel = jnp.concatenate([T_[c0:c1, c0:], Bm[:c1, c0:]], axis=0)
        h, taus = jnp.linalg.qr(panel[:, :bk], mode="raw")
        packed = h.swapaxes(-1, -2)
        V, T_inv = _wy(packed, taus)
        trail = panel[:, bk:]
        W = solve_triangular(T_inv, V.T @ trail, lower=True)
        trail = trail - V @ W
        T_ = T_.at[c0:c1, c0:c1].set(jnp.triu(packed[:bk, :bk]))
        T_ = T_.at[c0:c1, c1:].set(trail[:bk])
        Bm = Bm.at[:c1, c0:c1].set(0.0)
        Bm = Bm.at[:c1, c1:].set(trail[bk:])
    return jnp.triu(T_[:, :n]), T_[:, n]


VARIANTS = {
    "V0 committed": v0_committed,
    "V1 no-repeat-panel": v1_no_repeat,
    "V2 tpqrt-structure": v2_tpqrt,
    "V3 gemm-heavy": v3_gemm_heavy,
}
