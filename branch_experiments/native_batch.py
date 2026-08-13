"""Why is vmap over the structured QR slow, and can a native batched version fix it?

The vmap result (0.21-0.48x for B=2..8) should NOT be inherent. Batching B alphas
does not add sequential steps: the panel loop still runs n/b times, each step just
B times wider. If the routine is latency-bound, B alphas ought to cost close to
ONE sequential call, i.e. a gain near B. Measuring 0.2-0.48x means B=2 is ~4x
SLOWER than a single call -- a pathology in how the batched ops are emitted, not a
property of the algorithm.

Three suspects, each measured separately below:

  S1  BATCHED PANEL QR. Under vmap, jnp.linalg.qr(mode="raw") on the panel becomes
      a batched geqrf. On GPU, batched QR for these shapes may fall back to a
      serial loop over the batch inside cuSOLVER, which would destroy the win.

  S2  BATCHED GATHER/SCATTER. V0 addresses rows through a gathered index array
      (`M.at[idx, c0:].set(...)`). Under vmap that becomes a batched scatter with
      a dynamic index operand -- one of the slowest patterns XLA emits.

  S3  MEMORY TRAFFIC. B copies of a 2n x (n+1) work array. At n=1000, B=8 that is
      8 * 2000 * 1001 * 8 B = 128 MB rewritten every panel step.

The native implementation (`native_batched`) attacks all three:
  - batch is a LEADING axis on contiguous arrays; every update is an einsum, so
    XLA sees batched GEMM, not scatter
  - the WY triangular factor T is formed EXPLICITLY (as LAPACK's larft does), so
    applying it is a GEMM instead of a batched triangular solve
  - the panel QR is the only non-GEMM op, and it is called on the smallest shape
    that is mathematically necessary

BREAK-EVEN BAR. Batching also requires an algorithmic change: the Hebden iterates
are sequential (alpha_{k+1} depends on phi(alpha_k)), so a batched loop must
evaluate several candidate alphas speculatively per round. The sequential loop
needs 3.46 factorizations on average (measured, instrument_solve.py). A batched
scheme doing r rounds of B evaluations costs r*B factorizations of work but only
r rounds of latency, so it pays off only if

    per-alpha batched cost / per-alpha sequential cost  <  3.46 / (r*B)

For r=2, B=4 that bar is 0.43x; for r=2, B=8 it is 0.22x. This is the number the
measurements below have to beat -- not merely "batching helps".
"""

import functools
import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import qr_multiply, solve_triangular

sys.path.insert(0, ".")
from opt_struct import v0_committed, v3_gemm_heavy


def make_problem(m, n, seed=0, cond_exp=9.0):
    rng = np.random.default_rng(seed)
    k = min(m, n)
    U, _ = np.linalg.qr(rng.standard_normal((m, k)))
    V, _ = np.linalg.qr(rng.standard_normal((n, k)))
    s = np.logspace(0, -cond_exp, k)
    return jnp.asarray((U * s) @ V.T), jnp.asarray(rng.standard_normal(m))


def timeit(fn, *a, reps=7, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ------------------------------------------------------- native batched version
def _wy_T_batched(packed, taus):
    """Form V and the EXPLICIT WY factor T for a batch of panels.

    packed: (B, M, k) reflectors below the diagonal; taus: (B, k).
    Returns V (B, M, k) and T (B, k, k) with  Q = I - V T V'.
    Forming T explicitly (LAPACK larft) turns the application into two GEMMs,
    avoiding a batched triangular solve inside the panel loop.
    """
    B, M, k = packed.shape
    mask = jnp.tril(jnp.ones((M, k), bool), -1)
    V = jnp.where(mask[None], packed, jnp.eye(M, k, dtype=packed.dtype)[None])
    live = (taus != 0)[:, None, :]
    V = V * live
    tv = jnp.where(taus != 0, taus, 1.0)
    # T^-1 = V'V - diag(1/tau)  (Joffrain & Low). The identity holds for the LOWER
    # triangle only -- V'V is symmetric, and the reference implementation reaches
    # the same result via solve_triangular(..., lower=True), which implicitly
    # discards the upper part. Inverting the full symmetric matrix is WRONG
    # (measured: 65% relative error); take tril first.
    T_inv = jnp.einsum("bmk,bml->bkl", V, V) - jax.vmap(jnp.diag)(1.0 / tv)
    T = jnp.linalg.inv(jnp.tril(T_inv))
    return V, T


@functools.partial(jit, static_argnames=("block",))
def native_batched(R, z, alphas, block=128):
    """Structured QR of [R; sqrt(alpha_i) I] for a BATCH of alphas.

    Batch is the leading axis throughout; all updates are einsums on contiguous
    arrays. Returns (Rtil, Qtz) with leading batch dimension.
    """
    n = R.shape[1]
    k_ = R.shape[0]
    B = alphas.shape[0]

    top = jnp.zeros((B, n, n + 1))
    top = top.at[:, :k_, :n].set(R[None])
    top = top.at[:, :k_, n].set(z[None])
    eye = jnp.eye(n)
    bot = jnp.zeros((B, n, n + 1))
    bot = bot.at[:, :, :n].set(jnp.sqrt(alphas)[:, None, None] * eye[None])

    for kb in range((n + block - 1) // block):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        # contiguous slices only: b rows of `top` + the c1-row frontier of `bot`
        panel = jnp.concatenate([top[:, c0:c1, c0:], bot[:, :c1, c0:]], axis=1)
        h, taus = jnp.linalg.qr(panel[:, :, :bk], mode="raw")
        packed = jnp.swapaxes(h, -1, -2)
        V, T = _wy_T_batched(packed, taus)
        trail = panel[:, :, bk:]
        # Q'C = C - V T V' C  : three GEMMs, no triangular solve, no scatter
        W = jnp.einsum("bmk,bmw->bkw", V, trail)
        W = jnp.einsum("blk,bkw->blw", T, W)
        trail = trail - jnp.einsum("bmk,bkw->bmw", V, W)

        top = top.at[:, c0:c1, c0:c1].set(jnp.triu(packed[:, :bk, :bk]))
        top = top.at[:, c0:c1, c1:].set(trail[:, :bk])
        bot = bot.at[:, :c1, c0:c1].set(0.0)
        bot = bot.at[:, :c1, c1:].set(trail[:, bk:])

    return jnp.triu(top[:, :, :n]), top[:, :, n]


@functools.partial(jit, static_argnames=("block",))
def vmap_v0(R, z, alphas, block=128):
    return vmap(lambda a: v0_committed(R, z, a, block=block))(alphas)


@functools.partial(jit, static_argnames=("block",))
def vmap_v3(R, z, alphas, block=128):
    return vmap(lambda a: v3_gemm_heavy(R, z, a, block=block))(alphas)


# ------------------------------------------------------- primitive diagnostics
@functools.partial(jit, static_argnames=())
def prim_qr(A):
    return jnp.linalg.qr(A, mode="raw")


@jit
def prim_gemm(A, B_):
    return jnp.einsum("bmk,bkw->bmw", A, B_)


@jit
def prim_scatter(M, idx, S):
    return M.at[:, idx, :].set(S)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        assert dev.platform == "gpu", f"need a GPU, got {meta}"
    out = dict(meta=meta, prims=[], batched=[])

    rng = np.random.default_rng(0)
    sizes = json.loads(os.environ.get("NB_SIZES", "[491, 723, 1000]"))
    Bs = json.loads(os.environ.get("NB_BATCH", "[1, 2, 4, 8]"))
    block = int(os.environ.get("NB_BLOCK", "128"))

    # ---- S1/S2/S3: how do the primitives scale with batch? -----------------
    print("\nprimitive scaling with batch (n=723, b=128 shapes)")
    n_p, b_p = 723, 128
    for B in Bs:
        A = jnp.asarray(rng.standard_normal((B, b_p + n_p, b_p)))
        t_qr = timeit(prim_qr, A)
        G1 = jnp.asarray(rng.standard_normal((B, b_p + n_p, b_p)))
        G2 = jnp.asarray(rng.standard_normal((B, b_p, n_p)))
        t_gemm = timeit(prim_gemm, G1, G2)
        M = jnp.zeros((B, 2 * n_p, n_p + 1))
        idx = jnp.concatenate([jnp.arange(0, b_p), n_p + jnp.arange(0, b_p)])
        S = jnp.asarray(rng.standard_normal((B, idx.shape[0], n_p + 1)))
        t_sc = timeit(prim_scatter, M, idx, S)
        out["prims"].append(dict(B=B, t_qr=t_qr, t_gemm=t_gemm, t_scatter=t_sc))
        print(
            f"  B={B}: batched QR (B,{b_p+n_p},{b_p})={t_qr*1e3:8.3f} ms "
            f"({t_qr/out['prims'][0]['t_qr']:5.2f}x vs B=1) | "
            f"batched GEMM={t_gemm*1e3:7.3f} ms "
            f"({t_gemm/out['prims'][0]['t_gemm']:5.2f}x) | "
            f"batched scatter={t_sc*1e3:7.3f} ms "
            f"({t_sc/out['prims'][0]['t_scatter']:5.2f}x)",
            flush=True,
        )

    # ---- native batched vs vmap vs sequential ------------------------------
    print("\nnative batched vs vmap vs sequential (block=%d)" % block)
    print("   break-even bars: r=2,B=4 -> 0.43x ; r=2,B=8 -> 0.22x (per-alpha vs sequential)")
    for n in sizes:
        J, f = make_problem(3 * n, n)
        z, R = qr_multiply(J, f, mode="right")
        G = R.T @ R + 1.0 * jnp.eye(n)
        t_seq = timeit(v0_committed, R, z, 1.0, block=block)
        rec = dict(n=n, t_seq=t_seq, per_B={})
        print(f"\n  n={n}: sequential single call = {t_seq*1e3:7.2f} ms", flush=True)
        for B in Bs:
            al = jnp.asarray(np.logspace(-6, 0, B))
            e = {}
            try:
                t_nat = timeit(native_batched, R, z, al, block=block)
                Rt, zt = native_batched(R, z, al, block=block)
                # verify the FIRST batch member against the shifted Gram
                G0 = R.T @ R + float(al[0]) * jnp.eye(n)
                err = float(
                    jnp.linalg.norm(Rt[0].T @ Rt[0] - G0) / jnp.linalg.norm(G0)
                )
                e["native"] = dict(t=t_nat, per=t_nat / B, ratio=(t_nat / B) / t_seq,
                                   gain=(B * t_seq) / t_nat, gram_err=err)
            except Exception as ex:
                e["native"] = dict(error=f"{type(ex).__name__}: {ex}")
            for nm, fn in [("vmap_v0", vmap_v0), ("vmap_v3", vmap_v3)]:
                try:
                    t = timeit(fn, R, z, al, block=block)
                    e[nm] = dict(t=t, per=t / B, ratio=(t / B) / t_seq,
                                 gain=(B * t_seq) / t)
                except Exception as ex:
                    e[nm] = dict(error=f"{type(ex).__name__}: {ex}")
            rec["per_B"][str(B)] = e
            nv = e["native"]
            if "t" in nv:
                print(
                    f"    B={B}: native {nv['t']*1e3:8.2f} ms "
                    f"({nv['per']*1e3:6.2f}/alpha = {nv['ratio']:5.3f}x sequential, "
                    f"gain {nv['gain']:5.2f}x) gram_err={nv['gram_err']:.1e}"
                    + "".join(
                        f" | {nm} {e[nm]['t']*1e3:8.2f} ms ({e[nm]['ratio']:5.3f}x)"
                        if "t" in e[nm] else f" | {nm} FAILED"
                        for nm in ("vmap_v0", "vmap_v3")
                    ),
                    flush=True,
                )
            else:
                print(f"    B={B}: native FAILED {nv['error'][:120]}", flush=True)
        out["batched"].append(rec)
        json.dump(out, open("out/native_batch.json", "w"), indent=2)
