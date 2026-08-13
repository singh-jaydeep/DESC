"""Tighter test: measure the REAL tiled-QR primitives, not a bare matmul.

The previous bound used one b x b matmul as the per-tile-op floor, which is far
too optimistic: a TTQRT is a QR factorization of a stacked 2b x b pair, and a
TTMQR applies those reflectors to a 2b x b tile pair. Here we time the actual
primitives and multiply by the actual op counts.

We also test whether variant E2's measured cost is FUNDAMENTAL or an artifact
of its scatter/gather bookkeeping (`M.at[idx, c0:].set(...)` touches a
2n x (n+1) array every panel step, which is memory-bound). E2-slim does the
same math on a preallocated frontier without the full-array scatter.

Finally we measure the parallelism actually available: batched TTMQR over k
independent tiles, to see whether a q = 3-7 tile grid can fill an A100.
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
from jax.scipy.linalg import solve_triangular

sys.path.insert(0, ".")
from gpu_bench import iter_dense, iter_struct, make_problem, prep_qr, timeit


# ---------------- real tile primitives --------------------------------------
@jit
def ttqrt(Rii, Rji):
    """QR of a stacked pair of b x b triangles -> the reflectors + new triangle.

    This is the TT ("triangle on top of triangle") kernel: the operation a tiled
    algorithm performs at each (i, j) elimination.
    """
    A = jnp.vstack([Rii, Rji])
    h, taus = jnp.linalg.qr(A, mode="raw")
    return h.swapaxes(-1, -2), taus


@jit
def ttmqr(packed, taus, Cii, Cji):
    """Apply the TTQRT reflectors to a trailing tile pair (compact-WY)."""
    M2, k = packed.shape
    V = jnp.where(
        jnp.tril(jnp.ones((M2, k), bool), -1), packed, jnp.eye(M2, k, dtype=packed.dtype)
    )
    live = taus != 0
    V = V * live
    T_inv = V.T @ V - jnp.diag(1.0 / jnp.where(live, taus, 1.0))
    C = jnp.vstack([Cii, Cji])
    out = C - V @ solve_triangular(T_inv, V.T @ C, lower=True)
    return out[: Cii.shape[0]], out[Cii.shape[0] :]


@functools.partial(jit, static_argnames=("nb",))
def ttmqr_batched(packed, taus, Cii, Cji, nb):
    """nb independent TTMQRs fused into one batched call (best case for tiling)."""
    return vmap(ttmqr, in_axes=(None, None, 0, 0))(packed, taus, Cii, Cji)


def tile_counts(n, b):
    q = n // b
    ttqrt_n = sum(q - i for i in range(q))
    ttmqr_n = sum(sum(q - 1 - j for j in range(i, q)) for i in range(q))
    return q, ttqrt_n, ttmqr_n


# ---------------- E2 without the full-array scatter -------------------------
def _apply_QT_wy(packed, taus, C):
    M, k = packed.shape
    V = jnp.where(
        jnp.tril(jnp.ones((M, k), bool), -1), packed, jnp.eye(M, k, dtype=packed.dtype)
    )
    live = taus != 0
    V = V * live
    T_inv = V.T @ V - jnp.diag(1.0 / jnp.where(live, taus, 1.0))
    with jax.default_matmul_precision("highest"):
        return C - V @ solve_triangular(T_inv, V.T @ C, lower=True)


@functools.partial(jit, static_argnames=("block",))
def iter_struct_slim(R, z, alpha, block=256):
    """Same structured math as E2, but the frontier is carried as a small array
    instead of scattering into a 2n x (n+1) work matrix each panel step."""
    n = R.shape[1]
    k_ = R.shape[0]
    top = jnp.zeros((n, n)).at[:k_, :].set(R)
    rhs = jnp.zeros(n).at[:k_].set(z)
    sa = jnp.sqrt(alpha)
    B = sa * jnp.eye(n)  # bottom block
    bz = jnp.zeros(n)
    out_rows = []
    for kb in range((n + block - 1) // block):
        c0, c1 = kb * block, min(kb * block + block, n)
        bk = c1 - c0
        # active frontier: this panel of `top` plus the accumulated B rows 0..c1
        Sub = jnp.concatenate(
            [
                jnp.concatenate([top[c0:c1, c0:], rhs[c0:c1, None]], axis=1),
                jnp.concatenate([B[:c1, c0:], bz[:c1, None]], axis=1),
            ],
            axis=0,
        )
        h, taus = jnp.linalg.qr(Sub[:, :bk], mode="raw")
        Sub = _apply_QT_wy(h.swapaxes(-1, -2), taus, Sub)
        out_rows.append(Sub[:bk])
        B = B.at[:c1, c0:].set(Sub[bk:, :-1])
        bz = bz.at[:c1].set(Sub[bk:, -1])
    Rt = jnp.zeros((n, n))
    zt = jnp.zeros(n)
    off = 0
    for kb, rows in enumerate(out_rows):
        c0 = kb * block
        bk = rows.shape[0]
        Rt = Rt.at[c0 : c0 + bk, c0:].set(rows[:, :-1])
        zt = zt.at[c0 : c0 + bk].set(rows[:, -1])
    return jnp.triu(Rt), zt


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    assert dev.platform == "gpu", f"need a GPU, got {meta}"
    out = dict(meta=meta)

    rng = np.random.default_rng(0)

    # ---- 1. cost of the REAL tile primitives -------------------------------
    out["prims"] = []
    print("\nreal tile-primitive cost (single op, and marginal in a chain)")
    for b in [64, 128, 256, 512]:
        Rii = jnp.asarray(np.triu(rng.standard_normal((b, b))))
        Rji = jnp.asarray(np.triu(rng.standard_normal((b, b))))
        t_qrt = timeit(ttqrt, Rii, Rji, reps=5)
        packed, taus = ttqrt(Rii, Rji)
        Cii = jnp.asarray(rng.standard_normal((b, b)))
        Cji = jnp.asarray(rng.standard_normal((b, b)))
        t_mqr = timeit(ttmqr, packed, taus, Cii, Cji, reps=5)
        # batched over q independent trailing tiles
        bat = {}
        for nb in [2, 4, 8, 16]:
            CiiB = jnp.asarray(rng.standard_normal((nb, b, b)))
            CjiB = jnp.asarray(rng.standard_normal((nb, b, b)))
            bat[nb] = timeit(ttmqr_batched, packed, taus, CiiB, CjiB, nb, reps=5)
        out["prims"].append(dict(b=b, t_ttqrt=t_qrt, t_ttmqr=t_mqr,
                                 batched={str(k): v for k, v in bat.items()}))
        print(
            f"  b={b:4d}  TTQRT={t_qrt*1e6:8.1f} us  TTMQR={t_mqr*1e6:8.1f} us  "
            f"batched TTMQR x8={bat[8]*1e6:8.1f} us "
            f"(={bat[8]/t_mqr:4.2f}x one op -> batching gain {8*t_mqr/bat[8]:4.2f}x)",
            flush=True,
        )

    # ---- 2. tiled estimate from real primitives vs measured E2 -------------
    prim = {d["b"]: d for d in out["prims"]}
    out["compare"] = []
    print("\ntiled estimate (real primitives) vs measured column-blocked, m/n=3")
    for n in [1000, 2000, 4000]:
        J, f = make_problem(3 * n, n)
        z, R, _ = prep_qr(J, f)
        zp = jnp.concatenate([z, jnp.zeros(n)])
        t_dense = timeit(iter_dense, R, zp, 1.0, reps=3)
        e2 = {bb: timeit(iter_struct, R, z, 1.0, bb, reps=3) for bb in (64, 128, 256)}
        t_e2 = min(e2.values())
        slim = {}
        for bb in (128, 256):
            try:
                Rt, zt = iter_struct_slim(R, z, 1.0, bb)
                G = R.T @ R + 1.0 * jnp.eye(n)
                err = float(jnp.linalg.norm(Rt.T @ Rt - G) / jnp.linalg.norm(G))
                slim[bb] = (timeit(iter_struct_slim, R, z, 1.0, bb, reps=3), err)
            except Exception as e:
                slim[bb] = (float("nan"), float("nan"))
        for b in [128, 256]:
            q, nq, nm = tile_counts(n, b)
            # sequential: every tile op its own kernel
            seq = nq * prim[b]["t_ttqrt"] + nm * prim[b]["t_ttmqr"]
            # optimistic: each TTQRT's trailing TTMQRs batched into one call
            nb_key = min([2, 4, 8, 16], key=lambda k: abs(k - max(1, q - 1)))
            per_batched = prim[b]["batched"][str(nb_key)]
            opt = nq * (prim[b]["t_ttqrt"] + per_batched)
            rec = dict(n=n, b=b, q=q, n_ttqrt=nq, n_ttmqr=nm, t_dense=t_dense,
                       t_e2=t_e2, t_e2_by_block={str(k): v for k, v in e2.items()},
                       t_slim={str(k): v[0] for k, v in slim.items()},
                       slim_err={str(k): v[1] for k, v in slim.items()},
                       tiled_seq=seq, tiled_opt=opt)
            out["compare"].append(rec)
            print(
                f"  n={n:5d} b={b:4d} q={q:2d} | dense={t_dense*1e3:7.2f} "
                f"E2={t_e2*1e3:7.2f} slim={slim.get(b,(float('nan'),))[0]*1e3:7.2f} "
                f"| tiled seq={seq*1e3:8.2f} opt={opt*1e3:7.2f} ms "
                f"-> opt/E2={opt/t_e2:5.2f}x",
                flush=True,
            )
            json.dump(out, open("out/tile_real.json", "w"), indent=2)
