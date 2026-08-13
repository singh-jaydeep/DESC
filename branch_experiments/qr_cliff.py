"""Locate the batched-QR cliff, and test whether it can be worked around.

Measured on A100: a batched panel QR of shape (B, 851, 128) costs 1.39 ms at B=1,
2.35 at B=2, 4.40 at B=4, then 56.2 ms at B=8 -- a 12.8x jump for a 2x batch.
A batched GEMM over the same shapes costs 1.56x for 8x the work. So batching is
free for GEMM and catastrophic for QR, and the whole batched-alpha result is
explained by that one primitive.

Tests, in order of practical value:

  T1  MAP THE CLIFF. Sweep B = 1..16 and several panel shapes. Is the cliff at a
      fixed B, a fixed total element count, or a fixed shape ratio?

  T2  CHUNKING. If the cliff is at B >= 8, then B=8 issued as two batched calls of
      B=4 should cost ~2 x 4.4 = 8.8 ms instead of 56 ms. That is a trivial fix,
      and it decides whether "batching is a loss" is a real limit or an artifact
      of one library heuristic.

  T3  IS IT geqrf OR THE RAW MODE? Compare mode="raw" (what we use) against
      mode="reduced" and against lax.linalg.qr directly, all batched.

  T4  GEMM-ONLY PANEL FACTORIZATION. If cuSOLVER's batched QR is the problem,
      factor the panel with a recursive (divide-and-conquer) Householder scheme
      whose leaves are small QRs and whose combine steps are GEMMs. Tests whether
      a QR built from batch-friendly primitives beats the library's batched QR.
"""

import functools
import json
import os
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve_triangular


def timeit(fn, *a, reps=5, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@jit
def qr_raw(A):
    return jnp.linalg.qr(A, mode="raw")


@jit
def qr_reduced(A):
    return jnp.linalg.qr(A, mode="reduced")


@functools.partial(jit, static_argnames=("nchunk",))
def qr_chunked(A, nchunk=2):
    """Batched QR issued as `nchunk` separate batched calls."""
    B = A.shape[0]
    per = B // nchunk
    outs = [jnp.linalg.qr(A[i * per : (i + 1) * per], mode="raw") for i in range(nchunk)]
    return (
        jnp.concatenate([o[0] for o in outs], axis=0),
        jnp.concatenate([o[1] for o in outs], axis=0),
    )


@jit
def gemm_ref(A, Bm):
    return jnp.einsum("bmk,bkw->bmw", A, Bm)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        assert dev.platform == "gpu", f"need a GPU, got {meta}"
    rng = np.random.default_rng(0)
    out = dict(meta=meta, cliff=[], chunk=[], modes=[])

    # ---- T1: map the cliff --------------------------------------------------
    print("\nT1  batched QR cost vs batch size, several panel shapes")
    shapes = [(851, 128), (256, 128), (1128, 128), (851, 64), (2000, 256)]
    for (M_, k_) in shapes:
        base = None
        row = dict(M=M_, k=k_, per_B={})
        line = []
        for B in [1, 2, 3, 4, 6, 8, 12, 16]:
            A = jnp.asarray(rng.standard_normal((B, M_, k_)))
            try:
                t = timeit(qr_raw, A)
                if base is None:
                    base = t
                row["per_B"][str(B)] = dict(t=t, scaling=t / base, per_item=t / B)
                line.append(f"B={B}:{t*1e3:.2f}({t/base:.1f}x)")
            except Exception as e:
                row["per_B"][str(B)] = dict(error=str(e)[:80])
                line.append(f"B={B}:ERR")
        out["cliff"].append(row)
        print(f"  ({M_:5d},{k_:4d}): " + "  ".join(line), flush=True)

    # ---- T2: does chunking dodge it? ---------------------------------------
    print("\nT2  B=8 and B=16 as one batched call vs as chunks of 4")
    for (M_, k_) in [(851, 128), (1128, 128)]:
        for B in [8, 16]:
            A = jnp.asarray(rng.standard_normal((B, M_, k_)))
            t1 = timeit(qr_raw, A)
            res = dict(M=M_, k=k_, B=B, t_single=t1, chunks={})
            msg = f"  ({M_},{k_}) B={B}: one call {t1*1e3:8.2f} ms"
            for nc in [2, 4, B]:
                if B % nc:
                    continue
                try:
                    t = timeit(qr_chunked, A, nchunk=nc)
                    res["chunks"][str(nc)] = dict(t=t, gain=t1 / t, per_chunk=B // nc)
                    msg += f" | {nc} chunks of {B//nc}: {t*1e3:7.2f} ms ({t1/t:5.2f}x)"
                except Exception as e:
                    res["chunks"][str(nc)] = dict(error=str(e)[:80])
                    msg += f" | {nc} chunks: ERR"
            out["chunk"].append(res)
            print(msg, flush=True)

    # ---- T3: raw vs reduced -------------------------------------------------
    print("\nT3  mode='raw' vs mode='reduced' (batched)")
    for B in [1, 4, 8, 16]:
        A = jnp.asarray(rng.standard_normal((B, 851, 128)))
        t_raw = timeit(qr_raw, A)
        try:
            t_red = timeit(qr_reduced, A)
        except Exception:
            t_red = float("nan")
        out["modes"].append(dict(B=B, t_raw=t_raw, t_reduced=t_red))
        print(f"  B={B:2d}: raw {t_raw*1e3:8.2f} ms | reduced {t_red*1e3:8.2f} ms", flush=True)

    json.dump(out, open("out/qr_cliff.json", "w"), indent=2)
