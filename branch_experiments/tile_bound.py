"""Is a TILED (TT-kernel) QR of [R; sqrt(a) I] worth it at DESC's problem sizes?

The claim under test: at n ~ 1e3 with GPU-appropriate tile sizes b = 128-256,
the tile grid is only q = n/b = 4-8 across, so tiling cannot beat the simple
column-blocked scheme (variant E2).

Rather than implement tiled TT, we measure a LOWER BOUND on any tiled
implementation and compare it to E2's measured time. The bound is constructed
from the algorithm's own structure:

  For bottom row-block i (i = 0..q-1) eliminate against column-blocks j = i..q-1:
    - one TTQRT at (i,j)                          -> q(q+1)/2 total
    - TTMQR on trailing blocks l = j+1..q-1       -> ~q^3/6 total
  Every tile op is a b x b kernel.

Two bounds are measured:
  UNBATCHED   : all (TTQRT + TTMQR) ops issued sequentially. Cost >= n_ops x
                (time of ONE dependent b x b matmul) -- and a real TTQRT/TTMQR
                is several matmuls, so this is a strict lower bound.
  BATCHED     : optimistic. The TTMQRs following each TTQRT are fused into one
                batched (vmapped) op, so the critical path is the q(q+1)/2
                TTQRTs, each followed by one batched call.

If even the optimistic bound exceeds E2's measured runtime, tiling cannot win
at this size and the question is settled without writing the kernel.

Also measured: fp64 GEMM throughput vs matrix size, which shows directly
whether a b x b tile can reach a useful fraction of peak.
"""

import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap

sys.path.insert(0, ".")
from gpu_bench import iter_dense, iter_struct, make_problem, prep_qr, timeit


def tile_op_counts(n, b):
    """Tile-op counts for the structured elimination on a q x q tile grid."""
    q = n // b
    ttqrt = sum(q - i for i in range(q))
    ttmqr = sum(sum(q - 1 - j for j in range(i, q)) for i in range(q))
    # critical path if every TTQRT's trailing TTMQRs are batched into one call
    crit = ttqrt * 2
    return dict(q=q, ttqrt=ttqrt, ttmqr=ttmqr, total_ops=ttqrt + ttmqr, crit_path=crit)


def chain_cost(b, K, reps=3):
    """Median wall time of K DEPENDENT b x b matmuls (one kernel each)."""

    @jit
    def chain(A, X):
        for _ in range(K):
            X = A @ X
        return X

    A = jnp.asarray(np.random.default_rng(0).standard_normal((b, b)))
    X = jnp.eye(b)
    return timeit(chain, A, X, reps=reps)


def batched_chain_cost(b, K, batch, reps=3):
    """K dependent steps, each a BATCHED (vmapped) b x b matmul over `batch`."""

    @jit
    def chain(A, X):
        f = vmap(lambda a, x: a @ x)
        for _ in range(K):
            X = f(A, X)
        return X

    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((batch, b, b)))
    X = jnp.asarray(np.broadcast_to(np.eye(b), (batch, b, b)).copy())
    return timeit(chain, A, X, reps=reps)


def gemm_gflops(s, reps=3):
    @jit
    def mm(A, B):
        return A @ B

    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((s, s)))
    B = jnp.asarray(rng.standard_normal((s, s)))
    t = timeit(mm, A, B, reps=reps)
    return t, 2 * s**3 / t / 1e9


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    assert dev.platform == "gpu", f"need a GPU, got {meta}"

    out = dict(meta=meta)

    # ---- 1. fp64 GEMM throughput vs size: can a b x b tile reach peak? ------
    out["gemm"] = []
    print("\nfp64 GEMM throughput")
    for s in [64, 128, 256, 512, 1024, 2048, 4096]:
        t, gf = gemm_gflops(s)
        out["gemm"].append(dict(s=s, t=t, gflops=gf))
        print(f"  {s:5d}x{s:<5d} {t*1e6:10.1f} us {gf:9.1f} GFLOP/s", flush=True)

    # ---- 2. per-op floor: cost of ONE dependent tile-sized kernel ----------
    out["chain"] = []
    print("\ndependent-chain cost (per-op floor for a tile op)")
    for b in [128, 256]:
        t1 = chain_cost(b, 1)
        t50 = chain_cost(b, 50)
        per_op = (t50 - t1) / 49
        out["chain"].append(dict(b=b, t1=t1, t50=t50, per_op=per_op))
        print(
            f"  b={b:4d}  1 op={t1*1e6:8.1f} us  50 ops={t50*1e6:9.1f} us "
            f"-> per-op floor {per_op*1e6:7.2f} us",
            flush=True,
        )

    # ---- 3. tiled lower bounds vs the MEASURED column-blocked E2 -----------
    out["bounds"] = []
    print("\ntiled lower bound vs measured E2 / dense  (m/n = 3)")
    per_op = {d["b"]: d["per_op"] for d in out["chain"]}
    for n in [1000, 2000, 4000]:
        m = 3 * n
        J, f = make_problem(m, n)
        z, R, _ = prep_qr(J, f)
        zp = jnp.concatenate([z, jnp.zeros(n)])
        t_dense = timeit(iter_dense, R, zp, 1.0)
        t_e2 = min(timeit(iter_struct, R, z, 1.0, bb) for bb in (64, 128, 256))
        for b in [128, 256]:
            c = tile_op_counts(n, b)
            lb_unbatched = c["total_ops"] * per_op[b]
            # optimistic: critical path of tile ops, TTMQRs batched
            t_batched_step = batched_chain_cost(b, 20, max(1, c["q"])) / 20
            lb_batched = c["ttqrt"] * per_op[b] + c["ttqrt"] * t_batched_step
            rec = dict(
                n=n, b=b, **c, t_dense=t_dense, t_e2=t_e2,
                lb_unbatched=lb_unbatched, lb_batched=lb_batched,
            )
            out["bounds"].append(rec)
            print(
                f"  n={n:5d} b={b:4d} q={c['q']:2d} ops={c['total_ops']:5d} "
                f"crit={c['ttqrt']:4d} | dense={t_dense*1e3:7.2f} E2={t_e2*1e3:7.2f} "
                f"| tiled LB unbatched={lb_unbatched*1e3:8.2f} batched={lb_batched*1e3:7.2f} ms"
                f"  -> batched LB/E2 = {lb_batched/t_e2:5.2f}x",
                flush=True,
            )
            json.dump(out, open("out/tile_bound.json", "w"), indent=2)
