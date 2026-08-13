"""Which structured-QR implementation is fastest on GPU at DESC's real sizes?

Flop counting says the committed version (V0) does 1.3-3.9x the structured
minimum depending on block width, and that SMALLER blocks are much closer to the
minimum (b=32 -> 1.1-1.7x) while the earlier timing sweep found b=128 fastest.
That combination means the routine is NOT flop-bound at these sizes -- so the
question is where the wall time actually goes.

Measures, at the four real DESC sizes plus two larger ones:
  - all four variants across block widths 16..512
  - the dense QR reference (master's path)
  - a Cholesky lower bound (n^3/3 flops, the fastest possible dense route,
    numerically unacceptable but a speed floor)
  - achieved fraction of the dense QR's throughput

so we can say whether the remaining gap to the flop bound is recoverable by
better blocking or is a floor set by kernel launch and memory traffic.
"""

import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cho_factor, qr_multiply, solve_triangular

sys.path.insert(0, ".")
from opt_struct import VARIANTS


def make_problem(m, n, seed=0, cond_exp=9.0):
    rng = np.random.default_rng(seed)
    k = min(m, n)
    U, _ = np.linalg.qr(rng.standard_normal((m, k)))
    V, _ = np.linalg.qr(rng.standard_normal((n, k)))
    s = np.logspace(0, -cond_exp, k)
    return jnp.asarray((U * s) @ V.T), jnp.asarray(rng.standard_normal(m))


@jit
def dense_ref(R, zp, alpha):
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
    Qtz, Rtil = qr_multiply(A, zp, mode="right")
    return Rtil, Qtz


@jit
def chol_floor(R, z, alpha):
    """Speed floor: Cholesky of R'R + alpha I (n^3/3 flops). NOT numerically OK."""
    n = R.shape[1]
    G = R.T @ R + alpha * jnp.eye(n)
    c, low = cho_factor(G, lower=False)
    rhs = R.T @ z
    return c, rhs


def timeit(fn, *a, reps=7, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def flops_min(n):
    return sum(4 * (j + 2) * (n - j) for j in range(n))


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        assert dev.platform == "gpu", f"need a GPU, got {meta}"

    sizes = json.loads(os.environ.get("OPT_SIZES", "[327, 491, 561, 723, 1000, 2000]"))
    blocks = json.loads(os.environ.get("OPT_BLOCKS", "[16, 32, 64, 128, 256, 512]"))
    out = dict(meta=meta, rows=[])

    for n in sizes:
        m = 3 * n
        J, f = make_problem(m, n)
        z, R = qr_multiply(J, f, mode="right")
        zp = jnp.concatenate([z, jnp.zeros(n)])
        G = R.T @ R + 1.0 * jnp.eye(n)

        t_dense = timeit(dense_ref, R, zp, 1.0)
        t_chol = timeit(chol_floor, R, z, 1.0)
        fmin = flops_min(n)
        rec = dict(n=n, m=m, t_dense=t_dense, t_chol=t_chol,
                   gflops_dense=(10 * n**3 / 3) / t_dense / 1e9,
                   variants={})
        print(
            f"\nn={n:5d}: dense QR={t_dense*1e3:7.2f} ms "
            f"({(10*n**3/3)/t_dense/1e9:7.1f} GFLOP/s) | "
            f"Cholesky floor={t_chol*1e3:6.2f} ms "
            f"({t_dense/t_chol:5.2f}x faster than dense)",
            flush=True,
        )
        for nm, fn in VARIANTS.items():
            per_b = {}
            for b in blocks:
                if b > n:
                    continue
                try:
                    t = timeit(fn, R, z, 1.0, block=b)
                    Rt, zt = fn(R, z, 1.0, block=b)
                    err = float(jnp.linalg.norm(Rt.T @ Rt - G) / jnp.linalg.norm(G))
                    per_b[str(b)] = dict(t=t, speedup=t_dense / t, gram_err=err,
                                         eff_vs_min=fmin / t / 1e9)
                except Exception as e:
                    per_b[str(b)] = dict(error=f"{type(e).__name__}: {e}")
            ok = {b: v for b, v in per_b.items() if "t" in v}
            if ok:
                bb = min(ok, key=lambda b: ok[b]["t"])
                print(
                    f"  {nm:22} best b={bb:>4}: {ok[bb]['t']*1e3:7.2f} ms  "
                    f"{ok[bb]['speedup']:5.2f}x vs dense  "
                    f"gram_err={ok[bb]['gram_err']:.1e}  | all blocks: "
                    + " ".join(f"{b}:{ok[b]['t']*1e3:.1f}" for b in sorted(ok, key=int)),
                    flush=True,
                )
            rec["variants"][nm] = per_b
        out["rows"].append(rec)
        json.dump(out, open("out/opt_bench.json", "w"), indent=2)
