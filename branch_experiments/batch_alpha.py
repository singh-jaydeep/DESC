"""If the structured QR is latency-bound, two other routes should beat it.

The variant sweep showed all four structured implementations within 2% of each
other at 1.2-1.45x over dense, while a Cholesky of R'R+alpha*I -- HALF the flops
of the structured minimum -- runs 4.5-9.0x faster than dense. That says the panel
loop's n/b sequential steps, not arithmetic, set the cost. Two consequences worth
testing:

  (A) BATCH THE ALPHAS. The alpha loop is sequential because each alpha depends
      on the previous phi. But the Hebden iteration is a 1-D root-find whose
      iterates we can PREDICT: bracket alpha in [0, ||R'z||/Delta] and evaluate
      several alphas at once with vmap, then refine. Same total factorizations,
      but issued as ONE batched kernel per round instead of k sequential ones.
      Measures the batched cost of B alphas against B sequential structured
      calls, to see how much of the per-call cost is launch latency.

  (B) SEMI-NORMAL EQUATIONS WITH ONE CHOLESKY-LIKE STRUCTURE, DONE SAFELY.
      Cholesky of the Gram matrix is fast but squares kappa(J) -- we measured it
      failing at kappa >= 1e10, and real DESC R has kappa up to 4e12, so it is
      out. But the SPEED gap (4.5-9x vs 1.4x) is large enough that a middle route
      is worth measuring: form G = R'R + alpha*I ONCE per alpha as a GEMM, take
      its Cholesky, and repair the accuracy loss with one step of iterative
      refinement in the original R (a matvec, not a factorization). Tests whether
      refined-Cholesky recovers dense-QR accuracy at Cholesky speed.
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
from jax.scipy.linalg import cho_factor, cho_solve, qr_multiply, solve_triangular

sys.path.insert(0, ".")
from opt_struct import v0_committed, v2_tpqrt


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


# ---------------------------------------------------------------- (A) batching
@functools.partial(jit, static_argnames=("block",))
def struct_one(R, z, alpha, block=128):
    return v2_tpqrt(R, z, alpha, block=block)


@functools.partial(jit, static_argnames=("block", "B"))
def struct_batched(R, z, alphas, block=128, B=4):
    """Factorize B alphas in ONE batched call."""
    return vmap(lambda a: v2_tpqrt(R, z, a, block=block))(alphas)


@jit
def dense_one(R, zp, alpha):
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
    return qr_multiply(A, zp, mode="right")


@functools.partial(jit, static_argnames=("B",))
def dense_batched(R, zp, alphas, B=4):
    n = R.shape[1]

    def one(a):
        A = jnp.vstack([R, jnp.sqrt(a) * jnp.eye(n)])
        return qr_multiply(A, zp, mode="right")

    return vmap(one)(alphas)


# --------------------------------------------- (B) Cholesky + refinement
@jit
def chol_step(R, z, alpha):
    """p solving (R'R + alpha I) p = -R'z via Cholesky. Squares kappa."""
    n = R.shape[1]
    G = R.T @ R + alpha * jnp.eye(n)
    c, low = cho_factor(G, lower=False)
    return cho_solve((c, low), -(R.T @ z)), (c, low)


@functools.partial(jit, static_argnames=("n_ref",))
def chol_refined(R, z, alpha, n_ref=2):
    """Cholesky + iterative refinement, residual formed in the ORIGINAL R.

    The residual r = -(R'z) - (R'R + alpha I)p is computed with matvecs against
    R (never forming G), so the refinement sees the true operator conditioning
    rather than the squared one. Each round costs 2 matvecs + 1 triangular solve
    pair -- O(n^2), negligible against the O(n^3) factorization.
    """
    n = R.shape[1]
    G = R.T @ R + alpha * jnp.eye(n)
    c, low = cho_factor(G, lower=False)
    b = -(R.T @ z)
    p = cho_solve((c, low), b)
    for _ in range(n_ref):
        r = b - (R.T @ (R @ p) + alpha * p)
        p = p + cho_solve((c, low), r)
    return p


@jit
def dense_qr_step(R, zp, alpha):
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
    Qtz, Rtil = qr_multiply(A, zp, mode="right")
    dr = jnp.diag(Rtil)
    denom = jnp.where(dr == 0, 1, dr)
    dri = jnp.where(dr == 0, 0, 1 / denom)
    Rs = Rtil * dri[:, None]
    return solve_triangular(Rs, dri * (-Qtz), unit_diagonal=True, lower=False)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        assert dev.platform == "gpu", f"need a GPU, got {meta}"
    out = dict(meta=meta, batching=[], accuracy=[])

    sizes = json.loads(os.environ.get("BA_SIZES", "[327, 491, 723, 1000]"))

    # ---- (A) how much of the per-call cost is launch latency? --------------
    print("\n(A) batching alphas: B factorizations in one call vs B sequential")
    for n in sizes:
        J, f = make_problem(3 * n, n)
        z, R = qr_multiply(J, f, mode="right")
        zp = jnp.concatenate([z, jnp.zeros(n)])
        t_s1 = timeit(struct_one, R, z, 1.0, block=128)
        t_d1 = timeit(dense_one, R, zp, 1.0)
        rec = dict(n=n, struct_1=t_s1, dense_1=t_d1, batched={})
        for B in [2, 4, 8]:
            al = jnp.asarray(np.logspace(-6, 0, B))
            try:
                t_sB = timeit(struct_batched, R, z, al, block=128, B=B)
                t_dB = timeit(dense_batched, R, zp, al, B=B)
                rec["batched"][str(B)] = dict(
                    struct_B=t_sB, dense_B=t_dB,
                    struct_per=t_sB / B, dense_per=t_dB / B,
                    struct_gain=(B * t_s1) / t_sB, dense_gain=(B * t_d1) / t_dB,
                )
                print(
                    f"  n={n:5d} B={B}: structured {t_sB*1e3:7.2f} ms "
                    f"({t_sB/B*1e3:6.2f}/alpha vs {t_s1*1e3:6.2f} sequential "
                    f"-> {(B*t_s1)/t_sB:4.2f}x) | dense {t_dB*1e3:7.2f} ms "
                    f"({t_dB/B*1e3:6.2f}/alpha vs {t_d1*1e3:6.2f} "
                    f"-> {(B*t_d1)/t_dB:4.2f}x)",
                    flush=True,
                )
            except Exception as e:
                rec["batched"][str(B)] = dict(error=f"{type(e).__name__}: {e}")
                print(f"  n={n:5d} B={B}: FAILED {type(e).__name__}: {e}", flush=True)
        out["batching"].append(rec)
        json.dump(out, open("out/batch_alpha.json", "w"), indent=2)

    # ---- (B) does refinement recover QR accuracy at Cholesky speed? --------
    print("\n(B) Cholesky + iterative refinement vs dense QR: accuracy and speed")
    for n in [491, 723]:
        for cond_exp in [6.0, 9.0, 12.0]:
            J, f = make_problem(3 * n, n, cond_exp=cond_exp)
            z, R = qr_multiply(J, f, mode="right")
            zp = jnp.concatenate([z, jnp.zeros(n)])
            for alpha in [1e-8, 1e-3]:
                p_qr = dense_qr_step(R, zp, alpha)
                p_ch, _ = chol_step(R, z, alpha)
                p_rf = chol_refined(R, z, alpha, n_ref=2)
                b = -(R.T @ z)

                def kkt(p):
                    return float(
                        jnp.linalg.norm(R.T @ (R @ p) + alpha * p - b)
                        / max(float(jnp.linalg.norm(b)), 1e-300)
                    )

                t_qr = timeit(dense_qr_step, R, zp, alpha)
                t_ch = timeit(chol_step, R, z, alpha)
                t_rf = timeit(chol_refined, R, z, alpha, n_ref=2)
                t_st = timeit(struct_one, R, z, alpha, block=128)
                rec = dict(n=n, cond_exp=cond_exp, alpha=alpha,
                           kkt_qr=kkt(p_qr), kkt_chol=kkt(p_ch), kkt_refined=kkt(p_rf),
                           t_qr=t_qr, t_chol=t_ch, t_refined=t_rf, t_struct=t_st,
                           speedup_refined_vs_qr=t_qr / t_rf,
                           speedup_struct_vs_qr=t_qr / t_st)
                out["accuracy"].append(rec)
                print(
                    f"  n={n:4d} kappa~1e{cond_exp:.0f} alpha={alpha:.0e} | KKT: "
                    f"qr={kkt(p_qr):.2e} chol={kkt(p_ch):.2e} refined={kkt(p_rf):.2e} "
                    f"| ms: qr={t_qr*1e3:6.2f} struct={t_st*1e3:6.2f} "
                    f"refined={t_rf*1e3:6.2f} -> refined is {t_qr/t_rf:4.2f}x vs qr",
                    flush=True,
                )
                json.dump(out, open("out/batch_alpha.json", "w"), indent=2)
