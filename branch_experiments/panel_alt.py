"""Is the structured QR near-optimal in arithmetic intensity on A100, or is there headroom?

WHERE THIS STARTS
-----------------
Roofline analysis of the committed sweep (desc/optimize/tr_subproblems.py::
structured_retriangularize) says it is neither bandwidth- nor flop-bound:
arithmetic intensity is 15-60 flop/byte against A100 ridge points of 4.8-6.2
(HBM) and 1.9 (L2) -- compute-side by 5-6x. Yet it achieves 0.45% of fp64 peak,
34-353x off roofline. So the binding constraint is occupancy/latency, and there
is in principle a large amount of headroom.

Primitive timing localizes it. At panel shape (851,128) on A100-80GB:
    panel QR (geqrf)  1.390 ms   69.5% of the panel, 0.196% of peak
    GEMM (WY apply)   0.325 ms   16.2% of the panel, 1.161% of peak
    gather/scatter    0.286 ms   14.3% of the panel
geqrf on a narrow panel is internally a sequential loop over its b columns of
rank-1 (level-2 BLAS) work, so the n/b outer panel steps each nest b more
sequential steps.

WHAT IS RULED OUT, AND WHY (measured separately, not here)
----------------------------------------------------------
The obvious fix -- swap geqrf for a GEMM-only CholeskyQR2 -- is impossible, and
not for the reason one would guess. It is NOT a conditioning problem: the panels
are extremely well conditioned because their leading columns are already
orthogonalized and the sqrt(alpha)*I rows floor the smallest singular value.
Measured worst-case panel kappa over 60 (n, kappa(R), alpha, rank-deficiency)
combinations spanning the real DESC envelope is 5.3e7, and only 2.2e3 for
kappa(R) <= 3.7e12 with alpha > 0 -- comfortably inside CholeskyQR2's ~1e8 limit.

The obstruction is structural. The panel step must apply a full-row-space
orthogonal transform: the (rows - b) rows BELOW the panel's R factor are not
zero, they are the fill-in that later panels consume. Verified numerically:
those rows carry Frobenius norm 3.23 against 6.11 for the whole panel slice, and
discarding them breaks the Gram identity by O(1). CholeskyQR-family methods
return an orthonormal COLUMN basis (rows x b), not a full transform (rows x
rows), so they cannot produce the complement. Householder reflectors give the
full transform implicitly and in compact form -- the sequential column loop is
precisely the price of that.

WHAT THIS SCRIPT MEASURES
-------------------------
Three things, each of which either bounds the available gain or tests a
candidate that respects the full-transform requirement.

  1. CEILING. If the panel QR were free, what would the sweep cost? Measured as
     GEMM + scatter only, per real panel shape. This says whether ANY panel-level
     improvement is worth pursuing, before implementing one.

  2. TSQR (communication-avoiding QR) for the panel. Split the panel's rows into
     c >= 2 chunks of at least b rows, factor each INDEPENDENTLY (so they batch
     and fill the device), then merge pairwise up a reduction tree. This is the
     standard method for tall-skinny panels and it does yield a valid full
     transform. Risk: the leaf QRs go through the same batched geqrf that showed
     a 5x per-item cliff at batch 8; here c <= 6, below the measured cliff.

  3. PANEL SHAPE SENSITIVITY. geqrf cost over a (rows x b) grid. The committed
     sweep uses a uniform block width, so its panels are (2b x b) growing to
     (n+b x b) -- aspect ratios 2:1 through 6.6:1 at n=723. If geqrf's efficiency
     depends strongly on aspect ratio, a VARIABLE block schedule (narrow early,
     wide late, or vice versa) is a cheap win that needs no new kernel.
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
from jax.scipy.linalg import qr_multiply

A100_FP64 = 9.7e12


def timeit(fn, *a, reps=5, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# --- the three pieces of one panel, timed separately ------------------------
@jit
def piece_geqrf(A):
    h, taus = jnp.linalg.qr(A, mode="raw")
    return h, taus


@jit
def piece_wy_gemm(V, T, C):
    """The WY trailing update: two GEMMs plus a small triangular solve."""
    W = V.T @ C
    W = T @ W
    return C - V @ W


@jit
def piece_scatter(M, idx, Sub, c0):
    return jax.lax.dynamic_update_slice(M, Sub, (0, c0)) if idx is None else M.at[idx].set(
        jnp.zeros_like(M[idx])
    )


# --- TSQR: batched leaf QRs + pairwise merge tree ---------------------------
@functools.partial(jit, static_argnames=("nchunk",))
def tsqr_panel(A, nchunk=4):
    """Communication-avoiding QR of a tall-skinny panel.

    Rows are split into ``nchunk`` chunks, each factored independently (one
    batched call -> the leaves fill the device), then merged pairwise up a
    binary tree. Returns the R factor; the implicit Q is the composition of the
    leaf and node transforms, which is a valid FULL orthogonal transform of the
    row space (this is why TSQR is admissible where CholeskyQR2 is not).
    """
    rows, b = A.shape
    per = -(-rows // nchunk)  # ceil: PAD, never truncate -- dropping the
    # remainder rows silently corrupts R (measured: 2.8e-3 to 2.0e-1 relative
    # error at (851,128), which is a dropped row, not a TSQR property).
    A = jnp.pad(A, ((0, per * nchunk - rows), (0, 0)))
    leaves = A.reshape(nchunk, per, b)
    Rs = jnp.linalg.qr(leaves, mode="r")  # batched over the leading axis
    # Pairwise merge. nchunk MUST be a power of two: with an odd count the
    # `k//2` pairing silently drops the unpaired node (measured: 1.9e-1
    # relative error at nchunk=6). A general tree would carry the odd node
    # forward; powers of two are the standard TSQR shape and suffice here.
    assert nchunk & (nchunk - 1) == 0, "nchunk must be a power of two"
    cur = Rs
    k = nchunk
    while k > 1:
        pairs = cur.reshape(k // 2, 2 * b, b)
        cur = jnp.linalg.qr(pairs, mode="r")
        k = k // 2
    return cur[0]


@jit
def ref_panel_R(A):
    return jnp.linalg.qr(A, mode="r")


def make_problem(n, ratio=3.0, cond_exp=12.0, seed=0):
    rng = np.random.default_rng(seed)
    m = int(ratio * n)
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    J = jnp.asarray((U * np.logspace(0, -cond_exp, n)) @ V.T)
    f = jnp.asarray(rng.standard_normal(m))
    return J, f


def panel_shapes(n, b):
    """Exact (rows, bk, trailing_width) of each panel in the committed sweep."""
    out = []
    for kb in range((n + b - 1) // b):
        c0 = kb * b
        c1 = min(c0 + b, n)
        out.append((b + c1, c1 - c0, (n + 1) - c1))
    return out


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    dev = jax.devices()[0]
    meta = dict(platform=dev.platform, device=str(dev.device_kind), jax=jax.__version__)
    print("DEVICE:", meta, flush=True)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        assert dev.platform == "gpu", f"expected GPU, got {meta}"

    out = dict(meta=meta, ceiling=[], tsqr=[], shape_grid=[])
    rng = np.random.default_rng(0)

    # ---- 1. CEILING: cost of a panel with the geqrf removed ---------------
    print("\n=== 1. ceiling: what a panel costs if the panel QR were free ===", flush=True)
    for n in [327, 491, 723, 1000]:
        for b in [128]:
            tot_qr = tot_gemm = 0.0
            for rows, bk, wt in panel_shapes(n, b):
                A = jnp.asarray(rng.standard_normal((rows, bk)))
                C = jnp.asarray(rng.standard_normal((rows, max(wt, 1))))
                V = jnp.asarray(rng.standard_normal((rows, bk)))
                T = jnp.asarray(rng.standard_normal((bk, bk)))
                tot_qr += timeit(piece_geqrf, A)
                tot_gemm += timeit(piece_wy_gemm, V, T, C)
            rec = dict(n=n, b=b, t_qr_total=tot_qr, t_gemm_total=tot_gemm,
                       ceiling_ratio=(tot_qr + tot_gemm) / tot_gemm)
            out["ceiling"].append(rec)
            print(f"  n={n:5d} b={b}: panel-QR total {tot_qr*1e3:7.2f} ms, "
                  f"GEMM total {tot_gemm*1e3:6.2f} ms  ->  removing the QR entirely "
                  f"would be {rec['ceiling_ratio']:.2f}x faster", flush=True)
            json.dump(out, open("out/panel_alt.json", "w"), indent=2)

    # ---- 2. TSQR vs one geqrf, at the real panel shapes -------------------
    print("\n=== 2. TSQR (batched leaves + merge tree) vs a single geqrf ===", flush=True)
    for rows, bk in [(256, 128), (512, 128), (768, 128), (851, 128), (1128, 128)]:
        U, _ = np.linalg.qr(rng.standard_normal((rows, bk)))
        V, _ = np.linalg.qr(rng.standard_normal((bk, bk)))
        A = jnp.asarray((U * np.logspace(0, -6, bk)) @ V.T)
        t_ref = timeit(ref_panel_R, A)
        Rref = ref_panel_R(A)
        rec = dict(rows=rows, bk=bk, t_geqrf=t_ref, chunks={})
        for c in [2, 4, 8]:
            if rows // c < bk:
                continue
            try:
                t = timeit(tsqr_panel, A, nchunk=c)
                Rt = tsqr_panel(A, nchunk=c)
                # compare |R| (sign conventions differ per row)
                err = float(
                    jnp.linalg.norm(jnp.abs(jnp.triu(Rt)) - jnp.abs(jnp.triu(Rref)))
                    / jnp.linalg.norm(Rref)
                )
                rec["chunks"][c] = dict(t=t, gain=t_ref / t, R_relerr=err)
            except Exception as e:
                rec["chunks"][c] = dict(error=f"{type(e).__name__}: {e}"[:110])
        out["tsqr"].append(rec)
        print(f"  ({rows:5d},{bk:3d}) geqrf {t_ref*1e3:7.3f} ms | "
              + " | ".join(
                  f"TSQR c={c} {v['t']*1e3:7.3f} ms x{v['gain']:4.2f} err={v['R_relerr']:.1e}"
                  if "t" in v else f"c={c} FAIL"
                  for c, v in rec["chunks"].items()), flush=True)
        json.dump(out, open("out/panel_alt.json", "w"), indent=2)

    # ---- 3. geqrf cost over a (rows, b) grid ------------------------------
    print("\n=== 3. geqrf cost vs panel shape (is a variable block schedule better?) ===",
          flush=True)
    for b in [32, 64, 128, 256]:
        row_line = []
        for rows in [2 * b, 4 * b, 8 * b, 1128]:
            if rows < b:
                continue
            A = jnp.asarray(rng.standard_normal((rows, b)))
            t = timeit(piece_geqrf, A)
            f = 2 * rows * b**2 - (2 / 3) * b**3
            out["shape_grid"].append(dict(rows=rows, b=b, t=t,
                                         gflops=f / t / 1e9,
                                         pct_peak=f / t / A100_FP64 * 100,
                                         aspect=rows / b))
            row_line.append(f"{rows}x{b}:{t*1e3:.3f}ms/{f/t/1e9:.0f}GF")
        print(f"  b={b:4d}: " + "  ".join(row_line), flush=True)
        json.dump(out, open("out/panel_alt.json", "w"), indent=2)
