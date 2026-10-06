"""Controlled Levenberg-Marquardt loop for least-squares problems with bound rows.

All arms share the Jacobian, column scaling, LM damping rule and acceptance test; they
differ in the local model.
  std  : r_i + J_i p, with J_i = A_i on violated rows and 0 otherwise (what lsqtr sees)
  comp : bound rows kept exact as (a_i + A_i p - hi)_+ and (lo - a_i - A_i p)_+ of their
         pre-hinge values; target rows linearized
  compS: comp + structured secant estimate of S = sum r_i Hess(r_i) (NL2SOL-style)
  comp2: comp + exact S from Hessian-vector products (n per iteration)
usage: python ctr.py ARM START_TAG MAXITER OUT_TAG [Y_FILE]
START_TAG.json / START_TAG_allx.npy come from mkstage.py or run.py; Y_FILE (reduced
coordinates, e.g. a previous OUT_y.npy) overrides the start point. Writes OUT_hist.npy,
OUT_y.npy, OUT_x.npy (also every 50 iterations).
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import load_phase, load_run

from desc.batching import jacfwd_chunked

arm, start, maxiter, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
import json  # noqa: E402
import os  # noqa: E402

if os.environ.get("PHASE"):
    obj, proj, y, _ = load_phase(
        start, sys.argv[5], json.loads(os.environ.get("KW", "{}"))
    )
    y = jnp.asarray(y)
else:
    obj, proj, Y = load_run(start, json.loads(os.environ.get("KW", "{}")))
    y = jnp.asarray(np.load(sys.argv[5]) if len(sys.argv) > 5 else Y[-1])

# per-row target / bounds in scaled units
lo, hi, tgt, isb = [], [], [], []
for o in obj.objectives:
    sc = lambda v: np.asarray(o._scale(jnp.ones(o.dim_f) * v, constants=o.constants))
    norm = 1.0 if o._normalize_target else o.normalization
    if o.bounds is not None:
        lo.append(sc(np.asarray(o.bounds[0]) * norm))
        hi.append(sc(np.asarray(o.bounds[1]) * norm))
        tgt.append(np.zeros(o.dim_f))
        isb.append(np.ones(o.dim_f, bool))
    else:
        lo.append(np.full(o.dim_f, -np.inf))
        hi.append(np.full(o.dim_f, np.inf))
        tgt.append(sc(np.asarray(o.target) * norm))
        isb.append(np.zeros(o.dim_f, bool))
lo, hi, tgt, isb = map(np.concatenate, (lo, hi, tgt, isb))

s_fun = jax.jit(lambda y: obj.compute_scaled(proj.recover(y)))
cost_y = lambda y: 0.5 * jnp.sum(proj.compute_scaled_error(y) ** 2)
hvp = jax.jit(lambda y, v: jax.jvp(jax.grad(cost_y), (y,), (v,))[1])


def true_hessian(y):
    """Exact Hessian of the cost from one HVP per column."""
    I = np.eye(y.size)
    H = np.stack([np.asarray(hvp(y, jnp.asarray(I[k]))) for k in range(y.size)], 1)
    return 0.5 * (H + H.T)


A_fun = jax.jit(
    jacfwd_chunked(lambda y: obj.compute_scaled(proj.recover(y)), chunk_size=16)
)


def resid(s):
    """Scaled residuals from pre-hinge values: hinge on bounded rows, s - t else."""
    hinge = np.maximum(0, s - hi) - np.maximum(0, lo - s)
    return np.where(isb, hinge, s - tgt)


def cost_of(s):
    """Half the sum of squared residuals."""
    r = resid(s)
    return 0.5 * r @ r


def check_against_desc(y):
    """Max difference between our residuals and DESC's scaled error."""
    r_desc = np.asarray(proj.compute_scaled_error(y))
    return np.abs(resid(np.asarray(s_fun(y))) - r_desc).max()


def solve_comp(a, B, rT, lam, q0, Sd=None):
    """Semismooth Newton on the convex piecewise-quadratic composite model."""
    T = ~isb
    BT, rTv = B[T], rT[T]
    Bb, ab, lob, hib = B[isb], a[isb], lo[isb], hi[isb]

    def M(q):
        z = ab + Bb @ q
        e = rTv + BT @ q
        sq = 0.5 * q @ Sd @ q if Sd is not None else 0.0
        return (
            sq
            + 0.5 * e @ e
            + 0.5 * np.sum(np.maximum(0, z - hib) ** 2 + np.maximum(0, lob - z) ** 2)
            + 0.5 * lam * q @ q
        )

    HT = BT.T @ BT + (Sd if Sd is not None else 0)
    q = q0.copy()
    for k in range(50):
        z = ab + Bb @ q
        v = np.maximum(0, z - hib) - np.maximum(0, lob - z)
        S = v != 0
        g = (
            BT.T @ (rTv + BT @ q)
            + Bb[S].T @ v[S]
            + lam * q
            + (Sd @ q if Sd is not None else 0)
        )
        if np.linalg.norm(g) < 1e-12 * (1 + np.linalg.norm(q)):
            break
        H = HT + Bb[S].T @ Bb[S] + lam * np.eye(q.size)
        dq = -np.linalg.solve(H, g)
        t, m0 = 1.0, M(q)
        while M(q + t * dq) > m0 + 1e-4 * t * (g @ dq) and t > 1e-10:
            t *= 0.5
        q = q + t * dq
    return q, k + 1


def model_val(a, B, r, q):
    """Composite model value without the damping term."""
    z = a + B @ q
    rb = np.maximum(0, z - hi) - np.maximum(0, lo - z)
    rt = r + B @ q
    m = np.where(isb, rb, rt)
    return 0.5 * m @ m


print(f"arm={arm}  start={start}  n={y.size}  m={lo.size}  hinge rows={isb.sum()}")
print("max |resid - DESC scaled error| at start:", check_against_desc(y))
scale_inv = None
lam, nu = None, 2.0
# compS: structured secant estimate of S = sum r_i Hess(r_i), unscaled y-coordinates
Su, g_prev, step_prev, use_S, nsk = np.zeros((y.size, y.size)), None, None, False, 0
s = np.asarray(s_fun(y))
c = cost_of(s)
hist = []
t0 = time.time()
A = None
for it in range(maxiter + 1):
    if A is None:
        A = np.asarray(A_fun(y))
        r = resid(s)
        act = isb & (r != 0)
        Jstd = np.where((~isb | act)[:, None], A, 0.0)
        g = Jstd.T @ r
        cn = np.linalg.norm(A, axis=0)
        scale_inv = cn if scale_inv is None else np.maximum(scale_inv, cn)
        d = 1 / scale_inv
        B = A * d
        Bstd = Jstd * d
        gopt = np.abs(g * d).max()
        if lam is None:
            lam = 1e-3 * np.max(np.sum(Bstd**2, 0))
        Sd = lam_floor = None
        if arm == "comp2":
            Hs = true_hessian(y)
            Sd = d[:, None] * (Hs - Jstd.T @ Jstd) * d[None, :]
            ev = np.linalg.eigvalsh(Bstd.T @ Bstd + Sd)  # incl. active hinges
            lam_floor = max(0.0, -ev[0]) * 1.01 + 1e-14
        if arm == "compS":
            if step_prev is not None:
                # NL2SOL sized rank-2 update with y# = dg - G+ s (row-order independent)
                yv = g - g_prev
                ysh = yv - Jstd.T @ (Jstd @ step_prev)
                ys = yv @ step_prev
                sSs = step_prev @ Su @ step_prev
                if ys > 0:
                    if sSs != 0:
                        Su *= min(1.0, abs(step_prev @ ysh) / abs(sSs))
                    w = ysh - Su @ step_prev
                    Su = (
                        Su
                        + (np.outer(w, yv) + np.outer(yv, w)) / ys
                        - (w @ step_prev) * np.outer(yv, yv) / ys**2
                    )
                else:
                    nsk += 1
            Sd = d[:, None] * Su * d[None, :]
            ev = np.linalg.eigvalsh(Bstd.T @ Bstd + Sd)
            lam_floor = max(0.0, -ev[0]) * 1.01 + 1e-14
    hist.append((it, c, np.abs(g).max(), gopt, lam, time.time() - t0))
    if it % 50 == 0 and it > 0:  # checkpoint
        np.save(f"{out}_y.npy", np.asarray(y))
        np.save(f"{out}_x.npy", np.asarray(proj.recover(y)))
    if it % 10 == 0 or it == maxiter or arm == "comp2":
        print(
            f"{it:4d} cost={c:.12e} |g|inf={np.abs(g).max():.3e} "
            f"opt(scaled)={gopt:.3e} "
            f"lam={lam:.2e} active_hinges={int(act.sum())} t={time.time()-t0:.0f}s"
            + (
                f" model={'S' if use_S else 'GN'} skips={nsk}" if arm == "compS" else ""
            ),
            flush=True,
        )
    if it == maxiter or gopt < 1e-13:
        break
    # inner loop: find an acceptable step
    while True:
        if arm == "std":
            q = -np.linalg.solve(Bstd.T @ Bstd + lam * np.eye(y.size), Bstd.T @ r)
            pred = c - 0.5 * np.sum((r + Bstd @ q) ** 2)
            nin = 1
        elif arm == "comp":
            q, nin = solve_comp(s, B, r, lam, np.zeros(y.size))
            pred = c - model_val(s, B, r, q)
        elif arm == "compS" and not use_S:
            q, nin = solve_comp(s, B, r, lam, np.zeros(y.size))
            pred = c - model_val(s, B, r, q)
        else:
            q, nin = solve_comp(s, B, r, max(lam, lam_floor), np.zeros(y.size), Sd)
            pred = c - model_val(s, B, r, q) - 0.5 * q @ Sd @ q
        y_new = y + jnp.asarray(d * q)
        s_new = np.asarray(s_fun(y_new))
        c_new = cost_of(s_new)
        rho = (c - c_new) / pred if pred > 0 else -1.0
        if rho > 0:
            if arm == "compS":
                p_gn = c - model_val(s, B, r, q)
                p_s = p_gn - 0.5 * q @ (d[:, None] * Su * d[None, :]) @ q
                use_S = abs(p_s - (c - c_new)) < abs(p_gn - (c - c_new))
                step_prev, g_prev = d * q, g
            y, s, c = y_new, s_new, c_new
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
            A = None
            break
        lam *= nu
        nu *= 2
        if lam > 1e20:
            break
    if lam > 1e20:
        print("lambda blew up")
        break
    hist[-1] = hist[-1] + (rho, np.linalg.norm(q), nin)

np.save(f"{out}_hist.npy", np.array([h[:6] for h in hist]))
np.save(f"{out}_y.npy", np.asarray(y))
np.save(f"{out}_x.npy", np.asarray(proj.recover(y)))
print("done", out)
