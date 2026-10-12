"""Max first-order iota of a QA (or QH) near-axis field vs a cap on the axis vertical excursion max|Z|.
Free: axis harmonics R1..R5, Z1..Z5 (R0 = 1), eta_bar. Constraints (penalties, smooth max): max|Z| <= zcap,
elongation <= emax, curvature >= kmin (Frenet frame). Multi-start L-BFGS-B with JAX gradients. CPU, minutes."""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from nae import NearAxis, helicity, elongation

K = 5
def smax(x, t=200.0):
    return jax.scipy.special.logsumexp(t * x) / t

def run(nfp, hel, zcap, emax=5.0, kmin=0.15, starts=12, seed=0):
    na = NearAxis(nfp, 61)
    def unpack(v):
        rc = jnp.concatenate([jnp.ones(1), v[:K]]); zs = jnp.concatenate([jnp.zeros(1), v[K:2 * K]])
        return rc, zs, v[2 * K]
    def f(v):
        rc, zs, eta = unpack(v)
        q = na.solve(rc, zs, eta, hel, iota0=0.4 if hel == 0 else 1.2)
        e = elongation(q["X1c"], q["Y1s"], q["Y1c"])
        zm = smax(jnp.abs(q["Z"]))
        pen = 1e3 * (jnp.maximum(0, zm - zcap) / zcap) ** 2 + 1e2 * jnp.maximum(0, smax(e) - emax) ** 2 \
            + 1e3 * jnp.maximum(0, kmin - (-smax(-q["kappa"]))) ** 2
        return -jnp.abs(q["iota"]) + pen, (q["iota"], jnp.max(jnp.abs(q["Z"])), jnp.max(e), jnp.min(q["kappa"]))
    vg = jax.jit(jax.value_and_grad(lambda v: f(v)[0])); aux = jax.jit(lambda v: f(v)[1])
    rng = np.random.default_rng(seed); best = None
    for s in range(starts):
        v0 = np.zeros(2 * K + 1)
        v0[0] = rng.uniform(0.05, 0.25); v0[1:K] = rng.normal(0, 0.01, K - 1)
        v0[K:2 * K] = rng.normal(0, zcap / 2, K) / (1 + np.arange(K)) ; v0[2 * K] = rng.uniform(0.4, 1.5)
        rc, zs, _ = unpack(jnp.asarray(v0))
        if helicity(na.geometry(rc, zs)["n"], na.phi, nfp) != hel:
            continue
        try:
            r = minimize(lambda v: tuple(np.asarray(t, dtype=float) for t in vg(jnp.asarray(v))), v0, jac=True,
                         method="L-BFGS-B", options=dict(maxiter=400))
        except Exception:
            continue
        io, zm, em, km = [float(t) for t in aux(jnp.asarray(r.x))]
        rc, zs, _ = unpack(jnp.asarray(r.x))
        h = helicity(na.geometry(rc, zs)["n"], na.phi, nfp)
        ok = zm <= 1.02 * zcap and em <= 1.02 * emax and km >= 0.98 * kmin and h == hel and np.isfinite(io)
        if ok and (best is None or abs(io) > abs(best["iota"])):
            best = dict(iota=io, zmax=zm, elong=em, kmin=km, x=np.asarray(r.x).round(4).tolist())
    return best

out = {}
for nfp in (2, 3, 4):
    for zcap in (0.01, 0.02, 0.03, 0.05, 0.08):
        b = run(nfp, 0.0, zcap)
        out[f"QA nfp{nfp} zcap{zcap}"] = b
        print(f"QA nfp {nfp} max|Z| <= {zcap:.2f}: " + (f"max iota {abs(b['iota']):.3f} (|Z| {b['zmax']:.3f}, elong {b['elong']:.2f}, "
              f"Z harmonics {b['x'][K:2*K]})" if b else "no feasible start"), flush=True)
json.dump(out, open("runs2/iota_vs_z.json", "w"), indent=1)
