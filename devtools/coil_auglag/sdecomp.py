"""Split the dropped Gauss-Newton term S = sum_i r_i Hess(r_i) by objective group.

usage: python sdecomp.py TAG X_FILE
For each group k: S_k = Hess(0.5|r_k|^2) - J_k^T J_k, in the solver's column scaling.
Reports S_k along the directions where GN is slowest (largest -vSv / vGv) and the
coil parameters those directions move.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import groups, load_run

tag, xfile = sys.argv[1], sys.argv[2]
obj, proj, _ = load_run(tag)
y = proj.project(jnp.asarray(np.load(xfile)))
names, sl = groups(obj)
f = np.asarray(proj.compute_scaled_error(y))
J = np.asarray(proj.jac_scaled_error(y))
d = 1 / np.linalg.norm(J, axis=0)
n = y.size
I = np.eye(n)
G, S = {}, {}
for nm, s in zip(names, sl):
    cost = lambda y, s=s: 0.5 * jnp.sum(proj.compute_scaled_error(y)[s] ** 2)
    hvp = jax.jit(lambda v, cost=cost: jax.jvp(jax.grad(cost), (y,), (v,))[1])
    H = np.stack([np.asarray(hvp(jnp.asarray(I[i]))) for i in range(n)], 1)
    H = d[:, None] * 0.5 * (H + H.T) * d[None]
    Jk = J[s] * d
    G[nm], S[nm] = Jk.T @ Jk, H - Jk.T @ Jk
    r = f[s]
    print(
        f"{nm:16s} |r|={np.linalg.norm(r):.3e}  max|r|={np.abs(r).max():.3e}  "
        f"active={np.sum(r != 0)}  |S_k|2={np.abs(np.linalg.eigvalsh(S[nm])).max():.3e}"
    )
Gt, St = sum(G.values()), sum(S.values())
np.savez(
    f"{tag}_sdecomp.npz",
    d=d,
    **{"G_" + k: G[k] for k in G},
    **{"S_" + k: S[k] for k in S},
)
from scipy.linalg import eigh  # noqa: E402

mu, V = eigh(-St, Gt)  # -S v = mu G v ; GN step along v too short by 1/(1-mu)
order = np.argsort(-mu)[:5]
print("\nslowest GN directions: mu = -vSv/vGv (GN contraction per step ~ mu)")
x = np.load(xfile)
lab = obj.unpack_state(jnp.arange(x.size, dtype=float), False)[0]
labels, scale = {}, np.ones(x.size)
for c, p in enumerate(lab):
    for k, v in p.items():
        idx = np.atleast_1d(np.asarray(v)).astype(int)
        for j, t in enumerate(idx):
            labels[t] = f"c{c}.{k}[{j}]"
        scale[idx] = np.sqrt(np.mean(x[idx] ** 2)) + 1e-12
Z = np.asarray(jax.jacfwd(proj.recover)(y))
for i in order:
    v = V[:, i] / np.sqrt(V[:, i] @ Gt @ V[:, i])
    print(
        f"  mu={mu[i]:+.3f}  vGv=1  vSv by group: "
        + "  ".join(f"{nm}:{v @ S[nm] @ v:+.3f}" for nm in names)
    )
    dx = Z @ (d * v) / scale
    top = np.argsort(-np.abs(dx))[:6]
    print(
        "     moves (relative): " + ", ".join(f"{labels[t]}={dx[t]:+.2f}" for t in top)
    )
