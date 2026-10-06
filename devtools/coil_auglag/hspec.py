"""Hessian spectrum, Newton decrement and per-group curvature of flat directions."""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import groups, load_phase, load_run

if os.environ.get("PHASE"):
    obj, proj, y, _ = load_phase(
        sys.argv[1], sys.argv[2], json.loads(os.environ.get("KW", "{}"))
    )
    y = jnp.asarray(y)
else:
    obj, proj, Y = load_run(sys.argv[1])
    y = jnp.asarray(np.load(sys.argv[2]))
cost = lambda y: 0.5 * jnp.sum(proj.compute_scaled_error(y) ** 2)
hvp = jax.jit(lambda v: jax.jvp(jax.grad(cost), (y,), (v,))[1])
I = np.eye(y.size)
H = np.stack([np.asarray(hvp(jnp.asarray(I[k]))) for k in range(y.size)], 1)
H = 0.5 * (H + H.T)
J = np.asarray(proj.jac_scaled_error(y))
d = 1 / np.linalg.norm(J, axis=0)
Hd = d[:, None] * H * d[None]
ev, V = np.linalg.eigh(Hd)
print("eig(D H D) smallest:", ev[:6])
print("largest:", ev[-2:])
g = J.T @ np.asarray(proj.compute_scaled_error(y))
print("|g|inf", np.abs(g).max(), " |Dg|inf", np.abs(d * g).max())
gd = d * g
print(
    "Newton decrement 0.5 g^T H^-1 g =",
    0.5 * gd @ np.linalg.solve(Hd, gd),
    "  (cost =",
    float(cost(y)),
    ")",
)
w_, V_ = ev, V
c = V_.T @ gd
print(
    "decrement from 6 flattest dirs:",
    0.5 * np.sum(c[:6] ** 2 / w_[:6]),
    " from the rest:",
    0.5 * np.sum(c[6:] ** 2 / w_[6:]),
)
names, sl = groups(obj)
f = np.asarray(proj.compute_scaled_error(y))
for k in range(0):
    u = d * V[:, k]
    row = []
    for n, q in zip(names, sl):
        if not np.any(f[q]) and n.startswith(("plasma", "coil length")):
            continue
        fk = lambda yy: 0.5 * jnp.sum(proj.compute_scaled_error(yy)[q] ** 2)
        hk = jax.jvp(jax.grad(fk), (y,), (jnp.asarray(u),))[1]
        row.append(f"{n}={float(u@np.asarray(hk)):+.2e}")
    print(f"dir {k} (eig {ev[k]:+.2e}): ", "  ".join(row))
