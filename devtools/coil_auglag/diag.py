"""Endpoint report: per-objective cost and, with --hess, the Gauss-Newton vs exact
Hessian gap, scaled gradient and Newton decrement (the optimality certificate).

Constraint values and topology: check.py.

usage: python diag.py TAG X_FILE [--hess]
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import groups, load_run

tag, xfile = sys.argv[1], sys.argv[2]
obj, proj, _ = load_run(tag)
x = jnp.asarray(np.load(xfile))
y = proj.project(x)
f = np.asarray(proj.compute_scaled_error(y))
names, sl = groups(obj)
print(f"total cost {0.5 * f @ f:.10e}")
for n, s in zip(names, sl):
    print(f"  {n:16s} rows={s.stop - s.start:6d} cost={0.5 * f[s] @ f[s]:.6e}")

if "--hess" in sys.argv:
    cost = lambda y: 0.5 * jnp.sum(proj.compute_scaled_error(y) ** 2)
    hvp = jax.jit(lambda v: jax.jvp(jax.grad(cost), (y,), (v,))[1])
    I = np.eye(y.size)
    H = np.stack([np.asarray(hvp(jnp.asarray(I[i]))) for i in range(y.size)], 1)
    H = 0.5 * (H + H.T)
    J = np.asarray(proj.jac_scaled_error(y))
    J = np.where((f != 0)[:, None], J, 0.0)  # inactive hinge rows
    d = 1 / np.maximum(np.linalg.norm(J, axis=0), 1e-300)
    G = d[:, None] * (J.T @ J) * d[None]
    Hd = d[:, None] * H * d[None]
    S = Hd - G
    eg = np.linalg.eigvalsh(G)
    eh, V = np.linalg.eigh(Hd)
    print("eig GN  smallest", eg[:4], " largest", eg[-1])
    print("eig H   smallest", eh[:4], " largest", eh[-1])
    print("|S|_2 =", np.abs(np.linalg.eigvalsh(S)).max())
    for i in range(4):
        v = V[:, i]
        print(f"  flat dir {i}: vGv={v @ G @ v:.3e}  vSv={v @ S @ v:+.3e}")
    gd = d * (J.T @ f)
    print(
        "|Dg|inf",
        np.abs(gd).max(),
        " Newton decrement",
        0.5 * gd @ np.linalg.solve(Hd, gd),
    )
