"""Worst GN contraction mu after pinning candidate near-gauge parameters.

usage: python pintest.py TAG X_FILE   (needs TAG_sdecomp.npz from sdecomp.py)
mu = max generalized eig of (-S, G) restricted to {v : (Z d v)[pinned] = 0}.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from common import load_run
from scipy.linalg import eigh, null_space

tag, xfile = sys.argv[1], sys.argv[2]
obj, proj, _ = load_run(tag)
x = np.load(xfile)
y = proj.project(jnp.asarray(x))
z = np.load(f"{tag}_sdecomp.npz")
d = z["d"]
names = [k[2:] for k in z.files if k.startswith("G_")]
G = sum(z["G_" + k] for k in names)
S = sum(z["S_" + k] for k in names)
Z = np.asarray(jax.jacfwd(proj.recover)(y)) * d[None, :]  # dx per unit scaled y
lab = obj.unpack_state(jnp.arange(x.size, dtype=float), False)[0]
idx = {}
for c, p in enumerate(lab):
    for k, v in p.items():
        idx[(c, k)] = np.atleast_1d(np.asarray(v)).astype(int)
cs = obj.objectives[1].things[0]
pts = [np.asarray(c.compute("x", basis="xyz")["x"]) for c in cs]


def inplane(c):
    """Rows of Z giving the in-plane centre displacement of coil c (2 per coil)."""
    n = np.asarray(cs[c].normal, float)
    n = n / np.linalg.norm(n)
    # centre is stored in rpz; use its xyz Jacobian via the coil positions' mean
    a = np.cross(n, [0, 0, 1.0])
    a = a / np.linalg.norm(a)
    b = np.cross(n, a)
    return a, b


def mu_with(rows):
    if len(rows) == 0:
        return eigh(-S, G, eigvals_only=True).max()
    C = np.vstack(rows)
    Nn = null_space(C)
    return eigh(-(Nn.T @ S @ Nn), Nn.T @ G @ Nn, eigvals_only=True).max(), Nn.shape[1]


r1 = []
for c in range(len(cs)):
    r = idx[(c, "r_n")]
    r1 += [Z[r[6]], Z[r[8]]]  # modes n = -1, +1
cen = []
for c in range(len(cs)):
    ci = idx[(c, "center")]
    cen += [Z[ci[0]], Z[ci[2]]]  # R and Z of the centre (rpz basis), roughly in-plane
print("no pin                 mu = %.3f" % mu_with([]))
print("pin r_{+-1}            mu = %.3f (dim %d)" % mu_with(r1))
print("pin centre R,Z         mu = %.3f (dim %d)" % mu_with(cen))
print("pin both               mu = %.3f (dim %d)" % mu_with(r1 + cen))
rng = np.random.default_rng(0)
allrows = [i for (c, k), v in idx.items() if k in ("r_n", "center") for i in v]
for t in range(3):
    pick = rng.choice(allrows, 8, replace=False)
    print("control: 8 random      mu = %.3f (dim %d)" % mu_with([Z[i] for i in pick]))
