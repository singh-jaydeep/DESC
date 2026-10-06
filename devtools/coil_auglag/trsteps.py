"""Trust-region steps over a ladder of radii; attribute model mismatch to objectives."""

import sys

import jax.numpy as jnp
import numpy as np
from common import groups, load_run
from scipy.optimize import brentq

tag = sys.argv[1]
obj, proj, Y = load_run(tag)
y = jnp.asarray(Y[-1])
f = np.asarray(proj.compute_scaled_error(y))
J = np.asarray(proj.jac_scaled_error(y))
d = 1 / np.linalg.norm(J, axis=0)
Jh = J * d
U, s, Vt = np.linalg.svd(Jh, full_matrices=False)
uf = U.T @ f
del U
names, sl = groups(obj)
step = lambda al: -(Vt.T @ (s * uf / (s**2 + al)))
pn = lambda al: np.linalg.norm(step(al))
print(
    "scaled sv: max %.2e min %.2e ; GN step norm %.3e ; cost %.6e"
    % (s.max(), s.min(), pn(0), 0.5 * f @ f)
)
print("  Delta     alpha    ratio   " + " ".join(f"{n:>14s}" for n in names))
for Delta in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
    al = 0.0 if Delta >= pn(0) else brentq(lambda a: pn(a) - Delta, 0, 1e12)
    p = d * step(al)
    fn = np.asarray(proj.compute_scaled_error(y + jnp.asarray(p)))
    fl = f + J @ p
    pred = 0.5 * fl @ fl - 0.5 * f @ f
    act = 0.5 * fn @ fn - 0.5 * f @ f
    print(
        f"{Delta:7.2f} {al:9.2e} {act/pred:7.3f}   "
        + " ".join(f"{0.5*fn[q]@fn[q]-0.5*fl[q]@fl[q]:+14.3e}" for q in sl)
    )
