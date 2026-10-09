"""Subspace-Newton equilibrium solve (sandbox/subspace_newton.py) vs the exact-Hessian reference.

From the GN-"converged" k5 state and from 2 starts with 1e-6 random interior noise: does it reach the exact-
Hessian minimizer (force 1.1241e-7, QS 4.445482e-4), identically from every start, and at what cost?

    python work/ss_qh/joint/stall/sn_test.py EQ.h5 K [MAXITER]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.objectives.getters import maybe_add_self_consistency  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402
from gauge3 import force_cost, iota, qs_booz, qs_cost  # noqa: E402
from subspace_newton import subspace_newton  # noqa: E402

path, K = sys.argv[1], int(sys.argv[2])
maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 200
eq0, _ = S.load_equilibrium(path)
obj = O.ObjectiveFunction((O.ForceBalance(eq0),))
obj.build(verbose=0)
con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq0, O.get_fixed_boundary_constraints(eq0))))
lcp = LinearConstraintProjection(obj, con)
lcp.build(verbose=0)
print(f"reduced dim {lcp.dim_x}, rows {lcp.dim_f}, k {K}")

rng = np.random.default_rng(0)
results = []
for t in range(3):
    e = eq0.copy()
    if t > 0:
        for nm in ("R_lmn", "Z_lmn", "L_lmn"):
            x = np.asarray(getattr(e, nm))
            setattr(e, nm, x + 1e-6 * rng.standard_normal(x.shape))
    x0 = lcp.x(e)
    print(f"--- start {t} ({'GN-converged k5' if t == 0 else 'noisy'})", flush=True)
    x, info = subspace_newton(lcp, x0, k=K, maxiter=maxiter, verbose=1)
    eS = eq0.copy()
    eS.params_dict = eq0.unpack_params(np.asarray(lcp.recover(x)))
    q, b, i, fc = qs_cost(eS), qs_booz(eS), iota(eS), force_cost(eS)
    print(f"start {t}: {info['msg']} after {info['nit']} its, {info['wall']:.0f} s; force {fc:.6e}  QS {q:.7e}  "
          f"Booz {b[0]:.4e} {b[1]:.4e} {b[2]:.4e}  iota [{i.min():.6f}, {i.max():.6f}]", flush=True)
    results.append((q, i, fc))
q0, i0, _ = results[0]
for t in (1, 2):
    q, i, _ = results[t]
    print(f"spread start {t} vs start 0: QS rel {abs(q - q0) / q0:.2e}  iota max {np.abs(i - i0).max():.2e}")
print(f"vs exact-Hessian reference (QS 4.445482e-04, force 1.1241e-07): QS rel {abs(q0 - 4.445482e-4) / 4.445482e-4:.2e}")
