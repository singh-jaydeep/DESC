"""Warm re-solves, as in a proximal step: from the TRUE minimizer, a small boundary step, perturb, re-solve.

1. Converge the k5 equilibrium with subspace Newton (the true LSQ minimizer); save it.
2. For t in T: boundary step t * (BDRY boundary - k5 boundary), eq.perturb (order 2), then re-solve
   (a) GN (lsq-exact, tol 1e-10) and (b) subspace Newton, and (c) subspace Newton from (b)'s start + 1e-6 noise.
   Report iterations, wall, QS and iota, and the spread (b) vs (c). QS(t) from (b) should be smooth in t.

    python work/ss_qh/joint/stall/sn_warm.py EQ.h5 BDRY.h5 K T1,T2,...
"""

import os
import sys
import time

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
from gauge3 import force_cost, iota, qs_cost  # noqa: E402
from subspace_newton import subspace_newton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def lcp_for(eq):
    obj = O.ObjectiveFunction((O.ForceBalance(eq),))
    obj.build(verbose=0)
    con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq))))
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
    return lcp


def sn_solve(eq, K, maxiter=300):
    lcp = lcp_for(eq)
    t0 = time.time()
    x, info = subspace_newton(lcp, lcp.x(eq), k=K, maxiter=maxiter, gtol=1e-11, ftol=1e-13, xtol=1e-12, verbose=0)
    out = eq.copy()
    out.params_dict = eq.unpack_params(np.asarray(lcp.recover(x)))
    return out, info["nit"], time.time() - t0, info["msg"]


def gn_solve(eq):
    t0 = time.time()
    _, res = eq.solve(objective="force", constraints=O.get_fixed_boundary_constraints(eq), ftol=1e-10, xtol=1e-10,
                      gtol=1e-10, maxiter=300, verbose=0)
    return int(res["nit"]), time.time() - t0


path, bpath, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
T = [float(v) for v in sys.argv[4].split(",")]
eq0, _ = S.load_equilibrium(path)
eb, _ = S.load_equilibrium(bpath)
true_path = os.path.join(HERE, "eq_k5_true_minimizer.h5")
if os.path.exists(true_path):
    eqT, _ = S.load_equilibrium(true_path)
    print(f"loaded {true_path}")
else:
    eqT, nit, wall, msg = sn_solve(eq0, K)
    eqT.save(true_path)
    print(f"true minimizer: {nit} its, {wall:.0f} s ({msg}); saved {true_path}")
print(f"T0: force {force_cost(eqT):.6e}  QS {qs_cost(eqT):.8e}", flush=True)
dR = np.asarray(eb.Rb_lmn) - np.asarray(eq0.Rb_lmn)
dZ = np.asarray(eb.Zb_lmn) - np.asarray(eq0.Zb_lmn)
rng = np.random.default_rng(1)
for t in T:
    base = eqT.copy()
    base.perturb(deltas={"Rb_lmn": t * dR, "Zb_lmn": t * dZ}, order=2, verbose=0)
    qp = qs_cost(base)
    eG = base.copy()
    nG, wG = gn_solve(eG)
    eS, nS, wS, mS = sn_solve(base, K)
    eN = base.copy()
    for nm in ("R_lmn", "Z_lmn", "L_lmn"):
        x = np.asarray(getattr(eN, nm))
        setattr(eN, nm, x + 1e-6 * rng.standard_normal(x.shape))
    eN, nN, wN, _ = sn_solve(eN, K)
    qG, qS, qN = qs_cost(eG), qs_cost(eS), qs_cost(eN)
    iS, iN, iG = iota(eS), iota(eN), iota(eG)
    print(f"t {t:g}: perturbed QS {qp:.8e} | GN {nG} its {wG:.0f} s force {force_cost(eG):.6e} QS {qG:.8e} | "
          f"SN {nS} its {wS:.0f} s ({mS}) force {force_cost(eS):.6e} QS {qS:.8e} | SN-noisy {nN} its: QS spread "
          f"{abs(qN - qS) / qS:.1e} iota {np.abs(iN - iS).max():.1e} | GN vs SN: QS {abs(qG - qS) / qS:.1e} "
          f"iota {np.abs(iG - iS).max():.1e}", flush=True)
