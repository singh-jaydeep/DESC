"""Does the equal-arclength GaugeCondition (sandbox/gauge_fix.py) remove the branch choice without changing physics?

From branch A (BASE) and branch B (TRIAL's interior on BASE's boundary), solve ForceBalance + GaugeCondition(w)
with subspace Newton, for each weight w. Uniqueness: both starts must land on one state. Neutrality: its
gauge-invariant physics (Boozer QS, iota) vs the basis-10 reference, and the force residual vs the unfixed branches.

    python work/ss_qh/joint/stall/gauge_fix_test.py BASE.h5 TRIAL.h5 W1,W2,... [K]
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
from gauge_fix import GaugeCondition  # noqa: E402
from subspace_newton import subspace_newton  # noqa: E402

pbase, ptrial = sys.argv[1], sys.argv[2]
W = [float(v) for v in sys.argv[3].split(",")]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 128
REF = "basis-10 reference: QS 4.225-4.245e-04  Booz 2.07e-03 3.64e-03 7.83-7.84e-03  iota [1.1805, 1.2028-1.2030]"


def gauge_norm(eq):
    g = GaugeCondition(eq, weight=1.0)
    g.build(verbose=0)
    return float(np.linalg.norm(np.asarray(g.compute_scaled_error(eq.params_dict))))


def solve(eq, w, maxiter=500):
    terms = (O.ForceBalance(eq),) + ((GaugeCondition(eq, weight=w),) if w > 0 else ())
    obj = O.ObjectiveFunction(terms)
    obj.build(verbose=0)
    con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq))))
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
    x, info = subspace_newton(lcp, lcp.x(eq), k=K, maxiter=maxiter, gtol=1e-12, ftol=1e-14, xtol=1e-13, verbose=0)
    out = eq.copy()
    out.params_dict = eq.unpack_params(np.asarray(lcp.recover(x)))
    return out, f"{info['nit']} its {info['wall']:.0f} s"


def line(label, eq):
    b, i = qs_booz(eq), iota(eq)
    print(f"{label:<30} force {force_cost(eq):.6e}  |G| {gauge_norm(eq):.3e}  QS {qs_cost(eq):.7e}  "
          f"Booz {b[0]:.4e} {b[1]:.4e} {b[2]:.4e}  iota [{i.min():.6f}, {i.max():.6f}]", flush=True)
    return qs_cost(eq), i


A, _ = S.load_equilibrium(pbase)
Bt, _ = S.load_equilibrium(ptrial)
B = Bt.copy()
B.Rb_lmn, B.Zb_lmn = np.asarray(A.Rb_lmn), np.asarray(A.Zb_lmn)
print(REF)
line("branch A (unfixed)", A)
line("branch B (unfixed, A's bdry)", B)
for w in W:
    print(f"--- weight {w:g}", flush=True)
    eA, mA = solve(A.copy(), w)
    qa, ia = line(f"from A [{mA}]", eA)
    eB, mB = solve(B.copy(), w)
    qb, ib = line(f"from B [{mB}]", eB)
    dX = np.linalg.norm(np.asarray(eA.R_lmn - eB.R_lmn)) + np.linalg.norm(np.asarray(eA.Z_lmn - eB.Z_lmn))
    print(f"   A-start vs B-start: QS rel {abs(qa - qb) / qa:.2e}  iota max {np.abs(ia - ib).max():.2e}  "
          f"|dRZ_lmn| {dX:.2e}  |dL_lmn| {np.linalg.norm(np.asarray(eA.L_lmn - eB.L_lmn)):.2e}", flush=True)
