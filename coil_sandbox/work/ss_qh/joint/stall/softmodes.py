"""The soft modes of the fixed-boundary force balance: are they all theta relabelings?

SVD of the scaled ForceBalance Jacobian in the feasible (fixed-boundary) coordinates. For the K smallest
singular vectors: the displacement (dR, dZ) on interior flux surfaces split along e_theta (relabeling) and
normal; and how well dlambda matches the relabeling prediction. For theta -> theta + omega at fixed physical
point: dX = -omega X_theta (so omega = -dX_t/|X_theta|) and dlambda = -omega (1 + lambda_theta).
"gauge fit" = 1 - |dlambda - dlambda_pred| / |dlambda| (1: a pure relabeling).

    python work/ss_qh/joint/stall/softmodes.py EQ.h5 [K] [PIN_EPS PIN_P]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402
from gauge_fix import LambdaCondensation  # noqa: E402

path = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
pp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
eq, _ = S.load_equilibrium(path)
terms = [O.ForceBalance(eq)] + ([LambdaCondensation(eq, weight=eps, power=pp)] if eps > 0 else [])
obj = O.ObjectiveFunction(tuple(terms))
obj.build(verbose=0)
con = O.ObjectiveFunction(tuple(O.maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq)))
                          if hasattr(O, "maybe_add_self_consistency") else tuple(O.get_fixed_boundary_constraints(eq)))
try:
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
except Exception:
    from desc.objectives.utils import maybe_add_self_consistency

    con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq))))
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
xr = lcp.x(eq)
J = np.asarray(lcp.jac_scaled_error(xr))
f = np.asarray(lcp.compute_scaled_error(xr))
U, s, Vt = np.linalg.svd(J, full_matrices=False)
print(f"J {J.shape}; sv max {s[0]:.3e}; smallest {np.array2string(s[-K:][::-1], precision=3)}")
print(f"sv ratios to max: {np.array2string(s[-K:][::-1] / s[0], precision=2)}")
print(f"residual energy along each soft left vector (|u.f|): {np.array2string(np.abs(U[:, -K:][:, ::-1].T @ f), precision=2)}")
print(f"#sv < 1e-6 max {int(np.sum(s < 1e-6 * s[0]))}, < 1e-5 {int(np.sum(s < 1e-5 * s[0]))}, "
      f"< 1e-4 {int(np.sum(s < 1e-4 * s[0]))}, < 1e-3 {int(np.sum(s < 1e-3 * s[0]))}")

g = LinearGrid(rho=np.linspace(0.2, 0.9, 8), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
d0 = eq.compute(["R", "Z", "R_t", "Z_t", "lambda", "lambda_t"], grid=g)
Rt, Zt = np.asarray(d0["R_t"]), np.asarray(d0["Z_t"])
nt = np.hypot(Rt, Zt)
lt = np.asarray(d0["lambda_t"])
x_full0 = lcp.recover(xr)
p0 = eq.unpack_params(x_full0)
print(" k   sv/smax   tang frac   gauge fit   |dL|/|dRZ|")
for j in range(1, K + 1):
    v = Vt[-j]
    xf = lcp.recover(xr + 1e-6 * v)
    p1 = eq.unpack_params(xf)
    e2 = eq.copy()
    e2.params_dict = {k: p1[k] for k in ("R_lmn", "Z_lmn", "L_lmn")} | {k: v_ for k, v_ in p1.items()
                                                                     if k not in ("R_lmn", "Z_lmn", "L_lmn")}
    d1 = e2.compute(["R", "Z", "lambda"], grid=g)
    dR, dZ = np.asarray(d1["R"] - d0["R"]), np.asarray(d1["Z"] - d0["Z"])
    dl = np.asarray(d1["lambda"] - d0["lambda"])
    tang = (dR * Rt + dZ * Zt) / nt
    norm = (dR * Zt - dZ * Rt) / nt
    tf = np.sqrt(np.sum(tang**2) / (np.sum(tang**2) + np.sum(norm**2)))
    om = -tang / nt
    dl_pred = -om * (1 + lt)
    fit = 1 - np.linalg.norm(dl - dl_pred) / max(np.linalg.norm(dl), 1e-300)
    dLc = np.linalg.norm(np.asarray(p1["L_lmn"] - p0["L_lmn"]))
    dRZc = np.linalg.norm(np.asarray(p1["R_lmn"] - p0["R_lmn"])) + np.linalg.norm(np.asarray(p1["Z_lmn"] - p0["Z_lmn"]))
    print(f"{j:2d}  {s[-j] / s[0]:.2e}   {tf:.3f}      {fit:+.3f}      {dLc / max(dRZc, 1e-300):.3f}", flush=True)
