"""Implicit derivative of the equilibrium w.r.t. the boundary: zero-residual (DESC) vs full (with S).

At the true minimizer x* of 1/2 |F(x; c)|^2 (eq_k5_true_minimizer.h5) and a boundary direction dc
(BDRY boundary - k5 boundary, per unit t), predict dQS/dt from
  (a) zero-residual (DESC proximal / perturb):  dx = -(J^T J)^-1 J^T F_c
  (b) full implicit derivative:                  dx = -B^-1 d(J^T F)/dc,  B = J^T J + V (V^T S V) V^T
F_c and d(J^T F)/dc by central differences in c (two fixed-boundary projections with the boundary moved by
+-eps; their nullspace Z is the same, only the particular solution xp moves). Compare with the true slope
from converged subspace-Newton re-solves (sn_warm.py: ~ -3.8e-5).

    python work/ss_qh/joint/stall/sn_deriv.py EQ_TRUE.h5 EQ_K5.h5 BDRY.h5 [K] [EPS]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot  # noqa: E402

boot.setup(default="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.backend import jnp  # noqa: E402
from gauge3 import qs_cost  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.objectives.getters import maybe_add_self_consistency  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402


def lcp_for(eq):
    obj = O.ObjectiveFunction((O.ForceBalance(eq),))
    obj.build(verbose=0)
    con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq))))
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
    return lcp


ptrue, pk5, pb = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 128
eps = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
eqT, _ = S.load_equilibrium(ptrue)
e5, _ = S.load_equilibrium(pk5)
eb, _ = S.load_equilibrium(pb)
dR = np.asarray(eb.Rb_lmn) - np.asarray(e5.Rb_lmn)
dZ = np.asarray(eb.Zb_lmn) - np.asarray(e5.Zb_lmn)


def moved(s):
    e = eqT.copy()
    e.Rb_lmn = np.asarray(eqT.Rb_lmn) + s * dR
    e.Zb_lmn = np.asarray(eqT.Zb_lmn) + s * dZ
    return e


L0, Lp, Lm = lcp_for(eqT), lcp_for(moved(eps)), lcp_for(moved(-eps))
x = L0.x(eqT)  # reduced coordinates of the true minimizer
xf0 = np.asarray(L0.recover(x))
dxp = (np.asarray(Lp.recover(x)) - np.asarray(Lm.recover(x))) / (2 * eps)  # d x_full / dt at fixed x_red
print(f"|d xp/dt| {np.linalg.norm(dxp):.3e}; check Z equal: "
      f"{np.abs(np.asarray(Lp._Z) - np.asarray(Lm._Z)).max():.1e}", flush=True)

J = np.asarray(L0.jac_scaled_error(x))
F = np.asarray(L0.compute_scaled_error(x))
Fc = (np.asarray(Lp.compute_scaled_error(x)) - np.asarray(Lm.compute_scaled_error(x))) / (2 * eps)


def grad_of(L):
    return np.asarray(jax.grad(lambda y: 0.5 * jnp.sum(L.compute_scaled_error(y) ** 2))(x))


m = (grad_of(Lp) - grad_of(Lm)) / (2 * eps)  # d(J^T F)/dt, includes the mixed second-order term
A = J.T @ J
print(f"|J^T F| at x* {np.abs(J.T @ F).max():.2e}; |J^T F_c| {np.linalg.norm(J.T @ Fc):.3e}; "
      f"|m| {np.linalg.norm(m):.3e}; |m - J^T F_c| {np.linalg.norm(m - J.T @ Fc):.3e}", flush=True)

lam, Q = np.linalg.eigh(A)
V = Q[:, :K]
grad = jax.grad(lambda y: 0.5 * jnp.sum(L0.compute_scaled_error(y) ** 2))
HV = np.stack([np.asarray(jax.jvp(grad, (x,), (jnp.asarray(V[:, i]),))[1]) for i in range(K)], axis=1)
SV = HV - A @ V
Ssub = 0.5 * (V.T @ SV + SV.T @ V)
B = A + V @ Ssub @ V.T
B = 0.5 * (B + B.T)

dx_a = -np.linalg.lstsq(J, Fc, rcond=None)[0]           # zero-residual (DESC, no spectral shift)
w, P = np.linalg.eigh(A)
dx_a_shift = -P @ ((P.T @ (J.T @ Fc)) / (w + w[0]))     # DESC's sf += sf[-1] is s^2 + 2 s s_min + s_min^2 ~ shift
dx_b = -np.linalg.solve(B, m)                           # full implicit derivative
print(f"|dx_red| zero-residual {np.linalg.norm(dx_a):.3e}  full {np.linalg.norm(dx_b):.3e}; "
      f"angle cos {dx_a @ dx_b / np.linalg.norm(dx_a) / np.linalg.norm(dx_b):+.3f}", flush=True)


def qs_at(xfull):
    e = eqT.copy()
    e.params_dict = eqT.unpack_params(xfull)
    return qs_cost(e)


def slope(dx_red, h=1e-3):
    dfull = dxp + (np.asarray(L0.recover(x + dx_red)) - xf0)  # d x_full / dt (recover is affine)
    return (qs_at(xf0 + h * dfull) - qs_at(xf0 - h * dfull)) / (2 * h)


print(f"dQS/dt: zero-residual {slope(dx_a):+.4e}   zero-residual (shifted) {slope(dx_a_shift):+.4e}   "
      f"full implicit {slope(dx_b):+.4e}   [true from converged re-solves ~ -3.8e-5]", flush=True)
