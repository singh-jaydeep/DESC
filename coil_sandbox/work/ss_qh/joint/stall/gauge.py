"""Does the proximal re-solve drift along a force-balance null direction, and is it a theta relabeling?

Re-solves the given equilibrium in place (same objective and constraints as the proximal re-solve:
ForceBalance + the fixed-boundary set) at a tolerance, then compares it with the original:
  * force cost before/after, QS two-term (scale-invariant, single_stage's grid) and iota before/after
  * the displacement of the flux-surface points at FIXED computational (rho, theta, zeta), split into
    the part normal to the flux surface (physical) and the part along e_theta (a poloidal relabeling,
    which force balance cannot see).

    python work/ss_qh/joint/stall/gauge.py EQ.h5 TOL MAXITER [--perturb-order N]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from qs_scale_invariant import QuasisymmetryTwoTermScaleInvariant  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def metrics(eq, label):
    qg = LinearGrid(rho=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    qs = QuasisymmetryTwoTermScaleInvariant(eq, helicity=(1, 4), grid=qg)
    qs.build(verbose=0)
    r = np.asarray(qs.compute_scaled_error(eq.params_dict))
    fb = O.ObjectiveFunction(O.ForceBalance(eq))
    fb.build(verbose=0)
    fc = 0.5 * float(np.sum(np.asarray(fb.compute_scaled_error(fb.x(eq))) ** 2))
    ig = LinearGrid(rho=np.linspace(0.05, 1.0, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = ig.compress(eq.compute("iota", grid=ig)["iota"])
    print(f"{label}: force cost {fc:.6e}  QS |r|^2/2 {0.5 * np.sum(r**2):.8e}  iota[0,mid,-1] "
          f"{iota[0]:.8f} {iota[10]:.8f} {iota[-1]:.8f}", flush=True)
    return r, np.asarray(iota)


def main():
    path, tol, maxiter = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    eq0, _ = S.load_equilibrium(path)
    eq1 = eq0.copy()
    r0, i0 = metrics(eq0, "original")
    cons = O.get_fixed_boundary_constraints(eq1)
    eq1.solve(objective="force", constraints=cons, ftol=tol, xtol=tol, gtol=tol, maxiter=maxiter, verbose=1)
    r1, i1 = metrics(eq1, f"re-solved tol {tol:g}")
    print(f"QS residual change |dr|/|r| {np.linalg.norm(r1 - r0) / np.linalg.norm(r0):.3e}; "
          f"iota max change {np.abs(i1 - i0).max():.3e}")
    for nm in ["R_lmn", "Z_lmn", "L_lmn"]:
        a, b = np.asarray(getattr(eq0, nm)), np.asarray(getattr(eq1, nm))
        print(f"  {nm}: |d| {np.linalg.norm(b - a):.3e}  |x| {np.linalg.norm(a):.3e}")
    g = LinearGrid(rho=np.linspace(0.1, 1.0, 10), M=2 * eq0.M_grid, N=2 * eq0.N_grid, NFP=eq0.NFP, sym=False)
    d0 = eq0.compute(["R", "Z", "R_t", "Z_t", "lambda"], grid=g)
    d1 = eq1.compute(["R", "Z", "lambda"], grid=g)
    dR, dZ = np.asarray(d1["R"] - d0["R"]), np.asarray(d1["Z"] - d0["Z"])
    Rt, Zt = np.asarray(d0["R_t"]), np.asarray(d0["Z_t"])
    nt = np.hypot(Rt, Zt)
    tang = (dR * Rt + dZ * Zt) / nt  # along e_theta (in the R,Z plane at fixed zeta)
    norm = (dR * Zt - dZ * Rt) / nt  # normal to the surface cross-section
    rho = np.asarray(g.nodes[:, 0])
    print("rho   rms tangential   rms normal   ratio n/t   (m)   [theta shift omega = tang/|e_theta|, rms]")
    for rr in np.unique(rho):
        m = rho == rr
        om = tang[m] / nt[m]
        print(f"{rr:.2f}  {np.sqrt(np.mean(tang[m]**2)):.3e}   {np.sqrt(np.mean(norm[m]**2)):.3e}   "
              f"{np.sqrt(np.mean(norm[m]**2)) / max(np.sqrt(np.mean(tang[m]**2)), 1e-300):.3e}   "
              f"omega rms {np.sqrt(np.mean(om**2)):.3e}")
    dl = np.asarray(d1["lambda"] - d0["lambda"])
    print(f"lambda change rms {np.sqrt(np.mean(dl**2)):.3e}")


if __name__ == "__main__":
    main()
