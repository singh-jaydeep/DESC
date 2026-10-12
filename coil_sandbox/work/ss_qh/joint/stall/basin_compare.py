"""Two least-squares minima of the discrete equilibrium: same boundary? physical or discretization artifact?

    python work/ss_qh/joint/stall/basin_compare.py BASE.h5 TRIAL.h5 [K] [RES_UP]

1. Same boundary: TRIAL's interior with BASE's boundary, subspace-Newton solved (-> T'), and BASE's interior with
   TRIAL's boundary (-> B'). Distinct minima for one boundary iff T' != BASE (and B' != TRIAL).
2. The difference T' - BASE: force cost, QS (stock), Boozer QS, iota; displacement split along e_theta / normal
   and the relabeling fit of dlambda (1 = pure theta relabeling, which changes nothing physical in the continuum).
3. Radial profile of the force residual |F| (surface rms) for both, and where iota = 1.2 (12/10, n = 3 NFP).
4. Both raised to basis RES_UP (grids 2x) and re-solved: if they merge, the two minima are a discretization artifact.
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
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives.getters import maybe_add_self_consistency  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402
from gauge3 import force_cost, iota, qs_booz, qs_cost  # noqa: E402
from subspace_newton import subspace_newton  # noqa: E402

pbase, ptrial = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 128
RES_UP = int(sys.argv[4]) if len(sys.argv) > 4 else 10


def lcp_for(eq):
    obj = O.ObjectiveFunction((O.ForceBalance(eq),))
    obj.build(verbose=0)
    con = O.ObjectiveFunction(tuple(maybe_add_self_consistency(eq, O.get_fixed_boundary_constraints(eq))))
    lcp = LinearConstraintProjection(obj, con)
    lcp.build(verbose=0)
    return lcp


def sn(eq, maxiter=400):
    lcp = lcp_for(eq)
    x, info = subspace_newton(lcp, lcp.x(eq), k=K, maxiter=maxiter, gtol=1e-12, ftol=1e-14, xtol=1e-13, verbose=0)
    out = eq.copy()
    out.params_dict = eq.unpack_params(np.asarray(lcp.recover(x)))
    return out, f"{info['nit']} its {info['wall']:.0f} s ({info['msg']})"


def with_boundary(eq_interior, eq_boundary):
    e = eq_interior.copy()
    e.Rb_lmn, e.Zb_lmn = np.asarray(eq_boundary.Rb_lmn), np.asarray(eq_boundary.Zb_lmn)
    return e


def summary(label, eq):
    b, i = qs_booz(eq), iota(eq)
    print(f"{label:<34} force {force_cost(eq):.6e}  QS {qs_cost(eq):.7e}  Booz {b[0]:.4e} {b[1]:.4e} {b[2]:.4e}  "
          f"iota [{i.min():.6f}, {i.max():.6f}]", flush=True)


def diff(label, e0, e1):
    g = LinearGrid(rho=np.linspace(0.1, 0.95, 10), M=2 * e0.M_grid, N=2 * e0.N_grid, NFP=e0.NFP, sym=False)
    d0 = e0.compute(["R", "Z", "R_t", "Z_t", "lambda_t", "lambda"], grid=g)
    d1 = e1.compute(["R", "Z", "lambda"], grid=g)
    dR, dZ = np.asarray(d1["R"] - d0["R"]), np.asarray(d1["Z"] - d0["Z"])
    Rt, Zt = np.asarray(d0["R_t"]), np.asarray(d0["Z_t"])
    nt = np.hypot(Rt, Zt)
    tang, norm = (dR * Rt + dZ * Zt) / nt, (dR * Zt - dZ * Rt) / nt
    dl = np.asarray(d1["lambda"] - d0["lambda"])
    om = -tang / nt
    fit = 1 - np.linalg.norm(dl + om * (1 + np.asarray(d0["lambda_t"]))) / max(np.linalg.norm(dl), 1e-300)
    di = np.abs(iota(e1) - iota(e0)).max()
    print(f"{label:<34} |dX| rms {np.sqrt(np.mean(dR**2 + dZ**2)):.2e} m: tangential {np.sqrt(np.mean(tang**2)):.2e}, "
          f"normal {np.sqrt(np.mean(norm**2)):.2e}; omega rms {np.sqrt(np.mean(om**2)):.2e}; relabel fit {fit:+.3f}; "
          f"QS rel {abs(qs_cost(e1) - qs_cost(e0)) / qs_cost(e0):.2e}; iota max {di:.2e}", flush=True)


def force_profile(label, eq):
    rho = np.linspace(0.05, 1.0, 20)
    g = LinearGrid(rho=rho, M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    F = np.asarray(eq.compute("|F|", grid=g)["|F|"])
    r = np.asarray(g.nodes[:, 0])
    prof = np.array([np.sqrt(np.mean(F[r == rr] ** 2)) for rr in np.unique(r)])
    io = iota(eq)
    rho_i = np.linspace(0.05, 1.0, 20)
    cross = [f"{0.5 * (rho_i[j] + rho_i[j + 1]):.2f}" for j in range(len(io) - 1) if (io[j] - 1.2) * (io[j + 1] - 1.2) < 0]
    print(f"{label:<34} |F| rms by rho (0.05..1): " + " ".join(f"{v:.1e}" for v in prof) +
          f"   iota = 1.2 near rho {cross}", flush=True)


base, _ = S.load_equilibrium(pbase)
trial, _ = S.load_equilibrium(ptrial)
summary("BASE", base)
summary("TRIAL (its own boundary)", trial)
dRb = np.linalg.norm(np.asarray(trial.Rb_lmn) - np.asarray(base.Rb_lmn)) + np.linalg.norm(
    np.asarray(trial.Zb_lmn) - np.asarray(base.Zb_lmn))
print(f"boundary difference |dRb|+|dZb| {dRb:.3e}")

Tp, msg = sn(with_boundary(trial, base))
summary(f"T' = TRIAL interior, BASE bdry [{msg}]", Tp)
Bp, msg = sn(with_boundary(base, trial))
summary(f"B' = BASE interior, TRIAL bdry [{msg}]", Bp)
diff("T' vs BASE (same boundary)", base, Tp)
diff("B' vs TRIAL (same boundary)", trial, Bp)
force_profile("BASE", base)
force_profile("T'", Tp)

print(f"--- raise both to basis {RES_UP} and re-solve", flush=True)
ups = []
for lab, e in (("BASE", base), ("T'", Tp)):
    u = e.copy()
    u.change_resolution(L=RES_UP, M=RES_UP, N=RES_UP, L_grid=2 * RES_UP, M_grid=2 * RES_UP, N_grid=2 * RES_UP)
    u, msg = sn(u, maxiter=600)
    summary(f"{lab} at basis {RES_UP} [{msg}]", u)
    ups.append(u)
diff(f"T' vs BASE at basis {RES_UP}", ups[0], ups[1])
force_profile(f"BASE at basis {RES_UP}", ups[0])
