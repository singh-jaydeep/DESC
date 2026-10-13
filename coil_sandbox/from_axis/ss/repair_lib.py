"""repair(eq): same boundary surface, equal-arclength poloidal angle, interior re-solved from scratch by continuation.

Why: B.n cannot see a re-parametrization of the boundary, so a free-boundary solve drifts along that freedom until the
poloidal angle stalls at a point (|e_theta x e_zeta| -> 0: a mm-scale kink with a 10-20% B.n spike and a negative
Jacobian at the edge that the coils' field lines do not have; ss/tip.py, ss/tip_trace.py). Measured on li_qa2_iota0.30
at shell 6: the surface moves 0.14 mm mean / 1.7 mm max, B.n max 20% -> 0.7% (rms unchanged), sqrt(g) min/median at
rho = 1 from -0.05 to +0.59, QS within 3-7%; one more shell without a repair brings the kink back.
"""
import numpy as np
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid


def boundary_health(obj, NFP):
    """(smallest area element / median, largest principal curvature [1/m]) of the rho = 1 surface."""
    g = LinearGrid(rho=np.array([1.0]), M=64, N=64, NFP=NFP, sym=False)
    d = obj.compute(["|e_theta x e_zeta|", "curvature_k1_rho", "curvature_k2_rho"], grid=g)
    ar = np.asarray(d["|e_theta x e_zeta|"])
    k = np.maximum(np.abs(np.asarray(d["curvature_k1_rho"])), np.abs(np.asarray(d["curvature_k2_rho"])))
    return float(ar.min() / np.median(ar)), float(k.max())


def jac_min(eq, rhos=(0.8, 0.95, 1.0)):
    out = []
    for rho in rhos:
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        out.append(float(s.min() / np.median(s)))
    return out


def arclength_surface(eq, MN=None, nt=96, nz=96):
    """The eq's boundary refitted with an equal-arclength poloidal angle in every zeta plane (origin: the old theta = 0
    curve, which keeps stellarator symmetry). Returns (surface, max distance from the old surface [m])."""
    MN = MN or eq.M
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, nz, endpoint=False)
    tf = np.linspace(0, 2 * np.pi, 4001)
    tn = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    pts, th, old = [], [], []
    for z in zeta:
        d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([1.0]), theta=tf, zeta=np.array([z]), NFP=eq.NFP))
        R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(R), np.diff(Z)))])
        s = 2 * np.pi * s / s[-1]
        pts.append(np.column_stack([np.interp(tn, s, R), np.full(nt, z), np.interp(tn, s, Z)]))
        th.append(tn)
        old.append((R, Z))
    surf = FourierRZToroidalSurface.from_values(np.concatenate(pts), np.concatenate(th), M=MN, N=MN, NFP=eq.NFP, sym=eq.sym)
    err = 0.0
    for z, (R, Z) in list(zip(zeta, old))[:: max(1, nz // 16)]:
        d = surf.compute(["R", "Z"], grid=LinearGrid(rho=np.array([1.0]), theta=tn, zeta=np.array([z]), NFP=eq.NFP))
        dd = np.hypot(np.asarray(d["R"])[:, None] - R[None], np.asarray(d["Z"])[:, None] - Z[None])
        err = max(err, float(dd.min(1).max()))
    return surf, err


def repair(eq, chunk=200, verbose=0, say=print):
    """Returns (new_eq, info). The new equilibrium has the resolution, flux and profiles of `eq`."""
    a0, k0 = boundary_health(eq, eq.NFP)
    j0 = jac_min(eq)
    surf, err = arclength_surface(eq)
    new = Equilibrium(Psi=float(eq.Psi), NFP=eq.NFP, sym=eq.sym, L=eq.L, M=eq.M, N=eq.N, L_grid=eq.L_grid,
                      M_grid=eq.M_grid, N_grid=eq.N_grid, surface=surf, pressure=eq.pressure, current=eq.current,
                      iota=eq.iota)
    new = solve_continuation_automatic(new, verbose=verbose, jac_chunk_size=chunk)[-1]
    a1, k1 = boundary_health(new, eq.NFP)
    j1 = jac_min(new)
    info = dict(surface_moved_max_mm=err * 1e3, area_min_over_median=[a0, a1], kappa_max=[k0, k1],
                jac_min_over_median_rho_0p8_0p95_1=[j0, j1])
    say(f"REPAIR: surface moved <= {err * 1e3:.2f} mm; area element min/median {a0:.3f} -> {a1:.3f}; max curvature "
        f"{k0:.0f} -> {k1:.0f} /m; sqrt(g) min/median at rho .8/.95/1 {['%+.2f' % v for v in j0]} -> {['%+.2f' % v for v in j1]}")
    return new, info
