"""Cold-start candidate: precise_QH with its helical-axis excursion scaled down, re-solved (vacuum, zero current).

    python work/ss_qh/joint/cold/shrink_helix.py S [S ...]      (sandbox root, CPU)

The boundary's m = 0, n != 0 modes (R_0n, Z_0n) carry the helical axis (radius ~0.17 m > a), which puts the
cross-section up to 150 mm below z = 0. They are multiplied by S; everything else is kept; the equilibrium
is re-solved. Prints iota, A, Boozer QS and the vertical excursion; saves eq_helixS.h5.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S_
from desc.grid import LinearGrid
from desc.objectives import get_fixed_boundary_constraints, ForceBalance, ObjectiveFunction

D = "work/ss_qh/joint/cold"


def excursion(eq):
    zc = []
    for ph in np.radians(np.linspace(0, 45, 9)):
        g = LinearGrid(theta=np.linspace(0, 2 * np.pi, 361), zeta=np.array([ph]), NFP=1)
        Z = np.asarray(eq.surface.compute("Z", grid=g)["Z"])
        zc.append((Z.max() + Z.min()) / 2)
    return np.abs(zc).max(), float(np.mean([1]))


def report(lab, eq):
    import qs_scale_invariant  # noqa: F401  (path check only)
    from desc.grid import LinearGrid as LG
    iota = np.asarray(eq.compute("iota", grid=LG(rho=np.linspace(0.05, 1, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP))["iota"])
    A = float(eq.compute("R0/a")["R0/a"])
    qs = []
    for rho in (0.5, 1.0):
        g = LG(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP)
        d = eq.compute(["|B|_mn_B"] if False else ["|B|"], grid=g)
    from desc.objectives import QuasisymmetryBoozer
    for rho in (0.5, 1.0):
        o = QuasisymmetryBoozer(eq, helicity=(1, eq.NFP), grid=LG(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP))
        o.build(verbose=0)
        f = np.asarray(o.compute(eq.params_dict))
        B00 = float(np.mean(np.asarray(eq.compute("|B|", grid=LG(rho=np.array([rho]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP))["|B|"])))
        qs.append(np.linalg.norm(f) / B00)
    zx, _ = excursion(eq)
    print(f"{lab:14s} iota [{iota.min():.4f}, {iota.max():.4f}]  A {A:.3f}  QS |B_nonsym|/<B> rho 0.5 {qs[0]:.2e}  rho 1 {qs[1]:.2e}"
          f"  max |z-centre| {zx * 1e3:.0f} mm  V {float(eq.compute('V')['V']):.4f}", flush=True)


eq0, _ = S_.load_equilibrium("precise_QH")
report("precise_QH", eq0)
for s in [float(v) for v in sys.argv[1:]]:
    eq = eq0.copy()
    surf = eq.surface
    for name, basis in (("R_lmn", surf.R_basis), ("Z_lmn", surf.Z_basis)):
        c = np.asarray(getattr(surf, name)).copy()
        sel = (basis.modes[:, 1] == 0) & (basis.modes[:, 2] != 0)
        c[sel] *= s
        setattr(surf, name, c)
    eq.surface = surf
    eq.axis = eq.get_axis() if False else eq.axis
    eq.set_initial_guess(surf)  # interior from the NEW boundary (the old one is built around the old axis)
    _, res = eq.solve(verbose=0, ftol=1e-12, xtol=1e-12, gtol=1e-12, maxiter=500, copy=False)
    fb = float(np.max(np.abs(np.asarray(eq.compute("|F|_normalized" if False else "|F|", grid=LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10)))["|F|"]))))
    print(f"  solve: {res['message']} nit {res['nit']}, max |F| {fb:.2e}", flush=True)
    report(f"helix x{s}", eq)
    eq.save(f"{D}/eq_helix{s}.h5")
