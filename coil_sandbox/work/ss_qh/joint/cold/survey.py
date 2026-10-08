"""Survey QH equilibria as cold-start candidates for the z = 0 joint plane (sandbox root, CPU)."""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.grid import LinearGrid as LG
from desc.objectives import QuasisymmetryBoozer
for name in sys.argv[1:]:
    eq, kind = S.load_equilibrium(name)
    d = eq.compute(["p", "current"])
    vac = np.max(np.abs(d["p"])) < 1e-8 and np.max(np.abs(d["current"])) < 1e-8
    iota = np.asarray(eq.compute("iota", grid=LG(rho=np.linspace(0.05, 1, 20), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP))["iota"])
    R0, a = float(eq.compute("R0")["R0"]), float(eq.compute("a")["a"])
    qs = []
    for rho in (0.5, 1.0):
        try:
            o = QuasisymmetryBoozer(eq, helicity=(1, int(np.sign(iota.mean())) * eq.NFP), grid=LG(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP))
            o.build(verbose=0)
            B = float(np.mean(np.asarray(eq.compute("|B|", grid=LG(rho=np.array([rho]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP))["|B|"])))
            qs.append(np.linalg.norm(np.asarray(o.compute(eq.params_dict))) / B)
        except Exception as e:
            qs.append(np.nan)
    zc, strad = [], []
    for ph in np.linspace(0, np.pi / eq.NFP, 9):
        g = LG(theta=np.linspace(0, 2 * np.pi, 361), zeta=np.array([ph]), NFP=1)
        Z = np.asarray(eq.surface.compute("Z", grid=g)["Z"])
        zc.append((Z.max() + Z.min()) / 2); strad.append(Z.max() > 0 > Z.min())
    zc = np.abs(zc).max()
    print(f"{name:12s} NFP {eq.NFP} sym {eq.sym} vacuum {vac} R0 {R0:.3f} a {a:.4f} A {R0 / a:.2f} iota [{iota.min():.3f}, {iota.max():.3f}] "
          f"QS(1,±NFP) rho .5 {qs[0]:.2e} rho 1 {qs[1]:.2e} | max |z-centre| {zc * 1e3:.0f} mm = {zc / a:.2f} a, straddles z=0 {np.mean(strad):.2f}"
          f" | M,N {eq.M},{eq.N}", flush=True)
