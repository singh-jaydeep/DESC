"""WISTELL-A as a cold start for the joint plane: mirrored to positive iota and scaled to precise_QH's minor radius.

    python work/ss_qh/joint/cold/prep_wistell.py      (sandbox root, CPU)  -> work/ss_qh/joint/cold/eq_wistell.h5

Both maps are exact (no re-solve): mirror zeta -> -zeta (negate every R, Z, lambda, boundary mode with n < 0;
iota changes sign, helicity (1,-4) -> (1,4)); scale lengths by c = a_QH / a_W (R, Z modes x c, Psi x c^2 keeps |B|).
Checks: iota sign, force residual unchanged, a = precise_QH's.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.grid import LinearGrid as LG

eqW, _ = S.load_equilibrium("WISTELL-A")
eqQ, _ = S.load_equilibrium("precise_QH")
c = float(eqQ.compute("a")["a"]) / float(eqW.compute("a")["a"])


def fnorm(eq):
    g = LG(rho=np.linspace(0.2, 1, 9), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["|F|", "|grad(p)|", "|B|"], grid=g)
    return float(np.mean(np.asarray(d["|F|"])))


eq = eqW.copy()
for name, basis in (("R_lmn", eq.R_basis), ("Z_lmn", eq.Z_basis), ("L_lmn", eq.L_basis)):
    v = np.asarray(getattr(eq, name)).copy()
    v[basis.modes[:, 2] < 0] *= -1
    if name != "L_lmn":
        v *= c
    setattr(eq, name, v)
surf = eq.surface
for name, basis in (("R_lmn", surf.R_basis), ("Z_lmn", surf.Z_basis)):
    v = np.asarray(getattr(surf, name)).copy()
    v[basis.modes[:, 2] < 0] *= -1
    setattr(surf, name, v * c)
eq.surface = surf
eq.Psi = float(eqW.Psi) * c**2
ax = eq.axis
for name, basis in (("R_n", ax.R_basis), ("Z_n", ax.Z_basis)):
    v = np.asarray(getattr(ax, name)).copy()
    v[basis.modes[:, 2] < 0] *= -1
    setattr(ax, name, v * c)
eq.axis = ax
for lab, e in (("WISTELL-A", eqW), ("mirrored+scaled", eq)):
    iota = np.asarray(e.compute("iota", grid=LG(rho=np.linspace(0.05, 1, 20), M=e.M_grid, N=e.N_grid, NFP=e.NFP))["iota"])
    B = float(np.mean(np.asarray(e.compute("|B|", grid=LG(rho=np.array([0.5]), M=e.M_grid, N=e.N_grid, NFP=e.NFP))["|B|"])))
    print(f"{lab:16s} R0 {float(e.compute('R0')['R0']):.4f} a {float(e.compute('a')['a']):.4f} iota [{iota.min():.4f}, {iota.max():.4f}] "
          f"<|B|> {B:.4f}  mean |F| {fnorm(e):.3e}  V {float(e.compute('V')['V']):.4f}  L,M,N {e.L},{e.M},{e.N}", flush=True)
print(f"scale c = {c:.5f}")
eq.save("work/ss_qh/joint/cold/eq_wistell.h5")
print("wrote work/ss_qh/joint/cold/eq_wistell.h5")
