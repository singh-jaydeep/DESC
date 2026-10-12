"""Rebuild a clean (positive-Jacobian) fixed-boundary equilibrium on a (truncated) free-boundary surface.
    python fixjac.py EQ.h5 OUT.h5 --kb 8 --LMN 12
"""
import sys, argparse
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction

ap = argparse.ArgumentParser()
ap.add_argument("eq"); ap.add_argument("out")
ap.add_argument("--kb", type=int, default=99, help="keep boundary modes with max(|m|,|n|) <= kb")
ap.add_argument("--LMN", type=int, default=12)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
eq = load(a.eq)
surf = eq.surface.copy()
for b, c in ((surf.R_basis, "R_lmn"), (surf.Z_basis, "Z_lmn")):
    v = np.array(getattr(surf, c)); v[np.max(np.abs(b.modes[:, 1:]), 1) > a.kb] = 0; setattr(surf, c, v)
eq.change_resolution(L=a.LMN, M=a.LMN, N=a.LMN, L_grid=2 * a.LMN, M_grid=2 * a.LMN, N_grid=2 * a.LMN)
surf.change_resolution(M=a.LMN, N=a.LMN)
eq.surface = surf
eq.set_initial_guess(surf)   # fresh interior from the boundary (no inherited overlaps)
eq.axis = eq.get_axis()


def jac(eq, tag):
    out = []
    for rho in (0.5, 0.9, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        out.append(f"rho {rho} min/med {s.min() / np.median(s):+.3f}")
    print(tag, "  ".join(out), flush=True)


jac(eq, "initial guess")
eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)), verbose=1, maxiter=300, ftol=1e-10,
         xtol=1e-12, gtol=1e-10)
jac(eq, "solved")
o = ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)); o.build(verbose=0)
print("force cost", 0.5 * float(np.sum(np.asarray(o.compute_scaled_error(o.x(eq))) ** 2)))
eq.save(a.out)
