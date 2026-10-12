"""Raise resolution of an existing equilibrium (L, M=N separately), re-solve, report Jacobian.
    python fixjac2.py EQ.h5 OUT.h5 --L 24 --MN 12
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
ap.add_argument("--L", type=int, default=24); ap.add_argument("--MN", type=int, default=12)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
eq = load(a.eq)


def jac(eq, tag):
    out = []
    for rho in (0.5, 0.8, 0.9, 0.95, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        out.append(f"{rho}:{s.min() / np.median(s):+.3f}")
    print(tag, "sqrt(g) min/median", "  ".join(out), flush=True)


jac(eq, "in ")
eq.change_resolution(L=a.L, M=a.MN, N=a.MN, L_grid=2 * a.L, M_grid=2 * a.MN, N_grid=2 * a.MN)
eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)), verbose=0, maxiter=300, ftol=1e-10,
         xtol=1e-12, gtol=1e-10)
jac(eq, f"L{a.L} M{a.MN}")
o = ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)); o.build(verbose=0)
print("force cost", 0.5 * float(np.sum(np.asarray(o.compute_scaled_error(o.x(eq))) ** 2)), flush=True)
eq.save(a.out)
