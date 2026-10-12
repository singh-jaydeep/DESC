"""Validate a single-stage state: Jacobian folds, Boozer QS, iota, B.n, and a Poincare plot of the COILS' field with
the equilibrium surfaces overlaid.   python val.py EQ.h5 COILS.h5 OUT.png"""
import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np, matplotlib; matplotlib.use("Agg")
from desc.io import load
from desc.grid import LinearGrid
from desc.plotting import poincare_plot, plot_surfaces
import common as S
eq, cs, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
src = S.source_grid(cs, gl_m=64)[0]
fr = []
for rho in (0.5, 0.8, 0.9, 1.0):
    g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
    s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
    fr.append(f"{rho}:{s.min()/np.median(s):+.3f}({(s<0).mean()*100:.1f}%)")
print("sqrt(g) min/median (neg %)", " ".join(fr), flush=True)
fig, ax = plot_surfaces(eq)
gt = LinearGrid(rho=np.linspace(0.1, 1.1, 11), theta=np.array([0.0]), zeta=np.array([0.0]), NFP=1)
dt = eq.compute(["R", "Z"], grid=gt)
fig, ax = poincare_plot(cs, np.asarray(dt["R"]), np.asarray(dt["Z"]), ntransit=150, NFP=eq.NFP, ax=ax, grid=src, size=1.5)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
