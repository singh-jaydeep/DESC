import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from desc.io import load
from desc.grid import LinearGrid
eq = load(sys.argv[1]); out = sys.argv[2]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, z in zip(axs, np.linspace(0, np.pi / eq.NFP, 4)):
    for rho in np.linspace(0.1, 1, 10):
        g = LinearGrid(rho=np.array([rho]), theta=np.linspace(0, 2*np.pi, 400), zeta=np.array([z]), NFP=eq.NFP)
        d = eq.compute(["R", "Z"], grid=g)
        ax.plot(d["R"], d["Z"], lw=1.5 if rho == 1 else 0.6, color="k" if rho == 1 else "C0")
    for th in np.linspace(0, 2*np.pi, 24, endpoint=False):
        g = LinearGrid(rho=np.linspace(0, 1, 30), theta=np.array([th]), zeta=np.array([z]), NFP=eq.NFP)
        d = eq.compute(["R", "Z"], grid=g)
        ax.plot(d["R"], d["Z"], lw=0.4, color="C3")
    ax.set_aspect("equal"); ax.set_title(f"zeta {z:.2f}")
fig.savefig(out, dpi=110, bbox_inches="tight")
