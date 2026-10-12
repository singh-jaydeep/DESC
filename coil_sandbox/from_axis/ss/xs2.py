import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from desc.io import load
from desc.grid import LinearGrid
eq = load(sys.argv[1]); out = sys.argv[2]
zs = [float(v) for v in sys.argv[3].split(",")]
fig, axs = plt.subplots(1, len(zs), figsize=(5 * len(zs), 5))
for ax, z in zip(axs, zs):
    g = LinearGrid(rho=np.linspace(0.02, 1, 50), theta=np.linspace(0, 2*np.pi, 361), zeta=np.array([z]), NFP=eq.NFP)
    d = eq.compute(["R", "Z", "sqrt(g)"], grid=g)
    R = np.asarray(d["R"]).reshape(50, 361, order="F") if False else None
    R, Z, s = (np.asarray(d[k]) for k in ("R", "Z", "sqrt(g)"))
    rho = g.nodes[:, 0]
    for r in np.unique(rho)[::5].tolist() + [1.0]:
        m = rho == r; ax.plot(R[m], Z[m], lw=0.6, color="k")
    neg = s < 0
    ax.scatter(R[neg], Z[neg], s=2, color="r")
    ax.set_aspect("equal"); ax.set_title(f"zeta {z:.2f}: {neg.mean()*100:.1f}% neg")
fig.savefig(out, dpi=110, bbox_inches="tight")
