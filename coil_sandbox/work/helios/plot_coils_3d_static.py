"""Static companion to plot_coils_3d.py: the main coilset from three viewpoints (matplotlib).

    python work/helios/plot_coils_3d_static.py OUT.png path.h5 [title]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import _independent_coil_indices  # noqa: E402

PAL = [("#1f5fa8", "#7fb0e8"), ("#c2521d", "#f2a47a"), ("#16825b", "#7fd3b0")]
NARC = 200
out, path = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(path)
eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
cs = load(path)
s = np.linspace(0, 2 * np.pi, 2 * NARC + 1)
pos = np.asarray(cs._compute_position(grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False), basis="xyz"))
uniq = [int(i) for i in _independent_coil_indices(cs)]
per = pos.shape[0] // len(uniq)
gb = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, 40), zeta=np.linspace(0, 2 * np.pi, 160), NFP=1)
xb = np.asarray(eq.compute("x", grid=gb, basis="xyz")["x"]).reshape(160, 40, 3)

views = [("from above", 88, -90), ("oblique", 28, -55), ("side, one half period in front", 2, -45)]
fig = plt.figure(figsize=(18, 6.6))
for k, (lab, el, az) in enumerate(views):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.plot_surface(xb[..., 0], xb[..., 1], xb[..., 2], color="#d8d6cf", alpha=0.35, linewidth=0, shade=True)
    order = sorted(range(pos.shape[0]), key=lambda r: r in uniq)  # unique coils drawn last (on top)
    for r in order:
        u = min(r // per, len(uniq) - 1)
        base = r in uniq
        for a in range(2):
            seg = pos[r, a * NARC:(a + 1) * NARC + 1]
            ax.plot(*seg.T, color=PAL[u % 3][a], lw=3.2 if base else 1.1, alpha=1.0 if base else 0.6, zorder=5 if base else 2)
    H = np.array([pos[r, [0, NARC]] for r in uniq]).reshape(-1, 3)
    ax.scatter(*H.T, color="black", s=28, zorder=6, depthshade=False)
    ax.view_init(el, az)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_axis_off()
    ax.set_title(lab, loc="left", fontsize=10)
handles = [plt.Line2D([], [], color=PAL[u][a], lw=3, label=f"coil {u} arc {a}") for u in range(3) for a in range(2)]
handles.append(plt.Line2D([], [], color="black", marker="o", ls="", label="hinges"))
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
fig.suptitle(title + "  (thick: the 3 unique coils; thin: symmetry copies)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out, dpi=110)
print("wrote", out)
