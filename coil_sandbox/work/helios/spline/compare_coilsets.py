"""Overlay two coilsets on the Helios boundary, and compare them coil by coil.

    python work/helios/spline/compare_coilsets.py OUT.png ARC.h5 SPLINE.h5

Top row: three viewpoints, all 12 coils of each set over the plasma boundary.
Bottom row: each unique coil in its own plane, with the C0 corners marked and the
displacement between the two profiles.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
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

out, p_arc, p_spl = sys.argv[1], sys.argv[2], sys.argv[3]
A, B = load(p_arc), load(p_spl)
CA, CB = "#1f5fa8", "#d1541c"          # arc, spline
LA = os.path.basename(p_arc).replace(".h5", "")
LB = os.path.basename(p_spl).replace(".h5", "")

s = np.linspace(0, 2 * np.pi, 801)
g = Grid(np.stack([0 * s, 0 * s, np.mod(s, 2 * np.pi)], 1), sort=False, jitable=False)
posA = np.asarray(A._compute_position(grid=g, basis="xyz"))
posB = np.asarray(B._compute_position(grid=g, basis="xyz"))

eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
gb = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2 * np.pi, 48),
                zeta=np.linspace(0, 2 * np.pi, 192), NFP=1)
xb = np.asarray(eq.compute("x", grid=gb, basis="xyz")["x"]).reshape(192, 48, 3)

uA, uB = list(S.iter_unique(A)), list(S.iter_unique(B))
nc = len(uA)
fig = plt.figure(figsize=(6.0 * max(3, nc), 11.5))

for k, (lab, el, az) in enumerate([("from above", 88, -90), ("oblique", 26, -55), ("side", 4, -45)]):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")
    ax.plot_surface(xb[..., 0], xb[..., 1], xb[..., 2], color="#c9c6bf", alpha=0.30,
                    linewidth=0, antialiased=False, shade=True)
    for p, c in ((posA, CA), (posB, CB)):
        for q in p:
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color=c, lw=1.7)
    ax.view_init(elev=el, azim=az)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_axis_off()
    ax.set_title(lab, loc="left", fontsize=12)
    if k == 0:
        ax.plot([], [], color=CA, lw=2.5, label=f"{LA} (arc B=2)")
        ax.plot([], [], color=CB, lw=2.5, label=f"{LB} (spline)")
        ax.legend(frameon=False, fontsize=10, loc="upper left")

for i in range(nc):
    ax = fig.add_subplot(2, nc, nc + i + 1)
    xa = np.asarray(uA[i].compute("x", grid=g, basis="xyz")["x"])
    xb2 = np.asarray(uB[i].compute("x", grid=g, basis="xyz")["x"])
    ctr = xa.mean(0)
    nrm = np.linalg.svd(xa - ctr)[2][-1]
    ev = np.array([0.0, 0.0, 1.0]) - nrm[2] * nrm
    ev /= np.linalg.norm(ev)
    eh = np.cross(nrm, ev)
    for x, c, lw, z in ((xa, CA, 3.2, 2), (xb2, CB, 1.8, 3)):
        ax.plot((x - ctr) @ eh, (x - ctr) @ ev, color=c, lw=lw, zorder=z)
    brk = S.coil_breaks(uB[i])
    gb2 = Grid(np.stack([0 * brk, 0 * brk, brk], 1), sort=False, jitable=False)
    for x, c, m in ((np.asarray(uA[i].compute("x", grid=gb2, basis="xyz")["x"]), CA, "o"),
                    (np.asarray(uB[i].compute("x", grid=gb2, basis="xyz")["x"]), CB, "s")):
        ax.plot((x - ctr) @ eh, (x - ctr) @ ev, m, color=c, ms=9, mec="k", mew=0.8, zorder=5)
    d = np.linalg.norm(xb2 - xa, axis=1)
    mA, mB = S.curve_metrics(uA[i]), S.curve_metrics(uB[i])
    ax.set_aspect("equal")
    ax.set_title(f"coil {i}: max move {d.max():.2f} m\n"
                 f"$\\kappa$ {mA['kappa']:.2f} $\\to$ {mB['kappa']:.2f} /m, "
                 f"corner {mA['corner']:.0f}$\\to${mB['corner']:.0f}$\\degree$",
                 loc="left", fontsize=11)
    ax.set_xlabel("in-plane horizontal [m]")
    if i == 0:
        ax.set_ylabel("in-plane vertical [m]")

fig.suptitle(f"Helios encircling coils: {LA} (arc B=2) vs {LB} (SplineXYZ, C0 breaks at the hinges). "
             f"Markers: hinges / break knots.", x=0.005, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out, dpi=115)
print(f"wrote {out}")
