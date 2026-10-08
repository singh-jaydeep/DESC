"""Static pictures of the planar N7 history on the fixed QH boundary: (1) the 4 unique coils in 3D
over one half period of the boundary, per stage; (2) each unique coil in its own plane, with kappa(s)."""
import sys; sys.path.insert(0, "sandbox")   # run from the sandbox root
import boot; boot.setup(default="cpu")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as S
from desc.io import load
from desc.grid import LinearGrid

P = "work/ss_qh/fixed_ss_eq"
STAGES = [("cold start", "../sweep_qh/initial starts/start_planarN7.h5"),
          ("it 50", f"{P}/planarN7/ckpt_it0050.h5"),
          ("it 100", f"{P}/planarN7/ckpt_it0100.h5"),
          ("it 150", f"{P}/planarN7/ckpt_it0150.h5"),
          ("150+50, tr inf", f"{P}/planarN7_trinf/ckpt_it0050.h5"),
          ("150+100, tr inf", f"{P}/planarN7_trinf/ckpt_it0100.h5")]
KMAX = 13.672824284132604  # bounds/precise_qh_gil.json

eq, _ = S.load_equilibrium("work/ss_qh/arcB2/ss_k4/eq_k4.h5")
sg = LinearGrid(rho=1.0, theta=48, zeta=np.linspace(0, np.pi / eq.NFP, 24), NFP=1)
sd = eq.compute(["X", "Y", "Z"], grid=sg)
SX, SY, SZ = (np.asarray(sd[k]).reshape(sg.num_zeta, sg.num_theta) for k in "XYZ")  # zeta-major

g = LinearGrid(N=400)
data = []
for label, path in STAGES:
    rows = []
    for c in S.iter_unique(load(path)):
        d = c.compute(["x", "curvature", "length"], grid=g, basis="xyz")
        x = np.asarray(d["x"])
        seg = np.linalg.norm(np.roll(x, -1, 0) - x, axis=1)
        rows.append(dict(x=x, k=np.asarray(d["curvature"]), L=float(d["length"]), s=np.concatenate([[0], np.cumsum(seg)[:-1]])))
    data.append((label, rows))
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# (1) 3D, one panel per stage
fig = plt.figure(figsize=(18, 11))
for i, (label, rows) in enumerate(data):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    ax.plot_surface(SX, SY, SZ, color="0.8", alpha=0.25, linewidth=0)
    for j, r in enumerate(rows):
        x = r["x"]
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color=cols[j], lw=1.6)
        m = np.argmax(np.abs(r["k"]))
        ax.scatter(*x[m], color=cols[j], s=30, marker="x")
    kk = max(np.abs(r["k"]).max() for r in rows)
    ax.set_title(f"{label}: max |κ| {kk:.1f}/m ({kk / KMAX:.2f}×bound)", fontsize=10)
    ax.view_init(elev=28, azim=-60)
    ax.set_box_aspect((1, 1, 0.5)); ax.set_axis_off()
fig.suptitle("planar N7, 4 unique coils over half a field period (× = |κ| peak)")
fig.tight_layout()
fig.savefig(f"{P}/planar_history_3d.png", dpi=110)

# (2) in-plane shape and kappa(s), one row per unique coil
fig, axs = plt.subplots(len(data[0][1]), 2, figsize=(13, 3.6 * len(data[0][1])))
shade = plt.cm.viridis(np.linspace(0, 0.9, len(data)))
for j in range(len(data[0][1])):
    # one fixed in-plane frame per coil (from the start), so shape changes are comparable
    x0 = data[0][1][j]["x"]; c0 = x0.mean(0)
    _, _, vt = np.linalg.svd(x0 - c0, full_matrices=False)
    for i, (label, rows) in enumerate(data):
        r = rows[j]
        c = r["x"].mean(0)
        _, _, v = np.linalg.svd(r["x"] - c, full_matrices=False)
        e1, e2 = v[0], v[1]
        if e1 @ vt[0] < 0: e1 = -e1
        if np.cross(e1, e2) @ np.cross(vt[0], vt[1]) < 0: e2 = -e2
        u = (r["x"] - c) @ np.stack([e1, e2], 1)
        axs[j, 0].plot(u[:, 0], u[:, 1], color=shade[i], lw=1.4, label=label)
        axs[j, 1].plot(r["s"], r["k"], color=shade[i], lw=1.2, label=f"{label} (L {r['L']:.2f} m)")
    axs[j, 0].set_aspect("equal"); axs[j, 0].set_title(f"coil {j}: shape in its own plane (m)")
    axs[j, 1].axhline(KMAX, color="k", ls="--", lw=1); axs[j, 1].axhline(-KMAX, color="k", ls="--", lw=1)
    axs[j, 1].set_title(f"coil {j}: signed κ(s) (< 0: concave), dashed = ±bound {KMAX:.2f}/m"); axs[j, 1].set_xlabel("arclength s (m)")
    axs[j, 1].legend(fontsize=7)
axs[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{P}/planar_history_shapes.png", dpi=100)
print("wrote", f"{P}/planar_history_3d.png", f"{P}/planar_history_shapes.png")
