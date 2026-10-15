"""Where is the coils' B.n large on a free-boundary surface, and does the surface fold there?
    python tip.py EQ.h5 COILS.h5 OUT.png     (CPU is fine)"""
import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from desc.io import load
from desc.grid import LinearGrid
import common as S
eq, cs, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
src = S.source_grid(cs, gl_m=64)[0]
g = LinearGrid(rho=np.array([1.0]), M=64, N=64, NFP=eq.NFP, sym=False)
d = eq.compute(["x", "n_rho", "|e_theta x e_zeta|", "sqrt(g)", "R", "Z", "curvature_k1_rho", "curvature_k2_rho"], grid=g, basis="xyz")
B = np.asarray(cs.compute_magnetic_field(np.asarray(d["x"]), basis="xyz", source_grid=src))
w = np.asarray(g.weights) * np.asarray(d["|e_theta x e_zeta|"]); w /= w.sum()
Bm = np.linalg.norm(B, axis=1); Bavg = np.sum(w * Bm)
bn = np.abs(np.sum(B * np.asarray(d["n_rho"]), 1)) / Bavg
sg = np.asarray(d["sqrt(g)"]); med = np.median(sg)
kmax = np.maximum(np.abs(np.asarray(d["curvature_k1_rho"])), np.abs(np.asarray(d["curvature_k2_rho"])))
th, ze = g.nodes[:, 1], g.nodes[:, 2]
print(f"B.n/<B>: mean {np.sum(w * bn):.2e} rms {np.sqrt(np.sum(w * bn**2)):.2e} max {bn.max():.2e}; area above 1%: {np.sum(w[bn > 0.01]) * 100:.3f}%")
print(f"sqrt(g) at rho=1: min/median {sg.min() / med:+.3f}; negative on {100 * (sg < 0).mean():.2f}% of points; below 20% of median on {100 * (sg < 0.2 * med).mean():.2f}%")
for rho in (0.8, 0.9, 0.95):
    gi = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
    s2 = np.asarray(eq.compute("sqrt(g)", grid=gi)["sqrt(g)"])
    print(f"  rho {rho}: sqrt(g) min/median {s2.min() / np.median(s2):+.3f}, negative on {100 * (s2 < 0).mean():.2f}%")
print("largest B.n points (theta, zeta, R, Z, B.n/<B>, sqrt(g)/median, max principal curvature 1/m):")
for j in np.argsort(-bn)[:8]:
    print(f"   {th[j]:.3f} {ze[j]:.3f}  R {d['R'][j]:.4f} Z {d['Z'][j]:+.4f}  {bn[j]:.3e}  {sg[j] / med:+.3f}  {kmax[j]:.1f}")
print(f"max principal curvature of the boundary: {kmax.max():.1f} /m (1/a = {1 / float(eq.compute('a')['a']):.1f}); at the B.n max: {kmax[np.argmax(bn)]:.1f}")
far = sg > 0.2 * med
print(f"B.n max away from sqrt(g) < 20% median: {bn[far].max():.2e}")
# picture: the cross-section through the worst point, flux surfaces + where B.n > 0.5%
zs = [float(ze[np.argmax(bn)]), 0.0, np.pi / eq.NFP / 2, np.pi / eq.NFP]
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
for ax, z in zip(axs, zs):
    for rho in np.linspace(0.2, 1, 9):
        gg = LinearGrid(rho=np.array([rho]), theta=np.linspace(0, 2 * np.pi, 400), zeta=np.array([z]), NFP=eq.NFP)
        dd = eq.compute(["R", "Z"], grid=gg)
        ax.plot(dd["R"], dd["Z"], "k-", lw=1.4 if rho == 1 else 0.5)
    m = np.isclose(ze, ze[np.argmin(np.abs(ze - z))])
    sc = ax.scatter(np.asarray(d["R"])[m], np.asarray(d["Z"])[m], c=np.log10(bn[m] + 1e-6), s=14, cmap="viridis", vmin=-4, vmax=-1)
    ax.set_aspect("equal"); ax.set_title(f"zeta {z:.2f}  max B.n here {bn[m].max():.1e}")
fig.colorbar(sc, ax=axs, label="log10 |B.n| / <B>")
fig.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out)
