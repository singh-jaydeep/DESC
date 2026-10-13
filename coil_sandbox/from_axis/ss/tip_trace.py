"""Is a B.n spike on a free-boundary surface a real corner of the coils' flux surface or a crease in DESC's boundary?
Trace the coils' field lines started near the edge and overlay their punctures on the DESC cross-section through the
spike.   python tip_trace.py EQ.h5 COILS.h5 ZETA R Z OUT.png"""
import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from desc.io import load
from desc.grid import LinearGrid
from desc.magnetic_fields import field_line_integrate
import common as S
eq, cs = load(sys.argv[1]), load(sys.argv[2])
z0, Rs, Zs, out = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
src = S.source_grid(cs, gl_m=64)[0]
rhos = [0.8, 0.9, 0.95, 0.98, 1.0]
R0, Z0 = [], []
for r in rhos:  # start on the outboard side of the same plane, away from the spike
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([r]), theta=np.array([np.pi]), zeta=np.array([z0]), NFP=eq.NFP))
    R0.append(float(d["R"][0])); Z0.append(float(d["Z"][0]))
dR, dZ = R0[-1] - R0[-2], Z0[-1] - Z0[-2]
R0 += [R0[-1] + dR, R0[-1] + 2 * dR]; Z0 += [Z0[-1] + dZ, Z0[-1] + 2 * dZ]; labels = rhos + [1.02, 1.04]
NP = 400
phis = z0 + np.arange(NP + 1) * 2 * np.pi / eq.NFP
r, z = field_line_integrate(np.array(R0), np.array(Z0), phis, cs, source_grid=src, rtol=1e-9, atol=1e-9,
                            bounds_R=(0.35, 1.3), bounds_Z=(-0.5, 0.5), max_steps=4000 * NP, chunk_size=len(R0))
r, z = np.asarray(r), np.asarray(z)
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
for ax, zoom in zip(axs, (None, 0.03)):
    for rho in (0.8, 0.9, 0.95, 1.0):
        d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([rho]), theta=np.linspace(0, 2 * np.pi, 2000), zeta=np.array([z0]), NFP=eq.NFP))
        ax.plot(d["R"], d["Z"], "k-", lw=1.6 if rho == 1 else 0.6, label="DESC rho = 1" if rho == 1 else None)
    for j, lab in enumerate(labels):
        ok = np.isfinite(r[:, j])
        ax.plot(r[ok, j], z[ok, j], ".", ms=2.5, label=f"coil field line from rho {lab:g} ({ok.sum()} punctures)")
    ax.plot([Rs], [Zs], "r+", ms=18, mew=2, label="B.n spike")
    ax.set_aspect("equal")
    if zoom:
        ax.set_xlim(Rs - zoom, Rs + zoom); ax.set_ylim(Zs - zoom, Zs + zoom); ax.set_title(f"zoom: +-{zoom * 1e3:.0f} mm around the spike")
    else:
        ax.set_title(f"zeta = {z0:.3f}"); ax.legend(fontsize=7, loc="upper left")
fig.savefig(out, dpi=120, bbox_inches="tight")
for j, lab in enumerate(labels):
    ok = np.isfinite(r[:, j]); print(f"line from rho {lab:g}: {int(ok.sum())}/{NP + 1} punctures, closest approach to the spike {np.min(np.hypot(r[ok, j] - Rs, z[ok, j] - Zs)) * 1e3:.2f} mm")
print("wrote", out)
