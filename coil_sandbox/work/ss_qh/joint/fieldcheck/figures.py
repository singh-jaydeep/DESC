"""Figures for the coil-field QS write-up, per case: (1) Boozer |B| on rho 0.5 and 1, target (at rho x 0.95) vs coil
field (traced-surface equilibrium, res 12); (2) QS error vs rho for target, coil field, and the method control.
Usage: figures.py CASE EQ_TARGET"""
import os
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import desc.io  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.plotting import plot_boozer_surface  # noqa: E402
import common as S  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eqt_f = sys.argv[1:3]
eqt = S.load_equilibrium(eqt_f)[0]
eqs = desc.io.load(f"{F}/{case}/eq_traced_L12.h5")
ctl = f"{F}/{case}/control_L12.h5"
eqc = desc.io.load(ctl) if os.path.exists(ctl) else None

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for i, (e, tag, sc) in enumerate([(eqt, "target", 0.95), (eqs, "coil field", 1.0)]):
    for k, rv in enumerate([0.5, 1.0]):
        plot_boozer_surface(e, rho=rv * sc, ax=axs[i, k], fieldlines=4)
        axs[i, k].set_title(f"{case}: {tag}, rho {rv * sc:g}")
fig.tight_layout()
fig.savefig(f"{F}/{case}/boozer_L12.png", dpi=120)


def qs_profile(e, rhos):
    vals = []
    for r in rhos:
        o = O.QuasisymmetryBoozer(e, helicity=(1, e.NFP), normalize=False,
                                  grid=LinearGrid(rho=np.array([r]), M=2 * e.M_grid, N=2 * e.N_grid, NFP=e.NFP,
                                                  sym=False))
        o.build(verbose=0)
        B = float(e.compute("<|B|>_vol")["<|B|>_vol"])
        vals.append(float(np.linalg.norm(o.compute_unscaled(e.params_dict))) / B)
    return np.array(vals)


rh = np.linspace(0.1, 1.0, 10)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.semilogy(rh, qs_profile(eqt, 0.95 * rh), "k-o", ms=3, label="target (at rho x 0.95)")
if eqc is not None:
    ax.semilogy(rh, qs_profile(eqc, rh), "k--", label="method control (target surface through pipeline)")
qs_c = qs_profile(eqs, rh)
ax.semilogy(rh, qs_c, "r-o", ms=3, label="coil field (traced surface)")
ax.set_xlabel("rho (traced-surface equilibrium)")
ax.set_ylabel("QS error |B_nonsym| / <|B|>")
ax.set_title(f"{case}: quasisymmetry of the actual coil field")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{F}/{case}/qs_profile.png", dpi=130)
np.savetxt(f"{F}/{case}/qs_profile.txt", np.column_stack([rh, qs_c]), header="rho  QS_coilfield")
print("done")
