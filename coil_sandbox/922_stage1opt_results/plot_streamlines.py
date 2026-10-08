"""Can you SEE the difference between the two equilibria in their surface-current
streamlines on (theta, zeta) axes?  Short answer: not raw. This figure shows why.

Panel A  the streamlines as they actually are -- the two equilibria overlay almost exactly
Panel B  the same curves with the m=1 harmonic removed. m=1 is absorbed by TILTING the
         best-fit plane, so it costs nothing in eps_1; everything that matters is here
Panel C  the resulting out-of-plane residual, which is what eps_1 actually measures
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
from clamshell import best_breaks, freeze  # noqa: E402
from contour_curve import ContourClamshell  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.io import load  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"       # dataviz reference categorical slots 1, 2
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
CKW = dict(K=8, M=32, N=64, n=360, Mt=16, Nt=16)

eq0 = get("precise_QA")
eq1 = load("922_stage1opt_results/runs/prox_eps1_300/eq_lo.h5")


def extract(eq):
    o = ContourClamshell(eq, quantity="eps1", **CKW)
    o.build(verbose=0)
    xs, ws = o._chain(eq.params_dict, o._constants)
    th = np.asarray(o._constants["theta"])
    zetas, resids = [], []
    for k in range(CKW["K"]):
        X, W = np.asarray(xs[k]), np.asarray(ws[k])
        zetas.append(np.unwrap(np.arctan2(X[:, 1], X[:, 0])))
        fz = freeze(X, W, best_breaks(X, W, 1, stride=6)[1])
        D = X - (W[:, None] * X).sum(0) / W.sum()
        resids.append(D @ fz["normals"][0])
    return th, np.array(zetas), np.array(resids)


th, z0, r0 = extract(eq0)
_, z1, r1 = extract(eq1)


def strip_m1(z):
    """Remove mean and the m=1 harmonic -- the part a tilted fit plane absorbs."""
    out = np.empty_like(z)
    for k in range(z.shape[0]):
        d = z[k] - z[k].mean()
        F = np.fft.rfft(d)
        F[0] = 0
        F[1] = 0
        out[k] = np.fft.irfft(F, len(d))
    return out


thd = np.degrees(th)
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
for a in ax:
    a.grid(True, color=GRID, lw=0.6, alpha=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)
    a.set_xlim(0, 360)
    a.set_xticks([0, 90, 180, 270, 360])
    a.set_xlabel("poloidal angle $\\theta$  (deg)", color=MUTED, fontsize=10)

# ---- A: the streamlines as they are
for k in range(CKW["K"]):
    ax[0].plot(thd, np.degrees(z0[k]), color=BLUE, lw=1.6,
               label="precise_QA (baseline)" if k == 0 else None)
    ax[0].plot(thd, np.degrees(z1[k]), color=ORANGE, lw=1.6, ls=(0, (5, 3)),
               label="after $\\epsilon_1$ optimization" if k == 0 else None)
ax[0].set_ylabel("toroidal angle $\\zeta$  (deg)", color=MUTED, fontsize=10)
ax[0].set_title("A. The streamlines themselves\nthe two curves sit on top of each other",
                color=INK, fontsize=11, loc="left")
ax[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper right")

# ---- B: m=1 removed
s0, s1 = strip_m1(z0), strip_m1(z1)
for k in range(CKW["K"]):
    ax[1].plot(thd, np.degrees(s0[k]), color=BLUE, lw=1.6, alpha=0.85)
    ax[1].plot(thd, np.degrees(s1[k]), color=ORANGE, lw=1.6, ls=(0, (5, 3)), alpha=0.85)
ax[1].set_ylabel("$\\zeta$ with $m\\!=\\!1$ removed  (deg)", color=MUTED, fontsize=10)
ax[1].set_title("B. The part that actually costs planarity\n$m\\!=\\!1$ is absorbed by "
                "tilting the fit plane", color=INK, fontsize=11, loc="left")

# ---- C: out-of-plane residual
for k in range(CKW["K"]):
    ax[2].plot(thd, r0[k] * 1000, color=BLUE, lw=1.6, alpha=0.85)
    ax[2].plot(thd, r1[k] * 1000, color=ORANGE, lw=1.6, ls=(0, (5, 3)), alpha=0.85)
ax[2].set_ylabel("out-of-plane residual  (mm)", color=MUTED, fontsize=10)
ax[2].set_title(f"C. What $\\epsilon_1$ measures\nrms {np.sqrt((r0**2).mean())*1000:.2f} "
                f"$\\rightarrow$ {np.sqrt((r1**2).mean())*1000:.2f} mm", color=INK,
                fontsize=11, loc="left")

fig.suptitle("Surface-current streamlines on the $\\rho=1$ surface, before and after "
             "$\\epsilon_1$ optimization  (8 contours per field period)",
             color=INK, fontsize=12.5, x=0.005, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "922_stage1opt_results/streamlines_before_after.png"
fig.savefig(out, dpi=150, facecolor="white")
print("wrote", out)

# --- the numbers behind the question "can you see it?"
d = np.degrees(np.abs(z1 - z0)).max()
tot = np.degrees(np.ptp(z0, axis=1)).mean()
print(f"max |zeta_final - zeta_base|  {d:.4f} deg")
print(f"typical total wander           {tot:.4f} deg")
print(f"ratio                          {d/tot*100:.2f}%  of the curve's own excursion")
m10 = np.degrees(np.abs(s1 - s0)).max()
m1t = np.degrees(np.ptp(s0, axis=1)).mean()
print(f"with m=1 removed: difference {m10:.4f} deg vs structure {m1t:.4f} deg "
      f"-> {m10/m1t*100:.1f}%")
