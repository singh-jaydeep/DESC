"""Where does the polar-arc basis fail to follow the spline optimum?

For each point of the spline optimum, the DISTANCE TO THE NEAREST POINT of the polar
fit, as a function of the spline's own parameter, with the hinges marked. It must be a
geometric distance: the two classes parameterize the curve differently, so comparing
them pointwise at equal s measures the reparameterization, not the shape error (it does
not fall with M -- checked).
If the handoff's structural claim is right -- every shape mode vanishes at a hinge, at
any M -- the error of a fit to a corner-localized deformation should pile up AT the
hinge and refuse to fall with M there, while falling normally elsewhere.
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
from scipy.spatial import cKDTree  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.io import load  # noqa: E402

MS = [3, 5, 8, 10]
spl = list(S.iter_unique(load("work/helios/spline/stage2_n32/result.h5")))
fits = {M: list(S.iter_unique(load(f"work/helios/spline/refit/polarB2_M{M}.h5"))) for M in MS}
s = np.linspace(0, 2 * np.pi, 2001, endpoint=False)
g = Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False)
X = lambda c: np.asarray(c.compute("x", grid=g, basis="xyz")["x"])  # noqa: E731

fig, axes = plt.subplots(1, len(spl), figsize=(6.2 * len(spl), 4.6), squeeze=False)
print(f"{'coil':>5} {'M':>3} {'max dev':>9} {'dev within 0.3 rad of a hinge':>31} {'dev elsewhere':>15}")
for i in range(len(spl)):
    xs = X(spl[i])
    ax = axes[0][i]
    brk = S.coil_breaks(spl[i])
    near = np.zeros_like(s, dtype=bool)
    for b in brk:
        near |= np.abs(np.mod(s - b + np.pi, 2 * np.pi) - np.pi) < 0.3
    for M, col in zip(MS, ("#c9c6bf", "#8a5cd6", "#eb6834", "#2a78d6")):
        d = cKDTree(X(fits[M][i])).query(xs)[0]
        ax.plot(s, d, color=col, lw=1.8, label=f"M={M}")
        print(f"{i:>5} {M:>3} {d.max():>8.3f}m {d[near].max():>30.3f}m {d[~near].max():>14.3f}m")
    for b in brk:
        ax.axvline(b, color="#111", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("s")
    ax.set_ylabel("distance to the polar fit [m]")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"coil {i} (dotted: hinges)", loc="left")
fig.suptitle("Polar-arc B=2 fits to the spline optimum: where the basis cannot follow", x=0.005, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("work/helios/spline/refit_error.png", dpi=115)
print("\nwrote work/helios/spline/refit_error.png")
