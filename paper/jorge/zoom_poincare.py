"""Zoomed Poincare section (phi = 0) in bands of start offsets around suspected islands.

    python paper/jorge/zoom_poincare.py TRACE_DIR --band 0.085:0.115:0.0025 [--band ...]

TRACE_DIR is a paper/trace_lines.py output (coils.h5, score.json with axis_R_phi0).
Lines start on the outboard midplane at R = R_axis + offset, Z = 0. Per band: the R-Z
section, and the same points as (poloidal angle about the axis, distance from the axis),
where an island chain shows as a row of O's. Writes a PNG and an npz of the crossings.
Ported from coil_sandbox/paper_min/jorge/zoom_poincare.py (js/qa-poster).
"""

import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("trace_dir")
ap.add_argument("--band", action="append", required=True, help="lo:hi:step (m)")
ap.add_argument("--ntransit", type=int, default=600)
ap.add_argument("--rtol", type=float, default=1e-10)
ap.add_argument("--gl", type=int, default=32)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
a = ap.parse_args()

from desc import set_device  # noqa: E402

set_device(a.device)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from desc.coils import coil_source_grid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402

J = json.load(open(os.path.join(a.trace_dir, "score.json")))
Rax = J["axis_R_phi0"]
cs = load(os.path.join(a.trace_dir, "coils.h5"))
src = coil_source_grid(cs, n=a.gl)
name = os.path.basename(os.path.normpath(a.trace_dir))
out = a.out or os.path.join(a.trace_dir, "zoom.png")
bands = []
for b in a.band:
    lo, hi, st = (float(v) for v in b.split(":"))
    bands.append(np.round(np.arange(lo, hi + st / 2, st), 6))

phis = 2 * np.pi * np.arange(a.ntransit + 1)
fig, axs = plt.subplots(len(bands), 2, figsize=(15, 5.2 * len(bands)), squeeze=False)
save = {}
for k, offs in enumerate(bands):
    R0 = Rax + offs
    r, z = field_line_integrate(
        R0,
        np.zeros_like(R0),
        phis,
        cs,
        source_grid=src,
        rtol=a.rtol,
        atol=a.rtol,
        max_steps=int(2000 * phis[-1]),
        options={"throw": False},
    )
    r, z = np.asarray(r), np.asarray(z)
    save[f"band{k}_offs"], save[f"band{k}_R"], save[f"band{k}_Z"] = offs, r, z
    cols = plt.cm.viridis(np.linspace(0, 1, len(offs)))
    for j, o in enumerate(offs):
        ok = np.isfinite(r[:, j]) & np.isfinite(z[:, j])
        rr, zz = r[ok, j], z[ok, j]
        lab = f"{o:.4f}" + (f" (lost after {ok.sum()})" if (~ok).any() else "")
        axs[k, 0].plot(rr, zz, ".", ms=1.2, color=cols[j], label=lab)
        axs[k, 1].plot(
            np.arctan2(zz, rr - Rax), np.hypot(rr - Rax, zz), ".", ms=1.2, color=cols[j]
        )
        print(f"{name} band {k} offset {o:.4f}: kept {ok.sum()}/{len(ok)}", flush=True)
    axs[k, 0].plot([Rax], [0], "k+")
    axs[k, 0].set_xlabel("R (m)")
    axs[k, 0].set_ylabel("Z (m)")
    axs[k, 0].set_title(f"{name}: phi = 0, offsets {offs[0]:.4f}-{offs[-1]:.4f} m")
    axs[k, 0].legend(fontsize=6, markerscale=6, ncol=2, loc="upper left")
    axs[k, 1].set_xlabel("poloidal angle about axis (rad)")
    axs[k, 1].set_ylabel("distance from axis (m)")
fig.tight_layout()
fig.savefig(out, dpi=120)
np.savez(os.path.splitext(out)[0] + ".npz", Rax=Rax, **save)
print("wrote", out, flush=True)
