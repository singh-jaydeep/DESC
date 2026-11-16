"""Field lines of a coilset: magnetic axis, nested-surface check, Poincare sections.

    python paper/trace_lines.py COILS.h5 EQ.h5 OUT [--offsets 0.02,...] [--ntransit 150]

1. The magnetic axis of the coil field at phi = 0 (Z = 0 by symmetry): iterate the
   centroid of a nearby line, starting from the equilibrium's axis.
2. Field lines from the outboard midplane at R = R_axis + offset.
3. Each line is fitted by a surface in a straight-field-line angle,
   theta = iota phi, R = sum R_mn cos(m theta - n NFP phi), Z = sum Z_mn sin(...).
   The rms fit residual over the line's mean radius says whether the line lies on a
   smooth surface: islands and chaos give a large residual.
Writes OUT/score.json (axis, per-line iota, residual, mirror ratio) and OUT/lines.png.
Ported from the --trace-only path of coil_sandbox/1012_QI/score_trace.py (js/qa-poster).
"""

import argparse
import json
import os
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("coils")
ap.add_argument("eq", help="equilibrium for the axis guess and NFP")
ap.add_argument("out")
ap.add_argument(
    "--offsets", default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20"
)
ap.add_argument("--ntransit", type=int, default=150)
ap.add_argument("--rtol", type=float, default=1e-10, help="integration rtol = atol")
ap.add_argument("--nplane", type=int, default=160, help="samples per toroidal turn")
ap.add_argument("--Mfit", type=int, default=16)
ap.add_argument("--Nfit", type=int, default=16)
ap.add_argument("--tol", type=float, default=5e-3, help="largest relative residual")
ap.add_argument("--gl", type=int, default=32, help="Gauss-Legendre nodes per arc")
ap.add_argument("--max-steps-per-rad", type=float, default=2000)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
a = ap.parse_args()
assert a.nplane > 4 * (2 * a.Nfit + 1), "nplane too small for Nfit"

from desc import set_device  # noqa: E402

set_device(a.device)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from desc.coils import coil_source_grid  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402

os.makedirs(a.out, exist_ok=True)
t0 = time.time()


def stamp(msg):
    """Print with elapsed time."""
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


cs, eq = load(a.coils), load(a.eq)
NFP = int(eq.NFP)
g0 = LinearGrid(rho=np.array([0.0]), theta=np.array([0.0]), zeta=np.array([0.0]))
Rax = float(eq.compute("R", grid=g0)["R"][0])
cs.save(os.path.join(a.out, "coils.h5"))
src = coil_source_grid(cs, n=a.gl)
log = dict(coils=a.coils, eq=a.eq, ntransit=a.ntransit, nplane=a.nplane)


def trace(R0, ntransit):
    """Lines from (R0, phi=0, Z=0); lost lines come back non-finite."""
    phis = np.linspace(0, 2 * np.pi * ntransit, ntransit * a.nplane + 1)
    R0 = np.atleast_1d(np.asarray(R0, dtype=float))
    r, z = field_line_integrate(
        R0,
        np.zeros_like(R0),
        phis,
        cs,
        source_grid=src,
        rtol=a.rtol,
        atol=a.rtol,
        max_steps=int(a.max_steps_per_rad * phis[-1]),
        options={"throw": False},
    )
    return phis, np.asarray(r), np.asarray(z)


for it in range(4):  # magnetic axis at phi = 0
    phis, r, zz = trace([Rax], 40)
    new = float(np.mean(r[:: a.nplane, 0]))
    stamp(f"axis iteration {it}: R {Rax:.6f} -> {new:.6f}")
    done = abs(new - Rax) < 1e-6
    Rax = new
    if done:
        break
log["axis_R_phi0"] = Rax

offs = np.array([float(v) for v in a.offsets.split(",")])
phis, r, zz = trace(np.concatenate([[Rax], Rax + offs]), a.ntransit)
rA, zA = r[:, 0], zz[:, 0]
stamp(f"traced {len(offs)} lines, {a.ntransit} transits")
mm, nn = np.meshgrid(
    np.arange(0, a.Mfit + 1), np.arange(-a.Nfit, a.Nfit + 1), indexing="ij"
)
keep = (mm > 0) | (nn >= 0)
mm, nn = mm[keep], nn[keep]


def fit(iota, rr, zl):
    """Least-squares surface through a line for a trial iota: (residual, cR, cZ)."""
    ang = np.outer(iota * phis, mm) - np.outer(phis, nn * NFP)
    cR = np.linalg.lstsq(np.cos(ang), rr, rcond=None)[0]
    cZ = np.linalg.lstsq(np.sin(ang), zl, rcond=None)[0]
    res = np.sqrt(np.mean((np.cos(ang) @ cR - rr) ** 2 + (np.sin(ang) @ cZ - zl) ** 2))
    return res, cR, cZ


lines = []
for j, d in enumerate(offs):
    rr, zl = r[:, j + 1], zz[:, j + 1]
    if not (np.all(np.isfinite(rr)) and np.all(np.isfinite(zl))):
        lines.append(dict(offset=float(d), lost=True))
        stamp(f"offset {d:.3f}: line lost")
        continue
    w = np.unwrap(np.arctan2(zl - zA, rr - rA))
    i_w = float((w[-1] - w[0]) / (phis[-1] - phis[0]))
    span = 2 * np.pi / phis[-1]
    o = minimize_scalar(
        lambda i: fit(i, rr, zl)[0],
        bounds=(i_w - span, i_w + span),
        method="bounded",
        options=dict(xatol=1e-9),
    )
    res, cR, cZ = fit(o.x, rr, zl)
    amin = float(np.mean(np.hypot(rr - rA, zl - zA)))
    kk = slice(0, 40 * a.nplane)  # |B| along the first 40 transits
    x = np.stack([rr[kk], phis[kk], zl[kk]], 1)
    Bl = np.linalg.norm(
        np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=src)), axis=1
    )
    mirror = float((Bl.max() - Bl.min()) / (Bl.max() + Bl.min()))
    lines.append(
        dict(
            offset=float(d),
            lost=False,
            iota=float(o.x),
            iota_winding=i_w,
            res=float(res),
            res_rel=float(res / amin),
            mean_radius=amin,
            mirror=mirror,
            nested=bool(res / amin < a.tol),
        )
    )
    stamp(
        f"offset {d:.3f}: iota {o.x:+.5f}  mean radius {amin:.4f}  residual "
        f"{res / amin:.2e} relative  mirror {mirror:.3f}"
    )
log["lines"] = lines
ok = [ln for ln in lines if not ln["lost"] and ln["nested"]]
log["largest_nested_offset"] = max((ln["offset"] for ln in ok), default=None)
stamp(f"largest offset on a smooth surface: {log['largest_nested_offset']}")

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
npl = a.nplane // NFP
for ax, k in zip(axs.flat, range(0, npl, max(1, npl // 6))):
    for j in range(len(offs)):
        ax.plot(r[k::npl, j + 1], zz[k::npl, j + 1], ".", ms=1.5, color="C0")
    ax.plot(rA[k::npl], zA[k::npl], "k+")
    ax.set_aspect("equal")
    ax.set_title(f"phi = {phis[k]:.3f}")
fig.suptitle(f"{os.path.basename(a.coils)}: field lines")
fig.savefig(os.path.join(a.out, "lines.png"), dpi=110, bbox_inches="tight")
json.dump(log, open(os.path.join(a.out, "score.json"), "w"), indent=1)
stamp(f"wrote {a.out}")
