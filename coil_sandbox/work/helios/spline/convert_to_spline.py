"""Convert an arc coilset to SplineXYZCoil with C0 breaks at the hinges, and compare.

    python work/helios/spline/convert_to_spline.py --coils work/helios/polarB2_M5_cut30_final.h5

The arc coils put their B hinges at s = 2 pi i / B exactly, so a uniform knot vector of n
knots with n % B == 0 lands a knot on every hinge; `break_indices` then names those knots
and the spline is C0 there (`SplineXYZCurve._splinexyz_helper`).

Both curves are parameterized by the same s, so the comparison is pointwise: no fitting,
no nearest-neighbour search. Writes compare.json and compare.png to --out.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import CoilSet, MixedCoilSet, SplineXYZCoil  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.io import load  # noqa: E402


def sgrid(s):
    return Grid(np.stack([0 * s, 0 * s, np.mod(s, 2 * np.pi)], 1), sort=False, jitable=False)


def xyz(c, s, key="x"):
    return np.asarray(c.compute(key, grid=sgrid(s), basis="xyz")[key])


def to_spline(coil, n, method):
    """SplineXYZCoil with n uniform knots and a C0 break on each of the coil's B hinges."""
    B = int(coil.B)
    if n % B:
        raise ValueError(f"n={n} must be a multiple of B={B} so a knot lands on every hinge")
    knots = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return SplineXYZCoil.from_values(float(coil.current), xyz(coil, knots), knots=knots,
                                     method=method, basis="xyz",
                                     break_indices=np.arange(B) * (n // B))


def gl_length(coil, breaks, m=24):
    """Length on the composite Gauss-Legendre rule the sandbox uses for arc coils."""
    d = coil.compute(["x_s", "ds"], grid=S.gl_grid(breaks, m))
    return float(np.sum(np.linalg.norm(np.asarray(d["x_s"]), axis=-1) * np.asarray(d["ds"])))


def plane_normal(p):
    """Unit normal of the best-fit plane through points p, and their centroid."""
    c = p.mean(0)
    return np.linalg.svd(p - c)[2][-1], c


def compare(arc, spl, nd=2001):
    """Pointwise comparison of the two curves on a dense s grid (hinges included)."""
    B = int(arc.B)
    s = np.linspace(0, 2 * np.pi, nd, endpoint=False)
    xa, xs = xyz(arc, s), xyz(spl, s)
    dev = np.linalg.norm(xs - xa, axis=1)
    ka = np.abs(np.asarray(arc.compute("|curvature|", grid=sgrid(s))["|curvature|"]))
    ks = np.abs(np.asarray(spl.compute("|curvature|", grid=sgrid(s))["|curvature|"]))
    seg = np.minimum((s * B / (2 * np.pi)).astype(int), B - 1)
    # out-of-plane: each arc segment is planar by construction; measure the spline against
    # THAT plane, so the number is the non-planarity the conversion introduced.
    oop = np.zeros(nd)
    for i in range(B):
        m = seg == i
        nrm, c = plane_normal(xa[m])
        oop[m] = (xs[m] - c) @ nrm
    # corner angle at each hinge, from one-sided tangents
    eps = 1e-6
    ang = []
    for i in range(B):
        b = 2 * np.pi * i / B
        t = np.array(xyz(spl, np.array([b - eps, b + eps]), key="x_s"))
        t = t / np.linalg.norm(t, axis=1)[:, None]
        ang.append(float(np.degrees(np.arccos(np.clip(t[0] @ t[1], -1, 1)))))
    return dict(s=s, x_arc=xa, x_spl=xs, dev=dev, k_arc=ka, k_spl=ks, oop=oop, seg=seg,
                corner=ang, dev_max=float(dev.max()), oop_absmax=float(np.abs(oop).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coils", default="work/helios/polarB2_M5_cut30_final.h5")
    ap.add_argument("--out", default="work/helios/spline")
    ap.add_argument("--n", type=int, default=64, help="knots per coil for the plotted conversion")
    ap.add_argument("--method", default="cubic2", help="cubic (C1) or cubic2 (C2)")
    ap.add_argument("--scan", default="16,32,64,128,256", help="knot counts to tabulate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cs = load(a.coils)
    uniq = list(S.iter_unique(cs))
    B = int(uniq[0].B)
    print(f"{a.coils}: {len(uniq)} unique coils, {type(uniq[0]).__name__}, B={B}, M={int(uniq[0].M)}", flush=True)

    # ---- resolution scan: max deviation and length error, per method -------------------
    scan = {}
    for method in ("cubic", "cubic2"):
        for n in [int(v) for v in a.scan.split(",")]:
            rows = []
            for c in uniq:
                spl = to_spline(c, n, method)
                r = compare(c, spl)
                # both on the SAME composite-GL rule (per arc). A polyline length cuts the
                # C0 corners and loses ~5 mm on this coil at 4001 points, for the arc and
                # the spline alike -- an estimator artifact, not a difference between them.
                La, Ls = (gl_length(x, np.arange(B) * 2 * np.pi / B) for x in (c, spl))
                rows.append(dict(dev_mm=r["dev_max"] * 1e3, oop_mm=r["oop_absmax"] * 1e3,
                                 dL_mm=(Ls - La) * 1e3, k_max=float(r["k_spl"].max()),
                                 corner=r["corner"]))
            scan[f"{method}/{n}"] = rows
            print(f"  {method:7s} n={n:4d}  dev max {max(r['dev_mm'] for r in rows):8.2f} mm   "
                  f"out-of-plane {max(r['oop_mm'] for r in rows):7.2f} mm   "
                  f"dL {max(abs(r['dL_mm']) for r in rows):7.1f} mm   "
                  f"kappa max {max(r['k_max'] for r in rows):.3f}", flush=True)

    # ---- the plotted conversion --------------------------------------------------------
    res = [compare(c, to_spline(c, a.n, a.method)) for c in uniq]
    for i, r in enumerate(res):
        print(f"  coil {i}: dev max {r['dev_max']*1e3:.2f} mm, out-of-plane {r['oop_absmax']*1e3:.2f} mm, "
              f"corners " + ", ".join(f"{v:.1f}" for v in r["corner"]) + " deg", flush=True)

    nc = len(uniq)
    fig, axes = plt.subplots(4, nc, figsize=(5.0 * nc, 15.0), squeeze=False)
    for i, (c, r) in enumerate(zip(uniq, res)):
        xa, xs, s = r["x_arc"], r["x_spl"], r["s"]
        H = np.asarray(c.hinges).reshape(B, 3)
        # (a) in the coil's own best-fit plane
        nrm, ctr = plane_normal(xa)
        ev = np.array([0.0, 0.0, 1.0]) - nrm[2] * nrm
        ev /= np.linalg.norm(ev)
        eh = np.cross(nrm, ev)
        ax = axes[0, i]
        ax.plot((xa - ctr) @ eh, (xa - ctr) @ ev, color="#2a78d6", lw=3.0, label="arc B=2", zorder=2)
        ax.plot((xs - ctr) @ eh, (xs - ctr) @ ev, color="#eb6834", lw=1.4, ls="--",
                label=f"spline {a.method} n={a.n}", zorder=3)
        ax.plot((H - ctr) @ eh, (H - ctr) @ ev, "o", color="#111", ms=7, label="hinges", zorder=4)
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"coil {i}: in the coil plane [m]", loc="left")
        # (b) zoom on the first hinge
        ax = axes[1, i]
        w = 0.6
        ax.plot((xa - H[0]) @ eh, (xa - H[0]) @ ev, color="#2a78d6", lw=3.0)
        ax.plot((xs - H[0]) @ eh, (xs - H[0]) @ ev, color="#eb6834", lw=1.4, ls="--")
        ax.plot(0, 0, "o", color="#111", ms=7)
        ax.set_xlim(-w, w)
        ax.set_ylim(-w, w)
        ax.set_aspect("equal")
        ax.set_title(f"hinge 0 zoom, corner {r['corner'][0]:.0f} deg", loc="left")
        # (c) deviation and out-of-plane
        ax = axes[2, i]
        ax.plot(s, r["dev"] * 1e3, color="#eb6834", lw=1.5, label="|spline - arc|")
        ax.plot(s, np.abs(r["oop"]) * 1e3, color="#8a5cd6", lw=1.5, label="|out of arc plane|")
        for b in np.arange(B) * 2 * np.pi / B:
            ax.axvline(b, color="#111", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("s")
        ax.set_ylabel("mm")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("deviation (dotted: hinges)", loc="left")
        # (d) curvature
        ax = axes[3, i]
        ax.plot(s, r["k_arc"], color="#2a78d6", lw=2.5, label="arc")
        ax.plot(s, r["k_spl"], color="#eb6834", lw=1.2, label="spline")
        for b in np.arange(B) * 2 * np.pi / B:
            ax.axvline(b, color="#111", lw=0.8, ls=":")
        ax.set_xlabel("s")
        ax.set_ylabel("|kappa| [1/m]")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("curvature", loc="left")
    fig.suptitle(f"{os.path.basename(a.coils)} -> SplineXYZCoil, {a.method}, {a.n} knots, "
                 f"C0 breaks at the {B} hinges", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "compare.png"), dpi=110)
    plt.close(fig)

    json.dump(dict(coils=a.coils, n=a.n, method=a.method, B=B, scan=scan,
                   plotted=[dict(dev_max_mm=r["dev_max"] * 1e3, oop_absmax_mm=r["oop_absmax"] * 1e3,
                                 corner_deg=r["corner"]) for r in res]),
              open(os.path.join(a.out, "compare.json"), "w"), indent=1, default=float)
    print(f"wrote {a.out}/compare.png, compare.json")


if __name__ == "__main__":
    main()
