"""Diagnostic pictures of polar B=2 arc coils at successive stages, aimed at geometric degeneracies.

    python work/helios/scan_polar/polar_diag.py OUT.png "fit=path.h5" "repair=path.h5" ...

Per unique coil (rows) and stage (overlaid in colour):
  col 1  the coil in the plane of its FIRST stage (the planar start), hinges marked
  col 2  distance out of that plane along the coil (folds / tilted arcs)
  col 3  each arc's polar radius r(theta) / (|chord|/2) about its chord midpoint; the form
         degrades as this approaches 0 (the arc reaching its pole)
Printed per stage: hinge corners, arc_ref margins (1 = degenerate frame), min r / (|chord|/2),
hinge separation, and the closest approach of the coil to itself away from the hinges.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import common as S  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

COLS = ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d03b3b"]
N = 800


def polar_r(c, x):
    """r(theta)/(|chord|/2) for each arc, from sampled xyz (arc 0 = first half of the samples)."""
    H = np.asarray(c.hinges).reshape(-1, 3)
    out = []
    for b in range(2):
        seg = x[: N // 2 + 1] if b == 0 else np.vstack([x[N // 2:], x[:1]])
        h0, h1 = H[b], H[(b + 1) % 2]
        C = 0.5 * (h0 + h1)
        half = 0.5 * np.linalg.norm(h1 - h0)
        e = (h1 - h0) / (2 * half)
        P = seg - C
        xl = P @ e
        yl = np.linalg.norm(P - np.outer(xl, e), axis=1)
        theta = np.arctan2(yl, -xl)
        out.append((theta, np.linalg.norm(P, axis=1) / half))
    return out


def self_approach(x, excl=0.08):
    """Closest distance between points of the coil more than `excl` of a turn apart, ignoring the
    neighbourhoods of the two hinges (s = 0 and pi), where the arcs meet by construction."""
    n = len(x)
    s = np.arange(n) / n
    keep = (np.minimum(np.abs(s - 0.0), np.abs(s - 1.0)) > 0.03) & (np.abs(s - 0.5) > 0.03)
    idx = np.nonzero(keep)[0]
    tree = cKDTree(x[idx])
    best = np.inf
    for k, i in enumerate(idx):
        for j in tree.query_ball_point(x[i], r=min(best, 5.0)):
            jj = idx[j]
            ds = abs(s[i] - s[jj])
            if min(ds, 1 - ds) > excl:
                best = min(best, np.linalg.norm(x[i] - x[jj]))
    return best


def main():
    out = sys.argv[1]
    stages = [a.rsplit("=", 1) for a in sys.argv[2:]]
    sets = [(lab, list(S.iter_unique(load(p)))) for lab, p in stages]
    ncoil = len(sets[0][1])
    fig, axes = plt.subplots(ncoil, 3, figsize=(15, 4.6 * ncoil), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.3]))
    for i in range(ncoil):
        x0 = np.asarray(sets[0][1][i].compute(
            "x", grid=LinearGrid(zeta=np.linspace(0, 2 * np.pi, N, endpoint=False), NFP=1), basis="xyz")["x"])
        ctr = x0.mean(0)
        nrm = np.linalg.svd(x0 - ctr)[2][-1]
        ev = np.array([0, 0, 1.0]) - nrm[2] * nrm
        ev /= np.linalg.norm(ev)
        eh = np.cross(nrm, ev)
        if eh @ np.array([ctr[0], ctr[1], 0]) < 0:
            eh = -eh
        P = lambda X: np.stack([(X - ctr) @ eh, (X - ctr) @ ev], 1)
        a1, a2, a3 = axes[i]
        lines = []
        for k, (lab, coils) in enumerate(sets):
            c = coils[i]
            x = np.asarray(c.compute("x", grid=LinearGrid(zeta=np.linspace(0, 2 * np.pi, N, endpoint=False), NFP=1),
                                     basis="xyz")["x"])
            col = COLS[k % len(COLS)]
            lw = 2.6 if k == 0 else 1.5
            p = P(x)
            a1.plot(*np.vstack([p, p[:1]]).T, color=col, lw=lw, label=lab)
            if not hasattr(c, "hinges"):  # a reference coil without arcs: outline only
                a2.plot(np.arange(N) / N, (x - ctr) @ nrm, color=col, lw=lw, label=lab)
                continue
            H = P(np.asarray(c.hinges).reshape(-1, 3))
            a1.plot(*H.T, "--o", color=col, lw=0.8, ms=4)
            a2.plot(np.arange(N) / N, (x - ctr) @ nrm, color=col, lw=lw, label=lab)
            m = S.curve_metrics(c)
            rr = polar_r(c, x)
            rmin = min(float(r.min()) for _, r in rr)
            for b, (th, r) in enumerate(rr):
                a3.plot(np.degrees(th), r, color=col, lw=lw, ls="-" if b == 0 else ":",
                        label=f"{lab} arc {b}" if i == 0 else None)
            Hx = np.asarray(c.hinges).reshape(-1, 3)
            sa = self_approach(x)
            lines.append(f"{lab}: corner {m['corner']:.0f}°, κ {m['kappa']:.2f}, arc_ref margin "
                         f"{np.round(np.asarray(c.arc_ref_margin), 2)}, min r/(L/2) {rmin:.2f}, "
                         f"hinge gap {np.linalg.norm(Hx[1] - Hx[0]):.1f} m, self-approach {sa:.2f} m")
            print(f"coil {i} {lines[-1]}")
        a1.set_aspect("equal")
        a1.set_title(f"coil {i}: in the plane of '{sets[0][0]}'", loc="left", fontsize=9)
        a2.axvline(0.5, color="#cccccc", lw=0.8)
        lo, hi = a2.get_ylim()
        a2.set_ylim(min(lo, -0.1), max(hi, 0.1))  # rounding noise must not look like a fold
        a2.set_xlabel("coil parameter (turns); hinges at 0 and 0.5", fontsize=8)
        a2.set_ylabel("out of that plane (m)", fontsize=8)
        a3.axhline(0, color="#d03b3b", lw=0.8)
        a3.set_xlabel("θ about the chord midpoint (deg)", fontsize=8)
        a3.set_ylabel("r / (|chord|/2)", fontsize=8)
        a3.set_ylim(bottom=0)
        a2.set_title("\n".join(lines), loc="left", fontsize=7)
        for ax in (a1, a2, a3):
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    axes[0, 2].legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
    fig.suptitle(os.path.basename(out), x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
