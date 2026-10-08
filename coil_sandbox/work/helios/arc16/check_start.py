"""Pre-GPU checks for fitting B=2 polar arcs to a smooth coilset (handoff's 3 gates).

1. eps_B spectrum by the exact DP (917_results/planarizability.py) AND the optimal
   B=2 breakpoints, reported as coil parameter values so make_start can use them.
2. Polar-arc admissibility per candidate segment: a PolarPlanarArcCoil arc is
   r(theta) about the chord midpoint, so theta must advance MONOTONICALLY and
   sweep less than pi.  A segment can be planar to 1e-14 and still fail this.
3. Where the DP hinges sit relative to the centroid-line cut angle, so the result
   can be expressed as --cut-angle if that is close enough.

    python work/helios/arc16/check_start.py COILSET.h5 [COILSET.h5 ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, "917_results")
import common as S  # noqa: E402
import planarizability as PZ  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.io import load  # noqa: E402

NS = PZ.NS


def dense(coil, n=4000):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False,
                                               jitable=False), basis="xyz")["x"])
    return s, x


def arclen(x):
    xc = np.vstack([x, x[:1]])
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xc, axis=0), axis=1))])


def idx_to_s(coil, idx, n_ds=4000):
    """The DP samples NS points uniform in arclength starting at s=0; map back to parameter."""
    s, x = dense(coil, n_ds)
    a = arclen(x)                      # len n_ds+1, at s and 2pi
    sc = np.concatenate([s, [2 * np.pi]])
    target = a[-1] * (np.asarray(idx, float) / NS)
    return np.interp(target, a, sc)


def admissible(coil, s_hinge, n_per=600):
    """For each arc between consecutive hinges: fit its plane, project, and measure the
    polar sweep about the chord midpoint.  Returns per-arc diagnostics."""
    pts = S.resample_between(coil, s_hinge, n_per=n_per)
    B = len(s_hinge)
    out = []
    for k in range(B):
        P = pts[k * n_per:(k + 1) * n_per]
        # include the closing hinge of this arc (first point of the next piece)
        P = np.vstack([P, pts[((k + 1) % B) * n_per]])
        c = P.mean(0)
        u, sv, vt = np.linalg.svd(P - c)
        nrm = vt[-1]
        flat = float(np.abs((P - c) @ nrm).max())
        e1, e2 = vt[0], vt[1]
        chord = P[-1] - P[0]
        pole = 0.5 * (P[0] + P[-1])
        q = np.stack([(P - pole) @ e1, (P - pole) @ e2], 1)
        th = np.unwrap(np.arctan2(q[:, 1], q[:, 0]))
        th = th - th[0]
        if th[-1] < 0:
            th = -th
        dth = np.diff(th)
        r = np.linalg.norm(q, axis=1)
        out.append(dict(
            flat=flat,                                   # max out-of-plane of this arc, m
            sweep=float(th[-1]),                         # total polar sweep (should be ~pi)
            monotone=bool(dth.min() > 0),
            dtheta_min=float(dth.min()),
            n_back=int((dth <= 0).sum()),
            chord=float(np.linalg.norm(chord)),
            r_min_over_half=float(r.min() / (0.5 * np.linalg.norm(chord))),
            arclen=float(arclen(P)[-2]),
        ))
    return out, pts


def cut_angle_of(coil, s_hinge):
    """The in-plane angle (deg from vertical) of the chord direction, and how far the chord
    passes from the centroid, in units of the coil half-size -- i.e. what --cut-angle /
    --cut-offset would have to be."""
    s, x = dense(coil)
    c = x.mean(0)
    nrm = np.linalg.svd(x - c)[2][-1]
    ev = np.array([0.0, 0.0, 1.0]) - nrm[2] * nrm
    ev /= np.linalg.norm(ev)
    eh = np.cross(nrm, ev)
    if eh @ np.array([c[0], c[1], 0.0]) < 0:
        eh = -eh
    h = np.stack([np.interp(np.asarray(s_hinge), np.concatenate([s, [2 * np.pi]]),
                            np.concatenate([x[:, j], x[:1, j]])) for j in range(3)], 1)
    d = h[1] - h[0]
    psi = np.degrees(np.arctan2(d @ eh, d @ ev)) % 180.0
    mid = 0.5 * (h[0] + h[1])
    half = float(np.linalg.norm(x - c, axis=1).max())
    off = float(np.linalg.norm(mid - c - ((mid - c) @ nrm) * nrm) / half)
    return psi, off, h, half


def report(path):
    cs = load(path)
    print(f"\n=== {path} ===", flush=True)
    hinges_all = []
    for i, c in enumerate(S.iter_unique(cs)):
        sp, scale = PZ.spectrum(c)
        eps = " ".join(f"eps_{b} {sp[b][0]:.2e}" for b in range(1, PZ.BMAX + 1))
        cuts2 = sp[2][1]
        sh = sorted(idx_to_s(c, cuts2))
        hinges_all.append(sh)
        psi, off, h, half = cut_angle_of(c, sh)
        R = np.hypot(h[:, 0], h[:, 1])
        adm, _ = admissible(c, sh)
        print(f"\ncoil {i}: {eps}   in-plane extent {scale:.2f} m", flush=True)
        print(f"  B=2 DP hinges: s = {sh[0]:.4f}, {sh[1]:.4f} rad   "
              f"(R,Z) = ({R[0]:.2f}, {h[0,2]:.2f}), ({R[1]:.2f}, {h[1,2]:.2f}) m")
        print(f"  equivalent cut: psi {psi:.1f} deg from vertical, chord offset "
              f"{off:.3f} of half-size ({off*half:.2f} m)")
        for k, a in enumerate(adm):
            flag = "OK " if (a["monotone"] and a["sweep"] < np.pi * 1.02) else "BAD"
            print(f"  arc {k}: {flag} flat {a['flat']*1e3:7.1f} mm  sweep {np.degrees(a['sweep']):6.1f} deg  "
                  f"monotone {a['monotone']} (min dtheta {a['dtheta_min']*1e3:+.2f} mrad, "
                  f"{a['n_back']} backsteps)  L {a['arclen']:.2f} m  r_min/half-chord {a['r_min_over_half']:.2f}")
    return hinges_all


if __name__ == "__main__":
    paths = sys.argv[1:] or ["917_results/fourierXYZ_16coils.h5"]
    out = {}
    for p in paths:
        out[p] = report(p)
    print("\n--- hinge parameter values, for make_start --hinge-s ---")
    for p, hs in out.items():
        print(f"{p}\n  " + ";".join(",".join(f"{v:.6f}" for v in h) for h in hs))
