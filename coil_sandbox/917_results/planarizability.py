"""How piecewise-planar is a coil?  The defect spectrum eps_B, solved exactly by DP.

For a set of points the best-fit-plane residual is the smallest eigenvalue of their
arclength-weighted covariance, and a covariance over any contiguous range follows in
O(1) from prefix sums of w, w x and w x x^T. So the cost of every candidate arc is a
table, and the best split of the closed curve into B arcs is a dynamic program rather
than a search -- exact, and fast enough to sweep over equilibria.

  eps_B = sqrt( min over B-segmentations of  sum_seg w_seg * lambda_min(seg) / W )
          normalised by the coil's own in-plane extent, so it is dimensionless.

eps_1 is "how planar", eps_2 "how clamshell-able", and the decay of eps_B with B is the
signature. The optimal breakpoints come out as a by-product -- they are the hinge
placement a projection onto the arc classes would want.

CAVEAT, and it is the important one: this measures the distance from THIS coilset to
the piecewise-planar set. It is NOT the cost of imposing the restriction, which is the
gap between two CONSTRAINED optima and is much smaller -- on helios, unrestricted
FourierXYZ reaches bn 8.43e-3 while 2-plane polar arcs reach 1.045e-2, only 19% worse,
even though the projection distance between them is metres. Treat eps_B as a screening
heuristic whose correlation has to be established, not as a predictor.

    python 917_results/planarizability.py [coilset.h5 ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid  # noqa: E402
from desc.io import load  # noqa: E402

NS = 180          # samples per coil; the DP is O(NS^3 B) over all circular starts
BMAX = 4


def resample(coil, n=NS):
    """n points equally spaced in ARCLENGTH, so the weights are uniform and the metric
    is a property of the shape, not of the curve parameter."""
    s = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False,
                                               jitable=False), basis="xyz")["x"])
    xc = np.vstack([x, x[:1]])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xc, axis=0), axis=1))])
    return np.stack([np.interp(np.linspace(0, d[-1], n, endpoint=False), d, xc[:, j])
                     for j in range(3)], axis=1)


def cost_table(P):
    """C[i, L] = w * lambda_min for the arc of L+1 points starting at i (circular)."""
    n = len(P)
    Q = np.vstack([P, P])                       # double it so every arc is contiguous
    cw = np.concatenate([[0.0], np.arange(1, 2 * n + 1, dtype=float)])
    cx = np.vstack([np.zeros(3), np.cumsum(Q, axis=0)])
    cxx = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(Q[:, :, None] * Q[:, None, :], axis=0)])
    C = np.full((n, n + 1), np.inf)
    for L in range(2, n):                       # arcs of L+1 points
        i = np.arange(n)
        w = cw[i + L + 1] - cw[i]
        m = (cx[i + L + 1] - cx[i]) / w[:, None]
        M = (cxx[i + L + 1] - cxx[i]) / w[:, None, None] - m[:, :, None] * m[:, None, :]
        lam = np.linalg.eigvalsh(M)[:, 0].clip(0)   # smallest eigenvalue = mean sq. out-of-plane
        C[:, L] = w * lam
    C[:, 0] = C[:, 1] = 0.0                     # 1-2 points are trivially planar
    return C


def best_split(C, B):
    """Minimum total w*lambda over B arcs covering the closed curve, and the breakpoints."""
    n = C.shape[0]
    best, bestcut = np.inf, None
    for t in range(0, n, 2):                    # every other point as the first hinge
        # f[k][p] = best cost covering p points from t using k arcs
        f = np.full((B + 1, n + 1), np.inf)
        g = np.zeros((B + 1, n + 1), dtype=int)
        f[0][0] = 0.0
        for k in range(1, B + 1):
            for p in range(1, n + 1):
                Ls = np.arange(0, p)
                prev = f[k - 1][p - 1 - Ls]
                cur = C[(t + p - 1 - Ls) % n, Ls]
                tot = prev + cur
                j = int(np.argmin(tot))
                f[k][p], g[k][p] = tot[j], Ls[j]
        if f[B][n] < best:
            best = f[B][n]
            cuts, p = [], n
            for k in range(B, 0, -1):
                L = g[k][p]; p -= L + 1; cuts.append((t + p) % n)
            bestcut = sorted(cuts)
    return best, bestcut


def spectrum(coil, bmax=BMAX):
    P = resample(coil)
    n = len(P)
    C = cost_table(P)
    mu = P.mean(0)
    ev = np.linalg.eigvalsh(((P - mu).T @ (P - mu)) / n)
    scale = np.sqrt(np.sqrt(max(ev[1], 1e-30) * max(ev[2], 1e-30)))   # in-plane extent
    out = {}
    for B in range(1, bmax + 1):
        tot, cuts = best_split(C, B)
        out[B] = (float(np.sqrt(max(tot, 0) / n) / scale), cuts)
    return out, scale


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "917_results/planar_N7.h5", "917_results/arcB2_transverse_M5.h5",
        "917_results/polarB2_M5_cut30.h5", "917_results/arcB3_M5.h5",
        "917_results/spline_n32_from_cut30.h5"]
    print(f"eps_B: RMS out-of-plane defect of the best B-arc split, / in-plane extent")
    print(f"(sampled at {NS} points per coil, arclength-uniform)\n")
    print(f"{'coilset':<34} {'coil':>4} " + " ".join(f"{'eps_'+str(b):>9}" for b in range(1, BMAX + 1)))
    for p in paths:
        cs = load(p)
        for i, c in enumerate(S.iter_unique(cs)):
            sp, _ = spectrum(c)
            print(f"{os.path.basename(p).replace('.h5',''):<34} {i:>4} "
                  + " ".join(f"{sp[b][0]:>9.2e}" for b in range(1, BMAX + 1)), flush=True)
