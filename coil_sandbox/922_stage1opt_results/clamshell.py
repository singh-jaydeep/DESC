"""Differentiable eps_B and the clamshell ratio C = eps_2 / (eps_1 / 4).

eps_B is the RMS out-of-plane defect of the best split of a closed curve into B planar
arcs, normalised by the curve's own in-plane extent (the metric in
917_results/planarizability.py). The minimisation over segmentations is combinatorial, but
by Danskin's theorem the VALUE is differentiable in the curve wherever the optimal
segmentation is unique: solve the DP in numpy for the breakpoints, freeze them, and
differentiate only lambda_min of the arclength-weighted covariance -- smooth wherever that
eigenvalue is simple.

Danskin is applied ONE LEVEL DEEPER than just the segmentation: the per-segment best-fit
plane NORMALS are frozen too. The differentiated path then contains no eigendecomposition
at all -- the defect of a segment about a frozen unit normal nhat is

    W_seg * lambda_min  =  sum_i ww_i ((x_i - mu) . nhat)^2

which is polynomial in the points, and equals lambda_min exactly when nhat is the
minimiser (which it is, at the point where it was frozen). This matters: `eigvalsh`
gradients are meaningless on a degenerate eigenspace, and for a near-CIRCULAR curve the
two in-plane eigenvalues are exactly equal. Measured before the fix: no NaN, but a
gradient 30% wrong (0.996 analytic vs 1.390 finite-difference) -- silently wrong, which is
worse than a crash.

Three differences from planarizability.py, all needed for differentiability or robustness:
  * points are sampled on a UNIFORM PARAMETER grid and weighted by |x_s|, rather than
    resampled to uniform arclength (resampling is an interpolation, and its nonsmoothness
    would pollute the gradient). The two agree as the sampling is refined; `selftest`
    checks it.
  * segment boundaries AND plane normals are frozen, not re-optimised inside the
    differentiated call.
  * the DP cost table centres the points first. The textbook sum-of-squares-minus-square-
    of-sum form cancels catastrophically on uncentred coordinates: measured 116% error at
    a coordinate offset of 1e4 (eps_2 collapsed to eps_1). Real coil coordinates sit at
    R ~ 8 m with ~2 m extents, which is safe, but the failure is silent.

MINIMUM SEGMENT LENGTH. A segment with fewer than 3 points has a rank-deficient covariance
and no well-defined plane; two breakpoints at the same index make `eps_from_breaks` count
the whole curve TWICE (measured ratio exactly sqrt(2)). Candidate hinge sets must therefore
respect `MIN_SEG` and `_check_breaks` enforces it.

C = eps_2 / (eps_1 / 4) measures how the curve compares with the GENERIC decay: any smooth
curve has eps_B ~ const/B^2, so eps_1/4 is the null prediction for eps_2.

    C < 1  clamshells BETTER than a generic smooth curve   <- the direction we minimise
    C = 1  generic
    C > 1  worse than generic

C is scale-free in overall non-planarity, so minimising it cannot be gamed by flattening
the curve toward axisymmetry (where eps_1 = eps_2 = 0 and the raw metric is degenerate).
"""

if __name__ == "__main__":          # boot BEFORE desc/jax are imported
    import os as _os
    import sys as _sys
    _os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
    _sys.path.insert(0, "sandbox")
    import boot as _boot
    _boot.setup(default="cpu")

import numpy as np

# jax.numpy directly, not desc.backend: this module needs no DESC, and importing DESC here
# would let a consumer that forgets boot.setup() pull in the NON-vendored clone.
# The project rule still applies to the consumer -- call boot.setup() before importing this,
# because the first jax import fixes the platform.
import jax.numpy as jnp  # noqa: E402

__all__ = ["freeze", "eps_from_frozen", "best_breaks", "eps_B", "clamshell_ratio",
           "selftest", "MIN_SEG"]

MIN_SEG = 3        # a plane needs 3 points; fewer is rank-deficient


# ---------------------------------------------------------------------------
# freezing step (numpy): segmentation -> masks and plane normals
# ---------------------------------------------------------------------------
def _check_breaks(breaks, n):
    """Every segment must hold at least MIN_SEG points, and breaks must be distinct."""
    b = sorted(int(x) % n for x in breaks)
    if len(b) == 1:
        return b
    if len(set(b)) != len(b):
        raise ValueError(f"duplicate breakpoints {breaks}: a zero-length segment makes the "
                         f"whole curve count twice (measured ratio sqrt(2))")
    gaps = [(b[(k + 1) % len(b)] - b[k]) % n for k in range(len(b))]
    if min(gaps) < MIN_SEG:
        raise ValueError(f"segment of {min(gaps)} points < MIN_SEG={MIN_SEG}: the covariance "
                         f"is rank-deficient and has no well-defined plane")
    return b


def _mask(n, lo, hi, dtype):
    shifted = np.mod(np.arange(n) - lo, n)
    return (shifted < np.mod(hi - lo - 1, n) + 1).astype(dtype)


def _normal(Pn, wn, m):
    """Unit normal of the best-fit plane of the masked, weighted points (numpy)."""
    ww = wn * m
    W = ww.sum()
    D = Pn - (ww[:, None] * Pn).sum(0) / W
    ev, evec = np.linalg.eigh((D * ww[:, None]).T @ D / W)
    return evec[:, 0], ev


def freeze(P, w, breaks):
    """Freeze the segmentation AND the plane normals. Returns a dict for `eps_from_frozen`.

    Also reports `gap`, the eigenvalue separation that Danskin needs to be nonzero:
    `gap_seg` = lambda_1/lambda_2 per segment (0 means a straight, rank-deficient segment)
    and `gap_scale` = lambda_1/lambda_2 of the whole curve (1 means the in-plane pair is
    degenerate, which is fine here because only their PRODUCT is used).
    """
    Pn, wn = np.asarray(P, float), np.asarray(w, float)
    n = len(Pn)
    b = _check_breaks(breaks, n)
    masks, nrms, gaps = [], [], []
    for k in range(len(b)):
        m = _mask(n, b[k], b[(k + 1) % len(b)], Pn.dtype)
        nrm, ev = _normal(Pn, wn, m)
        masks.append(m)
        nrms.append(nrm)
        gaps.append(float(ev[1] / max(ev[2], 1e-300)))
    n0, ev0 = _normal(Pn, wn, np.ones(n, Pn.dtype))
    return dict(breaks=b, masks=np.array(masks), normals=np.array(nrms), n0=n0,
                gap_seg=min(gaps), gap_scale=float(ev0[1] / max(ev0[2], 1e-300)))


# ---------------------------------------------------------------------------
# differentiable part: no eigendecomposition anywhere below this line
# ---------------------------------------------------------------------------
def eps_from_frozen(P, w, fz):
    """eps for a frozen segmentation. Differentiable in P and w, exactly and stably."""
    masks = jnp.asarray(fz["masks"])
    nrms = jnp.asarray(fz["normals"])
    tot = 0.0
    for k in range(masks.shape[0]):
        ww = w * masks[k]
        W = jnp.sum(ww)
        D = P - jnp.sum(ww[:, None] * P, axis=0) / W
        r = D @ nrms[k]
        tot = tot + jnp.sum(ww * r**2)          # == W * lambda_min at the frozen normal
    # in-plane extent: lambda_1 * lambda_2 from matrix invariants, so the degenerate
    # in-plane pair of a near-circular curve is never differentiated individually
    W = jnp.sum(w)
    D = P - jnp.sum(w[:, None] * P, axis=0) / W
    M = (D * w[:, None]).T @ D / W
    lam0 = jnp.asarray(fz["n0"]) @ M @ jnp.asarray(fz["n0"])
    trM = jnp.trace(M)
    e2 = 0.5 * (trM**2 - jnp.trace(M @ M))
    l1l2 = e2 - lam0 * (trM - lam0)
    scale = jnp.sqrt(jnp.sqrt(jnp.clip(l1l2, 1e-30)))
    return jnp.sqrt(tot / W) / scale


# ---------------------------------------------------------------------------
# combinatorial part: the DP, in numpy, outside the differentiated call
# ---------------------------------------------------------------------------
def _cost_table(P, w):
    """C[i, L] = W * lambda_min for the arc of L+1 points starting at i (circular)."""
    n = len(P)
    P = np.asarray(P, float)
    P = P - P.mean(0)          # MUST centre: the prefix-sum covariance below cancels
    Pw = np.concatenate([P, P])
    ww = np.concatenate([w, w])
    cw = np.concatenate([[0.0], np.cumsum(ww)])
    cx = np.concatenate([[np.zeros(3)], np.cumsum(ww[:, None] * Pw, axis=0)])
    cxx = np.concatenate([[np.zeros((3, 3))],
                          np.cumsum(ww[:, None, None] * Pw[:, :, None] * Pw[:, None, :], axis=0)])
    C = np.full((n, n), np.inf)
    for L in range(0, n):
        i = np.arange(n)
        j = i + L + 1
        W = cw[j] - cw[i]
        S = cx[j] - cx[i]
        M = cxx[j] - cxx[i] - S[:, :, None] * S[:, None, :] / W[:, None, None]
        ev = np.linalg.eigvalsh(M / W[:, None, None])
        C[i, L] = W * np.clip(ev[:, 0], 0.0, None)
    return C


def best_breaks(P, w, B, stride=2):
    """Exact best B-segmentation of the closed curve. Returns (cost, breakpoints)."""
    C = _cost_table(P, w)
    n = len(P)
    best, bestcut = np.inf, None
    for t in range(0, n, stride):
        f = np.full((B + 1, n + 1), np.inf)
        g = np.zeros((B + 1, n + 1), dtype=int)
        f[0][0] = 0.0
        for k in range(1, B + 1):
            for p in range(1, n + 1):
                Ls = np.arange(0, p)
                tot = f[k - 1][p - 1 - Ls] + C[(t + p - 1 - Ls) % n, Ls]
                jbest = int(np.argmin(tot))
                f[k][p], g[k][p] = tot[jbest], Ls[jbest]
        if f[B][n] < best:
            best = f[B][n]
            cuts, p = [], n
            for k in range(B, 0, -1):
                L = g[k][p]
                p -= L + 1
                cuts.append((t + p) % n)
            bestcut = sorted(cuts)
    return best, bestcut


def eps_B(P, w, B, stride=2):
    """eps_B, its frozen segmentation, and its breakpoints."""
    _, breaks = best_breaks(np.asarray(P), np.asarray(w), B, stride)
    fz = freeze(P, w, breaks)
    return float(eps_from_frozen(jnp.asarray(P), jnp.asarray(w), fz)), fz


def clamshell_ratio(P, w, stride=2):
    """C = eps_2 / (eps_1 / 4), with the frozen data for each.

    C < 1 is a real two-plane preference (better than the generic B^-2 decay), C = 1 is
    generic, C > 1 is worse than generic. Lower is more clamshell-able."""
    e1, f1 = eps_B(P, w, 1, stride)
    e2, f2 = eps_B(P, w, 2, stride)
    return 4.0 * e2 / e1, dict(eps_1=e1, eps_2=e2, frozen_1=f1, frozen_2=f2)


# ---------------------------------------------------------------------------
def selftest(verbose=True):
    """(a) agreement with planarizability.py, (b) gradient vs finite differences."""
    import os
    import sys

    from jax import grad  # noqa

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "917_results"))
    import planarizability as PL

    ok = True

    def curve(p, n=180):
        """A closed 3-D curve depending smoothly on p; deliberately not planar."""
        s = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        R = 3.0 + 0.6 * jnp.cos(2 * s) + p[0] * jnp.cos(3 * s)
        Z = 1.3 * jnp.sin(s) + p[1] * jnp.sin(2 * s)
        Y = p[2] * jnp.sin(2 * s) + p[3] * jnp.cos(3 * s)
        return jnp.stack([R, Y, Z], axis=1)

    def pts_w(p, n=180):
        P = curve(p, n)
        d = jnp.roll(P, -1, axis=0) - jnp.roll(P, 1, axis=0)
        return P, jnp.linalg.norm(d, axis=1) / 2.0

    p0 = jnp.array([0.25, 0.4, 0.35, 0.2])

    # (a) against planarizability.py on the same points
    P, w = pts_w(p0, 360)
    for B in (1, 2, 3):
        mine, _ = eps_B(P, w, B, stride=4)
        Pn = np.asarray(P)
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
            np.diff(np.vstack([Pn, Pn[:1]]), axis=0), axis=1))])
        s = np.linspace(0, d[-1], PL.NS, endpoint=False)
        Q = np.stack([np.interp(s, d, np.vstack([Pn, Pn[:1]])[:, j]) for j in range(3)], 1)
        Cq = PL.cost_table(Q)
        tot, _ = PL.best_split(Cq, B)
        ev = np.linalg.eigvalsh(((Q - Q.mean(0)).T @ (Q - Q.mean(0))) / len(Q))
        ref = float(np.sqrt(max(tot, 0) / len(Q)) /
                    np.sqrt(np.sqrt(max(ev[1], 1e-30) * max(ev[2], 1e-30))))
        rel = abs(mine - ref) / ref
        ok &= rel < 5e-2
        if verbose:
            print(f"  [{'ok ' if rel < 5e-2 else 'BAD'}] eps_{B}: this {mine:.6f} vs "
                  f"planarizability {ref:.6f}  rel {rel:.2e}")

    # (b) gradient vs finite differences, breakpoints frozen
    n = 180
    P0, w0 = pts_w(p0, n)
    _, br1 = best_breaks(np.asarray(P0), np.asarray(w0), 1, stride=4)
    _, br2 = best_breaks(np.asarray(P0), np.asarray(w0), 2, stride=4)
    f1 = freeze(P0, w0, br1)
    f2 = freeze(P0, w0, br2)
    if verbose:
        print(f"  frozen gaps: seg lambda_1/lambda_2 = {f2['gap_seg']:.3f} (0 = straight "
              f"segment, bad), whole-curve = {f2['gap_scale']:.3f}")

    def C_of_p(p):
        P, w = pts_w(p, n)
        return 4.0 * eps_from_frozen(P, w, f2) / eps_from_frozen(P, w, f1)

    gA = np.asarray(grad(C_of_p)(p0))
    h = 1e-5
    gN = np.array([float((C_of_p(p0.at[i].add(h)) - C_of_p(p0.at[i].add(-h))) / (2 * h))
                   for i in range(len(p0))])
    rel = np.abs(gA - gN).max() / max(np.abs(gN).max(), 1e-30)
    ok &= rel < 1e-5
    if verbose:
        print(f"  C(p0) = {float(C_of_p(p0)):.6f}")
        print(f"  [{'ok ' if rel < 1e-5 else 'BAD'}] dC/dp analytic {np.array2string(gA, precision=6)}")
        print(f"         finite difference {np.array2string(gN, precision=6)}  rel {rel:.2e}")
    # (c) the three bugs found in review, as regressions
    #  c1: near-circular curve -> exactly degenerate in-plane pair; gradient must still be right
    sc = jnp.linspace(0, 2 * jnp.pi, 180, endpoint=False)

    def circ(q):
        return jnp.stack([jnp.cos(sc), jnp.sin(sc), q[0] * jnp.sin(3 * sc)], axis=1)

    def cw(P):
        return jnp.linalg.norm(jnp.roll(P, -1, 0) - jnp.roll(P, 1, 0), axis=1) / 2

    q0 = jnp.array([0.05])
    Pc, wc = circ(q0), cw(circ(q0))
    fc = freeze(Pc, wc, best_breaks(np.asarray(Pc), np.asarray(wc), 1, stride=6)[1])
    fe = lambda q: eps_from_frozen(circ(q), cw(circ(q)), fc)  # noqa: E731
    ga = float(grad(fe)(q0)[0])
    gn = float((fe(q0.at[0].add(1e-6)) - fe(q0.at[0].add(-1e-6))) / 2e-6)
    rel = abs(ga - gn) / abs(gn)
    ok &= rel < 1e-5
    if verbose:
        print(f"  [{'ok ' if rel < 1e-5 else 'BAD'}] degenerate in-plane pair "
              f"(lambda_2/lambda_1 = {fc['gap_scale']:.6f}): grad {ga:.6f} vs FD {gn:.6f}  rel {rel:.1e}")

    #  c2: duplicate / too-close breakpoints must be refused, not silently doubled
    caught = 0
    for bad in ([7, 7], [7, 8]):
        try:
            freeze(P0, w0, bad)
        except ValueError:
            caught += 1
    ok &= caught == 2
    if verbose:
        print(f"  [{'ok ' if caught == 2 else 'BAD'}] degenerate breakpoints refused ({caught}/2)")

    #  c3: a large coordinate offset must not change eps (cancellation in the DP table)
    e_ref, _ = eps_B(np.asarray(P0), np.asarray(w0), 2, stride=4)
    worst = 0.0
    for off in (1e2, 1e4, 1e6):
        e_off, _ = eps_B(np.asarray(P0) + np.array([off, 0, 0]), np.asarray(w0), 2, stride=4)
        worst = max(worst, abs(e_off - e_ref) / e_ref)
    ok &= worst < 1e-6
    if verbose:
        print(f"  [{'ok ' if worst < 1e-6 else 'BAD'}] coordinate offset up to 1e6: "
              f"worst relative change in eps_2 {worst:.2e}")
    return ok


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "sandbox")
    import boot
    boot.setup(default="cpu")
    print("clamshell.py selftest")
    sys.exit(0 if selftest() else 1)
