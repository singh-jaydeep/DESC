"""Shared pieces: equilibria, bounds, coil helpers, and the dense geometry check.

Import only after `boot.setup()`.
"""

import glob
import itertools
import json
import os
import threading
import time

import numpy as np
from scipy.spatial import cKDTree

from desc.grid import Grid, LinearGrid

# ---------------------------------------------------------------------------------------
# Equilibria
# ---------------------------------------------------------------------------------------


def load_equilibrium(spec):
    """An Equilibrium from: a desc.examples name ("precise_QA"), a DESC .h5 (an Equilibrium or
    an EquilibriaFamily, whose last member is used), a wout_*.nc, or a VMEC/DESC input file.

    From an input file only the boundary is used, so the returned Equilibrium is unsolved and
    carries no plasma current: usable only with --vacuum (`check_vacuum_choice`); see also
    `minor_radius`.
    """
    from desc.equilibrium import EquilibriaFamily, Equilibrium

    if os.path.exists(spec):
        base = os.path.basename(spec)
        if spec.endswith(".h5"):
            from desc.io import load

            eq = load(spec)
            if isinstance(eq, EquilibriaFamily):
                eq = eq[-1]
            return eq, "h5"
        if base.startswith("wout") and spec.endswith(".nc"):
            from desc.vmec import VMECIO

            return VMECIO.load(spec), "wout"
        from desc.geometry import FourierRZToroidalSurface

        surf = FourierRZToroidalSurface.from_input_file(spec)
        return Equilibrium(surface=surf, NFP=surf.NFP, sym=surf.sym, M=surf.M, N=surf.N), "input (boundary only)"
    from desc.examples import get

    return get(spec), "desc.examples"


def minor_radius(eq, override=None):
    """DESC's "a", or the override. For an UNSOLVED equilibrium built from a boundary, DESC's "a"
    comes from the interior initial guess and can be off by ~1% (MEASURED 0.85% on one case);
    pass `--a` if the bounds must be exact."""
    return float(override) if override else float(eq.compute("a")["a"])


# ---------------------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------------------
# Gil et al., PRE 114, 025202 (2026), Table III (stated set), for a reactor with a = 1.7 m.
REACTOR_A = 1.7
REACTOR = dict(length_per_half_period=160.0, kappa=1.0, kappa_MS=0.1, d_cc=0.8, d_pc=1.5, convex_excess=None)
UPPER = ("L", "Ltot", "kappa", "kms", "convex", "planar")  # bound keys where the value must stay below the bound
LOWER = ("dcc", "dpc")


def paper_bounds(eq, nc, a=None, bounds_json=None):
    """The bounds to judge against, for `nc` independent coils per half period.

    Default: the reactor set above scaled by f = a / 1.7 (lengths x f, curvature / f,
    kappa_MS / f^2), with the half-period length split evenly over the nc coils.
    `bounds_json`: a file with any of
      {"scale": "reactor"|"absolute", "length_per_half_period" or "L", "L_total", "kappa",
       "kappa_MS", "d_cc", "d_pc", "convex_excess"}
    "reactor" (default) values are scaled like the defaults; "absolute" values are used as given
    (metres, 1/m, 1/m^2; "L" is per coil). A value of null drops that bound (and its constraint).
    "L_total" bounds the SUM of the independent coils' lengths (one half period) instead of, or as
    well as, the per-coil "L"; set "L": null to use a total budget alone.
    "planar_max" (metres, never scaled) bounds each coil's worst out-of-plane knot distance;
    only spline coils have one, and it is None (unbounded) unless the bounds file sets it.
    "convex_excess" (radians, never scaled) bounds the total-curvature excess
    int |kappa| ds - 2 pi of each coil, which is 0 exactly when a planar coil is convex.
    Keys starting with "_" are comments.
    Returns dict(L, kappa, kms, dcc, dpc, convex) plus the scale factor f and the a used.
    """
    a = minor_radius(eq, a)
    f = a / REACTOR_A
    r = dict(REACTOR)
    scale = "reactor"
    if bounds_json:
        user = {k: v for k, v in json.load(open(bounds_json)).items() if not k.startswith("_")}
        scale = user.pop("scale", "reactor")
        r.update(user)

    def sc(v, p):
        return None if v is None else (v if scale == "absolute" else v * f**p)

    L = r["L"] if "L" in r else (None if r["length_per_half_period"] is None else r["length_per_half_period"] / nc)
    return dict(L=sc(L, 1), Ltot=sc(r.get("L_total"), 1), kappa=sc(r["kappa"], -1), kms=sc(r["kappa_MS"], -2),
                dcc=sc(r["d_cc"], 1), dpc=sc(r["d_pc"], 1), convex=r.get("convex_excess"),
                planar=r.get("planar_max"), f=f, a=a, scale=scale)


def scaled_set(b, inner):
    """The bound set moved `inner` (a fraction) inside every bound; None stays None."""
    return {k: (None if b[k] is None else b[k] * (1 - inner if k in UPPER else 1 + inner)) for k in UPPER + LOWER}


def fmt_bounds(b):
    return "  ".join(f"{k} {'-' if b[k] is None else format(b[k], '.4g')}" for k in UPPER + LOWER)


# ---------------------------------------------------------------------------------------
# Coil helpers
# ---------------------------------------------------------------------------------------
def iter_unique(coilset):
    """Every leaf coil, flattening nested (Mixed)CoilSets: the independent coils."""
    from desc.coils import CoilSet, MixedCoilSet

    for c in coilset:
        if isinstance(c, (CoilSet, MixedCoilSet)):
            yield from iter_unique(c)
        else:
            yield c


def n_unique(coilset):
    return len(list(iter_unique(coilset)))


def expected_coils(coilset, eq):
    """Physical coil count for a stellarator-symmetric, NFP-periodic set of these coils."""
    return n_unique(coilset) * eq.NFP * (2 if eq.sym else 1)


def is_arc(coilset):
    return hasattr(next(iter_unique(coilset)), "B")


def is_spline(coilset):
    return hasattr(next(iter_unique(coilset)), "knots")


def arc_breaks(coil):
    return np.arange(int(coil.B)) * 2.0 * np.pi / int(coil.B)


def coil_breaks(coil):
    """Parameter values of a coil's C0 corners: arc hinges, or a spline's break knots.

    Empty for a smooth coil. This is what makes a grid composite ACROSS the corners,
    which is what the length quadrature needs (see desc/objectives/_coils.py, the
    `override_grid=False` note: a uniform grid on a C0 curve converges only O(h))."""
    if hasattr(coil, "B"):
        return arc_breaks(coil)
    iv = np.asarray(getattr(coil, "intervals", [[]]))
    if iv.size == 0:
        return np.zeros(0)
    return np.sort(np.asarray(coil.knots)[np.asarray(iv)[:, 0].astype(int)])


def is_piecewise(coilset):
    """True if the coils have C0 corners (arc hinges, or spline breaks)."""
    return coil_breaks(next(iter_unique(coilset))).size > 0


def gl_grid(breaks, m):
    """Composite Gauss-Legendre, m nodes per piece; spacing[:, 2] carries the weights, so DESC's
    ds = grid.spacing[:, 2] makes its sums a composite rule across arc corners."""
    m = max(int(m), 4)
    xg, wg = np.polynomial.legendre.leggauss(m)
    e = np.sort(np.mod(np.asarray(breaks, dtype=float), 2 * np.pi))
    edges = np.concatenate([e, [e[0] + 2 * np.pi]])
    S, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        h = 0.5 * (hi - lo)
        S.append(lo + h * (xg + 1.0))
        W.append(wg * h)
    s = np.concatenate(S)
    nodes = np.zeros((s.size, 3))
    nodes[:, 2] = np.mod(s, 2 * np.pi)
    spacing = np.ones((s.size, 3))
    spacing[:, 2] = np.concatenate(W)
    return Grid(nodes, sort=False, jitable=False, spacing=spacing)


def source_grid(coilset, gl_m=24, uniform_n=400):
    """Biot-Savart source grid: composite GL for arcs, LinearGrid(N=uniform_n) otherwise.

    SPLINE coils take the uniform branch deliberately. `SplineXYZCoil._compute_A_or_B`
    uses the Hanson-Hirshman STRAIGHT-SEGMENT formula through the grid points rather
    than quadrature (coils.py: `biot_savart_hh`), so the grid supplies polyline
    vertices and its `ds` weights are ignored; GL nodes would cluster at the span ends
    and leave long chords in the middle."""
    c0 = next(iter_unique(coilset))
    if hasattr(c0, "B"):
        return gl_grid(arc_breaks(c0), gl_m), "gl"
    return LinearGrid(N=uniform_n), "uniform"


def quad_grid(coilset, gl_m=24, uniform_n=400, spline_m=8):
    """Grid for LENGTH (and any other arclength integral), composite across C0 corners.

    Same grid as `source_grid` for arc and smooth coils, so nothing changes for them.
    For splines it is composite GL per SPAN (the spline is only C2 at interior knots,
    so a rule that straddles them loses its spectral rate) -- MEASURED on the cut30
    coilset, GL per span and GL per break interval both give 35.999992 m, against
    35.999774 m for a 4001-point polyline, which cuts the C0 corners."""
    c0 = next(iter_unique(coilset))
    if hasattr(c0, "B"):
        return gl_grid(arc_breaks(c0), gl_m), "gl"
    if hasattr(c0, "knots"):
        return gl_grid(np.asarray(c0.knots), spline_m), "gl/span"
    return LinearGrid(N=uniform_n), "uniform"


# ---------------------------------------------------------------------------------------
# Dense checks
# ---------------------------------------------------------------------------------------
def surface_tree(eq, ntheta=256, nzeta=1024):
    """KD-tree of the FULL plasma boundary (no symmetry), for the dense d_pc."""
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    ze = np.linspace(0, 2 * np.pi, nzeta, endpoint=False)
    x = np.asarray(eq.compute("x", grid=LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False),
                              basis="xyz")["x"])
    return cKDTree(x)


def check_vacuum_choice(eq, kind, vacuum):
    """Stop early if the plasma field is wanted but the equilibrium cannot supply it."""
    if not vacuum and kind.startswith("input"):
        raise SystemExit("an equilibrium from an input file is boundary-only and unsolved, so it has no plasma "
                         "currents for virtual casing; solve it first, or pass --vacuum")


_BN_CACHE = {}
VC_MULT = 2  # virtual-casing source grid = VC_MULT x DESC's default (M_grid, N_grid)


def vc_source_grid(eq, m=None, n=None):
    """Virtual-casing source grid. DESC's default (M_grid, N_grid) was 0.18% of |B| off at the
    max on helios_repro (2x: 6e-5 T rms, 4e-4 T max vs 3x), so 2x is the default."""
    return LinearGrid(rho=np.array([1.0]), M=m or VC_MULT * eq.M_grid, N=n or VC_MULT * eq.N_grid, NFP=eq.NFP,
                      sym=False)


def _bn_grid_data(eq, nphi, ntheta, vacuum):
    """Boundary points, normals and the plasma field (virtual casing, rpz) on the check grid:
    one full field period, uniform, NFP = eq.NFP. The virtual-casing FFT interpolator needs
    exactly that, and truncates silently-ish (a warning) if the grid is coarser than its source
    grid, so ntheta, nphi are raised to at least the source grid's (2 M + 1, 2 N + 1).
    The equilibrium is fixed, so this is computed once per (eq, grid)."""
    key = (id(eq), nphi, ntheta, vacuum)
    if key not in _BN_CACHE:
        from desc.integrals.singularities import compute_B_plasma

        src = vc_source_grid(eq)
        grid = LinearGrid(rho=np.array([1.0]), theta=max(ntheta, src.num_theta), zeta=max(nphi, src.num_zeta),
                          NFP=eq.NFP, sym=False)
        d = eq.compute(["R", "phi", "Z", "n_rho"], grid=grid)
        x = np.stack([d["R"], d["phi"], d["Z"]], axis=1)
        Bp = np.zeros_like(x) if vacuum else np.asarray(compute_B_plasma(eq, grid, src, chunk_size=64))
        _BN_CACHE[key] = (eq, x, np.asarray(d["n_rho"]), Bp)  # eq kept so its id is not reused
    return _BN_CACHE[key][1:]


def bn_stats(coilset, eq, src_grid, nphi=72, ntheta=96, vacuum=False):
    """|B.n| on one field period (uniform, nphi x ntheta, raised to the virtual-casing source
    grid's size if smaller), with B = B_coils + B_plasma (B_plasma = 0 if `vacuum`).
    Unweighted means, as in the sweeps (which used 32 x 32 on half a period).

    mean, max: mean and max |B.n| / <|B|>      (the sweep's metric; <|B|> of the total field)
    local:     mean of |B.n| / |B| pointwise   (Kruger et al.'s "average percent error" / 100)
    max_T:     max |B.n| in tesla
    """
    x, n, Bp = _bn_grid_data(eq, nphi, ntheta, vacuum)
    B = np.asarray(coilset.compute_magnetic_field(x, basis="rpz", source_grid=src_grid)) + Bp
    Bn = np.abs(np.sum(B * n, axis=1))
    Bmag = np.linalg.norm(B, axis=1)
    mB = float(np.mean(Bmag))
    return dict(mean=float(np.mean(Bn)) / mB, max=float(np.max(Bn)) / mB, local=float(np.mean(Bn / Bmag)),
                max_T=float(np.max(Bn)))


def curve_metrics(c):
    brk = coil_breaks(c)
    arc = brk.size > 0
    kmax = float(np.max(np.abs(np.asarray(c.compute(["|curvature|"], grid=LinearGrid(N=400))["|curvature|"]))))
    qg = gl_grid(np.asarray(c.knots) if hasattr(c, "knots") else brk, 24) if arc else LinearGrid(N=400)
    d = c.compute(["|curvature|", "x_s", "ds"], grid=qg)
    k = np.abs(np.asarray(d["|curvature|"]))
    sp = np.linalg.norm(np.asarray(d["x_s"]), axis=-1)
    ds = np.asarray(d["ds"])
    L = float(np.sum(sp * ds))
    corner = 0.0
    if arc:
        eps = 1e-7
        s = np.concatenate([[(b - eps) % (2 * np.pi), (b + eps) % (2 * np.pi)] for b in brk])
        t = np.asarray(c.compute("x_s", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False))["x_s"])
        t = t / np.linalg.norm(t, axis=1)[:, None]
        corner = float(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", t[0::2], t[1::2]), -1, 1))).max())
    # concave turning int (|kappa_s| - kappa_s) ds per planar segment (whole coil, or each arc):
    # for a planar coil this is the total-curvature excess of Kruger et al. 2026, Eq. 5
    convex, kmin = (float("nan"), float("nan"))
    if type(c).__name__ in ("FourierPlanarCoil", "PiecewisePlanarArcCoil", "PolarPlanarArcCoil",
                            "SplineXYZCoil"):
        from convexity import convexity_excess

        convex, kmin = convexity_excess(c)
    # worst out-of-plane knot distance: 0 by construction for the structurally planar
    # classes, a real measurement only for splines (sandbox/planarity.py)
    planar = 0.0
    if hasattr(c, "knots"):
        from planarity import planarity_excess

        planar = planarity_excess(c)
    return dict(L=L, kappa=max(kmax, float(k.max())), kms=float(np.sum(k**2 * sp * ds) / L), corner=corner,
                convex=convex, kappa_s_min=kmin, planar=planar)


def linked_pairs(pos):
    """1-based linked pairs (pairwise Gauss linking integral on the given polylines)."""
    out = []
    for i, j in itertools.combinations(range(pos.shape[0]), 2):
        p, q = pos[i], pos[j]
        dp, dq = np.roll(p, -1, 0) - p, np.roll(q, -1, 0) - q
        r = (p + 0.5 * dp)[:, None, :] - (q + 0.5 * dq)[None, :, :]
        lk = np.sum(np.einsum("ijk,ijk->ij", r, np.cross(dp[:, None, :], dq[None, :, :]))
                    / np.linalg.norm(r, axis=-1) ** 3) / (4 * np.pi)
        if abs(lk) > 0.5:
            out.append([i + 1, j + 1, round(float(lk), 3)])
    return out


def evaluate(cs, eq, tree=None, vacuum=False):
    """Dense metrics: per-coil geometry, all-pairs d_cc on 401 nodes, d_pc against the full
    boundary, linked pairs, <|B.n|>/<|B|> (plasma field included unless `vacuum`), and the
    physical coil count."""
    tree = tree if tree is not None else surface_tree(eq)
    per = [curve_metrics(c) for c in iter_unique(cs)]
    pos = np.asarray(cs._compute_position(grid=LinearGrid(N=200)))
    trees = [cKDTree(p) for p in pos]
    d_cc = min(trees[i].query(pos[j])[0].min() for i, j in itertools.combinations(range(len(pos)), 2))
    d_pc = min(tree.query(p)[0].min() for p in pos)
    link = linked_pairs(np.asarray(cs._compute_position(grid=LinearGrid(N=64))))
    st = bn_stats(cs, eq, source_grid(cs)[0], vacuum=vacuum)
    return dict(per_coil=per, L=max(p["L"] for p in per), L_total=sum(p["L"] for p in per),
                kappa=max(p["kappa"] for p in per),
                kms=max(p["kms"] for p in per), corner=max(p["corner"] for p in per),
                convex=max(p["convex"] for p in per), planar=max(p["planar"] for p in per),
                d_cc=float(d_cc), d_pc=float(d_pc), linked=len(link), linked_pairs=link,
                bn=st["mean"], bn_max=st["max"], bn_local=st["local"], bn_max_T=st["max_T"], vacuum=bool(vacuum),
                n_coils=int(pos.shape[0]))


# margin name -> (metric key, bound key)
_MARGINS = dict(L=("L", "L"), L_total=("L_total", "Ltot"), kappa=("kappa", "kappa"), kMS=("kms", "kms"),
                convex=("convex", "convex"), planar=("planar", "planar"),
                d_cc=("d_cc", "dcc"), d_pc=("d_pc", "dpc"))


def margins(m, b):
    """value / bound for every bound that is set: satisfied when <= 1 for L, kappa, kMS, convex
    and >= 1 for d_cc, d_pc. Unset bounds are left out."""
    return {k: m[mk] / b[bk] for k, (mk, bk) in _MARGINS.items() if b.get(bk) is not None}


def failed(m, b):
    g = margins(m, b)
    bad = [k for k, v in g.items() if (v < 1 if _MARGINS[k][1] in LOWER else v > 1)]
    return bad + (["linked"] if m["linked"] > 0 else [])


def verdict(m, b):
    bad = failed(m, b)
    return "FEASIBLE" if not bad else "FAILS " + ",".join(bad)


def report(m, b):
    return "  ".join(f"{k} {v:.3f}" for k, v in margins(m, b).items())


def result_json_next_to(path):
    """The run record next to a result h5, if any."""
    js = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), "result.json")))
    return json.load(open(js[0])) if js else None


def memwatch(every=30):
    def loop():
        while True:
            fields = {}
            for line in open("/proc/self/status"):
                if line.startswith(("VmRSS", "VmHWM")):
                    fields[line.split(":")[0]] = int(line.split()[1]) / 1024**2
            print(f"  [mem] RSS {fields.get('VmRSS', float('nan')):.2f} GB  "
                  f"peak {fields.get('VmHWM', float('nan')):.2f} GB", flush=True)
            time.sleep(every)

    threading.Thread(target=loop, daemon=True).start()


# ---------------------------------------------------------------------------------------
# Arc fits of existing coils
# ---------------------------------------------------------------------------------------
def resample_between(coil, s_hinge, n_per=360, n_dense=4000):
    """Points of `coil` (xyz), resampled so that the given parameter values (increasing mod 2 pi)
    fall at equal intervals: n_per points per piece, evenly spaced in arclength within each piece.
    Passed to PiecewisePlanarArcCoil.from_values as uniform samples, this puts its hinges at
    exactly those points, whatever the source coil's parameterization."""
    s = np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")["x"])
    xc = np.vstack([x, x[:1]])
    arcl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xc, axis=0), axis=1))])  # at s and 2 pi
    sc = np.concatenate([s, [2 * np.pi]])
    total = arcl[-1]
    h = np.mod(np.asarray(s_hinge, dtype=float), 2 * np.pi)
    lh = np.interp(h, sc, arcl)
    out = []
    for k in range(len(h)):
        lo, hi = lh[k], lh[(k + 1) % len(h)]
        if hi <= lo:
            hi += total
        ls = np.mod(lo + (hi - lo) * np.arange(n_per) / n_per, total)
        out.append(np.stack([np.interp(ls, arcl, np.concatenate([xc[:, j]])) for j in range(3)], 1))
    return np.concatenate(out)


def long_axis_hinges(coil, n=2000):
    """Parameter values of the two points of `coil` farthest apart (its long axis)."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")["x"])
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return sorted([s[i], s[j]])


def arclength_hinges(coil, B, s0, offset=0.0, n=4000):
    """B parameter values of `coil` equally spaced in arclength, the first at arclength
    `offset` (fraction of the coil length) after parameter s0."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")["x"])
    xc = np.vstack([x, x[:1]])
    arcl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xc, axis=0), axis=1))])
    sc = np.concatenate([s, [2 * np.pi]])
    total = arcl[-1]
    l0 = np.interp(np.mod(s0, 2 * np.pi), sc, arcl)
    lh = np.mod(l0 + total * (offset + np.arange(B) / B), total)
    return list(np.interp(lh, arcl, sc))


def direction_hinges(coil, psi_deg, n=4000):
    """Parameter values of the two points of a (planar) coil that are extreme along the in-plane
    direction at angle psi from the plane's 'vertical' (the in-plane direction closest to +Z),
    turning toward the in-plane direction that points away from the machine axis.
    For B=2 arcs (each a graph over its chord) the hinges have to sit at such a pair of extremes,
    so psi is the one real hinge-placement choice per coil."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")["x"])
    c = x.mean(0)
    nrm = np.linalg.svd(x - c)[2][-1]
    ev = np.array([0.0, 0.0, 1.0]) - nrm[2] * nrm
    ev /= np.linalg.norm(ev)
    eh = np.cross(nrm, ev)
    radial = np.array([c[0], c[1], 0.0]) / np.hypot(c[0], c[1])
    if eh @ radial < 0:
        eh = -eh
    p = np.radians(psi_deg)
    u = np.cos(p) * ev + np.sin(p) * eh
    proj = (x - c) @ u
    return sorted([s[int(np.argmax(proj))], s[int(np.argmin(proj))]])


def centroid_line_hinges(coil, psi_deg, n=4000):
    """Parameter values where a planar coil crosses the line through its centroid at in-plane
    angle psi from vertical (same frame as `direction_hinges`). Any psi gives a valid B=2 split
    for polar arcs (PolarPlanarArcCoil), since a convex coil is star-shaped about any interior point."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.asarray(coil.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False),
                                basis="xyz")["x"])
    c = x.mean(0)
    nrm = np.linalg.svd(x - c)[2][-1]
    ev = np.array([0.0, 0.0, 1.0]) - nrm[2] * nrm
    ev /= np.linalg.norm(ev)
    eh = np.cross(nrm, ev)
    if eh @ np.array([c[0], c[1], 0.0]) < 0:
        eh = -eh
    p = np.radians(psi_deg)
    u = np.cos(p) * ev + np.sin(p) * eh
    side = (x - c) @ np.cross(nrm, u)  # signed distance from the cut line
    k = np.nonzero(np.sign(side) != np.sign(np.roll(side, -1)))[0]
    if len(k) != 2:
        raise ValueError(f"cut line at psi={psi_deg} crosses the coil {len(k)} times (not convex?)")
    out = []
    for i in k:  # linear interpolation of the crossing parameter
        j = (i + 1) % n
        t = side[i] / (side[i] - side[j])
        out.append(float(np.mod(s[i] + t * (2 * np.pi / n), 2 * np.pi)))
    return sorted(out)
