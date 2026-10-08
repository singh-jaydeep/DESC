"""Why does the KNOWN-GOOD arc configuration collapse under the c0 harness?.

The `coil_bundle/` runs (2026-08-05) optimized PiecewisePlanarArcCoil well with a
weights-based ladder. The c0 harness collapses on the same class. This script holds the
GEOMETRY fixed at the bundle's own cold init and swaps the CONFIGURATION one knob at a
time, so a single difference can be implicated or exonerated.

Seven knobs, each either "bundle" or "c0":

  src      Biot-Savart / field_grid. bundle: composite Gauss-Legendre, gl_m=24 nodes per
           arc, partition bracketing the hinges, GL weights in spacing[:,2].
           c0: LinearGrid(N=50), uniform in t, no weights, blind to the corners.
  curv     bundle: LinearGrid(N=50), bounds(-1,2), normalize_target=False, weight 750 --
           deliberately COARSE so the intended C0 faceting is not penalized.
           c0: LinearGrid(N=201) (403 nodes), bounds(0,4.657), normalize_target=True,
           weight sqrt(10)*403/(2pi) = 202.8 -- the 2pi/num_nodes quadrature weight is
           cancelled so the bound binds pointwise.
  dist     bundle: hard min. c0: use_softmin=True, softmin_alpha=100.
  weights  bundle ladder rung 0: qflux 20, ccdist 100, pcdist 10, curv 750, length 20,
           linking 50.  c0 "pen": qflux 1, everything else sqrt(mu0)=sqrt(10).
  current  bundle: FixCoilCurrent on coil 0 only.
           c0: FixSumCoilCurrent + a LinkingCurrentConsistency residual term.
  length   bundle: bounds(0, 6.0) absolute, grid=src.  c0: bounds(0, rescale_to_eq),
           grid=LinearGrid(N=50).
  linking  CoilSetLinkingNumber grid: src (bundle) vs LinearGrid(N=50) (c0).

Usage:
    python c0/bundle_probe.py --base bundle                 # the known-good rung 0
    python c0/bundle_probe.py --base c0                     # every knob at c0
    python c0/bundle_probe.py --base bundle --swap src      # one knob at a time
    python c0/bundle_probe.py --report c0/runs/bundle
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

KNOBS = (
    "src",
    "curv_grid",
    "curv_bound",
    "curv_weight",
    "dist",
    "weights",
    "current",
    "length",
    "linking",
)
# "curv" is a macro for its three components, which are separable and were confounded
# in the first sweep: the c0 side changes the grid (101 -> 403 nodes), the bound
# ((-1,2) physical -> (0,4.657) with normalize_target), AND multiplies the weight by
# num_nodes/(2pi) = 64 to cancel CoilCurvature's own quadrature weight.
MACROS = {"curv": ("curv_grid", "curv_bound", "curv_weight")}

CURV_BOUND = 4.657  # c0 side, 1/m, as used in the overnight runs
MU0 = 10.0  # c0 "pen" mode: engineering terms at weight sqrt(mu0)


def _cfg(base, swaps):
    other = {"bundle": "c0", "c0": "bundle"}
    cfg = {k: base for k in KNOBS}
    for s in swaps:
        for k in MACROS.get(s, (s,)):
            if k not in KNOBS:
                raise SystemExit(
                    f"unknown knob {s!r}; choose from " f"{KNOBS + tuple(MACROS)}"
                )
            cfg[k] = other[base]
    return cfg


# --- helpers copied VERBATIM from coil_bundle/code/driver_stage2.py --------------
# Copied, not imported: that module imports CoilSetPlanarity, which does not exist
# anywhere in this repo's history, so the bundle's driver cannot run against this tree
# at all. Keep these in sync by eye if the bundle is ever re-run.


def _iter_unique(coilset):
    """Yield every leaf coil, flattening nested (Mixed)CoilSets."""
    from desc.coils import CoilSet, MixedCoilSet

    for c in coilset:
        if isinstance(c, (CoilSet, MixedCoilSet)):
            yield from _iter_unique(c)
        else:
            yield c


def _arc_breaks(coil):
    """Parameter values where the arc integrand is non-smooth (the B hinges)."""
    import numpy as np

    return np.arange(int(coil.B)) * 2.0 * np.pi / int(coil.B)


def _gl_grid(breaks, m):
    """Composite Gauss-Legendre: m nodes inside each [breaks[i], breaks[i+1]) piece.

    spacing[:,2] carries the GL weights so DESC's ds = grid.spacing[:,2] turns
    biot_savart_quad's sum into the composite rule. jitable=False is required.
    """
    import numpy as np

    from desc.grid import Grid

    m = max(int(m), 4)
    xg, wg = np.polynomial.legendre.leggauss(m)
    e = np.sort(np.mod(np.asarray(breaks, dtype=float), 2 * np.pi))
    edges = np.concatenate([e, [e[0] + 2 * np.pi]])
    S, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        h = 0.5 * (b - a)
        S.append(a + h * (xg + 1.0))
        W.append(wg * h)
    s = np.concatenate(S)
    w = np.concatenate(W)
    nodes = np.zeros((s.size, 3))
    nodes[:, 2] = np.mod(s, 2 * np.pi)
    spacing = np.ones((s.size, 3))
    spacing[:, 2] = w
    return Grid(nodes, sort=False, jitable=False, spacing=spacing)


def _cold_init(
    eq, num_coils, B, M, r_over_a=3.0, proj_knots=120, fit_method="parameter"
):
    """Bundle cold init: modular coils -> PiecewisePlanarArcCoil(B, M).

    `fit_method` is passed straight to `from_values`; "parameter" (the default)
    reproduces every start built before the option existed. See CONVENTIONS.md
    sec 4.1 for what "chord" changes and why B=2 needs it.
    """
    import numpy as np

    from desc.coils import (
        CoilSet,
        MixedCoilSet,
        PiecewisePlanarArcCoil,
        initialize_modular_coils,
    )
    from desc.grid import LinearGrid

    base = initialize_modular_coils(eq, num_coils=num_coils, r_over_a=r_over_a)
    arcs = []
    for coil in base:
        pts = np.asarray(
            coil.compute("x", grid=LinearGrid(N=proj_knots), basis="xyz")["x"]
        )
        arcs.append(
            PiecewisePlanarArcCoil.from_values(
                current=float(coil.current),
                coords=pts,
                B=B,
                M=M,
                basis="xyz",
                fit_method=fit_method,
            )
        )
    return MixedCoilSet(*[CoilSet(c, NFP=base.NFP, sym=base.sym) for c in arcs])


def _arc_single_mode(cs, scale):
    """Replace each arc's fitted `shape` with ONE sine mode: w(t) = a*sin(pi t).

    WHY THIS EXISTS -- the B=2 start, measured 2026-09-10. `from_values` fits the
    transverse deviation of a circular arc as a graph over its own CHORD. At B=2 that
    chord is a DIAMETER and the target profile is a semicircle, `2r*sqrt(t(1-t))`, which
    has VERTICAL tangents at both endpoints. A truncated sine series cannot hold that,
    so the fit ripples -- and since kappa is a second derivative, every extra mode makes
    it WORSE, not better. MEASURED at r/a=3.3, kappa_MS against a 5.648225 bound:

        M=3   7.9275      M=5  13.5284      M=8  19.9798      M=10  27.1420

    so B=2 read structurally infeasible on mean-square curvature by 40.4%, and no r/a
    rescues it (kappa_MS falls with r/a but d_cc collapses faster; the windows never
    overlap -- kappa_MS is still 1.248x over at r/a=3.50 where d_cc has fallen to 1.107).

    The start does not have to BE a circle. A single mode is smooth, has kappa = 0 at the
    hinges by construction (`sin(m*pi*t)` and its second derivative both vanish at
    t=0,1), and peaks at only `a*pi^2/L_chord^2` at mid-arc -- a rounded lens rather than
    a rippling semicircle. `scale` is the bulge amplitude as a fraction of the chord's
    half-length, so `scale=1` puts each arc's midpoint exactly on the circle the fit was
    taken from and the coil touches that circle at every hinge and every mid-arc.

    Hinges, tilts and `arc_ref` are left exactly as `from_values` set them, so the arc
    planes and the frozen gauge are unchanged; only the in-plane profile is replaced.
    """
    import numpy as np

    if scale is None:
        return cs
    for c in _iter_unique(cs):
        B, M = int(c.B), int(c.M)
        H = np.asarray(c.hinges).reshape(B, 3)
        fitted = np.asarray(c.shape).reshape(B, M)
        new = np.zeros((B, M))
        for i in range(B):
            # KEEP THE FITTED SIGN. `shape` is a SIGNED deviation along the arc plane's
            # in-plane normal, and at B=2 the fit comes back NEGATIVE (-0.641 at
            # r/a=3.3). Forcing it positive turns every coil inside out: MEASURED d_pc
            # 0.05984 against a 0.171091 bound, the coil bulging through the plasma.
            sgn = np.sign(fitted[i, 0]) or 1.0
            if scale == 0:
                # pure TRUNCATION: keep the fitted fundamental, drop only the ripple
                # modes. The smallest change that removes the oscillation.
                new[i, 0] = fitted[i, 0]
            else:
                chord = np.linalg.norm(H[(i + 1) % B] - H[i])
                new[i, 0] = sgn * scale * chord / 2.0
        c.shape = new.reshape(-1)
    return cs


def _perturb(cs, seed, scale):
    """Seeded multistart jitter in the arc's own parameters (bundle's planararc arm)."""
    import numpy as np

    if not scale or scale <= 0:
        return cs
    rng = np.random.default_rng(seed)
    for c in _iter_unique(cs):
        B, M = int(c.B), int(c.M)
        H = np.asarray(c.hinges).reshape(B, 3)
        Rm = float(np.mean(np.linalg.norm(H - H.mean(0), axis=1))) + 1e-9
        c.hinges = (H + rng.normal(0.0, scale * Rm, size=H.shape)).reshape(-1)
        tl = np.asarray(c.tilts).reshape(B)
        c.tilts = tl + rng.normal(0.0, scale * np.pi, size=tl.shape)
        sh = np.asarray(c.shape).reshape(B, M)
        sc = scale * (np.mean(np.abs(sh)) + 0.05)
        c.shape = (sh + rng.normal(0.0, sc, size=sh.shape)).reshape(-1)
    return cs


def _tilt_planar(cs, deg, axis="rock"):
    """Hand-designed COHERENT tilt of a FourierPlanar set: rotate every unique coil's
    normal by the SAME `deg` about a machine axis, centers held fixed.

    Deterministic, not seeded -- this is the basin probe of `HANDOFF_multistart.md`
    sec 3 done by hand rather than by random draw. The `paper` cold start is an EXACT
    circle (all 28 non-zero `r_n` at ~1e-16, `kappa = 1/r_0` to machine precision), so
    a tilt is a rigid rotation of each coil about its own centre and changes NO
    curvature whatever -- it moves the coil without deforming it. Rotation about a
    coil's own `normal` is an exact null for a circle, which leaves exactly two
    non-degenerate axes:

        axis="rock" -> local radial e_R, so the coil plane leans in Z
        axis="yaw"  -> machine z-hat, so the coil plane leans radially in/out

    Applied to unique coils only; the CoilSet's NFP/sym container regenerates the
    images, so field-period symmetry is preserved by construction.
    """
    import numpy as np

    if not deg:
        return cs
    th = np.deg2rad(float(deg))
    for c in _iter_unique(cs):
        ctr = np.asarray(c.center, float)
        k = (
            np.array([ctr[0], ctr[1], 0.0])
            if axis == "rock"
            else np.array([0.0, 0.0, 1.0])
        )
        k = k / np.linalg.norm(k)
        v = np.asarray(c.normal, float)
        m = np.linalg.norm(v)
        v = (
            v * np.cos(th)
            + np.cross(k, v) * np.sin(th)
            + k * np.dot(k, v) * (1 - np.cos(th))
        )
        c.normal = v / np.linalg.norm(v) * m
    return cs


def _mode_kick(cs, amp, ns=(2, 3), axis_unused=None):
    """Hand-designed SHAPE perturbation of a FourierPlanar set: add a fixed ripple to
    the named cos modes of every unique coil.

    Amplitude for mode `n` is ``amp * r_0 / (1 + n**2)``. The weight is the point: a
    mode-n radial ripple of amplitude `a` on a circle of radius `r_0` perturbs
    curvature by ``(n**2 - 1) * a / r_0**2``, so a FLAT kick across modes -- e.g. one
    scaled by ``max|r_n|``, which for this start IS `r_0`, since the cold start is an
    exact circle and every other coefficient is ~1e-16 -- puts ~n^2 too much curvature
    into the high modes and returns a hairy coil. MEASURED at amp=0.05: flat jitter
    reaches kappa ~40 /m against an 11.64 bound, the weighted form ~2.2 /m. The weight
    equalizes the CURVATURE contribution instead, which is the quantity that disfigures.

    Deterministic: same sign and size on every unique coil, so it is inspectable and
    reproducible rather than seeded. `ns` are cos modes (positive n).
    """
    import numpy as np

    if not amp:
        return cs
    for c in _iter_unique(cs):
        n = np.asarray(c.r_basis.modes[:, 2])
        r = np.asarray(c.r_n, float).copy()
        r0 = float(np.abs(r[n == 0][0]))
        for k in ns:
            k = int(k)
            sel = np.where(n == k)[0]
            if sel.size:
                r[sel[0]] += amp * r0 / (1.0 + k**2)
        c.r_n = r
    return cs


def _tilt_planar_spec(cs, spec):
    """PER-COIL tilt of a FourierPlanar set, as (rock_deg, yaw_deg) for each unique coil.

    `_tilt_planar` applies the SAME angle to every coil, which cannot reach the
    orientation that separates the continued basin from the cold one. MEASURED off the
    saved checkpoints (CONVENTIONS.md 8p): between `planar_N7_cold` and all three
    `*_signed` continuations, coils 1-2 agree to within 1.5 deg while coil 3 swings
    yaw -19 deg / rock +9 deg and coil 4 swings rock -10 deg. The mean of that delta is
    only -4.3 deg yaw, so a COHERENT tilt is very nearly orthogonal to it -- which is
    why `rock 5` and the 3-axis probe both landed inside 0.15% of the baseline.

    `spec` is a sequence of (rock_deg, yaw_deg) pairs, one per unique coil, applied in
    iteration order. Axes are exactly `_tilt_planar`'s: rock = local radial e_R (the
    plane leans in Z), yaw = machine z-hat (the plane leans radially). Rotation about a
    coil's own normal is an exact null for a circle, so these remain the only two
    non-degenerate axes. Rock is applied first, then yaw.

    Centers are held fixed and the normal's MAGNITUDE is preserved, so as with
    `_tilt_planar` this is a rigid rotation of each coil about its own centre on the
    exactly-circular paper start: it changes NO curvature whatever. Applied to unique
    coils only; the CoilSet's NFP/sym container regenerates the images, so field-period
    symmetry is preserved by construction.
    """
    import numpy as np

    if spec is None:
        return cs
    spec = [(float(a), float(b)) for a, b in spec]
    coils = list(_iter_unique(cs))
    if len(spec) != len(coils):
        raise ValueError(
            f"--tilt-spec has {len(spec)} pairs but the coilset has "
            f"{len(coils)} unique coils"
        )
    if not any(r or y for r, y in spec):
        return cs

    def _rot(v, k, th):
        k = k / np.linalg.norm(k)
        return (
            v * np.cos(th)
            + np.cross(k, v) * np.sin(th)
            + k * np.dot(k, v) * (1 - np.cos(th))
        )

    for c, (rock_deg, yaw_deg) in zip(coils, spec):
        ctr = np.asarray(c.center, float)
        v = np.asarray(c.normal, float)
        m = np.linalg.norm(v)
        if rock_deg:
            v = _rot(v, np.array([ctr[0], ctr[1], 0.0]), np.deg2rad(rock_deg))
        if yaw_deg:
            v = _rot(v, np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg))
        c.normal = v / np.linalg.norm(v) * m
    return cs


def _kappa_max(cs, grid):
    import numpy as np

    return max(
        float(np.max(np.abs(c.compute("curvature", grid=grid)["curvature"])))
        for c in _iter_unique(cs)
    )


def build(
    cfg, num_coils=3, B=6, M=2, seed=1, perturb=0.15, qf_rung=20.0, curv_wmult=0.0
):
    """Bundle cold init + the objective stack selected by `cfg`."""
    import numpy as np
    from bench.bench import rescale_to_eq

    import desc.objectives as O
    from desc.examples import get
    from desc.grid import LinearGrid

    eq = get("precise_QA")

    # --- geometry: the bundle's own cold init, identical for every config ---------
    cs = _cold_init(eq, num_coils, B, M)
    kappa_clean = _kappa_max(cs, LinearGrid(N=50))
    cs = _perturb(cs, seed, perturb)
    kappa_init = _kappa_max(cs, LinearGrid(N=50))

    coil_grid = LinearGrid(N=50)
    dist_grid = LinearGrid(N=150)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    plasma_grid_nosym = LinearGrid(M=25, N=25, NFP=eq.NFP)
    curv_grid = LinearGrid(N=201)

    gl = _gl_grid(_arc_breaks(next(_iter_unique(cs))), 24)
    src = gl if cfg["src"] == "bundle" else coil_grid

    R0 = float(eq.compute("R0", grid=LinearGrid(M=8, N=8, NFP=eq.NFP, sym=False))["R0"])
    lim = rescale_to_eq(R0)

    # --- weights -----------------------------------------------------------------
    if cfg["weights"] == "bundle":
        w_qf, w_cc, w_pc, w_curv, w_len, w_link = (
            qf_rung,
            100.0,
            10.0,
            750.0,
            20.0,
            50.0,
        )
    else:
        w = float(np.sqrt(MU0))
        w_qf, w_cc, w_pc, w_curv, w_len, w_link = 1.0, w, w, w, w, w

    terms = [
        O.QuadraticFlux(
            eq,
            field=cs,
            eval_grid=plasma_grid,
            field_grid=src,
            vacuum=True,
            weight=w_qf,
            bs_chunk_size=10,
        )
    ]

    # --- distances ---------------------------------------------------------------
    soft = cfg["dist"] == "c0"
    sk = dict(use_softmin=True, softmin_alpha=100.0) if soft else {}
    cc_min = lim["d_cc_min"] if cfg["weights"] == "c0" else 0.1
    pc_min = lim["d_cs_min"] if cfg["weights"] == "c0" else 0.25
    terms += [
        O.CoilSetMinDistance(
            cs,
            bounds=(cc_min, np.inf),
            normalize_target=(cfg["dist"] == "c0"),
            grid=(coil_grid if cfg["dist"] == "c0" else dist_grid),
            weight=w_cc,
            dist_chunk_size=2,
            **sk,
        ),
        O.PlasmaCoilSetMinDistance(
            eq,
            cs,
            bounds=(pc_min, np.inf),
            normalize_target=(cfg["dist"] == "c0"),
            plasma_grid=(plasma_grid_nosym if soft else plasma_grid),
            coil_grid=(coil_grid if cfg["dist"] == "c0" else dist_grid),
            eq_fixed=True,
            weight=w_pc,
            dist_chunk_size=2,
            **sk,
        ),
    ]

    # --- curvature (three separable knobs) ---------------------------------------
    kgrid = coil_grid if cfg["curv_grid"] == "bundle" else curv_grid
    kbounds, knorm = (
        ((-1.0, 2.0), False)
        if cfg["curv_bound"] == "bundle"
        else ((0.0, CURV_BOUND), True)
    )
    kmult = (
        curv_wmult
        if curv_wmult > 0
        else (1.0 if cfg["curv_weight"] == "bundle" else kgrid.num_nodes / (2 * np.pi))
    )
    kweight = w_curv * kmult
    terms.append(
        O.CoilCurvature(
            cs, bounds=kbounds, normalize_target=knorm, grid=kgrid, weight=kweight
        )
    )

    # --- length ------------------------------------------------------------------
    if cfg["length"] == "bundle":
        terms.append(
            O.CoilLength(
                cs, bounds=(0, 6.0), normalize_target=True, grid=src, weight=w_len
            )
        )
    else:
        terms.append(
            O.CoilLength(
                cs,
                bounds=(0, lim["length_max"]),
                normalize_target=True,
                grid=coil_grid,
                weight=w_len,
            )
        )

    # --- linking number ----------------------------------------------------------
    terms.append(
        O.CoilSetLinkingNumber(
            cs,
            target=0,
            weight=w_link,
            grid=(src if cfg["linking"] == "bundle" else coil_grid),
        )
    )

    # --- currents ----------------------------------------------------------------
    if cfg["current"] == "bundle":
        idx = [False] * len(cs)
        idx[0] = True
        lin = (O.FixCoilCurrent(cs, indices=idx),)
    else:
        terms.append(O.LinkingCurrentConsistency(eq, cs, eq_fixed=True, weight=w_link))
        lin = (O.FixSumCoilCurrent(cs),)

    info = dict(
        cfg=cfg,
        curv_wmult=float(kmult),
        curv_weight=float(kweight),
        curv_nodes=int(kgrid.num_nodes),
        num_coils=num_coils,
        B=B,
        M=M,
        seed=seed,
        perturb=perturb,
        qf_rung=qf_rung,
        R0=R0,
        length_max_c0=lim["length_max"],
        d_cc_min_c0=lim["d_cc_min"],
        d_cs_min_c0=lim["d_cs_min"],
        kappa_clean=kappa_clean,
        kappa_init=kappa_init,
        src_nodes=int(src.num_nodes),
        src_kind=cfg["src"],
        dim_x=int(cs.dim_x),
        res=f"B={B} M={M}",
    )
    return eq, cs, O.ObjectiveFunction(tuple(terms)), lin, info


def term_costs(objective, things):
    """0.5*||f_scaled_error||^2 contributed by each sub-objective, and its dim_f."""
    import numpy as np

    x = objective.x(*things)
    f = np.asarray(objective.compute_scaled_error(x))
    out, i = {}, 0
    for sub in objective.objectives:
        n = int(sub.dim_f)
        blk = f[i : i + n]
        out[sub.__class__.__name__] = dict(
            cost=float(0.5 * np.sum(blk**2)),
            dim_f=n,
            rms=float(np.sqrt(np.mean(blk**2))) if n else 0.0,
            absmax=float(np.max(np.abs(blk))) if n else 0.0,
        )
        i += n
    assert i == f.size, f"block sizes {i} != dim_f {f.size}"
    return out


def summarize(step_log):
    """Trust-radius trajectory + acceptance statistics."""
    import numpy as np

    if not step_log:
        return {}
    tr = np.array([r["trust_radius"] for r in step_log])
    acc = np.array([r["accepted"] for r in step_log])
    rr = np.array([r["reduction_ratio"] for r in step_log])
    out = dict(
        n_attempts=len(step_log),
        n_accepted=int(acc.sum()),
        accept_frac=float(acc.mean()),
        tr_max=float(tr.max()),
        tr_final=float(tr[-1]),
        tr_collapse=float(tr.max() / max(tr[-1], 1e-300)),
        tr_median=float(np.median(tr)),
    )
    if acc.any():
        out["rr_accepted_median"] = float(np.median(rr[acc]))
    if (~acc).any():
        out["rr_rejected_min"] = float(rr[~acc].min())
        out["rr_rejected_median"] = float(np.median(rr[~acc]))
    for k in ("rank", "n_null", "n_sv", "n_x", "null_frac", "s_min", "s_max"):
        if k in step_log[0]:
            v = np.array([r[k] for r in step_log], dtype=float)
            out[k + "_final"] = float(v[-1])
            out[k + "_median"] = float(np.median(v))
    return out


def run(  # noqa: D103
    base,
    swaps,
    maxiter,
    outdir,
    tag,
    num_coils,
    B,
    M,
    seed,
    perturb,
    qf_rung,
    metrics,
    curv_wmult=0.0,
):
    cfg = _cfg(base, swaps)
    # build() imports driver_stage2, which calls set_device at import time -- nothing
    # that pulls in desc (and therefore jax) may run before it.
    eq, cs, objective, lin, info = build(
        cfg, num_coils, B, M, seed, perturb, qf_rung, curv_wmult
    )

    from desc.optimize import Optimizer

    print(f"[cfg] {cfg}")
    print(
        f"[init] dim_x={info['dim_x']} {info['res']} kappa {info['kappa_clean']:.2f}"
        f" -> {info['kappa_init']:.2f}  src={info['src_kind']}"
        f" {info['src_nodes']} nodes",
        flush=True,
    )

    (new_cs,), res = Optimizer("lsq-exact").optimize(
        cs,
        objective=objective,
        constraints=lin,
        maxiter=maxiter,
        verbose=3,
        ftol=1e-8,
        xtol=1e-10,
        gtol=1e-10,
        copy=True,
        options=dict(tr_method="svd", track_steps=True),
    )

    rec = dict(
        info,
        base=base,
        swaps=sorted(swaps),
        maxiter=maxiter,
        message=str(res["message"]),
        success=bool(res["success"]),
        nit=int(res["nit"]),
        nfev=int(res["nfev"]),
        cost=float(res["cost"]),
        optimality=float(res["optimality"]),
        step_log=list(res["step_log"]),
    )
    rec["summary"] = summarize(rec["step_log"])
    from desc.grid import LinearGrid

    rec["kappa_max_final"] = _kappa_max(new_cs, LinearGrid(N=201))
    rec["kappa_max_init"] = _kappa_max(cs, LinearGrid(N=201))
    try:
        rec["term_cost_final"] = term_costs(objective, (new_cs,))
        rec["term_cost_initial"] = term_costs(objective, (cs,))
    except Exception as e:  # noqa: BLE001
        rec["term_cost_error"] = repr(e)
    if metrics:
        from bench.bench import paper_metrics

        rec["metrics_initial"] = paper_metrics(cs, eq)
        rec["metrics"] = paper_metrics(new_cs, eq)

    os.makedirs(outdir, exist_ok=True)
    if tag is None:
        tag = base + ("_swap-" + "-".join(sorted(swaps)) if swaps else "")
        if curv_wmult > 0:
            tag += f"_kw{curv_wmult:g}"
    new_cs.save(os.path.join(outdir, f"bp_{tag}.h5"))
    with open(os.path.join(outdir, f"bp_{tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {outdir}/bp_{tag}.json")
    report_one(rec)


def report_one(rec):
    """Print the decisive columns for one run."""
    s = rec.get("summary", {})
    print(f"\n=== {rec['base']} swaps={rec['swaps'] or '-'} ===")
    print(f"  {rec['message']}")
    print(
        f"  nit={rec['nit']} cost={rec['cost']:.4e} "
        f"optimality={rec['optimality']:.3e}"
    )
    if rec.get("metrics"):
        print(
            f"  normalized_BdotN {rec['metrics_initial']['normalized_BdotN']:.4e}"
            f" -> {rec['metrics']['normalized_BdotN']:.4e}"
        )
    if s:
        print(
            f"  trust radius: max={s['tr_max']:.3e} final={s['tr_final']:.3e} "
            f"collapse={s['tr_collapse']:.3e}x  median={s['tr_median']:.3e}"
        )
        print(
            f"  attempts={s['n_attempts']} accepted={s['n_accepted']} "
            f"({s['accept_frac']*100:.0f}%)  "
            f"rr_acc_med={s.get('rr_accepted_median', float('nan')):.4f} "
            f"rr_rej_min={s.get('rr_rejected_min', float('nan')):.3e}"
        )
        print(
            f"  rank={s.get('rank_final')} n_null={s.get('n_null_final')} "
            f"n_sv={s.get('n_sv_final')} n_x={s.get('n_x_final')} "
            f"null_frac_med={s.get('null_frac_median', float('nan')):.3f}"
        )
    tc = rec.get("term_cost_final")
    if tc:
        tot = sum(v["cost"] for v in tc.values()) or 1.0
        t0 = rec.get("term_cost_initial", {})
        print(f"  {'term':<28} {'dim_f':>6} {'cost0':>10} {'cost':>10} {'share':>6}")
        for k, v in sorted(tc.items(), key=lambda kv: -kv[1]["cost"]):
            c0v = t0.get(k, {}).get("cost", float("nan"))
            print(
                f"  {k:<28} {v['dim_f']:>6} {c0v:>10.3e} {v['cost']:>10.3e} "
                f"{v['cost']/tot*100:>5.1f}%"
            )


def report(outdir):
    """Print every run found in `outdir` as one table."""
    hdr = (
        f"{'config':<28} {'nit':>4} {'w_curv':>9} {'tr_max':>10} {'tr_fin':>10} "
        f"{'collapse':>9} {'acc%':>5} {'optim':>9} {'bn_end':>10} {'kap_f':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(glob.glob(os.path.join(outdir, "bp_*.json"))):
        r = json.load(open(p))
        s = r.get("summary", {})
        name = r["base"] + ("+" + ",".join(r["swaps"]) if r["swaps"] else "")
        if r.get("curv_wmult", 1.0) not in (1.0,) and "kw" in os.path.basename(p):
            name += f"[kw{r['curv_wmult']:g}]"
        bn = r.get("metrics", {}).get("normalized_BdotN", float("nan"))
        print(
            f"{name:<28} {r['nit']:>4} {r.get('curv_weight', float('nan')):>9.2f} "
            f"{s.get('tr_max', 0):>10.3e} "
            f"{s.get('tr_final', 0):>10.3e} {s.get('tr_collapse', 0):>9.2e} "
            f"{s.get('accept_frac', 0)*100:>5.0f} {r['optimality']:>9.3e} "
            f"{bn:>10.3e} {r.get('kappa_max_final', float('nan')):>6.2f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", choices=["bundle", "c0"], default="bundle")
    ap.add_argument("--swap", default="", help="comma-separated knobs to flip")
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--out", default="c0/runs/bundle")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--num_coils", type=int, default=3)
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--perturb", type=float, default=0.15)
    ap.add_argument("--qf_rung", type=float, default=20.0)
    ap.add_argument(
        "--curv_wmult",
        type=float,
        default=0.0,
        help="override the CoilCurvature weight multiplier on top of "
        "w_curv; 0 = use whatever the curv_weight knob selects "
        "(1 for bundle, num_nodes/2pi for c0)",
    )
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--report", metavar="OUTDIR", default=None)
    a = ap.parse_args()

    if a.report:
        report(a.report)
        sys.exit(0)

    sys.path.insert(0, HERE)
    if a.device == "gpu":
        from desc import set_device

        set_device("gpu")

    run(
        a.base,
        [s for s in a.swap.split(",") if s],
        a.maxiter,
        a.out,
        a.tag,
        a.num_coils,
        a.B,
        a.M,
        a.seed,
        a.perturb,
        a.qf_rung,
        not a.no_metrics,
        a.curv_wmult,
    )
