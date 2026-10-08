"""Section 7 of `c0/CONVENTIONS.md`: the record every run must emit.

Feasibility is adjudicated ONLY here -- hard minima on a fine grid, never from
an objective's own returned values (which are softmin-biased and/or scaled).

    from verify import paper_bounds, verify
    rec = verify(coilset, eq, src_grid=src)
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

VERIFY_N = 400  # coil grid for every adjudicated geometric number
# Surface grid for the ADJUDICATED plasma-coil distance. A min-distance converges
# slowly: 0.195697 at M=25, 0.192683 at M=48, 0.191575 at M=128. The adjudicator must
# sit past that, not alongside the objective's own grid.
SURF_MN = (96, 96)


def paper_bounds(eq):
    """Hurwitz et al. Table 1 rescaled to this equilibrium. See CONVENTIONS sec 1, 3.

    NB `eq.compute("R0")`, NOT `compute_scaling_factors(eq)["R0"]` -- the latter is
    the R00 Fourier coefficient (1.0 here), which silently returns the paper's
    unscaled numbers.
    """
    from bench.bench import rescale_to_eq

    from desc.grid import LinearGrid

    R0 = float(eq.compute("R0", grid=LinearGrid(M=8, N=8, NFP=eq.NFP, sym=False))["R0"])
    lim = rescale_to_eq(R0)
    return {
        "R0": R0,
        "length_max": lim["length_max"],
        "kappa_max": lim["kappa_max"],
        "kappa_ms_max": lim["kappa_ms_max"],  # diagnostic only
        "d_cc_min": lim["d_cc_min"],
        "d_pc_min": lim["d_cs_min"],
    }


def arc_ref_margin(coilset):
    """Worst |e_par . arc_ref| over the set; -> 1.0 means the frozen gauge is degenerating.

    Returns None for representations that carry no frozen arc frame.
    """
    from bundle_probe import _iter_unique

    vals = [
        float(np.max(np.abs(np.asarray(c.arc_ref_margin))))
        for c in _iter_unique(coilset)
        if hasattr(c, "arc_ref_margin")
    ]
    return max(vals) if vals else None


def bn_grid_drift(coilset, eq, src_grid, nphi=32, ntheta=32):
    """`normalized_BdotN` recomputed at 2x source resolution.

    The bundle's own driver does this, and it would have caught its 7/31
    quadrature defect on day one. For a C0 arc the source grid must bracket every
    hinge, so `src_grid` is REQUIRED -- passing None here silently falls back to
    DESC's uniform `LinearGrid(N=2*coil.N+5)`, which does not.
    """
    from bench.bench import paper_metrics

    base = _bn_at(coilset, eq, src_grid, nphi, ntheta)
    fine = _bn_at(coilset, eq, _refine(coilset, src_grid), nphi, ntheta)
    return {
        "bn": base,
        "bn_2x_source": fine,
        "bn_grid_drift": abs(fine - base) / max(abs(base), 1e-300),
    }


def _refine(coilset, src_grid):
    """Double the source resolution, preserving the grid's character."""
    from bundle_probe import _arc_breaks, _gl_grid, _iter_unique

    from desc.grid import LinearGrid

    c0 = next(_iter_unique(coilset))
    if src_grid is None:
        return None
    # a composite-GL grid carries non-uniform spacing; rebuild it at 2x nodes/arc
    if hasattr(c0, "B") and not isinstance(src_grid, LinearGrid):
        m = int(round(src_grid.num_nodes / int(c0.B)))
        return _gl_grid(_arc_breaks(c0), 2 * m)
    return LinearGrid(N=2 * int((src_grid.num_nodes - 1) // 2) + 1)


def bn_stats(coilset, eq, src_grid, nphi=32, ntheta=32):
    """MEAN and MAX |B.n|/<|B|> on an explicit source grid, from ONE evaluation.

    Both statistics come from the same `Bn` array, so they can never again be
    reported off different grids. That split was a real bug: the headline mean was
    moved onto the solve's source grid when 8b.2 landed, but `max_BdotN_over_B` kept
    being taken from `paper_metrics`, which has NO `source_grid` parameter and so
    silently used DESC's default uniform LinearGrid(N=2*coil.N+5). On a faceted C0 arc
    that grid reads 1.7x-10.1x high (measured across al_B2..al_B5), and the resulting
    max/mean ratio tracked the inflation factor rather than the geometry -- 25.9 at
    B=3 against 3.7-4.2 for the two SMOOTH representations, which the defect cannot
    touch. Anything comparing peak error across representations must use this.
    """
    from desc.grid import LinearGrid

    grid = LinearGrid(
        theta=np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
        zeta=np.linspace(0, np.pi / eq.NFP, nphi, endpoint=False),
        NFP=1,
        sym=False,
    )
    xyz = np.asarray(eq.compute("x", grid=grid, basis="xyz")["x"])
    Bn = np.abs(
        np.asarray(
            coilset.compute_Bnormal(eq.surface, eval_grid=grid, source_grid=src_grid)[0]
        )
    )
    Bmag = np.linalg.norm(
        np.asarray(
            coilset.compute_magnetic_field(xyz, basis="xyz", source_grid=src_grid)
        ),
        axis=1,
    )
    mean_B = float(np.mean(Bmag))
    return {
        "mean": float(np.mean(Bn)) / mean_B,
        "max": float(np.max(Bn)) / mean_B,
        "mean_AbsB": mean_B,
    }


def _bn_at(coilset, eq, src_grid, nphi, ntheta):
    """`normalized_BdotN` in the paper's convention, on an explicit source grid."""
    return bn_stats(coilset, eq, src_grid, nphi, ntheta)["mean"]


def hard_geometry(coilset, eq, n=VERIFY_N):
    """Exact minima over the discretized geometry. Never softmin, never scaled."""
    from bench.bench import geometry_metrics, linking_metrics

    from desc.grid import LinearGrid

    g = LinearGrid(N=n)
    m = geometry_metrics(coilset, eq, g, surf_MN=SURF_MN)
    m.update(linking_metrics(coilset, g))
    m.update(_linking_gl(coilset))
    m.update(_dcc_segment(coilset, n, m["coil_coil_distance"]))
    m.update(_length_gl(coilset, m["max_length"]))
    return m


def _length_gl(coilset, uniform_value):
    """Coil length on the composite-GL grid, which is the trustworthy estimate.

    `bench.geometry_metrics` integrates length on a uniform LinearGrid. On a C0
    piecewise-arc coil that converges O(1/N) and always from ABOVE, so it manufactures
    violations of a bound the coil actually satisfies. MEASURED on v11, whose true
    length is 5.153326 against a 5.153350 bound (24 microns INSIDE):

        composite-GL gl_m=24/64   5.153326   <- converged, satisfied
        uniform N=400             5.157882   <- reported VIOLATED
        uniform N=12800           5.153418

    The margin at a converged solve is far smaller than any uniform grid's quadrature
    error, so the grid, not the geometry, decides the verdict. CONVENTIONS.md sec 8i.
    Same class of defect as `_dcc_segment`; the AL constrains length on the GL grid, so
    the check must too or it will contradict the solve it is checking.
    """
    from bundle_probe import _arc_breaks, _gl_grid, _iter_unique

    import desc.objectives as O
    from desc.grid import LinearGrid

    c0 = next(_iter_unique(coilset))
    grid = _gl_grid(_arc_breaks(c0), 24) if hasattr(c0, "B") else LinearGrid(N=400)
    o = O.CoilLength(coilset, bounds=(0.0, np.inf), normalize_target=False, grid=grid)
    o.build(verbose=0)
    gl = float(np.max(np.asarray(o.compute(coilset.params_dict))))
    return {"max_length_uniform": float(uniform_value), "max_length": gl}


def _dcc_segment(coilset, n, point_value):
    """Coil-coil distance from SEGMENT-segment minima, which is the trustworthy one.

    `bench.geometry_metrics` minimizes over distances between grid POINTS, so the
    closest approach has to land on a node. That cannot represent a crossing: at the v8
    iterate it reported 6.4e-03 m of clearance on the N=150 grid (1.6e-03 at N=400) for
    coils that INTERSECT -- the point-sampled minimum tracks the node spacing rather
    than converging, while the segment-segment distance drives to 2.7e-08 m.
    CONVENTIONS.md sec 8i.

    This matters here specifically because `checks["d_cc"]` decides feasibility. A
    feasibility check that point-samples shares the constraint's blind spot and will
    certify a self-intersecting coilset as feasible -- measured: an `lsq-exact` solve
    returning "Optimization terminated successfully" with the coils intersecting.

    Both are reported. `dcc_disagreement` is the flag.
    """
    import desc.objectives as O
    from desc.grid import LinearGrid

    o = O.CoilSetMinDistance(
        coilset,
        bounds=(0.0, np.inf),
        normalize_target=False,
        grid=LinearGrid(N=n),
        use_softmin=False,
        dist_chunk_size=2,
        distance_method="segment",
    )
    o.build(verbose=0)
    seg = float(np.min(np.asarray(o.compute(coilset.params_dict))))
    return {
        "coil_coil_distance_point": float(point_value),
        "coil_coil_distance": seg,
    }


def _linking_gl(coilset):
    """Linking number on the composite-GL grid, which is the trustworthy estimate.

    `bench.linking_metrics` sums a polygonal Gauss integral on a UNIFORM grid. For a
    C0 arc that converges O(1/N) (see CONVENTIONS 8e), and on nearly-touching coils the
    1/|r|^3 integrand makes it fail outright: at the v8 iterate (d_cc = 6.4 mm) it
    reported |Lk| = 5.23e-01 and "2 linked pairs" where the GL estimate gives 9.3e-06.
    0.523 is not an integer, so it was quadrature noise, not a topological link -- but
    it is enough to fail `checks["unlinked"]` and reject a sound coilset.

    Both are reported. `linking_disagreement` is the flag: when the two estimates differ
    by more than 0.1, neither verdict should be trusted without a closer look.
    """
    from bundle_probe import _arc_breaks, _gl_grid, _iter_unique

    import desc.objectives as O
    from desc.grid import LinearGrid

    c0 = next(_iter_unique(coilset))
    grid = _gl_grid(_arc_breaks(c0), 24) if hasattr(c0, "B") else LinearGrid(N=400)
    o = O.CoilSetLinkingNumber(coilset, target=0, grid=grid)
    o.build(verbose=0)
    v = float(np.max(np.abs(np.asarray(o.compute(coilset.params_dict)))))
    return {"max_abs_linking_number_gl": v, "n_linked_pairs_gl": int(v > 0.5)}


def verify(coilset, eq, src_grid, bounds=None, n=VERIFY_N, label=""):
    """The full section-7 record for one solved coilset."""
    from bench.bench import paper_metrics

    b = bounds or paper_bounds(eq)
    g = hard_geometry(coilset, eq, n=n)
    # `paper_metrics` passes NO source_grid, so it silently uses DESC's default uniform
    # LinearGrid(N=2*coil.N+5) = 111 nodes. On a faceted C0 arc that is badly
    # under-resolved: measured 1.632502e-03 against a converged 3.355912e-04, i.e. 4.9x
    # too high. The uniform rule needs >6400 nodes to reach what composite GL gets with
    # 120. So the HEADLINE bn is computed on the solve's own source grid, and the
    # default-grid value is kept only to expose the gap.
    pm = paper_metrics(coilset, eq)
    drift = bn_grid_drift(coilset, eq, src_grid)
    src_stats = bn_stats(coilset, eq, src_grid)

    # A binding constraint saturates: the AL drives it ONTO its bound, so the final
    # value differs from the bound only by solver tolerance. A strict `<=` then reports
    # a converged, feasible result as INFEASIBLE. MEASURED on v12, whose length exceeds
    # its bound by 4.265e-10 m -- 0.43 NANOMETRES, relative 8.3e-11, five orders below
    # the ctol=1e-6 the solve was run to, and with the AL's own constr_violation at
    # 7.5e-10. That is saturation, not violation. FEAS_RTOL is a floor on what counts
    # as a real violation; it is deliberately far tighter than any grid effect in 8i
    # (the smallest of those is length's +5.3 mm, i.e. 1e-3 relative), so it forgives
    # only floating-point saturation and never a quadrature artifact.
    FEAS_RTOL = 1e-8
    checks = {
        "length": g["max_length"] <= b["length_max"] * (1 + FEAS_RTOL),
        "kappa": g["max_max_kappa"] <= b["kappa_max"] * (1 + FEAS_RTOL),
        "d_cc": g["coil_coil_distance"] >= b["d_cc_min"] * (1 - FEAS_RTOL),
        "d_pc": g["coil_surface_distance"] >= b["d_pc_min"] * (1 - FEAS_RTOL),
        # adjudicate on the GL estimate; see `_linking_gl` for why the uniform-grid one
        # cannot be trusted on close coils
        "unlinked": g["n_linked_pairs_gl"] == 0,
    }
    margins = {
        "length": g["max_length"] / b["length_max"],
        "kappa": g["max_max_kappa"] / b["kappa_max"],
        "d_cc": g["coil_coil_distance"] / b["d_cc_min"],
        "d_pc": g["coil_surface_distance"] / b["d_pc_min"],
    }
    return {
        "label": label,
        "bounds": b,
        "verify_grid_N": n,
        "normalized_BdotN": drift["bn"],  # on the solve's source grid -- the result
        "normalized_BdotN_default_grid": pm["normalized_BdotN"],  # trap, see above
        "bn_default_grid_ratio": pm["normalized_BdotN"] / max(drift["bn"], 1e-300),
        "normalized_BdotN_areawt": pm["normalized_BdotN_areawt"],
        # on the solve's OWN source grid, from the same evaluation as the mean
        "max_BdotN_over_B": src_stats["max"],
        # the default-grid value, kept ONLY to expose the gap -- this is what used to
        # be reported as `max_BdotN_over_B` and it is inflated on faceted coils
        "max_BdotN_over_B_default_grid": pm["max_BdotN_over_B"],
        "max_default_grid_ratio": (
            pm["max_BdotN_over_B"] / max(src_stats["max"], 1e-300)
        ),
        "mean_AbsB": pm["mean_AbsB"],
        **drift,
        "max_length": g["max_length"],
        "max_length_uniform": g["max_length_uniform"],
        "length_grid_excess": g["max_length_uniform"] - g["max_length"],
        "max_kappa": g["max_max_kappa"],
        "max_kappa_MS": g["max_kappa_MS"],
        "max_arclength_variance": g["max_arclength_variance"],
        "d_cc": g["coil_coil_distance"],
        "d_cc_point": g["coil_coil_distance_point"],
        "dcc_disagreement": (
            g["coil_coil_distance_point"] / max(g["coil_coil_distance"], 1e-300)
        ),
        "d_pc": g["coil_surface_distance"],
        "n_linked_pairs": g["n_linked_pairs_gl"],
        "max_abs_linking_number": g["max_abs_linking_number_gl"],
        "n_linked_pairs_uniform": g["n_linked_pairs"],
        "max_abs_linking_number_uniform": g["max_abs_linking_number"],
        "linking_disagreement": abs(
            g["max_abs_linking_number_gl"] - g["max_abs_linking_number"]
        ),
        "currents": g["currents"],
        "total_current_base": g["total_current_base"],
        "arc_ref_margin": arc_ref_margin(coilset),
        "checks": checks,
        "margins": margins,
        "feasible": all(checks.values()),
        "failed": [k for k, v in checks.items() if not v],
    }


def print_verify(r):
    """One screenful, in physical units."""
    b = r["bounds"]
    print(f"\n=== VERIFY {r['label']} (hard minima, N={r['verify_grid_N']}) ===")
    print(
        f"  normalized_BdotN {r['normalized_BdotN']:.4e}"
        f"   (on the solve's source grid; max|Bn|/B {r['max_BdotN_over_B']:.3e},"
        f" max/mean {r['max_BdotN_over_B'] / max(r['normalized_BdotN'], 1e-300):.1f})"
    )
    if r.get("max_default_grid_ratio", 1.0) > 1.05:
        print(
            f"  default-grid max|Bn|/B {r['max_BdotN_over_B_default_grid']:.3e}"
            f"  = {r['max_default_grid_ratio']:.2f}x   <-- UNDER-RESOLVED, do not quote"
        )
    print(
        f"  paper_metrics default-grid bn {r['normalized_BdotN_default_grid']:.4e}"
        f"  = {r['bn_default_grid_ratio']:.2f}x"
        + (
            "   <-- DEFAULT GRID UNDER-RESOLVED, do not quote it"
            if r["bn_default_grid_ratio"] > 1.05
            else ""
        )
    )
    print(
        f"  bn_grid_drift    {r['bn_grid_drift']:.3e}"
        f"   (bn at 2x source: {r['bn_2x_source']:.4e})"
        + ("   <-- SOURCE GRID UNDER-RESOLVED" if r["bn_grid_drift"] > 1e-2 else "")
    )
    rows = [
        ("L_max   ", r["max_length"], b["length_max"], "<=", "length"),
        ("kappa   ", r["max_kappa"], b["kappa_max"], "<=", "kappa"),
        ("d_cc    ", r["d_cc"], b["d_cc_min"], ">=", "d_cc"),
        ("d_pc    ", r["d_pc"], b["d_pc_min"], ">=", "d_pc"),
    ]
    print(f"  {'':10s} {'value':>10s} {'bound':>10s} {'margin':>8s}")
    for name, v, bd, op, key in rows:
        # Label from the ADJUDICATED check, never a fresh strict comparison. A binding
        # constraint saturates onto its bound, so `checks` allows FEAS_RTOL for exactly
        # that (sec 8i); recomputing `v <= bd` here printed "VIOLATED" beside an overall
        # "FEASIBLE" on every run whose length block binds -- v12 by 0.43 nanometres.
        ok, m = r["checks"][key], r["margins"][key]
        print(
            f"  {name:10s} {v:10.5f} {op}{bd:9.5f} {m:8.3f}  {'ok' if ok else 'VIOLATED'}"
        )
    if r["length_grid_excess"] > 1e-4:
        print(
            f"       length on a uniform N={r['verify_grid_N']} grid reads"
            f" {r['max_length_uniform']:.5f}"
            f" (+{r['length_grid_excess']*1e3:.1f} mm)"
            f"   <-- O(1/N) on a C0 arc, always high; GL is the trustworthy one (8i)"
        )
    if r["dcc_disagreement"] > 1.5:
        print(
            f"       d_cc point-sampled reads {r['d_cc_point']:.5f}"
            f" = {r['dcc_disagreement']:.1f}x the segment value"
            f"   <-- closest approach falls BETWEEN nodes; segment is the"
            f" trustworthy one (sec 8i)"
        )
    print(
        f"  kappa_MS {r['max_kappa_MS']:10.5f}   (diagnostic, paper {b['kappa_ms_max']:.4f})"
    )
    print(
        f"  linked pairs {r['n_linked_pairs']}"
        f"   worst |Lk| {r['max_abs_linking_number']:.2e} (GL)"
        f"   arclength var {r['max_arclength_variance']:.2e}"
    )
    if r["linking_disagreement"] > 0.1:
        print(
            f"       uniform-grid estimate disagrees: |Lk|"
            f" {r['max_abs_linking_number_uniform']:.2e},"
            f" {r['n_linked_pairs_uniform']} pairs"
            f"   <-- quadrature failure on close coils, GL is the trustworthy one"
        )
    if r["arc_ref_margin"] is not None:
        print(
            f"  arc_ref_margin {r['arc_ref_margin']:.4f}"
            + ("   <-- REFREEZE" if r["arc_ref_margin"] > 0.95 else "")
        )
    print(
        f"  total current {r['total_current_base']:.6e}"
        f"   -> {'FEASIBLE' if r['feasible'] else 'INFEASIBLE: ' + ','.join(r['failed'])}"
    )


# ---------------------------------------------------------------------------
# solver health
# ---------------------------------------------------------------------------
# Every failure this campaign hit was visible in the numbers the solve already
# returns; each one cost hours because nothing looked at them. This does.
#
#   v3  n_outer=1 at nit=500                     -> outer loop dead
#   v5  mu 10 -> 42.65 with a frozen iterate     -> penalty runaway
#   8e  linking violated 2.3e-3 AT THE START     -> unsatisfiable equality
#
# The last one is the important one: an equality constraint already violated at the
# COLD START, by a margin that does not shrink by the end, is a target the solver can
# never reach. That single check would have found the linking-number defect in one run.


def feasible_descent(
    coilset,
    objective,
    constraints,
    tols=(1e-6, 1e-4, 1e-3),
    steps=None,
    ctol=1e-6,
    verbose=True,
):
    """Does a FEASIBLE descent step actually exist at this iterate? Direct test.

    Use this WITH `kkt_diag.kkt_split`, not instead of it. `kkt_split` answers the
    LINEAR question -- is `grad f` absorbed by the active constraint GRADIENTS -- and
    that question has a false-negative mode this campaign hit head on:

        A linear criterion is blind to constraint CURVATURE. Where the active set is
        sharply curved, the direction tangent to its linearization leaves the true
        feasible region almost immediately, so `|Pg|/|g|` stays large at a point that
        is effectively a constrained optimum.

    MEASURED (CONVENTIONS 8j): at the FourierPlanar iterate `kkt_split` reported
    `|Pg|/|g| = 0.268` with 6 wrong-sign multipliers -- reading as "real feasible
    descent left, the inner solve is genuinely short" -- while stepping along that same
    projected direction bought 1.05e-04 of cost before the worst relative violation
    grew 1500x. The point was effectively optimal; the linear test said otherwise. On
    the arc run that genuinely converged `kkt_split` gave 0.0067 and was exactly right.

    A step counts only if it lowers the objective WITHOUT pushing the worst constraint
    violation past `max(current violation, ctol)` -- the iterate itself may sit slightly
    outside, and a violation under `ctol` is what the solve was aiming for anyway.

    With NO constraint active at any threshold the direction is plain `-g`, so an
    unconstrained iterate is tested too rather than silently passing.

    Returns
    -------
    dict : `best_dcost`, `best_t`, `frac` (|Pg|/|g| for the winning threshold),
    `n_active`, `tol`, `f0`, `w0`, `stationary`.

    """
    import numpy as np

    from desc.objectives import ObjectiveFunction

    if steps is None:
        steps = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    of = ObjectiveFunction((objective, *constraints))
    of.build(verbose=0)
    fo = ObjectiveFunction((objective,))
    fo.build(verbose=0)
    x = of.x(coilset)
    n = objective.dim_f
    J = np.asarray(of.jac_scaled(x))
    g = J[:n].T @ np.asarray(of.compute_scaled_error(x))[:n]
    Jc = J[n:]
    lb, ub = (np.asarray(b)[n:] for b in of.bounds_scaled)
    gnorm = max(float(np.linalg.norm(g)), 1e-300)

    def worst_viol(xx):
        c = np.asarray(of.compute_scaled(xx))[n:]
        hi = np.where(np.isfinite(ub), c - ub, -np.inf)
        lo = np.where(np.isfinite(lb), lb - c, -np.inf)
        return float(np.max(np.concatenate([hi, lo])))

    f0, w0 = float(fo.compute_scalar(x)), worst_viol(x)
    # 1% slack on the violation budget: a step that leaves the worst violation
    # numerically where it already was must not be disqualified by drift in the last
    # digits. Without it the FourierPlanar iterate rejected every step because its own
    # violation ticked from 6.469e-07 to 6.474e-07.
    budget = max(w0, ctol) * 1.01
    c0 = np.asarray(of.compute_scaled(x))[n:]

    dirs = []
    for tol in tols:
        act = (np.isfinite(ub) & (c0 >= ub - tol)) | (
            np.isfinite(lb) & (c0 <= lb + tol)
        )
        if not act.any():
            continue
        A = Jc[act]
        d = -(g - A.T @ np.linalg.pinv(A @ A.T) @ A @ g)
        if np.linalg.norm(d) > 0:
            dirs.append((tol, int(act.sum()), d))
    if not dirs:  # nothing active anywhere -> plain steepest descent
        dirs.append((None, 0, -g))

    best = dict(
        best_dcost=0.0,
        best_t=0.0,
        frac=float("nan"),
        n_active=0,
        tol=None,
        f0=f0,
        w0=w0,
    )
    for tol, na, d in dirs:
        frac = float(np.linalg.norm(d) / gnorm)
        d = d / np.linalg.norm(d)
        for t in steps:
            ft, wt = float(fo.compute_scalar(x + t * d)), worst_viol(x + t * d)
            if wt <= budget and ft - f0 < best["best_dcost"]:
                best.update(
                    best_dcost=ft - f0, best_t=t, frac=frac, n_active=na, tol=tol
                )
    # Report the MAGNITUDE, do not just judge. `stationary` is a convenience verdict at
    # a 1e-4 RELATIVE cost threshold -- a judgment call, not a law. FourierPlanar's
    # remaining feasible descent is ~1e-05 relative, i.e. four orders below the 101x
    # flux gap between it and the arc representation: real, and irrelevant.
    best["best_dcost_rel"] = best["best_dcost"] / max(abs(f0), 1e-300)
    best["stationary"] = bool(best["best_dcost_rel"] > -1e-4)
    if verbose:
        print(
            f"  feasible_descent: f0={f0:.8e}  worst_viol={w0:+.3e}  "
            f"budget={budget:+.3e}"
        )
        print(
            f"    best feasible dcost {best['best_dcost']:+.3e} "
            f"({best['best_dcost_rel']:+.2e} relative) at t={best['best_t']:.0e}"
            f"  (|Pg|/|g|={best['frac']:.4f}, {best['n_active']} active @ tol="
            f"{best['tol']})"
        )
        print(
            "    -> "
            + (
                "STATIONARY (no useful feasible descent)"
                if best["stationary"]
                else "NOT stationary: feasible descent exists"
            )
        )
    return best


def constraint_breakdown(coilset, constraints):
    """Per-constraint max violation in scaled units, plus raw physical ranges.

    `constr_violation` is a single max over every row of every constraint, so it says
    nothing about WHICH constraint is responsible. This splits it.
    """
    import numpy as np

    from desc.objectives import ObjectiveFunction

    rows = []
    for c in constraints:
        of = ObjectiveFunction((c,))
        of.build(verbose=0)
        x = of.x(coilset)
        f = np.asarray(of.compute_scaled(x))
        lb, ub = (np.asarray(b) for b in of.bounds_scaled)
        viol = np.maximum(np.maximum(lb - f, f - ub), 0.0)
        try:
            raw = np.asarray(of.objectives[0].compute(coilset.params_dict))
        except Exception:  # noqa: BLE001 - objectives with a fixed eq take other args
            raw = np.array([np.nan])
        rows.append(
            {
                "name": c.name,
                "dim_f": int(c.dim_f),
                "raw_min": float(np.nanmin(raw)),
                "raw_max": float(np.nanmax(raw)),
                "max_viol": float(viol.max()),
                # an equality (target, or lb == ub) cannot absorb quadrature noise
                "equality": bool(np.allclose(lb, ub)),
            }
        )
    return rows


def solver_health(res, constraints, coils_start, coils_final, ctol=1e-6, maxiter=None):
    """Named, checkable failure modes from what the solve already returned."""
    import numpy as np

    diag = res.get("outer_diag") or []
    log = res.get("outer_log") or []
    reasons = {}
    for _, why in log:
        reasons[str(why)] = reasons.get(str(why), 0) + 1

    mu = np.asarray(res.get("penalty_param", [np.nan]))
    mu_final = float(np.mean(mu))
    mu0 = float(diag[0]["mu_mean"]) if diag else mu_final
    steps = [d.get("window_max_step", np.nan) for d in diag]
    nit = int(res.get("nit", 0))
    viol = float(res.get("constr_violation", np.nan))

    start = constraint_breakdown(coils_start, constraints)
    final = constraint_breakdown(coils_final, constraints)
    dom = max(final, key=lambda r: r["max_viol"]) if final else None

    flags = []
    if maxiter and nit >= maxiter and len(log) <= 1:
        flags.append("outer_loop_dead: <=1 outer update over the whole run")
    if mu0 > 0 and mu_final / mu0 > 10:
        flags.append(f"mu_runaway: {mu0:.3g} -> {mu_final:.3g}")
    if viol > ctol:
        flags.append(f"exit_infeasible: constr_violation {viol:.3e} > ctol {ctol:.1e}")
    # the check that would have caught 8e
    for a, b in zip(start, final):
        if a["max_viol"] > 100 * ctol and b["max_viol"] > 0.5 * a["max_viol"]:
            flags.append(
                f"unsatisfiable_constraint: '{a['name']}' violated "
                f"{a['max_viol']:.3e} at the START and {b['max_viol']:.3e} at the end"
                + (
                    " (EQUALITY -- target likely below its quadrature floor)"
                    if a["equality"]
                    else ""
                )
            )
    if steps and np.nanmax(steps) < 1e-4:
        flags.append("radius_never_recovered: no outer window exceeded a 1e-4 step")
    # NB `n_blocked` in outer_diag is `_sat & ~_pass`: rows that are SATISFIED but carry
    # a large slack residual, i.e. what the "violation" gate admits and "residual"
    # rejects. It does NOT count genuinely-violated rows -- those are excluded by BOTH
    # gates and never appear here. Do not read n_blocked==0 as "nothing was blocked".
    if diag and max(d.get("n_blocked", 0) for d in diag) > 0:
        flags.append(
            "residual_gate_blocking: satisfied rows with a large slack residual were "
            "withheld from the multiplier update (try --gate violation)"
        )
    # The v7 deadlock. When a row's violation is >= ctolk it is excluded from
    # `y <- y - mu*c` by both gates, so Conn & Gould's *penalty* branch is the only way
    # forward -- and `min_window_progress` can veto exactly that. Then the solver can
    # neither update the multiplier nor grow the penalty on the offending rows, and the
    # violation is frozen by construction.
    _deadlock = [
        d
        for d in diag
        if d.get("constr_violation", 0.0) >= d.get("ctolk", np.inf)
        and not d.get("mu_growth_allowed", True)
    ]
    if _deadlock:
        d = _deadlock[0]
        flags.append(
            f"penalty_growth_suppressed: at iteration {d['iteration']} the violation "
            f"{d['constr_violation']:.3e} was >= ctolk {d['ctolk']:.3e} (so violated "
            f"rows are excluded from the multiplier update) while mu growth was vetoed "
            f"by min_window_progress -- deadlock. Drop --mu-gate."
        )

    return {
        "nit": nit,
        "n_outer": int(res.get("n_outer", len(log))),
        "outer_reasons": reasons,
        "constr_violation": viol,
        "ctol": ctol,
        "mu_initial": mu0,
        "mu_final": mu_final,
        "mu_growth": (mu_final / mu0) if mu0 else float("nan"),
        "window_max_step_max": float(np.nanmax(steps)) if steps else float("nan"),
        "window_max_step_last": float(steps[-1]) if steps else float("nan"),
        "constraints_start": start,
        "constraints_final": final,
        "dominant_constraint": dom["name"] if dom else None,
        "flags": flags,
        "healthy": not flags,
    }


def print_health(h):
    """The screenful that turns a three-hour investigation into a glance."""
    print(f"\n=== SOLVER HEALTH ===")
    print(
        f"  nit={h['nit']}  n_outer={h['n_outer']}  reasons={h['outer_reasons']}\n"
        f"  constr_violation={h['constr_violation']:.3e} (ctol {h['ctol']:.1e})"
        f"   mu {h['mu_initial']:.4g} -> {h['mu_final']:.4g} (x{h['mu_growth']:.3g})\n"
        f"  window max step: peak={h['window_max_step_max']:.3e}"
        f" last={h['window_max_step_last']:.3e}"
    )
    print(f"  {'constraint':<30}{'max viol START':>16}{'max viol FINAL':>16}{'eq':>5}")
    for a, b in zip(h["constraints_start"], h["constraints_final"]):
        print(
            f"  {a['name']:<30}{a['max_viol']:>16.4e}{b['max_viol']:>16.4e}"
            f"{'  Y' if a['equality'] else '  .':>5}"
        )
    if h["healthy"]:
        print("  -> HEALTHY")
    else:
        print("  -> PROBLEMS:")
        for f in h["flags"]:
            print(f"       * {f}")
