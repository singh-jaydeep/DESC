"""Stage-2 coil optimization benchmark against Hurwitz, Landreman & Kaptanoglu.

Runs ONE case and writes a coilset + a JSON record. Sweeps are built by
`cases.py`, which imports `run_case` from here.

Reference
---------
S. Hurwitz, M. Landreman, A. Kaptanoglu, "Electromagnetic Coil Optimization for
Reduced Lorentz Forces". Data: doi.org/10.5281/zenodo.13913510.

Their published Pareto front (218 optima, `hurwitz_pareto_final.csv`) is the
target. Setup we are matching:

  equilibrium   Landreman-Paul 2021 precise QA, NFP=2, stellarator symmetric,
                major radius scaled to 1 m  (DESC's `precise_QA` has R0=1.0307 m,
                a=0.1718 m, A=6.0 -- the same configuration, 3% larger)
  coils         5 per half field period -> 20 total, order-16 Fourier XYZ
  currents      5 base currents summing to a FIXED total
  constraints   l <= 5 m, kappa <= 12 1/m, kappa_MS <= 6 1/m,
                d_cc >= 0.083 m, d_cs >= 0.166 m
  metric        normalized_BdotN = mean|B.n| / mean|B| on a 32x32 half-period
                surface grid, UNWEIGHTED means (see `paper_metrics`)

Two formulation notes that this script exists to get right
----------------------------------------------------------
1. CURRENT SCALE IS A DEGENERATE DIRECTION.  `QuadraticFlux(vacuum=True)` is
   exactly homogeneous of degree 2 in the coil currents: scaling every current
   by s scales the cost by s^2 (measured: s = 1, 0.5, 0.25, 0.1 gives cost
   99.76, 24.94, 6.235, 0.9976). So I -> 0 is a trivial minimizer. Meanwhile
   the *reported* metric mean|B.n|/mean|B| is scale-INVARIANT. An optimizer that
   shrinks current therefore reduces its cost while doing nothing for the metric
   we compare on -- real cost reduction, zero physics progress.

   `basic_AugLag.ipynb` fixes only coil 0's current (`FixCoilCurrent` with one
   index), leaving nc-1 currents free, and the saved runs took that ride: the
   nc=3 runs ended at 0.626x / 0.630x of the initial total current, with coil 1
   falling from 4.53e5 A to 6.7e4 A. The paper does not leave this open --
   `total_current.fix_all()` pins the sum. Use `current_mode="sum"`
   (`FixSumCoilCurrent`) to match; `current_mode="one"` reproduces the notebook
   for a controlled comparison.

   The paper's absolute current is arbitrary (their net gives |B| ~ 0.24 T, not
   precise_QA's ~1 T) and merely held fixed, which is all that is needed since
   the metric normalizes it out. Their `mean_AbsB > 0.22` filter is consistent
   with that scale.

2. DESC HAS NO MEAN-SQUARED-CURVATURE OBJECTIVE.  The paper constrains
   kappa_MS <= 6 1/m. `CoilIntegratedCurvature` is a different quantity
   (int kappa |dx/ds| ds, a convexity measure targeting 2*pi), so kappa_MS
   cannot be constrained directly. We measure it as a diagnostic and check it
   lands inside their bound rather than imposing it. Their pointwise
   kappa <= 12 1/m is representable and is imposed.
"""

import argparse
import itertools
import json
import os
import time
import uuid


def configure_device(device):
    """Must run before jax is imported anywhere."""
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    # Never preallocate: this box has one 8 GiB GPU that other sessions share.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


# ---------------------------------------------------------------------------
# unit conversion
# ---------------------------------------------------------------------------
# The paper states constraints at R0 = 1 m; DESC's precise_QA is R0 = 1.0307 m.
# Lengths scale as R0, curvatures as 1/R0. We convert their caps INTO DESC's
# metres, and convert our results BACK to R0 = 1 m for filter comparison.
PAPER_R0 = 1.0

PAPER_CONSTRAINTS = {  # at R0 = 1 m, Table 1
    "length_max": 5.0,  # m
    "kappa_max": 12.0,  # 1/m
    # 1/m^2, NOT 1/m: geometry_metrics computes (1/L)*int kappa^2 dl, with no square
    # root, so this is a MEAN-SQUARE curvature. It therefore rescales as 1/f^2, not
    # 1/f -- see rescale_to_eq. (Read as an RMS in 1/m the campaign's coils would sit
    # at 34-45% of it and it could never bind, which is not what the paper's own
    # front looks like; as a mean-square they sit at 68-118%.)
    "kappa_ms_max": 6.0,  # 1/m^2
    "d_cc_min": 0.083,  # m
    "d_cs_min": 0.166,  # m
}

# The filter actually applied in their analysis_tools.get_dfs(), which defines
# membership of the published front: 5% margin on Table 1, plus hygiene cuts
# that remove degenerate/detached coilsets.
PAPER_FILTER = {
    "margin_up": 1.05,
    "margin_low": 0.95,
    "d_cs_max": 0.375,
    "d_cc_max": 0.14,
    "length_min": 4.0,
    "arclength_var_max": 1e-2,
    "normalized_BdotN_max": 4e-3,
}


def rescale_to_eq(eq_R0):
    """Paper constraints expressed in metres / inverse-metres at DESC's R0."""
    f = eq_R0 / PAPER_R0
    c = PAPER_CONSTRAINTS
    return {
        "length_max": c["length_max"] * f,
        "kappa_max": c["kappa_max"] / f,
        # 1/m^2 -> f^2. Was `/ f`, which left the bound 3.07% too LOOSE at this R0
        # (5.821456 instead of 5.648276). No campaign verdict flips, but every
        # reported kappa_MS margin was that much optimistic.
        "kappa_ms_max": c["kappa_ms_max"] / f**2,
        "d_cc_min": c["d_cc_min"] * f,
        "d_cs_min": c["d_cs_min"] * f,
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def flat_coils(coilset):
    """Every leaf coil, flattening nested (Mixed)CoilSets."""
    out = []

    def walk(o):
        if hasattr(o, "coils") and not hasattr(o, "X_n") and not hasattr(o, "r_n"):
            for c in o.coils:
                walk(c)
        else:
            out.append(o)

    walk(coilset)
    return out


def paper_metrics(coilset, eq, nphi=32, ntheta=32):
    """`normalized_BdotN` exactly as the paper's code computes it, plus variants.

    Their optimization_tools.py does:

        BdotN     = np.mean(np.abs(np.sum(bs.B() * s.unitnormal(), axis=2)))
        mean_AbsB = np.mean(bs.AbsB())
        normalized_BdotN = BdotN / mean_AbsB

    -- UNWEIGHTED means over a `range="half period"` 32x32 surface grid. The
    paper's text writes this as <|B.n|>/<B> with angle brackets defined as
    flux-surface averages, but the code does not area-weight. Both are reported:
    `normalized_BdotN` (their convention, use for comparison) and
    `normalized_BdotN_areawt` (true area-weighted). They differ by O(10%), which
    matters at the 4e-5 end of their front.
    """
    import numpy as np

    from desc.grid import LinearGrid

    # half field period in zeta, full 2*pi in theta, matching simsopt's
    # "half period" range
    grid = LinearGrid(
        theta=np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
        zeta=np.linspace(0, np.pi / eq.NFP, nphi, endpoint=False),
        NFP=1,
        sym=False,
    )
    dat = eq.compute(["x", "|e_theta x e_zeta|"], grid=grid, basis="xyz")
    xyz = np.asarray(dat["x"])
    dS = np.asarray(dat["|e_theta x e_zeta|"])

    Bn = np.abs(np.asarray(coilset.compute_Bnormal(eq.surface, eval_grid=grid)[0]))
    Bmag = np.linalg.norm(
        np.asarray(coilset.compute_magnetic_field(xyz, basis="xyz")), axis=1
    )
    return {
        "normalized_BdotN": float(np.mean(Bn) / np.mean(Bmag)),
        "normalized_BdotN_areawt": float((Bn * dS).sum() / (Bmag * dS).sum()),
        # not on their front, but says whether error is spread or spiked --
        # the same question `track_residual` answers during the solve
        "max_BdotN_over_B": float(np.max(Bn / Bmag)),
        "mean_AbsB": float(np.mean(Bmag)),
        "BdotN": float(np.mean(Bn)),
    }


def geometry_metrics(coilset, eq, coil_grid, surf_MN=(48, 48)):
    """Length, curvature, kappa_MS, arclength variance, TRUE min distances.

    Distances are exact minima over the discretized geometry, not the softmin
    the objective uses -- the paper's `shortest_distance()` is exact, so a
    softmin value would flatter us by construction.
    """
    import numpy as np

    from desc.grid import LinearGrid

    pos = np.asarray(coilset._compute_position(grid=coil_grid))  # (ncoils, n, 3)
    base = flat_coils(coilset)

    lengths, kmax, kms, alvar = [], [], [], []
    for c in base:
        d = c.compute(["length", "|curvature|", "ds", "x_s"], grid=coil_grid)
        L = float(np.atleast_1d(d["length"])[0])
        k = np.abs(np.asarray(d["|curvature|"]))
        ds = np.asarray(d["ds"])
        speed = np.linalg.norm(np.asarray(d["x_s"]), axis=-1)
        lengths.append(L)
        kmax.append(float(k.max()))
        # kappa_MS = (1/L) * int kappa^2 dl, with dl = |x_s| ds
        kms.append(float(np.sum(k**2 * speed * ds) / L))
        alvar.append(float(np.var(speed)))

    d_cc = min(
        float(np.linalg.norm(pos[i][:, None, :] - pos[j][None, :, :], axis=-1).min())
        for i, j in itertools.combinations(range(pos.shape[0]), 2)
    )
    M, N = surf_MN
    sg = LinearGrid(M=M, N=N, NFP=eq.NFP, sym=False)
    xs = np.asarray(eq.compute("x", grid=sg, basis="xyz")["x"])
    d_cs = float(
        min(np.linalg.norm(p[:, None, :] - xs[None, :, :], axis=-1).min() for p in pos)
    )

    currents = [float(c.current) for c in base]
    return {
        "n_base_coils": len(base),
        "n_total_coils": int(pos.shape[0]),
        "max_length": max(lengths),
        "mean_length": float(np.mean(lengths)),
        "max_max_kappa": max(kmax),
        "max_kappa_MS": max(kms),
        "max_arclength_variance": max(alvar),
        "coil_coil_distance": d_cc,
        "coil_surface_distance": d_cs,
        "currents": currents,
        "total_current_base": float(sum(currents)),
    }


def linking_metrics(coilset, coil_grid):
    """Pairwise Gauss linking integral, computed independently of the objective.

    Deliberately NOT `CoilSetLinkingNumber` -- if that objective is misspecified
    (which is under investigation), a diagnostic built on it would inherit the
    same bug. A well-posed modular set has every pair unlinked.
    """
    import numpy as np

    pos = np.asarray(coilset._compute_position(grid=coil_grid))
    worst, n_linked, vals = 0.0, 0, []
    for i, j in itertools.combinations(range(pos.shape[0]), 2):
        a, b = pos[i], pos[j]
        da = np.roll(a, -1, axis=0) - a
        db = np.roll(b, -1, axis=0) - b
        r = a[:, None, :] - b[None, :, :]
        rn = np.maximum(np.linalg.norm(r, axis=-1) ** 3, 1e-30)
        cross = np.cross(da[:, None, :], db[None, :, :])
        lk = float(np.sum(np.einsum("ijk,ijk->ij", r, cross) / rn)) / (4 * np.pi)
        vals.append(lk)
        worst = max(worst, abs(lk))
        if abs(lk) > 0.5:
            n_linked += 1
    return {
        "max_abs_linking_number": worst,
        "n_linked_pairs": n_linked,
        "linking_max_frac_dev": float(
            max(abs(v - round(v)) for v in vals) if vals else 0.0
        ),
    }


def passes_paper_filter(m):
    """Would this coilset have survived the paper's own filter.

    Their engineering cuts (with the 5% margin they used) plus their hygiene
    cuts. `kappa_MS` is included since we can measure it even though we cannot
    constrain it. `unlinked` is ours: their linking check was commented out.
    """
    f, c = PAPER_FILTER, PAPER_CONSTRAINTS
    up, low = f["margin_up"], f["margin_low"]
    checks = {
        "length": m["max_length_paperR0"] < c["length_max"] * up,
        "kappa": m["max_max_kappa_paperR0"] < c["kappa_max"] * up,
        "kappa_MS": m["max_kappa_MS_paperR0"] < c["kappa_ms_max"] * up,
        "d_cc_min": m["coil_coil_distance_paperR0"] > c["d_cc_min"] * low,
        "d_cs_min": m["coil_surface_distance_paperR0"] > c["d_cs_min"] * low,
        "d_cc_max": m["coil_coil_distance_paperR0"] < f["d_cc_max"],
        "d_cs_max": m["coil_surface_distance_paperR0"] < f["d_cs_max"],
        "length_min": m["max_length_paperR0"] > f["length_min"],
        "arclength_var": m["max_arclength_variance"] < f["arclength_var_max"],
        "BdotN": m["normalized_BdotN"] < f["normalized_BdotN_max"],
        "unlinked": m["n_linked_pairs"] == 0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failed": [k for k, v in checks.items() if not v],
    }


# ---------------------------------------------------------------------------
# problem construction
# ---------------------------------------------------------------------------
def build_case(cfg):
    """Assemble equilibrium, coilset, objective, and constraint list."""
    import numpy as np

    import desc.io
    from desc import objectives as O
    from desc.coils import initialize_modular_coils
    from desc.examples import get
    from desc.grid import LinearGrid

    eq = get(cfg["eq"])

    if cfg.get("init_coils"):
        obj = desc.io.load(cfg["init_coils"])
        coilset = (
            obj
            if "Coil" in type(obj).__name__
            else next(o for o in obj if "Coil" in type(o).__name__)
        )
    else:
        c0 = initialize_modular_coils(eq, num_coils=cfg["nc"], r_over_a=cfg["r_over_a"])
        rep = cfg["rep"]
        if rep == "FourierXYZ":
            coilset = c0.to_FourierXYZ(N=cfg["order"])
        elif rep == "FourierPlanar":
            coilset = c0.to_FourierPlanar(N=cfg["order"])
        elif rep == "Spline":
            coilset = c0.to_SplineXYZ(knots=cfg.get("knots"))
        else:
            raise ValueError(f"unknown rep {rep!r}")

    coil_grid = LinearGrid(N=cfg["coil_grid_N"])
    curv_grid = LinearGrid(N=cfg["curv_grid_N"])
    plasma_grid = LinearGrid(
        M=cfg["plasma_M"], N=cfg["plasma_N"], NFP=eq.NFP, sym=eq.sym
    )
    # distance-to-plasma must NOT use a symmetric grid: the reflection is a
    # symmetry of the coilset, but the discretized surface points are the
    # "other object", so half of them would be missing.
    plasma_grid_nosym = LinearGrid(M=cfg["plasma_M"], N=cfg["plasma_N"], NFP=eq.NFP)

    R0 = float(eq.compute("R0", grid=LinearGrid(M=8, N=8, NFP=eq.NFP, sym=False))["R0"])
    lim = rescale_to_eq(R0)

    # Bounds in PHYSICAL units (metres, 1/m) with normalize_target=True so DESC
    # divides by its own normalization. `bound_scale` traces a front around the
    # paper's values (1.0 = paper exactly).
    bs = cfg.get("bound_scale", {})
    len_cap = lim["length_max"] * bs.get("length", 1.0)
    kap_cap = lim["kappa_max"] * bs.get("kappa", 1.0)
    dcc_min = lim["d_cc_min"] * bs.get("d_cc", 1.0)
    dcs_min = lim["d_cs_min"] * bs.get("d_cs", 1.0)

    qflux = O.QuadraticFlux(
        eq,
        field=coilset,
        eval_grid=plasma_grid,
        field_grid=coil_grid,
        vacuum=True,
        bs_chunk_size=cfg.get("bs_chunk_size", 10),
    )
    o_len = O.CoilLength(
        coilset, bounds=(0, len_cap), normalize_target=True, grid=coil_grid
    )
    o_kap = O.CoilCurvature(
        coilset,
        bounds=(0, kap_cap),
        normalize_target=True,
        grid=curv_grid,
        # under auglag `weight` scales residual and bound equally, so it only
        # sets penalty emphasis; this cancels CoilCurvature's 2*pi/num_nodes
        # quadrature weight so the bound binds pointwise as written.
        weight=curv_grid.num_nodes / (2 * np.pi),
    )
    o_cc = O.CoilSetMinDistance(
        coilset,
        bounds=(dcc_min, np.inf),
        normalize_target=True,
        grid=coil_grid,
        use_softmin=cfg.get("use_softmin", True),
        softmin_alpha=cfg.get("softmin_alpha", 100.0),
        dist_chunk_size=cfg.get("dist_chunk_size", 2),
    )
    o_cs = O.PlasmaCoilSetMinDistance(
        eq,
        coilset,
        bounds=(dcs_min, np.inf),
        normalize_target=True,
        plasma_grid=plasma_grid_nosym,
        coil_grid=coil_grid,
        eq_fixed=True,
        use_softmin=cfg.get("use_softmin", True),
        softmin_alpha=cfg.get("softmin_alpha", 100.0),
        dist_chunk_size=cfg.get("dist_chunk_size", 2),
    )
    o_link = O.CoilSetLinkingNumber(coilset, target=0, grid=coil_grid)
    o_alv = O.CoilArclengthVariance(coilset, target=0, grid=coil_grid)

    # --- current handling: the degeneracy fix (see module docstring) ---
    mode = cfg.get("current_mode", "sum")
    if mode == "sum":
        cur = O.FixSumCoilCurrent(coilset)
    elif mode == "one":  # reproduces basic_AugLag.ipynb
        cur = O.FixCoilCurrent(coilset, indices=[i == 0 for i in range(len(coilset))])
    elif mode == "all":
        cur = O.FixCoilCurrent(coilset)
    else:
        raise ValueError(f"unknown current_mode {mode!r}")

    named = {
        "quadratic_flux": qflux,
        "length": o_len,
        "curvature": o_kap,
        "coil_coil": o_cc,
        "plasma_coil": o_cs,
        "linking": o_link,
        "arclength_var": o_alv,
    }
    use = cfg.get(
        "terms", ["length", "curvature", "coil_coil", "plasma_coil", "linking"]
    )

    if cfg["method"] == "lsq-auglag":
        # the point of the exercise: engineering limits enter as BOUNDS, so no
        # weight decides how hard each one is enforced.
        objective = O.ObjectiveFunction((qflux,))
        constraints = tuple([cur] + [named[k] for k in use])
    elif cfg["method"] == "lsq-exact":
        # penalty baseline: same terms, folded in with weights.
        w = cfg.get("weights", {})
        for k in use:
            named[k].weight = w.get(k, 1.0)
        qflux.weight = w.get("quadratic_flux", 1.0)
        objective = O.ObjectiveFunction(tuple([qflux] + [named[k] for k in use]))
        constraints = (cur,)
    else:
        raise ValueError(f"unknown method {cfg['method']!r}")

    return dict(
        eq=eq,
        coilset=coilset,
        objective=objective,
        constraints=constraints,
        named=named,
        coil_grid=coil_grid,
        R0=R0,
        limits=lim,
        applied_bounds={
            "length_max_m": len_cap,
            "kappa_max_invm": kap_cap,
            "d_cc_min_m": dcc_min,
            "d_cs_min_m": dcs_min,
        },
    )


def run_case(cfg, outdir="runs"):
    """Solve one case; write `<tag>.h5` and `<tag>.json`. Returns the record."""
    import numpy as np

    from desc.optimize import Optimizer

    os.makedirs(outdir, exist_ok=True)
    tag = cfg.get("tag") or uuid.uuid4().hex[:10]
    built = build_case(cfg)
    eq, coilset = built["eq"], built["coilset"]

    m_before = {}
    if cfg.get("record_initial", True):
        m_before = paper_metrics(coilset, eq)
        m_before.update(geometry_metrics(coilset, eq, built["coil_grid"]))

    opt = Optimizer(cfg["method"])
    t0 = time.time()
    (new_coils,), res = opt.optimize(
        coilset,
        objective=built["objective"],
        constraints=built["constraints"],
        maxiter=cfg.get("maxiter", 300),
        verbose=cfg.get("verbose", 3),
        ftol=cfg.get("ftol", 1e-6),
        xtol=cfg.get("xtol", 1e-6),
        gtol=cfg.get("gtol", 1e-8),
        ctol=cfg.get("ctol", 1e-4),
        copy=True,
        options=dict(cfg.get("options", {})),
    )
    wall = time.time() - t0

    m = paper_metrics(new_coils, eq)
    m.update(geometry_metrics(new_coils, eq, built["coil_grid"]))
    m.update(linking_metrics(new_coils, built["coil_grid"]))

    # express geometry at the paper's R0 = 1 m so the filter compares directly
    f = built["R0"] / PAPER_R0
    m["max_length_paperR0"] = m["max_length"] / f
    m["max_max_kappa_paperR0"] = m["max_max_kappa"] * f
    m["max_kappa_MS_paperR0"] = m["max_kappa_MS"] * f**2  # 1/m^2, see rescale_to_eq
    m["coil_coil_distance_paperR0"] = m["coil_coil_distance"] / f
    m["coil_surface_distance_paperR0"] = m["coil_surface_distance"] / f
    filt = passes_paper_filter(m)

    def _get(k, default=None):
        try:
            v = res[k]
        except (KeyError, IndexError, TypeError):
            return default
        return default if v is None else v

    allf_max = _get("allf_max")
    rec = {
        "tag": tag,
        "config": {k: v for k, v in cfg.items() if k != "options"},
        "options": cfg.get("options", {}),
        "applied_bounds": built["applied_bounds"],
        "eq_R0": built["R0"],
        "wall_s": wall,
        "metrics": m,
        "metrics_initial": m_before,
        "paper_filter": filt,
        "solver": {
            # the convergence question: did it claim success, and how many
            # multiplier updates did the outer loop actually manage?
            "message": str(_get("message")),
            "success": bool(_get("success", False)),
            "cost": float(_get("cost", np.nan)),
            "constr_violation": float(_get("constr_violation", np.nan)),
            "nit": int(_get("nit", -1)),
            "n_outer": int(_get("n_outer", -1)),
            "nfev": int(_get("nfev", -1)),
            "njev": int(_get("njev", -1)),
            "allf_max_first": float(allf_max[0]) if allf_max is not None else None,
            "allf_max_last": float(allf_max[-1]) if allf_max is not None else None,
        },
        "current_drift": (
            m["total_current_base"] / m_before["total_current_base"]
            if m_before.get("total_current_base")
            else None
        ),
    }

    new_coils.save(os.path.join(outdir, f"{tag}.h5"))
    with open(os.path.join(outdir, f"{tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def summarize(rec):
    """One-line console summary of a solved case."""
    s, m = rec["solver"], rec["metrics"]
    drift = rec["current_drift"]
    return "\n".join(
        [
            f"=== {rec['tag']} ===",
            f"  normalized_BdotN     {m['normalized_BdotN']:.4e}"
            f"   (area-wt {m['normalized_BdotN_areawt']:.4e})",
            f"  max_length           {m['max_length_paperR0']:.3f} m   @R0=1",
            f"  max kappa            {m['max_max_kappa_paperR0']:.3f} 1/m",
            f"  kappa_MS             {m['max_kappa_MS_paperR0']:.3f} 1/m",
            f"  d_cc / d_cs          {m['coil_coil_distance_paperR0']:.4f}"
            f" / {m['coil_surface_distance_paperR0']:.4f} m",
            f"  linked pairs         {m['n_linked_pairs']}"
            f"  (worst |Lk| {m['max_abs_linking_number']:.3e},"
            f" frac dev {m['linking_max_frac_dev']:.3e})",
            f"  current drift        {'n/a' if drift is None else f'{drift:.4f}'}",
            f"  passes paper filter  {rec['paper_filter']['pass']}"
            f"  failed={rec['paper_filter']['failed']}",
            f"  solver               success={s['success']} nit={s['nit']}"
            f" n_outer={s['n_outer']} viol={s['constr_violation']:.2e}",
            f"  message              {s['message']}",
        ]
    )


DEFAULTS = dict(
    eq="precise_QA",
    nc=5,
    rep="FourierXYZ",
    order=16,
    r_over_a=3.0,
    coil_grid_N=50,
    curv_grid_N=12,
    plasma_M=25,
    plasma_N=25,
    method="lsq-auglag",
    current_mode="sum",
    maxiter=300,
    ctol=1e-4,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-8,
    softmin_alpha=100.0,
    bs_chunk_size=10,
    dist_chunk_size=2,
    verbose=3,
)


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="default cpu -- the GPU is shared, ask before taking it",
    )
    p.add_argument("--tag", default=None)
    p.add_argument("--outdir", default="runs")
    p.add_argument("--cfg", default=None, help="JSON file path or inline JSON dict")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="K=V",
        help="override a config key; value parsed as JSON",
    )
    args = p.parse_args()

    configure_device(args.device)

    cfg = dict(DEFAULTS)
    if args.cfg:
        raw = args.cfg
        if os.path.exists(raw):
            raw = open(raw).read()
        cfg.update(json.loads(raw))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    if args.tag:
        cfg["tag"] = args.tag
    if args.device == "gpu":
        from desc import set_device

        set_device("gpu")

    print(summarize(run_case(cfg, outdir=args.outdir)))


if __name__ == "__main__":
    main()
