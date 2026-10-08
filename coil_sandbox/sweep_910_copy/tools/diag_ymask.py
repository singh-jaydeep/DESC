"""Does the multiplier gate shut off the rows that carry constraint force?.

Hypothesis (see HANDOFF_convergence.md sec 4.4). `lsq_auglag` gates the multiplier
update on the slack-reformulated residual::

    y = jnp.where(jnp.abs(c) < ctolk, y - mu * c, y)

but `c` is `c(x) - s`, and minimising the AL over an interior slack gives
`s = c(x) - y/mu`, hence `c == y/mu`. So the gate reads `|y|/mu < ctolk`: a row
carrying a meaningfully non-zero multiplier is admitted only once
`mu > |y|/ctolk`. On those rows the method is a quadratic penalty, whose
feasibility floor is `~|y*|/mu` -- so only `mu` growth lowers it, which is the
runaway. The `mu` branch of the same update was already fixed to key on
`constr_violation_rows`; the `y` branch was not.

Predicted signature if true: `n_blocked > 0` on the active rows, `y_absmax`
flat for the whole run, `outer_log` dominated by `stall`.

Run (GPU, ~300 iterations each)::

    python c0/diag_ymask.py --gate residual  --out c0/runs/ymask
    python c0/diag_ymask.py --gate violation --out c0/runs/ymask

then compare with `python c0/diag_ymask.py --report c0/runs/ymask`.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------------
# DEVICE SELECTION MUST HAPPEN BEFORE ANYTHING IMPORTS `desc` (hence before the
# `from desc.utils import errorif` below), and this is not a style preference.
#
# `set_device` works by setting environment variables -- JAX_PLATFORMS,
# CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER -- which JAX reads ONCE, when it first
# initialises its backend. Importing any `desc` submodule imports jax immediately, and
# plain `import desc` defaults to CPU, which sets CUDA_VISIBLE_DEVICES="". A later
# `set_device("gpu")` then finds no visible GPU and falls back.
#
# MEASURED on this repo, on a box with an RTX 5070:
#     from desc.utils import errorif      # jax is now in sys.modules
#     set_device("gpu")                   # -> "CUDA_VISIBLE_DEVICES= did not match any
#                                         #    physical GPU, falling back to CPU"
#     jax.devices()                       # -> ['cpu']
# i.e. the whole run executes on CPU at roughly half speed while its command line says
# `--device gpu`. Reversing the order returns the GPU. There is no warning loud enough
# to survive a 41-hour sweep, so the order is enforced here instead.
#
# Guarded on __main__ so that importing this module (adjudicate.py, dump_initial_starts.py
# and promote_checkpoint.py all do) never seizes a device as a side effect.
if __name__ == "__main__":
    _dev = "gpu"  # matches the --device default below
    for _i, _a in enumerate(sys.argv):
        if _a == "--device" and _i + 1 < len(sys.argv):
            _dev = sys.argv[_i + 1]
        elif _a.startswith("--device="):
            _dev = _a.split("=", 1)[1]
    if _dev == "gpu":
        # Do not let JAX reserve most of the card up front; on a laptop GPU the display
        # is sharing it. Respect an explicit setting if the caller made one.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from desc import set_device as _set_device

    _set_device(_dev)
    _DEVICE_SET = _dev
else:
    _DEVICE_SET = None


from desc.utils import errorif

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CURV_BOUND = 4.657  # 1/m, as used in the overnight runs (LEGACY -- see PAPER below)
POLE_BOUND = 0.4

# --- paper conventions (c0/CONVENTIONS.md) --------------------------------------
# Bounds come from bench.rescale_to_eq(eq.compute("R0")) at solve time, in PHYSICAL
# units with normalize_target=True, so nothing re-anchors on the start. The start is
# BUILT, not loaded: the on-disk pw_start_B5M3.h5 is an r/a=2.5 artifact sitting 60%
# inside the plasma-coil bound and must never seed a sweep.
PAPER = dict(
    nc=4,  # unique coils; nc=5 leaves only a 0.035-wide feasible r/a window
    r_over_a=3.3,  # equal-margin point of the nc=4 window [3.151, 3.401]
    B=5,
    M=3,
    gl_m=24,  # composite-GL nodes per arc; bn error 2.1e-2 at gl_m=8 vs 3.8e-6 at 24
    dist_grid_N=150,  # hard min moves 0.14% from N=50 to N=800; not the sensitive knob
    curv_grid_N=150,  # UNIFORM. kappa is pointwise and grid-converged (2.1359 from
    #                   N=50 up); the GL weights vary 10x across rows and would put an
    #                   arbitrary 10x spread on a pointwise constraint block.
    # LENGTH uses the composite-GL source grid, but only because
    # `_CoilObjective.compute` now passes `override_grid=False` (see the comment
    # there). Without that, `Curve.compute` silently recomputes `length` -- the only
    # 0d key any coil objective requests -- on its own `LinearGrid(N=2*N+5)`, and the
    # constraint stops measuring the length. Do not revert one without the other.
    # LINKING also uses the composite-GL source grid -- see the comment on the objective.
    # A uniform grid converges O(1/N) there and makes target=0 unsatisfiable.
    uniform_src_N=400,  # smooth reps only
    plasma_M=25,  # QuadraticFlux eval grid; its cost is converged to 8 s.f. by M=18
    plasma_N=25,
    # SEPARATE, finer surface grid for PlasmaCoilSetMinDistance. A min-distance is a
    # pointwise extremum, not an integral, so it converges far more slowly than the
    # flux quadrature: measured d_pc = 0.195697 at M=25 against 0.191575 at M=128, i.e.
    # the M=25 grid over-reads the clearance by 2.2% (+0.0042 m). That is the same
    # class of optimistic bias as softmin, and the same size, so it gets fixed the same
    # way. M=64 lands within 0.1% of converged and costs 0.14 s to evaluate.
    dist_plasma_M=64,
    dist_plasma_N=64,
)
START = "c0/arc_start_B5M3.h5"
PW_START = "c0/pw_start_B5M3.h5"
# B=2 exact circles: shape=0 is the exact circle at B=2 (0.00% basis error,
# vs 2.85% at B=5 M=3), so curvature is exactly 1/R and nothing is
# infeasible at the start. tilts=[0,pi] -- [0,0] gives a DOUBLED SEMICIRCLE
# with identical (constant) curvature, so check the y-extent not kappa.
CIRC_START = "c0/circ_start_B2M3.h5"


# --- coil_bundle conventions (coil_bundle/code/driver_stage2.py + run_exp2.sh) -----
# Distances take DIMENSIONLESS fractions with normalize_target=False; the bound the
# solver enforces is fraction * objective.normalization. Measured on precise_QA with
# the pw_start coilset: CoilSetMinDistance normalizes by 0.42947 m (the start coil's
# equivalent circle radius L/2pi) and PlasmaCoilSetMinDistance by 0.20512 m, so 0.1 and
# 0.25 come out as 0.0430 m (0.25a) and 0.0513 m (0.30a). NOTE those normalizations
# re-anchor at build() on whatever coilset is passed, so the physical bound MOVES with
# the start -- deliberate here only because it is what generated the bundle.
#
# SOFTMIN BIAS. The bundle itself used a hard min; softmin is restored here because it
# was observed to work in the FourierXYZ AL runs. Be aware what it costs: softmin is a
# weighted AVERAGE of all pairwise distances, so it always reads HIGHER than the true
# minimum, by roughly 1/alpha. MEASURED on the AL result at alpha=100: softmin 0.0852 m
# against a true hard minimum of 0.0713 m, a 1.39 cm overestimate (1/alpha = 1 cm). The
# solver drives the SOFTMIN to the bound, so the real clearance lands ~1.4 cm short.
# Against the bundle coil-coil bound of 0.0430 m that is 32% of the bound, so verify any
# result with a hard-min check and raise `softmin_alpha` if the gap matters.
BUNDLE = dict(
    ccdist_min=0.1,  # fraction, normalize_target=False -> 0.0430 m
    pc_min=0.25,  # fraction, normalize_target=False -> 0.0513 m
    curv_hi=2.0,  # fraction, normalize_target=False
    length_max=6.0,  # METRES, normalize_target=True (run_exp2.sh --length_bound_abs)
    softmin_alpha=100.0,  # see the note on softmin bias below
    gl_m=24,  # composite-GL nodes per arc for the Biot-Savart source grid
    dist_grid_N=150,  # LinearGrid(N) for both distance objectives
    uniform_src_N=400,  # LinearGrid(N) source grid for smooth (non-arc) reps
)


def _source_grid(coilset, gl_m, uniform_n):
    """Biot-Savart source grid: composite GL bracketing the hinges for arc reps.

    The arc classes are C0, and the quad path reads ds = grid.spacing[:,2], so GL
    weights make DESC's own sum a composite rule across the corners. Smooth reps
    get a dense uniform grid. Never None: DESC would then default to
    LinearGrid(N=2*coil.N+5) with coil.N derived from the shape parameters, which
    would move resolution with the swept axis.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bundle_probe import _arc_breaks, _gl_grid, _iter_unique

    from desc.grid import LinearGrid

    c0 = next(_iter_unique(coilset))
    if hasattr(c0, "B") and gl_m and gl_m > 0:
        return _gl_grid(_arc_breaks(c0), gl_m), "gl"
    return LinearGrid(N=uniform_n), "uniform"


def _lift_start(target, source):
    """Put a solved coilset's geometry onto a same-B, higher-M start. REFINEMENT.

    Every PiecewisePlanarArc parameter except `shape` is M-independent -- `hinges`,
    `arc_ref`, `tilts`, `rotmat`, `shift`, `current` -- and `shape` is (B, M)
    row-major, so lifting M -> M+k is a zero-column append and the curve is
    geometrically IDENTICAL to the source.

    That identity is the whole point. It makes the higher-M feasible set a strict
    SUPERSET containing the source optimum, so a refinement run starts exactly at the
    lower-M answer and can only improve on it. A cold-started higher-M run cannot say
    that: it lands in its own basin, and the basin scatter (measured ~1% between
    al_B5_M4 and paper_al_v13_chord, in BOTH directions on the two metrics) is the
    same size as the effect the extra mode is being asked about.

    The caller must check that `bn` is unchanged after lifting -- that is the control
    on the padding convention, and it is cheap.
    """
    import numpy as np
    from bundle_probe import _iter_unique

    tg, sc = list(_iter_unique(target)), list(_iter_unique(source))
    errorif(
        len(tg) != len(sc),
        ValueError,
        f"lift: {len(sc)} source coils vs {len(tg)} target coils",
    )
    for t, src in zip(tg, sc):
        pd = {k: np.asarray(v) for k, v in src.params_dict.items()}
        if hasattr(src, "B"):
            # piecewise arc: `shape` is the only M-dependent parameter, (B, M)
            B, Ms, Mt = int(src.B), int(src.M), int(t.M)
            errorif(int(t.B) != B, ValueError, f"lift needs equal B: {B} -> {int(t.B)}")
            errorif(Mt < Ms, ValueError, f"lift only raises M: {Ms} -> {Mt}")
            sh = pd["shape"].reshape(B, Ms)
            pd["shape"] = np.concatenate([sh, np.zeros((B, Mt - Ms))], axis=1).ravel()
        else:
            # smooth reps (FourierPlanar, FourierXYZ). At IDENTICAL resolution a lift
            # is a plain warm start -- the path a length CONTINUATION takes: same rep,
            # same N, the previous length's optimum as the next length's start.
            #
            # A SPECTRAL continuation (raise N at fixed length) is the smooth-rep
            # analogue of the arc branch above, and it is exact for the same reason:
            # a Fourier series of order Ns is contained in one of order Nt > Ns, so
            # zero-filling the new modes leaves the curve geometrically IDENTICAL and
            # makes the higher-N feasible set a strict superset holding the source
            # optimum. `r_n` is indexed by MODE NUMBER over a centred range
            # ([-4..4] -> [-7..7]), so the new coefficients go in the MIDDLE, not on
            # the end -- `copy_coeffs` places them, which is the same convention
            # `FourierPlanarCurve.change_resolution` uses (desc/geometry/curve.py:743).
            # As above, the caller must check `bn` is unchanged after lifting.
            tp = t.params_dict
            bad = [k for k in pd if k not in tp or np.shape(tp[k]) != np.shape(pd[k])]
            if bad == ["r_n"] and hasattr(src, "r_basis") and hasattr(t, "r_basis"):
                from desc.utils import copy_coeffs

                Ns, Nt = int(src.r_basis.N), int(t.r_basis.N)
                errorif(Nt < Ns, ValueError, f"lift only raises N: {Ns} -> {Nt}")
                pd["r_n"] = np.asarray(
                    copy_coeffs(pd["r_n"], src.r_basis.modes, t.r_basis.modes)
                )
                bad = []
            errorif(
                bool(bad),
                ValueError,
                f"lift: source and target differ in {bad}; a smooth-rep lift needs "
                "the same representation, at the same resolution or a higher N",
            )
        t.params_dict = pd
    return target


def _paper_start(
    eq,
    rep,
    nc,
    B,
    M,
    r_over_a,
    fp_N,
    fit_method="parameter",
    xyz_N=6,
    tilt_deg=0.0,
    tilt_axis="rock",
    tilt_spec=None,
    arc_bulge=None,
    mode_kick=0.0,
    mode_ns=(2, 3),
):
    """The baseline start of `c0/CONVENTIONS.md` sec 4, BUILT rather than loaded.

    `initialize_modular_coils` -> `PiecewisePlanarArcCoil.from_values`. At
    ``nc=4, r_over_a=3.3`` every B >= 3 cold-starts feasible against all four paper
    bounds with ~14% margin on both clearances.

    `fit_method` selects the abscissa `from_values` fits against (sec 4.1). Under the
    default ``"parameter"`` the fit is exact against the WRONG target and returns a
    sine arch: at B=2 the start lies 94.5 mm from the circle it was fit to and
    violates d_pc (0.1135 vs 0.1711), so B=2 is refused at r/a=3.3 on that path.
    Under ``"chord"`` B=2 cold-starts FEASIBLE at r/a=3.3 (d_pc 0.19540, margin
    1.142 -- 99.7% of the exact circle's own clearance) and needs no separate r/a.

    `planar` is the FourierPlanar control and is built from the SAME modular start so
    the two representations differ only in parameterization.
    """
    from bundle_probe import (
        _arc_single_mode,
        _cold_init,
        _mode_kick,
        _tilt_planar,
        _tilt_planar_spec,
    )

    from desc.coils import initialize_modular_coils
    from desc.utils import errorif

    if rep in ("planar", "xyz"):
        base = initialize_modular_coils(eq, num_coils=nc, r_over_a=r_over_a)
        if rep == "planar":
            # hand basin probe: a rigid per-coil rotation, zero shape change
            _fp = _tilt_planar(base.to_FourierPlanar(N=fp_N), tilt_deg, tilt_axis)
            # per-coil aim; inert unless --tilt-spec is given (8p)
            _fp = _tilt_planar_spec(_fp, tilt_spec)
            return _mode_kick(_fp, mode_kick, mode_ns)
        # FourierXYZ is the unconstrained-shape control: it is the only rep here that
        # can leave the plane at all, so it bounds what the planar family gives up.
        # DOF/coil = 3*(2N+1): N=6 -> 39, against 35 for both arc B=5 M=3 (B*(4+M))
        # and FourierPlanar N=14 (3 + 3 + (2N+1)). One reparameterization gauge
        # direction is unobservable, so effective DOF is ~38 -- the closest honest
        # match on a 3*(2N+1) ladder, 33 at N=5 being the nearest below.
        return base.to_FourierXYZ(N=xyz_N)
    errorif(
        rep != "piecewise",
        ValueError,
        f"convention='paper' supports rep in ('piecewise','planar','xyz'), got "
        f"{rep!r}. "
        "The polar 'arc' rep cold-starts at kappa_max=17.39 against a bound of "
        "11.64 -- structurally infeasible, see CONVENTIONS.md sec 4.3.",
    )
    errorif(
        B == 2 and abs(r_over_a - 3.3) < 1e-9 and fit_method == "parameter",
        ValueError,
        "B=2 is infeasible at r_over_a=3.3 under fit_method='parameter' "
        "(d_pc 0.1135 < 0.1711) -- but that is a FIT artifact, not the geometry: "
        "the default fit returns a sine arch 94.5 mm from the circle. Pass "
        "--fit-method chord, which cold-starts B=2 feasible at r/a=3.3 "
        "(d_pc 0.19540, margin 1.142). CONVENTIONS.md sec 4.1.",
    )
    # `--arc-bulge` collapses the fitted transverse profile to its fundamental. Inert
    # when omitted, so every arc start built before it existed is bit-identical.
    return _arc_single_mode(
        _cold_init(eq, nc, B, M, r_over_a=r_over_a, fit_method=fit_method), arc_bulge
    )


def build(
    curv_grid_N=201,
    coil_grid_N=50,
    plasma_M=25,
    plasma_N=25,
    rep="arc",
    fp_N=14,
    mode="auglag",
    mu0=10.0,
    drop=(),
    curv_wmult=None,
    convention="paper",
    softmin_alpha=BUNDLE["softmin_alpha"],
    nc=PAPER["nc"],
    B=PAPER["B"],
    M=PAPER["M"],
    r_over_a=PAPER["r_over_a"],
    length_max=None,
    curv_max=None,
    dcc_min=None,
    feas_margin=0.0,
    dist_method="point",
    fit_method="parameter",
    xyz_N=6,
    dcc_pair_mode="per_coil",
    dcc_neighbors=None,
    tilt_deg=0.0,
    tilt_axis="rock",
    tilt_spec=None,
    arc_bulge=None,
    mode_kick=0.0,
    mode_ns=(2, 3),
    start_from=None,
    curv_ms_max=None,
    dcc_signed=False,
):
    """Objective + constraints, built ON the start coilset.

    `convention` selects how the engineering terms are specified:

    - ``"paper"`` (DEFAULT): `c0/CONVENTIONS.md`. Hurwitz et al. Table 1 rescaled to
      this equilibrium, in PHYSICAL units with ``normalize_target=True`` on every
      bounded term, so no bound re-anchors on the start. Hard min (no softmin bias).
      The start is BUILT by `_paper_start` at ``nc=4, r_over_a=3.3``, not loaded.
      ``FixSumCoilCurrent`` is returned as a linear constraint IN AUGLAG MODE TOO --
      QuadraticFlux(vacuum=True) is homogeneous of degree 2 in the currents, so
      ``I -> 0`` is a free minimizer and the currents must be pinned by a constraint
      that actually holds during the solve. ``LinkingCurrentConsistency`` is then
      redundant (measured: residual exactly 0.0 at the cold start, because
      ``initialize_modular_coils`` already sets the linking current) and is dropped.
      `length_max` / `curv_max` override one bound each, for the two sweeps that
      vary a bound; everything else stays frozen.

    - ``"bundle"``: the conventions that produced `coil_bundle/` on
      2026-08-05, which optimized well. Distances as dimensionless fractions with
      ``normalize_target=False`` and HARD min; curvature on the coarse coil grid with
      ``normalize_target=False`` and no quadrature-cancelling weight; length in metres;
      composite-GL Biot-Savart source grid bracketing the hinges, shared by
      QuadraticFlux, CoilLength and CoilSetLinkingNumber; currents pinned on coil 0
      only. Two deliberate departures from the bundle: the distance objectives use
      SOFTMIN (observed to work in the FourierXYZ AL runs; see the bias note above),
      and LinkingCurrentConsistency is present because the net poloidal current is a
      physical requirement the bundle's single pinned current does not enforce.
      ArcPoleDistance is still absent.

    - ``"c0"``: the legacy stack these runs used before 2026-08-28. Distances in metres
      from ``bench.rescale_to_eq`` with ``normalize_target=True`` and softmin
      (alpha=100); curvature on a 403-node grid with weight multiplied by
      ``num_nodes/(2*pi)``; ArcPoleDistance and LinkingCurrentConsistency present;
      currents pinned only in sum. Kept so earlier runs remain reproducible.

    MEASURED difference in what the two actually enforce, precise_QA + pw_start:

        objective                  c0 convention      bundle convention
        CoilSetMinDistance         0.0855 m (0.50a)   0.0430 m (0.25a)
        PlasmaCoilSetMinDistance   0.1711 m (1.00a)   0.0513 m (0.30a)

    The c0 plasma-coil bound is ~one full minor radius, which no
    ``initialize_modular_coils`` start below r/a ~ 3.5 satisfies -- both arc starts
    here sit at ~0.069 m, i.e. 60% inside it.

    `drop` removes named engineering terms entirely, `mode` selects how they enter
    (``"auglag"`` constraints, ``"unc"`` dropped, ``"pen"`` folded in at
    ``weight=sqrt(mu0)`` -- the first auglag subproblem at ``y=0``), and
    `curv_wmult` overrides the CoilCurvature weight multiplier (c0 convention only;
    ``None`` keeps the historical ``num_nodes/(2*pi)`` = 64.1).
    """
    import numpy as np
    from bench.bench import rescale_to_eq

    import desc.objectives as O
    from desc.examples import get
    from desc.grid import LinearGrid
    from desc.io import load
    from desc.utils import errorif

    errorif(
        convention not in ("paper", "bundle", "c0"),
        ValueError,
        f"unknown convention {convention!r}",
    )

    eq = get("precise_QA")
    if convention == "paper":
        coilset = _paper_start(
            eq,
            rep,
            nc,
            B,
            M,
            r_over_a,
            fp_N,
            fit_method,
            xyz_N,
            tilt_deg=tilt_deg,
            tilt_axis=tilt_axis,
            tilt_spec=tilt_spec,
            arc_bulge=arc_bulge,
            mode_kick=mode_kick,
            mode_ns=mode_ns,
        )
        if start_from is not None:
            # REFINEMENT: keep the structure just built (it carries the requested M)
            # but take the geometry from a solved lower-M coilset, zero-padding the
            # shape modes. See `_lift_start`. Objectives are built BELOW, so they bind
            # to the lifted coilset, not the cold one.
            from desc.io import load as _load

            _src = _load(start_from)
            print(f"REFINEMENT start: {start_from}", flush=True)
            coilset = _lift_start(coilset, _src)
    else:
        # NB build objectives on the coilset we actually start from: CoilCurvature,
        # CoilSetMinDistance and CoilLength all re-anchor their normalization at
        # build() (HANDOFF sec 5.1). Irrelevant under "paper", where every bound is
        # absolute -- but the RESIDUAL scale still moves, so constr_violation is
        # still not comparable across starts.
        _src = {"piecewise": PW_START, "circle": CIRC_START}.get(rep, START)
        coilset = load(_src)
        if rep == "planar":
            coilset = coilset.to_FourierPlanar(N=fp_N)
        elif rep == "xyz":
            coilset = coilset.to_FourierXYZ(N=xyz_N)
        elif rep not in ("arc", "piecewise", "circle"):
            raise ValueError(f"unknown rep {rep!r}")

    coil_grid = LinearGrid(N=coil_grid_N)
    curv_grid = LinearGrid(N=curv_grid_N)
    plasma_grid = LinearGrid(M=plasma_M, N=plasma_N, NFP=eq.NFP, sym=eq.sym)
    plasma_grid_nosym = LinearGrid(M=plasma_M, N=plasma_N, NFP=eq.NFP)

    w = float(np.sqrt(mu0)) if mode == "pen" else 1.0

    if convention == "paper":
        from verify import paper_bounds

        lim = paper_bounds(eq)
        # ENFORCED bounds = the paper bounds backed off by `feas_margin`, so the solver
        # aims strictly inside the spec and the returned design is feasible with room
        # rather than balanced on the boundary. `verify` still adjudicates against the
        # UNBACKED-OFF paper bounds, so a margin can only make a reported PASS more
        # honest, never manufacture one.
        #
        # Two distinct things this fixes, both measured:
        #  1. SOLVER-TOLERANCE overshoot. A binding constraint is driven onto its bound
        #     and stops within `ctol`, so it lands a hair outside. planar exceeded
        #     length by 3.33e-06 (65x FEAS_RTOL) and was called INFEASIBLE for it; v12
        #     by 4.3e-10. A margin of a few 1e-4 covers this and costs nothing.
        #  2. KNIFE-EDGE geometry. A bounds row that is satisfied contributes exactly
        #     zero Jacobian rows, so a near-active row is invisible to the
        #     linearization until already violated. At B=2, coil 0 sat 3.3e-05 INSIDE
        #     d_cc while coils 1,2 sat 5.0e-05 outside, and the active set flipped
        #     between them. A 2e-2 backoff made B=2 paper-FEASIBLE (margin 0.9990 ->
        #     1.0195) for 0.07% flux. See CONVENTIONS 4.1.1 and 9.
        #
        # Cost is NOT uniform across the four bounds -- length is the productive one
        # (longer coils buy flux), so a large length margin buys feasibility with
        # performance. Keep the margin small unless a specific bound is knife-edged.
        _m = float(feas_margin)
        errorif(
            _m < 0 or _m >= 0.5,
            ValueError,
            f"feas_margin must be in [0, 0.5), got {_m}",
        )
        # An EXPLICIT per-bound override is taken as given and never scaled, so a run
        # can carry a margin on the cheap bounds while holding length at spec:
        #     --feas-margin 0.03 --length-max 5.153350
        L_hi = (
            float(length_max)
            if length_max is not None
            else lim["length_max"] * (1 - _m)
        )
        K_hi = float(curv_max) if curv_max is not None else lim["kappa_max"] * (1 - _m)
        D_lo = float(dcc_min) if dcc_min is not None else lim["d_cc_min"] * (1 + _m)
        P_lo = lim["d_pc_min"] * (1 + _m)
        src, src_kind = _source_grid(coilset, PAPER["gl_m"], PAPER["uniform_src_N"])
        dist_grid = LinearGrid(N=PAPER["dist_grid_N"])
        dist_plasma_grid = LinearGrid(
            M=PAPER["dist_plasma_M"], N=PAPER["dist_plasma_N"], NFP=eq.NFP, sym=eq.sym
        )
        print(
            f"src={src_kind} {src.num_nodes} nodes/coil  "
            f"bounds{f' [feas_margin={_m:g}]' if _m else ''}: "
            f"L<={L_hi:.6f}"
            # the d_cc override announces itself; length and curvature did not, so a
            # length-sweep log read `L<=6.441687` with nothing marking it non-standard
            + (
                f" (OVERRIDE, paper {lim['length_max']:.6f})"
                if length_max is not None
                else ""
            )
            + f" kappa<={K_hi:.6f}"
            + (
                f" (OVERRIDE, paper {lim['kappa_max']:.6f})"
                if curv_max is not None
                else ""
            )
            + f" d_cc>={D_lo:.6f}"
            + (
                f" (OVERRIDE, paper {lim['d_cc_min']:.6f})"
                if dcc_min is not None
                else ""
            )
            + f" d_pc>={P_lo:.6f}"
            # ENFORCED only when --curv-ms-max is given; silent otherwise, so a run
            # carrying this extra constraint is visible in its own log header
            + (
                f" kappa_MS<={float(curv_ms_max):.6f}"
                f" (ENFORCED, paper {lim['kappa_ms_max']:.6f})"
                if curv_ms_max is not None
                else ""
            )
            + f" (R0={lim['R0']:.6f})"
        )
        qflux = O.QuadraticFlux(
            eq,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=src,
            vacuum=True,
            bs_chunk_size=10,
        )
        named = {
            "length": O.CoilLength(
                coilset, bounds=(0.0, L_hi), normalize_target=True, grid=src, weight=w
            ),
            # NB uniform grid, NOT src: CoilCurvature is the one _CoilObjective that
            # does not reset quad_weights to 1, so its rows inherit grid.spacing[:,2].
            # Those weights cancel against bounds_scaled (verified: max violation
            # exactly 0 on a feasible set, on both grids), so the FEASIBLE SET is
            # unaffected -- but on a composite-GL grid they range 0.00775..0.08039, a
            # 10x arbitrary spread across rows of a pointwise bound. Uniform keeps the
            # block equally weighted.
            "curvature": O.CoilCurvature(
                coilset,
                bounds=(0.0, K_hi),
                normalize_target=True,
                grid=LinearGrid(N=PAPER["curv_grid_N"]),
                weight=w,
            ),
            # OFF by default, so every run made before 2026-09-04 reproduces. The
            # paper's Table 1 carries a mean-square curvature spec that this campaign
            # has only ever REPORTED: `al_planar_N14` exceeds it by 8% and the x1.5
            # length point by 14%, both while satisfying the pointwise bound. The two
            # measure different things -- pointwise is a minimum bend RADIUS, this is
            # total bending -- and they bind different coils.
            **(
                {
                    "curvature_ms": O.CoilMeanSquaredCurvature(
                        coilset,
                        bounds=(0.0, float(curv_ms_max)),
                        normalize_target=True,
                        grid=LinearGrid(N=PAPER["curv_grid_N"]),
                    )
                }
                if curv_ms_max is not None
                else {}
            ),
            # distance_method: see CONVENTIONS 8i. "point" takes the min over
            # distances between grid POINTS, so the closest approach must land on a
            # node; at v8 it reported 6.4e-03 m of clearance for coils that INTERSECT
            # (true separation 0; segment-segment reads 9.0e-06 on the same grid).
            # Worse, its value moves the WRONG WAY under a step that genuinely
            # separates the coils, because its sampling error shrinks faster than the
            # true distance grows -- so it manufactures a barrier. With the correct
            # composite-GL length grid, "segment" is the ONLY setting that recovers the
            # v8 geometry (nit=6 to feasible, vs nit=1 reporting `ftol` SUCCESS with
            # the coils still intersecting). Use "segment".
            "coil_coil": O.CoilSetMinDistance(
                coilset,
                bounds=(D_lo, np.inf),
                normalize_target=True,
                grid=dist_grid,
                use_softmin=False,
                dist_chunk_size=2,
                distance_method=dist_method,
                pair_mode=dcc_pair_mode,
                num_neighbors=dcc_neighbors,
                signed=dcc_signed,
                weight=w,
            ),
            "plasma_coil": O.PlasmaCoilSetMinDistance(
                eq,
                coilset,
                bounds=(P_lo, np.inf),
                normalize_target=True,
                plasma_grid=dist_plasma_grid,
                coil_grid=dist_grid,
                eq_fixed=True,
                use_softmin=False,
                dist_chunk_size=2,
                weight=w,
            ),
            # MUST be the composite-GL source grid, NOT a uniform LinearGrid. The Gauss
            # linking integral is a smooth periodic integrand only for a C1 curve; the
            # 12-degree tangent jump at each hinge makes it non-smooth, and uniform
            # quadrature then converges O(1/N) instead of spectrally. Measured on the
            # cold start: N x max|link| = 0.339 for EVERY N from 25 to 400, so reaching
            # ctol=1e-6 would need ~340k nodes. A composite-GL grid that brackets the
            # hinges restores spectral convergence -- 6.99e-09 at gl_m=24 (120 nodes),
            # 1.20e-15 at gl_m=48, versus 8.49e-04 on a uniform 801-node grid. A smooth
            # FourierPlanar coilset reaches 1.62e-15 on a uniform 201-node grid, which
            # confirms the C0 faceting is the cause.
            #
            # NB unlike CoilLength, this objective is NOT affected by the override_grid
            # trap (8b.1): it calls `coilset._compute_linking_number` directly rather
            # than going through `Curve.compute`, so a raw Grid is honoured. Moving it
            # to a LinearGrid "for the same reason" made target=0 unsatisfiable and was
            # the sole source of the ~3.3e-03 constr_violation floor in v3/v4/v5.
            #
            # BOUNDS, NOT target=0. The linking number is a topological invariant: on
            # the unlinked component its true gradient is IDENTICALLY ZERO, so the
            # constraint carries no information -- everything it contributes is the
            # gradient of the quadrature error. That error-gradient goes as 1/d^4 in the
            # coil-coil separation, so it explodes exactly when coils approach. Measured
            # at the v8 iterate (d_cc = 6.4 mm):
            #
            #     block                        max |J row|    max |resid|
            #     coil-coil linking number        2.92e+12       9.25e-06
            #     (every other block)            <= 6.72e+00
            #
            # -- twelve orders above everything else while its own residual is ~0. With
            # `target=0` that row is always live, so `J^T J` is dominated by it, the
            # Gauss-Newton model becomes meaningless (measured: the FD ratio along -g is
            # -0.0000, where a correct gradient gives 1.0), and the solver cannot move to
            # repair d_cc. Positive feedback: coils approach -> Jacobian explodes ->
            # model dies -> d_cc cannot be repaired.
            #
            # With BOUNDS, `_shift`'s `jnp.where` SELECTS the zero branch for a row
            # inside the bounds, so a satisfied row contributes zero residual AND zero
            # Jacobian -- the same masking handoff 1.2 describes for the curvature NaN.
            # The row becomes a tripwire rather than a term in the model.
            #
            # 0.5 is the physically correct threshold: the linking number is an integer,
            # so |Lk| < 0.5 IS "unlinked". No softening of the VALUE can substitute -- a
            # weight scales residual and Jacobian equally (taming 2.9e+12 needs w~1e-12,
            # leaving a 1e-18 residual with no restoring force), and a saturating
            # transform has derivative ~1 at |Lk| = 9e-06, so both pass the exploding
            # gradient straight through. The smooth steering signal that keeps coils
            # apart is `coil_coil` (|J row| = 5.59 at that same iterate), not this.
            "linking_number": O.CoilSetLinkingNumber(
                coilset, bounds=(0.0, 0.5), normalize_target=True, grid=src, weight=w
            ),
        }
        # LINEAR, and returned in auglag mode too -- see the docstring.
        lin = (O.FixSumCoilCurrent(coilset),)
    elif convention == "bundle":
        src, src_kind = _source_grid(coilset, BUNDLE["gl_m"], BUNDLE["uniform_src_N"])
        dist_grid = LinearGrid(N=BUNDLE["dist_grid_N"])
        print(
            f"src={src_kind} {src.num_nodes} nodes/coil (shared by QuadraticFlux, "
            f"CoilLength, CoilSetLinkingNumber)"
        )
        qflux = O.QuadraticFlux(
            eq,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=src,
            vacuum=True,
            bs_chunk_size=10,
        )
        named = {
            # essential physics: the coils must carry the net poloidal current the
            # equilibrium needs. The bundle omitted it and pinned one coil's current
            # instead, which does not enforce the same thing.
            "linking_current": O.LinkingCurrentConsistency(
                eq, coilset, eq_fixed=True, weight=w
            ),
            "length": O.CoilLength(
                coilset,
                bounds=(0, BUNDLE["length_max"]),
                normalize_target=True,
                grid=src,
                weight=w,
            ),
            # coarse grid on purpose: the C0 faceting is intended, so do not resolve
            # and penalize the hinges. |curvature| >= 0, so the bundle's (-1, 2) lower
            # bound (written for the old SIGNED curvature) is inert; 0 is equivalent.
            "curvature": O.CoilCurvature(
                coilset,
                bounds=(0.0, BUNDLE["curv_hi"]),
                normalize_target=False,
                grid=coil_grid,
                weight=w,
            ),
            "coil_coil": O.CoilSetMinDistance(
                coilset,
                bounds=(BUNDLE["ccdist_min"], np.inf),
                normalize_target=False,
                grid=dist_grid,
                use_softmin=True,
                softmin_alpha=softmin_alpha,
                dist_chunk_size=2,
                weight=w,
            ),
            "plasma_coil": O.PlasmaCoilSetMinDistance(
                eq,
                coilset,
                bounds=(BUNDLE["pc_min"], np.inf),
                normalize_target=False,
                plasma_grid=plasma_grid,
                coil_grid=dist_grid,
                eq_fixed=True,
                use_softmin=True,
                softmin_alpha=softmin_alpha,
                dist_chunk_size=2,
                weight=w,
            ),
            "linking_number": O.CoilSetLinkingNumber(
                coilset, target=0, grid=src, weight=w
            ),
        }
        # pin coil 0's current only, leaving the rest free (run_exp2.sh)
        _idx = [False] * len(coilset)
        _idx[0] = True
        lin = (O.FixCoilCurrent(coilset, indices=_idx),)
    else:
        R0 = float(
            eq.compute("R0", grid=LinearGrid(M=8, N=8, NFP=eq.NFP, sym=False))["R0"]
        )
        lim = rescale_to_eq(R0)
        print(f"R0={R0:.4f}  kappa_max={lim['kappa_max']:.4f} (using {CURV_BOUND})")
        qflux = O.QuadraticFlux(
            eq,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            vacuum=True,
            bs_chunk_size=10,
        )
        named = {
            "linking_current": O.LinkingCurrentConsistency(
                eq, coilset, eq_fixed=True, weight=w
            ),
            "length": O.CoilLength(
                coilset,
                bounds=(0, lim["length_max"]),
                normalize_target=True,
                grid=coil_grid,
                weight=w,
            ),
            "curvature": O.CoilCurvature(
                coilset,
                bounds=(0, CURV_BOUND),
                normalize_target=True,
                grid=curv_grid,
                # historically curv_grid.num_nodes/(2*pi): cancels CoilCurvature's own
                # quadrature weight so the bound binds pointwise as written
                weight=w
                * (
                    curv_grid.num_nodes / (2 * np.pi)
                    if curv_wmult is None
                    else curv_wmult
                ),
            ),
            "coil_coil": O.CoilSetMinDistance(
                coilset,
                bounds=(lim["d_cc_min"], np.inf),
                normalize_target=True,
                grid=coil_grid,
                use_softmin=True,
                softmin_alpha=100.0,
                dist_chunk_size=2,
                weight=w,
            ),
            "plasma_coil": O.PlasmaCoilSetMinDistance(
                eq,
                coilset,
                bounds=(lim["d_cs_min"], np.inf),
                normalize_target=True,
                plasma_grid=plasma_grid_nosym,
                coil_grid=coil_grid,
                eq_fixed=True,
                use_softmin=True,
                softmin_alpha=100.0,
                dist_chunk_size=2,
                weight=w,
            ),
            "linking_number": O.CoilSetLinkingNumber(
                coilset, target=0, grid=coil_grid, weight=w
            ),
        }
        if rep in ("arc", "circle"):  # polar only: transverse has no pole
            from arc_pole_distance import ArcPoleDistance

            named["pole"] = ArcPoleDistance(
                coilset, bounds=(POLE_BOUND, np.inf), normalize=False, weight=w
            )
        lin = (O.FixSumCoilCurrent(coilset),)

    bad = set(drop) - set(named)
    errorif(bool(bad), ValueError, f"unknown terms to drop: {sorted(bad)}")
    cons = tuple(v for k, v in named.items() if k not in drop)
    print(
        f"rep={rep} mode={mode} convention={convention} dim_x={coilset.dim_x} "
        f"terms={[k for k in named if k not in drop]}"
        + (f" DROPPED={sorted(drop)}" if drop else "")
    )
    if mode == "auglag":
        # Under "paper", the current-fixing LINEAR constraint travels with the AL
        # constraints; Optimizer._parse_constraints splits them and projects the
        # linear ones out (optimizer.py:540, :720). The legacy conventions keep the
        # old behaviour -- lin dropped -- so their old runs still reproduce.
        return (
            eq,
            coilset,
            O.ObjectiveFunction((qflux,)),
            cons + (lin if convention == "paper" else ()),
            qflux,
        )
    if mode == "unc":
        return eq, coilset, O.ObjectiveFunction((qflux,)), lin, qflux
    if mode == "pen":
        return eq, coilset, O.ObjectiveFunction((qflux, *cons)), lin, qflux
    raise ValueError(f"unknown mode {mode!r}")


def run(
    gate,
    maxiter,
    outdir,
    device,
    smoke=False,
    mu_gate=None,
    tag=None,
    rep="arc",
    fp_N=14,
    curv_wmult=None,
    max_inner_iter=None,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-8,
    ctol=1e-6,
    convention="paper",
    softmin_alpha=100.0,
    nc=PAPER["nc"],
    B=PAPER["B"],
    M=PAPER["M"],
    r_over_a=PAPER["r_over_a"],
    length_max=None,
    curv_max=None,
    dcc_min=None,
    feas_margin=0.0,
    tau=None,
    dist_method="point",
    drop=(),
    max_trust_radius=None,
    fit_method="parameter",
    xyz_N=6,
    dcc_pair_mode="per_coil",
    dcc_neighbors=None,
    x_scale="auto",
    jac_scale_ratchet=True,
    tilt_deg=0.0,
    tilt_axis="rock",
    tilt_spec=None,
    arc_bulge=None,
    mode_kick=0.0,
    mode_ns=(2, 3),
    start_from=None,
    curv_ms_max=None,
    check_every=0,
    dcc_signed=False,
    dcc_floor=None,
    dcc_path_n=4,
    stop_on_stationary=False,
    stat_tol=1e-5,
    stat_patience=2,
):
    """Solve once with the given multiplier gate; dump outer_diag to JSON."""
    if _DEVICE_SET is None and device == "gpu":
        # Only reached when `run()` is called from an imported module rather than from
        # this file's __main__, where the block at the top already selected the device.
        # It is very likely TOO LATE here -- see that block -- so it warns rather than
        # pretending it worked.
        from desc import set_device

        set_device("gpu")

    # State the device that was ACTUALLY obtained, in every run log. `set_device("gpu")`
    # selects among available GPUs and does not fail loudly if the box has none, so a run
    # can silently execute on CPU at roughly half speed with nothing in the record to say
    # so. This line is the evidence; `tools/monitor.py` reads it back.
    try:
        import jax as _jax

        _devs = _jax.devices()
        print(
            f"DEVICE requested={device} obtained={_devs[0].platform} "
            f"({len(_devs)}x {_devs[0].device_kind})",
            flush=True,
        )
        if device == "gpu" and _devs[0].platform != "gpu":
            print(
                "DEVICE WARNING: --device gpu was requested but JAX is on "
                f"{_devs[0].platform!r}. The run will proceed, roughly 2x slower, and "
                "its numbers will not be bit-comparable with GPU runs in the same "
                "chain.",
                flush=True,
            )
    except Exception as _e:  # a diagnostic must never end a run
        print(f"DEVICE unknown ({type(_e).__name__}: {_e})", flush=True)

    import numpy as np

    from desc.optimize import Optimizer

    # coarse grids purely to check the plumbing end to end; not a physics result
    kw = dict(curv_grid_N=13, coil_grid_N=12, plasma_M=6, plasma_N=6) if smoke else {}
    eq, coilset, objective, constraints, qflux = build(
        rep=rep,
        fp_N=fp_N,
        curv_wmult=curv_wmult,
        convention=convention,
        softmin_alpha=softmin_alpha,
        nc=nc,
        B=B,
        M=M,
        r_over_a=r_over_a,
        length_max=length_max,
        curv_max=curv_max,
        dcc_min=dcc_min,
        feas_margin=feas_margin,
        dist_method=dist_method,
        drop=drop,
        fit_method=fit_method,
        xyz_N=xyz_N,
        dcc_pair_mode=dcc_pair_mode,
        dcc_neighbors=dcc_neighbors,
        tilt_deg=tilt_deg,
        tilt_axis=tilt_axis,
        tilt_spec=tilt_spec,
        arc_bulge=arc_bulge,
        mode_kick=mode_kick,
        mode_ns=mode_ns,
        start_from=start_from,
        curv_ms_max=curv_ms_max,
        dcc_signed=dcc_signed,
        **kw,
    )
    # the source grid the solve integrated on -- the verification must reuse it,
    # or it silently falls back to DESC's uniform LinearGrid(N=2*coil.N+5), which
    # does not bracket the hinges of a C0 arc.
    src_grid, _ = _source_grid(
        coilset,
        PAPER["gl_m"] if convention == "paper" else BUNDLE["gl_m"],
        PAPER["uniform_src_N"] if convention == "paper" else BUNDLE["uniform_src_N"],
    )
    # --- periodic stationarity check + checkpoint --------------------------------
    # Both planar runs to date spent ~70% of their wall time after the answer was
    # settled (cost within 0.1% of final by iter ~90, `viol` settled by ~130-170) and
    # then terminated on `maxiter` anyway. The AL exposes a callback that stops the
    # solve GRACEFULLY, so an early stop still returns through the normal path and the
    # h5/json get written. `--check-every` alone only INSTRUMENTS (it never stops);
    # `--stop-on-stationary` is what arms the early exit, and it is deliberately a
    # separate flag so the threshold can be calibrated from recorded data first.
    ckpt_log = []
    _cb = None
    os.makedirs(outdir, exist_ok=True)  # the callback checkpoints into it

    # --- topology guard: reject any trial step that takes d_cc below a floor -------
    # The floor is NOT the d_cc bound. It sits far below it (the bound is ~88 mm;
    # a floor of ~30 mm is 3x clear), so it is zero-effect in normal operation and
    # only fires in a region already adjudicated infeasible. It cannot move an
    # accepted optimum; it can only stop the solve walking a coil through another.
    _veto = None
    _veto_log = {"n": 0, "worst": float("inf")}
    if dcc_floor is not None:
        _dcc_obj = [
            c for c in constraints if getattr(c, "name", "").startswith("coil-coil")
        ]
        if not _dcc_obj:
            raise ValueError("--dcc-floor needs the coil_coil term; it was dropped")
        _dcc_obj = _dcc_obj[0]
        if not _dcc_obj.built:
            _dcc_obj.build(verbose=0)

        def _min_dcc(_x):
            """True minimum coil-coil distance at the state vector `_x`."""
            _p = objective.unpack_state(_x, False)[0]
            # compute_unscaled returns the distances themselves, in metres; the
            # scaled/error form is one-sided (exactly 0 inside the bound) and so
            # cannot see how far BELOW the floor a step would land.
            return float(np.min(np.asarray(_dcc_obj.compute_unscaled(_p))))

        def _veto(_x_new, _x_old, _n=dcc_path_n):
            # Check the straight line between the two iterates, not just the
            # endpoint: a crossing that happens mid-step and unwinds by the far end
            # would otherwise be invisible -- which is exactly the blind spot
            # CONVENTIONS 8i identified and never closed.
            for _t in np.linspace(1.0 / _n, 1.0, _n):
                _d = _min_dcc(_x_old + _t * (_x_new - _x_old))
                _veto_log["worst"] = min(_veto_log["worst"], _d)
                if not np.isfinite(_d) or _d < dcc_floor:
                    _veto_log["n"] += 1
                    return True
            return False

    if check_every:
        from kkt_diag import kkt_split
        from verify import bn_stats

        _nl = [c for c in constraints if not getattr(c, "linear", False)]
        _st = {"n": 0, "hits": 0}
        _ckpt_h5 = os.path.join(outdir, "checkpoint.h5")

        def _cb(x, *_a):
            _st["n"] += 1
            if _st["n"] % check_every:
                return False
            try:
                ck = coilset.copy()
                ck.params_dict = objective.unpack_state(x, False)[0]
                # Written EVERY check, so a killed run still leaves a usable iterate.
                # Also keep a per-iteration copy: the AL does NOT end on its best
                # iterate, so the trajectory has to be retained to choose from. Name it
                # HERE, where the iteration is known exactly -- an external watcher
                # cannot name these safely, because the h5 lands well before the log
                # line identifying it (kkt_split runs in between, taking ~30-60s).
                ck.save(_ckpt_h5)
                try:
                    ck.save(os.path.join(outdir, f"ckpt_it{_st['n']:05d}.h5"))
                except Exception:
                    pass  # the rolling checkpoint above is the one that must not fail
                # The actual field error, on the solve's OWN source grid. Without
                # this a checkpoint reports only AL cost, whose ratio to bn is not
                # stable across configurations (measured 2.90-4.01 across the B
                # sweep alone), so the trajectory could not be read in the metric
                # the campaign is judged on until the run had finished. One extra
                # Biot-Savart on the 32x32 half-period grid -- cheap next to
                # kkt_split, which dominates this callback.
                try:
                    _bs = bn_stats(ck, eq, src_grid)
                except Exception as _e:  # a diagnostic must never end a run
                    _bs = {
                        "mean": float("nan"),
                        "max": float("nan"),
                        "error": f"{type(_e).__name__}: {_e}",
                    }

                rows = kkt_split(ck, qflux, _nl)
                # sec 3a: read the largest tol whose active set has no wrong signs
                ok = [r for r in rows if r.get("n_wrong_sign") == 0]
                r = max(ok, key=lambda z: z["tol"]) if ok else {}
                dr = float(r.get("descent_rel", float("nan")))
                nr = float(r.get("null_resid", float("nan")))
                # No threshold with a clean active set means sec 3a cannot adjudicate
                # -- which is itself the verdict "nowhere near stationary". Keep the
                # tightest row anyway so a NaN checkpoint still carries a magnitude,
                # and record how far the active set is from clean.
                r0 = min(rows, key=lambda z: z["tol"]) if rows else {}
                minws = min((z.get("n_wrong_sign", -1) for z in rows), default=None)
                ckpt_log.append(
                    {
                        "iter": _st["n"],
                        "bn_mean": _bs["mean"],
                        "bn_max": _bs["max"],
                        "descent_rel": dr,
                        "null_resid": nr,
                        "n_active": r.get("n_active"),
                        "binds": r.get("binds"),
                        "adjudicable": bool(ok),
                        "min_n_wrong_sign": minws,
                        "descent_rel_tightest": float(
                            r0.get("descent_rel", float("nan"))
                        ),
                        "n_active_tightest": r0.get("n_active"),
                        # keep EVERY threshold: |Pg|/|g| spans ~9x across
                        # tols on one iterate (measured 0.4643 at 1e-6 down
                        # to 0.0499 at 1e-2), so the number is
                        # uninterpretable without the row it came from.
                        "rows": [
                            {
                                k: (
                                    float(z[k])
                                    if isinstance(z.get(k), float)
                                    else z.get(k)
                                )
                                for k in (
                                    "tol",
                                    "n_active",
                                    "n_wrong_sign",
                                    "null_resid",
                                    "descent_rel",
                                )
                            }
                            for z in rows
                        ],
                    }
                )
                if ok:
                    print(
                        f"  [ckpt it={_st['n']:>4}] "
                        f"bn={_bs['mean']:.4e} max={_bs['max']:.3e} "
                        f"desc/f0={dr:.3e} "
                        f"|Pg|/|g|={nr:.4f} n_active={r.get('n_active')} "
                        f"binds={r.get('binds')}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [ckpt it={_st['n']:>4}] "
                        f"bn={_bs['mean']:.4e} max={_bs['max']:.3e} -- "
                        f"NOT ADJUDICABLE (min wrong-sign "
                        f"{minws} at every tol) -- tightest tol={r0.get('tol')}: "
                        f"desc/f0={float(r0.get('descent_rel', float('nan'))):.3e} "
                        f"n_active={r0.get('n_active')}",
                        flush=True,
                    )
            except Exception as e:  # a diagnostic must never kill the solve
                print(
                    f"  [ckpt it={_st['n']}] FAILED {type(e).__name__}: {e}", flush=True
                )
                return False
            if not stop_on_stationary or not (dr == dr):
                return False
            # `desc/f0` ALONE IS NOT A STOP CRITERION. Sec 3a's thresholds were
            # calibrated on converged iterates, where `viol` was already tiny, so they
            # tacitly assume feasibility. MEASURED on the 3-axis run: iteration 100 read
            # desc/f0 = 2.03e-05 -- within 2x of the "finished" threshold -- while the
            # constraint violation was 1.4e-01. The active set is computed AT the
            # iterate, so an infeasible point can look stationary. Stopping there would
            # return a stationary-looking, badly infeasible coilset. So gate the stop on
            # hard feasibility, adjudicated the sec 7 way (fine grid, hard minima), and
            # only pay for it when `desc/f0` says we would otherwise stop.
            feas_ok = False
            if dr < stat_tol:
                try:
                    from verify import hard_geometry, paper_bounds

                    _b = paper_bounds(eq)
                    _g = hard_geometry(ck, eq, n=400)
                    _mg = {
                        "length": _g["max_length"] / _b["length_max"],
                        "kappa": _g["max_max_kappa"] / _b["kappa_max"],
                        "d_cc": _g["coil_coil_distance"] / _b["d_cc_min"],
                        "d_pc": _g["coil_surface_distance"] / _b["d_pc_min"],
                    }
                    feas_ok = (
                        _mg["length"] <= 1.0
                        and _mg["kappa"] <= 1.0
                        and _mg["d_cc"] >= 1.0
                        and _mg["d_pc"] >= 1.0
                        and _g["n_linked_pairs_gl"] == 0
                    )
                    ckpt_log[-1]["margins"] = {k: float(v) for k, v in _mg.items()}
                    ckpt_log[-1]["hard_feasible"] = bool(feas_ok)
                    print(
                        f"  [ckpt it={_st['n']:>4}] desc/f0 below {stat_tol:g}; hard "
                        f"feasibility {'OK' if feas_ok else 'FAILED'} "
                        f"{ {k: round(float(v), 4) for k, v in _mg.items()} }",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"  [ckpt] feasibility check failed "
                        f"{type(e).__name__}: {e} -- not stopping",
                        flush=True,
                    )
                    feas_ok = False
            # require `stat_patience` consecutive hits: one check is noisy, and `viol`
            # is non-monotonic across the tail (measured: 5.8e-06 -> 1.2e-04 -> 4.4e-05)
            _st["hits"] = _st["hits"] + 1 if (dr < stat_tol and feas_ok) else 0
            if _st["hits"] >= stat_patience:
                print(
                    f"  [ckpt] stationary {stat_patience}x consecutively "
                    f"(desc/f0 {dr:.3e} < {stat_tol:g}) -- stopping early",
                    flush=True,
                )
                return True
            return False

    opt = Optimizer("lsq-auglag")
    (new_coils,), res = opt.optimize(
        coilset,
        objective=objective,
        constraints=constraints,
        maxiter=maxiter,
        verbose=3,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        ctol=ctol,
        copy=True,
        # `x_scale="auto"` (DESC's default) scales each variable by its Jacobian
        # COLUMN NORM, via compute_jac_scale -- which RATCHETS:
        #     scale_inv = maximum(column_norms, prev_scale_inv)
        # so `scale = 1/scale_inv` only ever shrinks. A column norm that spikes once
        # throttles that variable's trust-region allowance permanently, even after the
        # Jacobian settles. This is standard MINPACK/scipy `x_scale="jac"` behaviour,
        # not a DESC defect, but it is not free: MEASURED start->final column-norm
        # growth is median 0.93 / max 10.7 for the arc run that converged on `gtol`
        # (1 of 232 columns grew >10x, none >100x), against median 2.23 / max 959 for
        # the planar run that stalled (47 of 192 grew >10x, 13 >100x). Pass
        # `--x-scale 1` to disable it and step in unscaled variables.
        x_scale=x_scale,
        options=dict(
            # Topology guard. `d_cc` is smooth and already built, and linking number
            # is a HOMOTOPY INVARIANT -- it cannot change unless two curves actually
            # intersect. So keeping d_cc away from zero ALONG THE PATH is sufficient
            # to preserve topology, and needs no derivative of anything topological.
            # Measured on `planar_N14_cert` (CONVENTIONS 8o): d_cc fell 85.7 -> 63.6
            # -> 30.2 -> 11.0 -> 5.0 mm over 20 iterations and the coils passed
            # through each other between it=15 and it=20, with every step well under
            # `--max-trust-radius`. Nothing in the objective could see it: the d_cc
            # penalty peaked at 0.134 against a flux term of ~8.3.
            # This is a HARD WALL -- the trust-region model does not know about it,
            # so a rejected step is re-proposed in the same direction on a smaller
            # radius. Whether that stalls or whether the outer (y, mu) update rescues
            # it (the radius resets on `al_changed`) is what `n_veto` measures.
            **({} if _veto is None else {"step_veto": _veto}),
            # graceful early exit / checkpointing; None unless --check-every
            **({} if _cb is None else {"callback": _cb}),
            tr_method="svd",
            # Penalty growth factor. DESC's default is 10, applied on the violated rows
            # at EVERY inner stall -- and `max_inner_stalls=50` below permits up to 50
            # of them, so `mu` can escalate by 10^26 (v8 reached 8.17e+04, saturating
            # against max_penalty). Each escalation adds a stiff block to the
            # least-squares system: measured at v9 iteration 133 the inner solve was
            # within 10x of gtolk, then three bumps in five iterations drove optimality
            # 260x WORSE and it never recovered. `tau` is the right lever rather than
            # `max_inner_stalls`, because the stall count also gates the trust-radius
            # reset (aug_lagrangian_ls.py:947) which sec 8d shows we need.
            **({} if tau is None else {"tau": tau}),
            # Cap on EVERY trust-region step, not just the first. The optimizer
            # takes discrete steps and nothing evaluates the path between them, so a
            # step large enough to carry a coil through another changes the topology
            # with d_cc satisfied at both endpoints -- invisible to every constraint.
            # v11 did this: window max step peak 3.381, one genuinely linked pair
            # (|Lk| = 1.000000, converged) with the coils 85 mm apart. See 8i.
            **(
                {}
                if max_trust_radius is None
                else {"max_trust_radius": max_trust_radius}
            ),
            # See the note in aug_lagrangian_ls.py: the jac-scale ratchet throttles a
            # variable permanently once its column norm spikes. Default True keeps the
            # historical (MINPACK/scipy) behaviour so old runs reproduce.
            **({} if jac_scale_ratchet else {"jac_scale_ratchet": False}),
            max_inner_stalls=50,
            **({} if max_inner_iter is None else {"max_inner_iter": max_inner_iter}),
            track_outer=True,
            # NB no `track_steps` here: that is a `least_squares.py` option. lsq_auglag
            # runs its own inner trust-region loop inline and rejects it outright
            # ("Unknown options: ['track_steps']"). Per-ATTEMPT diagnostics on an AL
            # subproblem would need the same instrumentation added to
            # aug_lagrangian_ls.py.
            y_update_gate=gate,
            **({} if mu_gate is None else {"min_window_progress": mu_gate}),
        ),
    )

    os.makedirs(outdir, exist_ok=True)
    if tag is None:
        tag = "gate" if rep == "arc" else (f"planarN{fp_N}" if rep == "planar" else rep)
        if rep == "xyz":
            tag = f"xyzN{xyz_N}"
        if convention == "paper":
            tag = f"{rep}_B{B}M{M}_nc{nc}_roa{r_over_a:g}"
            if length_max is not None:
                tag += f"_L{length_max:g}"
            if curv_max is not None:
                tag += f"_k{curv_max:g}"
        tag += f"_{gate}"
        if curv_wmult is not None:
            tag += f"_kw{curv_wmult:g}"
        tag += f"_{convention}"
        tag += "_mii-off" if max_inner_iter is None else f"_mii{max_inner_iter}"
        if (gtol, ctol) != (1e-8, 1e-6):
            tag += f"_g{gtol:g}_c{ctol:g}"
        if mu_gate is not None:
            tag += f"_mu{mu_gate:g}"
    tag += "_smoke" if smoke else ""
    new_coils.save(os.path.join(outdir, f"{tag}.h5"))
    from bench.bench import paper_metrics
    from verify import print_health, print_verify, solver_health, verify

    metrics = paper_metrics(new_coils, eq)
    metrics_initial = paper_metrics(coilset, eq)
    # CONVENTIONS.md sec 7: feasibility is adjudicated ONLY by hard minima on a fine
    # grid, never from the objective's own values.
    v_final = verify(new_coils, eq, src_grid, label=f"{tag} FINAL")
    v_start = verify(coilset, eq, src_grid, label=f"{tag} START")
    print_verify(v_start)
    print_verify(v_final)
    # CONVENTIONS.md sec 8f: named solver failure modes, from what `res` already
    # carries. `constraints` here excludes the linear current-fixing constraint.
    health = solver_health(
        res,
        [c for c in constraints if not getattr(c, "linear", False)],
        coilset,
        new_coils,
        ctol=ctol,
        maxiter=maxiter,
    )
    print_health(health)
    # Write the record BEFORE kkt_split, and again after it fills in. kkt_split is
    # the peak-memory step of the whole driver -- it OOM-killed al_B3 on 2026-09-03
    # at nit=496, after the h5 and both verify blocks were already done -- and a
    # SIGKILL is not an exception, so the try/except below cannot save the record
    # from it. Dumping first costs one extra write and means a kill during the
    # diagnostic loses the stationarity block only, not the run.
    rec = {
        "health": health,
        "stationarity": None,
        "checkpoints": ckpt_log,
        "rep": rep,
        "convention": convention,
        "nc": nc,
        "B": B,
        "M": M,
        # B/M are the PAPER DEFAULTS for a smooth rep and mean nothing there -- the
        # mode number is what identifies the run. Without these the run index reported
        # al_xyz_N4 as "4/5/3", i.e. as a B=5 arc.
        "xyz_N": xyz_N,
        "fp_N": fp_N,
        "r_over_a": r_over_a,
        "length_max_override": length_max,
        "curv_max_override": curv_max,
        # Without these a record cannot say whether a run carried a d_cc backoff or
        # enforced mean-square curvature -- the difference between a sweep-grade point
        # and a pilot. They were missing until 2026-09-04.
        "dcc_min_override": dcc_min,
        "curv_ms_max": curv_ms_max,
        "feas_margin": feas_margin,
        "verify": v_final,
        "verify_initial": v_start,
        "curv_wmult": curv_wmult,
        "max_inner_iter": max_inner_iter,
        "tau": tau,
        "tols": {"ftol": ftol, "xtol": xtol, "gtol": gtol, "ctol": ctol},
        "dim_x": int(coilset.dim_x),
        "metrics": metrics,
        "metrics_initial": metrics_initial,
        "gate": gate,
        "mu_gate": mu_gate,
        "maxiter": maxiter,
        "message": str(res["message"]),
        "success": bool(res["success"]),
        "nit": int(res["nit"]),
        "n_outer": int(res["n_outer"]),
        "cost": float(res["cost"]),
        "optimality": float(res["optimality"]),
        "constr_violation": float(res["constr_violation"]),
        "y_absmax_final": float(np.max(np.abs(np.asarray(res["y"])))),
        "mu_mean_final": float(np.mean(np.asarray(res["penalty_param"]))),
        "outer_log": [[int(i), str(r)] for i, r in res["outer_log"]],
        "outer_diag": res["outer_diag"],
    }
    path = os.path.join(outdir, f"{tag}.json")

    def _dump_rec():
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)

    _dump_rec()
    print(f"\nwrote {path}  (stationarity pending)")

    # CONVENTIONS.md sec 8j: did the solve stop because it was DONE, or because it ran
    # out of trust region? `solver_health` reads the solver's own telemetry and cannot
    # tell those apart -- planar reported `maxiter` with an apparently large leftover
    # gradient and was still effectively optimal. A LINEAR test (kkt_diag.kkt_split,
    # |Pg|/|g|) cannot tell them apart either: it read 0.89 there. Only stepping along
    # the projected direction and evaluating settles it, so do that and record it.
    print("\n=== STATIONARITY (kkt_split: direction quality AND step budget) ===")
    try:
        from kkt_diag import kkt_split

        rec["stationarity"] = kkt_split(
            new_coils,
            qflux,
            [c for c in constraints if not getattr(c, "linear", False)],
        )
    except Exception as e:  # never let a diagnostic lose the run record
        print(f"  kkt_split failed: {type(e).__name__}: {e}")
        rec["stationarity"] = [{"error": f"{type(e).__name__}: {e}"}]
    _dump_rec()
    print(f"\nwrote {path}")
    report_one(rec)


def _parse_tilt_spec(text):
    """'rock,yaw;rock,yaw;...' -> [(float, float), ...]. Degrees, one pair per coil."""
    if text is None or not str(text).strip():
        return None
    out = []
    for i, chunk in enumerate(str(text).split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--tilt-spec entry {i} is {chunk!r}; expected 'rock,yaw'"
            )
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--tilt-spec entry {i} is {chunk!r}; both fields must be numbers"
            )
    return out or None


def report_one(rec):
    """Print the decisive columns for one run."""
    d = rec["outer_diag"]
    print(
        f"\n=== rep={rec.get('rep','arc')} gate={rec['gate']} "
        f"mu_gate={rec.get('mu_gate')} dim_x={rec.get('dim_x')} ==="
    )
    m, m0 = rec.get("metrics"), rec.get("metrics_initial")
    if m:
        print(
            f"  normalized_BdotN: {m0['normalized_BdotN']:.4e} -> "
            f"{m['normalized_BdotN']:.4e}   "
            f"max_BdotN/B {m['max_BdotN_over_B']:.4e}  "
            f"mean|B| {m['mean_AbsB']:.4f}"
        )
    print(f"  {rec['message']}")
    print(
        f"  nit={rec['nit']}  n_outer={rec['n_outer']}  "
        f"cost={rec['cost']:.4e}  viol={rec['constr_violation']:.3e}"
    )
    reasons = {}
    for _, r in rec["outer_log"]:
        reasons[r] = reasons.get(r, 0) + 1
    print(f"  outer_log reasons: {reasons}")
    if not d:
        print("  no outer_diag rows")
        return
    print(
        f"  |y|max: first={d[0]['y_absmax']:.4e} last={d[-1]['y_absmax']:.4e}"
        f"   mu_mean: first={d[0]['mu_mean']:.3e} last={d[-1]['mu_mean']:.3e}"
    )
    blocked = [r["n_blocked"] for r in d]
    print(
        f"  n_rows={d[0]['n_rows']}  n_blocked: "
        f"min={min(blocked)} max={max(blocked)} "
        f"mean={sum(blocked)/len(blocked):.1f}"
    )
    print(
        f"  max |y| on a blocked row (over run): "
        f"{max(r['y_absmax_blocked'] for r in d):.4e}"
    )
    print(
        f"  max |dy| per update (over run):      "
        f"{max(r['dy_absmax'] for r in d):.4e}"
    )
    hdr = (
        "   iter reason      ctolk      viol    mu_mean   |y|max     |dy|"
        "   wprog    wstep   mu+"
    )
    print(hdr)
    step = max(1, len(d) // 25)
    for r in d[::step]:
        print(
            f"  {r['iteration']:5d} {r['reason']:6s} {r['ctolk']:.3e} "
            f"{r['constr_violation']:.3e} {r['mu_mean']:.3e} "
            f"{r['y_absmax']:.3e} {r['dy_absmax']:.3e} "
            f"{r.get('window_rel_progress', float('nan')):.2e} "
            f"{r.get('window_max_step', float('nan')):.2e} "
            f"{'Y' if r.get('mu_growth_allowed', True) else 'n'}"
        )


def report(outdir):
    """Print every run found in `outdir`, oldest first."""
    import glob

    for p in sorted(glob.glob(os.path.join(outdir, "*.json")), key=os.path.getmtime):
        rec = json.load(open(p))
        print(f"\n----- {os.path.basename(p)} -----")
        report_one(rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", choices=["residual", "violation"], default="residual")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--out", default="runs/ymask")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--report", metavar="OUTDIR", default=None)
    ap.add_argument(
        "--smoke", action="store_true", help="coarse grids, plumbing check only"
    )
    ap.add_argument(
        "--mu-gate",
        type=float,
        default=None,
        dest="mu_gate",
        help="min_window_progress: withhold the mu bump when the "
        "subproblem reduced its AL cost by more than this",
    )
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--rep",
        choices=["arc", "planar", "piecewise", "circle", "xyz"],
        default="piecewise",
    )
    ap.add_argument(
        "--convention",
        choices=["paper", "bundle", "c0"],
        default="paper",
        help="how the engineering terms are specified. 'paper' (DEFAULT) is "
        "c0/CONVENTIONS.md: Hurwitz Table 1 rescaled to this eq, in "
        "PHYSICAL units with normalize_target=True, hard min, start BUILT "
        "at nc=4 r/a=3.3, FixSumCoilCurrent enforced under AL. 'bundle' and "
        "'c0' are the legacy stacks, kept only so old runs reproduce -- "
        "their bounds re-anchor on the start and must not be used in a sweep.",
    )
    ap.add_argument(
        "--nc", type=int, default=PAPER["nc"], help="unique coils (paper convention)"
    )
    ap.add_argument(
        "-B",
        type=int,
        default=PAPER["B"],
        dest="B",
        help="arcs per coil; the facet sweep axis. B>=3 at r/a=3.3",
    )
    ap.add_argument(
        "-M", type=int, default=PAPER["M"], dest="M", help="transverse modes per arc"
    )
    ap.add_argument(
        "--r-over-a",
        type=float,
        default=PAPER["r_over_a"],
        dest="r_over_a",
        help="coil minor radius / eq minor radius. The nc=4 feasible "
        "window is [3.151, 3.401]; 3.3 is its equal-margin point.",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=None,
        help="penalty growth factor applied to violated rows at each "
        "inner stall. DESC default 10; with max_inner_stalls=50 that "
        "permits runaway (v8: mu -> 8.17e+04). Lower values (2) keep the "
        "escalation inside what the inner solve can absorb.",
    )
    ap.add_argument(
        "--length-max",
        type=float,
        default=None,
        dest="length_max",
        help="override the CoilLength bound in METRES (sweep axis). "
        "Default: the paper value, 5.153350.",
    )
    ap.add_argument(
        "--curv-max",
        type=float,
        default=None,
        dest="curv_max",
        help="override the CoilCurvature bound in 1/m (sweep axis). "
        "Default: the paper value, 11.642912.",
    )
    ap.add_argument(
        "--softmin-alpha",
        type=float,
        default=100.0,
        dest="softmin_alpha",
        help="softmin sharpness for both distance objectives. softmin "
        "overestimates the true minimum by ~1/alpha (measured: 1.39 cm "
        "at alpha=100), so raise it if the clearance margin matters.",
    )
    ap.add_argument("--ftol", type=float, default=1e-6)
    ap.add_argument("--xtol", type=float, default=1e-6)
    ap.add_argument(
        "--gtol",
        type=float,
        default=1e-8,
        help="floor on gtolk = max(omega/mean(mu)**alpha_omega, gtol). "
        "omega = min(g_norm, 1e-2), so gtolk starts at 1e-3 unless "
        "gtol is raised above it.",
    )
    ap.add_argument(
        "--ctol",
        type=float,
        default=1e-6,
        help="floor on ctolk = max(eta/mean(mu)**alpha_eta, ctol). "
        "eta=1e-2, alpha_eta=0.1, so ctolk starts at 7.94e-3 at "
        "mu=10 unless ctol is raised above it.",
    )
    ap.add_argument(
        "--max-inner-iter",
        type=int,
        default=0,
        dest="max_inner_iter",
        help="cap on inner iterations per subproblem. DESC's own default "
        "is None (off); this harness has historically forced 50, "
        "which makes every outer update a forced one labelled "
        "'stall' and prevents gtolk from ever firing. Pass 0 for "
        "None (off).",
    )
    ap.add_argument(
        "--curv-wmult",
        type=float,
        default=None,
        dest="curv_wmult",
        help="CoilCurvature weight multiplier; default (None) keeps the "
        "historical num_nodes/(2*pi) = 64.1. Under AL the weight "
        "cancels out of the feasible set (compute_scaled and "
        "bounds_scaled carry it equally), so this changes only the "
        "conditioning of the constraint block.",
    )
    ap.add_argument(
        "--max-trust-radius",
        type=float,
        default=None,
        dest="max_trust_radius",
        help="cap on every trust-region step. Uncapped, one large step can carry a "
        "coil THROUGH another between two feasible endpoints, changing the topology "
        "invisibly: v11 peaked at 3.381 and ended with a real linked pair while d_cc "
        "was satisfied. d_cc cannot guard against this and neither can the linking "
        "objective (unreadable at contact). See CONVENTIONS.md sec 8i.",
    )
    ap.add_argument(
        "--distance-method",
        choices=["point", "segment"],
        default="point",
        dest="dist_method",
        help="how CoilSetMinDistance measures coil-coil distance. 'point' "
        "(default, historical) minimizes over grid POINTS and cannot see a "
        "crossing that falls between nodes -- it read 6.4e-03 m of clearance "
        "for the intersecting v8 coils. 'segment' minimizes over the line "
        "SEGMENTS between points and is O(h^2); it is REQUIRED to recover a "
        "collapsed geometry. See CONVENTIONS.md sec 8i.",
    )
    ap.add_argument(
        "--drop",
        nargs="*",
        default=[],
        metavar="TERM",
        help="named engineering terms to remove entirely, e.g. "
        "--drop linking_number. The linking number is UNREADABLE whenever "
        "coils are close (gradient 2.9e+12 at v8 while its own residual is "
        "~0) and should be dropped. NOTE d_cc does NOT subsume it -- this help "
        "claimed otherwise until 2026-09-10 and the claim is FALSE: "
        "planar_N14_cert finished with d_cc at margin 1.029 and SIX LINKED "
        "PAIRS. Separation and topology are independent. Nothing constrains "
        "topology unless you pass --dcc-signed, and `linked pairs` must be read "
        "from the pairwise MATRIX on every result. CONVENTIONS 8i, 8n, 8o.",
    )
    ap.add_argument(
        "--fit-method",
        choices=["parameter", "chord"],
        default="parameter",
        dest="fit_method",
        help="abscissa PiecewisePlanarArcCoil.from_values fits the transverse "
        "deviation against. 'parameter' (default, historical) fits against the "
        "sample's CURVE PARAMETER while the curve reconstructs it at that CHORD "
        "FRACTION -- a different variable. The fit is then exact against the wrong "
        "target (residual 4e-16) and returns a sine arch: at B=2, 94.5 mm from the "
        "circle it was fit to, which is the whole reason B=2 read infeasible at "
        "r/a=3.3. 'chord' fits against the actual chord fraction. See "
        "CONVENTIONS.md sec 4.1.",
    )
    ap.add_argument(
        "--no-jac-scale-ratchet",
        action="store_false",
        dest="jac_scale_ratchet",
        help="recompute the Jacobian column-norm scaling FRESH each iteration instead "
        "of maximum(current, previous). The default ratchet means one column-norm "
        "spike throttles that variable's trust-region allowance for the rest of the "
        "solve; planar's column norms grew up to 959x (47 of 192 above 10x) where the "
        "arc run that converged grew 10.7x at most. Not the same as --x-scale 1, "
        "which removes scaling entirely and collapsed planar by iteration 13.",
    )
    ap.add_argument(
        "--x-scale",
        default="auto",
        dest="x_scale",
        help="variable scaling passed to Optimizer.optimize. 'auto'/'jac' (DESC "
        "default) scales by Jacobian column norm and RATCHETS -- compute_jac_scale "
        "takes maximum(column_norms, prev), so a variable throttled by one column-norm "
        "spike never recovers. Pass a number (e.g. 1) for fixed scaling. Planar's "
        "column norms grew up to 959x during its solve (47 of 192 columns >10x) where "
        "the converged arc run's grew 10.7x at most.",
    )
    ap.add_argument(
        "--feas-margin",
        type=float,
        default=0.0,
        dest="feas_margin",
        help="back every paper bound off by this FRACTION before enforcing it: "
        "length and curvature by (1-m), d_cc and d_pc by (1+m). `verify` still "
        "adjudicates against the unbacked-off paper bounds, so this can only make a "
        "reported PASS more honest. Fixes two measured failures: solver-tolerance "
        "overshoot (planar exceeded length by 3.33e-06 and was called INFEASIBLE; a "
        "few 1e-4 covers it, free) and knife-edge geometry (B=2's d_cc active set "
        "flipped between two rows 8.3e-05 apart; 2e-2 made it feasible for 0.07%% "
        "flux). NOTE length is the productive bound -- a large margin there buys "
        "feasibility with performance. CONVENTIONS 4.1.1 and 9.",
    )
    ap.add_argument(
        "--dcc-pair-mode",
        choices=["per_coil", "per_pair"],
        default="per_coil",
        dest="dcc_pair_mode",
        help="how CoilSetMinDistance reduces across neighbours. 'per_coil' (default, "
        "historical) emits one row per coil = the min over ALL other coils; that "
        "outer min is kinked wherever two neighbours are equidistant, and at B=2 two "
        "pairs sat 8.3e-05 apart straddling the bound with the argmin swapping inside "
        "one trust-region step. 'per_pair' emits one row per (coil, neighbour), so no "
        "min is taken across neighbours. Pair with --dcc-neighbors: dim_f grows "
        "n_neighbours-fold and the reverse-mode Jacobian cost with it (all 15 "
        "neighbours at N=150 wants 40.9 GB).",
    )
    ap.add_argument(
        "--dcc-neighbors",
        type=int,
        default=None,
        dest="dcc_neighbors",
        help="restrict the coil-coil distance to the N nearest coils by centroid. "
        "Default None = all. MEASURED lossless at N=3,4,6 for nc=4 B=2 (identical to "
        "all 15), because only 2-3 pairs are anywhere near the bound.",
    )
    ap.add_argument(
        "--dcc-min",
        type=float,
        default=None,
        dest="dcc_min",
        help="override the ENFORCED coil-coil distance bound (metres). This is the "
        "SAGITTA BACKOFF of SWEEP_BLUEPRINT sec 5.1 and is PRESCRIBED for the C0 arc "
        "family: the optimizer parks exactly on its own enforcement grid's bound, and "
        "on a piecewise-straight curve that grid over-reads clearance by up to 5.4%%. "
        "Enforce paper_bound + 1.4*kappa_bound*(L_bound/N)^2/8; `verify` still "
        "adjudicates against the unbacked-off paper bound, so it can only make a "
        "reported PASS honest. Points solved at DIFFERENT backoffs are still solving "
        "different problems, so size it from the formula rather than by hand. NOT "
        "needed for the smooth reps -- their enforcement-grid error is 0.01-0.23%%, "
        "~25x smaller than the rule prescribes. CONVENTIONS 8l, 8o.",
    )
    ap.add_argument(
        "--xyz-N",
        type=int,
        default=6,
        dest="xyz_N",
        help="FourierXYZ mode number for --rep xyz. DOF/coil = 3*(2N+1); N=6 gives "
        "39 against 35 for arc B=5 M=3 and FourierPlanar N=14, the closest match "
        "available on this ladder (N=5 gives 33).",
    )
    ap.add_argument(
        "--fp-N",
        type=int,
        default=14,
        dest="fp_N",
        help="FourierPlanar mode number; 14 matches arc B=5 M=3 DOF",
    )
    ap.add_argument(
        "--tilt-deg",
        type=float,
        default=0.0,
        dest="tilt_deg",
        help="COHERENT hand tilt of the FourierPlanar cold start (degrees): "
        "rotate every unique coil's normal by this angle about --tilt-axis, "
        "centers fixed. A rigid per-coil rotation -- the start is an exact "
        "circle, so it changes no curvature at all. Default 0 leaves the start "
        "bit-identical to every run made before the flag existed. --rep planar "
        "only; the basin probe of HANDOFF_multistart.md sec 3.",
    )
    ap.add_argument(
        "--tilt-axis",
        choices=["rock", "yaw"],
        default="rock",
        dest="tilt_axis",
        help="axis for --tilt-deg. 'rock' = local radial e_R (coil plane leans "
        "in Z); 'yaw' = machine z-hat (leans radially in/out). Rotation about "
        "the coil's own normal is an exact null for a circle, so these two are "
        "the only non-degenerate choices.",
    )
    ap.add_argument(
        "--tilt-spec",
        default=None,
        dest="tilt_spec",
        type=_parse_tilt_spec,
        help="PER-COIL tilt of the FourierPlanar start: "
        "'rock1,yaw1;rock2,yaw2;...' in degrees, one pair per unique coil (4 at "
        "nc=4), applied rock-then-yaw about the same axes as --tilt-deg. Unlike "
        "--tilt-deg, which rotates every coil by the SAME angle, this can reach a "
        "DIFFERENTIAL orientation. Measured (8p), the cold and continued basins differ "
        "almost entirely on coils 3-4 -- coil 3 by yaw -19 deg, coil 4 by rock -10 deg "
        "-- while the mean delta is only -4.3 deg, so a coherent tilt is nearly "
        "orthogonal to the direction that separates them. Composes with --tilt-deg "
        "(applied after it). Omit to disable.",
    )
    ap.add_argument(
        "--arc-bulge",
        type=float,
        default=None,
        dest="arc_bulge",
        help="ARC starts only: replace each arc's fitted transverse profile with a "
        "SINGLE sine mode, w(t) = a*sin(pi t), keeping the fitted hinges, tilts and "
        "arc_ref. Pass the amplitude `a` as a fraction of the chord half-length, or 0 "
        "to keep the fitted fundamental and drop only the higher modes. Omit to "
        "disable, which leaves every arc start bit-identical to before this existed. "
        "WHY: `from_values` fits the deviation as a graph over the CHORD, and at B=2 "
        "that chord is a DIAMETER whose target profile is a semicircle -- a sqrt with "
        "VERTICAL endpoint tangents that a truncated sine series cannot hold. The fit "
        "ripples, and because kappa is a second derivative every extra mode makes it "
        "WORSE: kappa_MS 7.93 (M=3) / 13.53 (M=5) / 19.98 (M=8) / 27.14 (M=10) against "
        "a 5.648225 bound. One mode is smooth, has kappa=0 at every hinge by "
        "construction, and peaks at only a*pi^2/L_chord^2. CONVENTIONS 8q.",
    )
    ap.add_argument(
        "--mode-kick",
        type=float,
        default=0.0,
        dest="mode_kick",
        help="hand SHAPE perturbation of the FourierPlanar start: add "
        "amp*r_0/(1+n^2) to each cos mode in --mode-ns. The 1/(1+n^2) weight "
        "equalizes each mode's CURVATURE contribution; a flat kick across "
        "modes reaches kappa ~40 /m against an 11.64 bound. 0 disables.",
    )
    ap.add_argument(
        "--mode-ns",
        default="2,3",
        dest="mode_ns",
        help="comma-separated cos modes for --mode-kick (default 2,3)",
    )
    ap.add_argument(
        "--dcc-signed",
        action="store_true",
        dest="dcc_signed",
        help="SIGN the coil-coil distance: negate the pairs that are LINKED, so a "
        "linked pair goes negative, wins the min, and reads as a violation that GROWS "
        "as the coils separate on the wrong side. Fixes the blind spot that the "
        "unsigned minimum RECOVERS after a crossing (measured: 85.7 -> 5.0 mm through "
        "the crossing, then back up through 28, 48, 75 mm while linked), which is why "
        "nothing caught the topology change. The sign is stop_gradient'd -- exact, "
        "since it is locally constant and d=0 at a flip -- so the linking number's "
        "2.9e+12 derivative never enters the graph. CONVENTIONS 8i, 8o.",
    )
    ap.add_argument(
        "--dcc-floor",
        type=float,
        default=None,
        dest="dcc_floor",
        metavar="METRES",
        help="topology guard: REJECT any trial step whose straight-line path takes "
        "the true coil-coil distance below this floor, shrinking the trust radius "
        "as for any rejected step. Linking number is a homotopy invariant, so it "
        "cannot change unless two curves intersect -- keeping d_cc off zero ALONG "
        "THE PATH preserves topology without differentiating anything topological. "
        "The floor is NOT the d_cc bound: set it well below (bound ~0.0880, floor "
        "~0.030) so it is zero-effect in normal operation and only fires in a "
        "region already infeasible. NOTE this is a hard wall the trust-region model "
        "cannot see, so a rejected step is re-proposed in the same direction on a "
        "smaller radius; `n_veto` in the record says whether that stalled. "
        "CONVENTIONS 8o.",
    )
    ap.add_argument(
        "--dcc-path-n",
        type=int,
        default=4,
        dest="dcc_path_n",
        help="how many points along each candidate step --dcc-floor tests. 1 checks "
        "only the endpoint and reproduces the blind spot of CONVENTIONS 8i (a "
        "crossing that unwinds by the far end is invisible). Cost is this many "
        "extra coil-coil distance evaluations per trial step, no Jacobians.",
    )
    ap.add_argument(
        "--check-every",
        type=int,
        default=0,
        dest="check_every",
        help="run kkt_diag.kkt_split every N accepted steps, log desc/f0 and "
        "|Pg|/|g|, and write outdir/checkpoint.h5 each time so a killed run still "
        "leaves a usable iterate. INSTRUMENTS ONLY -- it never stops the solve. "
        "0 (default) disables it entirely, leaving old runs bit-identical.",
    )
    ap.add_argument(
        "--stop-on-stationary",
        action="store_true",
        dest="stop_on_stationary",
        help="arm early exit: end the solve once --check-every reports desc/f0 below "
        "--stat-tol on --stat-patience consecutive checks. The AL callback terminates "
        "GRACEFULLY, so the h5/json are still written by the normal path. Requires "
        "--check-every. Calibrate the threshold from a --check-every run FIRST.",
    )
    ap.add_argument(
        "--stat-tol",
        type=float,
        default=1e-5,
        dest="stat_tol",
        help="desc/f0 below which an iterate counts as stationary "
        "(CONVENTIONS sec 3a: <1e-05 finished, >1e-03 not)",
    )
    ap.add_argument(
        "--start-from",
        default=None,
        dest="start_from",
        metavar="H5",
        help="REFINEMENT: build the start with the requested -B/-M, then transplant "
        "the geometry of this solved coilset onto it, zero-padding the shape modes "
        "(same B, lower-or-equal M). The lifted curve is geometrically IDENTICAL to "
        "the source, so the higher-M feasible set is a strict superset containing the "
        "source optimum and the run can only improve on it. Use this, not a cold "
        "higher-M run, to ask whether extra modes buy anything: cold runs land in "
        "their own basin and the scatter is the same size as the effect.",
    )
    ap.add_argument(
        "--curv-ms-max",
        type=float,
        default=None,
        dest="curv_ms_max",
        metavar="K",
        help="ENFORCE mean-square curvature (1/L)*int kappa^2 dl <= K per coil, in "
        "m^-2. OFF by default -- and this flag is the ENABLE SWITCH, not merely the "
        "bound: omit it and `curvature_ms` is silently absent from terms=[...]. That "
        "is how al_planar_N14 came to exceed it by 10.9%% while satisfying the "
        "pointwise bound, and be void. The paper's rescaled value is 5.648225 (NOT "
        "5.821456, which this help advertised until 2026-09-10: the quantity is "
        "(1/L)*int kappa^2 dl with no square root, so it is 1/m^2 and rescales as "
        "/f^2. A single power of f leaves the bound 3.07%% too loose -- "
        "SWEEP_BLUEPRINT sec 4.1.1). Distinct from --curv-max, which bounds "
        "|kappa| at every node INDEPENDENTLY: pointwise is a minimum bend RADIUS "
        "(winding) limit, this is a total-bending (stress) limit, and measured on "
        "al_B5 x1.5 they bind DIFFERENT coils.",
    )
    ap.add_argument(
        "--stat-patience",
        type=int,
        default=2,
        dest="stat_patience",
        help="consecutive stationary checks required before stopping",
    )
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(
            a.gate,
            a.maxiter,
            a.out,
            a.device,
            a.smoke,
            a.mu_gate,
            a.tag,
            a.rep,
            a.fp_N,
            a.curv_wmult,
            None if a.max_inner_iter == 0 else a.max_inner_iter,
            a.ftol,
            a.xtol,
            a.gtol,
            a.ctol,
            a.convention,
            a.softmin_alpha,
            a.nc,
            a.B,
            a.M,
            a.r_over_a,
            a.length_max,
            a.curv_max,
            a.dcc_min,
            a.feas_margin,
            a.tau,
            a.dist_method,
            tuple(a.drop),
            a.max_trust_radius,
            a.fit_method,
            a.xyz_N,
            a.dcc_pair_mode,
            a.dcc_neighbors,
            (a.x_scale if a.x_scale in ("auto", "jac") else float(a.x_scale)),
            a.jac_scale_ratchet,
            tilt_deg=a.tilt_deg,
            tilt_axis=a.tilt_axis,
            tilt_spec=a.tilt_spec,
            arc_bulge=a.arc_bulge,
            mode_kick=a.mode_kick,
            mode_ns=tuple(int(v) for v in str(a.mode_ns).split(",") if v.strip()),
            check_every=a.check_every,
            dcc_signed=a.dcc_signed,
            dcc_floor=a.dcc_floor,
            dcc_path_n=a.dcc_path_n,
            stop_on_stationary=a.stop_on_stationary,
            stat_tol=a.stat_tol,
            stat_patience=a.stat_patience,
            start_from=a.start_from,
            curv_ms_max=a.curv_ms_max,
        )
