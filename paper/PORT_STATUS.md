# Port status: piecewise-planar coil paper -> js/pp-coils

Branch `js/pp-coils`, based on upstream/master `ef9e018ae`. Ported from js/qa-poster (worktree
`DESC2-poster`, `desc/` = js/coil-auglag) following `coil_sandbox/paper_min/PORT_PLAN.md` (the reasons)
and `PORT_CHECKLIST.md` (the order) there. This file records what was ported, how it was checked,
and every place the port departs from the plan. No upstream PRs: all fixes are commits on this branch.

## Library (desc/)

| step | what | check |
|---|---|---|
| 1b | `CoilCurvature` bounds `\|curvature\|` (new compute quantity; signed `curvature` built on it) | peanut coil: signed curvature changes sign, objective is its magnitude; finite Jacobian |
| 1c | `_CoilObjective` evaluates length on its own grid (`override_grid=False`) | 3-node GL grid: 13.712 (its own) not 13.318 (the default grid's) |
| 2a | `ObjectiveFunction.row_ids`, `keep_rows`, `scaled_bounds` (+ constraint wrappers) | default ids, bounds/targets split, through `LinearConstraintProjection` |
| 2b-d | `composite.py`, `lsq_composite`, `lsq_auglag_composite` | closed-form minimizers with active hinges (gn, secant); AL KKT point and multipliers; rows re-sorted every evaluation keep their multipliers by id |
| 2e-f | `lsq-composite`, `lsq-auglag-composite` registered | through `Optimizer`; upstream all-optimizers test |
| 3 | `PiecewisePlanarArcCurve/Coil`, `PolarPlanarArcCurve/Coil`, `SplineXYZ` break indices, `FixCurveArcReference` | branch values on F/Jorge starts identical; FD derivatives, C0 hinges, planar arcs, save/load, arc_ref fixed |
| 4 | `CoilSetDistanceRows`, `PlasmaCoilSetDistanceRows`, `node_gap_estimate` | bit-identical to the branch (values, ids, kept rows, Jacobians) on design F, chunked and not |
| 5 | `ArcFold`, `FixHingeHeight`, `HingeLevel`, `HingeMidHeightRelative`, `HingeTilt`, `PolarArcRadius`, `CoilCurrentBand`, `PointwiseBn`, `VacuumBoundaryBn` | bit-identical to the sandbox classes (values, Jacobians) on Jorge and design F |
| 6 | `desc.coils.coil_quad_grid`, `coil_source_grid`, `composite_gauss_legendre_grid`, `coil_curvature_grid` | identical to sandbox grids; GL lengths vs adaptive quadrature |

Tests: `tests/test_composite.py`, `test_arc_coils.py`, `test_distance_rows.py`, `test_piecewise_planar.py`,
additions in `test_objective_funs.py`, `test_compute_everything.py`. The touched upstream suites pass;
`test_compute_everything` fails only on `SplineXYZ` values that fail identically on clean upstream with
interpax 0.3.14 (environment). The full suite has not been run on this branch yet.

## Scripts (paper/)

| stage (PORT_PLAN 9) | script | status |
|---|---|---|
| stage 2 (fixed eq, hard bounds) | `stage2.py`, `lib/dense_check.py` | ported; build-only check on design F reproduces the start metrics |
| 4 single stage | `single_stage.py` | see below |
| 3 free boundary, 3b repair | `free_boundary.py`, `lib/bdry_repair.py`, `lib/surface_fit.py` | see below |
| Jorge chain (9.4) | `jorge/` | see below |
| 1-2 near-axis | `near_axis/` | see below |
| 6 metrics | `metrics/` | see below |
| faces sweep | `faces/` | see below |

## Departures from the plan

1. **1a dropped**: `CoilSetLinkingNumber` is correct upstream (the writhe bug came from the branch's own
   `indices=` change). **1d dropped**: the 1e-14 floor is in `compute_hess_scale`, already upstream;
   `compute_jac_scale` was never changed.
2. **Inner solver default kept** at `{"hessian": "secant", "finish_steps": 5}` (what every reference run
   used; PORT_PLAN said gn). `finish_steps` needs the exact second-order term, so that code stays as the
   finishing mode; `"gn"` and `"secant"` are the user choices.
3. **`lsq-auglag-composite` runs without nonlinear constraints** (empty block), as upstream `lsq-auglag`.
4. **Arc kernels take their inputs explicitly** (`hinges, tilts, shape, arc_ref, s`), so each compute
   function reads what it declares; six functions used `arc_ref` without declaring it. Values unchanged.
5. **Distance-row candidate scan is chunked** (`scan_block_size`, kept rows as a sorted id array):
   compiled temporaries 6.06 GB -> 0.14 GB on Jorge (12 coils x 1026 nodes). Results bit-identical.
6. **`PlasmaCoilSetDistanceRows(eq_fixed=...)`** like upstream `PlasmaCoilSetMinDistance`: the boundary
   moves with `eq` unless `eq_fixed=True` (stage 2).
7. **Plasma-row gap needs a fine plasma grid**: the gap grows as (cell size)^2 x surface curvature (design
   F's boundary reaches 51 /m): 47 mm at 32 x 32, 5 mm at 128 x 128. `stage2.py` defaults to a 128 x 128
   half grid at select 1.2 d_pc (6k active rows on F's final coils). No poster run used these rows.
8. **Curvature**: the |kappa| peaks of the paper's arc coils are inside the arcs, not at the hinges, so a
   per-arc grid alone does not remove the poster's undershoot (up to 0.12% at LinearGrid(N=300)); node
   density does. `coil_curvature_grid` with 1000 nodes per arc (< 0.01%) is the driver default.
9. **`select_distance` limitation** (paper methods): a row exists only inside the selection band, so a step
   longer than the band can jump coils through each other unseen; unsigned rows cannot separate crossed
   coils. Starts must be feasible and the band wider than one step (`stage2.py`: 2.5 d_cc coil-coil,
   1.2 d_pc plasma-coil).
10. **`CoilArclengthResidual` ported** (decision 2026-11-16) for the free FourierXYZ arm of the
    planar/hinged/free comparison; `CoilMeanSquaredCurvature` not ported.

## Not done

- Acceptance runs (checklist step 8): 8a design F solver row, 8b Jorge chain, 8c planar control, 8d
  memory, 8e field lines. Run with the scripts above once the GPU is free.
- Full test suite.
