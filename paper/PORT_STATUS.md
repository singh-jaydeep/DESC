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

| stage (PORT_PLAN 9) | script | checked (no long runs) |
|---|---|---|
| 1-2 near-axis, to DESC | `near_axis/` (README) | test_nae identical to poster; B's state continues at gtol; to_desc on B |
| 3 free boundary, 3b repair | `free_boundary.py`, `lib/bdry_repair/` (PR #2337 fit vendored) | design F scores reproduced; repair gate |
| 4 single stage | `single_stage.py` | `--dry` on F; 2 iterations at basis 4 with repair |
| 5 stage-2 polish | `stage2.py`, `lib/dense_check.py` | build on F, Jorge (arc, planar), faces (arc, FourierXYZ) |
| 6 metrics | `metrics/make_metrics.py` | design F metrics reproduced (eq, coils, md) |
| Jorge chain (9.4) | `jorge/chain.sh`, `trace_lines.py`, `jorge/zoom_poincare.py` | builds; tracer on C06 |
| faces sweep | `faces/run_sweep.sh`, `faces/score.py`, `faces/table.py` | score + table on a start |
| planar/hinged/free | `compare/run.sh` | builds of each arm |

What needs runs or a decision: `OPEN_ITEMS.md`.

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

11. **`stage2.py` select distances are relative** (`--cc-sel 2.5` x d_cc, `--pc-sel 1.2` x d_pc) and the plasma
    grid defaults to 128 x 128 (Jorge: 128 x 512); FourierXYZ coils get the phase pin and speed bounds.
12. **Single stage drops the backoffs** and is weighted (as the poster); stage 2 restores hard feasibility.
13. **`coil_curvature_grid` needed quadrature spacing** to be usable by `CoilCurvature` (fixed, tested).
