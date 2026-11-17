# Port summary: piecewise-planar coil paper on js/pp-coils

**Branch** `js/pp-coils` (fork `singh-jaydeep/DESC`), worktree `DESC2-paper`, based on upstream/master `ef9e018ae`
(2026-11-16). **Source**: branch js/qa-poster / js/coil-auglag (worktree `DESC2-poster`), following
`coil_sandbox/paper_min/PORT_PLAN.md` and `PORT_CHECKLIST.md` there. The branch is not meant to be merged; any
upstream PRs are cut from it later. Done 2026-11-16/17.

Read next: `PORT_STATUS.md` (every ported piece, how it was checked, every departure from the plan),
`OPEN_ITEMS.md` (the experiments and method decisions that remain), `findings/8a_basins.md` (the end-to-end check).

## What is on the branch

**Library (`desc/`)**, each piece checked against the source (bit-identical where the code is the same) and
with tests (`tests/test_composite.py`, `test_arc_coils.py`, `test_distance_rows.py`, `test_piecewise_planar.py`):

- Solvers: `lsq-composite` (Levenberg-Marquardt with bounded rows kept exact as hinges; Gauss-Newton or secant
  second-order term) and `lsq-auglag-composite` (slack-free augmented Lagrangian over it, multipliers kept per row
  id; optional `initial_penalty_rows`). Row identities on `ObjectiveFunction`.
- Coils: `PiecewisePlanarArcCurve/Coil` (transverse arcs), `PolarPlanarArcCurve/Coil`, `SplineXYZ` with break
  indices; `FixCurveArcReference`; corner-aware grids (`coil_quad_grid`, `coil_source_grid`,
  `coil_curvature_grid`).
- Distances: `CoilSetDistanceRows`, `PlasmaCoilSetDistanceRows` (top-k node-pair rows with a gap that makes each
  a lower bound; chunked candidate scan: 6.06 GB -> 0.14 GB on Jorge; closest-first when slots overflow;
  plasma rows follow a moving boundary unless `eq_fixed`), `node_gap_estimate`.
- Constraints: `ArcFold`, `FixHingeHeight`, `HingeLevel`, `HingeMidHeightRelative`, `HingeTilt`,
  `PolarArcRadius`, `CoilCurrentBand`, `PointwiseBn`, `VacuumBoundaryBn`, `CoilArclengthResidual`.
- Fixes: `CoilCurvature` bounds `|curvature|` (the signed one jumps); coil length on the objective's own grid.

**Scripts (`paper/`)**: `stage2.py` (stage 2, every bound hard), `single_stage.py`, `free_boundary.py` with boundary
repair (`lib/bdry_repair`, DESC PR #2337's surface fit vendored), `near_axis/` (with pyQSC), `metrics/`, the
example drivers `jorge/`, `faces/`, `compare/`, `trace_lines.py`, the dense check `lib/dense_check.py`, and
`inputs/` (every start, near-axis state, equilibrium and bounds file, checksummed). Experimental:
`lib/experimental.py` (per-pair rows; the link guard).

## What changed on the way (decisions, with dates in PORT_STATUS)

- Kept the AL's inner default `{"hessian": "secant", "finish_steps": 5}` (every reference run used it).
- Plasma-coil distance: upstream's hard minimum by default, top-k rows where it stalls (Jorge planar control).
- Curvature bounded on 1000 nodes per arc: the |kappa| peaks are inside the arcs, and the poster's
  `LinearGrid(N=300)` missed them by up to 0.12%.
- Link guard restored as an experimental option after two 8a runs linked coils; whether it is part of the method
  is open.
- Not ported: the reworked `lsq-auglag`, penalties and the distance field, curve-mode and signed rows (except the
  experimental guard), `CoilMeanSquaredCurvature`, the demountability diagnostic.

## The end-to-end check, and what it found

Design F stage 2 (acceptance run 8a) runs to a KKT point on the ported code. The ported problem is identical to
the poster's apart from inactive rows, yet the runs end at B.n 9.3e-4 (port) and 6.0e-5 (poster). Both are KKT
points of the same problem; inactive rows change the solver's variable scaling, and with that the basin. The
straight path between the two optima runs unique coil 2 into its own stellarator mirror image on the symmetry line
(1.2 mm at the midpoint). For the paper: report stage-2 B.n as best-of-several with the spread, and decide on the
scaling (`OPEN_ITEMS.md` 2.1). The Jorge-specific pieces are checked at the objective level; the first Jorge
chain is their end-to-end run.

## How to run

From the repository root, one heavy job at a time on the laptop:

```bash
PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
    python paper/stage2.py --eq EQ --start COILS.h5 --bounds BOUNDS.json --out paper/runs/NAME [options]
```

Run outputs go to `paper/runs/` (ignored by git). Python env: `desc-env2` (numpy 2.4.x; interpax 0.3.14 makes
`test_compute_everything` fail on SplineXYZ values, on clean upstream too).
