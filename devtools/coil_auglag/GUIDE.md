# Stage-2 coils with smooth distance rows and the augmented Lagrangian: a guide

How to set up and run a stage-2 coil optimization with the objectives and solvers added
on `js/coil-auglag`, using precise_QH with planar coils as the worked example (run AL8
in `notes/auglag_plan.md`, converged from circles). Everything in the example uses the
`desc` API only; the scripts in this directory are our versions of the same steps.

## 1. Where things are

| what | where |
|---|---|
| composite least-squares solver (penalty form) | `desc/optimize/least_squares_composite.py`, method `"lsq-composite"` |
| hinge augmented Lagrangian | `desc/optimize/aug_lagrangian_composite.py`, method `"lsq-auglag-composite"` |
| composite hinge model | `desc/optimize/composite.py` |
| distance rows | `desc/objectives/_coils.py`: `CoilSetDistanceRows`, `PlasmaCoilSetDistanceRows` |
| per-node arclength | `desc/objectives/_coils.py`: `CoilArclengthResidual` |
| row ids, kept rows, scaled bounds | `ObjectiveFunction.row_ids`, `keep_rows`, `scaled_bounds` |
| tests | `tests/test_optimizer.py` (`TestComposite`, `TestAugLagComposite`), `tests/test_objective_funs.py` (`*distance_rows*`) |
| precise_QH / precise_QA builders | `devtools/coil_auglag/setup_qh.py`, `setup.py` |
| drivers with logs | `run_al.py` (AL), `run_composite.py` (penalty) |
| brute-force feasibility and linking | `check.py` (`feasibility()`) |
| optimality certificate | `diag.py TAG X --hess` |
| FourierXYZ phase gauge | `common.py` (`phase_tangent`, `load_phase`) |
| history and decisions | `notes/auglag_plan.md` (status, every run), `notes/stage2_*_handoff.md` |

Not yet in `desc/` but general-purpose: the feasibility check, the phase gauge fix and the
certificate. Everything else in this directory is problem-specific or investigation
tooling (`ctr.py`, the prototype `lsq-composite` replaced, `sdecomp.py`, `hspec.py`,
`trsteps.py`, `pintest.py`, `lk.py`).

## 2. The objectives

### Distance rows

`CoilSetDistanceRows` (coil-coil) and `PlasmaCoilSetDistanceRows` (plasma-coil) replace
`CoilSetMinDistance` / `PlasmaCoilSetMinDistance`. Instead of one minimum per coil, which
is kinked wherever the closest pair of points changes, they return one smooth row per close
contact, so the solver sees every contact and its own derivative.

How a row set is built at each evaluation:

1. **Pairs.** One coil pair per symmetry orbit, independent coil first (72 on precise_QH:
   4 independent coils, 32 physical).
2. **Candidates.** Every distance that could matter, with a permanent id:
   - `distance="curve"`: node i of one coil to the *exact* other curve (Newton from the
     nearest polyline point), both directions ("side 0": a to b, "side 1": b to a); id
     `(side, pair, node)`. Fourier coils only.
   - `distance="node"` (default): node i of coil a to node j of coil b, minus a gap (below);
     id `(pair, i, j)`. Any coil class.
3. **Selection.** A cheap pass without derivatives keeps candidates closer than
   `select_distance`.
4. **Slots.** JAX needs fixed sizes, so the selected rows fill `max_active_rows` slots in id
   order; unused slots read `select_distance` (inactive) with id -1. A warning names the
   objective when slots overflow. `active_row_count(params)` gives the count at a state.
5. **Pair rows.** The last `num_pairs` rows hold each pair's minimum distance, negated if
   the pair is linked. Their lower bound is 0 whatever `bounds` says: they only guard
   topology, inactive for an unlinked pair and violated (pulling the coils back through)
   for a linked one. The distance limit itself is enforced by the slots, which hold every
   contact.

Plasma-coil rows work the same way with plasma surface points in place of the first coil
(no sides, no pair rows).

Rows change identity between evaluations (slot 17 holds different candidates over a run),
so anything attached to a row is keyed by id: the augmented Lagrangian keeps multipliers
and penalties per id, and `keep_rows` keeps every candidate that carries a multiplier
selected even if it moves past `select_distance`.

**The gap (node mode).** Node-node distances overestimate the true curve distance by at
most `gap = backoff * ((h_a + h_b)^2 / (8 d) + (k_a h_a^2 + k_b h_b^2) / 8)` (h: largest
arclength between nodes, k: largest curvature, d: the lower bound). Rows report `d_ij - gap`,
so every row is a lower bound on the true distance. It is computed once at build from the
coils at that moment, with `gap_backoff=1.5`, or set with `gap=`. Built on circles it
uses their curvature (about 1/R), not the optimum's (about 14 on precise_QH): pass a
larger backoff or an explicit gap when starting from circles. Curve mode has no gap; it
misses up to about the same amount between nodes (0.45 mm at 257 nodes, 0.11 mm at 513 on
precise_QH).

### Other engineering rows

- `CoilLength(bounds=(0, Lmax))`, `CoilCurvature(bounds=(-inf, kmax))`: per-node rows. The
  default `CoilCurvature(signed=False)` bounds |curvature|.
- `CoilArclengthResidual` (3D coils only): one row per node, `(|x_s|_i - mean) / sqrt(N)`.
  Use it rather than `CoilArclengthVariance`, whose single squared residual makes the cost
  quartic and invisible to Gauss-Newton.

## 3. Per-representation setup (gauges)

| representation | fix | why |
|---|---|---|
| all | `FixSumCoilCurrent` | fixes the overall field scale |
| all Fourier and planar | `rotmat`, `shift` fixed (DESC does this for coil sets) | duplicate `center`/`normal` or the Fourier coefficients |
| `FourierPlanarCoil` | pin the largest `normal` component per coil (`FixParameters`) | scaling `normal` leaves the coil unchanged |
| `FourierPlanarCoil` | nothing for tangential freedom | the curve parameter is the polar angle in the plane: no reparameterization exists |
| `FourierXYZCoil` | `CoilArclengthResidual` | arbitrary reparameterization s -> phi(s) |
| `FourierXYZCoil` | pin one n=+-1 coefficient per coil (`common.load_phase`) | the phase s -> s + c survives any arclength term |
| `SplineXYZCoil` | probably the same as FourierXYZ | untested |

Symptom of a gauge left free: cost flat to 6-7 digits, steps accepted, raw gradient
bouncing instead of decreasing.

## 4. Choosing resolutions

Measured on precise_QH (83 free parameters, AL8 state); one inner iteration costs about one
Jacobian:

| objective | rows | resolution | Jacobian |
|---|---|---|---|
| QuadraticFlux | 1326 | plasma grid `M=N=25`, field grid 101 nodes/coil | 2.2 s |
| coil-coil rows (curve) | 3072 slots, ~550 active | 257 nodes/coil | 0.27 s |
| plasma-coil rows (curve) | 6000 slots, ~670 active | 257 nodes/coil | 0.49 s |
| curvature | 404 | 101 nodes/coil | 11 ms |

Rules of thumb:

- **QuadraticFlux sets the cost.** Its Jacobian scales with plasma points x field nodes,
  so keep its grids where field accuracy needs them and refine everything else on separate
  grids.
- **Curvature:** 101 nodes per coil missed peaks (fine-grid max 14.7 vs bound 13.7); 401
  nodes fixed it for about 40 ms per Jacobian. Give it its own grid.
- **Distance nodes:** the between-node miss (curve) or gap (node) scales as spacing^2;
  doubling nodes quarters it at about twice the row cost.
- **`select_distance`** must exceed the bound plus the gap with room for a step (we used
  2x the bound). Too small and unused slots read as violated (a warning says so).
- **`max_active_rows`:** size from `active_row_count` with margin (1.5x); rows grow as
  (contact length / spacing)^2 for node rows.
- **Plasma grid for node rows:** its spacing alone sets a floor on the plasma-coil gap
  (6.4 cm on an 11 cm bound at `M=N=25`); use curve mode for plasma-coil or a finer grid.
- **Tolerances:** QuadraticFlux evaluation noise limits the measurable decrease to a scaled
  gradient of a few 1e-8 on precise_QH; set `gtol` above that (1e-6..1e-7).

Peak memory for the whole AL8 problem was about 2 GB.

## 5. Worked example: precise_QH, planar coils, augmented Lagrangian

QuadraticFlux is the objective; every engineering limit is a constraint with its bound.
Weights on constraints only scale their rows, so leave them at 1.

```python
import numpy as np

import desc.examples
from desc.coils import initialize_modular_coils
from desc.grid import LinearGrid
from desc.objectives import (
    CoilCurvature,
    CoilLength,
    CoilSetDistanceRows,
    FixParameters,
    FixSumCoilCurrent,
    ObjectiveFunction,
    PlasmaCoilSetDistanceRows,
    QuadraticFlux,
)
from desc.optimize import Optimizer

eq = desc.examples.get("precise_QH")
coils = initialize_modular_coils(eq, num_coils=4, r_over_a=2.5)
for c in coils:
    c.change_resolution(N=7)

plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
field_grid = LinearGrid(N=50)  # Biot-Savart quadrature, 101 nodes per coil
row_grid = LinearGrid(N=256)  # distance-row nodes, 513 per coil

objective = ObjectiveFunction(
    QuadraticFlux(
        eq,
        field=coils,
        eval_grid=plasma_grid,
        field_grid=field_grid,
        vacuum=True,
        bs_chunk_size=10,
        jac_chunk_size=16,
    ),
    deriv_mode="blocked",
)
constraints = (
    FixSumCoilCurrent(coils),
    # planar gauge: pin the largest normal component of each coil
    FixParameters(coils, [{"normal": [np.argmax(np.abs(c.normal))]} for c in coils]),
    CoilSetDistanceRows(
        coils,
        select_distance=0.116,
        bounds=(0.058, np.inf),
        grid=row_grid,
        max_active_rows=3000,
        distance="curve",
        jac_chunk_size=16,
    ),
    PlasmaCoilSetDistanceRows(
        eq,
        coils,
        select_distance=0.16,
        bounds=(0.11, np.inf),
        plasma_grid=plasma_grid,
        coil_grid=row_grid,
        max_active_rows=6000,
        distance="curve",
        jac_chunk_size=16,
    ),
    CoilCurvature(coils, bounds=(-np.inf, 13.7), grid=LinearGrid(N=200)),
    CoilLength(coils, bounds=(0, 2.92), grid=field_grid),
)

(coils_opt,), result = Optimizer("lsq-auglag-composite").optimize(
    coils,
    objective=objective,
    constraints=constraints,
    ftol=0,
    xtol=0,
    gtol=1e-6,  # scaled gradient of the subproblem merit
    ctol=1e-6,  # max scaled constraint violation
    maxiter=800,  # inner iterations in total
    verbose=1,
    copy=True,
)
```

On precise_QH this converges from circles in 10 outer / 352 inner iterations (AL8):
QuadraticFlux cost 8.0733, all limits met on a fine grid except about 0.11 mm of coil-coil
clearance between rows (section 2), no linked coils.

Useful options (`options={...}` to `optimize`):

- inner solve (passed to `lsq_composite`): `hessian` (`"secant"` default here),
  `finish_steps` (exact-Hessian steps when an inner solve stalls, default 5),
  `track_steps` (log every attempted step in `step_log`).
- outer loop: `initial_penalty_parameter` (10), `tau` (10), `max_stalls` (3),
  `outer_callback` (called after each outer iteration with its log entry and the full
  state `x`; `run_al.py` uses it to checkpoint and print the per-group table).
- `callback`: called with the full state after every accepted step.

The result holds `y` and `penalty_param` (per constraint row at the final state, keyed by
`con_ids`), `constr_violation`, and `outer_history`.

### Penalty form instead

Put the limits in the objective (with weights) and use `"lsq-composite"`; it keeps their
hinges exact in the model. `options={"hessian": "secant", "finish_steps": 8}` reproduces
the E1 run from the earlier session (`run_composite.py`).

## 6. Reading a run

`run_al.py` prints, per outer iteration: inner iterations, the inner gradient against its
tolerance (and whether it was met), the max violation against its tolerance, and per
constraint group mu (min/median/max), max |y|, rows with a multiplier and max violation,
then a brute-force check (closest distances on fine polylines, lengths, curvature on a fine
grid, linked pairs).

What healthy looks like (AL7/AL8): every inner solve meets its tolerance, the violation
falls 3-10x per outer iteration once mu has settled, and the objective changes
monotonically with its relative change shrinking geometrically.

What went wrong before, and what it looked like:

| symptom | cause | fix |
|---|---|---|
| inner solve stalls; every trial step, however small, raises the merit by the same amount | a discontinuity in some row: curve mode used to exclude each pair's argmin node from the slots, then the pair row jumped at polyline ties | both fixed; find any new one by perturbing the stall state by 1e-11 and comparing rows (`notes/auglag_plan.md`, AL4/AL5) |
| inner solve stalls near a kinked row | `CoilSetMinDistance`-style minima in active rows | use rows, never min objectives |
| violation bounces around 1e-3, tolerances reset every other outer iteration | mean mu taken over all rows | now taken over active rows; new ids start at that level |
| exact steps rejected at a scaled gradient of a few 1e-8 | QuadraticFlux evaluation noise | set `gtol` above it |

## 7. Checking a result

1. Feasibility on fine grids, independent of the rows: `python check.py KW X_x.npy`.
2. Linking: `CoilSetDistanceRows.linked_pairs(params)` (in the check above).
3. Certificate: `python diag.py TAG X_x.npy --hess` (positive definite Hessian, small Newton
   decrement); written for the penalty problem; the Lagrangian version is still to do.

## 8. Open

- Node rows from circles: gap from the circles' curvature (backoff), plasma grid floor.
- Move the feasibility check, phase gauge fix and certificate into `desc/`.
- Run the precise_QA (FourierXYZ) problem through the AL.
- `ProximalProjection` with row ids is untested.
