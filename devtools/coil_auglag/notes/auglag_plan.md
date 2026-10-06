# Augmented Lagrangian for stage-2 coils: plan and task list

Written 2026-10-06 on branch `js/coil-auglag` (worktree `../DESC-coil-auglag`). Read
`../README.md` first, then `stage2_qh_planar_handoff.md` §4-5 (why Gauss-Newton crawls on
precise_QH and what fixes it). Line numbers below are at commit `cf85f0d1a`.

## Why

The c0-testing AL inner solves cannot reach optimality, so its outer loop misbehaves
(`tests/test_optimizer.py:984 test_auglag` now stops infeasible: "subproblems repeatedly
stalled", violation 2.2e-3). On precise_QH the penalty subproblem converges only with three
solver ingredients that today live in `devtools/coil_auglag/ctr.py`, outside DESC:

1. the **composite hinge model** (bound rows kept exact as `max(0, .)` of their
   linearized pre-hinge value; inner semismooth Newton `solve_comp`),
2. a **structured secant** estimate of the dropped Gauss-Newton term S (`compS`), which
   is dominated by the OBJECTIVE (QuadraticFlux S alone is 7.4x the GN curvature;
   c0-testing's `second_order="constraints"` omits it),
3. a few **exact-Hessian** steps to finish (`comp2`).

## Decisions taken (with the user)

- Inequalities: **slack-free hinge AL**. Row `sqrt(mu) * max(0, c - ub + y/mu)` (and the
  mirror for `lb`), `y >= 0`, update `y <- max(0, y + mu (c - ub))`. Equalities keep
  `sqrt(mu)(c - t) - y/sqrt(mu)`, no slacks. The hinge rows are exactly what the composite
  model solves, and their dropped term is the Lagrangian term `(mu c - y) Hess(c)`, so
  compS/comp2 cover objective and constraint curvature in one mechanism.
- Second order: compS in every inner solve; comp2 only to finish (final outer
  iteration, or an inner solve stalled above tolerance).
- Tolerances: Conn-Gould schedule (master), with the inner `gtolk` floored near what the
  inner solve reaches (~1e-6 scaled). Measure decrease row-paired (task A6).
- Termination: master's logic; on a genuine inner stall raise `mu` on violated rows only.
  No "repeatedly stalled" exit at an infeasible point.
- Placement: **new methods** `lsq-composite` (penalty) and `lsq-auglag-composite` (AL),
  registered beside the c0-testing `lsq-auglag`, which stays until the new one matches it.
- Distances: **node-node rows with an explicit gap bound** replace the exact-curve rows as
  default (task group B). The bound delta is computed once (with a backoff factor) at
  build; no delta updates in the outer loop unless the backoff proves too generous.
- Selected rows keep **slots with identity tracking**: a selected-row objective returns its
  K slot rows plus a static-size array of candidate ids (pair, i, j). The optimizer uses the
  ids to pair rows across iterates (A6) and to attach multipliers (B4). Considered and
  rejected: one fixed row per pair via log-sum-exp (the c0-testing objectives instead sum
  squared hinges or take a min per pair). It is conservative
  (`m_alpha <= min d <= m_alpha + log(n_eff)/alpha`) but its curvature varies on a 1/alpha
  scale: alpha ~2000/m (2-3 mm conservatism) bends on sub-mm scales, far below a useful
  step, so neither secant nor exact second order would resolve it.
- Done criterion (precise_QH from circles): feasible to `ctol` (judged by brute force,
  `check.py`), no linked pairs, scaled inner gradient <= 1e-6 at each outer update, final
  certificate (PD Hessian, small Newton decrement). Also precise_QA notebook case and
  `test_auglag`.

## Open decisions (flagged; ask the user)

| id | question | default if not asked |
|---|---|---|
| Q1 | Public accessor for per-row scaled bounds/targets on `ObjectiveFunction` (A3) vs reading `_scale`, `_normalize_target`, `normalization` privately as `ctr.py` does | add a public method |
| Q2 | ~~Row pairing when slot rows change identity~~ resolved: pair by candidate id among rows with nonzero residual (A6) | |
| Q3 | How the optimizer passes "keep these candidates selected" (rows with `y > 0`) to a selected-row objective (B4) | boolean candidate mask in `constants` |
| Q4 | Keep the exact-curve rows as `distance="curve"` for Fourier coils, or drop them | keep as an option |
| Q5 | Backoff factor on delta and default node count | 1.5x, nodes chosen so delta <= 2% of the bound |
| Q6 | Retire `second_order="constraints"` from `lsq_auglag` once C lands | keep, document as superseded |

## A. Composite least-squares solver in DESC (penalty form)

**A1. Module `desc/optimize/composite.py` with the model pieces from `ctr.py`.**
Move `resid`/`cost_of` (`ctr.py:73-83`), `solve_comp` (`:91`), `model_val` (`:131`) and
the per-row bound/target assembly (`:40-54`) into pure functions on arrays:
`composite_resid(s, lo, hi, tgt, isb)`, `solve_composite(a, B, rT, lam, isb, lo, hi, S=None)`
(semismooth Newton on the convex piecewise-quadratic model, Armijo backtracking, returns
step and iteration count), `composite_model(a, B, r, q, isb, lo, hi)`.
Why: one implementation for the penalty and AL methods. Check: unit test that
`solve_composite` with no active hinge equals the LM step `-(BᵀB + lam I)^-1 Bᵀr`, and on a
1-D hinge problem returns the analytic minimizer.

**A2. Loop `lsq_composite` in `desc/optimize/least_squares_composite.py`.**
Port `ctr.py:151-247`: Jacobian of pre-hinge scaled values, ratcheted column scaling,
Nielsen LM damping, acceptance on `rho > 0`, checkpoints through `callback`. Options:
`model={"gn","composite"}` (`gn` = what lsqtr sees, for comparison), `hessian={"gn",
"secant","exact"}`, `finish_steps=k` (switch to exact for the last k accepted steps once
`gtol` is reached with the secant), `jac_chunk_size`. Return an `OptimizeResult` shaped like
`lsqtr` (`least_squares.py:32`) so `Optimizer` can report it. Check: reproduces the E1 run
(README "Hands-off precise_QH run") to the same certificate within a factor of 2 in
iterations.

**A3. Register `lsq-composite`** in `desc/optimize/_desc_wrappers.py` following
`_optimize_desc_least_squares` (`:471-552`): `scalar=False, equality_constraints=False,
inequality_constraints=False` (bounds live in the objective), `hessian=False`. Needs the
per-row `(lo, hi, tgt, is_bound)` in scaled units (Q1). Check: `Optimizer("lsq-composite")`
runs `devtools/coil_auglag/setup_qh.py`'s problem from circles.

**A4. Secant S (`hessian="secant"`).** Port `ctr.py:144` and the update block: NL2SOL
sized rank-2 update with `y# = dg - G+ s` (row-order independent; needed because selected
rows change identity), skip when `yᵀs <= 0`, keep S in unscaled coordinates, per-iteration
switch between GN and augmented model by which predicted the last step better, damping floor
from `eig(BstdᵀBstd + Sd)`. Check: from G3o-like states the scaled gradient drops ~1e-2 ->
1e-6 in < 100 iterations (measured 77).

**A5. Exact S (`hessian="exact"`).** HVP per column (`ctr.py:58-66`), damping floor from
the full Hessian incl. active hinge rows (the fixed `comp2` floor). Check: from a compS
endpoint, quadratic decrease of the scaled gradient over 3-5 steps.

**A6. Row-paired decrease.** The acceptance ratio needs `cost(x) - cost(x_new)`; near
convergence that is 1e-10..1e-17 against a ~2e5 cost (ulp 3e-11), so the difference of two
totals is noise and lambda blows up (compS stopped near 1e-6, comp2 near 1e-9..1e-10).
Compute `sum((r - r_new)(r + r_new))/2` instead, so large equal parts cancel row by row.
Fixed-position rows pair by position. Selected-row objectives pair by candidate id: rows
entering or leaving the selection are inactive (zero residual) on both sides because
`select_distance` exceeds the bound, so only ids with a nonzero residual need pairing; any
unmatched remainder falls back to the plain difference. This removes the summation and
cancellation part of the floor, not rounding inside each residual (Biot-Savart sums); if
that dominates, accept on decrease of the gradient norm once the cost cannot resolve
progress. Check: measure the floor before and after; comp2 on QH reaches <= 1e-10
reliably (C1n got there by two lucky 3-ulp acceptances, E1 stopped at 7.7e-9).

## B. Node-node distance rows with an explicit gap bound

**B1. Rows.** Add `distance="node"` (default) to `CoilSetDistanceRows`
(`_coils.py:2959`) and `PlasmaCoilSetDistanceRows` (`:3200`): rows are node-node
(node-plasma point) distances `d_ij`, candidates `(pair, i, j)` within `select_distance`,
re-selected each evaluation into slots, fixed candidate ids. The per-pair signed row is the
pair's minimum node distance times the linking sign (kinked only at ties, and only active
near a crossing). Works for every coil class: only node positions are needed, so
`_fourier_curve_kind`/`_point_curve_distance` (`:2813-2950`) become the `distance="curve"`
path (Q4).

**B2. Gap bound delta.** At the true closest pair the connector is perpendicular to both
tangents, so the node estimate overshoots by at most
`delta = (h_a + h_b)^2 / (8 d) + (kappa_a h_a^2 + kappa_b h_b^2) / 8`, with `h` the largest
arclength between consecutive nodes and `kappa` the maximum curvature (curves C^2 between
nodes). Use `d = d_min` (feasibility is what is certified). Bounds become
`d_ij >= d_min + delta`. Compute `h`, `kappa` once at build from the initial coils on a
fine grid times the backoff (Q5); for Fourier coils certify between nodes with Bernstein
(`max|x'| <= max_nodes|x'| / (1 - N ds / 2)`). Plasma-coil: plasma grid spacing and surface
principal curvature in place of curve b's. Numbers for precise_QH (d_min 5.8 cm, kappa ~14,
L ~3.1 m): 129 nodes -> delta 0.7 cm (12%); 401 nodes -> 0.07 cm (1.2%). Check: property test
on random Fourier coils and arcs: `min d_ij - true distance <= delta` (true distance by
`_segment_segment_distance` on a 10^4-node polyline).

**B3. Piecewise classes.** Arcs (`PiecewisePlanarArc`, `PolarPlanarArc`): grid with nodes on
every hinge (node count a multiple of `B`) so each stretch between nodes is smooth; per-arc
`h`, `kappa`. Splines with breaks: nodes on the breaks; derivative bounds per piece from the
spline (Bernstein does not apply). Check: the B2 property test on both classes.

**B4. Multipliers on selected rows.** Return a static-size int array of candidate ids with
the slot rows (padding id -1), from the same `jnp.nonzero(mask, size=K)` that fills the
slots. Selection keeps
any candidate with a nonzero multiplier (Q3), so a row carrying force cannot drop out and
make the merit jump. Store multipliers sparsely (only `y > 0`). Check: an AL run where a
contact moves along the coil keeps `sum y` continuous across selection changes.

**B5. Tests** (compute-level, per the objective-test convention): bound property test, FD
Jacobian, NaN gradient, signed row on threaded rings (pattern of
`tests/test_objective_funs.py` `test_coil_set_distance_rows`), registration in the
`specials` lists.

## C. Slack-free hinge augmented Lagrangian

**C1. `lsq_auglag_composite`** (new module, e.g. `desc/optimize/aug_lagrangian_composite.py`).
Inequality rows as hinge rows with per-row shift `y/mu` and scale `sqrt(mu)` fed to the A
machinery (pre-hinge value `c`, bounds shifted); equalities as in `lsq_auglag`'s `lagfun`
(`aug_lagrangian_ls.py:404`). No `inequality_to_bounds` (`:390`), no `z2xs`.
**C2. Outer loop.** Conn-Gould updates (master), `gtolk` floored near 1e-6 scaled; `mu`
grows only on rows still violated (the hinge residual IS the violation, so the c0-testing
true-violation bookkeeping `:429-450` and `y_update_gate` are unnecessary); keep
`max_multiplier`/`max_penalty_parameter` as caps.
Keep the Jacobian handling of PR #2298 (already the effect of the vendored
`scale_tail_rows` rescale at `aug_lagrangian_ls.py:1449`): rescale the constraint rows by
`sqrt(mu_new/mu_old)` at an outer update instead of re-evaluating, and do the update right
after the accepted step that built J so no extra unscaled copy has to live across
iterations (the vendored code keeps one).
**C3. Stalls.** Master termination; an inner stall raises `mu` on violated rows and
restarts the inner solve; never return success at an infeasible point.
**C4. Second order.** Inner solves with `hessian="secant"`; `finish_steps` exact steps at
the last outer iteration.
**C5. Register `lsq-auglag-composite`** next to `lsq-auglag` (`_desc_wrappers.py:329-470`),
`equality_constraints=True, inequality_constraints=True`; constraints come from the
`constraints=` objectives with their per-row bounds (`optimizer.py:579`).
**C6. Tests and benchmarks.** `test_auglag` (`tests/test_optimizer.py:984`) passes with the
new method; a small test with active inequalities; precise_QA notebook problem; precise_QH
from circles to the done criterion above, with the coil-coil, plasma-coil, length and
curvature limits as constraints and QuadraticFlux as the objective.

## D. Remaining items

- **D1.** c0-testing AL issues (README "Known issues"): `fmin_auglag` stall counter reset,
  `step_veto` loop to `maxiter`, `_trust_region_step_eigh` (`aug_lagrangian_ls.py:35`) on
  singular PSD models. Fix where `lsq-auglag`/`fmin-auglag` stay in use, or document as
  superseded; decide `test_auglag` for the old `lsq-auglag` once C passes it.
- **D2.** ~~Planar coils NaN for normals within ~1e-8 of +z~~ fixed by folding in PR #2277
  (`rotate_vector_to_vector`); `CoilSetDistanceRows`' evaluator uses it too.
- **D3.** Q6 (`second_order="constraints"`).
- **D5.** `desc.objectives.utils.softmin` (used by `CoilSetMinDistance(use_softmin=True)`)
  is a Boltzmann-weighted average, which is >= the true minimum, so a lower bound on it is
  not conservative. Raise separately (generic, present on master).
- **D4.** API docs for the new methods and options. The user writes CHANGELOG entries.

## Order and constraints

A1 -> A2 -> A3 -> A4 -> A5 -> A6, then B1-B5 (independent of A, but B4 is needed by C),
then C1-C6, D last. One heavy job at a time on the 8 GB machine; keep `jac_chunk_size` set;
run benchmarks from `devtools/coil_auglag/` with outputs left untracked.
