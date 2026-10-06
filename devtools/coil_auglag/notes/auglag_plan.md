# Augmented Lagrangian for stage-2 coils: plan and task list

Branch `js/coil-auglag` (worktree `/Users/singh/PycharmProjects/DESC-coil-auglag`), written
2026-10-06; line numbers are at commit `144433233`. Read `../README.md` first, then
`stage2_qh_planar_handoff.md` §4-5 (why Gauss-Newton crawls on precise_QH and what fixed
it). The user prefers a task list to follow: each task says where, what, why, and how to
check it.

## Status (2026-10-06, end of first implementation session)

Q1, Q3-Q6 taken at their defaults. Done, with tests (`tests/test_optimizer.py`
`TestComposite`, `TestAugLagComposite`; `tests/test_objective_funs.py` `*_distance_rows*`):

- A1 `desc/optimize/composite.py`; matches `ctr.py`'s `solve_comp` to 1e-15.
- A2-A5 `lsq_composite` (`least_squares_composite.py`), `lsq-composite` registered, Q1 as
  `ObjectiveFunction.scaled_bounds()`. Exact S is built from `fun` by autodiff
  (`sum rho_i Hess s_i`; agrees with `ctr.py`'s `H - J^T J` to 3e-14).
- A6 paired actual and predicted decrease, by row id. Floor is evaluation noise, see runs.
- B1-B5 `distance="node"` default (exact-curve rows kept as `"curve"`), gap per pair,
  default node grid for 2% gap, nodes on arc hinges and spline knots, candidate ids
  (`row_ids`, `compute_row_ids`), keep mask (`keep_rows`).
- C1-C6 `lsq_auglag_composite` (`aug_lagrangian_composite.py`), `lsq-auglag-composite`
  registered. Multipliers and penalties per row id; shift and sqrt(mu) applied through
  `lsq_composite(rows=...)`, so the Jacobian does not depend on mu (C2).
- C7 `setup_qh.build(al=True, distance="node", pair_N=None)`, `run_al.py`,
  `check.feasibility()`; `run_composite.py` for the penalty method.
- D3 documented, D5 API entries added.

Runs (untracked, `devtools/coil_auglag/`):

| tag | what | result |
|---|---|---|
| LC1 | A2 check: `lsq-composite` from circles, secant + 8 exact, totals decrease | secant 192 its -> 1.8e-6, exact -> 3.2e-8, 21 min; same optimum as E1n (distances, lengths, curvature), PD Hessian, Newton decrement 3e-12 |
| LC2 | exact steps from `E1o_x.npy`, paired actual decrease only | stalls 3.5e-7 |
| LC3 | as LC2, paired predicted decrease too | 2.6e-8, then damping blow-up |
| LC4 | LC3 with `track_steps` | from ~5e-8 every trial step measures an increase of 1e-10..4e-10 whatever its size |
| AL1 | `run_al.py` from circles, node rows for both distances | stopped: plasma gap 6.4 cm makes unused slots (0.16 m) read as violated |
| AL2 | as AL1 with `pc="hard"` (plasma minimum distance) | both outer iterations' inner solves stalled (1.2e-2, 2.2e-2 vs gtolk 1e-3), so no multiplier update; stopped. Suspect the kinked plasma minimum; not diagnosed |
| AL3/AL4 | AL, exact-curve rows both distances (E1 setup) | outer 3 stalls: merit jumps 6.4e-4 under 1e-11 steps. Cause: curve mode excluded each pair's closest side-0 node (argmin) from the slots; a tie swapped slot membership and the multipliers with it. Fixed: closest node stays in the slots |
| AL5 | after that fix | stalls at outers 2, 4-6, jumps 1.2e-4 / 7.6e-6. Cause: the per-pair row took the exact distance of the polyline-argmin node, which jumps at polyline ties, and as an unlinked duplicate of the closest contact it was violated. Fixed: per-pair rows bounded below by 0 (topology only), value = min of the pair's exact side-0 slot rows |
| AL6 | after both fixes, maxiter 800 | no stalls in 12 of 13 outer iterations; ends on maxiter. cc 5.75 cm, pc 11.00 cm, length 2.920, nothing linked, max viol ~1e-3; QuadraticFlux cost plateaus at 8.07 (+-2e-4 rel per outer, sign alternating with the violation) vs 4.10 at the infeasible penalty optimum E1 |
| AL7 | AL6 with the schedule fix: mean mu over active rows (nonzero y or violated), unseen ids start at that level | **converged**: 9 outer / 255 inner its, scaled gradient 9.8e-7, violation 2.4e-7; QuadraticFlux 8.0815079, relative change per outer 4e-3 -> 1e-7; pc 11.000 cm, length 2.9200, nothing linked; cc 5.755 cm by brute force (curve rows miss ~0.45 mm between nodes) |
| AL8 | AL7 with finer grids: `pair_N=256` (513 nodes/coil), `curv_N=200` (401 nodes/coil) | **converged**: 10 outer / 352 inner its, gradient 7.5e-7, violation 3.4e-7; QuadraticFlux 8.0733402. Brute force: curvature 13.700 (bound 13.7), length 2.9200, pc 11.000 cm, cc 5.789 cm (0.11 mm short, between-node miss), nothing linked |
| AL9 | AL8 with node-node coil-coil rows (E1): explicit gap 2.102 mm (kmax, AL8 spacing), `cc_sel=0.1`, ~5.7k active rows | **converged**: 10 outer / 402 inner its, gradient 8.8e-7, violation 1.1e-7, every inner solve met its tolerance; QuadraticFlux 8.127 (+0.7% vs AL8). Brute force: cc 5.988 cm (1.9 mm above the bound: the gap is conservative, no miss), pc 11.000 cm, length 2.9200, curvature 13.700, nothing linked |
| AL10 | AL9 with the E2 gap from the bounds: `node_gap_estimate(513, 2.92, 13.7, 0.058)` = 2.342 mm, from circles | **converged**: 9 outer / 404 inner its, gradient 5.3e-7, violation 7.5e-7, every inner solve met its tolerance; QuadraticFlux 8.133. Brute force: cc 6.012 cm, pc 11.000 cm, length 2.9200, curvature 13.700, nothing linked; gap used 2.342 mm vs 2.097 mm needed at the final state (no flag) |
| AL11 | AL10 with `PlasmaCoilDistanceField` for plasma-coil (E3): 10 mm field, `node_margin` 0.554 mm from the bounds (h = 2.0 Lmax/513, kmax, 1/d_min) | **converged**: 13 outer / 491 inner its, gradient 7.9e-7, violation 2.2e-7, every inner solve met its tolerance; QuadraticFlux 8.247 (+1.4% vs AL10). Against the EXACT boundary: pc 11.055 cm, cc 6.013 cm, length 2.9200, curvature 13.700, nothing linked. Field build ~1.5-3 min, spline error 1e-6 m |

Correction (2026-10-06): `check.py` used to measure plasma-coil distance to the
objective's own plasma grid points (M=N=25), so AL8-AL10's "pc 11.000 cm" held only at
those points. Against the exact boundary (Newton projection, now what `check.py` does)
AL8 and AL10 are at 10.875 cm, 1.25 mm inside the bound. AL11's field rows fix this.

Noise floor: each QuadraticFlux (and length) residual carries 1e-13..1.6e-12 of rounding
from its own evaluation (up to ~200 ulp), about 2e-10 in any measured decrease, so the
ratio test fails below a scaled gradient of a few 1e-8. Decision (user): set tolerances
above it (gtol ~1e-7 on precise_QH), no machinery to beat it. The A6 target of 1e-10 is
dropped.

B6 on precise_QH (AL builder, node rows, built on circles): coil-coil 231 nodes, gap
1.0 mm (1.8%), ~2000 active rows at circles and at LC1; plasma-coil 65 nodes, gap 6.4 cm
on an 11 cm bound, ~3400-4200 active rows; one constraint Jacobian 0.1 s, peak 2.4 GB.

Done (AL7): fixed the outer update schedule. What AL6 showed:
In AL6 ctolk keeps resetting to ~7e-3 (`eta / mean(mu)^alpha_eta`, Conn-Gould via master)
because mean(mu) is over all constraint rows, almost all never active and at mu0 = 10, so
the violation hovers near 1e-3 and the subproblems alternate tight/loose. Contacts also
slide to new candidate ids, which start at mu0 and y = 0. Ideas: mean over active /
violated rows, or per-group; new ids start at their group's current mu.

Also seen: the curvature constraint on the 50-point grid misses peaks (fine-grid max
14.3-14.7 vs bound 13.7, in E1 as well).

Remaining for the done criterion: the last 0.11 mm of brute-force coil-coil clearance at
AL8 (curve rows have no gap: more nodes, node rows, or a margin on the bound), and the
final certificate (Hessian of the Lagrangian at AL8). Curvature is fixed by `curv_N=200`.

Open (for the user):
- Gap from circles: built on circles the coil-coil gap uses their curvature (~1/R); at
  the optimum curvature is ~14, where 231 nodes need ~3x the gap. Options: build the gap
  from the curvature bound and length bound, or rebuild after a first solve.
- Plasma-coil gap: the M=N=25 plasma grid alone gives ~1.6 cm of gap (x1.5 backoff);
  node rows need a finer plasma grid (cost grows with points x nodes) or the gap rule
  revisited.
- `ProximalProjection` inherits `ObjectiveFunction.row_ids`, untested there.
- D1, D4 untouched.

## Why

The c0-testing AL (now on this branch) cannot get its inner solves to optimality, and its
outer loop misbehaves around that: `tests/test_optimizer.py:984 test_auglag` stops
infeasible ("subproblems repeatedly stalled", violation 2.2e-3) where master converges. On
precise_QH the penalty subproblem converges only with three solver pieces that live in
`devtools/coil_auglag/ctr.py`, outside DESC:

1. the **composite hinge model**: bound rows kept exact as `max(0, .)` of their linearized
   pre-hinge value, solved by semismooth Newton (`solve_comp`);
2. a **structured secant** estimate of the dropped Gauss-Newton term S (`compS`). S is
   dominated by the OBJECTIVE (QuadraticFlux's alone is 7.4x the GN curvature), which
   c0-testing's `second_order="constraints"` leaves out;
3. a few **exact-Hessian** steps to finish (`comp2`).

## Where things are

| what | where |
|---|---|
| distance objectives (exact-curve rows, slots, signed rows) | `desc/objectives/_coils.py`: `CoilSetDistanceRows` :2951, `PlasmaCoilSetDistanceRows` :3192, curve evaluator `_fourier_curve_kind` :2811 to `_polyline_seed` :2928 |
| c0-testing AL | `desc/optimize/aug_lagrangian_ls.py` (`lsq_auglag` :130), wrapper `desc/optimize/_desc_wrappers.py:329-470` |
| stock trust region | `desc/optimize/least_squares.py:32` (`lsqtr`), wrapper `_desc_wrappers.py:471-552` |
| solver prototype (reference until A2 reproduces it) | `devtools/coil_auglag/ctr.py`: bound assembly :40-54, `resid`/`cost_of` :73-83, `solve_comp` :91, `model_val` :131, compS :144, loop :151-247 |
| problem builders, run tags, checks | `devtools/coil_auglag/`: `setup_qh.py`, `setup.py`, `mkstage.py`, `check.py` (brute-force feasibility and linking), `diag.py --hess` (certificate), `sdecomp.py` |
| reference states and logs | main checkout `/Users/singh/PycharmProjects/DESC/scratchpad_crawl/` (untracked): `QH_E1.h5`, `E1o_*`, `E1n_*` (penalty optimum, scaled gradient 7.7e-9, pair (2,3) linked), `G3o_*`, `H1_*`, `C1*` |
| tests touching this work | `tests/test_objective_funs.py` (`test_coil_set_distance_rows` :1280, curve evaluator, linking, arclength residual), `tests/test_optimizer.py` (`test_auglag` :984, `test_lsq_auglag_second_order` :1096), `tests/test_curves.py` |

## Decisions taken (with the user)

- **Inequalities: slack-free hinge AL.** For a scaled constraint row `c` with bounds
  `lb <= c <= ub`, multipliers `y_u, y_l >= 0` and penalty `mu > 0`, the subproblem rows are
  `sqrt(mu) * max(0, c - ub + y_u/mu)` and `sqrt(mu) * max(0, lb - c + y_l/mu)`; the merit
  is half their sum of squares (the constant `-y^2/(2 mu)` is dropped). Outer updates:
  `y_u <- max(0, y_u + mu (c - ub))`, `y_l <- max(0, y_l + mu (lb - c))`. Equalities:
  `sqrt(mu)(c - t) - y/sqrt(mu)`, no slacks. These rows are exactly what the composite model
  solves, and their dropped term is the Lagrangian term `(mu c - y) Hess(c)`, so compS and
  comp2 cover objective and constraint curvature in one mechanism.
- **Second order:** compS in every inner solve; comp2 only to finish (final outer
  iteration, or an inner solve stalled above tolerance).
- **Tolerances:** Conn-Gould schedule (master), inner `gtolk` floored near what the inner
  solve reaches (~1e-6 scaled). Decrease measured row by row (A6).
- **Termination:** master's logic; on a genuine inner stall raise `mu` on violated rows
  only. Never return success at an infeasible point.
- **Placement:** new methods `lsq-composite` (penalty) and `lsq-auglag-composite` (AL)
  beside c0-testing's `lsq-auglag`, which stays until the new one matches it.
- **Distances:** node-node rows with an explicit gap bound delta are the default (B); delta
  computed once at build with a backoff factor; no delta updates in the outer loop unless
  the backoff proves too generous.
- **Selected rows:** slots with identity tracking. A selected-row objective returns its K
  slot rows plus a static-size array of candidate ids; the optimizer pairs rows (A6) and
  attaches multipliers (B4, C1) by id. Rejected: one row per pair via log-sum-exp, which is
  conservative but bends on a 1/alpha scale (sub-mm for 2-3 mm conservatism), far below a
  useful step.
- **Done criterion** (precise_QH from circles): feasible to `ctol` by brute force
  (`check.py`), no linked pairs, scaled inner gradient <= 1e-6 at each outer update, final
  certificate (PD Hessian, small Newton decrement). Also the precise_QA notebook problem and
  `test_auglag`.

## Open decisions (ask the user; defaults in the last column)

| id | question | default |
|---|---|---|
| Q1 | Public accessor for per-row scaled `(lo, hi, target, is_bound)` on `ObjectiveFunction` (A3), or read `_scale`, `_normalize_target`, `normalization` privately as `ctr.py:40-54` does | public method, e.g. `ObjectiveFunction.scaled_bounds()` |
| Q3 | How the optimizer tells a selected-row objective which candidates to keep (rows with `y > 0`, B4) | boolean candidate mask in `constants`, updated between outer iterations |
| Q4 | Keep the exact-curve rows as `distance="curve"` for Fourier coils, or drop them | keep as an option |
| Q5 | Backoff factor on delta; default node count | 1.5x; nodes so that delta <= 2% of the bound |
| Q6 | Retire `second_order="constraints"` from `lsq_auglag` once C lands | keep, document as superseded |

(Q2, row pairing, is resolved: by candidate id, see A6.)

## Before serious AL runs: checklist

All of these, in this order; each links to its task.

1. Composite solver in DESC reproduces the E1 penalty run from circles (A1-A3).
2. Secant S in DESC reaches scaled gradient ~1e-6 at Gauss-Newton cost (A4).
3. Node-node rows with delta, candidate ids, keep-mask; property test passes; memory fits
   the 8 GB machine at the chosen node count (B1, B2, B4, B6).
4. `lsq-auglag-composite` passes `test_auglag` and a small active-inequality test (C1-C6).
5. AL builder, driver and outer-loop log exist (C7).
6. One short precise_QH AL run (2-3 outer iterations) whose log shows inner solves reaching
   their `gtolk`, `mu` growing only on violated groups, and the signed row's multiplier
   growing on the linked pair (C8).

Exact-Hessian finish (A5) and the row-paired decrease (A6) are needed for the final
certificate, not to start runs.

## A. Composite least-squares solver in DESC (penalty form)

**A1. `desc/optimize/composite.py`: the model pieces from `ctr.py`.** Pure functions on
arrays: `composite_resid(s, lo, hi, tgt, isb)` (from `resid`/`cost_of`, `ctr.py:73-83`),
`solve_composite(a, B, rT, lam, isb, lo, hi, S=None)` (from `solve_comp`, `:91`: semismooth
Newton on the convex piecewise-quadratic model, Armijo backtracking, returns step and
iteration count), `composite_model(a, B, r, q, isb, lo, hi)` (from `model_val`, `:131`).
Why: one implementation for the penalty and AL methods. Check: with no active hinge,
`solve_composite` equals the LM step `-(BᵀB + lam I)^-1 Bᵀr`; on a 1-D hinge problem it
returns the analytic minimizer.

**A2. `lsq_composite` in `desc/optimize/least_squares_composite.py`.** Port the loop
(`ctr.py:151-247`): Jacobian of pre-hinge scaled values, ratcheted column scaling, Nielsen
LM damping, acceptance on `rho > 0`, checkpoints through `callback`. Options:
`model={"gn","composite"}` (`gn` = what lsqtr sees, for comparison),
`hessian={"gn","secant","exact"}`, `finish_steps=k` (exact for the last k accepted steps
once `gtol` is met with the secant), `jac_chunk_size`. Return an `OptimizeResult` shaped
like `lsqtr`'s (`least_squares.py:32`). Check: reproduces E1 (README "Hands-off precise_QH
run") to the same certificate within 2x the iterations. Keep `ctr.py` as the reference
until this passes.

**A3. Register `lsq-composite`** in `_desc_wrappers.py`, following
`_optimize_desc_least_squares` (:471-552): `scalar=False, equality_constraints=False,
inequality_constraints=False` (bounds live in the objective), `hessian=False`. Uses Q1.
Check: `Optimizer("lsq-composite")` runs `setup_qh.py`'s problem from circles.

**A4. Secant S (`hessian="secant"`).** Port `ctr.py:144` and its update block: NL2SOL
sized rank-2 update with `y# = dg - G+ s` (row-order independent, required because selected
rows change identity), skip when `yᵀs <= 0`, S in unscaled coordinates, per-iteration
switch between GN and augmented model by which predicted the last step better, damping
floor from `eig(BstdᵀBstd + Sd)`. Check: from `G3o_y.npy`-like states the scaled gradient
drops 1e-2 -> 1e-6 in < 100 iterations (measured 77).

**A5. Exact S (`hessian="exact"`).** One HVP per column (`ctr.py:58-66`), damping floor
from the full Hessian including active hinge rows (the fixed `comp2` floor). Check: from a
compS endpoint, quadratic decrease of the scaled gradient over 3-5 steps.

**A6. Row-paired decrease.** Acceptance needs `cost(x) - cost(x_new)`; near convergence
that is 1e-10..1e-17 against a ~2e5 cost (ulp 3e-11), so a difference of two totals is
noise and lambda blows up (compS stopped near 1e-6, comp2 near 1e-9..1e-10). Compute
`sum((r - r_new)(r + r_new))/2`, cancelling large equal parts row by row. Fixed-position
rows pair by position; selected-row objectives pair by candidate id among rows with a
nonzero residual (rows entering or leaving the selection are inactive on both sides, since
`select_distance` exceeds the bound); an unmatched remainder falls back to the plain
difference. This removes the summation and cancellation part of the floor, not rounding
inside each residual; if that dominates, accept on decrease of the gradient norm once the
cost cannot resolve progress. Check: measure the floor before and after; comp2 on QH
reaches <= 1e-10 reliably (C1n got there by two lucky 3-ulp acceptances, E1 stopped at
7.7e-9).

## B. Node-node distance rows with an explicit gap bound

**B1. Rows.** Add `distance="node"` (default) to `CoilSetDistanceRows` (:2951) and
`PlasmaCoilSetDistanceRows` (:3192): rows are node-node (node-plasma point) distances
`d_ij`, candidates `(pair, i, j)` within `select_distance`, re-selected each evaluation into
slots. Node positions are smooth in the parameters for every coil class, so this needs no
per-class evaluator; the exact-curve path (:2811-2950) becomes `distance="curve"` (Q4). The
per-pair signed row is the pair's minimum node distance times the linking sign (kinked only
at ties, and only active near a crossing).

**B2. Gap bound delta.** At the true closest pair the connector is perpendicular to both
tangents, so the node estimate overshoots by at most
`delta = (h_a + h_b)^2 / (8 d) + (kappa_a h_a^2 + kappa_b h_b^2) / 8`
(`h` = largest arclength between consecutive nodes, `kappa` = max curvature, curves C^2
between nodes, `d` = `d_min`). Distance rows get bound `d_ij >= d_min + delta`; the signed
row does not (it only guards topology near d = 0). Compute `h`, `kappa` once at build from
the initial coils on a fine grid, times the backoff (Q5); for Fourier coils certify between
nodes with Bernstein (`max|x'| <= max_nodes|x'| / (1 - N ds / 2)`). Plasma-coil: plasma grid
spacing and surface principal curvature in place of curve b's. precise_QH (d_min 5.8 cm,
kappa ~14, L ~3.1 m): 129 nodes -> 0.7 cm (12%), 401 nodes -> 0.07 cm (1.2%). Check:
property test on random Fourier coils and arcs, `min d_ij - true distance <= delta`, true
distance from `_segment_segment_distance` (`_coils.py:1194`) on a 1e4-node polyline.

**B3. Piecewise classes.** Arcs (`PiecewisePlanarArc`, `PolarPlanarArc`): nodes on every
hinge (node count a multiple of `B`) so each stretch between nodes is smooth; per-arc `h`,
`kappa`. Splines with breaks: nodes on the breaks; derivative bounds per piece from the
spline (Bernstein does not apply). Check: the B2 property test on both.

**B4. Candidate ids and multipliers.** Return a static-size int array of candidate ids with
the slot rows (padding -1), from the same `jnp.nonzero(mask, size=K)` that fills the slots.
The per-pair signed rows and the plasma rows follow the same scheme (signed rows have a
fixed id per pair). Selection keeps any candidate with a nonzero multiplier (Q3), so a row
carrying force cannot drop out and make the merit jump. Multipliers stored sparsely
(only `y > 0`), keyed by id. Check: an AL run where a contact slides along the coil keeps
`sum y` continuous across selection changes.

**B5. Tests** (compute-level, per the objective-test convention): bound property test, FD
Jacobian, NaN gradient, signed row on threaded rings (pattern: `test_coil_set_distance_rows`,
`tests/test_objective_funs.py:1280`), registration in the `specials` lists.

**B6. Memory sizing.** Rows scale as (contact length / h)^2 per contact: roughly 5-20k
active rows at 400 nodes on precise_QH. Measure peak memory of one Jacobian at the chosen
node count with `jac_chunk_size` set; size `max_active_rows` from `active_row_count()` with
margin. Check: fits on the 8 GB machine with the QuadraticFlux Jacobian alongside.

## C. Slack-free hinge augmented Lagrangian

**C1. `lsq_auglag_composite`** in a new `desc/optimize/aug_lagrangian_composite.py`.
Inequality rows as in "Decisions": each row's pre-hinge value `c` goes to the A machinery
with its bound shifted by `-y/mu` and scaled by `sqrt(mu)`; for selected-row objectives,
`y` is gathered per slot by candidate id (B4). Equalities as in `lsq_auglag`'s `lagfun`
(`aug_lagrangian_ls.py:404`). No `inequality_to_bounds` (:391), no `z2xs`.

**C2. Outer loop.** Conn-Gould updates (master), `gtolk` floored near 1e-6 scaled; `mu`
grows only on rows still violated (the hinge residual is the violation, so c0-testing's
true-violation bookkeeping :429-450 and `y_update_gate` :629 are unnecessary); keep
`max_multiplier` / `max_penalty_parameter` as caps. Jacobian at an outer update as in PR
#2298 (already the effect of the vendored rescale at :1449): rescale constraint rows by
`sqrt(mu_new/mu_old)` instead of re-evaluating, and do the update right after the accepted
step that built J, so no extra unscaled copy lives across iterations (the vendored code
keeps one).

**C3. Stalls.** Master termination; an inner stall raises `mu` on violated rows and
restarts the inner solve; never success at an infeasible point.

**C4. Second order.** Inner solves with `hessian="secant"`; `finish_steps` exact steps at
the last outer iteration (A5).

**C5. Register `lsq-auglag-composite`** next to `lsq-auglag` (`_desc_wrappers.py:329-470`):
`equality_constraints=True, inequality_constraints=True`; constraints come from the
`constraints=` objectives and their per-row bounds (`optimizer.py:579`).

**C6. Tests.** `test_auglag` (`tests/test_optimizer.py:984`) passes with the new method; a
small test with active inequalities and a known solution.

**C7. AL tooling in `devtools/coil_auglag/`.**
- Builder: `setup_qh.py` today puts every engineering limit in the objective as a penalty.
  Add an AL mode returning QuadraticFlux as the objective and coil-coil, plasma-coil, length
  and curvature as `constraints=` (weights irrelevant there), plus the linear constraints.
- Driver: run `Optimizer("lsq-auglag-composite")` from a tag (`mkstage.py`) with
  checkpoints, writing `TAG_x.npy` per outer iteration.
- Outer-loop log, one line per outer iteration: `mu` (min/median/max per constraint
  group), `max y` per group, true violation per group (`check.py`'s brute force), inner
  iterations, whether the inner solve met `gtolk`, linked pairs, signed-row multiplier.
Check: a 2-outer-iteration run on precise_QH produces a readable log and resumable state.

**C8. Benchmarks.** precise_QA notebook problem; precise_QH from circles to the done
criterion. Run one heavy job at a time.

## D. Remaining items

- **D1.** c0-testing AL issues (README "Known issues"): `fmin_auglag` stall counter reset,
  `step_veto` looping to `maxiter`, `_trust_region_step_eigh`
  (`aug_lagrangian_ls.py:35`) on singular PSD models. Fix where `lsq-auglag` and
  `fmin-auglag` stay in use, or document as superseded; decide `test_auglag` for the old
  `lsq-auglag` once C passes it.
- **D2.** Resolved: planar-coil NaN near +z, fixed by folding in PR #2277
  (`rotate_vector_to_vector`).
- **D3.** Q6 (`second_order="constraints"`).
- **D4.** `desc.objectives.utils.softmin` (used by `CoilSetMinDistance(use_softmin=True)`)
  is a Boltzmann-weighted average, which is >= the true minimum, so a lower bound on it is
  not conservative. Raise separately (generic, present on master).
- **D5.** API docs for the new methods and options. The user writes CHANGELOG entries.

## E. Distance constraints shared by every coil representation

Decided with the user (2026-10-06): the representations being compared (FourierPlanar,
FourierXYZ, PolarPlanarArcCoil) get identical distance constraints. Coil-coil: node-node
`CoilSetDistanceRows`; curve mode cannot take arc coils, and a whole-curve Newton
projection would kink across the bisector of every concave corner. Plasma-coil: a fixed,
splined distance field, since the plasma does not move in stage 2. Dense-grid node-node
rows were rejected: the gap needs M=N=200 (1.6 mm, ~430k active rows), and the selection
pass and candidate ids scale with plasma points x coil nodes (~2.6e9 ids).

**E1. Node-mode coil-coil in the AL (run AL9).** AL8 with `cc_distance="node"`, explicit
`cc_gap=2.102e-3` (1.5 x `_node_gap` with h from the AL8 coils at 513 nodes, k = kmax
13.7), `cc_sel=0.1`, `cc_K=10000` (~3-6k active). Check: converges like AL8 (no stalled
inner solve, violation falling each outer), QuadraticFlux a little above 8.0733, and
`check.py` finds brute-force coil-coil >= 5.8 cm with no between-node miss.

**Done (AL10).** **E2. Gap set in advance from the bounds (decided with the user, 2026-10-06).** Spacing
is not fixed during a run: at 513 nodes h is 3.8 mm on the circles and 10.8 mm at
AL8/AL9, so a gap built on the circles is ~8x too small. Rule: `h = s * Lmax / n` and
`k = kmax`, gap `= backoff * _node_gap(h, h, kmax, kmax, d_min)`, where `s` is the ratio
of the largest node spacing to the mean. Measured `s` (h n / L, per coil): circles 1.00;
AL8 and AL9 1.68-1.89 (FourierPlanar, polar parameter). Default `s = 2.0`; the gap goes as
s^2, so that is ~11% over the 1.9 seen. With s = 2.0 at 513 nodes: h = 11.4 mm, gap
2.3 mm (4% of 5.8 cm); AL9's hand-set 2.102 mm against 2.094 mm needed at its final state.
- `desc/objectives/_coils.py`: a public helper, e.g.
  `node_gap_estimate(n_nodes, max_length, max_curvature, min_distance, speed_ratio=2.0,
  backoff=1.5)`, returning the gap to pass as `gap=`. The objectives keep their current
  default (gap from the coils at build); no new options on them.
- `check.py`: report the gap needed at the checked state (`_coil_gap_data` on the row
  grid, `_node_gap` with the measured h and k) next to the gap used, and flag when the
  used gap is more than 2% short.
- Arc coils: nodes are split evenly per arc, so `s` there includes the spread of arc
  lengths; measure it on the first PolarPlanarArcCoil run before trusting 2.0.
Escalation (user): only if runs regularly come out more than 2% short do we build
anything adaptive (e.g. gap updates between outer iterations).
Check: `check.py` on AL9 reports 2.094 mm needed against 2.102 mm used, no flag; a run
from circles with the helper's gap ends with no flag.

**Done (AL11).** **E3. `PlasmaCoilDistanceField` in `desc/objectives/_coils.py`.** Rows: `phi(x_i) - margin`
for each node of the field-period-independent coils (corners on nodes via
`_coil_node_grid`), bounds `(d_min, inf)`. Fixed row count and identity: no selection,
slots, ids or keep mask. Build:
- cylindrical (R, phi, Z) grid over one field period, periodic in phi, padded ~0.3 m
  past the boundary; spacing default 10 mm (3M nodes, 24 MB on precise_QH);
- exact distance by Newton projection on the boundary (`x`, `e_theta`, `e_zeta`, second
  derivatives; seed from a KD-tree of a dense boundary cloud) at nodes in the band around
  the bound, cloud distance elsewhere; signed (negative inside the plasma) so steps across
  the boundary are pushed back;
- smooth saturation beyond the band (constraint inactive there), monotone;
- cubic B-spline coefficients (`scipy.ndimage.spline_filter`, `mode="grid-wrap"`);
- `margin` = measured spline error (max over a test set in the band) plus the
  between-node dip `h_c^2/8 (kmax + 1/d_min)` of the coil.
Why: ~2k rows (4 coils x 513 nodes) against 150-430k for dense node-node rows, cost
independent of plasma resolution, works for any coil class. Prototype
(`sdf_probe.py`, precise_QH): max error 0.026 mm at h = 20 mm, 0.008 mm at
10 mm against Newton-exact distances in d in [0.08, 0.16] m.

**Done (AL11).** **E4. Tricubic B-spline evaluation in JAX.** `jax.scipy.ndimage.map_coordinates` stops at
order 1 and interpax's cubic is only C1; write the 4x4x4 gather with cubic B-spline
weights (C2, which the exact-Hessian finish needs). Check: matches
`scipy.ndimage.map_coordinates(order=3, prefilter=False)` to 1e-13; `jax.jacfwd` against
finite differences.

**Done (AL11).** **E5. Reach check at build.** The field is smooth near d_min only if the boundary's
concave radius (positive principal curvature in DESC's convention, where convex is
negative) exceeds d_min and no two distant parts of the boundary are equally close.
precise_QH: concave radius >= 23 cm (k1 max 4.3), convex edges to 1.1 cm (k2 min -92,
harmless). Warn when 1/max(k_concave) < 1.5 d_min, and report the largest spline error in
the band from a random test set. Check: warns on a deliberately dented boundary.

**E6. Benchmarks.** Rerun the AL9 setup with E3 for plasma-coil; then precise_QA
(FourierXYZ) and PolarPlanarArcCoil on the same constraints; then the user's test cases.
Check: `check.py` feasibility (fine-grid distances) with no shortfall beyond `margin`.

## Order and dependencies

A1 -> A2 -> A3 -> A4 (needed by C); A5, A6 can follow in parallel with B.
B1 -> B2 -> B3; B4 needs B1; B5, B6 close B. C needs A1-A4, B1 and B4.
C1 -> C2 -> C3 -> C4 -> C5 -> C6 -> C7 -> C8. D last.
One heavy job at a time on the 8 GB machine; keep `jac_chunk_size` set; run benchmarks from
`devtools/coil_auglag/` with outputs left untracked.
