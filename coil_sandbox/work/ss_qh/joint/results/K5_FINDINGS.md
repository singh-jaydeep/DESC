# The k = 5 single-stage stall: findings and takeaways

Answers `K5_STALL.md` (2026-10-08/09). Details, every measurement and all scripts: `../stall/DIAGNOSIS.md` and
`../stall/`. Setup: precise_QH-derived boundary, arcB2 coils (M5) with every hinge at z = 0, `single_stage.py`,
k = 5 (boundary |m|, |n| ≤ 5 free), proximal-lsq-exact, equilibrium basis L = M = N = 8.

## Bottom line

* **The stall is not about the coils, and not about the outer optimizer. It is the equilibrium sub-problem.** At
  basis 8, the fixed-boundary equilibrium that the proximal method re-solves at every trial point is not a
  well-defined, smooth function of the boundary. Its soft directions are poloidal-angle (theta) relabelings
  that the truncated basis cannot represent exactly. Along them (a) DESC's Gauss-Newton solve does not
  converge, so the objective is noisy, and (b) the least-squares problem has several branches (local minima
  with near-equal force residual but ~5% different QS) that end at folds. The optimizer drives the boundary to
  the edge of the QS-favorable branch and stalls there: it is optimizing gauge-dependent truncation error.
* **At basis 12 all of this shrinks below what matters.** The gauge-induced physics spread falls ~5% (basis 8) →
  0.5% (10) → 0.02% (12). The same k5 step at basis 12 does not stall.
* **Recommended recipe, measured:** basis 12 (chunked ForceBalance), `--solve-tol 1e-6`, `--method lsq-composite
  --hessian secant`. From the k4b state: cost **1.634 → 0.818** in 140 iterations (~1.5 h), against 1.634 → 1.625
  for the original basis-8 k5. B.n 6.82e-3 → **5.12e-3** (−25%), Boozer QS at rho = 0.5 3.65e-3 → **1.41e-3**. Validated
  by a tight re-solve at basis 12 (0.8182) and at basis 14 (0.8183). At the endpoint the linear model is exact down
  to 1e-7 steps (ratio 0.9998; it was −318 at basis 8), so the remaining limit is ordinary convergence speed.

## What happens, step by step

1. **Noise from the inner solve.** At the stalled k5 point, every trial whose equilibrium re-solve iterates gets
   a step-size-independent jump in the QS cost (3-7e-6 at `--solve-tol 1e-10`; worse, ~3e-5, at 1e-13). The
   steps the trust region can still afford predict ~2e-6, so they are all rejected and the radius collapses
   to xtol. B.n, the coil terms and volume match their linear models to ≤1e-3. QS and iota do not.
2. **The noise is theta-gauge drift.** Re-solving the unchanged equilibrium leaves the force cost exactly
   flat while flux-surface points move ~92% tangentially (a poloidal relabeling ω ~ 1e-4) and QS changes by
   ~1e-4 relative. Of the 16 softest force-balance modes, 13 are almost pure relabelings.
3. **Why Gauss-Newton fails there.** The discrete equilibrium is a NONZERO-residual least-squares minimizer.
   Along the soft modes JᵀJ is ~1e-7 of its maximum and the dropped residual-curvature term S = Σ rᵢ∇²rᵢ is
   ~4000x larger and indefinite. Gauss-Newton does not converge there: it stops wherever gtol is met, so the
   result depends on the start and the tolerance. The exact Hessian (and the subspace Newton below) converges,
   deterministically to ~1e-7, to a lower force residual. The "converged" GN k5 equilibrium was not the
   minimizer.
4. **Branches and folds.** With the inner solve converged (subspace Newton), the stall persists. Rejected
   trials now jump by a constant ~1% of the cost at any step ≥ 3e-4 and converge to a different, lower-force
   minimum (branch B) with 5.5% worse QS. Both branches exist at the same boundary. The QS-favorable branch A
   ceases to exist ~1e-7 (in boundary coefficients) from where the optimizer parked it. The perturbation
   predictor is not the cause (orders 0 and 2 give identical jumps). The branches are the same continuum field
   in two gauges: their difference is 98% tangential, and at basis 10 it is 99.9% tangential with 12x smaller
   physics differences. A vacuum field in a fixed boundary is unique, so this is a discretization artifact.
   It is not islands: the force residual does not localize at iota = 1.2.
5. **Resolution.** At basis 12 the soft modes are ~1000x softer (closer to an exact symmetry), and walking
   along them changes QS by only ~2e-4 relative. The k5 step runs cleanly there.

## Results (all costs re-scored at basis 12 with the same weights; lower is better)

| run | where optimized | cost | B.n | Boozer QS rho 0.25/0.5/0.75/1 (e-3) | iota range |
|---|---|---|---|---|---|
| k4b start | basis 8 | 1.634 | 6.823e-3 | 2.08/3.65/4.92/7.87 | [1.1804, 1.2031] |
| original k5 (stalled) | basis 8, GN tol 1e-10 | 1.625 | 6.809e-3 | 2.08/3.65/4.92/7.87 | [1.1804, 1.2031] |
| k5, solve-tol 1e-6 | basis 8 | 1.496 | 6.481e-3 | 1.95/3.49/4.86/7.84 | [1.1807, 1.2002] |
| k5, composite+secant, 1e-6 | basis 8 | 1.421 | 6.295e-3 | 1.87/3.18/4.37/7.34 | [1.1798, 1.2044] |
| k5, lambda-pinned composite | basis 8 (biased path) | 0.996 | 5.495e-3 | 2.12/2.21/3.30/7.51 | [1.1788, 1.2088] |
| **k5, composite+secant, 1e-6** | **basis 12, 60 its** | **1.035** | **5.541e-3** | **1.08/1.80/3.37/7.12** | [1.1845, 1.2001] |
| (same, re-scored at basis 14) | | 1.037 | 5.540e-3 | 1.09/1.79/3.39/7.14 | [1.1845, 1.2001] |

| **k5, continued +80 its** | **basis 12, 140 its total** | **0.818** | **5.120e-3** | **0.91/1.41/2.99/6.67** | [1.1854, 1.2008] |
| (same, re-scored at basis 14) | | 0.818 | 5.122e-3 | 0.91/1.41/2.99/6.67 | [1.1854, 1.2008] |

The final design (`../stall/k5_b12_comp2/`: `eq_k5_basis12.h5`, `coils_k5.h5`): A 8.316, R0 1.014, a 0.122;
coils L 1.001 (FAILS L by 0.1%, as every run did), kappa 0.999, kappa_MS 0.888, d_cc 1.009, d_pc 1.149, unlinked,
hinges at z = 0 exactly. Still descending at 80 its (~5e-4 per iteration). It has not had the AL stage-2 refinement
(`run_al.py`), which the basis-8 k5 needed to reach 6.356e-3.

Endpoint probe at basis 12 (`../stall/probe_b12_end`, solve-tol 1e-6): ratios 0.9985-0.9998 for steepest-descent
steps 1e-3 … 1e-7 and 0.94/0.999 along the Gauss-Newton direction. The QS cost change matches its prediction to
~1% on both, including the direction where basis 8 was off by 2x.

## Actionable takeaways

1. **Run single stage at equilibrium basis ≥ 12 for k ≥ 5 on this QH.** Basis 8 for an |m|, |n| ≤ 5 boundary is
   under-resolved: it gets the physics wrong by 2-5%, and its gauge ambiguity stalls the optimizer. Basis 12 fits
   on the 16 GB GPU with `--force-chunk 200` (equilibrium-solve peak ~2 GB; proximal ~5.5 GB).
2. **Set the ForceBalance `jac_chunk_size` explicitly.** DESC's "auto" puts all columns in one batch and OOMs at
   basis 12 (8.7 GB allocation); 100-400 costs nothing in speed here.
3. **Use `--solve-tol 1e-6` for the proximal re-solve.** A tighter tolerance is WORSE: more inner iterations means
   more gauge drift. At basis 12 the loose tolerance costs nothing measurable (a tight re-solve of the result left
   the cost unchanged). At basis 8 it cost ~0.01 and became path-dependent.
4. **Prefer `--method lsq-composite --hessian secant`** (needs the uncommitted wrapper fix below). It reached 1.035
   at basis 12 in 60 iterations; lsqtr spends its first ~8 evaluations shrinking an initial trust radius of 125
   (`--init-tr` exists for it).
5. **Always validate a single-stage result**: re-solve the equilibrium tightly at the run's basis AND at a higher
   basis, then re-score with `single_stage.py --dry` (`stall/rescore12.py`, `stall/rescore_all.sh`). Basis-8
   scores were off by up to ~0.01 in cost and 2-5% in QS.
6. **Do not use penalty "gauge pins"** (`--lambda-pin`, the tangential pin, or the equal-arclength
   `GaugeCondition` at large weight). They make the re-solve deterministic but admit non-equilibria, or pull the
   physics when the gauge they ask for is not representable in the basis. Verified: after an unpinned re-solve
   the apparent gains shrink or vanish.
7. **When diagnosing an optimizer stall, probe the model term by term** (`stall/probe.py`): predicted vs actual
   change per objective along fixed directions, at several step sizes. A step-size-independent mismatch means
   noise or a discontinuity in the objective, not a bad model. This took the k5 question from "hinges?" to "inner
   solve" in one run.

## Code changes (all uncommitted)

* `desc/optimize/_constraint_wrappers.py`: `LinearConstraintProjection.scaled_bounds` and
  `ProximalProjection.scaled_bounds`/`row_ids` delegate to the wrapped objective. Without them
  `proximal-lsq-composite` fails (`'ObjectiveFunction' object has no attribute 'bounds'`). Relevant optimizer
  tests: 16 pass; 1 (`test_all_optimizers_lsq[lsq-auglag-composite]`) fails with or without the change.
* `coil_sandbox/single_stage.py`: `--method`, `--hessian`, `--finish-steps`, `--init-tr`, `--track-steps`
  (step_log in result.json), `--solve-options`, `--force-chunk`, `--inner-sn`, `--lambda-pin` (marked DO NOT USE).
* `coil_sandbox/sandbox/subspace_newton.py`: subspace-Newton least-squares solver (S on the k smallest
  eigenvectors of JᵀJ, eigenbasis trust region, paired decrease) and `install_proximal_hook`. It makes the basis-8
  inner solve deterministic (~1e-7) at ~1 min cold / ~15 s warm, but does not remove the branch folds, and at basis
  12 the valley is so flat that it does not converge (nor does it need to). A diagnostic tool, not the recommendation.
* `coil_sandbox/sandbox/gauge_fix.py`: `LambdaCondensation`, `TangentialPin`, `GaugeCondition` (negative results,
  kept for the record).

## Open questions

* A discrete-consistent gauge condition (stationarity of a VMEC-style spectral width along the basis-projected
  relabeling directions) could remove the branch ambiguity at moderate resolution. Not built; basis 12 made it
  unnecessary here.
* The proximal's zero-residual implicit derivative is accurate along the boundary direction tested at the true
  minimizer, but QS was off by ~2x along a Gauss-Newton direction at basis 8. Not re-checked at basis 12.
* The coils still FAIL L by 0.1% and iota min sits just below 1.19 in every run: both are weighted penalties.
  Use the AL stage-2 refinement (`run_al.py`) for hard limits.
* The k4b "zig-zag" (constant step, ratio 0.33-0.46) is the same family: model over-prediction at basis 8. Not
  separately re-tested at basis 12.
