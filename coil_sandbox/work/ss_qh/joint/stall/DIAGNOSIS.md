# K5 stall: diagnosis (2026-10-08, local box, desc-dev env)

Answers `../results/K5_STALL.md`. Everything here is in `work/ss_qh/joint/stall/` (`run.sh NAME [flags]`
reproduces each run; `probe.py`, `gauge.py`).

## Verdict

The k = 5 stall is **noise in the objective from the proximal equilibrium re-solve**, not the coil hinges.
Each re-solve slides along DESC's unfixed poloidal-angle gauge (theta relabeling): force balance cannot see
it, but QS and iota evaluated on the fixed M=N=8 computational grid are not exactly gauge-invariant. Any
trial point whose re-solve iterates at all gets a step-size-independent jump in the QS cost. At the k5
endpoint that jump is 3-7e-6 at `--solve-tol 1e-10`. Steps the trust region can still afford there predict
only ~2e-6, so every one is rejected and the radius collapses to `xtol`.

Of the three hypotheses in K5_STALL.md:
1. coil hinges at their bounds: real, but not what stalls k5 (see "Coil terms" below).
2. proximal model mismatch / solve noise: **yes**, but the noise is gauge drift, and it gets WORSE with a
   tighter solve (more inner iterations means more drift). It is not the 1e-10 solve tolerance's own noise.
3. scaling of the new modes / the 9 rejections on iteration 1: the first trust radius is 1.254e2 (the
   `"scipy"` default ‖x/scale‖), and 9 quarterings give 4.8e-4, which matches the 3.7e-4 step. That costs
   time, not progress.

## Evidence

### Probe (`probe.py`): actual vs linear-model change per term along fixed directions

At the k5 START (k4b end), steepest descent, every term matches the model to <1e-3 relative for
‖h‖ ≥ 1e-5, except QS (~9%) and iota (noise only). Along the full Gauss-Newton step (‖h‖ = 881, scaled-J
condition 2.8e4):
* B.n: the linearization error is proportional to the step (good Jacobian).
* coil length / curvature: very strong 2nd order, valid only for ‖h‖ ≲ 1e-4 along GN. At ‖h‖ = 0.88 the
  curvature cost goes 2e-5 → 0.43. The GN step piles motion into coil directions the inactive hinge rows
  cannot see. This is what the trust region defends against early on.
* QS: the error is FLAT at ~0.5 for t = 1e-2 … 1e-4. It is not a smoothness error (see the gauge below).

At the k5 ENDPOINT (`probe_k5end`), steepest descent:

| ‖h‖ | predicted | actual | QS term actual/pred |
|---|---|---|---|
| 1e-4 | 2.24e-5 | 2.26e-6 | −9.8e-6 / +4.6e-7 |
| 1e-5 | 2.24e-6 | −1.22e-6 | −3.4e-6 / +4.6e-8 |
| 1e-6 | 2.24e-7 | −5.27e-6 | −5.5e-6 / +4.6e-9 |
| 1e-7 | 2.24e-8 | −7.14e-6 | −7.2e-6 / +4.6e-10 |

B.n, length, curvature, d_cc and volume match to ≤1e-3 at the same points.

### The re-solves (`PROBE_SOLVE_LOG=1`)

Each trial re-solve at tol 1e-10 takes 7-21 iterations and ends at the SAME force cost, 1.154e-7. The stored
point itself takes 0 iterations. At tol 1e-13 every solve hits maxiter 100, the force cost is still 1.154e-7,
and the QS jump grows to 2.5-4.9e-5.

### It is the theta gauge (`gauge.py`)

Re-solving the unchanged k5 equilibrium (tol 1e-13, 100 iterations): force cost 1.153717e-7 unchanged.
The flux-surface points move ~92% tangentially, a poloidal shift ω ≈ 1e-4 rad (normal part 7-9%), λ
changes by 8.5e-5 rms, the QS residual changes by 3.8e-4 relative, and iota changes by up to 1.5e-5. DESC
leaves θ free (only `FixThetaSFL`, λ = 0, would pin it), and the Gauss-Newton inner step moves far along
this very soft mode.

The DESC proximal Jacobian also shifts the force-Jacobian spectrum (`sf += sf[-1]`,
`desc/optimize/_constraint_wrappers.py:1364`, upstream since 2024), which halves the softest modes'
response. Removing it (`PROX_REG=none`) cuts the flat QS error along GN from ~0.5 to ~0.3, but leaves the
trust-region ratios essentially unchanged. A minor contributor.

### Remedy probes at the endpoint (steepest descent, ‖h‖ = 1e-3 … 1e-7)

| inner solve | small steps (≤1e-5) | ‖h‖ ≥ 1e-4 |
|---|---|---|
| tol 1e-10 (as run) | 7-21 its, ratios −0.5 … −318 | ratio 0.10 |
| tol 1e-10 + `initial_trust_radius: conngould` | identical | identical |
| tol 1e-8 | 0-1 its, QS jumps ~4e-7 | ratio 0.55 |
| tol 1e-6 | **0 its, ratios 1.005-1.012** | 1 it: QS jump ~5e-5, ratio 0.59 |

A single inner iteration already jumps along the gauge.

## The fix, tested: k5 rerun with `--solve-tol 1e-6` (`k5_tol6/`, else identical to `../ss_z0.sh k5`)

| | original k5 (tol 1e-10) | `--solve-tol 1e-6` |
|---|---|---|
| exit | xtol after 12 its (35 nfev) | maxiter 100 (113 nfev), still descending |
| cost | 1.620 | **1.493** |
| dense B.n | 6.806e-3 | **6.480e-3** (−4.8%; k4b 6.820e-3) |
| QS Boozer rho 0.25/0.5/1 | 2.11e-3 / 3.70e-3 / 7.98e-3 | 1.97e-3 / 3.53e-3 / 7.69e-3 |
| iota min (bound 1.19) | 1.1807 | 1.1800 |
| wall | 5.7 min | 6.0 min |

Validity: re-solving the result at tol 1e-10 (`resolve.py`, 76 its to gtol) leaves the force cost unchanged at
3.861e-7, and the re-scored table (`k5_tol6/verify`, `--dry`) is identical (cost 1.493, B.n 6.4802e-3). The
loose solve did not leave the equilibrium unconverged. The higher force floor (3.9e-7 vs 1.15e-7) is this
boundary's least-squares floor at M = N = 8. Coils still FAIL L by 0.1% (as k4b and k5 did: a weighted
penalty, not a constraint).

## What limits it now (the k4b "zig-zag" regime)

`k5_tol6/result.json` step_log: from iteration ~30 on, the trust radius is pinned at 0.0612 and EVERY step is
accepted with ratio 0.33-0.46. That is inside the [0.25, 0.75] band where lsqtr neither grows nor shrinks the
radius. The model over-predicts each step ~2.5x (2e-3 predicted, ~8e-4 actual). This is a systematic
model error, not noise. It is the regime K5_STALL.md saw at the end of k4b (step 3.4e-3, ~7e-4 per
iteration). It is consistent with Gauss-Newton dropping the second-order term on a large-residual problem
(cost ~1.5). The 8 rejections on iteration 1 (initial radius 125) are unchanged.

## Second lever: `proximal-lsq-composite` + `hessian="secant"` (`k5_comp_secant/`)

Needed a DESC fix first: `ProximalProjection` keeps `[objective, constraint]` as its `objectives`, so the
inherited `scaled_bounds()` failed (`'ObjectiveFunction' object has no attribute 'bounds'`). Added explicit
delegation, `LinearConstraintProjection.scaled_bounds` and `ProximalProjection.scaled_bounds`/`row_ids`
(`desc/optimize/_constraint_wrappers.py`, uncommitted). `single_stage.py --method lsq-composite --hessian secant`.

Same start and flags as the original k5 plus `--solve-tol 1e-6`:

| | original k5 | lsqtr tol 1e-6 (100 its) | composite+secant tol 1e-6 (12 its) |
|---|---|---|---|
| cost, after a tight re-solve + re-score | 1.620 | 1.493 | **1.430** (1.420 as run) |
| dense B.n | 6.806e-3 | 6.480e-3 | **6.300e-3** (−7.4%) |
| QS Boozer rho 0.25/0.5/0.75/1 | 2.11/3.70/4.82/7.98e-3 | 1.97/3.53/4.84/7.69e-3 | 1.91/3.31/4.33/7.37e-3 |
| iota min (1.19) | 1.1807 | 1.1800 | 1.1801 |
| d_pc / kMS | 1.053 / 0.971 | 1.065 / 0.967 | 1.086 / 0.952 |
| force LSQ floor at M=N=8 | 1.15e-7 | 3.9e-7 | **1.44e-6** (max normalized 8.5e-4) |
| wall | 5.7 min | 6.0 min | 3.4 min |

All the progress came in the first 6 accepted steps (cost 1.628 → 1.420, step norms 7-15). There was no
damping start-up waste, because LM starts at 1e-3 instead of an initial trust radius of 125. Then it stalled
on a new form of the same disease:
from iteration 6 every rejected trial has actual reduction **−7.70e-4 exactly**, at any step from 1e-5 down to
3e-9. After the big steps, the stored equilibrium is a *perturbed* state that passes gtol 1e-6 without being
re-solved. Any trial whose re-solve iterates lands on the nearby true solution, which is 7.7e-4 worse. Only
trials small enough for a 0-iteration solve get accepted. The loose tolerance trades the gauge noise for a
path-dependent objective. Re-solving the end state tightly cost ~0.01 of the claimed gain (1.420 → 1.430).

The force floor rising 12x is a warning: the big steps moved the boundary into shaping that M = N = 8
resolves noticeably worse. Check k5 boundaries at higher equilibrium resolution before trusting their physics.

## Recommendations

1. **Now:** run k steps with `--solve-tol 1e-6` (or 1e-7), and finish each with a tight re-solve plus a
   re-score (`resolve.py`, then `single_stage.py --dry`). That is cheap and clears the k5 stall.
2. **Better:** `--method lsq-composite --hessian secant` (with the wrapper fix). It gets ~2x the reduction
   of lsqtr in a third of the iterations, and avoids the 125 initial-radius waste. If staying on lsqtr, pass
   `--init-tr` (e.g. `1e-2`, the scale of the steps it actually accepts) to skip the ~8 rejected
   evaluations per k step.
3. **Root cause, not fixed here:** f(x) is not a deterministic smooth function of x, because the inner
   solve has a soft theta-gauge mode and a nonzero force floor. Options, roughly by cost: (a) evaluate QS
   and iota on finer grids, so they are closer to gauge-invariant (the jump size is the discretization
   error of a gauge-invariant quantity); (b) raise eq L/M/N for k >= 5 (lower force floor, smaller
   jumps); (c) fix the gauge in the inner solve (`FixThetaSFL` is the only built-in; it changes the
   representation); (d) in ProximalProjection, always re-solve the accepted point (store the solved,
   not the perturbed, state), so trials and the stored point agree.
4. The `sf += sf[-1]` shift in the proximal Jacobian (upstream DESC) biases soft-mode responses by up to
   2x. It is measurable but minor here; worth raising upstream.
5. Coil hinges (hypothesis 1): the hinge terms are extremely nonlinear along Gauss-Newton directions (the
   linear model holds only for ‖h‖ ≲ 1e-4 along GN) and drive the early rejections. The composite model,
   which keeps them exact, handles that. They were not the cause of the k5 stall.

## Files

`run.sh NAME [flags]` (one capped job); `probe.py` (`--gn-t`, `--sd-h`; env `PROX_REG=none`,
`PROBE_SOLVE_LOG=1`); `gauge.py EQ TOL MAXITER`; `resolve.py IN OUT [TOL MAXITER]`. Runs: `probe_k5`,
`probe_k5end`, `probe_reg_{shift,none}`, `probe_end_{tol10,tol13,tol8,tol6,cg}`, `k5_tol6` (+ `verify`),
`k5_comp_secant` (+ `verify`). `single_stage.py` gained `--method`, `--hessian`, `--finish-steps`,
`--init-tr`, `--track-steps` (step_log in result.json), and `--solve-options`.

## Follow-up (same day): grids, soft modes, and a lambda pin (negative result)

* **Finer objective grids do not help.** `gauge2.py measure`: after a drift re-solve, QS changes by 8.1e-5 (stock
  measure) and 1.05e-4 (area measure), and iota by 1.46e-5, identically on 1x/2x/3x grids. The drifted state is a
  slightly different field: the relabeling is a symmetry only in the continuum, and at L=M=N=8 it changes the
  truncation error. Higher EQUILIBRIUM resolution is the lever (less truncation per unit drift), untested.
* **Soft modes** (`softmodes.py`): the scaled force Jacobian (5346 x 856) has condition number ~5e6; 10 sv < 1e-6 max,
  54 < 1e-5, 153 < 1e-4. Of the 16 softest, 13 are theta relabelings (displacement 98-100% tangential, dlambda fits
  the relabeling prediction at 0.65-0.985); modes 3, 11, 13 are mixed (fit 0.16-0.50). The gauge is the dominant
  source, as a family of modes, with some physical mixing at the bottom.
* **lambda pin** (`sandbox/gauge_fix.py`, `--lambda-pin`): rows eps (L_lmn - L_ref). Unanchored (L_ref = 0) it
  pulls toward lambda = 0 and distorts. Anchored (at eps = 3e-2) it made the re-solves deterministic (spread 20-100x
  smaller, inner solves 4-9x faster) and the probe consistent at tol 1e-10. **But it admits non-equilibria**:
  lambda is not pure gauge. Once R and Z (the labels) are fixed, lambda is set by physics (the straight-field-line
  map), and the pin resists that physical response. After an UNPINNED tight re-solve:
  pinned lsqtr 1.415 → 1.506 (QS Boozer rho 0.25 1.75e-3 → 2.29e-3, iota min 1.185 → 1.176); pinned
  composite 0.916 → 1.004 (QS rho 0.25/0.5/1: 0.61/1.46/6.52e-3 → 2.28/2.22/7.42e-3). The force cost drops 40-60%
  when lambda is freed. Only the dense B.n (boundary + coils) survives: 5.49e-3. Do not use `--lambda-pin`.
  Lesson: check any change to the inner solve with an unregularized tight re-solve + re-score.
* A physically neutral gauge fix must act only on the gauge orbit: penalize TANGENTIAL displacement relative
  to the anchor ((dR R_t + dZ Z_t)/|X_t| on interior surfaces; normal displacements and lambda stay free), or
  impose a proper gauge condition (VMEC-style spectral condensation on R, Z), or use FixThetaSFL (exact PEST
  gauge, needs more modes). Untested.

## Stepping back: the root cause is a zero-residual solver on a nonzero-residual problem

Penalty pins are a dead end. `gauge3.py bias` (boundary walk k5 → k5_tol6 in 10 perturb+solve steps, then an
unpinned re-solve) shows both pins biased even when re-anchored every step: tangential 1e-1 force +86%, iota 1e-2;
lambda 3e-2 force +4.6%, QS 1.8e-2. That is 100x the noise they remove. Do not use either.

Resolution (`resolution.py`, re-solve spread from 1e-6 noise):

| basis / grid | force floor | QS two-term spread (stock / area) | Boozer QS spread | iota spread |
|---|---|---|---|---|
| 8 / 16 | 1.15e-7 | ≤ 1e-4 / ≤ 1.2e-4 | ≤ 1.3e-4 | ≤ 1.4e-5 |
| 8 / 24 | 1.09e-7 | 5-8e-5 / 5-8e-5 | 1-4e-5 | ≤ 1e-6 |
| 10 / 20 | 1.9e-9 | 5-7e-5 / 6-8e-5 | ≤ 1.2e-5 | ~8e-7 |

The force floor is set by the BASIS, not the collocation grid. More basis quiets iota and Boozer QS but not the
two-term QS (stock or area-weighted, so the measure is not the cause either). Basis 12 OOMs the 16 GB GPU here.

**The inner solve is the problem** (`INNER=exact gauge3.py det`): re-solving the k5 equilibrium with
`lsq-composite(hessian="exact")` (Gauss-Newton plus the residual-curvature term S) from two different
perturbed starts lands on the SAME state to all printed digits (spread < 1e-7, vs 1e-5..1e-4 for GN).
It reaches a LOWER force cost (1.1241e-7 vs 1.1537e-7): the GN-"converged" k5 equilibrium was not the
least-squares minimizer. The true minimizer differs by 5.5% in QS (4.215e-4 → 4.445e-4) and 4e-4 in iota.

Interpretation: the discrete equilibrium is a unique nonzero-residual LSQ minimizer, with soft directions
(mostly relabelings) where J^T J is ~1e-7 of its max and the dropped term S dominates. Gauss-Newton does not
converge along those directions: it stops wherever gtol is met, which depends on the start and the tolerance.
That gives the outer objective's noise, and the "tighter is worse" behavior. The proximal Jacobian
`-J^+ J_c` is the zero-residual implicit derivative, and very likely also why the QS Jacobian is ~30-50% off
along GN directions. Both are fixed in principle by treating eq*(x) as the stationary point of ½|F|², i.e.
using S in the solve and (JᵀJ + S) in the implicit derivative. Cost: ~180 exact-Hessian iterations
(~10 min) per solve here, so the practical form is a hybrid (secant S, or GN then an exact finish).

### Hybrids for the inner solve (same determinism test, k5)

* `lsq-composite(hessian="secant")`: does NOT reach the minimizer (force stays 1.1536e-7 after 300 its) and its
  spread is WORSE than GN (QS 1.6-2.7e-3). The secant S is learned only along the steps taken (and gated on
  predicting better than GN), so it never sees the soft subspace.
* **Subspace Newton** (`sandbox/subspace_newton.py`, `sn_test.py`): B = JᵀJ + V(VᵀSV)Vᵀ on the k = 128 smallest
  eigenvectors of JᵀJ (k forward-over-reverse HVPs per iteration), eigenbasis trust region (More-Sorensen; S is
  INDEFINITE there: VᵀSV in [-3.2e-2, +3.4e-2] against λ_min(JᵀJ) = 8.9e-6, so a Levenberg-Marquardt shift of
  |min eig| crawls), and a paired decrease (the true decrease is below the rounding of a 1e-7 cost).
  From the GN-converged state and 2 noisy starts: force 1.124075e-7 every time, QS 4.4454817e-4 (vs the
  exact-Hessian 4.445482e-4: 6.5e-8), spread QS ~8e-8, iota ~7e-10. 182-192 its, 49-59 s per cold solve
  (exact Hessian: ~10 min). The ~180 its are the walk down the valley from the GN point; the final ~15 are
  quadratic Newton steps.

### Warm re-solves and the implicit derivative (`sn_warm.py`, `sn_deriv.py`)

From the TRUE minimizer (`eq_k5_true_minimizer.h5`) with boundary steps t × (k5_tol6 − k5):

| t | GN re-solve | subspace Newton | SN noisy-restart spread | GN vs SN (QS) |
|---|---|---|---|---|
| 1e-3 | 10 its, 6 s | 21 its, 14 s | QS 2.8e-7, iota 2e-9 | 1.6e-5 |
| 1e-2 | 28 its, 4 s | 22 its, 15 s | QS 2.9e-7, iota 2e-9 | 4.5e-5 |
| 1e-1 | 30 its, 4 s | 30 its, 16 s | QS 7.6e-8, iota 3e-10 | 8.3e-5 |

SN's QS is smooth in t (ΔQS/t = −3.90, −3.78, −3.68e-5); GN's is not, even from the true minimizer
(−3.19e-5 at t = 1e-3). Warm SN costs ~3x a GN re-solve.

**Correction:** the proximal's ZERO-residual implicit derivative is accurate at the true minimizer:
dQS/dt = −3.674e-5 (−3.688e-5 with DESC's spectral shift) vs −3.7..−3.9e-5 from converged re-solves. The
~30-50% QS "Jacobian error" measured earlier was against noisy GN re-solves, i.e. the noise again. A boundary
change barely excites the soft (relabeling) modes (F_c ⟂ them), so the S term does not matter for the
derivative. Including it via B^-1 only amplifies finite-difference error along B's near-zero eigenvalue
(−2.2e-2, wrong). The "wrong sign" seen in `sn_warm.py`'s perturbed state came from `eq.perturb`'s default
`include_f=True` (a GN correction −J⁺F that pushes x* back along the soft modes). ProximalProjection uses
include_f=False. Also retracted: the `sf += sf[-1]` shift is irrelevant here (−3.688 vs −3.674e-5).

**Root cause, final:** the inner GN solve does not converge along the soft modes. Fix: replace the proximal's
inner `eq.solve` with subspace Newton; keep the existing proximal Jacobian.

### Subspace Newton inside the proximal loop (`--inner-sn 128`, `install_proximal_hook`)

Endpoint probe (`probe_end_sn`), same steps as with GN:

| SD step | GN inner ratio | SN inner ratio |
|---|---|---|
| 1e-5 | −0.54 | **1.0001** |
| 1e-6 | −23 | **0.91** |
| 1e-7 | −318 | 0.14 |

The noise floor drops ~350x (cost ~7e-6 → ~2e-8; the remaining floor matches SN's stopping tolerance). The
build moves the endpoint to the true minimizer (181 its, 53 s); its cost there is 1.636, not 1.620. All GN-scored
single-stage results are flattered by ~5% in QS.

**Narrower Jacobian conclusion:** along steepest descent QS matches its model to 6%, but along the GN direction it
is off by a consistent ~2x (8.98e-5 actual vs 5.61e-5 predicted at both t = 1e-4 and 1e-5), and iota has
the wrong sign. That is systematic, not noise. The zero-residual proximal derivative is accurate along the
boundary direction tested in `sn_deriv.py`, but not along every direction (plausibly those that excite soft
modes). Open; a proper full implicit derivative needs a better-conditioned formulation than B^-1 with FD
mixed terms.

### k5 with the subspace-Newton inner solve (`k5_sn/`, lsqtr, maxiter 40)

Passes the original stall point (iteration 11 accepted on the first evaluation, vs 10 evaluations and
collapse), but stops on xtol at iteration 20 at the same place (cost 1.620, optimality 0.0397). Step log: the
REJECTED trials have a near-constant actual reduction of −0.0135 … −0.0158 at steps from 1.5e-2 down to 2.4e-4
(iterations 8, 9, 12), each with a converged inner solve (precision floor). A deterministic, step-independent
jump of ~1% of the cost to a well-converged state = the re-solve lands in a DIFFERENT LOCAL MINIMUM of the
force least-squares problem. Consistent with S indefinite in the soft subspace (non-convex valley), B_min ~ 6e-6,
and the huge full implicit derivative (|dx/dt| ~ 12): the solution path sits near a fold. (The ~8.5e-9
jumps at iteration 19 are the SN tolerance floor.)

Picture: (1) GN never converged along the soft modes, giving the 1e-5-level noise; SN removes it. (2) The
discrete fixed-boundary equilibrium at L=M=N=8 is not unique: the near-gauge valley has several local minima
with near-equal force residual and different QS. With a converged inner solve f(x) is smooth within a basin
and JUMPS between basins, and the stall persists. Pins pick a basin by distorting physics; loose tolerances
avoid switching only by not converging.

Principled fix candidates: an EXACT gauge condition as constraint rows (one per gauge dof; VMEC-style spectral
condensation on R, Z, or FixThetaSFL), and/or higher basis resolution (the basins differ by truncation error).
Unconfirmed: save two trial equilibria from opposite sides of a jump and compare force residuals and
soft-mode coordinates.

### The two branches: a fold, the same continuum field, and a discretization artifact (`basin_compare.py`)

Endpoint probe from `k5_sn` (SN inner solve), steepest-descent steps, perturbation order 2 vs 0 (`basins_po*`):
IDENTICAL results, so the perturbation predictor is NOT the cause. Steps ≥ 3e-4 (scaled) all take ~180 SN its and
land on the same state (QS-term cost +1.75e-2, iota-term −1.67e-3, identical for all four); steps ≤ 1e-4 stay
(12-18 its) and match the model. That is a FOLD (saddle-node): the base branch ceases to exist between 1e-4 and 3e-4.

Two branches at k5:
* A (base; also the original GN k5 state): force 1.153671e-7, QS 4.2157e-4. A converged local minimum (SN: 3 its).
* B (the "true minimizer" found earlier): force 1.124095e-7 (lower), QS 4.4455e-4 (+5.5%).

Same boundary: T' (B's interior, A's boundary) stays on B (20 its); B' (A's interior, B's boundary, 1.1e-7 away)
walks to B (192 its). Two distinct minima for one boundary, and the optimizer sits within ~1e-7 of A's fold.

| A vs B, same boundary | basis 8 | basis 10 (both re-solved, 600-it cap hit) |
|---|---|---|
| displacement rms | 9.0e-4 m (tangential 8.9e-4, normal 1.6e-4) | 4.3e-4 m (tangential 4.33e-4, normal 1.9e-5) |
| relabeling fit of dlambda | 0.71 | 0.93 |
| QS rel. difference | 5.5% | 0.47% |
| Boozer QS rho 0.25 / 1 | 1.0% / 2.6% | 0.26% / 0.12% |
| iota max difference | 4.0e-4 | 2.7e-4 |

The force residual does NOT localize at iota = 1.2 (rho ≈ 0.38). It rises smoothly to the edge (3.2e3 at rho = 1
vs ~350 mid-radius) at both resolutions, so this is not an island/resonance effect. At basis 10 both states have
QS 4.22-4.24e-4: the LOWER-force branch B (4.45e-4 at basis 8) is the WORSE approximation of the converged physics.

**Conclusion.** A vacuum field inside a fixed boundary with given flux is unique; the only continuum family is
the theta relabeling (physics-invariant). The branches are that one field in two gauges, made physically distinct
(~5% QS) by truncation at basis 8; with resolution they converge to pure relabelings. Each branch is stable where
it exists, but A ends at a fold, and the single-stage optimizer drives the boundary to that edge because A's
gauge makes QS look better: **it is optimizing gauge-dependent truncation error.** Fixes: an exact gauge condition
(one constraint per gauge dof: spectral condensation, or FixThetaSFL) removes the branch choice and the folds.
Resolution shrinks the stakes (5.5% → 0.5% from basis 8 to 10) but does not remove them. Meanwhile, judge
single-stage gains at basis ≥ 10.

### Gauge condition: equal-arclength spectral condensation (`GaugeCondition`, `gauge_fix_test.py`)

Rows: Galerkin <phi_k, R_t R_tt + Z_t Z_tt>/a^2 for the m != 0 lambda-basis functions (the stationarity of
W = ∮|X_t|^2 along continuum relabelings, i.e. equal arclength), plus lambda_{m=0} = 0 for the theta shift.
Same-boundary test from branch A and branch B:

| weight | A vs B starts | force | QS | Boozer 0.25 / 1 | iota | abs(G) |
|---|---|---|---|---|---|---|
| unfixed A | | 1.1537e-7 | 4.216e-4 | 2.109 / 7.985e-3 | [1.1807, 1.2038] | 0.620 |
| unfixed B | | 1.1241e-7 | 4.445e-4 | 2.130 / 8.191e-3 | [1.1811, 1.2035] | 0.615 |
| 1e-3 | identical (1e-8) | 1.1301e-7 | 4.454e-4 | 2.123 / 8.196e-3 | [1.1812, 1.2031] | 0.613 |
| 1e-2 | identical | 9.7e-7 | 5.23e-4 | 1.727 / 9.114e-3 | [1.1817, 1.1895] | 0.566 |
| 1e-1 | identical | 1.8e-4 | 9.6e-3 | broken | [0.985, 1.402] | 0.413 |

Uniqueness at every weight; neutrality fails as the weight grows. abs(G) barely moves: the equal-arclength gauge is
NOT reachable in the basis-8 space for the shaped QH cross-sections, so enforcing it pulls the physics (criterion
1 holds in the continuum, fails discretely). At 1e-3 it is a tie-breaker: unique, physics inside the A/B
truncation spread, force +0.5% over B.

Lesson: in a truncated basis the gauge orbit is not a symmetry; ANY gauge choice selects among states that differ
by truncation error (~5% QS here). A gauge condition can make the equilibrium deterministic, not accurate;
accuracy needs resolution. A good discrete condition should be (1) satisfiable in the basis: stationarity of a
spectral width along the DISCRETE gauge directions (relabeling generators projected onto the basis), and (2)
select the most compactly representable gauge (VMEC's normalized m^4/m width). Options: A) the 1e-3
tie-breaker in the loop (fold probe + k5); B) discrete-consistent condensation; C) basis 10-12 (memory work).

## Higher resolution (2026-10-08/09 overnight)

**Memory.** Basis 12 (L=M=N=12, grids 24; dim_x 4074, 16 562 force rows) fits easily once the ForceBalance Jacobian
is chunked: peak 2.47 GB at jac_chunk_size 400, 1.61 GB at 100, same speed (~1.6 s per GN iteration). The basis-12
OOM was DESC's "auto" chunk (all columns in one batch). `single_stage.py --force-chunk N` sets it for the proximal
ForceBalance; the full proximal at basis 12 peaks ~2-5.5 GB. Proximal Jacobian 33 s, a GN re-solve evaluation
156-188 s (`res_mem.py`, `probe_b12`).

**Plain GN at basis 12 does not fix the inner solve.** The endpoint probe at tol 1e-10 shows a constant actual
reduction of ~1.1e-4 at every step size (ratios 3.6, 27, 261 at 1e-4, 1e-5, 1e-6): the basis-12 start (300 GN its,
maxiter) is not converged along the soft modes and every re-solve keeps sliding.

**But at basis 12 the gauge valley barely changes the physics.** The soft modes are ~1000x softer (min eig of JᵀJ
~1e-8 vs 9e-6 at basis 8), so SN walks a long flat valley (400 its, not converged). Along it: QS −2e-4 rel,
Boozer ~1e-4, iota 1e-5. The gauge-induced physics spread falls 5% (basis 8) → 0.5% (10) → ~0.02% (12).

**Fair re-score at basis 12** (`rescore_all.sh`, `rescore12/summary.txt`; same objective and weights, each run's coils):

| run | cost scored at basis 8 | at basis 12 | B.n | Boozer QS 0.25/0.5/0.75/1 (e-3) |
|---|---|---|---|---|
| k4b start | 1.628 | 1.634 | 6.823e-3 | 2.08/3.65/4.92/7.87 |
| original k5 (= k5_sn) | 1.620 | 1.625 | 6.809e-3 | 2.08/3.65/4.92/7.87 |
| k5 tol 1e-6 | 1.493 | 1.496 | 6.481e-3 | 1.95/3.49/4.86/7.84 |
| k5 composite + secant | 1.430 | 1.421 | 6.295e-3 | 1.87/3.18/4.37/7.34 |
| k5 pinned composite | 0.916 pinned / 1.004 | 0.9955 | 5.495e-3 | 2.12/2.21/3.30/7.51 |

All basis-8 gains survive at basis 12 (within ~0.01; ranking unchanged). Basis-8 truncation shifts the total cost
slightly and roughly uniformly, so it hampered the optimization DYNAMICS (noise, folds) without biasing the
end results much. The pinned composite's path ran through non-equilibria, but its final boundary + coils are
genuinely good (cost 1.634 → 0.996 at basis 12, B.n −19%, mid-radius QS −40%): a biased search can find good
designs, provided the end state is validated at high resolution.

### Basis-12 k5 run (the recommended recipe) and endpoint probe

`k5_b12_comp` (60 its) + `k5_b12_comp2` (80 its): composite + secant, solve-tol 1e-6, force-chunk 200, from the
basis-12 k4b state. Cost 1.634 → 1.035 → 0.818 (still descending). B.n 5.120e-3, Boozer QS 0.91/1.41/2.99/6.67e-3,
A 8.316, iota [1.1854, 1.2008]. Tight re-solve at basis 12: 0.8182; basis 14: 0.8183. Endpoint probe
(`probe_b12_end`): ratio 0.9998 down to 1e-7 SD steps; QS cost matches the model to ~1% along SD and GN. The basis-8
pathologies (noise, folds, the 2x Jacobian error) are all gone at basis 12. Summary: `../results/K5_FINDINGS.md`.
