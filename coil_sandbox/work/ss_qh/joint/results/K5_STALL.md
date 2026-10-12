# Open issue: the k = 5 single-stage step stalls (trust-region collapse)

**Resolved 2026-10-09: summary and takeaways in `K5_FINDINGS.md`; lab notebook `../stall/DIAGNOSIS.md`.** Earlier note: The proximal re-solve drifts along the unfixed theta gauge, so
QS/iota jump by a step-independent amount at every trial point (5e-6 at solve-tol 1e-10); the stall is that noise, not the
coil hinges. `--solve-tol 1e-6` clears it (cost 1.620 → 1.493); `--method lsq-composite --hessian secant` reaches 1.430.

Not investigated, on purpose: the joint-plane study moved on (2026-10-08). This note is for a session that wants to
make `single_stage.py` converge better. The same family of early stops shows up in the matrix
(`../../matrix/RESULTS.md`: k steps stopping on ftol 1e-6), so a fix here probably helps every single-stage run.

## What happened

`../ss_z0.sh k5` (log `../ss_z0_k5/run.log`): proximal-lsq-exact, precise_QH-derived boundary, arcB2 M5 with every hinge
fixed at z = 0, boundary modes |m|,|n| ≤ 5 freed (eq/boundary resolution M = N = 8), started from the end of k4b.
Flags: `--kmin 5 --kmax 5 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12` plus the matrix arcB2 weights
(`QH`, `QHSS`, `ANCH` in `../ss_z0.sh`).

```
Iter  nfev   cost       reduction   step norm   optimality
 0     1    1.628e+00                           2.505e-01
 1    10    1.627e+00   7.4e-04     3.7e-04     2.486e-01   <- 9 rejected steps before the first acceptance
 2-10 11-22 ...         ~1e-4..2e-3 1e-5..4e-4  6.8e-02 -> 3.95e-02
11    32    1.620e+00   4.4e-10     6.0e-11     3.953e-02   <- 10 evaluations, step 6e-11
12    35    1.620e+00   5.5e-11     7.5e-12     3.953e-02   -> stops on xtol 1e-12
```

Optimality stayed at 0.04 (not a minimum; k4b ended at 0.028–0.029), and B·n moved only 6.82e-3 → 6.81e-3. There was
no equilibrium-solve failure in the log. Cost is 76% B·n and 22% QS. Several coil terms are weighted one-sided
penalties sitting exactly on their bounds:

| term | value | bound |
|---|---|---|
| L | 2.928 | ≤ 2.926 |
| κ | 13.61 | ≤ 13.6 |
| d_cc | 1.001 | — (penalty nonzero on a few pairs) |

## A related symptom at the end of k4b (k = 4, the step before)

`../ss_z0_k4b/run.log`, last iterations: every step is accepted, and the step norm is pinned at ~3.4e-3. Cost falls by
~7e-4 per iteration, and optimality alternates 2.75e-2 / 2.93e-2 on every step. That is a zig-zag: a narrow valley, or an
active-set flip of a one-sided penalty from step to step. It is not convergence.

## Hypotheses (none tested)

1. **One-sided penalties at their bounds.** The squared hinges `max(0, g)²` on L, κ and d_cc have a kink in their
   second derivative at the bound. The Gauss-Newton model cannot see a term switching on, so steps are predicted well
   and realised badly. That fits both the zig-zag and the rejections. The old `lsq-auglag(second_order="constraints")`
   cleared a stall of this kind in stage 2 (CLAUDE.md, "degenerate at the bound").
2. **Proximal model mismatch.** The predicted reduction uses the perturbed equilibrium (`perturb_order 2`). The actual
   reduction uses the re-solved equilibrium (solve_tol 1e-10). At iteration 11 the reductions are ~1e-10, so the
   solve's own noise can decide acceptance. That explains the final collapse but not iteration 1.
3. **Scaling of the newly freed modes.** The |m| or |n| = 5 boundary modes start at ~0, with Jacobian-based scaling.
   The first step may be dominated by them, which would fit the 9 rejections on iteration 1.

**Solver context (checked 2026-10-08).** The single stage runs `proximal-lsq-exact`, which is plain Gauss-Newton:
no secant or exact second-order term. Its coil limits are weighted squared hinges inside one least-squares cost.
Stage 2 (`run_al.py`) uses `lsq-auglag-composite`, whose subproblems default to `{"hessian": "secant", "finish_steps": 5}`,
with the coil limits as augmented-Lagrangian rows. So stage-2 runs reaching optimality on the same coils does NOT
clear the single-stage coil terms: they are a different formulation, solved without the curvature defence. The optimizer
accepts any `proximal-<method>`, so `proximal-lsq-composite` with `options={"hessian": "secant"}` is a candidate; it is
untested with ProximalProjection, and `single_stage.py` has no switch for it yet.

## Cheap diagnostics, in order

* Rerun k5 under `proximal-lsq-composite` with `hessian="secant"` (needs a solver switch in single_stage.py).
* Rerun k5 with `--fix-coils`. If the equilibrium alone converges, hypothesis 1 (coil penalties) is the prime suspect.
* Rerun k5 with the coil bounds loosened by 1%, or the L/κ terms dropped. Does the stall go away?
* Turn up solver verbosity, or log the actual-to-predicted ratio per evaluation. Do the rejections come from the
  equilibrium or the coil side?
* `--perturb-order 1`, and a looser or tighter `--solve-tol`. These test hypothesis 2.
* Repeat k = 4 from k4b's end with the coils frozen. Does the zig-zag persist?

Files: `../ss_z0_k5/` (run.log, eq_k5.h5, coils_k5.h5, result.json), `../ss_z0_k4b/`, `../ss_z0.sh` (functions `k4b`,
`k5`). The k5 coils were refined afterwards (`../on_z0_k5/`): 6.356e-3, against 6.372e-3 after k4b.
