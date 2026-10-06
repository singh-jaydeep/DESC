# Stage-2 coil "crawl" — handoff

Written at the end of the diagnosis session (2026-10-05). Read this before trying the
methods on another stalled coil optimization.

**Branch:** `js/demo-stage2notebook`. Commits on top of master:

| commit | what |
|---|---|
| `3392c5c73` | `scratchpad_crawl/`: prototypes and diagnostics |
| `46db87173` | `CoilCurvature(signed=...)`, lsqtr logging of trust radius / alpha / cond(J), notebook |
| `2d4299412` | README pointer |

**Write-up for humans:** https://claude.ai/artifact/QXwrvAovHM2mcjqoyZ1JHz (full numbers,
charts). This file is the agent-facing version: what to run, when, and what to watch for.

**Status:** the precise_QA notebook case is solved and certified (gradient ~1e-7 raw,
1e-11 scaled, positive definite Hessian). Nothing has been upstreamed into `desc/`; all
fixes are prototypes in `scratchpad_crawl/`. Not yet tested on any other case.

---

## 1. The problem and the motivation

The first coil optimization in `docs/notebooks/tutorials/coil_stage_two_optimization.ipynb`
(precise_QA, 3 FourierXYZ coils, N=10, lsq-exact) "converges" on ftol while the trust
radius collapses, then crawls forever: optimality ~1e-2, zig-zagging. The user wanted a
certificate of a tiny gradient. Their earlier attempts (unsigned curvature, arclength
variance, finer grids, curvature target) did not fix it.

## 2. What was found

Four independent defects, each hiding the next. All four are needed; the exact Hessian is not.

| # | defect | kind | fix (prototype) |
|---|---|---|---|
| F1 | `CoilSetMinDistance` is a hard `jnp.min` over point pairs; it was active and sitting at a tie between pairs | non-smooth objective | `CoilSetPairDistance`: every neighbour point-pair distance as its own hinge residual |
| F2 | `CoilArclengthVariance` is one residual per coil (r = Var), so the cost is Var², quartic; J is rank 1 per coil and vanishes at the target | Gauss-Newton blind to its curvature | `CoilArclengthResidual`: r_i = w(\|x_s\|_i - mean)/sqrt(N), sum of squares = Var |
| F3 | bounds are `max(0,·)` hinges; lsqtr linearizes the hinged residual, so inactive rows have zero Jacobian rows and new activations break the model | solver's local model | composite model: linearize the pre-hinge value, keep the hinge exact (`ctr.py comp`) |
| F4 | FourierXYZ has an exact gauge s -> s+c (phase of the curve parameter) even with shift/rotmat fixed; arclength variance is invariant to it | gauge | one `FixParameters` per coil on the n=±1 coefficient with the largest phase-tangent component |

Order in which they surfaced: hard min -> curvature-hinge activations (+ scalar arclength) -> phase gauge -> converged.

Key evidence (details in the write-up):
- Baseline stall: trust-region model mismatch at Δ=1 was +5.5e-2 from coil-coil vs ≤1e-5 from everything else. Ties: coil 0 at 0.051298 m (coil 9) vs 0.051447 m (coil 11).
- Scalar arclength: along the weakest J directions, JᵀJ curvature 1e-6..1e-3 vs dropped term S = 600..4200.
- Composite vs standard model, same loop, same start, 300 its: cost 1.370 -> 0.5891 (flat) vs 1.370 -> 1.124 (sliding).
- Phase gauge: phase-tangent curvature 1e-7..1e-5 vs max eigenvalue 58; after pinning, plain composite GN took |g| 0.80 -> 2.2e-6 in 60 its. The gradient was real, not a gauge artifact.
- From the circular initial coils, all fixes (run F1): cost 4.5e6 -> 0.15006, |g|∞ 1.8e-7, scaled optimality 1.1e-11, no bound violated, 700 its, ~1.7 s/it.
- Same but scalar arclength (run F2): not converged after 700 its (optimality 1e-4..5e-4).
- Gauge fix alone on the original notebook objectives (run GA): same crawl as baseline.

Ruled out: monotone jac scaling in `compute_jac_scale` (fresh restart crawls identically);
plasma-coil distance and coil length were never active; exact-Hessian Newton alone crawls;
softmin coil-coil (α=500) still throttles; curvature *target* is a large-residual problem;
softplus curvature hinge and a Sobolev mode penalty help little or not at all.

## 3. How to tell which failure you have (from the lsqtr log, verbose=3)

| signature | likely cause | next step |
|---|---|---|
| alpha > 0 almost every iteration, trust radius parks at a small value, steady tiny cost decrease, period-2 zig-zag in optimality | hinge or kink (F1/F3) | run `trsteps.py`; the objective carrying the mismatch is the culprit |
| as above, and the culprit is a min/max-type objective at a tie | hard min (F1) | per-pair/per-point hinges; check ties with distances of the top pairs |
| alpha/λ tiny, steps accepted, cost flat to 6–7 digits but creeping, raw \|g\| bounces instead of decreasing | gauge (F4) | `hspec.py` (Hessian spectrum), test known symmetries with `gauge.py`-style tangent overlap |
| one objective with few rows and a residual far from zero carries the hidden curvature | aggregated residual (F2) | split it into per-node/per-pair signed residuals |
| restart with fresh scaling gives the same collapse | not a scaling-history problem | — |

Rule of thumb for objectives under Gauss-Newton: pass signed residuals that cross zero at
the solution, one row per condition. Never pass a residual that is itself a sum of squares,
variance, energy or mean-squared quantity. A hard min/max over points is a kink. This only
matters for GN/LM solvers (lsq-*); quasi-Newton on a scalar objective does not care.

## 4. The code (`scratchpad_crawl/`, see its README.md)

Run with `/Users/singh/PycharmProjects/DESC/.venvDEV/bin/python` (default `python` in the
user's shell). Run scripts from inside `scratchpad_crawl/`.

- `setup.py` — `build(**kw)`: the notebook's precise_QA problem with switches
  `cc` (`hard`/`soft`/`pairs`), `arclen` (`scalar`/`vector`/`none`), `curv`
  (`bound`/`target`/`soft`), `reg_w`, weights, grids. Also defines the prototype objectives
  `CoilSetPairDistance`, `CoilArclengthResidual`, `CoilCurvatureSoftHinge`,
  `CoilModeRegularization`. **This builder is hard-wired to the notebook case.** For a
  new case, write a new builder with the same signature/return (`eq, coilset, obj, cons`)
  and reuse the prototype classes.
- `run.py TAG [--cc pairs --arclen vector --arclen_w 5 --phase --start X.h5 --maxiter N]` —
  lsqtr driver. Writes `TAG.json` (the kwargs), `TAG.h5`, `TAG_allx.npy`.
- `ctr.py ARM START_TAG MAXITER OUT [YFILE]` — the controlled trust-region loop.
  ARM: `std` (lsqtr-like linearization), `comp` (composite hinge model), `comp2`
  (composite + exact second-order term, ~n HVPs per iteration, slow). START_TAG supplies
  the problem config via `START_TAG.json`. With `PHASE=1` it adds the phase constraints and
  YFILE may be a full state `*_x.npy`; `KW='{"arclen":"vector"}'` overrides config keys.
  The full fixed run from scratch was:
  `PHASE=1 KW='{"arclen":"vector","arclen_w":5}' python ctr.py comp P2b 700 F1 init_x.npy`
  (`init_x.npy` = packed initial coil params; P2b.json = pairs + vector arclength + curvature bound).
  Writes `OUT_hist.npy` (it, cost, |g|∞, scaled opt, λ, t), `OUT_y.npy`, `OUT_x.npy`.
- `trsteps.py TAG` — trust-region ladder at the end of a `run.py` run; prints the
  reduction ratio and per-objective model mismatch for Δ = 0.01..100.
- `hspec.py TAG YFILE` (or `PHASE=1 ... X_FILE`) — Hessian spectrum (jac-scaled) and Newton
  decrement ½gᵀH⁻¹g. Use it for the final certificate.
- `gauge.py` — phase-tangent finite-difference check and overlap with the flattest
  eigenvectors. Paths are hard-coded to P2b/Ccomp2; adapt before reuse.
- `newton.py TAG OUT MAXITER [YFILE]` — scipy trust-krylov exact-Hessian polish. Slow, and
  it stalls on hinges; prefer `ctr.py comp`.
- `common.py` — `load_run`, `load_phase`, `phase_tangent`, `groups`.

Run outputs (`*.npy`, `*.h5`) are not committed; they are on disk in the original checkout
only. Regenerate as needed.

## 5. Applying this to a new stalled case — suggested procedure

1. Reproduce with lsqtr `verbose=3` and save `allx` (write a builder in the style of
   `setup.build`, use `run.py`). Note which bounds are active at the end (compare
   `compute_unscaled/normalization` to the bounds; the summary table prints max/min of
   per-coil values and can hide a hinge sitting at its bound).
2. Read the log against the table in §3. Run `trsteps.py` on the endpoint.
3. Look for, in this order: hard min/max objectives that are active (`CoilSetMinDistance`,
   `PlasmaCoilSetMinDistance`, any `jnp.min/max` in a compute), aggregated scalar residuals
   (`CoilArclengthVariance`, anything returning a variance/energy/integral of a square per
   coil), active bounds, and gauges of the coil representation.
4. Fix what applies, then run `ctr.py comp` (composite model). Add `PHASE=1` for FourierXYZ.
5. Certify with `hspec.py`: positive definite Hessian, small Newton decrement, small raw
   gradient. A bouncing raw gradient with a flat cost means a gauge is left.

Representation notes: the phase gauge was derived and verified for **FourierXYZ** only.
`SplineXYZCoil` likely has an analogous reparameterization freedom but was not tested.
`FourierPlanarCoil`/`FourierRZCoil` have a unique parameterization for arclength purposes
(CoilArclengthVariance returns 0 for them); do not assume a phase gauge there without
checking (build the tangent and test overlap as in `gauge.py`).

## 6. Pitfalls hit in this session

- **Memory:** the box is CPU-only with 8 GB. Two parallel runs with the per-pair objective
  crashed the machine. Run one heavy job at a time; keep `jac_chunk_size` set (setup.py
  uses 16). The per-pair objective at pair grid N=30 has ~15k rows.
- **Scratch files:** `/private/tmp` was wiped by the crash. Keep work in the repo.
- **Normalizations:** coil objectives compute their normalization from the coilset at build
  time. To evaluate or restart at another point, build on the *initial* coils first, then
  load params (see `run.py --start` and `common.load_phase`). Building on the optimized
  coils gives wrong costs (27270 instead of 18.73).
- **`FixParameters` targets** are taken at build time: set the coil params before building it.
- **Shell:** `pgrep -f "python run.py X"` matches its own waiting shell; zsh does not
  word-split `$var` for argument lists.
- **pre-commit:** flake8 with docstring checks runs on `scratchpad_crawl/*.py`; black
  reformats. Keep new scripts lint-clean rather than bypassing hooks.
- `ctr.py` uses a Levenberg-Marquardt λ rule (Nielsen), not lsqtr's radius rule; the
  composite model has not been put inside `lsqtr` itself.

## 7. Open questions

- Does the composite model alone fix a hard-min objective if the min is exposed as separate
  pieces? Not tested.
- Would a structured quasi-Newton S (NL2SOL-style) give most of the exact-Hessian speedup cheaply?
- Upstreaming: composite step in `lsqtr` (`least_squares.py` ~318–347, pre-hinge values from
  `ObjectiveFunction.compute_scaled`, bounds from `_Objective._shift`); per-pair mode for
  `CoilSetMinDistance` (`_coils.py:1056`); per-node arclength (`_coils.py:1600`); phase
  constraint in `maybe_add_self_consistency` (`getters.py:334`).
- Normal-field metrics (⟨|B·n|⟩/⟨|B|⟩) were never computed for the final run F1.
