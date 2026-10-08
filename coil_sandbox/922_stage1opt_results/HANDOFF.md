# Handoff: stress-testing the eps_1 interpretation

> ## STATE CHANGED 2026-09-23 (later) -- READ `NOTES.md` 9o FIRST
>
> **The agenda in section 3 below is largely answered, and not in eps_1's favour.**
>
> Four resolution-converged runs with the constraints VERIFIED (not assumed):
>
> | eps_1 | A | iota in bound? | L_gradB/a min | pdrot*a median |
> |---|---|---|---|---|
> | **-10.31%** | 6.010 | yes | **+7.77%** | **-9.26%** |
> | -27.21% | 6.251 | - | -23.55% | +65.63% |
> | **-32.91%** | **6.010** | **yes** | **-23.55%** | **+13.52%** |
> | -33.78% | 6.010 | leaked +0.013 | -25.81% | +12.02% |
>
> **The coil-complexity damage tracks the SIZE of the eps_1 reduction, not any constraint.**
> Aspect ratio and iota were both investigated as confounds and both ruled out: the -32.91% run
> holds aspect at 6.010 and iota inside its baseline-referenced bound and is damaged just as
> much as the loose one. Optimizing eps_1 makes the boundary demand coils that are CLOSER
> (min L_gradB -24%) and LESS PLANAR (pdrot up) -- the opposite of the intent.
>
> Confirmed at stage 2 (matched equilibrium quality, identical coil problem, baseline control):
> **planar -0.45% (unchanged), free coils +17.32% worse, ratio -15.15%.** Planar coils never
> improved in ANY run at ANY eps_1. Section 3.1's null is answered on the numerator side --
> 9l's planar degradation was equilibrium quality -- but the ratio still moves only through the
> denominator.
>
> **Also required reading before trusting any number here:**
> * **No stage-1 run has ever converged** (9n.4). Every eps_1 figure is a lower bound.
> * The -10.31% run, the only one whose metrics improved, was TRUST-REGION limited, not
>   physics limited (9o). A possible benign regime below ~10% is *consistent with* the data but
>   NOT established -- it is one point from a stalled run. `--target-frac 0.85/0.80` tests it.
> * Sections 3.3 (QS confound) and 3.4 (B-ladder) remain untouched. Section 5 (the C question)
>   is unsafe: the eps_1/C tension it rests on is not reproducible across constraint settings
>   (C moved -3.13%, +9.59%, +14.16% and +24.61% on runs reaching similar eps_1).
>
> **UPDATE (evening): the objective was asking for the wrong thing -- see `NOTES.md` 9p.**
> `--target-mode worst` (one-sided bound on the WORST contours) + an `L_grad(B)` floor gives
> **better coils and a better equilibrium together**: eps_1 -17.8% with planar coils -9.23%
> (0.519 %/point vs proportional's 0.450), min L_gradB -1.81% (vs -25.81%), pdrot median
> -11.85% (vs +12.02%), foliation spread 1.73x (vs 3.43x), and BOTH coil representations stay
> in the baseline's constraint regime. That is the live direction.
>
> **Still unexplained and it matters: only 1 stage-1 leg in 3 is productive**, with identical
> configuration and trust radius. Four mechanisms were proposed and falsified in one session.
> **Do not conclude "configuration X does not work" from a single stalled run.**
>
> **Cheapest next test, no GPU:** QUASR and Constellaration ship equilibria WITH optimized
> coils, so eps_1 / pdrot / L_gradB can be regressed against ACHIEVED coil complexity (length,
> curvature, eta_SVD) over thousands of devices. If eps_1 adds nothing over pdrot (R^2 = 0.700
> univariate against coil non-planarity), the question is closed. This inverts 5c's failure,
> which was 8 hand-picked configurations with no dynamic range.

---

## 1. What exists

### Code (`922_stage1opt_results/`)

| file | what |
|---|---|
| `phi_contours.py` | current potential on rho=1; `K_vc` -> `Phi` -> contours. T0-verified |
| `clamshell.py` | differentiable `eps_B` (Danskin, frozen segmentation AND plane normals) and `C = eps_2/(eps_1/4)`. `python clamshell.py` self-tests; grad vs FD 7.8e-10 |
| `contour_curve.py` | **`ContourClamshell`**, the DESC objective. Contours built INSIDE it, so the curves are exact streamlines and equal-current by construction |
| `stage1_v1.py` | the stage-1 driver (`--quantity C\|eps1`, `--arm`, `--from`, `--eq-res`, bounds flags) |
| `analyze_result.py` | audit: constraints, frozen-vs-fresh eps_1, numerator/denominator gaming check, proxy validity, boundary diff, C |
| `geometry_why.py`, `plot_streamlines.py` | why eps_1 changed, geometrically |
| `resolution_check.py` | eps_1/C/force-balance vs spectral resolution |
| `run_coil_validation.sh` | the 6 stage-2 runs; resumable, retries the CUDA fault |
| `view_{baseline,leg1,leg2}.html` | **offline 3D viewers**, one per equilibrium: plasma surface + planar result + FourierXYZ result + the common circles start |

### Vendored DESC

Two PRs merged into `vendor/desc`, both **additions only** against the vendored tree,
both verified (`check.py` passes; `polarB2_M5_cut30` re-evaluates to 1.044989e-2 unchanged):

* **#2285** `SurfaceCurve` -- `FourierRZSurfaceCurve/Coil`, `SurfaceCurveConsistency`.
* **#2298** proximal + augmented Lagrangian -- `ProximalState`; this is what makes
  `proximal-lsq-auglag` work with nonlinear constraints.

`vendor/desc_changes_vs_master.patch` is **stale**; regenerate before relying on it.

### Equilibria (the eps_1 ladder)

| name | path | eps_1 (L=8) |
|---|---|---|
| baseline | `precise_QA` (desc.examples) | 0.06763 |
| leg 1 | `runs/loose_eps1_L1/eq_lo.h5` | 0.05071 (-25.0%) |
| leg 2 | `runs/loose_eps1_L2/eq_lo.h5` | 0.03959 (-41.5%) |

Produced by minimising eps_1 with QS DROPPED and loose iota/aspect bounds. They are
**materially different devices**: QS rms 583x / 1880x worse, iota drifted to 8.0e-2,
minor radius -1.7%, volume -1.3%.

---

## 2. THE RESULT, AND WHY IT IS PROBABLY NOT A VALIDATION

Six stage-2 runs, identical coil problem throughout (absolute bounds, 16 coils, r/a 2.5
cold start from circles, no warm starts, `convex_excess: null` so planar and FourierXYZ are
judged alike, `--vacuum`).

| equilibrium | eps_1 | bn_planar | bn_free | ratio |
|---|---|---|---|---|
| baseline | 0.06763 | 6.110e-2 | 1.331e-2 | **4.590** |
| leg 1 | 0.05071 | 6.303e-2 | 1.575e-2 | **4.002** |
| leg 2 | 0.03959 | 6.470e-2 | 2.087e-2 | **3.100** |

Monotonic, `ratio ~ eps_1^0.73`, Pearson r = +0.97. **Do not report this as a validation.**
Decomposed:

```
bn_planar   6.110e-2 -> 6.470e-2    +5.9%   planar coils got WORSE
bn_free     1.331e-2 -> 2.087e-2   +56.8%   free coils got MUCH worse
ratio          4.590 -> 3.100      -32.5%   driven ENTIRELY by the denominator
```

**We did not improve planar coils. We degraded unrestricted coils more.** And the free
optimum barely moved toward planarity: eps_1 of the optimised FourierXYZ coils went
0.23044 -> 0.21868, only **-5.1%** for a -41.5% change in the equilibrium's eps_1.

Two readings, and the data does not yet separate them:

* **(A) Real.** The equilibrium became one where 3D freedom buys less -- genuinely closer
  to planar-compatible.
* **(B) Artefact.** Making an equilibrium harder compresses ALL coil types toward a common
  floor, so the best-performing representation loses the most and every ratio falls. This
  would happen for any degradation, with nothing to do with planarity.

**(B) is the null hypothesis and it has not been tested.** Everything below is about
killing or confirming it.

---

## 3. THE AGENDA, in priority order

### 3.1 The null test -- do this first, it can end the project

Degrade the baseline by an amount that raises `bn_free` by ~57%, in a way that has NOTHING
to do with eps_1, and measure the ratio. Candidates, cheapest first:

1. **Random boundary perturbation** of the same magnitude as leg 2 (|dR|max ~ 4% of a),
   several draws, re-solved.
2. **Degrade QS deliberately** (maximise QS error to match leg 2's 1880x) while holding
   eps_1 at baseline -- this also attacks the confound in 3.3.
3. **Maximise eps_1** (`--arm hi`, `--target-frac` > 1). If the ratio falls here TOO, the
   ratio is just a difficulty gauge.

**If any of these reproduces a ~30% ratio drop, reading (B) wins and eps_1 as we measure it
is not a planarity proxy.** That is a publishable negative and worth knowing fast.

### 3.2 Matched-difficulty comparison

The fixed-budget design confounds "better for planar" with "harder overall". Instead give
each equilibrium whatever budget it needs to reach a COMMON `bn_free` (tune the length cap,
or coil count), then compare `bn_planar` at matched free performance. If leg 2 still wins,
reading (A) survives a much harder test.

### 3.3 Break the QS confound

Every ladder equilibrium has QS destroyed, monotonically with eps_1. On this data
"planarity cost falls with eps_1" and "planarity cost falls with QS error" are
indistinguishable. Run stage 1 with `QuasisymmetryTwoTerm` bounded at baseline (drop
`--no-qs`) and see whether eps_1 still moves and the ratio still falls.

### 3.4 The B-ladder

If eps_1 measures PLANARITY specifically, the B=1 penalty should shrink faster than B=2 or
B=3 as eps_1 falls. If all shrink together, it is a generic difficulty effect. Run arcB2
and arcB3 on the three equilibria (`make_start.py --rep polararc --B 2/3`).

### 3.5 Reverse dose-response

One-sided ladders are weak. `--arm hi` with `--target-frac 1.4` gives eps_1 ABOVE baseline;
the ratio should RISE. Two-sided is much more convincing than three monotone points.

---

## 4. Traps -- all of these cost real time

### Optimizer / DESC

* **`ForceBalance` under proximal must be an EQUALITY (`bounds=None`).** Proximal SOLVES it;
  `_parse_nonlinear_constraints` only routes bounds-free equilibrium constraints there. But
  under SINGLE-STAGE AL it must have BOUNDS -- `target=0` is infeasible (no equilibrium
  reaches exact force balance) and the penalty ramps to 2.4e6 while the trust region
  collapses. **Opposite conventions for the same constraint.**
* **Single-stage AL is unusable here.** With the 1145 interior DOF free, the optimizer
  reduces the objective by breaking force balance instead of reshaping the boundary:
  measured eps_1 -14.9% with the boundary moving 0.004% of a, force balance 20x over bound.
  Proximal removes the interior DOF and the cheat with them.
* **Constraints meaning "stay acceptable" are inequalities with bounds and weight 1.** Do
  not encode feasibility in weights. Objective weight = `1/|baseline - target|` so its
  residual starts at exactly 1.0.
* **Never call a stall before the AL penalty has stepped.** Observed 3 times: flat for
  45-55 iterations, then an outer update drops the cost immediately (10->25, 10->40,
  190->1840). A TRUE stall is flat cost AND flat optimality.
* **Do NOT use `tr_method="cho"` to save memory.** It forms J^T J and squares the condition
  number. `qr` is the default and correct; memory comes from chunking.
* **`jac_chunk_size` on an `ObjectiveFunction` requires `deriv_mode="batched"`**; with
  several things DESC picks "blocked" and raises. Chunk per sub-objective instead.
* **Shape-defining ints must be static attrs, never in `_constants`** -- jit traces them and
  `reshape` fails on a traced shape.
* `ObjectiveFromUser` returns `data["x"]` in **RPZ, not XYZ**. Silent wrong answers.
* `GenericObjective("iota")` already compresses to `dim_f = 5`; shrinking its grid to
  M=0,N=0 saves nothing and returns **-3.84 instead of 0.42** (iota needs a real theta/zeta
  grid for the surface average).

### GPU

* **`--qf-chunk 5` is REQUIRED for FourierXYZ.** At the default 25 every xyz run dies with
  `CUDA_ERROR_ILLEGAL_ADDRESS` (4/4) while planar always succeeds -- looks exactly like
  CLAUDE.md's 1-in-8 random fault but is representation-specific. `--coil-jac-chunk 4` also
  works, so it is an oversized fused kernel, not one objective. Even with the fix it still
  fails intermittently (~1 in 2-3), so **retry up to 3 times**.
* **After killing a run, verify `nvidia-smi` shows ZERO compute apps.** A leftover
  `run_one.py` holding 734 MiB made two unrelated runs die with the same CUDA error.
* **`pkill -f <pattern>` matches the shell whose command line contains the pattern** -- it
  will kill itself (`exit 144`). Put the kill in its own call with no other mention.
* L=M=N=12 re-solve fits in **~3.6 GB with `jac_chunk_size=100`** (334 s). An earlier
  "needs more than 10 GB" claim was from an unchunked run and was wrong.

### Physics / measurement

* **No eps_1 or C number is quotable at the resolution the equilibrium ships with.** The
  gain erodes monotonically: -41.5% (L=8) -> -37.0% (L=10) -> -33.4% (L=12), extrapolating
  to ~-28%. Re-measuring afterwards catches it but does not avoid it -- **run stage 1 at
  raised resolution**. L=10 stage 1 on GPU OOMs (4.11 GiB) and on CPU runs at ~150 s/it.
* `--vacuum` is REQUIRED for precise_QA (p=0, I=0); the sandbox defaults to finite beta.
* Bounds must be **absolute**, not a-scaled: `a` varies 0.1718 -> 0.1689 across the ladder
  and a-scaled bounds would move with the thing under test. Input JSON keys are
  `kappa_MS`, `d_cc`, `d_pc`, `convex_excess`, `planar_max` -- NOT the short names of the
  returned dict, and a wrong key silently falls back to the reactor default.
* `convex_excess: null` so planar and FourierXYZ are judged alike. On Helios the FourierXYZ
  reference never carried convexity and the comparison was unfair by 17% until noticed.
* **All six coil runs ended pinned on BOTH `L` and `kMS`.** Every number is from a
  budget-limited regime. This is consistent across the comparison but the absolute ratios
  are specific to this budget -- relevant to 3.2.

---

## 5. The C question, parked

`C = eps_2/(eps_1/4)` measures clamshell-ability (< 1 better than the generic `B^-2` decay).
**eps_1 and C are in TENSION**: minimising eps_1 RAISES C by +25.8% (L=8) to +37.4% (L=12),
`|dC/d eps_1| ~ 0.6`, opposite sign, and it STRENGTHENS with resolution. So SPEC [ADV-3]'s
design -- hold eps_1 fixed as a control while moving C -- is invalid. **Target C directly.**
But do not start until section 3 has settled whether eps_1 means anything, because C is
built on the same contour machinery and inherits any flaw.

Also unresolved from the cross-configuration sweep (`NOTES` 5c): across precise_QA/QH, HSX,
ESTELL, W7-X, NCSX, ARIES-CS and Helios, `eps_2` sits on the generic `B^-2` null within
+-25% and `eps_1` spans only 2.7x, with HSX LOWEST. That is why the observational program
was abandoned for interventions -- but it is also a standing reason to doubt that these
metrics discriminate real configurations.
