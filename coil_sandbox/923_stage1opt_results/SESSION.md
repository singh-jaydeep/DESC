# 2026-09-23: stage-1 eps_1 optimization — what we did, what it cost, what survived

Companion to `922_stage1opt_results/` (the code) and `NOTES.md` sections **9m–9p** (the
chronology, including every wrong turn). This folder holds the artifacts: every equilibrium and
coilset produced today, the offline viewers, the diagnostic scripts, and all run logs.

**Read the caveats in section 6 before quoting any number here.**

---

## 1. The headline

Two changes to the objective — both the user's — turned eps_1 optimization from something that
damaged the equilibrium into something that improves coils *and* the equilibrium together, which
had not happened before in this program.

| | proportional targeting | **worst-offender + L_gradB floor** |
|---|---|---|
| eps_1 | −32.9% (`aspin_leg2_iotatied`) | **−17.8%** (`worst_leg3`) |
| planar coil B·n | −14.81% | **−9.23%** |
| **planar % per eps_1 point** | 0.450 | **0.519** |
| min L_gradB/a | **−25.81%** | **−1.81%** |
| pdrot·a median | **+12.02%** (worse) | **−11.85%** (better) |
| foliation spread | **3.43×** | **1.73×** (baseline 1.66×) |
| coil constraint regime | crowded onto `d_cc` (1.008) | **same as baseline** (`d_cc` 2.235) |

The old objective asked every proxy contour for the same *fractional* reduction, which subsidises
the curves that are already planar because they are cheapest to move. A planar coilset is limited
by its **worst** coil, so that was backwards.

---

## 2. What changed in the code (all in `922_stage1opt_results/`)

### `contour_curve.py` — eps_1 no longer freezes the best-fit plane

The B=1 metric held its plane normal from build time. That made eps_1 **not rotation-invariant**:
a rigid tilt of the curve — same curve, same planarity — inflated it 1.08× at 2°, 1.28× at 4°,
1.89× at 8°, against a whole-ladder signal of 0.60×. A planar basin at a different plane
orientation was unreachable.

`_lam_min` (custom JVP, `d lambda_0/dM = v0 v0^T`) makes it exact and rotation-invariant with no
division by an eigenvalue difference, so the degenerate in-plane pair is still never
differentiated. Validated: live == frozen at the freeze point to 7.7e-14; rotation-invariant to
1.6e-13 at 1/5/20/90°; reverse-mode gradient vs central FD **1.5e-8** at baseline and **2.2e-8**
at a displaced geometry; exactly degenerate in-plane pair (λ1/λ2 = 1.000000000) grad vs FD
1.3e-11. B≥2 keeps the freeze — there the segmentation is genuinely combinatorial.

### `stage1_v1.py` — new flags

| flag | what it does |
|---|---|
| `--target-mode worst` | single one-sided bound `(0, frac·max(base))` instead of per-contour proportional targets |
| `--lgradb-floor/-weight/-grid/-backoff/-chunk` | one-sided floor on `L_grad(B)`, per node |
| `--fb-weight`, `--fb-chunk` | soft `ForceBalance` in the objective |
| `--soft-weight` | scales bounded constraints when the method has no AL |
| `--ftol/--xtol/--gtol`, `--tr-ratio` | outer optimizer tolerances and initial trust radius |
| `--frozen-eps1` | restores the pre-2026-09-23 frozen normal |

Plus two **reference leaks fixed**: `iota0` and the L_gradB floor are now saved and reloaded
across `--from` restarts, so a chained leg inherits the *original* equilibrium's bounds instead of
being granted a fresh ± budget every leg.

### `view.py` — streamlines and a surface heatmap

* `--streamlines K` draws K current-potential contours per field period (the proxy modular
  coils), replicated over the torus, **shaded by each curve's own eps_1**. They are cut at equal
  ΔΦ so they are equal-current: the bunching is physical, not a drawing choice.
* `--field-ref` adds an `L_gradB` heatmap — sequential blue for absolute, diverging blue↔red with
  a gray midpoint for the ratio-vs-reference, which is what shows *where* the field changed.

---

## 3. Contents of this folder

```
equilibria/    11 stage-1 results (.h5)
coilsets/      8 stage-2 coilsets (.h5 + .json) + their circles starts
viewers/       5 offline HTML viewers — open directly, no server
logs/          every stage-1 and stage-2 run log, plus the chain drivers' output
diagnostics/   the measurement scripts, so every table here can be regenerated
```

**Equilibria** (all vacuum precise_QA descendants; baseline eps_1 = 0.067633, A = 6.0):

| file | eps_1 | A | how |
|---|---|---|---|
| `AL_live_eps1.h5` | −22.77% | 6.29 | `lsq-auglag`, live eps_1 — **stalled at iteration 105** |
| `lsq_noFB.h5` | −39.55% (−28.88% at L=12) | 6.25 | `proximal-lsq-exact`, no force term |
| `lsq_fb50/165/500.h5` | −38.70 / −26.52 / −27.50% | 6.25 | force-weight sweep |
| `aspin_leg1.h5` | −10.31% | **6.010** | aspect pinned — trust-region limited |
| `aspin_leg2_iotaleak.h5` | −33.78% | 6.010 | **iota exceeded its bound by 0.013 — do not use** |
| `aspin_leg2_iotatied.h5` | −32.91% | 6.010 | iota tied to baseline; proportional targeting |
| `worst_leg1/2/3.h5` | −0.5 / −17.0 / **−17.8%** | 6.010 | **worst-offender + L_gradB floor** |

**Coilsets** — identical problem for all four equilibria, so `baseline_*` is the control:
absolute bounds (`bounds/precise_qa_planar.json`), 16 coils, r/a 2.5 cold start, 200 iterations,
`--vacuum`, `--qf-chunk 5`.

| | planar B·n | xyz B·n | ratio |
|---|---|---|---|
| `baseline` | 6.1104e-02 | 1.3311e-02 | 4.590 |
| `fb500` | 6.0826e-02 | 1.5616e-02 | 3.895 |
| `aspin3` | 5.2057e-02 | 1.6483e-02 | 3.158 |
| `worst3` | **5.5463e-02** | 1.5078e-02 | 3.678 |

---

## 4. The two results worth keeping

### 4a. Targeting the worst offenders inverts who moves

Change in eps_1 by baseline rank, easiest → hardest quartile:

```
aspin  (proportional, mean -10.3%)   -28.8%   -2.4%   -6.0%    -8.9%     spread 1.66x -> 2.35x
aspin3 (proportional, mean -32.9%)   -56.0%  -16.4%  -24.9%   -37.1%     spread 1.66x -> 3.43x
worst3 (worst-mode,   mean -17.9%)   -19.9%   -4.2%  -16.9%   -29.6%     spread 1.66x -> 1.73x
```

Proportional **stretches** the foliation and barely touches the hard curves. Worst-mode moves the
hardest quartile most and preserves the spread. There is a manufacturing argument on top of the
coil-physics one: a real modular set is built from a few unique coil types, so a foliation
spanning 3.4× instead of 1.7× means the coils become more unlike each other even at equal
average planarity.

### 4b. The L_gradB floor costs almost nothing and buys a lot

`GenericObjective("L_grad(B)", bounds=(floor, inf))` — `dim_f` is per node, so it is one-sided
per node and only the offending patch is penalised. At eps_1 −17.8% it held min L_gradB to
−1.81%, against the −25.81% proportional targeting destroyed at −32.9%, and pdrot's median
*improved*.

**Backoff must be 1.0.** At 1.01 the floor sits 1% above the baseline's own min, so 14/2401 nodes
violate at the start and the term costs 4.0 — as much as the entire eps_1 objective — to correct
a 0.7% grid error. Grid convergence: min L_gradB/a is 3.1788 (M=N=8) → 3.1623 (M=N=64), so
M=N=24 costs 0.7%.

---

## 5. Where the L_gradB damage actually is (`local_global.py`)

Ratio of each optimized boundary's L_gradB/a to baseline's, at matched (θ, ζ):

```
run       eps_1     p5      p50     p95    % worse   % worse by >10%
aspin    -10.3%   0.912   0.975   1.110     64%            3%
lsqfb    -27.2%   0.607   0.951   1.203     66%           29%
aspin3   -32.9%   0.505   0.909   1.359     59%           49%
```

**The extent is global and constant** — ~60% of the boundary degrades in every run, including the
mild one. **The severity is local and grows**: the fraction degrading by more than 10% goes
3% → 29% → 49%. Summary statistics like "min L_gradB −24%" hide a bimodal picture. The location
of the global minimum also **moves** off baseline's (ζ=0°, θ=180°), so the place where coils are
forced closest migrates, not just the value.

---

## 6. Caveats — read before quoting anything above

1. **Only 1 stage-1 leg in 3 is productive, and nobody knows why.** `worst1`/`worst2`/`worst3`
   had identical configuration; leg 2 did all the work and converged on `xtol`, legs 1 and 3 did
   nothing and died on "bad approximation". Legs 2 and 3 had the *same* trust radius; legs 1 and
   2 had the *same* floor status. Four mechanisms were proposed today (proximal re-solve noise,
   initial trust radius, the floor's zero slack, a kink in the worst-mode bound) and **all four
   were falsified within minutes**. `--tr-ratio` does not help: a 2500× larger initial radius
   changed nothing, and leg 1's first step was 1.003e-03 against a radius of 4637 — never
   trust-region limited at all.
   → **Do not conclude "configuration X does not work" from a single stalled run.** Restart first.
   → Untried isolation: save `precise_QA` to `.h5` and run the fresh config through `--from`.
     Same equilibrium, same objective; only the fresh-build-vs-restart code path differs.
2. **No stage-1 run has ever converged to a stationary point.** Best optimality reached is
   3.5e-02 (`lsq_fb50`); most stop on a budget limit or model failure. Every eps_1 figure is a
   lower bound.
3. **The worst-mode result rests on one productive leg.** It deserves a repeat before it is
   load-bearing.
4. **Objective-block scaling is badly unbalanced.** Measured pre-proximal at the start: the force
   term's residual is 77× *smaller* than eps_1's and its Jacobian 171× *larger*, carrying 99.997%
   of ‖J‖². Balancing Jacobians instead of costs would put `--fb-weight` near 3, not 500.
   `ProximalProjection` should remove much of that, but any leakage is multiplied by 500 — and
   there is a real conflict, because the resolution-gap control *wants* a large weight
   (w=500 → 0.29 pt gap, w=50 → 4.96). The force term may belong somewhere other than the
   objective.
5. **QS is destroyed in every equilibrium here** (`--no-qs` throughout; QS rms up to 380×
   baseline). HANDOFF §3.3's confound is untouched.
6. **pdrot's tail still degrades** even in the good runs — `worst3` has median −11.85% but mean
   +18.73% and 95th +54.68%, and umbilic proximity falls 0.480 → 0.330. Quote the **median**:
   pdrot carries a `1/(c²+s²)` factor, so near-umbilic points inflate the max without bound.
7. **`loose_eps1_L2` (the old ladder's leg 2) is unusable** for complexity work: L_gradB/a min =
   0.0068 and pdrot max = 3e5, on an equilibrium whose force balance is 1.75e-3.

---

## 7. Corrections this session made to earlier conclusions

| earlier claim | status |
|---|---|
| §9k: "the eps_1/C tension is the most solid result of the session" | **artifact** — C reverses sign (−3.13%) when aspect is pinned |
| §9j: the −1.98% ceiling was the tight iota bound | **wrong** — it was aspect ratio; a loose-iota run with a different optimizer crawled identically |
| §9l: planar coils degrade monotonically with eps_1 | **artifact** — equilibrium quality, not eps_1. −0.45% at matched force balance |
| §9m: "the converged eps_1 reduction is about −27%" | that was at A = 6.25, and "converged" meant *resolution*-converged, not optimizer-converged |
| §9n: "eps_1 spends constraint slack" | **retracted** — built on a trust-region-limited run |
| §9o: damage is a dose-response in eps_1 alone | **superseded** — it depends on *how* eps_1 is driven, not only how far (§9p) |

---

## 8. Next

1. **Repeat the worst-mode run.** One productive leg is not a result.
2. **Isolate the fresh-build vs restart difference** (caveat 1). It undermines every single-run
   conclusion until understood, and it is one cheap test.
3. **Intermediate targets** (`--target-frac` 0.85, 0.80) to map how far worst-mode goes before
   the complexity metrics turn.
4. **The database sweep, no GPU required.** QUASR and Constellaration ship equilibria *with*
   optimized coils, so eps_1 / pdrot / L_gradB can be regressed against *achieved* coil
   complexity (length, curvature, eta_SVD) over thousands of devices. If eps_1 adds nothing over
   pdrot (R² = 0.700 univariate against coil non-planarity), the question closes without another
   intervention. This inverts §5c's failure, which was 8 hand-picked configurations with no
   dynamic range.
5. **A pdrot objective**, if the L_gradB floor keeps paying off. It is the predictor most
   directly opposed to what we want, and nothing currently blocks it — but it needs the director
   field in `coil_complexity.py` made differentiable.
