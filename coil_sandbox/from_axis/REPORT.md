# Growing QS axes for z = 0-joint two-face coils: near-axis single stage (report, 2026-10-09)

**Question.** Can the near-axis single-stage strategy of Giuliani et al. (JCP 2022) identify axes, and then QS
equilibria, that admit coils with **two planar faces per coil and every hinge on z = 0** (one demountable joint
plane)? Perturbing existing QS equilibria toward this coil class has been expensive (precise_QH: 2.8× B·n at fixed
equilibrium, 4.3× after single stage, `../work/ss_qh/joint/results/RESULTS.md`).

**Answer so far.** At the near-axis level, **yes for QA, and for QH at higher nfp**. With the coil class built into
the parameterization, positive bounded currents, and your Gil engineering bounds, the joint optimization of axis +
coils finds:

* **QA, nfp 2 and nfp 3:** every seed basin reproduces the first-order QS field on the axis to ~1% or better,
  with ι on target. nfp 3 gives the best fits of the study (ΔB < 0.05%, Δ∇B 0.3%) and a **nearly flat axis**
  (max |Z| 0.022-0.029), which is the geometry a z = 0 joint plane wants.
* **QH, nfp 5:** the best seed hits ι = 1.30 with ΔB 2.6% / Δ∇B 1.7%, **near-uniform currents** (0.88-1.21×) and a
  helical excursion of only |Z| 0.052 (precise_QH ~0.13-0.15). nfp 4 has one good basin (ι 1.22, |Z| 0.076);
  nfp 3 does not reach its ι target.
* The optimizer trades the helical excursion down by itself for this coil class: the "inversion" works.

**What is not yet shown:** whether these seeds stay QS away from the axis. Everything here is first order on the
axis; the honest finite-radius test (vacuum free-boundary solve with the coils, Phase 2 of `PLAN.md`) is not
built yet. The one rough finite-radius check made with a stand-in bridge was poor (QS 2-3e-2), for reasons
explained in §5 that do not condemn the method but do mean the off-axis question is open.

**Update (later 2026-10-09): the sweep-1 QH runs were stall-limited.** The clearance and coil-coil rows used the
distance to the *nearest* node, which is kinked wherever the nearest node changes; with many active clearance
rows the trust region collapses (the "stopped on ftol at ~1e-5 optimality" runs). `run_na2.py` now uses a smooth
soft-min over the 8 nearest nodes (τ = 2 mm, errs safe by ≤ 4 mm). Continuing QH nfp 5 s2 with it took the
on-axis mismatch from 2.6% / 1.7% to 0.6% / 0.6% in 15 iterations (ι 1.301). **Sweep 1's QH numbers are therefore
pessimistic and should be rerun with the smooth distances**; the QA ones had converged (optimality ~1e-7).

Viewer: `viewer.html` (every config: coils, hinges, axis, NAE surface, numbers). Plan and agreed rules:
`PLAN.md`. Orientation for agents: `CLAUDE.md`.

---

## 1. The paper's strategy

Giuliani, Wechsung, Cerfon, Stadler, Landreman, *Single-stage gradient-based stellarator coil design: optimization
for near-axis quasi-symmetry*, JCP 459, 111147 (2022) (`singlestage_QS.pdf`).

1. **Unknowns:** coil shapes and currents, an *independent* expansion axis (R_k cos kNφ, Z_k sin kNφ), and η̄.
2. **Target from the axis:** first-order Garren-Boozer QS. Given (axis, η̄), solve the σ equation (periodic
   Riccati ODE) for σ(φ) and ι; this fixes the QS field on the axis, B_QS = B0 t, and its full gradient ∇B_QS
   (their eqs. 1-4).
3. **Objective:** ∫_axis |B_coils − B_QS|² + ∫_axis |∇B_coils − ∇B_QS|² + ((ι − ι0)/ι0)² + regularization
   (lengths, curvature, coil-coil distance). σ, ι are eliminated by the implicit function theorem; exact
   gradients by forward/adjoint sensitivity; BFGS (steepest descent fails: the Hessian spans ~12 decades).
4. **At the optimum** the coils' magnetic axis coincides with the expansion axis, and the coil field is QS near it.
5. **No boundary and no equilibrium solve.** Vacuum throughout. Afterwards they trace field lines and compute
   flux surfaces and Boozer coordinates *of the coil field* (aspect ratio is an outcome).
6. **Limits:** vacuum; QS controlled only near the axis (it degrades outward, their Figs. 7-8); simple engineering
   terms. Their Hessian study finds many nearly flat directions in coil space: room for engineering constraints.
7. **Follow-ups (from memory, not in this folder):** Giuliani et al. JPP 2022 (Boozer surfaces of a coil field);
   PoP 30, 042511 (2023) (start from near-axis coils, then optimize QS on the coils' own surfaces at finite
   radius); QUASR (2024) runs that pipeline at scale.

**Why it fits this problem.** The coil class is a hard constraint that existing equilibria were not designed for.
Here it is *built into the parameterization*, and the axis/plasma adapts to what those coils can make. Each run
takes minutes with no equilibrium solve, so many seeds are affordable.

## 2. Rules agreed for staying in the spirit of the papers

1. Coils are primary; the plasma is whatever the coils' field makes. Vacuum.
2. The NAE is only the on-axis target and an initial guess; it never defines the evaluated boundary.
3. The scoring solve (vacuum free boundary with the coils, given enclosed flux) has **no NAE term**: it has at
   most one answer, the coils' own surface. NAE-vs-solved is a diagnostic.
4. Poincaré checks guard against fake nested surfaces.
5. Drift away from the NAE is allowed only in the refinement stage, and monitored (z = 0 wants a small excursion).
6. Currents stay constrained: each in [0, 2× uniform] (no reversed coils).

## 3. What was built (`na/`)

| piece | what | verification |
|---|---|---|
| `nae.py` | first-order NAE in JAX: σ equation by Newton plus one differentiable step (= exact implicit derivative), B_QS, ∇B_QS (Cartesian), helicity | ι and ∇B match pyQSC to ~1e-9 (QA r1 §5.1, QH r1 §5.2) |
| `folded.py` | the coil class: hinges (R, φ) at z = 0; upper and lower arcs each in a plane through the chord at its own tilt; shape w(t) = c Σ a_k sin kπt; stellarator image + nfp copies; exact polyline Biot-Savart; ∇B by jacfwd | loop-centre field; div B = curl B = 0 to 1e-15 |
| `run_na2.py` | the paper's objective as rows for the branch's **composite least squares** (`lsq_composite`, hinges exact in the model, secant Hessian). Bound rows: coil → NAE surface at r = a ≥ d_pc, coil-coil ≥ d_cc, κ ≤ κ_max, L ≤ L_max, currents in [0, 2× uniform], elongation ≤ cap | all runs end feasible |
| `to_desc.py` | NAE → `Equilibrium.from_near_axis` → fixed-boundary solve; coils → `PiecewisePlanarArcCoil` (B = 2) CoilSets; scored with `common.evaluate` + Boozer QS | conversion reproduces the field to 0.4-0.9%, hinges exactly z = 0. **The fixed-boundary part is not the agreed scoring** |
| `export_viewer.py`, `viewer_template.html` | offline 3D viewer | |
| `summarize.py` | results table | |

Joint optimization: one parameter vector (axis R0..R5, Z1..Z5, η̄, and per coil 2 hinges, 2 tilts, 2×5 shape
coefficients, current: 80 parameters for nc = 4). σ and ι are solved at every evaluation. nfp, the coil count and
the helicity (an integer, taken from the starting axis) are fixed per run.

## 4. Results

### 4.1 First prototype (scipy TRF + squared hinges; currents unconstrained; clearance to the axis)

| run | ΔB / Δ∇B | ι (target) | note |
|---|---|---|---|
| QA nfp2 z0 | 1.8% / 3.6% | 0.410 (0.42) | not converged |
| QA nfp2 free hinges | 0.2% / 1.2% | 0.419 | |
| QH nfp4 z0 | 0.6% / 1.4% | 1.243 (1.25) | **relied on a reversed, large current** (−0.28 vs +0.03..+0.18) |
| QH nfp4 free hinges | 1.3% / 1.7% | 1.210 | single cold seeds: basin noise |

Adding current bounds with this solver **stalled** (status xtol, optimality 0.18) at 15-21% mismatch: squared
hinges have zero gradient at the bound, which Gauss-Newton cannot handle (the lesson already in the sandbox notes).
Coil-to-axis clearance was also the wrong measure: with elongation ~5 the a = 0.124 surface reaches ~0.28 from the
axis, so the QH coils sat 0.073 m from the plasma (bound 0.11).

### 4.2 Composite solver, NAE-surface clearance, constrained currents (`run_na2.py`)

**Pass 1 (QH nfp 4, nc 4, 32 coils):** every bound met with positive currents (0.31-1.62×), ΔB 3.5% / Δ∇B 2.4%,
ι 1.225, axis |Z| 0.077, elongation 5.3 (`runs2/qh_z0_nc4`). nc = 5, 6 were dropped (32 coils is enough).

**Sweep 1: breadth of axis types** (z0, d_cc relaxed to 0.04, elongation ≤ 6, 4 randomized seeds per case;
bounds = Gil set scaled by a: d_pc 0.885 a, κ 1.699/a, L 23.6 a):

| case (coils) | a | good seeds | best ΔB / Δ∇B | ι (target) | elongation | axis max abs(Z) | currents (best) |
|---|---|---|---|---|---|---|---|
| QA nfp 2 (16) | 0.167 | 4/4 | 0.2% / 1.1% | 0.420 (0.42) | 4.5-5.1 | 0.074-0.087 | 0.58-1.14× |
| QA nfp 3 (24) | 0.143 | 3/4 | <0.05% / 0.3% | 0.400 (0.40) | 3.4-5.5 | **0.022-0.029** | 0.24-1.53× |
| QH nfp 3 (24) | 0.143 | 0/4 | 2.1% / 4.3% | 1.01-1.05 (1.10) | 5.4-6.0 (cap) | 0.117-0.157 | |
| QH nfp 4 (32) | 0.124 | 1/4 (+ pass 1) | 4.1% / 2.2% | 1.224 (1.25) | 4.1-6.0 | 0.076-0.114 | 0.34-1.65× |
| QH nfp 5 (30) | 0.110 | 2/4 | 2.6% / 1.7% (s2) | 1.299 (1.30) | 2.6-6.0 | **0.045-0.102** | **0.88-1.21×** |

Full per-seed table: `cd na && python summarize.py "runs2/s1_*.json"`.

Observations:

* **QA is easy and consistent** with z = 0 joints; QA nfp 3 is the standout.
* **QH depends on nfp and on the seed.** Failed QH seeds sit on bounds: a current at 0 or 2×, or the elongation
  cap. nfp 3 QH never reaches its ι (it keeps a large helix, |Z| 0.12-0.16).
* **Trend:** higher nfp and a smaller helical excursion help QH with this coil class.
* **Geometry checks** (all 21 configs): every hinge is exactly at z = 0; every unique coil links the axis exactly
  once (Gauss linking number +1.00), so these are genuine modular coils.
* **Coil shapes are poorly regularized.** The coils reach R 1.3-1.9 (axis R ≈ 1) and |z| up to 0.8, with length and
  curvature at their bounds and irregular arcs (visible in the viewer). This is the paper's flat-direction finding:
  the on-axis objective barely sees distant coil parts, so they wander until a bound stops them. It needs a
  regularizer before these coils are taken further (§6).

## 5. The one finite-radius check so far, and why it is not a verdict

`to_desc.py` builds the first-order NAE boundary at r = a and solves the fixed-boundary vacuum equilibrium inside
it (an early stand-in, before the rules in §2 were agreed):

| seed | a, A | Boozer QS ρ .25 / .5 / .75 / 1 | ι (NAE) | B·n | d_pc |
|---|---|---|---|---|---|
| QA nfp2 z0 (prototype) | 0.173, 5.3 | 2.4e-2 / 2.0e-2 / 3.3e-2 / 6.4e-2 | 0.39-0.42 (0.41) | 3.5e-2 | 0.166 |
| QH nfp4 z0 (prototype) | 0.131, 6.8 | 3.1e-2 / 3.2e-2 / 4.6e-2 / 8.3e-2 | 1.12-1.27 (1.24) | 1.6e-1 | 0.073 |

Reference: precise_QA free arcB2 B·n ≈ 2e-3; precise_QH z0 1.8e-2 (fixed eq), 6.4e-3 (single stage).

These mix three errors: (i) the boundary is truncated at first order, but QS off the axis depends on the
second-order shape; (ii) the QH coils were too close (the old clearance); (iii) genuine coil error away from the
axis. ι survived the transfer (good sign for the first-order content). Only the free-boundary solve with the coils
measures (iii) cleanly; these seeds are also from the prototype, not sweep 1.

## 6. Assessment

* **Axis identification: yes.** The near-axis stage is cheap (minutes per seed on CPU), keeps the coil class
  exact, and finds z = 0-compatible axes the perturbative route could not: flat QA axes, and QH axes with a third
  of precise_QH's excursion. Its output is a *seed*: axis, η̄, ι and coils.
* **QS equilibria: open.** The first-order target controls the field only near the axis. Two specific risks:
  (a) QS off the axis needs the second-order shape, which the near-axis stage does not control; (b) the coils are
  under-determined far from the axis (§4.2), so their off-axis field (ripple from two-face coils) is unshaped.
  The papers' answer is the next stage: optimize the coils for QS on their own surfaces at finite radius.
* **Where to look first:** QA nfp 3 (s3, s0, s2), QA nfp 2 (s2), QH nfp 5 (s2, s3), QH nfp 4 (s2 / pass 1).

## 7. Caveats

* On-axis numbers only; all finite-radius statements wait for Phase 2.
* The clearance uses the first-order NAE surface at r = a (an approximation of the plasma's shape).
* QH runs often stopped on ftol with optimality ~1e-5: not fully converged. Four seeds is thin where variance is high.
* d_cc was relaxed to 0.04 (precise_QH: 0.0585); the elongation cap (6) binds in several QH seeds.
* The free-hinge control has not been rerun with the new solver, so the joint-plane cost per axis type is unmeasured.
* The polyline Biot-Savart (80 segments per arc) differs from DESC's smooth arcs by 0.4-0.9% in B.

## 8. Recommended next steps

1. **Regularize the coils:** a weak total-length (or Σ curvature²) term, as the paper does with length targets,
   so the flat directions settle on compact coils instead of the bounds. Rerun the best seeds.
2. **Phase 2 scorer:** vacuum free-boundary DESC solve with the converted coils (Ψ for the target a; NAE surface as
   the initial guess only), Boozer QS and ι on four surfaces, axis vs NAE, Poincaré check.
3. **Free-hinge counterpart of sweep 1**, same seeds, to measure the joint-plane cost per axis type.
4. More seeds and continuation for QH nfp 4/5 (and nfp 6 if the trend holds).
5. Then refinement in the PoP 2023 spirit (QS on the coils' own surfaces), with the user's go-ahead.

## 10. Phase 2 results: the coils' own surfaces (vacuum free boundary, `na/score_fb.py`)

Seeds: `elong_qa2s2_e5`, `elong_qh5s2_e5` (regularized w = 1e-3, elongation <= 5). Coils converted exactly
(`na/convert.py`, pointwise 1e-15 m; GL 64 nodes per arc, since 24 left a 7e-4 quadrature error for these arcs).

| seed, a (A) | Boozer QS rho .25 / .5 / .75 / 1 | iota axis (NAE) / edge | B.n rms | d_pc (bound) | Poincare |
|---|---|---|---|---|---|
| QA nfp2, a 0.167 (5.4) | 1.3e-2 / 3.4e-2 / 6.3e-2 / 1.0e-1 | 0.446 (0.420) / 0.331 | 2.2e-3 | 0.132 (0.148) | boundary at the edge of the confined region; superseded |
| **QA nfp2, a 0.12 (7.6)** | **3.1e-3 / 1.3e-2 / 3.1e-2 / 6.0e-2** | **0.418 (0.420) / 0.372** | 3.2e-4 | 0.190 | **clean nested surfaces; DESC boundary lies on the traced field lines** |
| QH nfp5, a 0.11 (8.2) | 6.3e-3 / 2.4e-2 / 5.8e-2 / 1.1e-1 | 1.247 (1.295) / 1.045 | 4.1e-3 | 0.067 (0.097) | boundary at the edge of the confined region (chaos just outside) |

* The coils realize their first-order design (QA: iota on axis to 0.5%, axis R1 exact), but QS degrades as r^2
  away from the axis: the second-order terms were never controlled.
* Second-order diagnosis (`na/nae2.py`, validated vs pyQSC): B20 residual 0.56 (QA seed) and 1.92 (QH seed), against
  0.135 / 0.007 for Landreman-Sengupta's reference QA / QH. B2c alone only reaches 0.48 / 1.85: the axis must change.
* Solver health: most free-boundary shells end on a trust-region stall ("bad approximation"), ~1 rejected step per
  accepted one, force error grows across shells. QA a 0.12 is validated independently (Poincare, iota); the QH
  numbers are provisional. Suspected cause: inner proximal solve drift (as in the k5 stall); diagnostic deferred.
* Coil-coil distance against the dense check is 0.031 (bound 0.040) for QA: node-node rows on an 80-segment
  polyline miss up to ~1 cm between nodes. The branch's distance rows (with the gap) fix exactly this.

## 11. Second-order near-axis stage (in progress)

`run_na2.py --order 2`: B2c free; rows (B20 - <B20>) (second-order QS breaking, weight w2) and coil grad grad B vs
NAE grad grad B (weight wgg x a/2, so B, gradB, gradgradB rows all measure field error at r = a). At the first-order
seeds the coils' grad grad B is 96% off any NAE on their axis, so the weights are ramped (`na/o2.sh`).

### 11.1 Second-order ladder on the two seeds (`na/o2.sh`), and what it does off-axis

| run | coils' gradgradB vs NAE | B20 residual | first order dB / dG | iota (target) | axis abs(Z) |
|---|---|---|---|---|---|
| QA nfp2 seed | 96% | 0.48 (best B2c) | 0.11% / 0.85% | 0.420 (0.42) | 0.074 |
| QA step 2 (wgg 1e-2, w2 1e-3) | 63% | 0.28 | 1.2% / 3.1% | 0.414 | 0.110 |
| QA step 3 (1e-1, 1e-2) | 26% | 0.42 | 2.2% / 4.7% | 0.411 | 0.135 |
| QA step 4 (1, 1e-1) | 13% | 0.24 | **11.4%** / 6.5% | 0.403 | 0.131 |
| QH nfp5 seed | 96% | 1.85 | 0.18% / 0.50% | 1.295 (1.30) | 0.048 |
| QH step 3 | 25% | 0.48 | 2.2% / 4.0% | 1.230 | 0.120 |
| QH step 4 | 19% | 0.16 | 4.7% / 5.8% | 1.283 | 0.136 |

Step 4 overpowers the first-order rows (the expansion axis stops being the coils' axis), so ladders now stop at
step 3. In both cases **lowering second-order QS breaking raised the axis out of the midplane** (QA 0.074 -> 0.135,
QH 0.048 -> 0.136).

Free boundary (coils' own surfaces) of step 3:

| | rho .25 / .5 / .75 / 1 | iota axis / edge | B.n rms |
|---|---|---|---|
| QA first-order seed, a 0.12 | 3.1e-3 / 1.3e-2 / 3.1e-2 / 6.0e-2 | 0.418 / 0.372 | 3.2e-4 |
| **QA step 3, a 0.123** | 6.7e-3 / **1.0e-2 / 1.7e-2 / 2.8e-2** | 0.399 / 0.344 | 5.5e-4 |
| QH first-order seed, a 0.11 | 6.3e-3 / 2.4e-2 / 5.8e-2 / 1.1e-1 | 1.247 / 1.045 | 4.1e-3 |
| **QH step 3, a 0.088** | 2.7e-2 / 2.9e-2 / 3.4e-2 / 4.6e-2 | 1.186 / 1.193 | 1.6e-5 |

**The second-order extension works off-axis** (QA edge 2.2x better; the r^2 growth is tamed: QA rises 4x from rho
.25 to the edge vs 19x before; QH profile nearly flat), but the weighting gave up first order, which raised the
near-axis floor (QA 3.1e-3 -> 6.7e-3; QH ~3%). Fix under test: `--w1` (stiff first-order rows) in step 3.

### 11.2 The near-planar axis search (`--zmax`, `na/chainA.sh`, `na/sweepP2.sh`)

Theory: with zero torsion the first-order sigma equation forces iota = 0 for QA (sigma' = -iota (eta^4/kappa^4 + 1 +
sigma^2) cannot be periodic), and a planar axis has helicity 0 (no QH). Iota must come from torsion; small, fast
vertical wiggles (higher nfp or higher Z harmonics) give torsion at small abs(Z).

| capped run | first order dB / dG, iota | second order (step 3) | verdict |
|---|---|---|---|
| QA nfp2, abs(Z) <= 0.04, seeds 0 and 1 | 0.28-0.32% / 1.4-1.7%, iota 0.417-0.418 | s0: B20 0.34, first order 7.7% / 6.5%, iota **0.331** | first order fine; second order costs iota |
| **QA nfp3, abs(Z) <= 0.02** | **0.01% / 0.07%, iota 0.400** | B20 stuck at 0.74, iota **0.284** | best first-order fit of the study; second order impossible at this cap |
| QH nfp5, abs(Z) <= 0.03 | 14.7% / 9.1%, iota 1.269, a current at 0 | (stopped) | fails at first order |
| QH nfp4, abs(Z) <= 0.04 | 12.2% / 7.8%, iota 1.115 | (stopped) | fails at first order |

**Finding:** near-planar QA axes exist at first order with these coils (nfp 3 at 2 cm excursion), but second-order QS
then costs rotational transform; near-planar QH fails already at first order. Levers left: an intermediate cap (the
knee of the trade-off), more elongation (the cap of 5 binds in every capped run), a lower iota target; or a different
coil class (per-coil horizontal hinges, as the user suggested).

### 11.3 Stiff first order (`--w1 10`) and the session's outcome (2026-10-09 night)

Free boundary (coils' own surfaces), QA nfp2, a ~ 0.12:

| configuration | Boozer QS rho .25 / .5 / .75 / 1 | iota axis / edge | axis abs(Z) | Poincare |
|---|---|---|---|---|
| first-order seed | 3.1e-3 / 1.3e-2 / 3.1e-2 / 6.0e-2 | 0.418 / 0.372 | 0.074 | clean |
| second order, step 3, w1 = 1 | 6.7e-3 / 1.0e-2 / 1.7e-2 / 2.8e-2 | 0.399 / 0.344 | 0.135 | - |
| **second order, step 3, w1 = 10** (`o2w1_qa2s2_gg1e-1_w1e-2_w1x10`) | **3.2e-3 / 6.6e-3 / 1.3e-2 / 2.3e-2** | 0.396 / 0.325 | 0.101 | **clean nested surfaces past the boundary** |
| near-planar abs(Z) <= 0.04, step 3, w1 = 10 | 8.1e-3 / 1.3e-2 / 2.3e-2 / 4.0e-2 | 0.266 / 0.225 | 0.040 | (B.n max 0.11: edge approximate) |

* **Best result of the line:** z = 0 two-face coils (16 coils, every hinge on z = 0, currents 0.82-0.97x uniform)
  whose own field has nested surfaces at A 6.8 and QS 0.3% near the axis, 0.7% at mid-radius, 2.3% at the edge.
  Two independent seeds (s0 and s2) converge to the same state, so it is a robust basin.
* Stiff first order is the right weighting for the second-order steps; it is now the default in `na/chainA.sh`.

Near-planar search (`--zmax`), all seeds:

* QA nfp2 abs(Z) <= 0.04 (2 seeds) and QA nfp3 abs(Z) <= 0.02 (2 seeds): excellent first-order fits (QA nfp3: 0.01% /
  0.07%), but second-order QS then costs rotational transform: iota 0.33 (nfp2) and 0.28 (nfp3) with w1 = 1; with
  w1 = 10 the cost moves entirely into iota (0.27). B20 residual cannot go below ~0.74 at nfp3, 2 cm.
* QH nfp4 abs(Z) <= 0.04 and nfp5 <= 0.03: fail already at first order (12-15% field mismatch, a current at 0).
* Uncapped QH nfp5 s0 with stiff first order: B20 residual stays 3.8-6.2 and iota falls 17%; QH seed s2 (w1 = 1)
  remains the only usable second-order QH (B20 0.48; flat ~3% QS profile on its surface).

**Conclusion for this coil class:** second-order QS, rotational transform and a flat axis form a three-way trade.
The near-axis theory says why (iota comes from torsion, which needs out-of-plane excursion; QH needs a rotating
normal). With z = 0 joints, the best QA keeps abs(Z) ~ 0.10; flattening to 0.04 halves QS quality and loses a third of
iota. Next levers: a lower iota target for near-planar QA (accept iota ~ 0.27 and design for it), more elongation,
higher nfp with Z harmonics, or a different coil class (per-coil horizontal hinges).

## 9. Files

* `PLAN.md`: plan, rules, decisions, status. `CLAUDE.md`: orientation for agents.
* `viewer.html`: built by `na/export_viewer.py`.
* `na/runs_snapshot/`: the prototype runs (`run_na.py`) and their DESC scorings.
* `na/runs2/`: `run_na2.py` runs: `qh_z0_nc4`, `s1_<case>_z0_s<seed>` (`.json` numbers, `.npz` parameters,
  `.log` solver trace).
