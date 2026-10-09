# Growing z = 0-joint QS configurations from the axis (Giuliani-style near-axis single stage)

Started 2026-10-09. Common reference for the from-axis line of work: what we are testing, the rules we
agreed on, what has been run, and what comes next. Update the **Status** and **Results** sections as runs land.

## The question

Perturbing existing QS equilibria (precise_QH, WISTELL-A, ...) toward coils with **two planar faces per coil and
both hinges on z = 0** (one demountable joint plane) is expensive: on precise_QH the joint plane cost 2.8×
(fixed eq) and 4.3× after single stage, because the helical axis's vertical excursion is untouched
(`../work/ss_qh/joint/results/RESULTS.md`). Instead of starting from an equilibrium, start from the coils:
**find axes, and then QS equilibria, that this coil class can actually produce.**

## The paper (`singlestage_QS.pdf`: Giuliani, Wechsung, Cerfon, Stadler, Landreman, JCP 459, 111147, 2022)

* Variables: coil shapes + currents, an **independent expansion axis** (R_k cos, Z_k sin), and η̄.
* Given (axis, η̄), first-order Garren-Boozer QS fixes the on-axis field B_QS = B0 t and its full gradient ∇B_QS,
  after solving the σ equation (a periodic Riccati ODE) for σ(φ) and ι.
* Objective: ∫_axis |B_coils − B_QS|² + ∫_axis |∇B_coils − ∇B_QS|² + ((ι − ι0)/ι0)² + regularization
  (coil/axis length, curvature, coil-coil distance). Exact gradients by forward/adjoint sensitivity; BFGS.
* At the optimum the coils' real magnetic axis coincides with the expansion axis and the field is QS near it.
* **No boundary, no equilibrium solve.** Everything is vacuum. Afterwards they trace field lines (Poincaré)
  and compute flux surfaces + Boozer coordinates **of the coil field** to report ι(r), the |B| spectrum and QS
  on surfaces of chosen aspect ratio (Figs. 6-8). The aspect ratio is an outcome, set by which coil-field surface
  you call the boundary.
* Limits they state: vacuum only; QS controlled only near the axis (degrades outward); simple engineering terms.
* Follow-ups (from memory, not in this folder): Giuliani et al. JPP 2022 (surfaces of a coil field directly in
  Boozer coordinates); Giuliani et al. PoP 30, 042511 (2023) (start from near-axis-optimized coils, then optimize
  coils for QS/ι/volume on the coils' own surfaces at finite radius); QUASR (2024) is that pipeline at scale.

## Rules we agreed on (the "spirit of the papers")

1. **Coils are the primary object; the plasma is whatever the coils' field makes.** Vacuum throughout.
2. **The NAE is used only as (a) the on-axis target the coils are fitted to and (b) an initial guess.**
   It never defines a boundary in the final evaluation.
3. **The scoring solve has no NAE term.** A vacuum free-boundary solve with given coils and enclosed flux Ψ has
   (at most) one answer: the coils' own flux surface. Adding NAE fidelity would over-determine it and make the
   result stop being the coils' field. NAE-vs-solved (axis, ι, on-axis elongation) is a **diagnostic**.
4. **Guard against fake surfaces.** DESC assumes nested surfaces; if the coils make islands/chaos it returns a
   smoothed fake. Every scored configuration gets a Poincaré check with the coils.
5. **Drift is allowed only in the refinement stage** (the PoP 2023 step), and is monitored: if the axis drifts
   back toward a large vertical excursion, add the near-axis term as a weak regularizer or DESC's
   `FixNearAxisR/Z/Lambda`.
6. The first-order-ellipse boundary + fixed-boundary solve (`to_desc.py` as first written) is **not** in this
   spirit. Keep it only as a rough, comparable indicator between cases.

## The coil class (exactly)

Per unique coil (nc per half period): hinges Ha (inboard), Hb (outboard) **at z = 0**, given as (R, φ) each.
Upper arc x(t) = Ha + t c + w_u(t) u_u, lower arc x(t) = Hb − t c + w_l(t) u_l, t ∈ [0, 1], c = Hb − Ha,
w(t) = |c| Σ_k a_k sin(kπt), u = unit vector ⟂ c at tilt α from the vertical (separate α_u, α_l).
Each arc lies in span(c, u), so a coil is exactly two planar faces joined at the two z = 0 hinges (C0 corners).
This is the same "hinge + tilt + transverse sine" family as DESC's `PiecewisePlanarArcCoil` (B = 2), and converts
to it exactly up to fit/quadrature error. Full set: stellarator image (x, −y, −z) with current −I, then nfp
rotations. `--mode free` puts the hinges at any (R, φ, z): the free-hinge arcB2 control.

## Code (`na/`)

| file | what | state |
|---|---|---|
| `nae.py` | first-order NAE in JAX: σ equation (Newton + one differentiable step = exact implicit derivative), B_QS, ∇B_QS (Cartesian, G[i,j] = ∂_i B_j), helicity count | matches pyQSC to 1e-9 (QA r1 §5.1, QH r1 §5.2); σ0 ≠ 0 and I2 ≠ 0 are not supported (not needed) |
| `folded.py` | the coil class above, exact polyline Biot-Savart, ∇B by jacfwd, arc curvature | loop-centre field and div/curl checked |
| `run_na.py` | the paper's objective as least squares (scipy TRF, JAX Jacobian). Penalties: coil-axis distance `--dpa`, coil-coil `--dcc`, curvature `--kmax`, length `--Lmax`, elongation `--welong/--emax`, currents `--imax` (each I ∈ [0, imax × uniform]) | runs; see Results |
| `to_desc.py` | pyQSC → `Equilibrium.from_near_axis(r)` → fixed-boundary solve; coils → `PiecewisePlanarArcCoil` CoilSets; scores with `common.evaluate` (B·n, d_pc, ...) + Boozer QS at ρ = .25/.5/.75/1 (as `single_stage.py` Physics) | runs on GPU; **to be replaced by the free-boundary scorer (Phase 2)** |
| `test_nae.py` | NAE vs pyQSC check (`python test_nae.py pylib`) | passes |
| `pylib/qsc` | pyQSC 0.1.3, vendored (not installed in desc-env/desc-dev) | |
| `chain_qa.sh`, `chain_qh.sh` | the runs below | |
| `runs_snapshot/` | npz/json/logs of the runs (copied from the session scratchpad) | |

Units: B0 = 1, axis length 2π (so R0 ≈ 1); currents in μ0/4π units (DESC amps = I / 1e-7). Run near-axis jobs on
CPU (`JAX_PLATFORMS=cpu`, tiny); DESC scoring on GPU (`boot.setup(default="gpu")`), one GPU job at a time, under
`systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`.

## Results so far (2026-10-09)

Near-axis stage. **None converged**: every run stopped at its evaluation cap (1500-1700), so differences between
single cold runs are partly basin noise.

| run | cost | max ΔB | max Δ∇B (rel) | ι (target) | max elong. | axis R1 / Z1 |
|---|---|---|---|---|---|---|
| QA nfp2 nc4 z0 (`qa_z0b`) | 6.1e-3 | 1.8% | 3.6% | 0.410 (0.42) | 5.6 | 0.21 / −0.07 |
| QA free (`qa_free`) | 9.4e-4 | 0.2% | 1.2% | 0.419 | 4.0 | 0.18 / −0.16 |
| QA z0 → freed (`qa_z0b_to_free`) | 2.3e-4 | 0.1% | 0.8% | 0.420 | 7.4 | 0.21 / −0.08 |
| QH nfp4 nc4 z0 (`qh_z0`) | 1.4e-3 | 0.6% | 1.4% | 1.243 (1.25) | 4.9 | 0.10 / −0.10 |
| QH free (`qh_free`) | 2.8e-3 | 1.3% | 1.7% | 1.210 | 3.7 | 0.13 / −0.14 |

QA bounds: dpa 0.35, L 3.6, κ 10, d_cc 0.08. QH: dpa 0.28, L 2.93, κ 13.7, d_cc 0.0585 (precise_QH's Gil set).
QH z0 reached ι 1.24 on a helix ~40% smaller than precise_QH's (≈0.16 / 0.13), but with unbalanced currents
(one reversed coil: −0.28 vs +0.03..+0.18).

Rough indicator only (first-order-ellipse boundary, fixed-boundary solve, not the agreed scoring):

| run | a | A | Boozer QS ρ .25 / .5 / .75 / 1 | ι range (NAE) | B·n | d_pc |
|---|---|---|---|---|---|---|
| QA z0 | 0.173 | 5.3 | 2.4e-2 / 2.0e-2 / 3.3e-2 / 6.4e-2 | 0.39-0.42 (0.41) | 3.5e-2 | 0.166 |
| QH z0 | 0.131 | 6.8 | 3.1e-2 / 3.2e-2 / 4.6e-2 / 8.3e-2 | 1.12-1.27 (1.24) | 1.6e-1 | **0.073 (< 0.11 bound)** |

References: precise_QA free arcB2 B·n ≈ 2e-3; precise_QH z0 arcB2 1.8e-2 (fixed eq), 6.4e-3 (single stage);
free arcB2 5.4e-3 / 1.5e-3 (`../work/ss_qh/joint/results/RESULTS.md`, `../work/SINGLE_STAGE_HANDOFF.md`).

### Sweep 1 (run_na2, z0, currents in [0, 2x], NAE-surface clearance, d_cc 0.04, elongation <= 6, 4 seeds/case)

Bounds: Gil set scaled by a (d_pc 0.885 a, kappa 1.699/a, L 23.6 a). `na/sweep1.sh`, table: `python na/summarize.py`.

| case (coils) | seeds | cost range | B / gradB mismatch (best) | iota (target) | elong. | axis abs(Z) | verdict |
|---|---|---|---|---|---|---|---|
| QA nfp2 nc4 (16), a 0.167 | 4/4 | 4.3e-4 .. 8.4e-4 | 0.2% / 1.1% | 0.420 (0.42) | 4.5-5.1 | 0.074-0.087 | good, consistent |
| QA nfp3 nc4 (24), a 0.143 | 4/4 | 3.4e-5 .. 1.6e-3 | <0.05% / 0.3% | 0.400 (0.40) | 3.4-5.5 | 0.022-0.029 | best so far; nearly flat axis |
| QH nfp3 nc4 (24), a 0.143 | 4/4 | 1.2e-2 .. 3.7e-2 | 2.1% / 4.3% | 1.01-1.05 (1.10) | 5.4-6.0 (cap) | 0.117-0.157 | poor: iota short, cap binding |
| QH nfp4 nc4 (32), a 0.124 | 4/4 | 6.1e-3 .. 5.1e-2 | 4.1% / 2.2% | 1.18-1.22 (1.25) | 4.1-6.0 | 0.076-0.114 | one good basin (s2, = pass-1 nc4); others pinned on current/elong. bounds |
| QH nfp5 nc3 (30), a 0.110 | 4/4 | 2.2e-3 .. 3.1e-2 | 2.6% / 1.7% (s2); 0.8% / 2.1% (s3) | 1.15-1.31 (1.30) | 2.6-6.0 | **0.045-0.102** | best QH: s2 hits iota with abs(Z) 0.052 and near-uniform currents (0.88-1.21x) |

All 20 meet every bound (the bound rows are satisfied; poor runs are poor FITS, not infeasible). Many runs stopped on
ftol after 80-160 iterations with optimality ~1e-5, so the QH ones are not fully converged.

Incidents (fixed): xargs stripped quotes (all runs died on argparse; use `xargs -d '\n'`); fixed-size starts
(rcoil 0.42) were infeasible for QA (coils 8 mm from the surface) and stuck the solver in an infeasible corner
-> starts are now sized from the NAE cross-section (chord = R range + 2.4 d_pc each side, sine arches clearing
by 1.6 d_pc, randomized starts redrawn until elongation <= cap); 32-coil runs hit the 10 GB cap (OOM) because the
B/gradB Jacobian was vmapped over axis points -> `lax.map` (peak 1.4 GB).

## Problems found (to fix before reading anything into comparisons)

1. **Coil-axis distance is the wrong clearance.** With elongation ~5, the a = 0.124 surface reaches ~0.28 from
   the axis, so dpa = 0.28 let QH coils sit almost on the plasma (d_pc 0.073). Replace with the distance from the
   coils to the NAE surface at r = a: x0 + a (X1 n + Y1 b), which `nae.py` already has the pieces for. (Using the
   NAE surface for a clearance penalty is consistent with rule 2: it is part of fitting the coils, not a boundary.)
2. **Current imbalance** (QH): reversed/large currents can match the axis while doing badly off-axis. `--imax 2`
   added; first constrained runs queued.
3. **Not converged / basin noise:** need longer runs or a better stopping rule, and several starts per case
   before comparing z0 with free.
4. Coil conversion error 0.5-0.9% (polyline NPT = 80 vs DESC's smooth arcs). Fine for screening; raise NPT for
   final runs.

## Plan

### Phase 1: near-axis stage done properly (CPU, minutes per run)
* Coil-to-NAE-surface clearance at r = a (problem 1) replaces `--dpa`; current bound; elongation cap (~4).
* Converge: more evaluations, then stop on relative cost change; report optimality.
* Multi-start: ~5-10 seeds per case (axis amplitude, η̄, coil tilt/arch heights). Report best and spread.
* Cases: QA nfp 2 (ι 0.42, A 6) and QH nfp 4 (ι 1.25, A 8); z0 and free, so the joint-plane cost is measured
  at the near-axis level from matched multi-start statistics.
* Optional knob: penalize the axis vertical excursion max|Z_axis| directly, to map ι/QS vs excursion for z0.

### Phase 2: honest scoring, the coils' own surfaces (GPU, one job at a time)
* Vacuum **free-boundary** DESC solve with the converted coils as the external field: p = 0, net current 0,
  Ψ set for the target a. NAE surface as the initial guess only (rule 3). Sandbox piece:
  `sandbox/bnormal.py` (`VacuumBoundaryBn`).
* Report: Boozer QS (ρ = .25/.5/.75/1), ι profile, A; axis/ι/elongation vs the NAE prediction (diagnostic);
  coil metrics vs bounds (`common.evaluate`).
* Poincaré check with the coils at several Ψ (rule 4).
* Optional better initial guess: pyQSC `order='r2'` with B2c chosen to minimize second-order QS breaking.

### Phase 3: refinement in the spirit of PoP 2023 (needs the user's OK: long runs)
* Goal: optimize the coils for QS and ι measured on the coils' own surfaces at finite radius, keeping z = 0 hinges.
* DESC routes: (a) free-boundary-in-the-loop single stage (new code; the plasma is the coils' field by
  construction); (b) pragmatic: `run_al.py --fix-hinge-z 0` then `single_stage.py --fix-hinge-z 0` from
  (Phase 2 equilibrium, converted coils). Note: (b) couples a fixed-boundary eq to the coils by a B·n penalty,
  unlike the papers.
* Monitor axis drift (rule 5). Compare with the precise_QH joint-plane numbers above.

### Phase 4: scale (QUASR-style), only if Phases 2-3 look good
* Sweep nfp, nc, ι, A with many seeds; rank by Phase 2 scores; a viewer like `../view.py`.

## Decisions, 2026-10-09 (later)

* **Focus on the near-axis stage for now:** build a wide library of axes + coil candidates before Phase 2.
* **Currents stay constrained:** each current in [0, 2 x uniform] (no reversed coils).
* **Clearance to the NAE surface at r = a** (`--a`, `--dpc`), not to the axis.
* **More coils allowed** (nc = 5, 6 per half period; QH nfp 4 so 40 / 48 coils), with a looser coil-coil
  distance (`--dcc 0.04` vs precise_QH's 0.0585) if needed to fit them.
* **Solver: the branch's composite least squares** (`lsq_composite`, hinges exact in the model, secant
  Hessian), called directly with the prototype's rows: `na/run_na2.py`. `run_na.py` (scipy TRF + squared
  hinges, which stalled at active bounds) is kept only to reproduce the first runs.

## Decision (2026-10-09, user): use the branch's objectives wherever DESC objects exist

* **Phase 2 and later must use the branch's machinery**, not hand-rolled rows: `PlasmaCoilSetDistanceRows`,
  `CoilSetDistanceRows` (smooth per-contact rows with a gap, stable `row_ids`), the composite solvers
  (`lsq-composite`, `lsq-auglag-composite`), `keep_rows`; see `../../devtools/coil_auglag/GUIDE.md`. The branch
  built these specifically to avoid the stalls the prototype hit with kinked min-distances.
* Near-axis stage (`run_na2.py`): its soft-min distances are a stopgap; port the branch's row scheme (candidate
  pairs within `select_distance`, gap, stable ids passed as `row_ids` to `lsq_composite`) before the sweep-1 QH
  reruns. The regularizer ladder (`reg1.sh`) runs to completion on the soft-min version.

## Decision gates
* After Phase 2: do the best z0 seeds have nested surfaces and QS meaningfully better than the perturbed-precise_QH
  route's starting point? If not, check whether the gap is first-order truncation (try r2 guess) or coil ripple.
* After Phase 3: does z0 close to within ~1.5× of free arcB2 on its own boundary (vs 4.3× from precise_QH)?

## Status
* Phase 0 (prototype + validation): done.
* Constrained QH with the old solver (`qh_z0_c`, `qh_z0_c_warm`: currents in [0, 2x], elongation <= 4): both
  stalled (scipy status 3, optimality 0.18) at 15-21% field mismatch, ι 1.10-1.16, with one coil's current
  pinned at 0 (it wanted to reverse). Not conclusive: solver stall + old clearance.
* Pass 1 nc = 4 (`runs2/qh_z0_nc4`): QH nfp4 z0 meets every bound with positive currents (0.31-1.62x), mismatch
  3.5% / 2.4%, iota 1.225, axis abs(Z) 0.077 (~60% of precise_QH), elong 5.3. nc = 5, 6 dropped (user: 32 coils is enough).
* Sweep 1 done (table above).
* Report: `REPORT.md`; viewer: `viewer.html`; agent guide: `CLAUDE.md` (2026-10-09).
* Found: coils link the axis once (all 21 configs) but are poorly regularized (length/curvature at bounds).
* Soft-min distances (kinked argmin distances stalled sweep-1 QH runs; rerun them). Coil regularization started on
  QA nfp2 s2 and QH nfp5 s2 (`na/reg1.sh`: w = 0, 1e-5, 1e-4, 1e-3 for length + bending).
* Next (proposed, REPORT §8): coil regularizer; Phase 2 free-boundary scorer; free-hinge counterpart of sweep 1;
  more QH seeds.
