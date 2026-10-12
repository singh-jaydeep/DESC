# precise_QH coil study: summary of runs 2026-10-07 → 10-10

What we ran, in order, with the settings, what each step taught us, and where its viewers and reports live. The last
two sections list the best coilsets for z = 0 hinges and level hinges, and assess whether any are ready to publish.

**Defaults, unless a row says otherwise:**
* QH, 4 field periods, vacuum.
* Coil bounds: precise_QH's (Gil Table III, scaled to the minor radius; `bounds/precise_qh_gil*.json`).
* 4 coils per half-period (32 coils).
* Equilibrium resolution L = M = N = 8.
* Stage 2: `run_al.py`, the slack-free augmented-Lagrangian solver. "Converged" means it stopped on gtol, with
  optimality and violation < 1e-6.
* Single stage: `single_stage.py` (proximal-lsq-exact), boundary-mode steps k = 1–4 plus a k4b step, then a stage-2
  refine on the final boundary.

**Definitions:**
* **B·n:** ⟨|B·n|⟩/⟨|B|⟩ from the dense check on the target boundary.
* **QS:** Boozer |B_nonsym|/⟨B⟩.
* **Removable:** the exact vertical lift-out test (`sandbox/demount.py`). It is checked after each run and never
  constrained.

Detailed timeline: `joint/NOTE.txt`, `matrix/NOTE.txt`.

---

## 1. Boundary × coils matrix (2026-10-07/08)

**Files:** `matrix/RESULTS.md`, `matrix/matrix.md`. **Viewer:** `matrix/view.html`.

**Settings:**
* Three coil representations: planar N7, arcB2 M5 and FourierXYZ N6.
* For each one, a single stage from precise_QH with identical weights: B·n tolerance 5e-3, and the QS tolerance taken
  from the start equilibrium.
* Then a converged stage 2 for every representation on every boundary: 12 cells, all gtol.
* Every representation got an extra k4b step (ftol = xtol = 1e-12, anchors frozen).
* arcB2 used `--cc-k 80000`.
* The XYZ single stage needed `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.

**Result** (B·n; the factor in parentheses is relative to the same coils on precise_QH):

| boundary ↓ / coils → | planar N7 | arcB2 M5 | XYZ N6 | edge QS vs precise_QH |
|---|---|---|---|---|
| precise_QH | 2.205e-2 | 5.437e-3 | 9.332e-4 | 1× |
| B_planar | **6.866e-3 (0.31)** | 2.335e-3 | 1.098e-3 | 6.9× worse |
| B_arcB2 | 1.687e-2 | **1.484e-3 (0.27)** | **5.171e-4** | 3.3× worse |
| B_xyz | 2.166e-2 | 3.079e-3 | 6.260e-4 | 1× |

**Learned:**
* Planar and arcB2 do best on their own single-stage boundary. XYZ does not: its B·n already sat inside the
  tolerance, so its single stage barely moved the boundary.
* B_arcB2 helps all three representations.
* The gains are paid for in QS.
* Reference values used below: free arcB2 on B_arcB2, 1.484e-3; free arcB2 on precise_QH, 5.437e-3.

---

## 2. All hinges on one plane, z = 0, on precise_QH (2026-10-08)

**Files:** `joint/results/RESULTS.md`, `joint/base_*`, `joint/ss_z0*`, `joint/on_z0*`. **Viewer:**
`joint/results/view.html`.

**Settings:**
* `run_al.py --fix-hinge-z 0`: hinge z is snapped to 0 and fixed exactly.
* Start: a constructed "lens" (`make_z0_start.py`, arches h·sin(πt)^0.7, clearance 2·d_pc), repacked into one
  CoilSet.
* Link guard `--link-mu 1e8`: an initial penalty on the signed per-pair coil–coil rows only. It uses the solver option
  `initial_penalty_rows`, committed in 31ed99491 together with a scaling fix.
* The single stage used the matrix arcB2 weights plus `--fix-hinge-z 0`, then a k4b step and a refine.

| run | boundary | B·n | notes |
|---|---|---|---|
| `base_free` (no guard) | precise_QH | 2.11e-2 | **coils 0 and 1 linked** in outer iteration 1; the run stalled |
| `base_free_link` | precise_QH | 6.421e-3 | guard on; converged, never linked |
| `base_z0` | precise_QH | 1.801e-2 | **2.80×** free from the same start; removable |
| `on_z0/arcB2_z0` | B_z0 (single stage) | 6.372e-3 | **4.3×** free arcB2 on B_arcB2; QS at ρ 0.5 17× precise_QH's |
| `on_z0_k5` | B_z0 at k = 5 | 6.356e-3 | stalled on xtol; no gain |

**Learned:**
* precise_QH's helical axis swings far vertically. At φ = 28° the whole cross-section lies below z = 0, so arcs hinged
  at z = 0 must be D-shaped there.
* Single stage does not remove that excursion (the maximum |z| of the section centre went from 144 to 156 mm). So the
  gap stays open.

---

## 3. Level hinges, each coil at its own height (2026-10-08)

**Files:** `joint/base_level`, `joint/ss_level*`, `joint/on_level`, `joint/results/K5_STALL.md`. **Viewer:**
`joint/results/view.html`.

**Settings:** `--horizontal-hinges` (`sandbox/demount.HingeLevel`, linear) in both `run_al.py` and
`single_stage.py`, with the same weights as the z = 0 run.

| run | B·n | ÷ free | removable |
|---|---|---|---|
| `base_level` (fixed precise_QH) | 8.768e-3 | 1.37× | yes (+1.9 cm) |
| `on_level/arcB2_level` (single stage) | 2.859e-3 | 1.93× | yes (+4.7 cm) |

**Learned:**
* Letting each coil sit at its own height recovers about 80% of the z = 0 cost. The rest is the cost of horizontality
  itself: free arcB2 chords tilt 37–80°.
* QS stays close to B_arcB2's, and the minimum ι is 1.187.
* Two intermittent GPU segfaults occurred; identical reruns passed both times.
* The k = 5 stall is documented in `K5_STALL.md`.

---

## 4. Other equilibria, fixed equilibrium (2026-10-08)

**Files:** `joint/quasr1328095/`, `joint/devices/`. **Viewer:** `joint/results/view_devices.html`. Its margins are
always measured against precise_QH's bounds.

**Settings:**
* precise_QH bounds, 4 coils per half-period, arcB2 M5.
* Each free control was warm-started from the level result, so level ÷ free is a lower bound.
* Ψ was rescaled to precise_QH's ⟨|B|⟩.

| equilibrium | ι | free | level | level ÷ free | level removable |
|---|---|---|---|---|---|
| precise_QH | 1.25 | 5.437e-3 | 8.768e-3 | 1.61 | yes |
| **QUASR 1328095** | 1.31 | **2.693e-3** | **5.644e-3** | 2.10 | no (5% of an arc blocked) |
| QUASR 1632037 | 1.40 | 4.510e-3 | 7.672e-3 | 1.70 | no |
| QUASR 1632468 | 1.70 | 6.340e-3 | 8.908e-3 | 1.41 | no |
| QUASR 1775228 | 2.00 | 5.726e-3 | 1.117e-2 | 1.95 | no |
| WISTELL-A (mirrored, scaled) | 1.06 | 1.346e-2 | 1.457e-2 | 1.08 | yes |

**Learned:**
* 1328095, the one device chosen for arcB2 geometry, is the best.
* Removability fails on every QUASR level result.
* A cold free start on 1328095 collapsed into a poor basin (B·n 0.10).

---

## 5. Overnight 2026-10-08/09

**Files:** `joint/results/REPORT.md` §2–5, `joint/cont/`, `joint/quasr1328095/nc5/`, `joint/z0screen/`,
`joint/qa_z0/`.

### Coil continuation (`cont/cont.sh`)

**Settings:** level hinges, ramping the length bound ×0.80 → 1.0 and the arc modes M 3 → 5 together with the boundary
step k = 1 → 4.

| step | length × | M | B·n | QS ρ 0.5 | ι min |
|---|---|---|---|---|---|
| k = 1 | 0.80 | 3 | 1.63e-2 | 5.3e-3 | 1.239 |
| k = 3 | 0.933 | 5 | 3.50e-3 | 8.6e-4 | 1.194 |
| k = 4 / k4b | 1.0 | 5 | 2.44e-3 | 7.0e-4 | 1.208 |
| **refine** | 1.0 | 5 | **2.2661e-3** | | |

**Result:** 21% better than `ss_level`, and 1.53× free arcB2. It is removable (+7.4 cm). It ends on tolerances, not
maxiter.

**Caveats:**
* Only one schedule was run, so the effects of the length ramp and the mode ramp cannot be separated.
* Free hinges never got the same continuation.

### 1328095 at its own design

**Settings:** 5 coils per half-period with QUASR's bounds (`bounds/quasr1328095_own.json`).

**Result:**
* A feasibility-only solve (`--obj-weight 1e-6`) reaches a feasible point in 5–7 inner iterations.
* The B·n solves stall, even from the feasible point. With L, κ, d_cc and d_pc all binding, the inner subproblems stop
  converging.
* So the stall belongs to the solver, not the geometry.

### z = 0 screen (613 QUASR boundaries)

* **QH:** z = 0 misses at least one cross-section on every device.
* **QA:** z = 0 suits only boundaries with ι ≤ 0.2.
* **precise_QA:** z = 0 costs 5.4× and level 3.8×.
* **QUASR 804112** (QA, ι 0.2) has the cheapest plane: 1.70× free, removable. Its absolute B·n is poor (free 8.9e-3).
* The R-extreme height (`ext_med`) predicts the cost of a z = 0 plane; the cut fraction alone does not.

---

## 6. Equilibrium resolution 12 on the continuation result (2026-10-09)

**Files:** `joint/hires/`, `joint/results/REPORT.md` §7.

**Settings:**
* `upres.py` (resolution 8 → 12, grids 24).
* k4b capped at 30 iterations, with `--eq-jac-chunk 300` (the automatic chunk ran out of GPU memory).
* Same refine flags as before.

**Result:**
* Refined B·n 2.2746e-3, against 2.2661e-3 at resolution 8.
* QS at ρ 0.5 improved by 18% (5.76e-4) and ι by a little.
* The coils are unchanged.

**Decision:** keep resolution 8. When a resolution-8 step stalls, a resolution-12 step is the first fix to try; it
costs about 13× more per evaluation.

---

## 7. Field quality of the actual coils (2026-10-09)

**Files:** `joint/fieldcheck/FIELDCHECK.md`. **Figures:** `fieldcheck/{level,free,z0}/poincare.png`, `boozer*.png`,
`qs_profile.png`, `fieldcheck/level_vs_free.png`.

**Method** (`pipeline.sh`):
1. Trace the coil field for 400 field periods.
2. Fit the traced ρ0 = 0.95 surface.
3. Deform the target equilibrium onto it and re-solve at resolution 12.
4. Compute Boozer QS and ε_eff with DESC `EffectiveRipple`.

The control and a direct Biot–Savart Boozer fit agree with this to 1–6%. Free-boundary solves were dropped because
they were unreliable for this.

**Learned:**
* **Level coils (`cont/refine`):** nested surfaces with no islands; ι within 0.002 of the target. Coil-field QS is
  2.4–3.5× the target's: 1.95e-3 against 6.9e-4 at ρ 0.5. About 70% of the excess is the 32-coil (0, 8) ripple.
  ε_eff stays below 0.35%.
* **Free arcB2:** coil-field QS is 2.5× its target's at ρ 0.5.
* **z = 0 coils:** nested surfaces, but ρ wanders 3× more than for level, and edge ι is low by 0.025.

---

## 8. Overnight 2026-10-09/10: which coilset has the best QS?

**Files:**
* Report: `joint/fieldcheck/BEST_QS.md`.
* Screen: `fieldcheck/spectrum/screen_table.md`.
* Figure: `fieldcheck/best_coilsets.png`.
* Case directories: `fieldcheck/{xyzX,xyzB,xyz5,xyz4L}/`.
* Runs: `nc5xyz/`.

### Would a ripple term help? (`spectrum.py`)

Mostly no:
* 86% (level) and 65% (free) of the squared B·n sits in the ripple band.
* That band transfers to |B| symmetry-breaking about 1:1. Lower-order modes transfer at only 0.2–0.5.
* So B·n already targets the ripple.

### Screen of 29 coilsets

**Method:** the quick proxy check, i.e. coil |B| on the target's surfaces, fitted in the target's Boozer angles. It
takes about 1 minute per coilset, reads 5–40% below the traced value, and keeps the same ranking.

**Learned:** the ranking is XYZ < arcB2 < planar, and it follows B·n.

### Full field checks (traced QS at ρ 0.25 / 0.5 / 0.75 / 1; ε_eff at ρ 0.5 / 0.9)

| case | QS | ε_eff |
|---|---|---|
| level (`cont/refine`) | 1.80 / 1.95 / 2.86 / 5.71 e-3 | 4.2e-4 / 1.29e-3 |
| free arcB2 | 0.98 / 1.73 / 2.65 / 5.25 e-3 | 1.51e-3 / 3.44e-3 |
| xyzB (XYZ on precise_QH) | 5.8 / 6.8 / 9.1 / 21.1 e-4 | 3.1e-4 / 4.3e-4 |
| **xyzX** (XYZ on B_xyz, `matrix/on_xyz/xyzN6`) | 2.8 / 4.0 / 6.1 / 15.6 e-4 | 1.3e-4 / 1.8e-4 |
| xyz4L (XYZ, L ×1.25) | 2.5 / 5.0 / 5.7 / 15.2 e-4 | 3.1e-4 at ρ 0.5 |
| xyz5 (XYZ, 5 coils/hp, L ×1 per coil) | 1.65 / 2.8 / 4.0 / 13.0 e-4 | 1.3e-4 / 2.2e-4 |

**Caveat for XYZ:** the target's own coordinates fit these coil fields as well as or better than the traced
reconstruction does, so the traced values over-read. The truth lies between the proxy and the traced value, at
ρ 0.5:
* xyz5: 1.7–2.8e-4
* xyz4L: 2.0–5.0e-4
* xyzX: 3.8–4.0e-4

**Risk:** B_xyz's edge ι is 5/16 per field period with near-zero shear.

### Length versus coil count (stage 2, cold or warm starts)

**Settings:** `--cc-k 30000` for XYZ, 80000 for arcB2; `--link-mu 1e8`; arcs use `--cc-gap coils`.

| run | coils/hp | length per coil | B·n | proxy QS ρ 0.5 |
|---|---|---|---|---|
| `xyz4_cold` (B_xyz) | 4 | ×1.0 | 6.38e-4 | 3.95e-4 |
| `xyz5_Lfix` | 5 | ×0.8 (same total) | 8.78e-4 | 3.71e-4 |
| `xyz4_L125` | 4 | ×1.25 | 2.98e-4 | 2.03e-4 |
| `xyz5` | 5 | ×1.0 (+25% total) | **1.91e-4** | **1.68e-4** |
| `arc5` (B_arcB2, free) | 5 | ×1.0 | 6.60e-4 | 7.72e-4 |
| `arcL125_free` (B_arcB2) | 4 | ×1.25 | 6.12e-4 | 7.61e-4 (target 7.11e-4) |
| **`arcL125_level`** (B_cont) | 4 | ×1.25 | **1.285e-3** (−43%) | **1.08e-3** (−39%) |

**Learned:**
* Conductor length is the lever.
* A fifth coil adds nothing at fixed total length. With the extra length it helps XYZ a little and arcB2 not at all.
* With +25% length, free arcB2 is limited by its target, not its coils.
* Length and κ bind in every run; κ ends 0.04–0.1% over its bound.

**Not done: `arc5_level`** (5 coils/hp, level hinges). It ran out of memory three times during setup: twice on the GPU
(once with a 0.92 memory fraction), then at the 12 G cap on CPU. The memory goes into the coil–coil node buffer `d0`,
which holds pairs × N² float64 values. A fix needs a DESC change (float32 `d0`, or a per-pair min/argmin) or a coarser
node grid. That is the user's call.

**No 3D viewer yet** covers the `nc5xyz/` runs.

---

## 9. Best configurations now

| | **Level hinges, L ×1.25** | **Level hinges, nominal L** | **z = 0 hinges** |
|---|---|---|---|
| path | `nc5xyz/arcL125_level` | `joint/cont/refine` (res-12 twin: `joint/hires/refine`) | `joint/on_z0/arcB2_z0` |
| boundary | `cont/k4b` | `cont/k4b` | `ss_z0` k4b |
| config | arcB2 M5, 4 coils/hp, `bounds/precise_qh_gil_nc4_L125.json`, warm from `cont/refine`, `--horizontal-hinges --link-mu 1e8 --cc-gap coils --cc-k 80000` | continuation L ×0.8 → 1.0, M 3 → 5 with k, then refine (same flags) | `--fix-hinge-z 0`, matrix arcB2 single-stage weights, then refine |
| B·n | **1.29e-3** | 2.27e-3 | 6.37e-3 |
| target QS ρ 0.5 / edge | 6.9e-4 / 2.9e-3 | 6.9e-4 / 2.9e-3 | 3.7e-3 / 8.0e-3 (17× precise_QH) |
| coil-field QS ρ 0.5 | 1.08e-3 (proxy only) | 1.95e-3 traced (proxy 1.77e-3) | 6.8e-3 (proxy) |
| ε_eff ρ 0.5 / 0.9 | not run | 4.2e-4 / 1.3e-3 | not run |
| surfaces | not traced | nested, no islands, ι within 0.002 | nested, ρ wanders 3× more, edge ι −0.025 |
| removable (hinge clearance) | yes (+4.1 cm) | yes (+7.4 cm) | yes (+14.4 cm) |
| ÷ free arcB2 | 2.1× (free at L ×1.25: 6.1e-4) | 1.53× | 4.3× |
| constraint state | L, κ active; κ 1.001 | L, κ active | all five active |

**Reference, the best coil field of any kind:**
* xyz5 (XYZ, 5 coils/hp, +25% length): B·n 1.9e-4, QS 1.7–2.8e-4 at ρ 0.5.
* At nominal length, xyzX: B·n 6.3e-4, QS about 4e-4, ε_eff ≤ 1.8e-4.

---

## 10. Publishable?

**z = 0:** not as a positive result (4.3× free, with QS 17× worse). As a negative result it is solid: the
vertical-excursion argument, the 613-boundary screen and the precise_QA comparison could form one section of a paper.

**Level hinges:** close, but not yet.

*For:*
* At L ×1.25, B·n 1.3e-3 is comparable to QUASR 1328095's own coils (9.9e-4 on that equilibrium, measured here).
* L ×1.25 is about 3.66 m per coil, matching QUASR designs (3.6–3.7 m), while our plasma–coil distance bound is
  stricter (0.11 m vs 0.15–0.21 m).
* The coils are removable with nested surfaces.
* Coil continuation is a reusable method result.

*Missing:*
1. A single stage with continuation at the chosen length. The L ×1.25 result is stage 2 on the nominal-length boundary.
2. The full field check on that result: traces, traced QS and ε_eff.
3. Strict κ feasibility: a small backoff, or a finer enforcement grid.
4. More than one start, and free hinges run with the same continuation, so the level ÷ free ratio is fair.
5. More than one equilibrium, or a stated reason why precise_QH alone suffices.
6. A check of literature numbers before claiming "competitive". The QUASR comparison is the only one measured here.

**Most efficient next step:** one chain covering items 1–3. Sweep the length bound at ×{1.0, 1.1, 1.25}, run the
continuation single stage at the chosen length, then the field check. About one evening of GPU time.

## Open items

* `arc5_level`: blocked on DESC memory (the `d0` buffer); needs a decision.
* Length-budget decision.
* `K5_STALL.md`: single-stage k = 5 stall (Gauss–Newton; try a secant Hessian or resolution 12).
* Removability is not constrained. `HingeShadowClearance` and `CoilDemountPenalty` exist in `demount.py` but are not
  wired into `run_al.py`.
* No viewer for the `nc5xyz/` runs.
