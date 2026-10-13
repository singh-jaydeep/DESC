# Best z = 0 two-face-coil QA designs so far (reference for anyone trying to beat them), 2026-10-13

QA, nfp 2, 16 coils (4 per half period), each coil two planar faces, **every hinge on z = 0**, all scaled engineering
bounds met. The coils' own vacuum field has nested surfaces at aspect ratio 6.9 with the largest symmetry-breaking Boozer
mode at 0.2-0.7% of B00. This file says where it is, how it was made, how it scores, what is not yet trustworthy,
and what a better result has to show. Details and every dead end: `ss/NOTES.md` (single stage), `REPORT.md` sections
10-11 and `NAE_HANDOFF.md` (near-axis stage).

## 00. Update 2026-10-13: the "ss4 redux" (design C) and a literature-style comparison of all three

Design C answers one question: was B better than A because of its lower iota, or because its single stage ran on a
clean boundary? C is A's near-axis parent (iota target 0.42) put through B's exact pipeline, with the new boundary
repair (`../sandbox/bdry_repair/README.md`). **C lands near A, not near B: most of B's gain came from the lower iota.**

All numbers below are on vacuum free-boundary equilibria of the final coils (no QS term), in the literature's
conventions. B and C are quoted on their repaired surfaces (positive Jacobian everywhere, no crease); the repair moved
them 0.09 and 0.07 mm rms and changed QS by under 7%.

| | **A: `ss4`** | **C: redux** (new) | **B: `li030` k=6** | precise_QA | reactor_QA | NCSX | ARIES-CS |
|---|---|---|---|---|---|---|---|
| iota at rho .25 / .5 / .75 / 1 | .352 / .340 / .325 / .313 | .304 / .293 / .307 / .349 | .226 / .218 / .220 / .232 | 0.42 | .49 - .63 | .39 - .66 | .45 - .67 |
| aspect ratio | 6.89 | 6.90 | 7.26 | 6.0 | 6.0 | 4.5 | 4.5 |
| largest breaking Boozer mode / B00, rho .25 | 0.36% | 0.24% | **0.13%** | 0.006% | 0.27% | 0.69% | 0.97% |
| ... rho .5 | 0.20% | 0.25% | **0.14%** | 0.003% | 0.18% | 0.66% | 1.05% |
| ... rho .75 | 0.37% | 0.32% | **0.22%** | 0.003% | 0.29% | 1.30% | 1.43% |
| ... rho 1 | 0.69% | 0.85% | **0.45%** | 0.02% | 0.51% | 2.70% | 2.15% |
| L2 of breaking modes / B00, rho .25/.5/.75/1 | 0.42 / 0.43 / 0.84 / 1.74% | 0.36 / 0.47 / 0.74 / 1.47% | **0.25 / 0.28 / 0.47 / 0.94%** | 0.008 / 0.004 / 0.006 / 0.03% | 0.29 / 0.29 / 0.41 / 0.73% | 0.97 / 1.28 / 1.95 / 3.88% | 1.08 / 1.36 / 1.92 / 3.60% |
| two-term rms(f_C / B^3), rho .25/.5/.75 | 2.4e-2 / 3.2e-2 / 5.9e-2 | 2.4e-2 / 3.2e-2 / 5.4e-2 | 1.6e-2 / 2.2e-2 / 3.5e-2 | 2e-4 / 1e-4 / 1e-4 | 4.4e-3 / 4.5e-3 / 6.5e-3 | 4.6e-2 / 5.3e-2 / 7.6e-2 | 2.3e-2 / 2.9e-2 / 5.1e-2 |
| eps_eff, rho .25/.5/.75 | 1.1e-3 / 1.1e-3 / 2.8e-3 | 8.6e-4 / 1.1e-3 / 2.0e-3 | **6.9e-4 / 7.4e-4 / 1.3e-3** | 4.5e-7 / 3.2e-7 / 1.2e-6 | 2.2e-4 / 8.1e-4 / 2.8e-3 | 2.0e-3 / 1.9e-3 / 3.8e-3 | 6.3e-4 / 2.3e-4 / 8.0e-4 |
| eps_eff^(3/2), rho .25/.5/.75 | 3.7e-5 / 3.4e-5 / 1.4e-4 | 2.5e-5 / 3.6e-5 / 9.1e-5 | 1.8e-5 / 2.0e-5 / 4.6e-5 | 3e-10 / 2e-10 / 1e-9 | 3e-6 / 2e-5 / 1.5e-4 | 9e-5 / 8e-5 / 2.3e-4 | 1.6e-5 / 3e-6 / 2.3e-5 |
| coils' B.n on the surface, rms / max | 0.04% / 9.6% (crease) | 0.23% / 1.5% | 0.10% / 0.44% | (fixed boundary, no coils) | | | |
| coils feasible | yes | yes | length 0.03% over | | | | |

Reference columns are DESC's example equilibria (fixed boundary, no coils), computed with the same scripts.

**Reading it against the literature.**
* QS: B is better than `reactor_QA` on the largest mode at every radius and level with it on the L2; A and C are about
  1.3x worse than `reactor_QA` and 2-4x better than NCSX and ARIES-CS. All three are 20-100x worse than `precise_QA`.
  The two-term measure tells the same ranking, but ours is 4-5x worse than `reactor_QA`'s there (it weights many small
  modes that the largest-mode measure ignores).
* eps_eff: B is 7e-4 - 1.3e-3, between ARIES-CS and NCSX, and level with `reactor_QA` at mid radius. The values are
  converged: doubling the bounce-integral resolution changes them by 0-6% (`ss/ref042r/eps_conv.json`, `ss/eps_conv.py`).
  This supersedes the "not converged" eps_eff row in section 0.
* Poincare (coil field, 6 planes): nested surfaces from the axis to beyond the equilibrium boundary in both B and C,
  no island chains visible inside the boundary. `ss/li030/fb_norepair_local/poincare.png`, `ss/ref042r/fb/poincare.png`.
* C's iota profile is not monotonic (minimum 0.29 near mid radius, 0.35 at the edge); A's and B's are.

**C in detail** (Modal job `ss12_ref042_q15r`, commit d97172432, 77 min on an A100; files in `ss/ref042r/`):
start repair moved the surface 0.12 mm rms and kept QS (0.35/0.67/1.31/2.37% -> 0.29/0.71/1.34/2.37%); the gate then
found no repair needed after any of the six k steps. Single stage k=6: L2 0.39 / 0.42 / 0.69 / 1.42%, iota 0.299-0.352.
Viewer: entry 6 of `ss/viewer_ss.html`.

**What is weak about C (do not skip):** its free-boundary rescore did not converge. Both shells stopped with "a bad
approximation caused failure to predict improvement" and left B.n where it started (0.185% -> 0.187% rms; A reached
0.04%), and the boundary creased inside the last shell (repaired afterwards). So C's score sits close to its
single-stage numbers partly because the surface barely moved, and it is softer than A's. Why the shells stall here is
open; it also bears on B's 0.10-0.14%.

**B's new numbers** come from `ss/li030/repaired/eq.h5` (the section-0 state repaired with the new package): they
replace the "0.25-0.28% / ... / 0.9-1.2%" range quoted in section 0 with one clean state, and B.n on it is 0.10% rms,
0.44% max.

## 0. Update 2026-10-12 (late): a second design, lower iota, better QS

There are now TWO reference designs. They are two points on a QS-versus-transform trade, not a winner and a loser.
Sections 1-6 below describe design A; this section describes design B and compares them.

| | **A: `ss4`** (sections 1-6) | **B: `li030` k=6** (new) |
|---|---|---|
| near-axis parent | `o2w1_qa2s2_gg1e-1_w1e-2_w1x10` (iota target 0.42) | `li_qa2_iota0.30` (iota target 0.30) |
| iota, axis -> edge (coils' own surface) | 0.356 -> 0.313 | **0.228 -> 0.221** (flat) |
| aspect ratio, a, R0 | 6.89, 0.122, 0.843 | 7.26, 0.121, 0.877 |
| largest breaking Boozer mode / B00, rho .25/.5/.75/1 | 0.36% / 0.20% / 0.37% / 0.69% | **0.14% / 0.14% / 0.22% / 0.38%** |
| L2 of breaking modes / B00 | 0.42% / 0.43% / 0.84% / 1.74% | **0.25% / 0.28% / 0.46% / 0.88%** |
| two-term rms(f_C / B^3), rho .25/.5/.75 | 2.4e-2 / 3.2e-2 / 5.9e-2 | 1.6e-2 / 2.2e-2 / 3.4e-2 |
| eps_eff (one resolution, not converged), rho .25/.5/.75 | 1.0e-3 / 1.0e-3 / 2.7e-3 | 7.0e-4 / 7.0e-4 / 1.1e-3 |
| B.n of the coils on that surface, rms | 0.040-0.046% | 0.14% |
| coils: L / kappa / d_cc / d_pc over their bounds | 0.999 / 0.999 / 1.024 / 1.109 | 1.0003 / 0.9996 / 1.025 / 1.201 |
| hinges | all on z = 0 | all on z = 0 |
| free-boundary score | basis 12 and basis 14, shells to 8-12, agree to 1-5% | basis 12, shells 4 and 8 |

B is 35-50% better than A on every QS measure at every radius, and better than DESC's `reactor_QA` example on the
largest mode (0.27% / 0.18% / 0.29% / 0.51%). It has two thirds of A's transform.

**What is weaker about B's numbers than A's (do not skip):**
* B's free-boundary equilibrium sits on the coils' surface to 0.14% rms, A's to 0.04%. Its boundary is creased again
  after the two shells (area element min/median < 0, B.n 23% at the crease, 1.5% away from it; 0.13% of the area above
  1%). A second re-score WITH repairs after each shell (`li030/fb_repair_modal/`) has a clean boundary but sits 0.39%
  off the coils' surface and gives L2 0.28% / 0.38% / 0.55% / 1.22%, iota 0.19 -> 0.28. Quote B as
  **L2 0.25-0.28% / 0.28-0.38% / 0.46-0.55% / 0.9-1.2%**; even the pessimistic end beats A at every radius.
* B's coil length is 1.1 mm (0.03%) over its bound (3.9423 vs 3.9412 m): not feasible to the letter. A is feasible.
* No basis-14 score, no traced-field check, no Poincare plot of the final coils yet for B.
* In the last single-stage step (k=6, all boundary modes free) the plasma shrank 2.3% (a 0.1236 -> 0.1207, A 7.04 -> 7.26)
  and the edge gain did not survive the independent re-solve; k=5 (`li030/ss/eq_k5.h5`, `coils_k5.h5`: re-solved L2
  0.25% / 0.27% / 0.50% / 0.98%, iota 0.217-0.245, A 7.04) may be the better design and has NOT been free-boundary scored.

**Files (B)**, under `ss/li030/` (harvested from the Modal Volume `from-axis-runs:/ss12_li030_q15`, VERIFIED):
`ss/coils_k6.h5` + `fb_norepair_local/eq_fb.h5` (the pair to quote), `fb_norepair_local/score.json`, `tip.png`;
`ss/eq_kK.h5`, `raw_eq_kK.h5` (before each repair), `coils_kK.h5`, `trials_kK.txt` (every evaluation), `result.json`;
`fb_repair_modal/` (the repaired re-score); `qs_metrics.json`; `run.log`, `meta.json` (commit 6993369dc, A100).
Start: `na/runs2/li_qa2_iota0.30_fb_a0.12_L12_rep/` (L2 0.19% / 0.43% / 0.89% / 1.67%, iota 0.28 -> 0.25, before any
single stage). Viewer: `ss/viewer_ss.html` now has both designs and both starts.

**How B was made** (job `ss12_li030_q15` in `cloud/jobs.py`; one hour on a Modal A100):
near-axis stage with the vacuum iota target lowered to 0.30 -> free-boundary solve at basis 12 (laptop) ->
single stage at basis 12, the SAME recipe as A (Boozer QS, tau_bn 5e-4, tau_qs 0.15 x start, distance rows, hinges on
z = 0, same bounds), iota >= 0.22, A in 6.6-7.6, k = 1..6, with the boundary repaired before k=1 and after every k step.
Single stage by step, each re-solved independently (L2, rho .25/.5/.75/1; iota range):
start 0.17 / 0.43 / 0.92 / 1.60% (0.25-0.29); k1 0.36 / 0.37 / 0.70 / 1.62% (0.25-0.28); k2 0.34 / 0.36 / 0.64 / 1.59%
(0.22-0.25); k3 0.28 / 0.28 / 0.54 / 1.27% (0.22-0.23); k4 0.27 / 0.28 / 0.51 / 1.18% (0.21-0.24); k5 0.25 / 0.27 / 0.50 /
0.98% (0.22-0.25); k6 0.25 / 0.27 / 0.46 / 1.05% (0.21-0.26).

**What changed in the method, and matters for anyone repeating A or B:**
* **The free-boundary solve and the single stage crease the boundary.** B.n (and QS) cannot see a re-parametrization
  of the boundary, so the solver drifts along it until the poloidal angle stalls at a point: a mm-scale kink, a 10-20%
  B.n spike, a negative Jacobian at the edge. The coils' field lines are smooth there (`ss/tip.py`, `ss/tip_trace.py`,
  `na/runs2/li_qa2_iota0.30_fb_a0.12_L12/tip_trace_k6.png`). A's "9.6% B.n spot" (section 3) is very likely the same
  thing; it has not been re-checked.
* **`ss/repair_lib.py`** removes it: same surface, equal-arclength poloidal angle, interior re-solved by continuation.
  With it (`single_stage.py --repair-steps`) the single stage took 12-30 accepted iterations per k step, nearly all on
  the first trial, and converged at every step; A's runs stalled after 0-7.
* **The repair is not free:** the refit moves the surface 0.5-1.7 mm, i.e. 0.2-0.4% of B.n. In a free-boundary score it
  must therefore come BEFORE a shell, never after the last one (`--repair` now does that). Open: a re-parametrization
  that slides points along the existing surface instead of refitting it.
* **The re-solve after each step is an honesty check:** gains that existed only in one solve path disappeared (the
  edge at k=1, k=2, k=6); at basis 12 the edge value depends on the path by ~15%.
* It is NOT yet known how much of B's gain over A is the lower iota and how much is the solver now working: A was never
  run with repairs. Job `ss12_ref042_q15` (the same pipeline on A's parent, iota >= 0.30) answers that.

## 1. The files

All under `coil_sandbox/from_axis/ss/qa2best/`. Two states that are the same design to 4 digits (the hinges agree to
1e-4 m); quote the first, which was scored at the higher resolution.

| what | file |
|---|---|
| **coils** (DESC `CoilSet` of 4 `PiecewisePlanarArcCoil`, B = 2, M = 5, NFP 2, sym) | `ss7_b14/coils_k4.h5` |
| **their free-boundary vacuum equilibrium**, L = M = N = 14 | `ss7_b14_fb/eq_fb.h5` |
| its score (QS, iota, B.n, coil check) | `ss7_b14_fb_score/score.json` |
| basis-12 twin: coils, free-boundary equilibrium, score, Poincare plot | `ss4/coils_k6.h5`, `ss4_fb/eq_fb.h5`, `ss4_fb/score.json`, `ss4_fb/poincare.png` |
| literature QS metrics with reference equilibria; B.n statistics | `qs_metrics.json`, `bn_report.json` |
| 3D viewer (start, ss4, this state, each on its own free-boundary surface) | `../viewer_ss.html` |
| the near-axis parent (parameters, numbers, log) | `../../na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10.{npz,json,log}` |
| the parent's free-boundary score (Phase 2) | `../../na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10_fb_a0.12_L10/` |

Units: the near-axis normalization, B0 = 1 T on axis, axis length 2 pi (R0 ~ 0.84 m). Psi = 0.04524 Wb.

## 2. What it is

| quantity | value |
|---|---|
| R0, a, aspect ratio | 0.843 m, 0.122 m, 6.89 |
| iota, axis -> edge | 0.356 -> 0.313 (crosses 1/3) |
| axis excursion from the midplane | max abs(Z) ~ 0.10 m |
| cross-sections | elongation ~5 at zeta = 0 (thin D), pointed tips near the half period |
| coil currents | 281, 298, 257, 292 kA (0.91-1.06 x their mean) |
| hinges (R, phi), all z = 0 exactly | inner 0.723 / 0.559 / 0.429 / 0.322 m, outer 1.746 / 1.511 / 1.416 / 1.397 m |
| fold between a coil's two faces | 33, 73, 94, 55 deg |

Coil check against the bounds (`../../bounds/fa_qa2_a0167.json`: the Gil et al. set scaled by a = 0.167, absolute):

| | value | bound | ratio |
|---|---|---|---|
| length per coil (max) | 3.938 m | <= 3.941 | 0.999 |
| curvature (max) | 10.16 /m | <= 10.17 | 0.999 |
| coil-coil distance (min) | 0.0409 m | >= 0.040 | 1.024 |
| plasma-coil distance (min) | 0.164 m | >= 0.148 | 1.109 |
| linked pairs | 0 | | |

Length and curvature are active. Relaxing the length bound by 10% changed nothing (`ss6_L110`), so they are not what
limits QS.

## 3. Scores

All on the coils' OWN surface: a vacuum free-boundary DESC solve with these coils and no other objective
(`ss/fb_rescore.py`), then metrics on that equilibrium. This is not the single-stage equilibrium (see section 5).

Boozer amplitudes are single helical modes cos(m theta_B - n zeta_B) (as `QuasisymmetryBoozer`), over B00, with
M_booz = N_booz = 2 x the equilibrium's. Surfaces rho = 0.25 / 0.5 / 0.75 / 1.

| metric | this state | Phase 2 start (its parent) |
|---|---|---|
| largest breaking Boozer mode / B00 | 0.36% / 0.20% / 0.37% / 0.69% | 0.22% / 0.32% / 0.64% / 0.98% |
| L2 of all breaking modes / B00 | 0.42% / 0.43% / 0.84% / 1.74% | 0.32% / 0.66% / 1.33% / 2.32% |
| two-term rms(f_C / B^3) | 2.4e-2 / 3.2e-2 / 5.9e-2 / unusable (fold) | 1.9e-2 / 3.5e-2 / unusable / unusable |
| effective ripple eps_eff, rho .25/.5/.75 (one resolution, NOT converged) | 1.0e-3 / 1.0e-3 / 2.7e-3 | 5.0e-4 / 1.3e-3 / 3.2e-3 |

The same code on DESC's example equilibria (fixed-boundary TARGETS, no coils, so they are flattered relative to us):

| largest breaking mode / B00 | rho 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|
| precise_QA | 0.006% | 0.003% | 0.003% | 0.02% |
| reactor_QA | 0.27% | 0.18% | 0.29% | 0.51% |
| NCSX | 0.69% | 0.66% | 1.3% | 2.7% |
| ARIES-CS | 0.97% | 1.1% | 1.4% | 2.2% |

Normalized B.n of the coils on the free-boundary surface (abs(B.n) / surface-mean abs(B), area-weighted, 193 x 193 grid):

| | this state | ss4 (basis 12) | Phase 2 start |
|---|---|---|---|
| mean | 0.018% | 0.024% | 0.048% |
| rms | 0.040% | 0.046% | 0.092% |
| 99th / 99.9th percentile | 0.14% / 0.35% | 0.16% / 0.39% | 0.39% / 0.72% |
| max | 9.6% | 8.1% | 7.5% |
| max away from the fold | 0.45% | 0.46% | 1.2% |

The 9.6% is one spot (0.011% of the area above 1%) at the pinched inboard tip of the flattest cross-section
(R ~ 0.57, Z ~ 0, zeta ~ 1.30), where DESC's surface folds (section 5). This B.n is NOT a stage-2 B.n: the surface was
fitted to the coils, so it only says the solve found the coils' flux surface.

Poincare (`ss4_fb/poincare.png`): nested out to the boundary, which sits on a traced surface; a weak island chain just
outside (rho ~ 1.1).

## 4. How it was made

1. **Near-axis stage** (`na/run_na2.py`, CPU, `na/chainA.sh qa2 2`): joint optimization of the axis, eta-bar and the
   coils so the coils reproduce the near-axis QS field on the axis (Giuliani et al. JCP 2022), hinges on z = 0 by
   construction. First order -> length/bending regularizer -> second-order ladder (B20 and grad grad B rows) with stiff
   first order (`--w1 10`). Result `o2w1_qa2s2_gg1e-1_w1e-2_w1x10`; seed s0 reaches the same state.
2. **Phase 2 score** (`na/score_fb.py`): exact conversion to DESC coils, NAE initial guess at a = 0.12, vacuum free
   boundary in shells. QS L2 0.32% / 0.66% / 1.3% / 2.3%.
3. **Single stage** (`../single_stage.py` via `ss/run1.sh`): proximal-lsq-composite (secant), `--fix-hinge-z 0`,
   the branch's distance rows (`--dist rows --pc-gap 0.005`), **Boozer QS objective** (`--qs-form boozer`, rho .25/.5/.75/1),
   B.n on the moving boundary with the current sum fixed, bands on A (6.5-7), volume (+-2%), iota >= 0.30.
   Boundary continuation k = 1..6. What mattered:
   * `ss2` (tau_qs 0.3x start, tau_bn 2e-3): first real gain. `ss3` (tau_qs 1.5e-3): big reported gain, half of it fake
     (the boundary left the coils' surface, B.n 0.5%).
   * `ss4` (tau_bn 5e-4, tau_qs 1.5e-3, L enforced at 0.999x, start = ss3 coils + THEIR free-boundary equilibrium):
     the result. Reported and free-boundary numbers agree to 3-12%.
   * `ss7_b14`: the same at basis 14, solve-tol 1e-8, one k step: the coils moved by < 1e-3; its free-boundary score
     at basis 14 reproduces ss4's basis-12 score to 1-5%.
4. **Rule-2 score** after every candidate: `ss/fb_rescore.py EQ COILS OUT --shells 4,8[,12]`.

To reproduce the last step from the parent: `ss/prep.py` (coils into one CoilSet, equilibrium to basis 12), then
`run1.sh` with the env vars and flags listed per run in `ss/NOTES.md`.

## 5. What is not yet trustworthy (read before quoting or comparing)

* **Traced-field check, done** (`ss/trace_check.py`, adapted from `work/ss_qh/joint/fieldcheck/fieldcheck.py`;
  `ss/qa2best/trace_ss7k4/`): 13 lines of the coils' Biot-Savart field, 200 field periods, mapped into the
  free-boundary equilibrium's flux coordinates. No line lost; each stays on one flux label to +-0.004-0.011 in rho for
  rho 0.1-0.9 (0.5-1.3 mm); traced iota within 0.002 of the equilibrium's out to rho 0.95. At rho = 1 the spread is 0.06
  and iota is off by 0.014. On the boundary abs(B_coil) differs from abs(B_eq) by 0.019% mean (5% at the fold spot).
  So the equilibrium represents the coil field inside rho ~ 0.9 but not exactly: worst-case estimate +-0.1-0.15%
  (absolute) on Boozer amplitudes from a 0.01 rho surface offset; not measured. A traced-surface equilibrium fit
  (`tracedsurf.py`, `polish.py`, `directcheck.py` in that directory; rho spread 0.0002-0.003 on precise_QH) has NOT
  been run and would settle it.
* **Every QS number goes through a DESC equilibrium**, not the coils' Biot-Savart field directly (see the
  check above for how close the two are).
* **The free-boundary equilibrium folds at the edge**: sqrt(g) < 0 on ~1% of points at rho = 1 (none at rho <= 0.8 at
  basis 14), at the pinched cross-section tips. Flux surfaces are nested; the poloidal angle folds. Pointwise
  quantities (two-term, B.n max) are wrong there; Boozer spectra are less sensitive. Trust rho <= 0.75 more than rho = 1.
* **Basis 12 has branch solutions on this boundary**: re-solving after a 1e-5 nudge gives QS differing by ~20% (mid)
  to ~50% (edge), independent of tolerance. Basis 14: 0.5-4%. Score at basis >= 14.
* **The outer free-boundary shell (all modes free) runs out of GPU memory at basis 14** (16 GB card); the basis-14
  boundary is refined to mode 8 only.
* **The single stage stalls**: after 0-7 accepted steps per k step it rejects trials until it gives up (basis 12:
  "precision loss"; basis 14: 7 good steps then ~10 rejections, run killed). The design is NOT a converged optimum
  of the single-stage problem; nobody knows how much is left.
* **The single stage over-reports** when B.n is loosely weighted. Only free-boundary scores count.
* **eps_eff** was computed at one resolution. **The edge tip** (B.n 9.6% spot, island chain outside) is unexplained:
  a DESC corner artifact or proximity to a separatrix.
* **No baseline**: the same pipeline with free hinges or conventional coils under the same bounds has not been run,
  so the cost of the z = 0 plane is unknown.

## 6. To beat it

A better result must, on its coils' own free-boundary surface at basis >= 14 (and ideally by traced field):
largest breaking Boozer mode below 0.36% / 0.20% / 0.37% / 0.69% (rho .25/.5/.75/1) or L2 below
0.42% / 0.43% / 0.84% / 1.74%, with nested Poincare surfaces, coils inside the same scaled bounds, unlinked,
currents within [0, 2x] of uniform, every hinge on z = 0, at comparable aspect ratio (~7) and iota (>= 0.3) or with
the difference stated. Leads not yet scored by free boundary are in `NAE_HANDOFF.md` (lower vacuum iota with z = 0
hinges; per-coil level hinges, which leave the single plane).
