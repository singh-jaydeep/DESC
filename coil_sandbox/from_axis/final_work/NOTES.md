# Finalizing B and D: working notes (started 2026-10-14; plan in `../FINALIZE_HANDOFF.md`)

## 1a. Traced iota of the coil fields vs the equilibria (`iota/`, `../ss/iota_trace.py`, laptop GPU, ~1 min per coil set)

Method: magnetic axis of the coil field = fixed point of the one-period map (Newton); 30 lines on Z = 0, phi = 0 from
the axis to 1.08x the outboard boundary, 500 field periods, 16 samples per period; iota = Birkhoff-weighted rotation
number about the traced axis (half-vs-half error 1e-6 - 1e-9 except next to a rational); punctures mapped into each
equilibrium (rho mean / std, iota_eq at rho mean). Every line in every case stays confined out to 1.08x.

| coil set | traced iota, axis -> rho~1 | equilibria vs trace (iota_eq - iota_trace) | eq axis vs traced axis |
|---|---|---|---|
| relaxed 5-coil (`relaxed/nc5lh`) | 0.415 -> 0.365 | shell 8: -0.016 core, +0.015 edge; **repaired: -0.072 core, +0.058 edge** (shear reversed) | 2.9 mm |
| relaxed 6-coil (`relaxed/nc6z0`) | 0.416 -> 0.367 | shell 8: -0.043 core; repaired: +0.006 core, -0.02 mid | 2.7 mm |
| **B k6** | **0.238 -> 0.219** (not flat) | fb -0.010 core / +0.002 edge; fb_repaired -0.01 / +0.012; ss_k6 -0.02 mid / +0.024 edge | **5.8 mm** |
| **B k5** | 0.240 -> 0.222 | ss_k5 within 0.005 everywhere | 3.5 mm |
| **D k6** | **0.399 -> 0.352** | fb +0.015 core / +0.010 edge; ss_k6 -0.05 core / +0.04 edge (shear reversed) | 1.5 mm |

* **D never reaches 2/5**: the coil field's iota is 0.399 at the axis and falls outward, so the 2/5 resonance the
  equilibrium's 0.415 suggested is not in the plasma (it sits just outside the axis value). Re-check with the
  resolved equilibrium; dense trace near the axis for a 2/5 island anyway.
* **The repair's iota shift (RESULTS 7b, 8) is not physics**: neither the repaired nor the unrepaired equilibrium is
  consistently closer to the trace. The basis-12 equilibria do not reproduce the coils' interior field accurately
  enough to give iota to better than 0.01-0.07.
* D's single-stage iota floor (>= 0.35) was enforced on an equilibrium whose iota is off by up to 0.05.

## 1b. Interior field: equilibrium vs coils (`iota/interior_*.json`, `../ss/interior_check.py`)

|B_coil - B_eq| / |B_eq| rms: 0.15-0.5% inside rho 0.75 (direction 0.1-0.3 deg), 0.3-0.8% at 0.9, 0.7-2% at 1.
Vacuum residual mu0 |J| a / |B| rms: ~1e-2 inside, 4-10e-2 at rho 0.9, 0.15-0.35 at 1.
Why iota is so sensitive: the poloidal pitch is iota r / R ~ 0.01 rad (0.8 deg), so a 0.2 deg direction error is
10-30% of the field-line pitch near the axis; the spurious current is as large as the field's poloidal part.
**The interior |B| error (0.2-0.5%) is the size of the QS breaking we quote (0.25-0.5%).** The QS numbers are only
meaningful once the interior is resolved: next, the resolution ladder (1c).

## 1c. Resolution ladder, design D (`res/D_fb/`, `../ss/resladder.py`: the fb boundary held fixed, vacuum re-solve)

| basis | iota axis -> edge | eq axis vs traced | mu0|J|a/|B| rho .1/.5/.9 | interior |dB|/|B| rms | solve |
|---|---|---|---|---|---|
| 12 (tight tol) | 0.416 -> 0.366 | 1.5 mm | 1.6e-2 / 8.6e-3 / 4.2e-2 | 0.15-0.3% | 1.1 min; same as the quoted state: converged, so it is resolution |
| 14 | 0.407 -> 0.347 | 0.4 mm | 6.2e-3 / 4.7e-3 / 2.0e-2 | 0.05-0.07% | 7.3 min |
| 16 | 0.395 -> 0.349 | 0.2 mm | 9.1e-4 / 1.7e-3 / 8.1e-3 | 0.016-0.04% | 38 min (13 GB GPU) |
| trace | 0.399 -> 0.352 | | | | |

Basis 16 vs trace: iota within 0.0035 in the core, 5e-4 from mid radius out; traced lines lie on its surfaces to rho std 0.002-0.004.

QS (`res/D_qs_ladder.json`, `qs_metrics.py`):

| basis | L2 % rho .25/.5/.75/1 | largest mode % | eps_eff .25/.5/.75 |
|---|---|---|---|
| 12 (quoted) | 0.304 / 0.279 / 0.469 / 0.954 | 0.189 / 0.111 / 0.166 / 0.319 | 2.65e-4 / 2.76e-4 / 6.83e-4 |
| 14 | 0.262 / 0.266 / 0.489 / 1.108 | 0.173 / 0.086 / 0.200 / 0.420 | 2.21e-4 / 2.35e-4 / 6.11e-4 |
| 16 | 0.252 / 0.256 / 0.462 / 1.110 | 0.176 / 0.091 / 0.170 / 0.428 | 2.57e-4 / 1.96e-4 / 5.70e-4 |

D's QS survives: core and mid slightly better, the edge 16% (L2) / 34% (max) worse than quoted, converged 14 -> 16.
Caveat: the boundary is still the basis-12 free-boundary one (B.n 0.097%); a basis-16 free-boundary shell would remove it.

## 1d. QS convention: the handoff's B-vs-D table mixed two conventions

`ss/qs_metrics.py` writes Boozer amplitudes in DESC's raw product basis; the literature convention (cos(m theta_B -
n zeta_B), ptolemy transform, as QuasisymmetryBoozer) needs `ss/qs_booz_fix.py` run on its json afterwards (ss/NOTES.md,
2026-10-12: raw reads 1.2-2x larger). B's `li030/qs_metrics.json` was fixed; **D's `nc5lh/qs_metrics.json` was not**, so
D's quoted QS (0.30/0.28/0.47/0.95) is inflated. Boozer resolution is converged (2M, 3M, 4M identical). The repaired-B
"jump" seen with raw qs_metrics (0.47 -> 0.63 at rho .75) is a raw-basis artefact; the literature value and the
two-term measure do not move with the repair.

Literature convention (QuasisymmetryBoozer amplitudes / B00), fixed-boundary re-solves on each design's basis-12
free-boundary boundary (`res/*_qs_ladder.json` after qs_booz_fix):

| | L2 % rho .25/.5/.75/1 | largest mode % | eps_eff .25/.5/.75 | traced iota |
|---|---|---|---|---|
| D basis 12 (quoted state) | 0.273 / 0.227 / 0.345 / 0.722 | 0.189 / 0.111 / 0.160 / 0.281 | 2.65e-4 / 2.76e-4 / 6.83e-4 | 0.399 -> 0.352 |
| **D basis 16** | **0.239 / 0.208 / 0.341 / 0.817** | **0.176 / 0.091 / 0.168 / 0.414** | 2.57e-4 / 1.96e-4 / 5.70e-4 | |
| B basis 12 (quoted state) | 0.247 / 0.279 / 0.468 / 0.935 | 0.130 / 0.140 / 0.223 / 0.450 | 6.89e-4 / 7.36e-4 / 1.29e-3 | 0.238 -> 0.219 |
| **B basis 16** (300 its, not converged) | **0.231 / 0.272 / 0.476 / 0.910** | **0.118 / 0.139 / 0.214 / 0.442** | 6.08e-4 / 6.79e-4 / 1.33e-3 | |

**D is better than B, not level with it**: 25% lower L2 at rho .5-.75, a little lower at the edge, level in the core,
2-3x lower eps_eff, at 1.7x the transform. Every number in the packages must be produced by ONE script that applies the
literature convention (fold qs_booz_fix into qs_metrics).

B basis ladder vs trace (`iota/B_ladder/`): b14 iota 0.237 -> 0.221 (within 0.004 of the trace), b16 0.230 -> 0.222
(0.008 low in the core: hit maxiter 300, not converged); axis 0.3 mm from the traced one at b16.

## 1e. Modal group `final_fb14` (2026-10-14 18:04-18:48, A100-40GB, commit 29f1c5e36): ~2 h 07 min used

Free-boundary shell at basis 14 (fb_rescore --LMN 14 --solve-tol 1e-8, 60 its, no repair). Watchdog at 40 min, then
extended by hand ~3.5 min for Bk6 (budget overrun ~7 min; Bk6 still did not finish).
* **fb14_D: done, VERIFIED** (`../ss/cloud_runs/fb14_D/`). One shell, all modes free: B.n rms 9.7e-4 -> **1.6e-5**
  (max-iterations, still improving; NO "bad approximation" stall: the stall was the default inner solve tolerance
  and/or basis 12). Coils FEASIBLE (L 1.000, kappa 0.998, d_cc 1.028, d_pc 1.147 of their bounds).
* fb14_Bk6: lost (its shell needed ~37 min after a 5-min resolution step). fb14_Bk5: shell k=4 only (B.n 0.19%,
  converged on gtol), `../ss/cloud_runs/fb14_Bk5/`.
* Lesson: fb_rescore writes eq_fb.h5 only at the end of a shell; for time-boxed runs use short shells (maxiter ~20)
  repeated, so every chunk is saved.

**D on its basis-14 free-boundary state = the consistent coil/equilibrium pair** (`iota/D_fb14/`, `iota/interior_D_fb14.json`,
`res/D_fb14_qs.json`):
* traced iota vs equilibrium: within 6e-4 at every radius; axis 0.04 mm from the traced axis; traced lines on its
  surfaces to rho std 5e-4 - 2e-3.
* interior |B_coil - B_eq| / |B| 0.004-0.008% (rho <= 0.9), vacuum residual mu0|J|a/|B| 3-8e-4 inside.
* A 7.558; iota 0.399 -> 0.350 (never reaches 2/5).
* QS (literature) L2 0.240 / 0.213 / 0.354 / 0.777 %, largest mode 0.175 / 0.091 / 0.188 / 0.319 %,
  two-term 0.012 / 0.017 / 0.029 / 0.057, eps_eff 2.66e-4 / 1.99e-4 / 7.16e-4.

**Resolution check of D's basis-14 free-boundary state** (`res/D_fb14/eq_b16.h5`: same boundary, interior at basis 16,
78 min laptop, hit maxiter 300; `res/D_fb14_b16_qs.json`):

| | iota axis -> edge | L2 % rho .25/.5/.75/1 | largest mode % | eps_eff .25/.5/.75 |
|---|---|---|---|---|
| fb14 (basis 14) | 0.3985 -> 0.3501 | 0.240 / 0.213 / 0.354 / 0.777 | 0.175 / 0.091 / 0.188 / 0.319 | 2.66e-4 / 1.99e-4 / 7.16e-4 |
| same boundary, basis 16 | 0.3991 -> 0.3494 | 0.240 / 0.213 / 0.351 / 0.785 | 0.175 / 0.091 / 0.183 / 0.332 | 2.55e-4 / 2.06e-4 / 7.05e-4 |

Converged to 1-4% (vacuum residual 1e-4 - 1e-3 at basis 16). **These are D's numbers to quote.**

## Status at end of 2026-10-14 session

* D: step 1 done except the 2/5-adjacent dense trace (iota 0.399 at the axis: check for a 2/5 island chain right at
  the axis) and island-width estimates.
* B: NOT settled. Needs a free-boundary shell at basis 14 (as D got) for k=6 and for k=5, then the k5/k6 decision and
  the coil length polish. Laptop estimate several hours per state; use short repeated shells (maxiter ~20).
* All quoted QS must come from qs_metrics + qs_booz_fix (literature convention); fold the fix into qs_metrics.
