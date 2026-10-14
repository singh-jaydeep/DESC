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
