# Single stage on the from_axis candidates (started 2026-10-10)

Driver: `../../single_stage.py` (proximal-lsq-composite, secant, solve-tol 1e-6, basis 12, `--fix-hinge-z 0`),
wrapped by `run1.sh` (env vars OUT, EQ, CS, TAUQS, QSFORM, QSRHO, AMIN, AMAX, IOTAMIN, TAUBN, KMIN, KMAX, MAXIT).
Starts: `prep.py` (coils repacked into one CoilSet(NFP, sym); free-boundary eq raised to basis 12 and re-solved).
Bounds: `../../bounds/fa_qa2_a0167.json` (the Phase 1 bounds, absolute).

## Changes to single_stage.py

* `--dist rows`: the branch's `CoilSetDistanceRows` + `PlasmaCoilSetDistanceRows` (node mode) instead of the
  penalty / kinked min. The plasma-coil rows hold the plasma fixed; they are rebuilt on the current boundary at
  every k step. `--pc-gap` sets the plasma-coil gap explicitly: the build-time estimate came out at 52 m on this
  boundary (the surface-curvature term), which made every row read as violated.
* `--qs-form boozer`: one `QuasisymmetryBoozer` per `--qs-rho` surface, weighted so that tau_qs is the rms over
  surfaces of each surface's L2 norm.

## ss1 (stopped after k=2): the QS two-term objective chased a defect

QA nfp2 best (`o2w1_qa2s2_gg1e-1_w1e-2_w1x10`), QS two-term (scale-invariant), tau_qs 0.5x start.

| step | Boozer QS rho .25/.5/.75/1 | iota | note |
|---|---|---|---|
| start | 3.5e-3 / 6.7e-3 / 1.3e-2 / 2.4e-2 | 0.32-0.37 | |
| k=1 (28 its, 30 min) | 1.35e-2 / 9.2e-3 / 1.6e-2 / 2.4e-2 | 0.32-0.45 | objective 15.3 -> 0.09 |
| k=2 (24 its, 18 min) | 1.8e-2 / 1.9e-2 / 2.5e-2 / 3.5e-2 | 0.33-0.39 | |

Cause: two stellarator-symmetric boundary points with f_C/|B|^3 = +-1.25e4 (typical 0.7), which held essentially
all of the QS objective. The optimizer removed the spike and gave up real QS to do it.

The spike sits where the **free-boundary equilibrium's Jacobian is negative**: sqrt(g) < 0 on 0.1-0.8% of points
for rho >= 0.8 (min/median -0.04 at rho 0.8, -0.11 at rho 1), at the pinched tips of the strongly shaped
cross-sections (zeta = 0 is a thin D, elongation ~5). The flux surfaces themselves are nested (`qa2best/xs_neg.png`);
the poloidal angle folds locally. It is not resolution: L 24 / M 12 and L 20 / M 10 re-solves leave it unchanged
(force cost 5e-6). A fresh `set_initial_guess` from the boundary is worse (non-nested). **The Phase 2 edge QS numbers
for this case (rho = 1) were computed on this mapping**: Boozer is spectral and less sensitive, but treat them as
indicative.

## ss2: Boozer QS objective

Same start, `--qs-form boozer --qs-rho 0.25,0.5,0.75,1.0`, tau_qs 0.3x start (QS cost ~5.5 at the start).

| step | Boozer QS rho .25/.5/.75/1 | iota | B.n | coils | solver |
|---|---|---|---|---|---|
| start | 3.5e-3 / 6.7e-3 / 1.3e-2 / 2.4e-2 | 0.32-0.37 | 9.5e-4 | L, kappa, d_cc fail by 0.2-1.4% (polyline Phase 1) | |
| k=1 | 4.2e-3 / 5.2e-3 / 1.0e-2 / 2.0e-2 | 0.32-0.36 | 1.5e-3 | feasible but L 1.000 | converged |
| k=2 | 4.5e-3 / 5.5e-3 / 1.0e-2 / 1.8e-2 | 0.318-0.328 | 1.7e-3 | FEASIBLE | 3 its, then 14 rejected trials (noise?) |

After k=2 the edge QS is 61% of the cost; iota is drifting toward its 0.30 floor (on-axis 0.36 -> 0.33).
| k=3 | 4.4e-3 / 4.3e-3 / 8.8e-3 / 1.65e-2 | 0.31-0.38 | 1.8e-3 | L 1.000 | converged |
| k=4 | 4.4e-3 / 4.3e-3 / 8.8e-3 / 1.64e-2 | 0.31-0.38 | 1.8e-3 | L 1.000 | 1 it (early stop) |
| k=5 | 4.6e-3 / 4.0e-3 / 8.3e-3 / 1.43e-2 | 0.31-0.35 | 1.7e-3 | L 1.000 | 4 its, precision loss |
| k=6 | 4.0e-3 / 4.8e-3 / 7.9e-3 / 1.40e-2 | 0.31-0.35 | 1.7e-3 (max 3.7e-2) | L 1.000, kappa .998, d_cc 1.024, d_pc 1.068 | converged |

ss2 k=6 validation (`val.py`, `qa2best/ss2/poincare_k6.png`): the coils' own field lines trace clean nested surfaces
out past rho = 1.1; the boundary sits on a traced surface. The angle fold remains (sqrt(g) < 0 on 0.4-0.8% of
points at rho 0.9-1) but no longer drives the objective. Steps stop early (few iterations, then precision loss):
the same evaluation-noise signature as the k5 study.

ss2c: k=6 continuation from ss2, same anchors (tau_qs 4.276e-3, V0, bn_norm), ftol = xtol = 1e-12, maxiter 150.

ss2c result: no progress (1 iteration, then precision loss; cost 2.552 -> 2.551). The state is at the optimizer's
noise floor for these weights. Probe (`../../work/ss_qh/joint/stall/probe.py`, `qa2best/probe_ss2c.log`): steepest
descent matches its model to 4% at a step of 1e-4, 41% at 1e-5, and goes the wrong way below that (noise ~5e-5 in cost). CAVEAT:
the probe reported coil-coil rows at a cost of 130.8 where the run's own table has 3.8e-4. The row objective gives a
different value through the probe's evaluation path (ProximalProjection with row objectives is untested per the
branch GUIDE), so the probe's directions are contaminated by that term; only the noise floor is taken from it. Worth a
look in daylight.

ss3: from ss2 k=6, QS weighted ~3x harder (tau_qs 1.5e-3 instead of 4.28e-3), k = 4..6, 60 its each, same anchors.

ss3 (tau_qs 1.5e-3, tau_bn 2e-3), single-stage states:

| step | Boozer QS rho .25/.5/.75/1 | iota | B.n |
|---|---|---|---|
| k=4 | 3.5e-3 / 3.8e-3 / 5.8e-3 / 9.95e-3 | 0.32-0.34 | 4.6e-3 |
| k=5 | 3.1e-3 / 3.6e-3 / 5.9e-3 / 9.5e-3 | 0.32-0.34 | 4.6e-3 |
| k=6 | 3.0e-3 / 3.3e-3 / 5.2e-3 / 8.6e-3 | 0.32-0.35 | 4.8e-3 |

**Rule-2 score of ss3 k=6** (`fb_rescore.py`, free boundary with the ss3 coils, shells 4/8/12, `qa2best/ss3_fb/`):
B.n 5.4e-3 -> 4.4e-4; Boozer QS **4.4e-3 / 4.7e-3 / 9.2e-3 / 1.87e-2**, iota 0.315-0.36, A 6.88, a 0.122; coils L 3.9423
(bound 3.9412: 0.03% over), kappa 10.17 (ok), d_cc 0.041 (ok), d_pc 0.161 (ok), unlinked, hinges at z = 0; Poincare
nested (`qa2best/ss3_fb/poincare.png`). Against the Phase 2 free-boundary score of the start (3.2e-3 / 6.6e-3 / 1.3e-2 /
2.3e-2): -29% at rho 0.5, -29% at 0.75, -19% at the edge, +36% at 0.25. Real, but half of what the single stage
reported: with tau_bn 2e-3 the boundary left the coils' surface (B.n 4.8e-3) and collected QS the coils do not make.
Lesson: weight B.n at least as hard as QS (tau_bn <= 5e-4 here) and always rescore by free boundary.

ss4: start = ss3 coils + their free-boundary equilibrium (a consistent pair), tau_bn 5e-4, tau_qs 1.5e-3, L enforced at
0.999x, k = 4..6.

ss4 (tau_bn 5e-4, tau_qs 1.5e-3, L 0.999x; start ss3 coils + ss3_fb eq): single-stage k=6 3.9e-3 ... reported
4.45e-3 / 4.49e-3 / 8.3e-3 / 1.49e-2 with B.n 1.06e-3 (k=5 made no progress: noise floor).
**Rule-2 score of ss4** (`qa2best/ss4_fb/`): B.n 4.5e-4; Boozer QS **4.37e-3 / 4.35e-3 / 8.58e-3 / 1.67e-2**, iota
0.314-0.360, A 6.88; coils L 0.999, kappa 0.999, d_cc 1.024, d_pc 1.105 of their bounds: **FEASIBLE**, unlinked, hinges at
z = 0. vs ss3_fb: -7% / -7% / -10% at rho .5/.75/1. vs the Phase 2 start: -34% / -34% / -27%.
The single-stage numbers now overstate the free-boundary ones by only 3-12% (ss3: 30-115%).
Active coil bounds: length (0.999) and curvature (0.999); d_cc and d_pc have 2-10% margin.
ss5: from ss4 pair, tau_qs 1.0e-3, tau_bn 5e-4, k 4..6; rescore queued
ss6_L110 (DIAGNOSTIC, not a design): from ss4 pair, coil length bound x1.10, tau_qs 1.5e-3, tau_bn 5e-4. ss5 (tau_qs 1e-3) stalled at k=4-5 (1 and 0 its): stopped.

ss6_L110 diagnostic: with the coil length bound relaxed 10%, the optimizer did not use it (L 0.909 of the relaxed
bound) and stalled after 2 iterations: **the length bound is not the limit; the optimizer's noise floor is.**

Re-solve noise (`noise.py`, on the ss4_fb equilibrium): perturb every interior coefficient by 1e-5 (random), re-solve
at tol 1e-6 (the proximal's --solve-tol), Boozer QS per surface:
* basis 12: std/mean 19% / 23% / 13% / 55% (rho .25/.5/.75/1); single trials up to +133% at the edge
* basis 14: std/mean 8% / 5% / 4% / 2%
QS at the 1e-2 level is very sensitive to equilibrium convergence on this strongly shaped boundary, so a solve-tol of
1e-6 is not converged enough for it (the k5 study's recommendation was for a different regime).
* tol 1e-9 instead: basis 12 unchanged (19% / 22% / 13% / 50%); basis 14: 0.5% / 4% / 2% / 2%.
**At basis 12 the re-solves are converged but land on DIFFERENT force-balance solutions** (branches, as in the k5
study at basis 8) whose QS differs by up to 2x at the edge. Basis 12 is under-resolved for this boundary; QS values
quoted at basis 12 carry roughly +-20% (mid) / +-50% (edge) branch uncertainty. Move to basis 14, solve-tol 1e-8.
ss7_b14: basis 14 (force cost 6e-7), solve-tol 1e-8, from ss4 pair, tau_qs 1.5e-3, tau_bn 5e-4, L 0.999x, k 4..6
ss7_b14 k=4 (7 its, ~8 min/eval at basis 14): 4.34e-3 / 4.55e-3 / 8.57e-3 / 1.55e-2, iota 0.317-0.336, B.n 1.02e-3. Same frontier as ss4 (basis 12) within branch uncertainty.

## 2026-10-12: basis 14

* The 10-12 launch of "finish basis 14" never ran: a rebase rewrote `run1.sh` with CRLF (core.autocrlf) and it failed
  with "cannot execute"; the chain fell through to a free-boundary rescore of ss7_b14 k=4. Fixed (LF, `.gitattributes`).
* **Basis-14 rule-2 score of ss7_b14 k=4** (`qa2best/ss7_b14_fb/eq_fb.h5`, shells 4 and 8; shell 14 ran out of GPU
  memory, 6.25 GB allocation with all boundary modes free; `fb_rescore.py` now scores the last completed shell):
  B.n 9.1e-4 -> 4.6e-4 -> 4.0e-4; Boozer QS **4.25e-3 / 4.29e-3 / 8.45e-3 / 1.75e-2**, iota 0.313-0.356, A 6.89;
  coils L 0.999, kappa 0.999, d_cc 1.024, d_pc 1.109: FEASIBLE (`qa2best/ss7_b14_fb_score/score.json`).
  This agrees with the basis-12 score of ss4 (4.37e-3 / 4.35e-3 / 8.58e-3 / 1.67e-2) to 1-5%: the ss4 numbers hold at
  basis 14. The k=4 single-stage edge gain at basis 14 (1.55e-2 reported) did not survive the free boundary (1.75e-2).
* ss7b_b14 (k = 5, 6 from ss7_b14 k=4, basis 14, solve-tol 1e-8, --track-steps): k=5 iterations 1-7 cost 21.00 -> 19.59,
  14 evaluations, steps 1.4-3.6 (no collapse, unlike basis 12).

## Literature QS metrics (2026-10-12; `qs_metrics.py`, `qs_booz_fix.py`, `qa2best/qs_metrics.json`)

ss7b_b14 (k=5) killed after 7 accepted steps and ~35 min of rejected trials. State scored: ss7_b14 k=4 coils on their
own basis-14 free-boundary surface (`qa2best/ss7_b14_fb/eq_fb.h5`). References: DESC example equilibria (fixed-boundary
targets, NO coils). Boozer amplitudes are cos(m theta_B - n zeta_B) modes (ptolemy transform, as QuasisymmetryBoozer;
the raw `|B|_mn_B` product basis gives 1.2-2x larger numbers and is NOT the literature convention) over B_00.

max breaking Boozer mode / B00 at rho .25/.5/.75/1:
  ours k=4 b14   3.6e-3 / 2.0e-3 / 3.7e-3 / 6.9e-3      phase 2 start 2.2e-3 / 3.2e-3 / 6.4e-3 / 9.8e-3
  precise_QA     6.3e-5 / 2.7e-5 / 3.3e-5 / 2.1e-4      reactor_QA    2.7e-3 / 1.8e-3 / 2.9e-3 / 5.1e-3
  NCSX           6.9e-3 / 6.6e-3 / 1.3e-2 / 2.7e-2      ARIES-CS      9.7e-3 / 1.1e-2 / 1.4e-2 / 2.2e-2
L2 of breaking modes / B00:
  ours 4.2e-3 / 4.3e-3 / 8.4e-3 / 1.7e-2   precise_QA 8.1e-5 / 4.2e-5 / 5.6e-5 / 2.8e-4   reactor_QA 2.9e-3 / 2.9e-3 / 4.1e-3 / 7.3e-3
  NCSX 9.7e-3 / 1.3e-2 / 2.0e-2 / 3.9e-2   ARIES-CS 1.1e-2 / 1.4e-2 / 1.9e-2 / 3.6e-2
two-term rms(f_C / B^3): ours 2.4e-2 / 3.2e-2 / 5.9e-2 / (138: fold spike; median 6.0e-2); precise_QA 2.1e-4 / 9.8e-5 /
  1.3e-4 / 5.8e-4; reactor_QA 4.4e-3 / 4.5e-3 / 6.5e-3 / 1.2e-2; NCSX 4.6e-2 / 5.3e-2 / 7.6e-2 / 1.5e-1; ARIES-CS 2.3e-2 / 2.9e-2 / 5.1e-2 / 1.3e-1
eps_eff (DESC "effective ripple", Bounce2D X=Y=32, Y_B=128, 20 transits; resolution NOT convergence-checked), rho .25/.5/.75:
  ours 1.0e-3 / 1.0e-3 / 2.7e-3; precise_QA 4.6e-7 / 3.2e-7 / 1.3e-6; reactor_QA 2.2e-4 / 8.1e-4 / 2.8e-3;
  NCSX 2.0e-3 / 1.9e-3 / 3.8e-3; ARIES-CS 6.3e-4 / 2.3e-4 / 8.0e-4
Viewer: `viewer_ss.html` (`../../view.py`; phase 2 start, ss4, ss7 k=4, each on its own free-boundary surface).
