# Stage 2 on the fixed single-stage QH boundary (2026-10-02)

Equilibrium `work/ss_qh/arcB2/ss_k4/eq_k4.h5` (vacuum), bounds `bounds/precise_qh_gil.json` (absolute),
all runs `--kappa-grid-n 600`, GPU. Dense check, margins vs the UNbacked-off bounds (`table.py`).
Viewer: `view.html` (since 2026-10-07: the three lsq-auglag-composite results, table.md; the old-solver
versions are view_oldsolver.html and table_oldsolver.md, and the table below). Planar history:
`planar_history.html`, `planar_history_*.png`.

| coilset | ⟨\|B·n\|⟩/⟨B⟩ | max | ÷ XYZ | L | κ | κ_MS | d_cc | d_pc | linked | active | over by >1e-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arcB2 M5, single-stage start | 1.919e-03 | 8.01e-03 | 3.67 | 1.000 | 0.997 | 0.994 | 1.205 | 1.133 | 0 | L, kappa | L |
| arcB2 M5, stage 2 | 1.449e-03 | 6.35e-03 | 2.77 | 1.000 | 0.995 | 0.815 | 1.123 | 1.296 | 0 | L, kappa | none |
| planar N7, stage 2 (cold, 150+400 its) | 1.858e-02 | 7.77e-02 | 35.52 | 1.000 | 1.109 | 1.000 | 1.016 | 1.010 | 0 | L, kMS | kappa |
| FourierXYZ N6, stage 2 | 5.231e-04 | 2.14e-03 | 1.00 | 1.000 | 1.000 | 0.828 | 1.022 | 1.335 | 0 | L, kappa | kappa |

* arcB2: 275 its, solver stalled at a feasible point; flat since it 100. L "over" at the start is 2.5e-4.
* FourierXYZ N6: warm start (precise_QH stage-2 it 300). Stopped at it 200 (B.n -0.36% from it 150);
  kappa 1.00034 = dense-grid read between enforcement nodes. First GPU attempt crashed (ILLEGAL_ADDRESS, before it 1); rerun clean.
* planar N7: COLD start (8.7e-2). 150 its at cap 0.5 (planarN7/), then 400 its from ckpt_it0150 with
  --max-tr inf + --diag-log (planarN7_trinf/). Trust radius was never binding (steps interior, ratio ~1).
  Restart reset the multipliers; kappa overshot to 1.37 and the AL only clawed it back to 1.11 by maxiter.
  B.n flat at 1.86e-2 for the last 250 its. Infeasible on kappa; and the start is not comparable to the
  two warm starts, so this does not yet measure what planar N7 can do on this boundary.

## Planar optimality study (2026-10-02, after the deliverable)

Why could planar N7 not converge? Step-level logs (`--diag-log`, summarized by `step_stats.py`,
`compare_klb.py`). All planar runs from the cold start; dense check vs the bounds each run used.

| run | change | B.n | infeasible | outer updates | optimality, last 50 its (min / median) |
|---|---|---|---|---|---|
| planarN7_trinf | restart it150, max-tr inf | 1.858e-2 | kappa 1.109 | 12 | 6.7e-4 / 1.6e-3 |
| planarN7_klb | + kappa lower bound -inf | 1.861e-2 | kappa 1.021 | 18 | 5.8e-4 / 1.5e-3 |
| easy_R1 | N=3, max-tr inf | 2.590e-2 | kappa 1.556, d_cc 0.973 | 17 | 4.4e-2 / 4.3e-1 |
| easy_R2 | kappa bound x2 (inactive, 0.65) | 1.862e-2 | none (L, kMS 1.000 noise) | 24 | 2.5e-4 / 7.3e-4 |
| easy_R3 | no pointwise kappa (kMS kept) | 1.857e-2 | none (L, kMS 1.000 noise) | 9 | 5.4e-4 / 1.7e-3 |
| arcB2_diag | arcB2 restart at its optimum, 50 its | 1.449e-3 | none | 19 | 3.9e-7 / ~5e-5 |

Findings (measured unless marked):
* The trust radius never limited N7 (steps interior, model ratio ~1). N3 with the 0.5 cap sat ON the cap.
* lsq-auglag is Conn-Gould AL + scipy TRF on slacks; checked, no bug. Steps leave the slack box 56-100%
  of the time in EVERY run incl. arcB2 (92%). TRF then truncates/reflects; with a weakly active slack
  (|g_s| ~ 0) the stride -> 0 (derived: p_s ~ v(|g|+mu dc)/(|g|+mu v); hits iff dc > v). Pinned
  distance-penalty slacks sit ~1e-38 from their 0 bound.
* Which rows truncate: per-node kappa rows (median stride 0.001-0.01) whenever kappa is active; C1 distance
  penalties otherwise (stride ~0.01-0.03). kappa lb 0 additionally floors inflection nodes: removing it
  raised the early stride 0.0013 -> 0.22.
* Not planar-specific: arcB2 shows the same truncation, sign flips, and 5x MORE near-zero-|kappa| nodes,
  and converges -- but it restarted AT its optimum (confound).
* Every cold N7 run reaches B.n ~1.86e-2 and optimality ~1e-3 by the end; with kappa active, kappa keeps
  oscillating 1.02-1.4 and never becomes feasible. With kappa out of the way (R2, R3) the run is feasible
  and the AL behaves textbook (gtolk exits, tightening tolerance), with a distance-penalty floor ~5e-4-2e-3.
* So kappa does not limit planar B.n on this boundary (L and distances do). Planar N7 cold ~1.86e-2.
* N3 is worse: fewer DOFs, larger kappa excursions, penalties up to 1e5, never feasible.
* Not reached in any planar run: optimality << 1e-3.
Code changes (sandbox only, defaults unchanged): run_one.py `--kappa-lb {zero,none}`; a null kappa bound
skips CoilCurvature. Bounds variants: bounds/precise_qh_gil_kappa2x.json, precise_qh_gil_nokappa.json.

## Slack-free hinge AL on planar N7 (2026-10-07, main repo `js/coil-auglag`)

`run_al.py` + `sandbox/al_problem.py` (the ../devtools/coil_auglag AL10/AL11 recipe: node coil-coil rows,
gap 2.334 mm from the bounds; PlasmaCoilDistanceField, margin 0.556 mm; per-node |kappa| on 601 nodes; kappa_MS;
length; QuadraticFlux field grid N=50, checked = N=100 to 5e-15). Cold start, absolute precise_qh_gil bounds,
no backoffs. Out: `al_planarN7/` (run.log, outer.json, result.h5/json).

* **Converged**: `gtol` met, 11 outer / 453 inner its, 500 s; scaled gradient 2.9e-7, violation 6.5e-7;
  every inner solve met its tolerance. Old lsq-auglag planar runs: floor ~1e-3, 36-48 min, kappa infeasible.
* Dense check: **B.n 1.8516e-2**, L 1.000, kMS 1.000, d_cc 1.038, d_pc 1.005, linked 0; active L, kappa, kMS, d_pc.
  B.n matches the old planar runs (1.857-1.862e-2): planar N7 on this boundary is ~1.85e-2, now certified.
* kappa 1.002 on the dense check: the optimizer puts the peak BETWEEN the 601 enforcement nodes (exactly
  13.6728 on that grid, 13.699 on N=3000). Fix: a ~0.3% kappa backoff or a finer kappa grid.

## All three reps with the slack-free AL (2026-10-07)

Same starts as the original stage-2 runs (planar cold, arcB2 single-stage coils repacked into one
CoilSet(NFP=4, sym=True) by `repack_arcs.py`, XYZ warm), same bounds, no backoffs, gtol = ctol = 1e-6.
arcB2: coil-coil gap and plasma node margin from the coils (`--cc-gap coils`: 1.061 mm on 1086 nodes,
0.997 mm), because arc node spacing is ~3x uneven. XYZ: phase pinned, relative speed rows +-0.5, gap with
speed ratio 1.5 (1.313 mm); the speed rows evened the parameterization in the first outer iteration.

| run | B.n | max | / XYZ | old solver B.n | L | kappa | kMS | d_cc | d_pc | linked | outer/inner | opt | viol | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| planar N7 | 1.8516e-02 | 7.77e-02 | 35.67 | 1.8610e-02 (kappa 1.02, infeasible) | 1.0000 | 1.0019 | 1.000 | 1.038 | 1.005 | 0 | 11/453 | 2.9e-07 | 6.5e-07 | 500 s |
| arcB2 M5 | 1.4489e-03 | 6.35e-03 | 2.79 | 1.4492e-03 | 1.0000 | 1.0007 | 0.816 | 1.123 | 1.296 | 0 | 5/72 | 9.5e-07 | 7.9e-08 | 143 s |
| FourierXYZ N6 | 5.1916e-04 | 2.12e-03 | 1.00 | 5.2306e-04 | 1.0000 | 1.0003 | 0.753 | 1.022 | 1.333 | 0 | 11/94 | 3.0e-07 | 3.0e-07 | 218 s |

(plus ~3 min field build per run). Every run converged with every inner solve meeting its tolerance.
* The old arcB2 and XYZ results were at (or within 0.7% of) their optima; the old planar result was not
  feasible, but its B.n was right. The ranking and ratios stand: arcB2 2.8x XYZ, planar ~36x.
* kappa reads 0.03-0.19% over on the dense check in all three: the peak sits between the 601 enforcement
  nodes. A ~0.3% kappa backoff would make all three strictly feasible.
* New-code caveats found: CoilSetDistanceRows/PlasmaCoilDistanceField read NFP/sym from the top-level
  coilset only (MixedCoilSet of symmetric CoilSets -> 4 coils instead of 32: IndexError); devtools
  check.py measures length on a uniform grid, which is wrong across arc corners (read 2.932 vs 2.926).
