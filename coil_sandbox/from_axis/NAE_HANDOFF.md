# Near-axis (NAE) stage: handoff, 2026-10-12

For an agent continuing the near-axis work. Read `CLAUDE.md` (how to run, solver, traps) and `REPORT.md` (sections
1-11, the story up to second order) first; this file adds what was done after `REPORT.md` section 11 and what is open.
The single-stage / free-boundary side is in `ss/NOTES.md` and is separate.

All numbers below are NEAR-AXIS numbers (the coils' fit to the expansion on the axis), not free-boundary scores. They
screen ideas; rule 2 in `CLAUDE.md` (score by vacuum free boundary + Poincare) still decides.

## Reference state

`na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10`: QA nfp 2, 4 coils per half period (16 total), every hinge on z = 0, second
order with stiff first order (`--w1 10 --wgg 0.1 --w2 0.01`), bounds = Gil set scaled by a = 0.167
(d_pc 0.1478, kappa 10.17, L 3.941, d_cc 0.04), iota target 0.42. Every scan below continues from it with
`na/cont.py` and changes ONE thing.

Columns: field cost (`cost_field`: B, grad B, B20, grad grad B rows + iota row), on-axis |dB| (`max_dB`, B0 = 1),
grad grad B mismatch (`max_dGG_rel`), iota, B20 residual (the expansion's own second-order QS breaking), axis max |Z|.

## New options in `na/run_na2.py`

* `--mode free --wlevel 1e4 --hzmax H`: per-coil LEVEL hinges: each coil's two hinges at one free height (a horizontal
  joint per coil), |z| <= H. Without `--wlevel`: hinges fully free. A z0 run can be continued in free mode (`--init`
  converts the parameters).
* `--wplanar W`: per-coil PLANAR rows (alpha_u + alpha_l = 0): no fold, the coil is a planar curve (still two
  sine-series arcs, so an in-plane corner at the hinges is allowed: an upper bound on smooth planar coils).
* The final json now has `fold_deg`, `tilt_u_deg`, and in free mode `hinge_z`, `max_abs_hinge_z`, `max_hinge_dz`.

## 1. Level hinges (`na/lvscan.sh`, runs `lv_qa2_h*`, `lvfree_qa2_h0.20`)

| hinge cap H | field cost | on-axis dB | grad grad B | iota | B20 res. | axis max Z | hinge heights |
|---|---|---|---|---|---|---|---|
| 0 (z = 0) | 0.074 | 1.0% | 56% | 0.372 | 0.40 | 0.101 | 0 |
| 0.02 | 0.060 | 0.83% | 43% | 0.383 | 0.43 | 0.108 | -H, +H, +H, +H |
| 0.05 | 0.041 | 0.44% | 28% | 0.397 | 0.49 | 0.119 | -H, +H, +H, +H |
| 0.10 | 0.029 | 0.29% | 27% | 0.408 | 0.50 | 0.124 | -0.10, 0.056, 0.092, 0.10 (maxiter) |
| 0.20 | 0.014 | 0.14% | 21% | 0.417 | 0.39 | 0.134 | -0.20, 0.061, 0.124, 0.20 (maxiter) |
| 0.20, not level | 0.010 | 0.13% | 14% | 0.413 | 0.32 | 0.136 | per-coil dz up to 0.40 |

* No threshold: the fit improves steadily with H. Coils sit at the cap for H <= 0.05; at H >= 0.10 only the end coils do.
* "Level" costs little against fully free hinges (0.014 vs 0.010).
* B20 does NOT improve until H = 0.20: level hinges help the coils reproduce the target, not the target's own QS.
* Scale: plasma a ~ 0.12, coils reach z ~ +-0.3.

## 2. Lower vacuum iota, hinges on z = 0 (`na/lowiota.sh`, runs `li_qa2_iota*`)

Question: if the plasma (current) supplied part of iota, how much easier is the coils' job?

| iota target | iota | field cost | on-axis dB | grad grad B | B20 res. | axis max Z | max elongation |
|---|---|---|---|---|---|---|---|
| 0.42 | 0.372 | 0.074 | 1.0% | 56% | 0.40 | 0.101 | 4.7 |
| 0.30 | 0.281 | 0.047 | 0.53% | 50% | 0.35 | 0.084 | 4.0 |
| 0.20 | 0.193 | 0.028 | 0.35% | 41% | 0.30 | 0.065 | 3.8 |
| 0.10 | 0.099 | 0.013 | 0.17% | 31% | 0.20 | 0.044 | 3.1 |

* Everything improves together, roughly in proportion to iota, INCLUDING B20 and the elongation (the elongation-5
  cross-section is what DESC struggles with, `ss/NOTES.md`).
* iota 0.2 at z = 0 ~ level hinges at H = 0.10 in field cost, with a flatter axis and better B20.
* This only removes the coils' burden. NOT done: an expansion with on-axis current (I2 != 0; `na/nae.py` and
  `na/nae2.py` are vacuum only, the vendored pyQSC supports I2), where the coil target is the total field minus the
  plasma's own, and the finite-current free-boundary scoring.

## 3. Planar coils (`na/planar.sh`, `na/planar_cold.sh`, runs `pl_z0_*`, `pl_free_*`, `plc_qa2_s*`)

Question: is the fold doing anything, or would planar coils fit the expansion as well?
The folded optima use the fold heavily: fold angles 35-93 deg (z = 0) and 48-86 deg (free hinges).

By continuation in the planarity weight from the folded optima (final step, fold 0.00 deg):

| class | iota (target 0.42) | on-axis dB | grad B | grad grad B | B20 res. | axis max Z | max elong. |
|---|---|---|---|---|---|---|---|
| folded, z = 0 | 0.372 | 1.0% | 2.3% | 56% | 0.40 | 0.101 | 4.7 |
| planar, z = 0 hinges | 0.056 | 0.54% | 3.1% | 31% | 0.05 | 0.045 | 1.4 |
| folded, free hinges | 0.413 | 0.13% | 1.1% | 14% | 0.32 | 0.136 | 5.0 |
| planar, free hinges | 0.064 | 0.56% | 3.1% | 25% | 0.06 | 0.047 | 1.4 |

Cold starts, planar from the beginning (free hinges |z| <= 0.2, full chain, `chainP.sh`):

| stage | seed 0: iota / on-axis dB | seed 2: iota / on-axis dB | folded z = 0 seed 0 |
|---|---|---|---|
| first order | 0.402 / 24.5% | 0.409 / 23.3% | 0.420 / 0.12% |
| + regularizer | 0.422 / 16.9% | 0.421 / 17.1% | 0.420 / 0.14% |
| second order, step 1 | 0.192 / 11.2% | 0.146 / 9.3% | |
| second order, step 3 | 0.064 / 0.59% | 0.064 / 0.59% | 0.372 / 1.0% (seed s2) |

* Planar coils (4 per half period, these bounds) can hold iota 0.42 only at a 17-25% on-axis mismatch; once second
  order is weighted they give up the transform and settle on a nearly axisymmetric state (iota 0.06, elongation 1.4).
  Cold start and continuation reach the same state, so it is not a starting-point artifact.
* So at the near-axis level the fold is what makes the transform at this coil count.
* Caveats: first-order planar runs stopped on maxiter (400), not converged; 4 coils per half period only; "planar"
  allows an in-plane corner at the hinges.

## Open, in rough order of value

1. Free-boundary score (rule 2) of level hinges H = 0.05 and H = 0.20, and of the iota 0.20 state: none has been scored.
   `na/score_fb.py` works for z0 and free mode (`convert.py`). Check the Jacobian (`ss/jac.py`) and use basis >= 14
   (`ss/NOTES.md`: basis 12 has branch solutions on the elongation-5 boundary).
2. On-axis current in the expansion (I2), see section 2.
3. More coils per half period (5, 6) for QA nfp 2: the innermost free-boundary surface sits at 0.3-0.4% QS error whatever
   is done further out (`ss/NOTES.md`), which suggests a floor in the coils' near-axis field. Not tested.
4. Level hinges combined with lower iota; QA nfp 3 (best near-planar first-order fit, `REPORT.md` section 11.3) with either.
5. Planar: converge the first-order cold starts (more iterations), and repeat with 6 coils per half period before
   quoting the planar gap outside this setting.
6. Port the branch's distance-row scheme into `run_na2.py` (still the soft-min stopgap; polyline d_cc is ~1 mm
   optimistic, which showed up as 0.2-1.4% bound violations when the coils were converted to DESC).

## Traps met since `CLAUDE.md` was written

* The repo has `core.autocrlf = true`: a rebase rewrote shell scripts with CRLF and they failed with "cannot execute:
  required file not found". `from_axis/.gitattributes` now forces LF for `*.sh`; if a script fails that way, run
  `sed -i 's/\r$//' script.sh`.
* Memory: one near-axis run (~1.9 GB) fits beside a GPU single-stage job (~7 GB); two CPU jobs with 5 GB caps beside
  it got the side jobs killed. Cap CPU runs at 3 GB (`systemd-run ... -p MemoryMax=3G`), one at a time.
* Continuing a folded optimum with a new hard constraint can simply drop the iota target (planar scan); check `iota`
  in the final json before reading any other column.
