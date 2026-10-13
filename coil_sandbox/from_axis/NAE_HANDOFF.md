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

## 4. Thin hinge band with a tilt bound (`na/loose.sh`, `na/loose_table.py`, runs `lo_qa2_i*_h*_t*`, `hi_qa2_iota0.50`)

Hinges of every coil within |z| <= H (H = 0.1, 0.2, 0.3 a; a = 0.167, so 0.0167, 0.0334, 0.0501), hinge line level,
tilted <= 2 degrees (`--tiltmax 2`), or free inside the band; iota target 0.30, 0.42, 0.50, each continued from its own
z = 0 state. `/z0` is the field cost over that z = 0 state's. All 27 converged (ftol/gtol).

```
iota tgt   H/a   tilt |    cost   /z0   iota    B20    dGG elong | max|z| max tilt(deg) nit status
0.30         0    z=0 |  0.0472  1.00  0.281  0.354  0.499  3.97 | 0.0000  0.00 208 Optimization terminated succ
0.30       0.1      2 |  0.0375  0.79  0.286  0.388  0.392  4.05 | 0.0167  1.97 188 `gtol` condition satisfied. 
0.30       0.1   free |  0.0375  0.79  0.286  0.387  0.392  4.04 | 0.0167  1.97 147 Optimization terminated succ
0.30       0.1  level |  0.0408  0.87  0.284  0.358  0.429  3.86 | 0.0167  0.00 153 `gtol` condition satisfied. 
0.30       0.2      2 |  0.0296  0.63  0.289  0.387  0.319  3.92 | 0.0334  2.00 206 `gtol` condition satisfied. 
0.30       0.2   free |  0.0279  0.59  0.290  0.381  0.302  3.98 | 0.0334  3.57 172 `gtol` condition satisfied. 
0.30       0.2  level |  0.0322  0.68  0.288  0.390  0.335  3.78 | 0.0334  0.00 386 `gtol` condition satisfied. 
0.30       0.3      2 |  0.0242  0.51  0.293  0.410  0.232  3.68 | 0.0501  2.00 182 `gtol` condition satisfied. 
0.30       0.3   free |  0.0216  0.46  0.294  0.368  0.222  3.68 | 0.0501  5.38 303 Optimization terminated succ
0.30       0.3  level |  0.0287  0.61  0.292  0.384  0.274  3.72 | 0.0501  0.00 216 `gtol` condition satisfied. 
0.42         0    z=0 |  0.0738  1.00  0.372  0.403  0.564  4.65 | 0.0000  0.00 278 `gtol` condition satisfied. 
0.42       0.1      2 |  0.0598  0.81  0.382  0.392  0.472  4.50 | 0.0167  1.74 214 `gtol` condition satisfied. 
0.42       0.1   free |  0.0598  0.81  0.382  0.392  0.472  4.50 | 0.0167  1.74 193 Optimization terminated succ
0.42       0.1  level |  0.0626  0.85  0.380  0.414  0.467  4.56 | 0.0167  0.00 213 `gtol` condition satisfied. 
0.42       0.2      2 |  0.0493  0.67  0.390  0.427  0.375  4.57 | 0.0334  2.00 212 `gtol` condition satisfied. 
0.42       0.2   free |  0.0424  0.57  0.394  0.451  0.333  4.67 | 0.0334  3.75 168 `gtol` condition satisfied. 
0.42       0.2  level |  0.0523  0.71  0.387  0.445  0.369  4.52 | 0.0334  0.00 148 `gtol` condition satisfied. 
0.42       0.3      2 |  0.0340  0.46  0.403  0.492  0.258  4.44 | 0.0501  2.00 274 `gtol` condition satisfied. 
0.42       0.3   free |  0.0303  0.41  0.403  0.447  0.243  4.43 | 0.0501  5.28 177 Optimization terminated succ
0.42       0.3  level |  0.0406  0.55  0.397  0.493  0.281  4.46 | 0.0501  0.00 264 `gtol` condition satisfied. 
0.50         0    z=0 |  0.0938  1.00  0.424  0.370  0.609  4.93 | 0.0000  0.00 100 Optimization terminated succ
0.50       0.1      2 |  0.0773  0.82  0.436  0.324  0.536  4.66 | 0.0167  1.81 420 `gtol` condition satisfied. 
0.50       0.1   free |  0.0773  0.82  0.436  0.324  0.536  4.66 | 0.0167  1.81 379 Optimization terminated succ
0.50       0.1  level |  0.0803  0.86  0.431  0.380  0.548  4.87 | 0.0167  0.00 260 `gtol` condition satisfied. 
0.50       0.2      2 |  0.0418  0.45  0.479  0.485  0.332  4.40 | 0.0334  2.00 340 Optimization terminated succ
0.50       0.2   free |  0.0403  0.43  0.478  0.477  0.327  4.40 | 0.0334  3.98 308 `gtol` condition satisfied. 
0.50       0.2  level |  0.0664  0.71  0.441  0.461  0.409  5.00 | 0.0334  0.00 334 Optimization terminated succ
0.50       0.3      2 |  0.0365  0.39  0.481  0.482  0.320  4.67 | 0.0501  2.00 562 `gtol` condition satisfied. 
0.50       0.3   free |  0.0335  0.36  0.483  0.454  0.308  4.77 | 0.0501  5.46 212 Optimization terminated succ
0.50       0.3  level |  0.0467  0.50  0.456  0.523  0.284  5.00 | 0.0501  0.00 361 `gtol` condition satisfied.
```

* Field cost: 0.1 a gives 13-21% off, 0.2 a 29-57%, 0.3 a 39-64%. Roughly linear in H, no threshold.
* Tilt matters as much as height. At 0.1 a the band only permits ~1.8 degrees, so "2" and "free" coincide. At 0.2-0.3 a
  the 2-degree bound captures most of the free gain (free uses 3.6-5.5 degrees); level gives about half to two thirds.
* The gain grows with iota: at 0.3 a free, cost is 0.46 / 0.41 / 0.36 of z = 0 for iota 0.30 / 0.42 / 0.50. The
  iota 0.50 target with 0.2 a and 2 degrees reaches iota 0.48 at a cost (0.042) below the z = 0 iota 0.30 state (0.047).
* Achieved iota is under target in every run and moves toward it as the band widens; the cost includes that miss.
* The second-order residual B20 does **not** improve (0.35-0.52 everywhere, often worse). max dGG (second-order
  gradient mismatch) halves. So the band buys first-order field accuracy and transform, not second-order QS.
* Caveat on section 1: the heights there were quoted against a = 0.12; with a = 0.167 they are 0.12, 0.30, 0.6, 1.2 a.
* Not free-boundary scored. The single stage (`ss/single_stage.py`) has no height cap or tilt bound yet.

## 5. Weakening the engineering bounds (`na/loose2.sh`, `na/loose2_table.py`, runs `l2_qa2_*`)

One knob at a time at iota 0.42, at z = 0 and at (0.2 a, 2 degrees): plasma-coil distance 0.8/0.7/0.6 a, max curvature
and max length x1.15/x1.3, coil-coil distance 0.03/0.02, coils per half period 5/6 (cold starts with a cold 4-coil
control). Results: `na/runs2/loose2_scan.out`.

```
hinge         knob |    cost /base   iota    B20    dGG elong |    dpc    dcc   kmax   Ltot nit status
z0            base |  0.0738  1.00  0.372  0.403  0.564  4.65 | 0.1478 0.0400  10.17  15.78 278 `gtol` condition satisfied.
z0       Lmax4.532 |  0.0441  0.60  0.390  0.282  0.506  4.63 | 0.1509 0.0400  10.17  17.61 157 `gtol` condition satisfied.
z0       Lmax5.124 |  0.0257  0.35  0.401  0.346  0.373  5.00 | 0.1721 0.0400  10.17  20.50 138 `gtol` condition satisfied.
z0         dcc0.02 |  0.0684  0.93  0.377  0.419  0.493  4.59 | 0.1478 0.0200  10.17  15.77 77 Optimization terminated succ
z0         dcc0.03 |  0.0710  0.96  0.375  0.409  0.528  4.62 | 0.1478 0.0300  10.17  15.78 61 Optimization terminated succ
z0       dpc0.1002 |  0.0719  0.97  0.372  0.297  0.515  4.78 | 0.1042 0.0400  10.17  15.71 512 Optimization terminated succ
z0       dpc0.1169 |  0.0723  0.98  0.372  0.319  0.550  4.77 | 0.1169 0.0400  10.17  15.70 377 Optimization terminated succ
z0       dpc0.1336 |  0.0732  0.99  0.373  0.386  0.555  4.63 | 0.1336 0.0400  10.17  15.65 153 Optimization terminated succ
z0       kmax11.70 |  0.0711  0.96  0.375  0.413  0.522  4.57 | 0.1478 0.0400  11.70  15.78 97 Optimization terminated succ
z0       kmax13.23 |  0.0689  0.93  0.377  0.414  0.497  4.34 | 0.1478 0.0400  13.23  15.78 136 Optimization terminated succ
lh            base |  0.0493  1.00  0.390  0.427  0.375  4.57 | 0.1478 0.0400  10.18  15.77 212 `gtol` condition satisfied.
lh       Lmax4.532 |  0.0231  0.47  0.406  0.417  0.259  4.80 | 0.1535 0.0400  10.17  18.13 323 `gtol` condition satisfied.
lh       Lmax5.124 |  0.0130  0.26  0.412  0.338  0.219  4.90 | 0.1563 0.0400  10.17  20.12 454 `gtol` condition satisfied.
lh         dcc0.02 |  0.0339  0.69  0.409  0.454  0.275  3.75 | 0.1478 0.0200  10.17  15.77 305 `gtol` condition satisfied.
lh         dcc0.03 |  0.0418  0.85  0.396  0.454  0.317  4.49 | 0.1478 0.0300  10.18  15.77 322 `gtol` condition satisfied.
lh       dpc0.1002 |  0.0444  0.90  0.394  0.502  0.311  4.87 | 0.1308 0.0400  10.18  15.77 246 `gtol` condition satisfied.
lh       dpc0.1169 |  0.0444  0.90  0.394  0.502  0.311  4.87 | 0.1308 0.0400  10.18  15.77 229 Optimization terminated succ
lh       dpc0.1336 |  0.0441  0.90  0.395  0.503  0.296  4.68 | 0.1336 0.0400  10.17  15.77 446 `gtol` condition satisfied.
lh       kmax11.70 |  0.0412  0.84  0.397  0.488  0.298  4.48 | 0.1478 0.0400  11.70  15.77 350 `gtol` condition satisfied.
lh       kmax13.23 |  0.0380  0.77  0.401  0.491  0.278  4.18 | 0.1478 0.0400  13.23  15.77 392 `gtol` condition satisfied.
```

* **Length is the big lever**: +15% per-coil length gives 0.60 (z0) / 0.47 (lh) of the base cost, +30% gives 0.35 / 0.26.
  It compounds with the hinge band: lh + 30% length is 0.013, 18% of the z = 0 base. The longer coils sit *further*
  from the plasma (d_pc 0.147 -> 0.15-0.17), and +15% at z = 0 is the only change so far that lowers B20 (0.40 -> 0.28).
* Curvature x1.3: 0.93 (z0), 0.77 (lh). Coil-coil 0.02: 0.93 (z0), 0.69 (lh). Both matter more once hinges are loose.
* Plasma-coil distance: almost nothing (0.97-0.99 at z0; 0.90 at lh, where the coils stop at 0.78 a and go no closer).
  Its only visible effect is B20 at z = 0 (0.40 -> 0.30 at 0.6 a).
* The cold starts for 5 and 6 coils (and the 4-coil control) at second order did not converge (600 iterations, cost > 1);
  discard them. Coil count is redone with a staged chain: `na/ncchain.py`, runs `nc{4,5,6}_s{1..4}`.


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
