# Acceptance run 8a (design F stage 2): two KKT optima 15x apart in B.n

2026-11-17. Problem: design F's equilibrium and bounds (fa_qa2_a0167), start `F_z0_c25_p07` (z = 0 lens), every
hinge on z = 0, fold <= 60, current band 0.5, `lsq-auglag-composite` (secant inner, finish_steps 5), coil-coil rows
(gap from the bounds, 1025 nodes, 30k slots), plasma-coil hard minimum (32 x 32). Runs in `paper/runs/` (not
committed); commands in each run's `cmd.txt`.

## Result

| run | ended | mean B.n / <B> | linked | note |
|---|---|---|---|---|
| poster (js/coil-auglag) | KKT, 190 s | **6.0e-5** | no | reference |
| port, plasma rows (128 x 128) | stopped | 2.3e-3 | **yes** | slot overflow dropped contacts (fixed: closest-first) |
| port, hard min, 1000 nodes/arc | KKT, 572 s | **9.25e-4** | no | |
| port, hard min, LinearGrid(N=300) | KKT, 448 s | **9.26e-4** | no | curvature grid ruled out |
| port + branch's 55 per-pair rows | stopped | 2.8e-3 | **yes** | |
| port + link guard (signed rows, mu 1e8) | KKT, 1101 s | **9.42e-4** | no | parked a pair at contact, then separated |
| warm start from poster's optimum | KKT, 10 its | 6.005e-5 | no | stays |
| warm start from port's optimum | KKT, 19 its | 9.24e-4 | no | stays |

## Findings

1. **The port is correct.** At the start the two codebases build identical problems (objective, every constraint,
   the projected problem, to 1e-15) except that the branch carries 55 extra per-pair coil-coil rows (always
   inactive: lower bound 0) and 10 plasma-coil rows where upstream's minimum has 20 (each distance twice, one per
   field period). With the 55 rows restored the port follows the poster's iterates to 2e-10 for 5 iterations.
2. **Both coilsets are KKT points of the same problem**: each warm start stays put (L and kappa active; kappa
   0.09% / 0.11% over on the dense check).
3. **The basin is chosen by the variable scaling.** `x_scale="jac"` divides each variable by the column norm of the
   whole weighted Jacobian, inactive rows included. The branch's extra inactive rows change some column norms by up
   to 1e15 relative to the port, the early steps differ, and the runs separate into two basins. Every port variant
   that stayed unlinked (two curvature grids, with and without the link guard) reached 9.3e-4.
4. **The basins are separated by a coil-coil clearance barrier.** On the straight line between the two optima, B.n
   is monotone (no barrier in the field), L, kappa and d_pc stay at their bounds, but d_cc falls to 0.03 of its bound
   (1.2 mm) at the midpoint: unique coil 2's stellarator image (coil 9) and its next-period copy (coil 12) pass
   within 1.2 mm. Coil 2 is the one that differs (3.87 m long at the poster's optimum, 3.63 m at the port's; the
   others within 0.5%). The optima differ in how coil 2 sits relative to its own symmetry images.
5. **Linking is a separate, path-dependent hazard.** Two runs linked coil pairs in one accepted step (a step that
   jumps a pair through each other leaves every unsigned row positive). The poster's run did not, by path. The
   link guard (signed per-pair rows with a large initial penalty) prevents it and keeps steps healthy; it does not
   change the basin.

## Consequences for the paper

- Report stage-2 B.n as the best over a few starts or scalings, and state the spread; a single run's value depends
  on solver-path details as small as inactive rows in the variable scaling.
- The sensitivity itself is a result worth stating: KKT points 15x apart in B.n for one formulation, separated by a
  clearance barrier between a coil and its symmetry images.
- Keep the link guard available (paper/lib/experimental.py, stage2.py --exp-pair-rows signed --link-mu 1e8).
- Open: a scaling that ignores inactive rows (reproducible, independent of rows with no role) vs the current
  max-column-norm rule; multistart cost.
