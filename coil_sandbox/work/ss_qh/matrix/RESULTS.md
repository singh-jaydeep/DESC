# precise_QH: boundary × coils matrix (run 2026-10-07/08, overnight)

## The answer

**Does a single-stage boundary built for one coil representation help that representation more than boundaries built
for other representations? Yes for the two restricted representations (planar and arcB2), no for FourierXYZ. One
boundary, B_arcB2, helps all three, and it is the best boundary of the three for FourierXYZ.**

⟨|B·n|⟩/⟨B⟩ after converged stage-2 refinement (`run_al.py`, every cell `gtol`, optimality and violation < 1e-6).
The factor in parentheses is ÷ the same column on precise_QH. Bold marks each column's best boundary.

| boundary ↓ / coils → | planar N7 | arcB2 M5 | FourierXYZ N6 | QS at the edge vs precise_QH |
|---|---|---|---|---|
| B0: precise_QH | 2.205e-2 | 5.437e-3 | 9.332e-4 | 1× |
| B_planar (single stage with planar coils) | **6.866e-3 (0.31)** | 2.335e-3 (0.43) | 1.098e-3 (1.18) | 6.9× worse |
| B_arcB2 (single stage with arcB2 coils) | 1.687e-2 (0.76) | **1.484e-3 (0.27)** | **5.171e-4 (0.55)** | 3.3× worse |
| B_xyz (single stage with XYZ coils) | 2.166e-2 (0.98) | 3.079e-3 (0.57) | 6.260e-4 (0.67) | 1.0× |
| reference: old arcB2 boundary `ss_k4` | 1.852e-2 (0.84) | 1.449e-3 (0.27) | 5.192e-4 (0.56) | 3.2× worse |

* **Planar N7 is best on its own boundary, by a wide margin.** It reaches 0.31× its precise_QH value on B_planar,
  compared with 0.76× on B_arcB2 and 0.98× on B_xyz. That is 2.5× lower B·n than on the next-best boundary.
  Planar coils gain almost nothing on a boundary that was not shaped for them.
* **arcB2 M5 is best on its own boundary,** at 0.27×. B_planar is second (0.43×, 1.6× worse than its own) and B_xyz
  third (0.57×).
* **FourierXYZ N6 is not best on its own boundary.** B_arcB2 beats B_xyz for XYZ (5.17e-4 vs 6.26e-4, 1.2×), and
  B_planar makes XYZ worse than precise_QH (1.18×). This row is a weak test, though. With identical weights, XYZ's
  B·n (9e-4) is already well inside the B·n tolerance τ = 5e-3. Its single stage therefore had little reason to move
  the boundary: its own coils improved only 29% and QS did not change at all.
* **The diagonal effect is real, and so is a general one.** Within each column the targeted boundary wins for the two
  restricted representations. B_arcB2 is also coil-friendly in general: it is the only boundary that improves all
  three, and its QS cost is moderate.
* **The gains are paid for in QS.** Roughly, the more QS a boundary gave up, the more it helped the representation it
  was made for:
  * B_planar gave up the most: QS is 6.9× worse at the edge and 14× worse at ρ = 0.5, and it has a sharp concave
    crease with a 6.2 mm radius of curvature (see below).
  * B_arcB2 lost 3–4× in QS at ρ ≥ 0.5.
  * B_xyz lost nothing; its QS is slightly better at ρ ≤ 0.5.
* **Ratio to the best coils on the same boundary (÷ XYZ).** arcB2 is 5.8× XYZ on precise_QH, 2.9× on B_arcB2 and
  2.1× on B_planar. Planar is 24×, 33× and 6.3× XYZ on the same three boundaries.
* **Reproducibility check.** The redone arcB2 boundary matches the old `ss_k4` boundary: refined arcB2 1.484e-3 vs
  1.449e-3, with the same QS, A and volume.

## Caveats (measured, not fixed)

* **κ between nodes (trap 5).** Dense-check κ reads up to 1.0019 and κ_MS up to 1.006 (arcB2 on B_xyz), because the
  optimizer puts peaks between the 601 enforcement nodes. No backoff was added, so the cells stay comparable with
  earlier results. L and κ_MS reading 1.000 are solver noise.
* **Single starts.** Each cell is one local optimum. The off-diagonal starts on B_planar were far from fitting that
  boundary (start B·n 6.5e-2 for arcB2, 6.4e-2 for XYZ; R0 moved 13 mm inward), and XYZ converged there after only
  17 inner iterations. A different start, such as the diagonal coils converted to another representation, could move
  those cells.
* **One set of weights.** τ_QS = 0.01695 (×10 of precise_QH's QS rms) and τ_bn = 5e-3 for every representation, as
  requested. With these weights XYZ is under-driven (see above).
* **B_planar is not a clean boundary.** It has a concave crease: maximum concave curvature 161 /m at θ ≈ 0.54,
  ζ·NFP ≈ 4.3, and 1.7% of the area has a concave radius below 165 mm. The other boundaries keep 240–260 mm, like
  precise_QH (232 mm). The crease raised the plasma distance-field spline error to 2.6e-5 m, still 5% of the
  0.556 mm node margin, and every verdict comes from the exact dense check.
* **Volume.** Every single-stage boundary sits at the lower edge of the ±2% volume band (V/V0 = 0.980). The band
  binds on every boundary; B·n alone would shrink the plasma.

## Deviations from the handoff (all logged in NOTE.txt)

1. **arcB2 `--cc-k 80000`, not 40000.** On precise_QH 35,937 of 40,000 coil-coil row slots were already active at
   the start. At 80,000 the peak memory was 6.4 GB and no slot overflowed in any run.
2. **An extra k = 4 continuation ("k4b") for every representation.** The k = 1–4 single-stage chains hit early stops
   on `ftol` 1e-6: planar at k = 2 after 3 iterations, arcB2 at k = 4 after 8 iterations with optimality 8.7e-3.
   The original arcB2 boundary had received a full 100-iteration k = 4 with `xtol` 1e-12. So each representation got
   one more k = 4 step of 100 iterations from its own k = 4 result, with `ftol = xtol = 1e-12` and the frozen
   anchors (τ_QS 0.01695, V0 0.30352, B·n normalization 0.42526). The matrix rows are the k4b boundaries.

   | | k4b iterations | B·n of the single-stage coils, k = 4 → k4b |
   |---|---|---|
   | planar | 100 | 7.40e-3 → 6.95e-3 |
   | arcB2 | 61 (stopped on `xtol`) | 2.37e-3 → 2.02e-3 |
   | XYZ | 100 | 6.83e-4 → 6.65e-4 |
3. **GPU workarounds, neither of which changes the numerics.** Both are XLA settings that only affect which kernels
   run and how GPU memory is reserved.
   * The arcB2 single stage died once with "Autotuner could not compile any configs". The rerun used
     `XLA_FLAGS=--xla_gpu_autotune_level=0`, the fallback from `work/compare/queue2.sh`.
   * The XYZ single stage died twice on a cuFFT batched-plan failure, for 301- and 201-node FFTs; a size that had
     passed once failed later, so it is not size-specific. The Biot–Savart chunk sizes made no difference. The fix was
     `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, which leaves VRAM for cuFFT's own plan memory. This is the first QH XYZ
     single stage (H4) to complete on this machine.
   * The first planar single-stage attempt segfaulted at the first optimizer call; an identical rerun passed.

## Files

* `matrix.md`, `matrix.json`: every cell (margins, linked, active, solver outcome, ÷ row XYZ, ÷ column B0) and the
  boundary table (QS at 4 radii, A, R0, a, iota, V/V0, single-stage coils B·n). Built by `matrix.py`.
* `base/`, `on_planar/`, `on_arcB2/`, `on_xyz/`: `run_al.py` results (`result.h5`, `result.json`, `outer.json`,
  `run.log`). Folders ending in `*_build` hold the `--maxiter 0` checks.
* `ss/<rep>/`: the k = 1–4 single stages. `ss/<rep>_k4b/` holds the continuations; their `eq_k4.h5` files are the
  matrix boundaries. `ss/*fail*`, `ss/planar_segv1`, `ss/arcB2_autotune_fail1` are the failed attempts.
* `view.html`: all 12 coilsets, each on its own boundary.
* Scripts: `step0_build.sh`, `step1_base.sh`, `step2_ss.sh`, `step2b_chain.sh`, `step2c_xyz.sh`, `step3_cross.sh`,
  `step3_builds.sh`, `step3_all.sh`, `view.sh`, `matrix.py`. `NOTE.txt` has the timeline and every decision.

## Possible next steps (not started)

* Rerun the XYZ single stage with a tighter τ_bn (for example 5e-4), so XYZ's B·n carries weight. That would test
  XYZ's diagonal properly.
* Start the off-diagonal cells from the diagonal coils converted to the other representation (an open decision in the
  handoff), to check the single-start caveat on B_planar.
* Add a ~0.3% κ/κ_MS backoff if strict dense-check feasibility matters.
