# arcB2 with every hinge on z = 0 (one demountable joint plane): precise_QH, 2026-10-08

## The answer so far

**From near-converged starts, single stage does not close the joint-plane gap.** On its own single-stage boundary
the joint-plane arcB2 reaches B·n 6.37e-3, while free-hinge arcB2 on its own single-stage boundary (B_arcB2, the
matrix) reaches 1.48e-3: still **4.3×**, against 3.3× on precise_QH. Single stage helps the joint-plane coils as much
as it helps free ones (0.35× vs 0.27×), but it does not change the geometry that makes the joint plane expensive:
precise_QH's helical axis puts the cross-section up to 150 mm below z = 0, and **the single-stage boundaries keep
that excursion unchanged** (see "Why" below). The next test is a boundary from a different basin (cold start), not
more boundary modes: **k = 5 gave nothing** (stalled after 12 iterations, B·n unchanged).

⟨|B·n|⟩/⟨B⟩, every row converged with `run_al.py` (lsq-auglag-composite, gtol: optimality and violation < 1e-6)
unless marked; dense check against the unbacked-off bounds.

| coils (arcB2 M5) | boundary | B·n | ÷ free on same boundary | notes |
|---|---|---|---|---|
| free hinges, warm start (matrix) | precise_QH | 5.437e-3 | 1 | |
| free hinges, cold start + link guard | precise_QH | 6.421e-3 | 1.18 (cold-start cost) | `base_free_link`; never linked |
| **all hinges z = 0** | precise_QH | **1.801e-2** | **3.3 (2.8 vs the same cold start)** | `base_z0`; removable vertically |
| **hinges level per coil, own heights** | precise_QH | **8.768e-3** | **1.37 vs the same cold start (1.6 vs warm)** | `base_level` (from base_z0); removable; coil z −161, −45, −64, +62 mm |
| free hinges (matrix) | B_arcB2 | 1.484e-3 | 1 | |
| **all hinges z = 0** | **B_z0** (joint-plane single stage, k4b) | **6.372e-3** | **4.3 vs free on B_arcB2** | `on_z0/arcB2_z0`; removable, 0.144 m hinge clearance |
| all hinges z = 0 | B_z0 at k = 5 | 6.356e-3 | 4.3 | `on_z0_k5/arcB2_z0`; −0.25% vs k4b |

All joint-plane results: every hinge exactly at z = 0 (max |z| 0.0), no linked pair, every arc liftable vertically
(sandbox/demount.py ray casting). L, κ read 1.000/1.001 (between-node peaks, as in the matrix).

### Single stage (`single_stage.py --fix-hinge-z 0`, matrix arcB2 weights, from `base_z0`)

| step | B·n of the single-stage coils | QS ρ = 0.5 | QS edge | ι min | A |
|---|---|---|---|---|---|
| start (precise_QH) | 1.80e-2 | 2.2e-4 | 1.4e-3 | 1.252 | 8.00 |
| k = 1 | 1.62e-2 | 3.4e-3 | 7.5e-3 | 1.255 | 7.98 |
| k = 2 | 1.08e-2 | 3.9e-3 | 1.2e-2 | 1.190 | 8.11 |
| k = 3 | 8.25e-3 | 3.8e-3 | 8.6e-3 | 1.182 | 8.09 |
| k = 4 | 6.97e-3 | 3.7e-3 | 8.0e-3 | 1.181 | 8.09 |
| k4b (k = 4, +100 its) | 6.82e-3 | 3.7e-3 | 8.0e-3 | 1.181 | 8.09 |
| k = 5 (stalled, 12 its) | 6.81e-3 | 3.7e-3 | 8.0e-3 | 1.181 | 8.09 |
| ref: B_arcB2 (matrix, free hinges) | 2.0e-3 | 8.8e-4 | 4.8e-3 | ~1.24 | 8.17 |

* Identical weights to the matrix (τ_QS 0.01695, τ_bn 5e-3, V0, B·n norm), but B·n started at 4.5× its tolerance
  (82% of the cost) against 1.3× (25%) in the matrix arcB2 run, so this run traded QS for B·n much harder: QS at ρ = 0.5
  is 17× precise_QH (B_arcB2: 4×), and ι sits just under its 1.19 bound (a weighted penalty, not a constraint).
* k = 1..4 each ran to maxiter; k4b barely moved (−2%). The |m|,|n| = 4 boundary shell grew 5× (2e-4 → 1.1e-3), so the
  top shell was in use, which motivated k = 5.

### (b) k = 5: feasible, but no gain as run

* Feasible: eq and boundary resolution M = N = 8, so |m|,|n| ≤ 5 (120 free modes vs 80) is well resolved; memory
  ~5 GB and ~12 min/100 its, like k = 4.
* Result: stopped on xtol after 12 iterations; cost 1.628 → 1.620 (−0.5%); B·n 6.82e-3 → 6.81e-3; nothing else moved.
  Not a minimum: optimality stayed 0.04 while the steps shrank 1e-4 → 1e-11 with ~10 function evaluations per step
  (trust region collapsing on rejected steps). No equilibrium-solve failure in the log. Same family as the matrix's
  early ftol/xtol stops; my unverified guess is the weighted-penalty hinges of the coil terms sitting exactly at their
  bounds (L 1.001, d_cc 1.001). **Not recommended to push k further** from this state; the limit is the basin (and this
  stall), not the mode count.

### Per-coil horizontal hinges (user's relaxation, 2026-10-08)

`run_al.py --horizontal-hinges` (sandbox/demount.HingeLevel: z_1 − z_0 = 0 per coil, linear, eliminated exactly; each
coil's height free), from the converged base_z0 coils, link guard on. Converged (gtol, 11 outer / 215 inner, 6.3 min).
B·n 1.801e-2 → **8.768e-3**: the per-coil heights (−161, −45, −64, +62 mm) recover ~80% of what the single plane cost on
precise_QH; what is left (1.37× the free cold start) is roughly the price of horizontality. Removable, min hinge clearance
+0.019 m (base_z0 +0.112). For reference, free arcB2 chords tilt 37–80° (warm) or 11–38° (cold) from horizontal.
Single stage with the same constraint: `../ss_level.sh` (results below when done).

## Why: the vertical excursion is untouched

precise_QH's magnetic axis is a helix of radius ~0.17 m (R_01 = 0.161, Z_0,-1 = −0.131 on the boundary) with a =
0.124 m; it carries most of ι ≈ 1.25. Per toroidal angle, the cross-section centre sits up to 150 mm below z = 0:

| boundary | max \|z-centre\| | \|z\| of the R-extremes / a (median, max) | z = 0 chord ÷ R extent (min, median) |
|---|---|---|---|
| precise_QH | 144 mm | 1.62, 2.15 | 0.00, 0.27 |
| B_arcB2 (matrix ss) | 159 mm | 1.69, 2.20 | 0.00, 0.17 |
| B_z0 (joint ss) | 156 mm | 1.55, 2.19 | 0.00, 0.30 |
| HSX (DESC example, mirrored) | 135 mm | 1.36, 2.43 | 0.02, 0.36 |
| **WISTELL-A (DESC example, mirrored, scaled)** | 72 mm (0.58 a) | **1.08, 1.73** | **0.34, 0.56** |

(Transverse arcs are graphs over their chord, so the cross-section's R-extremes must lie near the z = 0 chord; on
precise_QH z = 0 misses some cross-sections entirely, which forces D-shaped coils.) The single stage reduced B·n by
reshaping cross-sections locally while keeping the helix: the joint-plane boundary is 0.27 → 0.30 on the last column.

## (c) Cold-start candidate: WISTELL-A, mirrored and scaled to precise_QH's minor radius

**Why:** of the QH equilibria available, it is the only one where z = 0 cuts every cross-section through a large
part of its width (≥ 34%, median 56%), with the R-extremes ~1 a from z = 0 (precise_QH 1.6 a). It is a real QH
(helicity (1, ∓4), NFP 4, vacuum) from a different basin: excursion 0.58 a, A 6.67, |ι| 1.03–1.10, QS at ρ = 0.5
8.9e-3 (40× precise_QH) — i.e. genuinely cold for QS, warm for joint-plane geometry.

**Prepared (CPU only; nothing launched):**
* `cold/eq_wistell.h5`: WISTELL-A mirrored ζ → −ζ (ι > 0, helicity (1, 4) like precise_QH) and scaled by c = 0.411
  (R0 0.830, a 0.1243 = precise_QH's, so `bounds/precise_qh_gil.json` applies as is; |B| unchanged). Both maps exact:
  force residual scales by exactly 1/c, ι flips sign. L, M, N = 12, 12, 6 (dim_x larger than precise_QH's 8, 8, 8).
* `cold/starts/z0_wistell_packed.h5`: z = 0 lens start (make_z0_start.py, power 0.7, clearance 2.0 d_pc), better margins
  than on precise_QH: L 0.884, κ 1.33 (only violation), κ_MS 0.70, d_cc 1.17, d_pc 1.05, unlinked, B·n 0.26.

**Proposed sequence (each needs your OK on the config):**
1. **W1, fixed-equilibrium baselines (~2 × 20 min):** `run_al.py` on `cold/eq_wistell.h5` from the lens start, with
   `--fix-hinge-z 0 --link-mu 1e8` and a free-hinge control with `--link-mu 1e8`. This alone answers "does a
   z0-friendly QH make the joint plane cheap?" (on precise_QH: 2.8×).
2. **W2, single stage** from W1's joint-plane coils, `--helicity 1,4 --fix-hinge-z 0`. Decisions for you:
   * targets: its own (A ≈ 6.67 → band [6.2, 7.2], ι_min ≈ 1.0) or precise_QH's (A [7.5, 8.5], ι ≥ 1.19, which would
     force a large reshaping first);
   * QS weight: τ_QS = 0.01695 absolute (matrix value) — WISTELL starts far above it, so QS will dominate the cost at
     first, i.e. a true cold start for QS; or τ_QS relative to its own start;
   * resolution: keep 12, 12, 6 (build check needed; memory unknown) or re-solve at 8, 8, 8 first.
3. Alternatives considered: HSX (same minor radius, A 10, but z = 0 coverage only marginally better than precise_QH:
   min 0.02); a reduced-helix precise_QH (scaling the m = 0 modes and re-solving diverged, |F| ~ 1e20; it would need a
   perturbation continuation, and it attacks ι directly since the helix carries it).

## Files

* `../base_z0`, `../base_free_link` (fixed scaling), `../base_free_link_v1` (before the fix), `../base_free`
  (unguarded, linked), `../smoke*`, `../build`: stage-2 runs on precise_QH.
* `../ss_z0` (k = 1..4), `../ss_z0_k4b`, `../ss_z0_k5`: single stage; `eq_k*.h5`, `coils_k*.h5`.
* `../on_z0/arcB2_z0`, `../on_z0_k5/arcB2_z0`: converged refinements on B_z0.
* `../cold/`: `eq_wistell.h5`, `prep_wistell.py`, `survey.py`, `extremes.py`, `shrink_helix.py` (failed attempt),
  `starts/`.
* `view.html` (`view.sh`): every coilset on its own boundary.
* Scripts: `../run_base.sh`, `../ss_z0.sh` (dry | chain | k4b | refine | all | k5 | refine5 | all5),
  `../make_z0_start.py`, `../z0_hinges.py`, `../from_checkpoint.py`. Timeline: `../NOTE.txt`.
* Code: `run_al.py --fix-hinge-z/--link-mu`, `single_stage.py --fix-hinge-z`; main repo commit 31ed99491
  (`initial_penalty_rows` + jac-scale exclusion + unit test).
