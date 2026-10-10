# Which coilset has the best coil-field QS? (overnight 2026-10-09 → 10)

**Task (user).** Test whether a coil-field ripple term would help, then, "depending on what you find, look at different
approaches to identifying the best QS coilset we have."

## Short answer

1. **A ripple term on its own would add little.** B·n is already 65–86% ripple, and that is the part that turns into
   |B| symmetry-breaking (§1).
2. **The best existing coilset is FourierXYZ N6 on its own single-stage boundary (`matrix/on_xyz/xyzN6`, "xyzX").**
   * Coil-field QS error about 4e-4 at ρ 0.5, about 5× better than the arcB2 level example.
   * ε_eff ≤ 1.8e-4.
   * Clean nested surfaces.
   * A 29-coilset screen and four full checks agree (§2–3).
3. **The lever that moves coil-field QS is conductor length.** The objective barely matters, and coil count matters only
   modestly (§4–5). Length binds in every run.

| | 25% more length per coil (L × 1.25) |
|---|---|
| XYZ | B·n halves; QS 2.0–5.0e-4 |
| arcB2 free | B·n 2.4× lower; QS reaches its target's (7.6e-4 vs 7.1e-4), so it is now **target-limited** |
| arcB2 level hinges | B·n −43%, QS −39%; still removable (clearance 7.4 → 4.1 cm) |

   * A 5th coil per half-period adds nothing at fixed total length.
   * With the extra length it helps XYZ a little (best coil field so far, xyz5: QS 1.7–2.8e-4 at ρ 0.5) and arcB2 not at all.
4. **The full evaluation pipeline has a resolution limit**, which matters for the best coils (§3). For XYZ coils the
   target's own coordinates describe the coil field as well as or better than the traced-surface reconstruction. Their
   QS is therefore bracketed by the proxy and the traced value, which differ by up to 2.5× at the 2–5e-4 level.

## 1. Would a coil-field ripple term help? Mostly no (`spectrum.py`)

On the target boundary, the B·n residual and the coil's |B| symmetry-breaking were split into toroidal bands. Here n
is in units of NFP, and |n| 6–9 is the ripple of 4 coils per half-period.

| | level example | free arcB2 |
|---|---|---|
| share of squared B·n in the ripple band | **86%** | **65%** |
| \|B\|-breaking per unit B·n, ripple band | 1.15 | 1.08 |
| \|B\|-breaking per unit B·n, \|n\| ≤ 2 / 3–5 | 0.23 / 0.52 | 0.21 / 0.49 |

* As I argued earlier, low-order B·n is indeed about 4× less harmful to QS than ripple.
* But low-order modes are only 14–35% of the residual, and B·n already puts most of its weight on ripple.
* So a ripple or QS term could only re-weight a small remainder. The ripple is a floor set by the coils (length, count,
  distance), not something the objective neglects.

## 2. Screening every coilset with a cheap proxy (`spectrum.py`, `screen.sh`, `spectrum/screen_table.md`)

**The proxy.** Evaluate the actual coil |B| (Biot–Savart, currents rescaled) on the *target's* flux surfaces, and fit
Boozer modes in the target's Boozer angles. It needs no tracing or solve and takes about 1 min per coilset.

**Screen of 29 QH coilsets, QS at ρ 0.5** (full table in `spectrum/screen_table.md`):

| rank | coilset | B·n | proxy QS | target QS |
|---|---|---|---|---|
| 1 | XYZ on B_xyz (`matrix/on_xyz/xyzN6`, xyzX) | 6.3e-4 | 3.8e-4 | 1.4e-4 |
| 2 | XYZ on precise_QH (`matrix/base/xyzN6`, xyzB) | 9.3e-4 | 5.9e-4 | 2.2e-4 |
| 3–4 | XYZ on B_arcB2 (fixed_ss_eq, matrix) | 5.2e-4 | 7.3–7.7e-4 | 6.7–7.1e-4 |
| 5 | best arcB2 (`fixed_ss_eq/al_arcB2`) | 1.45e-3 | 1.15e-3 | 6.7e-4 |
| 7 | best level hinges (`hires/refine`, res 12) | 2.28e-3 | 1.71e-3 | 5.7e-4 |
| — | planar N7 (all) | ≥ 6.9e-3 | ≥ 8.7e-3 | |
| — | z = 0 (precise_QH) | 1.8e-2 | 5.4e-2 | |

* **Coil-field QS ranks with B·n**, and the representation dominates: XYZ < arcB2 < planar.
* **A good target only pays off with good coils.** On B_arcB2 (target 7e-4), XYZ is target-limited (1.1×). On
  precise_QH and B_xyz (targets 1.4–2.2e-4), it is coil-limited (2.6–2.8×).
* Three QUASR "device" equilibria (q1632037, q1632468, q1775228) read target QS of about 0.1. They are unsolved
  boundaries, so ignore those rows.

## 3. Full checks of the leaders (traced-surface pipeline at res 12, plus direct and ε_eff checks)

| ρ | 0.25 | 0.5 | 0.75 | 1.0 | ε_eff ρ 0.5 / 0.9 | proxy ρ 0.5 |
|---|---|---|---|---|---|---|
| arcB2 level (cont/refine) | 1.80e-3 | 1.95e-3 | 2.86e-3 | 5.71e-3 | 4.2e-4 / 1.29e-3 | 1.77e-3 |
| arcB2 free (B_arcB2) | 9.8e-4 | 1.73e-3 | 2.65e-3 | 5.25e-3 | 1.51e-3 / 3.44e-3 | 1.27e-3 |
| XYZ on precise_QH (xyzB) | 5.8e-4 | 6.8e-4 | 9.1e-4 | 2.11e-3 | 3.1e-4 / 4.3e-4 | 5.90e-4 |
| **XYZ on B_xyz (xyzX), best existing** | **2.8e-4** | **4.0e-4** | **6.1e-4** | **1.56e-3** | **1.3e-4 / 1.8e-4** | **3.81e-4** |
| XYZ nc4, L × 1.25 (xyz4L, new) | 2.5e-4 | 5.0e-4 | 5.7e-4 | 1.52e-3 | 3.1e-4 / 2.8e-4 | 2.03e-4 |
| XYZ nc5, L × 1.25 total (xyz5, new) | 1.65e-4 | 2.8e-4 | 4.0e-4 | 1.30e-3 | 1.3e-4 / 2.2e-4 | 1.68e-4 |
| target of the B_xyz cases (ρ × 0.95) | 1.35e-4 | 1.39e-4 | 3.1e-4 | 1.16e-3 | 1.9e-5 / 1.5e-4 | 1.37e-4 |

![QS and eps_eff of four checked coilsets](best_coilsets.png)

**Surfaces.**
* Every case is nested, with no islands, confined out to 6% beyond the boundary, and ι matches its target to ≤ 0.003.
* **Edge rational on B_xyz:** ι = 1.2503 at the edge, i.e. 5/16 per field period, so the Poincaré punctures show 16
  slowly filling arcs.
  * Shear there is nearly zero (ι 1.2503–1.2508 over ρ 0.9–1.06).
  * No island is resolved, but any resonant m = 16 error would open a chain.
  * This is a robustness risk of the B_xyz *target*, not a defect of its coils.

**Which equilibrium describes the coil field better? This sets how far each row can be trusted.** The measure is the
traced lines' ρ spread (max − min, ρ0 0.3–0.9) in each coordinate system:

| case | target's coordinates | traced-surface equilibrium | better |
|---|---|---|---|
| level | 0.028 | 0.003 | traced |
| free arcB2 | 0.021 | 0.011 | traced |
| xyzB | 0.0033 | 0.0078 | **target** |
| xyzX | 0.0020 | 0.0030 | **target** |
| xyz5 | 0.0026 | 0.0026 | tie |
| xyz4L | 0.0024 | 0.0067 | **target** |

What this means:
* The pipeline (trace one line at ρ 0.95, fit the surface, deform the target, re-solve at res 12) has a floor of about
  0.003–0.008 in ρ spread.
* For arcB2 coils, which miss their target by 0.02–0.03, that floor is fine and the traced numbers stand.
* For XYZ coils, which already sit on the target's surfaces to 0.002–0.003, the reconstruction adds more error than it
  removes. Their traced QS over-reads, and the proxy, evaluated on the better-fitting target surfaces, is the closer
  number.
* **So XYZ QS at ρ 0.5 lies between proxy and traced:**

| case | range |
|---|---|
| xyz5 | 1.7–2.8e-4 |
| xyz4L | 2.0–5.0e-4 |
| xyzX | 3.8–4.0e-4 |
| xyzB | 5.9–6.8e-4 |

* The ranking is the same either way, except that xyz4L vs xyzX is uncertain.
* The direct Biot–Savart Boozer fit on each traced equilibrium agrees with that equilibrium's own |B| (2–27%). This
  confirms the |B| on those surfaces, not that the surfaces are the true flux surfaces.

**ε_eff.**
* XYZ on B_xyz is 0.013–0.018%, 3–10× below the arcB2 coil fields.
* The L × 1.25 and nc5 runs do not improve ε_eff over xyzX; xyz4L is worse at ρ 0.5. The caveat above applies: these
  come from the traced equilibria.

## 4. Coil count vs conductor, FourierXYZ (`../../nc5xyz/`)

All runs: XYZ N6 on B_xyz, `run_al.py` (maxiter 800, `--cc-k 30000 --link-mu 1e8`), precise_QH bounds except length.
Cold starts are circles at r/a 2.6. Each run takes 5–12 min.

| run | coils per half-period | L per coil | total L (unique) | B·n | proxy QS ρ 0.5 | d_pc / bound |
|---|---|---|---|---|---|---|
| xyz4_cold | 4 | 1.00 × 2.93 m | 11.7 m | 6.38e-4 | 3.95e-4 | 1.26 |
| xyzX (matrix, warm) | 4 | 1.00 | 11.7 m | 6.26e-4 | 3.81e-4 | 1.26 |
| xyz5_Lfix | 5 | 0.80 | 11.7 m | 8.78e-4 | 3.71e-4 | 1.13 |
| xyz4_L125 | 4 | 1.25 | 14.6 m | 2.98e-4 | 2.03e-4 | 1.25 |
| **xyz5** | 5 | 1.00 | 14.6 m | **1.91e-4** | **1.68e-4** | 1.19 |

* **The cold start reproduces the warm matrix result**, so start history is not a factor.
* **At fixed total conductor, a 5th coil buys nothing.** Shorter coils lose shaping, and the ripple moves to |n| = 10.
* **25% more length per coil halves B·n and the proxy QS excess.** A 5th coil on top gains another 1.6× in B·n but only
  about 1.2× in QS.
* **Length, κ and κ_MS bind in every run.** κ ends 0.04–0.1% over its bound, the same tolerance class as earlier
  accepted results. This matches Helios (memory: length is the only expensive constraint).

## 5. arcB2: the same levers (`../../nc5xyz/`)

| run | coils per half-period | L per coil | start | B·n | proxy QS ρ 0.5 | target QS | notes |
|---|---|---|---|---|---|---|---|
| free, matrix on_arcB2 | 4 | 1.00 | — | 1.48e-3 | 1.27e-3 (traced 1.73e-3) | 7.1e-4 | reference |
| arcL125_free | 4 | 1.25 | warm (reference) | **6.12e-4** | **7.61e-4** | 7.1e-4 | κ_MS 0.85, d_pc 1.47 |
| arc5 | 5 | 1.00 (total 1.25×) | cold arcs | 6.60e-4 | 7.72e-4 | 7.1e-4 | κ_MS 0.90, d_pc 1.20 |
| level, cont/refine | 4 | 1.00 | — | 2.27e-3 | 1.77e-3 (traced 1.95e-3) | 6.9e-4 | removable, +7.4 cm |
| **arcL125_level** | 4 | 1.25 | warm (cont/refine) | **1.29e-3** | **1.08e-3** | 6.9e-4 | **removable, +4.1 cm** |
| arc5_level | 5 | 1.00 | warm (arc5) | *(running on CPU)* | | 6.9e-4 | GPU OOM twice at build |

* **For arcB2 free, length is the whole effect.** At L × 1.25, 4 or 5 coils give the same B·n and the same QS, both at
  the target's own value. The free arcB2 coils become target-limited: further gains need a better single-stage target,
  not better coils.
* **Level hinges: −43% B·n and −39% QS from L × 1.25, still demountable.** The level/free ratio grows from 1.5× to 2.1×.
  The free coils use the extra length better, and level's remaining error is still about 70% ripple
  (proxy band 7.9e-4 of 1.08e-3).
* These runs keep the boundary fixed. Coil continuation and single stage at L × 1.25 were not run.

## 6. Recommendations

1. **Decide the length budget first.** It is the binding constraint and the largest single lever on coil-field QS
   (about 2× for 25%). Gil's Table III length is a design choice, and every result so far is priced at that value. A
   curve of B·n and QS vs L × {1.0, 1.1, 1.25, 1.5}, for level arcB2 and the XYZ reference, is cheap (about 10 min per
   warm run) and would show where the return flattens.
2. **For QS, use XYZ on B_xyz as the reference coilset** ("best QS coilset we have"). For the demountable track,
   `arcL125_level` is the new best level result, if 25% more length is acceptable.
3. **Don't add a ripple or QS term to the coil objective.** B·n already carries it (§1). Ripple falls with
   length (and count) instead.
4. **Fix the evaluation for near-perfect coils.**
   * The traced-surface pipeline over-reads once the coil field matches its target to about 0.003 in ρ.
   * Options: compute Boozer surfaces of the coil field directly (simsopt-style `BoozerSurface`); fit several traced
     surfaces instead of one; or pick per case whichever equilibrium has the smaller traced-line spread, as done here.
   * Until then, report XYZ numbers as a proxy–traced bracket.
5. **Single stage at the new length.** Once the length budget is set, re-run the level continuation (`cont.sh`) with
   that length. The boundary may then trade some of the coil gain into QS.
6. **Watch the B_xyz edge rational** (ι = 5/16 per period, zero shear) if that target is used further.

## Files

| file | contents |
|---|---|
| `spectrum.py`, `screen.sh`, `screen_table.py`, `spectrum/` | B·n vs \|B\| bands, proxy QS, screen table |
| `chain_xyz.sh`, `xyzX/`, `xyzB/`, `xyz5/`, `xyz4L/` | full pipeline + direct check + ε_eff + figures |
| `best_coilsets.png` | QS and ε_eff, four checked coilsets |
| `../../nc5xyz/` | count vs conductor runs (`queue*.sh`, `starts/`, `xyz5/`, `xyz5_Lfix/`, `xyz4_L125/`, `xyz4_cold/`, `arc5/`, `arcL125_free/`, `arcL125_level/`, `arc5_level/`) |
| `bounds/precise_qh_gil_nc5_Lfix.json`, `bounds/precise_qh_gil_nc4_L125.json` | L × 0.8 and L × 1.25 |
