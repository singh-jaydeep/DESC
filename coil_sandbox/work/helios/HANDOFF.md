# Helios coil trials: status and handoff (2026-09-17)

Read this together with `coil_sandbox/CLAUDE.md` (machine rules, the recipe and its reasons).

## What we are doing

- **Target.** `sample_equilibria/helios_repro.h5`, a DESC reproduction of Thea Energy's Helios QA plant: NFP 2, R0 7.96 m, a 1.77 m, A 4.5, ⟨β⟩ 2.6%, 4.6 MA net toroidal current. It matches Table 1 of Kruger, Elder, Gates et al., *Planar coil design for the Helios stellarator fusion power plant*, Fusion Eng. Des. 230 (2026) 115893 (`sample_equilibria/Thea_planar_coils_Helios.pdf`). `Thea_planar_coils.pdf` is the earlier Eos paper, not this machine.
- **Coils.** 12 encircling coils (3 unique per half period), stage 2 only (no shaping coils yet).
- **Constraints.** From the paper's Table 2, as given (`bounds/helios_kruger2026.json`):
  - κ ≤ 1.5 /m
  - coil–plasma distance ≥ 1.2 m
  - coil–coil distance ≥ 1.2 m
  - convexity
  - length ≤ 36 m **per coil**. This is the paper's *average* encircling length used as a per-coil cap (the user's choice); every result below sits on it.
- **Finite beta.** The plasma's own B·n is about 2.4% of ⟨|B|⟩ (0.166 T mean). Everything is optimized and judged on (B_coils + B_plasma)·n, with the plasma field from virtual casing.
- **The question (the user and their advisor).** The paper's planar encircling + 324 shaping coils reach 0.23% average |B·n|/B_axis. Our planar encircling coils alone reach 4.04%.
  - How much of that gap can non-planar-but-simple coils close?
  - Above all: **what is the best B=2 (two planar arcs) coil on this equilibrium?**
- **"Gap closed"** below = (planar − x)/(planar − 0.23%), and in log terms ln(planar/x)/ln(planar/0.23%).

## Results (all at the same bounds, plasma field included)

⟨|B·n|⟩ is area-weighted over the whole boundary; B_axis = 6.056 T. "bn" is the sandbox metric ⟨|B·n|⟩/⟨|B|⟩.

| coilset | bn | ⟨\|B·n\|⟩/B_axis | max \|B·n\| | gap closed (lin / log) | file |
|---|---|---|---|---|---|
| paper: planar + 324 shaping coils | – | 0.23% | 0.11 T | – | – |
| arcB3 M=5 (transverse arcs) | 1.008e-2 | 1.04% | 0.24 T | 79% / 47% | `work/helios/arcB3_M5_final_it200.h5` |
| **polar arcB2 M=5, cut 30°** | **1.045e-2** | **1.11%** | 0.28 T | **77% / 45%** | `work/helios/polarB2_M5_cut30_final.h5` |
| polar arcB2 M=5, cut 60° | 1.071e-2 | 1.12% | 0.26 T | 77% / 45% | `scan_polar/cut60/legB/result.h5` |
| polar arcB2 M=5, cut 90° | 1.250e-2 | 1.27% | 0.34 T | 73% / 40% | `scan_polar/cut90/legC/result.h5` |
| polar arcB2 M=5, cut 0° | 1.266e-2 | 1.30% | 0.32 T | 72% / 39% | `scan_polar/cut0/legB/result.h5` |
| transverse arcB2 M=5 | 1.320e-2 | 1.42% | 0.36 T | 69% / 37% | `work/helios/arcB2_M5_stage2_w1/result.h5` |
| polar arcB2 M=5, cut 150° | 1.415e-2 | 1.47% | 0.35 T | 67% / 35% | `scan_polar/cut150/legB/result.h5` |
| polar arcB2 M=5, cut 120° | 1.449e-2 | 1.50% | 0.37 T | 67% / 35% | `scan_polar/cut120/legB/result.h5` |
| planar N=7 (FourierPlanar) | 4.023e-2 | 4.04% | 0.98 T | – | `work/helios/planarN7_final_it300.h5` |

`scan_polar/` means `work/helios/scan_polar/`.

- **Feasibility.** All are feasible against the paper bounds. Some are flagged "FAILS L" only because the dense check has no tolerance: lengths are micrometres over 36 m (arcB3 coil 0: 1.3 mm).
- **Geometry.** Best polar arcB2 (cut 30°): κ 0.35 of bound, d_cc 1.35, d_pc 1.33, convexity 0.03 of tolerance, currents 21.6/21.2/22.1 MA.
  - The price is geometric: hinge corners up to about 100° and folds of 25–83° between the two halves.
  - Transverse arcB2 corners are 90–96°. Corners are reported but not constrained, by the user's choice.
- **Polar scan stages.** bn at fit → repair → 50 iterations with hinges held → final:

| cut | fit | repair | hinges held | final |
|---|---|---|---|---|
| 0° | 0.040 | 0.045 | 0.041 | 0.0127 |
| 30° | 0.040 | 0.045 | 0.029 | 0.0104 |
| 60° | 0.040 | 0.044 | 0.039 | 0.0107 |
| 90° | 0.040 | 0.042 | 0.033 | 0.0125 (+150 iterations) |
| 120° | 0.041 | 0.042 | **0.027** | **0.0145** |
| 150° | 0.040 | 0.043 | 0.038 | 0.0141 |

- **Pictures.**
  - `scan_polar/summary.png`, `summary.json` (per-coil hinges, folds, corners, pairwise coilset distances)
  - `scan_polar/cut*/diag_final.png` (every stage, per coil: in-plane shape, out-of-plane fold, polar r(θ), degeneracy numbers)
  - `work/helios/best_arcB2_3d.html` (interactive 3D; other coilsets toggle in the legend) and `best_arcB2_3d.png`
  - `work/helios/compare_planar_arcB2_arcB3.png` (B·n maps)

## What we learned (why things are set up the way they are)

1. **Paper-form convexity stalls the solver.**
   - The paper's form is Σ|κ|ds = 2π (`CoilIntegratedCurvature`). Nearly straight coil legs put many grid nodes near κ = 0, and each is a kink in Σ|κ|. The trust radius collapsed to 2e-5.
   - Replacement: `sandbox/convexity.py: CoilSignedCurvature`, κ_s ≥ 0 at every node. The plane normal is Σ(x − p)×x_s ds per planar segment (the whole coil, or each arc), which is exact whatever the grid. It is smooth and verified: values to 1e-15 against an independent calculation, derivatives to 1e-10 against finite differences, on planar, transverse-arc and polar-arc coils.
   - This is the default (`run_one --convex-form signed`). The verdict still uses the paper's excess, ∫(|κ_s| − κ_s)ds ≤ 1e-3 rad per coil, which for a planar coil equals ∫|κ|ds − 2π.
   - The AL tolerance lets κ_s dip to about −2e-3 /m, which can exceed that excess. `--convex-kmin` > 0 is the lever if it matters.
2. **Transverse arcs** (`PiecewisePlanarArcCoil`; each arc is a graph *over* its chord) force B=2 hinges onto the coil's diameter.
   - Off-diameter fits miss by 1–3 m. The optimizer always ends with chord = coil span.
   - Both transverse arcB2 starts (weak-anchor repair: pointed ends, tilted; heavy-anchor repair: unfolded) converged to the **same** optimum, within 19 mm.
   - In that optimum all parameter groups move a lot (hinges 1.4–5 m, tilts up to 0.9 rad). Nothing is "frozen"; see `work/helios/arcB2_motion.{json,png}`.
3. **Polar arcs** (`PolarPlanarArcCoil`; r(θ) about the chord midpoint) accept any chord of a convex coil, which makes hinge placement a real choice.
   - **The vendored polar fitter had a sign bug.** When perp flips, the tilt was negated, putting arcs out of plane and making fits miss by metres. It is fixed in `vendor/desc/geometry/curve.py` and in `vendor/desc_changes_vs_master.patch` (regenerated from DESC master 33076c84; round-trip verified).
   - The user reports polar arcs have had other bugs and degeneracies, so always make the diagnostic pictures.
   - Checks passed so far: derivatives vs finite differences, length, 24-node GL Biot–Savart (4e-4 of |B|), per-arc convexity.
4. **Hinge placement matters, and the winning layout is set coil by coil.**
   - Cuts 30° and 60° (the two best) end with **coil 2's hinges at (R, Z) ≈ (9.45, 3.9) and (4.3, −2.2) m**. The four worse cuts end with coil 2 at ≈ (12.2, −3.4) and (5.0, 2.1).
   - Coil 0 ends almost the same in cuts 30°, 60° and 90° (hinges ≈ (6.4, 0.0), (13.4, 3.8)).
   - Coil 1 differs between cuts 30° and 60° but they score almost the same.
   - Cut 60° is only 0.64 m from the arcB3 coils.
5. **The held-hinge bn does not predict the final basin.** Cut 120° had the best held stage and the worst final; its plasma–coil distance became active. Cut 30° switched basin in its free stage (iterations 100–175, d_cc 1.05 → 1.33).
6. **Repairs.**
   - Fitted starts carry sine-series ripple, so a feasibility "repair" runs first:
     - `run_one --mode feasibility --anchor-weight 1 --margin 0.25 --margin-skip L`
     - The 25% margin is the user's rule, so the AL doesn't start on the edge.
     - Length is exempt because every coil sits on the cap.
   - With the default weak anchor (1e-3) a repair reshapes coils freely. With weight 1 it only removes ripple.
   - Repairs often end at a few mrad of concave turning, which stage 2 clears. The chain carries on regardless.
7. **Degeneracy watch.**
   - `arc_ref_margin` (1 = degenerate frame) peaked at 0.77 (cut 0°, coil 2). The chain warns above 0.9. Use `refreeze_arc_ref()` between stages if it gets close.
   - Thin polar halves: min r/(chord/2) 0.39–0.40 on coil 0 for cuts 0° and 150°. Cut 0° coil 0 also comes within 0.67 m of itself.
8. **Operations.**
   - Arc stage 2 takes about 7–10 s/iteration plus about 4 min compile on the RTX 5070.
   - Host memory grows over long runs (8 → 10.6 GB by iteration 550 of an 800-iteration run). The 12 GB cgroup cap stops the run cleanly, so avoid needlessly long runs.
   - One CUDA_ERROR_ILLEGAL_ADDRESS happened at setup; the chain retries once.
   - Runs usually converge by 100–250 free iterations.

## Hinge-placement scan and the clearance finding (2026-09-17)

The equal-angle scan varies one number per coil (the cut angle), so it samples a 1-D curve of the
2-D hinge space per coil, and always starts with the chord through the centroid. The finals never
stay there: they end at chord offsets 0.17–0.52 of the coil half-size with unequal halves
(shorter arc 0.20–0.39 of the length). So the start angle only selects a basin.

**Mixed-angle design.** `scan_polar/build_mixed.sh` builds 10 starts in which every coil PAIR sees
every nonzero cut-angle difference (mod 180°) exactly twice; the six equal-angle runs cover
difference 0. No exactly balanced design exists (Z6 has no complete mapping), so this is the closest
by a local search. `scan_polar/run_mixed.sh` is the queue; it skips labels that already have
`legB/result.h5`, so it resumes. **3 of 10 were run** (`0,120,60`, `0,150,120`, `30,90,0`); the
4th (`30,120,60`) was interrupted and its partial `repair/`, `legA/` remain (rerun redoes it).

**`scan_polar/classify.py`** groups each coil across all finals (symmetric max nearest-point
distance, 30 cm threshold) and prints the per-coil layout of every run. Layout names: coil 0
A (cut60) A0 (cut0) C (cut120) D (cut150); coil 1 X (cut30) Y (cut60); coil 2 P (cut60) Q (cut90).

| run | bn | d_cc/bound | c0 | c1 | c2 |
|---|---|---|---|---|---|
| 30 | **1.045e-2** | 1.35 | A | X | P |
| 30,90,30 | 1.071e-2 | 1.33 | A | Y | P |
| 60 | 1.071e-2 | 1.33 | A | Y | P |
| 90 | 1.250e-2 | 1.03 | A | Y | Q |
| 30,90,0 | 1.253e-2 | 1.03 | new | Y | Q |
| 0 | 1.266e-2 | 1.03 | A0 | new | Q |
| 0,150,120 | 1.289e-2 | 1.03 | A0 | new | Q |
| 150 | 1.415e-2 | 1.03 | D | new | Q |
| 120 | 1.449e-2 | 1.04 | C | new | new |
| 0,120,60 | 1.510e-2 | 1.04 | new | new | new |

New layouts do exist (the mixed runs found several), but **none beats P**, and the three mixed runs
cost ~90 min for no new best. Starting coil 1 at 90° lands it in Y whatever the other coils do
(`30,90,30` reproduced cut 60 to 0.2–0.6 cm and 3.5 kA).

**The winners are exactly the runs with slack clearance — but that turned out to be a symptom, not
a cause** (see next step 2(e): relaxing the bound in a losing basin buys nothing). Read the table
below as a marker of which basin a run fell into (`geom_compare.py` → `geom_compare.{json,png}`).
Pairwise minimum gaps, by coil group (u0u0 = coil 0 against its own mirror):

| run | bn | u0u0 | u0u1 | u0u2 | u1u1 | u1u2 | u2u2 |
|---|---|---|---|---|---|---|---|
| 30 / 60 / 30,90,30 | 1.045–1.071e-2 | 1.60–1.61 | 1.62–1.63 | 3.2–3.3 | 4.0–4.3 | **1.70–1.75** | 2.3 |
| 90, 30,90,0 | 1.25e-2 | 1.62 | 1.71 | 2.7 | 4.1 | **1.23** | 2.1 |
| 0, 0/150/120, 150, 120, 0/120/60 | 1.27–1.51e-2 | **1.24** | 1.24–2.07 | 1.3–3.2 | 2.2–4.0 | 1.24–1.46 | 2.1–3.1 |

- In the three winners **no pair is near the bound** (tightest 1.60 m, a third clear) and length is
  the only active constraint. In every loser at least one pair sits at 1.24 m, where the distance
  penalty settles against the ×1.025 backed-off bound.
- κ is never the limit (peak 0.26–0.50 of the 1.5 /m bound); d_pc binds only in cut 120 (1.21 m).
- The winning coil 2 (P) is ~20% more curved overall than Q (sector-mean κ: outboard .25/.22,
  top-out .19/.14, top-in .16/.09, inboard-low .32/.29, bottom .21/.14), reaches 0.5 m further
  outboard and stands 0.8 m lower. Slack clearance is spent on shaping.
- Coil 1's apparent κ-peak difference between winners and losers is two near-equal peaks swapping
  rank, not a shape change. Do not read `kappa_max` location without the sector profile.

**Why the optimizer does not repair a pinch locally** (the user's question: high modes are B·n-cheap
by exponential attenuation, so a local dent should buy clearance almost free).

1. *The loser's pinch is ON a hinge.* The polar arc shape is `r(φ) = |chord|/2 + Σ a_m sin(mπφ)`
   with hinges at φ = 0, 1, so **every mode vanishes at a hinge, at any M**. Measured: cut90 coil 2's
   closest-approach point is 0.00 m from its hinge, basis amplitude 0.00 for m = 1…5, and perturbing
   any of its five coefficients by 0.2 m changes the minimum gap by 0.000 m (while still costing bn
   and length). Winner coil 2 meets its neighbour mid-arc (5.3–5.7 m from a hinge, basis 0.3–1.0).
   Only hinge motion retracts a corner, and that redefines both arcs' chord, pole and baseline
   radius — a basin change, not a local repair. Raising M cannot help at a hinge.
2. *Where modes do act, they are cheap in bn but expensive in LENGTH.* On cut90's coil 1 (pinch
   3.1 m from a hinge), per +0.2 m of coefficient: a1 +0.068 m gap / +0.000 m length / Δbn 2.6e-4;
   a2 +0.109 / +0.024 / 2.8e-4; a3 +0.099 / +0.176 / 1.8e-4; a4 +0.046 / +0.230 / 1.0e-5;
   a5 +0.013 / +0.316 / 1.1e-4. Field-error cost per 0.1 m of clearance falls ~18× from a1 to a4
   (the attenuation argument is right), but clearance per metre of length falls from 4.5 (a2) to
   0.04 (a5). A dent of depth A over wavelength λ costs length ~π²A²/λ, so locality is quadratically
   expensive. Closing this pinch's 0.47 m deficit over the 0.4 m where the gap is < 1.3 m would cost
   ~1.5 m of length, 4% of the 36 m budget, taken from low-order shaping that does reduce bn.

3. *Translating the pinched hinge works geometrically but is not local.* Measured on cut90 coil 2,
   moving its pinched hinge directly away from the neighbour: +0.1 m → gap +0.100 m, ΔL +0.027 m,
   Δbn +2.8e-4; +0.2 m → +0.200 m, +0.058 m, +1.0e-3; +0.4 m → +0.233 m (saturating), +0.129 m,
   +3.7e-3; +0.8 m → the gap gets WORSE (−0.380 m) as the coil swings into another neighbour.
   Clearance comes ~1:1 and cheaply in length, but the arc is built ABOUT ITS CHORD (pole = chord
   midpoint, baseline radius |chord|/2, direction along the chord), so moving one hinge translates,
   rotates and rescales both adjacent arcs. Displacement profile along the coil for a 0.2 m move,
   ten bins from the fixed hinge: 0.04 0.13 0.21 **0.25** 0.22 0.17 0.10 0.07 0.06 0.03 — the peak
   is MID-ARC at 1.25x the hinge motion, not at the corner. That is a low-order deformation, which
   is what B·n sees.

**Structural conclusion: a B=2 polar arc coil has no local degree of freedom.** Cost of 0.1 m of
clearance: hinge translation Δbn 2.8e-4 / ΔL 0.027 m; mode a2 2.6e-4 / 0.022 m; mode a4
**2.2e-5 / 0.50 m**. Nothing is cheap in both. Hinges and tilts act affinely on an 18 m arc; the
sine modes span that arc and vanish at its ends.

**Counterexample worth keeping in mind: arcB3 is PINNED at 1.24 m and is still the best coilset
(1.008e-2).** Its pinches sit 1.3–2.4 m from a hinge on 12 m arcs, versus 18 m arcs for B=2. So the
lesson is not "keep clearance" but **a constraint is expensive when the freedom acts on a much
larger scale than the constraint does**. That argues for more arcs rather than more modes, and
polar B=3 has never had its hinges scanned.

## Tools added this session (all in `coil_sandbox/`)

- **`run_one.py`**
  - `--vacuum` (the default now includes the plasma field; virtual-casing source grid 2× by default, `--vc-m/--vc-n/--vc-chunk`)
  - `--convex-form signed|total`, `--convex-kmin`, `--convex-backoff`
  - `--anchor-weight`, `--margin-skip`, `--fix-hinges`
  - a bound set to null in the bounds file drops its constraint
- **`make_start.py`**
  - `--from-coils FILE` fits arcs to an existing coilset.
  - `--rep polararc`.
  - `--cut-angle A` or `A0,A1,A2`: polar B=2 hinges on the centroid line at A° from in-plane vertical, one value per coil allowed.
  - `--hinges axis` and `--phase` for transverse arcs; `--chord-angle` exists but only works near the long axis.
- **`sandbox/common.py`**
  - The dense check includes the plasma field (`bn_local`, `bn_max_T` too) and reports convexity as `convex` and `kappa_s_min`.
  - `paper_bounds` handles null bounds and `convex_excess`.
  - Hinge helpers: `resample_between`, `long_axis_hinges`, `arclength_hinges`, `direction_hinges`, `centroid_line_hinges`.
- **`sandbox/convexity.py`** — `CoilSignedCurvature`, `convexity_excess`.
- **`work/helios/scan_polar/`**
  - `run_chain.sh LABEL`: fit at `cutLABEL/fit.h5` → repair → 50 iterations with hinges held (then asserts they didn't move) → free stage (`FREE_IT`, default 250). `run_chain.sh LABEL continue N` continues the free stage.
  - `run_scan.sh` (the queue), `polar_diag.py` (degeneracy pictures), `scan_summary.py`
    (only reads `cut0…cut150`; needs a small change to pick up labelled folders).
  - LABEL is only a directory suffix, so any string works (for example `c2x30_c0x45_c1x60`).
  - **2026-09-17:** `build_mixed.sh` (the mixed-angle design, builds fits + `fit_diag.png`),
    `run_mixed.sh` (resumable queue), `classify.py` (per-coil layout grouping across all finals).
- **`work/helios/plot_coils_3d.py`** (interactive) and `plot_coils_3d_static.py`.
- **`work/helios/geom_compare.py`** — winner/loser geometry: position, hinges, fold and tilt, where
  κ peaks, and where each coil comes closest to the plasma and to its neighbours (poloidal angle
  about the plasma centre: 0 outboard midplane, 90 top, 180 inboard). Writes `geom_compare.{json,png}`;
  `why_P_wins.png` is the clearance-profile picture for coil 2.
- **`work/helios/clearance_test.sh`** with `bounds/helios_dcc1.0.json` and `bounds/helios_dcc1.5.json`
  — the shadow-price runs (below).

Run everything with `/home/singh/miniforge3/envs/desc-env2/bin/python`, one GPU job at a time, under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` (`run_chain.sh` does this).

## The subdominant coil is ROBUST (2026-09-18, answering the user's question)

Every 16-coil arcB2 run inherits the planar optimum's currents EXACTLY (make_start copies
`c.current`; fit and heavy-anchor repair never change them), so "one coil ends subdominant" could
have been an artefact of that lineage, or of leg A. It is neither.

| run | leg A? | final currents (MA) | min/max | short coil L | bn |
|---|---|---|---|---|---|
| cut 30 | yes | **5.33** 21.16 19.56 18.91 | 0.25 | 25.19 m | 8.7195e-3 |
| 0,0,120,30 | yes | **5.51** 21.66 19.03 18.76 | 0.25 | 25.60 m | 9.7813e-3 |
| **nolegA (from cut30 repair)** | **NO** | **6.43** 21.05 19.25 18.23 | 0.31 | 26.83 m | 8.7897e-3 |
| 60,150,60,60 | yes | 17.64 14.91 11.45 20.96 | 0.55 | - | 9.2937e-3 (unconverged) |

**Three independent paths - different hinge layouts, one with hinges NEVER frozen - converge to the
same attractor**: coil 0 subdominant at 5.3-6.4 MA and shortened to 25-27 m, the other three pinned
on the 36 m cap at 18-22 MA. The three that reach it are the three best scores. The one run that
avoided it (60,150,60,60) scored worse AND had not converged.

The current SUM is constrained (64.96 MA at every stage of every run), so current can only be moved
between coils; concentrating it on three necessarily starves the fourth.

### CONFIRMED BY A COLD START: five routes, five subdominant coils

`work/helios/arc16/run_cold.sh` breaks the lineage completely: polar arcB2 fitted to CIRCLES
(r/a 2.5), four EQUAL currents (16.24 MA each), fit deviation 3.5 mm, convexity excess exactly 0,
feasibility repair at a 10% margin (reached `target FEASIBLE` at iteration 100 - the only repair all
session that terminated on its callback instead of maxiter), then 300 free iterations, no leg A.

It splits the currents anyway, by iteration 50, and holds the split:

| cold stage | c0 | c1 | c2 | c3 | min/max |
|---|---|---|---|---|---|
| start (by construction) | 16.24 | 16.24 | 16.24 | 16.24 | 1.00 |
| it 50 | 15.40 | **9.73** | 17.34 | 22.49 | 0.43 |
| it 150 | 15.01 | 10.08 | 18.15 | 21.73 | 0.46 |
| final (it 300) | 15.22 | 10.99 | 17.68 | 21.07 | 0.52 |

All five 16-coil arcB2 runs:

| run | legA? | lineage | bn | currents (MA) | min/max | lengths (m) |
|---|---|---|---|---|---|---|
| cut 30 | yes | planar | **8.7195e-3** | 5.33 21.16 19.56 18.91 | 0.25 | 25.19 36 36 36 |
| nolegA | NO | planar | 8.7897e-3 | 6.43 21.05 19.25 18.23 | 0.31 | 26.83 36 36 36 |
| 60,150,60,60 | yes | planar | 9.2937e-3 | 17.64 14.91 11.45 20.96 | 0.55 | 36 36 33.19 36 |
| 0,0,120,30 | yes | planar | 9.7813e-3 | 5.51 21.66 19.03 18.76 | 0.25 | 25.60 36 36 36 |
| **cold** | **NO** | **circles** | 1.0804e-2 | 15.22 10.99 17.68 21.07 | 0.52 | 34.58 34.24 36 36 |

**Neither the planar lineage nor leg A causes it** - removing each in turn changes the SCORE but
never removes the effect. It follows from the constrained current SUM (64.96 MA in every run at
every stage): with 16 coils at this packing B=2 cannot find field-error-reducing work for a fourth
independent coil, so the optimizer moves its current to neighbours that can use it.

**Deeper starvation correlates with a BETTER score.** The three runs that drove a coil to ~5-6 MA
AND shortened it to 25-27 m scored 8.72 / 8.79 / 9.78e-3; the two that only reached min/max ~0.52
with all coils near full length scored 9.29e-3 and 1.08e-2. Sacrificing a coil is how B=2 wins here,
not a pathology to fix. WHICH coil is sacrificed is basin-dependent (coil 0 in three runs, coil 1 or
2 in the others); THAT one is sacrificed is not.

**The planar warm start is worth ~24% of final bn** (8.72e-3 vs the cold 1.08e-2) and the cold run
converged to a genuinely worse basin rather than merely lagging - it was flat from iteration 200.
So the warm start earns its place as a seed while being innocent of the structural effect.

### Constraint sensitivity: space is REAL but SECOND-ORDER (2026-09-18)

Warm restarts from cut30 legB (8.7195e-3), one bound changed, 300 iterations each,
`work/helios/arc16/run_sensitivity.sh` -> `sens_dpc10/`, `sens_dcc10/`, `sens_L30/`.

| test | bn | vs base | what the optimizer did with the room |
|---|---|---|---|
| baseline | 8.7195e-3 | - | d_cc 1.243 m, d_pc 1.221 m (both on their enforced bounds) |
| d_pc 1.2 -> 1.0 | 8.7432e-3 | **+0.3%** | took only 7 cm of 20 (1.221 -> 1.147), gained NOTHING |
| d_cc 1.2 -> 1.0 | **8.4427e-3** | **-3.2%** | took ALL 20 cm (1.243 -> 1.039) and gave plasma clearance BACK (d_pc 1.221 -> 1.357) |
| L 36 -> 30 | 1.1829e-2 | +35.7% | - |

1. **Plasma-coil clearance is NOT the limitation.** Offered 20 cm toward the plasma the coils
   declined it and ended marginally worse - the same signature as the 12-coil d_cc shadow-price
   test. Do not spend runs on d_pc.
2. **Coil-coil clearance IS real but weak.** The coils took every centimetre and spent it moving
   AWAY from the plasma: what they want is room from EACH OTHER (the "clamshell" crowding), not
   proximity to the plasma. But 3.2% for a 17% bound relaxation cannot touch a 2.30x gap.
3. **Length: effective elasticity at 16 coils is -1.67, vs -2.10 measured at 12.** The -2.1 law
   predicts 1.2787e-2 at L=30; actual 1.1829e-2, so freed space recovered **7.5%** of the loss.
   Crowding costs something at 16 coils that it did not at 12 - but staying short is still a large
   net loss, so "keep the coils in their own area" is not a win.
4. **The subdominant coil tracks available space**: its current is 5.3 MA at baseline, 7.5 MA with
   d_cc relaxed, 12.7 MA at L=30 (min/max 0.25 -> 0.39 -> 0.70). The starving IS a crowding
   response and relieving crowding partially reverses it - but the reversal comes with WORSE field
   error, reinforcing that sacrificing a coil is how B=2 copes, not a defect to fix.

**Conclusion: the constraint set is not what holds arcB2 back.** The three levers move bn by -3.2%,
+0.3% and +35.7% while the gap to unrestricted shapes is 2.30x. The binding limitation is the B=2
REPRESENTATION. kappa corroborates this: arcB2 uses only 0.32-0.50 of its curvature bound in every
run while FourierXYZ at 16 coils uses 1.00 - the restricted coils are not even spending the shape
freedom they already have. **-> arcB3 at 16 coils is the next substantive test.**

### CORNERS ARE NEARLY FREE; RAISING M IS THE BIG LEVER (2026-09-21)

Two things the recipe had never done: bound the hinge corner angle, and use more than M=5 modes.

**1. Raising modes per arc M=5 -> 10 buys -11.3%, feasible: bn 7.7549e-3 -> 6.8770e-3.**
`work/helios/arc16/M10_L36/`. Start built by `raise_M.py`, which zero-pads the sine coefficients
(r(phi) = |chord|/2 + sum a_m sin(m pi phi), so higher modes with zero amplitude leave the curve
POINTWISE identical -- verified 0.000 um and bn identical to 7 digits before writing). NOTE the arc
frame depends on `arc_ref`, which is a PARAMETER not a constructor argument; rebuilding a coil
without copying it moves the curve by METRES.
   - kappa goes 0.45 -> 0.89 of bound: at M=5 the coils were never curvature-limited because M=5
     could not express the shapes that would use the curvature.
   - COSTS: corners open 117 -> **144.5 deg**, currents spread min/max 0.93 -> **0.73**, d_pc goes
     active. Not a free lunch.
   - M=10 -> 15 was -1.6% by iteration 50 (vs -11.3% for 5->10) and was killed; M ~ 10 is the
     practical plateau, consistent with the coefficient decay (a_9, a_10 ~ 0.005-0.08 m vs a_1 ~ 1-4 m).

**2. The extra modes CANNOT act where the coils are closest to the plasma, because that point is a
HINGE.** `mode_use.py` on the M=10 optimum: three of four coils have their closest approach at
phi = 0.000 exactly, and every mode is sin(m pi phi), which vanishes at phi = 0 and 1. New-mode
displacement there is **+0.0 mm at any M**. The one coil whose closest approach is mid-arc
(phi 0.805, +65 mm of new-mode displacement) sits at 1.95 m, nowhere near the bound. So the m>5
modes (4.6-13.6% of total |a|) buy their 11.3% through general mid-arc shaping, NOT by relieving
the plasma pinch. Same structural trap as the 12-coil coil-coil pinch finding. **Raising M cannot
fix a pinch at a corner; only more hinges (B=3) can.**

**3. Corners are nearly free.** New `sandbox/corners.py: CoilCornerAngle` (cos of the tangent turn
angle at EVERY hinge, bounded below by cos(theta_max); per hinge, not a max over hinges) wired in as
`run_one --corner-max DEG`. Validated: values match `common.curve_metrics` to 1e-4 deg, Jacobian
matches finite differences to 1.9e-6. **Trap:** the framework allocates one residual per GRID NODE
(2B per coil) while a turn angle is one number per HINGE (B per coil) -- emit each value at both of
its nodes or the build dies in `_scale` with a broadcasting error. Physics validation did not catch
this; only a real run did.

Dual length+corner ladder (`run_dual_ladder.sh`), tight->loose from the cold L=28 coilset:

| L | corner cap | bn | vs UNCONSTRAINED ladder | currents min/max | max corner |
|---|---|---|---|---|---|
| 28 | 60 | 1.2804e-2 | 1.2351e-2, **+3.7%** | 0.88 | 60.0 |
| 30 | 70 | 9.9952e-3 | - | 0.90 | 70.0 |
| 32 | 80 | 8.6201e-3 | 8.6095e-3, **+0.1%** | 0.93 | 80.0 |
| 34 | 90 | 8.0020e-3 | 7.9766e-3, **+0.3%** | 0.95 | 90.0 (convex 1.29, marginally infeasible) |
| **36** | **100** | **7.7913e-3** | 7.7549e-3, **+0.5%** | **0.95** | 100.0 |

The cap saturates exactly at every rung, so it genuinely binds - the coils WANT sharper corners and
giving them up costs almost nothing. **You can have the length-ladder result AND buildable corners.**
Currents also got MORE even along the ladder (0.88 -> 0.95, better than the unconstrained 0.93), so
constraining corners does not reintroduce the sacrifice. It is not really a new basin: rung 5's
corners are the unconstrained ones with the two worst (114.8, 117.1) clipped to 100, the rest barely
moved - consistent with the tiny cost.

**Open, one run:** does M=10 keep its 11.3% under a corner cap? Its 144.5 deg corners are far outside
anything the ladder needed.

### THE CONVEXITY PRICE IS 17.2% - and the FourierXYZ reference never paid it (2026-09-21)

The user's hypothesis: the FourierXYZ "unrestricted ceiling" coilsets are strongly non-convex, so
part of the arcB2-vs-FourierXYZ gap may be the CONVEXITY constraint rather than shape freedom.
**Confirmed, and it is the largest single lever found this session.**

`xyz16` ran with `bounds/helios_kruger2026_noconvex.json`, `convex = None` - convexity is undefined
for a non-planar coil, so it was dropped. Every arcB2 run carries it. The comparison was never
like-for-like.

| | bn | ratio to FourierXYZ | active constraints |
|---|---|---|---|
| arcB2, convexity ON (paper bounds) | 7.7549e-3 | 2.04x | L only |
| **arcB2, convexity OFF** | **6.4227e-3** | **1.69x** | L, **kappa**, **d_pc** |
| FourierXYZ 16 (never had convexity) | 3.7991e-3 | 1.00 | L, kappa |

**-17.2%** from dropping convexity alone (`work/helios/arc16/noconvex_L36/`), warm-started from the
L=36 ladder optimum. So ~1/3 of what we had been calling "the cost of two planar arcs" is the convex
floor. The remaining **1.69x** is the genuine representation gap.

**Why the coils were never curvature-limited:** with convexity on, arcB2 sits at 0.32-0.50 of the
kappa bound in EVERY run. Drop convexity and kappa goes to 1.00 within 100 iterations, and d_pc
becomes active too. Convexity was masking both - the coils were held flat before they could reach
them. (Diagnostic: `work/helios/arc16/convex_pressure.py`. convexity_excess ~ 0 does NOT mean the
constraint is inactive; it is enforced POINTWISE as kappa_s >= 0, so a coil that wants to be concave
sits pinned at kappa_s ~ 0 over a stretch while the integral excess still reads 0. Measured: up to
19% of a coil's arclength within 5% of flat in the sacrificed baseline, 2.7-6.6% in the uncrowded
short-coil set - pressure tracks crowding.)

**"Relax, optimize, re-impose" does NOT work here.** Re-imposing convexity from the non-convex
optimum (`reconvex_L36/`) lands at **8.1019e-3, 4.5% WORSE than the 7.7549e-3 ladder result**. The
start was 3194x bound = **3.19 rad** of concave turning (half a turn - a genuinely different shape
class); the repair restored convexity cleanly by iteration 50 (no n_outer stall, n_outer 16) but
then drifted UPWARD from 8.067e-3 to 8.102e-3 over the last 100 iterations while the AL kept
tightening convexity. **The non-convex optimum has no good convex neighbour.** So convexity is not
trapping the ladder in a poor basin - it is a real price, and the ladder finds about the best convex
configuration available. Do not try this recipe on the 12-coil results expecting a gain.

### THE LENGTH LADDER: grow from short coils, +11% field error for free (2026-09-21)

**Best 16-coil arcB2 is now bn 7.7549e-3** (`work/helios/arc16/warm_L36_paper/result.h5`), feasible
on the FULL paper bounds with all four coils healthy. The like-for-like comparison is the result:

| | bn | conductor | currents min/max |
|---|---|---|---|
| baseline cut30 (start long, sacrifice a coil) | 8.7195e-3 | 576 m | 0.25 |
| **ladder @ 36 (grow from short, all healthy)** | **7.7549e-3** | 576 m | **0.93** |

**Same bounds, same conductor, 11.1% better field error, purely from the PATH taken to the optimum.**

The ladder, each rung a warm start from the one below, paper bounds except the length cap:

| cap | bn | conductor | currents min/max | d_cc | paper-feasible |
|---|---|---|---|---|---|
| 28 (cold from circles, d_cc/d_pc relaxed to 1.0) | 1.2351e-2 | 448 m | 0.90 | 1.412 m (SLACK) | n/a |
| 32 | 8.6095e-3 | 512 m | 0.92 | 1.241 m | yes |
| 34 | 7.9766e-3 | 544 m | 0.95 | 1.242 m | yes |
| **36** | **7.7549e-3** | 576 m | 0.93 | 1.243 m | **yes** |

Every rung improved and the even-current state survived all of them. At 32 m three coils were pinned
at the clearance bound with coil 1 squeezed between coils 0 and 2 (the user spotted this in the 3D
viewer) - but it resolved rather than tipping into a sacrifice: currents got MORE even, 0.92 -> 0.95,
over the next rung.

Recipe, for reuse: `run_cold.sh` with `bounds/helios_cold_L28_d10.json` (L 28, d_cc/d_pc 1.0 - note
circles at r/a 2.5 are already feasible on these, so the repair converges in 25 iterations), then
warm restarts through `helios_L32_d10` -> `helios_L32` -> `helios_L34` -> `helios_L36_ladder`,
400 iterations each, no leg A. About 25 min per rung.

**Buildability caveat:** hinge corners reach 117 deg at L=36 (vs ~100 deg in the 12-coil results).
Corners are reported but not constrained (user's choice), so this coilset buys field error partly
with sharper corners. Relevant if next-step 3 (a corner limit) is ever imposed.

### BEST 16-COIL arcB2: grow into the length from SHORT coils (2026-09-18, the user's idea)

**bn 8.6095e-3, FEASIBLE against the FULL paper bounds, with all four coils healthy and 11% LESS
conductor than the previous best.** `work/helios/arc16/warm_L32_paper/result.h5`

| | previous best (cut30) | **this** |
|---|---|---|
| bn | 8.7195e-3 | **8.6095e-3** (-1.2%) |
| length cap | 36 m | **32 m** |
| conductor | 576 m | **512 m** |
| currents (MA) | 5.3 21.2 19.6 18.9 (min/max 0.25) | **15.4 16.7 16.4 16.4 (min/max 0.92)** |
| d_cc / d_pc | 1.243 / 1.221 m | 1.241 / 1.271 m |
| kappa | 0.47 of bound | 0.41 of bound |

**The subdominant coil is PATH-DEPENDENT, not required by the bounds.** Every run that started from
long crowded coils drove one coil to 5-6 MA. This recipe never does:

1. **Cold start, SHORT coils, relaxed clearances** (`run_cold.sh`, `bounds/helios_cold_L28_d10.json`:
   L 28, d_cc 1.0, d_pc 1.0). Circles at r/a 2.5 are ALREADY feasible on all of these, so the repair
   converges in 25 iterations. Result: bn 1.2351e-2, currents min/max 0.90, and the coils keep
   clearances **voluntarily slack** (d_cc 1.412 m, d_pc 1.348 m against 1.0 m bounds) - the first run
   all session where no clearance binds. Short coils simply do not crowd.
2. **Raise L to 32, warm start** (`bounds/helios_L32_d10.json`). All four grow in lockstep to the new
   cap and stay even (min/max 0.98 at it 50, 0.91 final). bn 8.5579e-3 - but they re-crowd to the
   relaxed 1.036 m, so this is NOT paper-feasible.
3. **Restore d_cc/d_pc to 1.2, warm start** (`bounds/helios_L32.json`). d_cc climbs 1.036 -> 1.241 m,
   FEASIBLE by iteration 50, and the currents STAY even. Converged (flat from it 250, n_outer 111).

So enforcing the paper's 1.2 m clearance does NOT force the sacrifice - starting long and crowded
does. The earlier finding that "deeper starvation correlates with better bn" was an artefact of all
those runs sharing the same crowded lineage.

**Open**: L 32 -> 34 -> 36 by the same walk-up has not been tried. If the even state survives to 36 m
it should beat 8.61e-3 again, on the same conductor as the old baseline. That is the obvious next run.

### Leg A is approximately NEUTRAL, not harmful (the hypothesis is answered)

`work/helios/arc16/run_nolegA.sh`: cut30's repair -> free stage directly, 300 iterations = the same
stage-2 budget cut30 got (50 held + 250 free), so the ONLY variable is whether hinges were frozen.
Result **8.7897e-3 vs cut30's 8.7195e-3** - a 0.8% tie, and the no-legA run was still falling 7% per
25 iterations at the cap, so it would probably go lower with more. Comparison on equal stage-2
iterations (no-legA leads early, cut30 leads in the middle, they converge):

| stage-2 it | 25 | 50 | 75 | 100 | 125 | 300 |
|---|---|---|---|---|---|---|
| cut 30 | 2.28e-2 | 2.31e-2 | 1.27e-2 | 1.06e-2 | 9.48e-3 | 8.7195e-3 |
| no-legA | 1.80e-2 | 1.26e-2 | 1.17e-2 | 1.12e-2 | 1.07e-2 | 8.7897e-3 |

So legA does not create the subdominant coil and does not cost the endpoint. Important corollary:
**the 12-coil results in the table are NOT compromised** by having gone through the same held stage.

### The 250-300 iteration budget is too small at 16 coils

3 of 4 runs hit their cap while still descending (run 1 -2.7%/25it, run 2 -1.0%, no-legA **-7%**).
Finals therefore UNDERSTATE their configurations. Budget 400-500 for 16-coil runs, or continue.

### The repair leaves a large convexity violation, contrary to this handoff's own note

It says repairs end "at a few mrad of concave turning". The 16-coil arc repairs end at **1.3 rad**
(cut30 per-coil excess 0.450/1.302/0.751/0.132 rad, kappa_s down to -0.36 /m), three orders of
magnitude more, because the AL never advances its outer loop (`n_outer 1` in 120 iterations) while
chasing L, convex, d_cc and d_pc at once. Planar-16's repair was the same (0.709 rad, n_outer 1).
Stage 2 does clear it every time (cut30 -> 0.199 of bound), but it is handed a much worse start than
intended, which is part of why the iteration budget is tight.

## Next steps (agreed with the user)

1. **Hinge placement — mostly answered, and the remaining moves are different from the old plan.**
   - The mixed-angle scan (above) found new layouts but nothing better than P, and the equal-angle
     scan's own results say the start angle only picks a basin. Finishing the remaining 7 mixed runs
     (`run_mixed.sh`, ~3.5 h, resumable) is a completeness check, not a promising search.
   - **The untested region is the second hinge degree of freedom**: all starts put the chord through
     the centroid, while every final sits at an offset of 0.17–0.52 of the coil half-size. A
     `--cut-offset o0,o1,o2` in `make_start.py` (a small change to `centroid_line_hinges`) would let
     (ψ, offset) be scanned, i.e. start where the optima actually live.
   - Given the clearance finding, the targeted move is to place hinges so that each coil meets its
     neighbours **mid-arc, not at a corner** (P does, Q does not), and to seed from the cut-30
     optimum replacing one coil at a time (still needs a small "mix coilsets" helper).
   - Make `polar_diag.py` pictures for every start. The user wants to see them because of past polar-arc degeneracies.
2. **Separate physics limits from our own constraints** (from the session's closing discussion). Each run starts from an existing result, so each costs one stage-2 run (~30–50 min). Judge every one against the same paper bounds, except where a bound is deliberately changed.
   - **(e) Shadow price of the coil–coil bound — DONE 2026-09-17, `work/helios/clearance/relax90/`.
     The bound is NOT what limits the losers.** cut90 restarted with d_cc = 1.0 m (everything else
     identical) went from bn 1.2502e-2 / gap 1.235 m to **bn 1.2504e-2 / gap 1.199 m**: given 20 cm
     of room it took 3.6 cm and gained nothing, stalling feasible after 95 iterations with only
     length active. So the clearance correlation in the table above is a SYMPTOM of the basin, not
     a cause, and "keep the coils clear" is not a design rule. Treat the whole winner/loser split as
     basin quality.
   - `tight60` (cut60 at d_cc = 1.5 m) was ill-posed and abandoned: a minimum-distance bound only
     pushes coils apart, and the winner's tightest pair is already 1.60 m, so the constraint is
     slack (verified: bn unchanged at 1.0712e-2, gap 1.596 m, through 100 iterations). To probe the
     winner one must force MORE clearance (d_cc ≈ 1.8 m), not less.
   - A companion test if the representation is doubted: rerun a loser with M = 8–10. It should
     change little, because the pinch is at a hinge where no mode has amplitude.
   - **(f) Price of length — DONE 2026-09-17, `work/helios/clearance/len37/`. Length is by far the
     most valuable resource, and it cannot close the gap.** cut30 restarted with
     `bounds/helios_L37.json` (L = 37 m, everything else identical): bn **1.0450e-2 → 9.8547e-3**
     (−5.7%), FEASIBLE, all three coils on the new cap, max |B·n| 0.278 → 0.272 T. Converged by
     iteration 25 of 250. κ rose to 0.536 /m (0.36 of bound) and the tightest coil gap opened from
     1.61 to 1.88 m — with more length the coils buy clearance for free, another sign that d_cc was
     never the binding resource.
     - Elasticity d(ln bn)/d(ln L) = **−2.1**: bn falls about as 1/L².
     - Extrapolating that (naively, far outside the measured range) to the paper's 0.23% needs coils
       ~2.1× longer, about 75 m each. **Longer encircling coils cannot close the 4.8× gap**; the
       paper's result comes from its 324 shaping coils, not from conductor length.
     - This also settles the local-fix economics above: a metre of length is worth ~6e-4 of bn, so
       the a4-style route (0.50 m of length per 0.1 m of clearance, Δbn 2.2e-5) is a bad trade by
       more than an order of magnitude, and the hinge translation (0.027 m of length, Δbn 2.8e-4)
       is roughly break-even. Nothing cheap is available; that is why the losers stay put.
   - **(a) Ceiling with unrestricted shapes.**
     - Convert the planar result to FourierXYZ coils (for example N = 7–10; `CoilSet.to_FourierXYZ`) and run stage 2 with the same κ, distance and length bounds. Drop convexity, which is undefined for non-planar coils: set `"convex_excess": null` in a copy of the bounds file.
     - `make_start.py --rep xyz` currently builds only from circles; add `--from-coils` support for xyz (a few lines, reusing the conversion).
     - This measures what 12 coils at ≥ 1.2 m standoff can do at all, and therefore how much of the gap from arcB2 (1.11%) to the paper (0.23%) is the cost of restricted shapes.
     - Calibration: in the Eos paper, modular coils with the same count reached 0.32%, close to planar + shaping. Arrange the runs so the ceiling isn't limited by resolution.
   - **(b) Best arcB2 with convexity off.**
     - Run stage 2 from `work/helios/polarB2_M5_cut30_final.h5` with a bounds file that has `"convex_excess": null` (`run_one` then drops the constraint).
     - Question: are the simple shapes set by convexity (wiggles appear and bn drops) or by the ~1.2–1.9 m standoff filtering out short-wavelength coil detail (little change)?
     - Evidence so far: κ is at 0.3–0.5 of its bound, while length and per-arc convexity are the active constraints.
   - **(c) Best arcB2 with a TOTAL length budget** (the user's specification).
     - Replace the per-coil cap (each unique coil ≤ 36 m) with Σ over the 3 unique coils ≤ 3 × 36 = **108 m per half period** (the previous average times the number of coils).
     - Needs building:
       - A total-length constraint: a small sandbox objective summing `CoilLength` over the unique coils, or a `CoilLength` subclass returning the sum.
       - A bounds-file key, for example `"L_total": 108` (with `"L": null`), handled in `paper_bounds`, `margins` and `failed`.
       - A `run_one` switch.
     - Keep the same backoffs otherwise. Every coil in every run so far sits on the per-coil cap, so this is the most likely constraint to be costing field error. For a fair comparison, run it for the planar baseline too.
     - **Now confirmed worth doing by (f):** length is the only resource with a large shadow price
       (−5.7% of bn per +1 m per coil). A total budget lets the optimizer move length between coils,
       which the per-coil cap forbids.
   - **(d) Where the residual sits.** Cheap, CPU only, no new optimization.
     - For the planar and best-arcB2 results: the toroidal/poloidal mode spectrum of the residual B·n (coils + plasma), and the fraction of the surface above given |B·n| thresholds.
     - Coil ripple from only 12 coils would show at toroidal mode 6 per period.
     - This tells whether arcB2 could get by with *fewer* shaping coils (concentrated residual) or only *weaker* ones (a thin spread-out residual).
     - The direct test is to rerun the paper's shaping-coil step (fixed identical circular coils of 133.7 cm diameter on an offset winding surface, currents only, |I| ≤ 4.4 MA, pruning small currents) on top of each encircling set and count the surviving coils. That is a larger build.
3. **Test a hinge-corner limit** on the best result. There is no corner constraint yet; it would be a smooth bound on the tangent angle at each hinge. Today's corners are about 100°, and a buildable limit will cost field error.
4. **Scan polar B=3 hinges — now the most promising direction, not an optional extra.** B3 already
   leads (1.008e-2), its arcs are 12 m instead of 18 m, and the scale argument above says that is
   where the headroom is: more corners give freedom closer to the scale at which the distance
   constraints act. Raising M instead is the weaker lever (modes 4–5 had little influence on
   transverse arcB2, and they buy clearance only at a large cost in length).
5. **Not done:** the paper's own coil data (available on request from Thea) was never available, so there is no direct comparison with their encircling coils. Also note that their coils were optimized for current × length and the X-point objective as well, which costs field error, and their "0.23%" is loosely defined. Treat the 4.8× gap as an estimate.

**Suggested order after the coil-2-centred scan:** 2(d) (free) → 2(b) → 2(c) → 2(a) (needs the most new code).

Session memory: `~/.claude/projects/-home-singh-Documents-DESC2/memory/helios-coil-trials.md` has the full chronology.

---

# Session 2026-09-17/18: splines, the unrestricted ceiling, and coil count

## What we established

| coilset | coils | conductor | bn | ⟨\|B·n\|⟩/B_axis | gap closed |
|---|---|---|---|---|---|
| planar N=7 | 12 | 432 m | 4.0223e-2 | 4.04% | – |
| transverse arcB2 M=5 | 12 | 432 m | 1.3201e-2 | 1.42% | 69% |
| polar arcB2 M=5 cut30 | 12 | 432 m | 1.0450e-2 | 1.11% | 77% |
| arcB3 M=5 | 12 | 432 m | 1.0086e-2 | 1.04% | 79% |
| spline, 2 C0 breaks | 12 | 432 m | 1.0052e-2 | 1.04% | 79% |
| **FourierXYZ N=10 (ceiling)** | 12 | 432 m | **8.4220e-3** | **0.897%** | **82%** |
| **FourierXYZ, 16 coils** | 16 | 576 m | **3.7991e-3** | **0.405%** | **95%** |
| paper: planar + 324 shaping | 12+324 | – | – | 0.23% | 100% |

Three results, in order of importance.

1. **Coil count dominates coil shape by ~an order of magnitude.** Every shape lever moved bn by
   16-24%; 12 -> 16 coils moved it by 120%. Decomposed against the measured elasticity
   d(ln bn)/d(ln L) = -2.1, the +33% conductor alone predicts 1.83x of the observed 2.22x, so count
   contributes ~21% beyond conductor. **The controlled test (16 coils at a 27 m cap = same 432 m
   total) has NOT been run and is the cleanest way to separate the two.**
2. **The piecewise-planar restriction is cheap.** Two planar arcs cost 1.24x the unrestricted
   ceiling's field error, three cost 1.20x - about 11% of equivalent conductor. This is the
   quantitative answer to "what does restricting shapes cost" and it is small.
3. **Splines are a refinement basis, not a cold-start basis.** Warm-started from the cut30 arc
   optimum a `SplineXYZCoil` with C0 breaks gives -3.75% (1.044989e-2 -> 1.005757e-2), kappa
   converged at every resolution. Cold from circles it fails every time: kappa = |x_s x x_ss|/|x_s|^3,
   so the optimizer slows the parameterization BETWEEN constraint nodes and buys field error with
   curvature the constraint cannot see (measured: kappa 0.90 on the 301-node grid, 3.45 on a fine
   one). Fixes tried and found inadequate: finer grids (do not fit 8 GB), `CoilArclengthVariance`
   (global, misses a narrow dip), a pointwise |x_s| floor (itself aliased). See the memory file.

## Code changes (all additive; arc and Fourier runs are bit-identical)

- **`vendor/desc/compute/_curve.py: _splinexyz_helper` - `break_indices` was first-order broken and
  is fixed.** Interior breaks were O(h) with kappa ringing; now O(h^4). Required a STATIC
  `SplineXYZCurve._break_idx` passed as the `spline_breaks` compute kwarg - `transforms["intervals"]`
  arrives traced because an objective passes `_constants` as a jit argument. Patch regenerated and
  round-trip verified.
- **`sandbox/planarity.py`** - `CoilSetPlanarity`. Spline planarity is an EXACT algebraic condition
  on the knots (a segment is planar iff its knots are coplanar; measured to 1e-14 m), so no torsion
  objective is needed - and `CoilTorsion`'s denominator vanishes on exactly the straight legs these
  coils have.
- **`917_results/planarizability.py`** - the eps_B piecewise-planarity defect, exact by dynamic
  programming. Validated: it recovers the true plane count of five coilsets it was not told about.
  Writeup published as an artifact; source `917_results/epsilon_metric.html`.
- **New `run_one` flags:** `--kappa-grid-n`, `--coil-jac-chunk`, `--speed-floor`,
  `--arclength-weight`, `--torsion-weight`, `--planarity`/`--planarity-tol`. `common.py` gained
  `quad_grid` (length quadrature, split from the Biot-Savart `source_grid` - spline coils use
  Hanson-Hirshman POLYLINE Biot-Savart, so their source grid must be uniform).
- **`917_results/`** - nine coilsets + equilibrium, `build_data.py`, and an interactive B·n viewer
  (artifact). Mean AND max normalized B·n per coilset.

## NEXT STEP (agreed with the user): polar arcB2 on the 16-coil configuration

**The question.** At 12 coils, polar arcB2 recovers the unrestricted result to within 1.24x. If that
ratio carries over to 16 coils, polar arcB2 would reach ~3.80e-3 x 1.24 = **4.7e-3, about 0.50% of
B_axis** - roughly 2x the paper's 0.23%, and by far the closest a buildable piecewise-planar coilset
has come. This is the best available shot at striking distance of the paper, and it tests whether
the cheap-restriction result holds at higher coil count.

**Start from** `917_results/fourierXYZ_16coils.h5` (4 unique coils, L 36.000 m, kappa 1.500 /m
exactly on its bound, d_cc 1.226 m, currents 15.4-18.0 MA).

**Do these cheap checks BEFORE spending GPU** - this session lost ~45 min to skipping them:
1. `planarizability.py` on it. Report eps_2 and the optimal 2-arc breakpoints; those breakpoints ARE
   the hinge placement to use (better than a centroid-line cut angle).
2. Per-segment admissibility: theta must advance monotonically about each chord midpoint. A spline
   optimum this session was planar per segment to 1e-14 m and STILL outside the polar class because
   a deep re-entrant corner made it overshoot theta = pi. `make_start` will happily produce garbage
   if this fails.
3. Gate on the fit: proceed only if the fit deviation is <~0.2 m and the convexity excess is <~20x
   bound. MEASURED this session: a 0.63 m / 284x fit could not be repaired - the feasibility solve
   spent everything on convexity and drove bn 1.33e-2 -> 2.86e-2 within 50 iterations.
   - **CORRECTED 2026-09-18: the convexity half of this gate is wrong; DEVIATION is the whole gate.**
     All 17 twelve-coil fits that repaired fine sit at deviation 136-261 mm and convexity excess
     **1027-1753x** bound, i.e. every one of them would have failed a 20x test, and the fit that
     could NOT be repaired was 630 mm at only **284x** - lower convexity excess than any success.
     Use deviation < ~0.35 m (headroom over the worst success) and report convexity for information.
     Implemented in `work/helios/arc16/run_chain.sh`.

**Then** `make_start.py --from-coils ... --rep polararc --B 2 --M 5` at the DP's hinges, a
feasibility repair (`--mode feasibility --anchor-weight 1 --margin 0.25 --margin-skip L`) with an
**abort gate: if bn more than doubles by iteration 25, discard the start**, then stage 2.

**Expect d_cc to bind.** The 16-coil FourierXYZ optimum sits at d_cc 1.226 m against the 1.2 m bound
with kappa exactly at its limit; a 2-arc restriction has fewer ways to dodge. Note that no CIRCULAR
16-coil start satisfies both clearances (d_cc >= 1.2 needs r/a <~ 2.4, d_pc >= 1.2 needs r/a >~ 2.6,
no overlap) - the feasibility solve found a shaped one from r/a 2.5 in 6.6 min, so use
`work/helios/xyz16/feas/result.h5` if a fresh non-circular start is wanted.

## Incomplete / failed

- **N=8 resolution check for the 12-coil ceiling: failed twice** on `CUDA_ERROR_ILLEGAL_ADDRESS`
  (the second time it hung 20 min at iteration 22 first). So "the ceiling is not
  resolution-limited" rests on a spectral argument - the n=9,10 modes carry 1.2% of the n=1
  amplitude and are attenuated ~8x by the 1.2 m standoff - not on a measurement. Rerun on a fresh
  GPU session.
- The 12-coil ceiling started from the PLANAR optimum, so it is a lower bound (possibly
  basin-limited), not a proven ceiling.
- 16-coil run is 300 iterations, `Maximum number of iterations exceeded` - converged on bn to the
  4th significant figure but not formally converged.
- Handoff item 2(b) (arcB2 with convexity off) and 2(c) (total length budget) still untouched.

---

# Session 2026-09-18: polar arcB2 on the 16-coil configuration

## The answer

**At 16 coils, polar arcB2 reaches bn 8.7227e-3 (0.905% of B_axis), FEASIBLE on every paper bound.
But the piecewise-planar restriction costs 2.30x the unrestricted ceiling, not the 1.24x it cost at
12 coils. The "cheap restriction" result does NOT carry over to higher coil count.**

| coilset | coils | conductor | bn | aw ⟨\|B·n\|⟩/B_axis | max \|B·n\| | file |
|---|---|---|---|---|---|---|
| planar N=7 | 12 | 432 m | 4.0223e-2 | 4.036% | 0.984 T | `planarN7_final_it300.h5` |
| **planar N=7** | **16** | **576 m** | **3.3448e-2** | **3.365%** | 0.810 T | `work/helios/planar16/stage2/result.h5` |
| polar arcB2 cut30 | 12 | 432 m | 1.0450e-2 | 1.107% | 0.278 T | `polarB2_M5_cut30_final.h5` |
| **polar arcB2 cut30** | **16** | **576 m** | **8.7227e-3** | **0.905%** | 0.262 T | `work/helios/arc16/cut30/legB/result.h5` |
| FourierXYZ N=10 | 12 | 432 m | 8.4224e-3 | 0.880% | 0.181 T | `fourierXYZ_N10_ceiling.h5` |
| FourierXYZ | 16 | 576 m | 3.7978e-3 | 0.373% | 0.141 T | `fourierXYZ_16coils.h5` |

(`bn` reproduces every earlier row exactly; the area-weighted column reproduces the earlier table
to the quoted digits. Both recomputed together by `917_results/build_data.py` +
`work/helios/arc16/metrics.json`.)

**Restriction cost (arcB2 / FourierXYZ at the same coil count):**

| coils | arcB2 | FourierXYZ | ratio |
|---|---|---|---|
| 12 | 1.0450e-2 | 8.4224e-3 | **1.24x** |
| 16 | 8.7227e-3 | 3.7978e-3 | **2.30x** |

The predicted 4.7e-3 (from assuming the 1.24x ratio carries over) was wrong by 1.85x.

**What adding 4 coils bought each representation** (12 -> 16, +33% conductor):

| rep | 12 coils | 16 coils | change |
|---|---|---|---|
| planar N=7 | 4.0223e-2 | 3.3448e-2 | **-16.8%** |
| polar arcB2 | 1.0450e-2 | 8.7227e-3 | **-16.5%** |
| FourierXYZ | 8.4224e-3 | 3.7978e-3 | **-54.9%** |

The measured length elasticity d(ln bn)/d(ln L) = -2.1 predicts -45% from the +33% conductor alone.
FourierXYZ beats that; **planar and arcB2 both fall far short of it**. Extra coils are worth much
less to a shape-restricted coilset than raw conductor is, because the coils now pack tightly and a
restricted shape cannot dodge its neighbours.

**Why, and it is the same scale argument as before.** At 16 coils every representation sits near the
clearance bound (d_cc/bound: arcB2 1.036, planar 1.023, FourierXYZ 1.022 = 1.23-1.24 m) whereas the
12-coil arcB2 winner had 1.35 (1.62 m) and *only length was active*. The handoff's own conclusion —
"a constraint is expensive when the freedom acts on a much larger scale than the constraint does" —
now bites: a B=2 coil's freedom (hinges, tilts, 5 sine modes spanning an 18 m arc) is far coarser
than the local clearance it must respect, and at 16 coils clearance is finally binding. **This is a
direct argument for B=3 at 16 coils**, where the arcs are ~12 m rather than ~18 m.

**Framing worth keeping:** 16-coil arcB2 (8.72e-3) ≈ 12-coil UNRESTRICTED FourierXYZ (8.42e-3).
Four extra coils with two planar arcs each buys you exactly what fully free 3D shapes bought at 12.

## How it was done, and two corrections to the previous handoff

1. **The planned route (fit arcs to `fourierXYZ_16coils.h5` at the DP's breakpoints) is DEAD, and the
   pre-GPU checks caught it.** That optimum is genuinely 3D: eps_1 0.089-0.29, eps_2 0.049-0.095.
   Its own optimal 2-arc fit deviates **0.64-1.06 m** (vs the 0.63 m case known to be unrepairable)
   with convexity 2194x. Per-segment admissibility PASSED (theta monotone, 0 backsteps) so the
   obstruction is planarity, not the polar class. Tool: `work/helios/arc16/check_start.py`.
2. **The fit gate in this handoff was wrong (corrected above too).** It demanded convexity excess
   < 20x bound; all 17 twelve-coil fits that repaired fine are at **1027-1753x**, and the one that
   could NOT be repaired was at **284x**. Convexity excess does not discriminate. **Fit deviation
   does**: successes 136-261 mm, failure 630 mm. Gate on deviation < ~0.35 m alone.
3. **The route that works is the 12-coil recipe**: converged PLANAR optimum -> `--cut-angle` fit ->
   repair -> legA -> legB. Needed building the planar-16 optimum first (it did not exist, and it is
   a table row we were missing).
4. **`make_start.py --hinge-s "s0,s1;s0,s1;..."`** (new) takes hinge parameter values explicitly, so
   a chord need not pass through the centroid — this is also the `--cut-offset` lever the previous
   handoff asked for, in a more general form.

Chain: `work/helios/arc16/run_chain.sh CUT` (fit + deviation gate + repair + legA + hinge assertion
+ legB). Trajectory for cut 30: fit 3.327e-2 -> repair 3.389e-2 -> legA (held) 2.3135e-2 -> legB
8.7195e-3. Hinge change over leg A was exactly 0. legB reached FEASIBLE at iteration 175 and
converged by 225.

## The 16-coil arcB2 optimum ABANDONS one of its four coils

Per-coil, `work/helios/arc16/cut30/legB/result.h5` (verified, `arc_ref` margins 0.086-0.367, healthy):

| | coil 0 | coil 1 | coil 2 | coil 3 |
|---|---|---|---|---|
| current | **5.33 MA** | 21.16 MA | 19.56 MA | 18.91 MA |
| length | **25.19 m** | 36.000 m | 35.999 m | 36.000 m |
| corner | 36.7 deg | 53.2 deg | 73.9 deg | 108.6 deg |

Coil 0 carries a QUARTER of the others' current and is the only coil off the length cap. The
16-coil FourierXYZ optimum by contrast puts all four on the cap (L 35.99-36.00) at 15.4-18.0 MA.

So arcB2 actually drew only **+23% conductor** (4 x 133.2 = 532.8 m vs 432 m), not the +33% it was
allowed. **This is a concrete mechanism for the weak -16.5% gain: handed a fourth coil, the
restricted representation cannot find useful work for it and shrinks it away, while the unrestricted
one exploits it fully.** Whether this is intrinsic to B=2 or an artefact of the cut-30 basin is
exactly what the cut-angle scan would settle — it is now the most interesting reason to run it.

d_cc 1.243 m, d_pc 1.221 m, linked 0 (12-coil winner for comparison: 1.618 / 1.594 m).

## Caveats on this number

- **Only ONE hinge placement was run (cut 30).** At 12 coils the cut-angle spread was
  1.045e-2 to 1.510e-2 (**44%**), and the handoff's own finding is that the start angle selects a
  basin. So **8.72e-3 is an upper bound on what arcB2 can do at 16 coils**, and the 2.30x ratio is
  correspondingly an upper bound on the restriction cost. A cut-angle scan is the obvious next step.
- The 16-coil FourierXYZ reference stopped on max-iterations (not converged) and was seeded from a
  feasibility start, so it is itself a lower bound on unrestricted performance. If it improves, the
  2.30x grows.
- The planar-16 feasibility repair never reached the 25% margin (n_outer stayed at 1 through 300
  iterations) and the arc repair likewise (n_outer 1 through 120). Stage 2 cleared both. Worth a
  look if a future repair misbehaves.
- One `CUDA_ERROR_ILLEGAL_ADDRESS` (planar-16 stage 2, ~it 80); identical rerun passed and
  reproduced the earlier checkpoints bit-for-bit. The documented 1-in-8 fault.

## Next steps

0. **IF THE SCAN IS NOT DECISIVE: test whether leg A (50 iterations with hinges HELD) is itself the
   problem** (the user's hypothesis, registered 2026-09-18). Evidence already supports it: the
   current starving FIRST APPEARS IN LEG A, the constrained stage. Per-coil currents down the cut-30
   chain (sum is constrained, so current can only be MOVED between coils):

   | stage | currents (MA) |
   |---|---|
   | fit and repair | 17.31 14.30 15.79 17.57 (= the planar source, copied exactly) |
   | **legA, hinges held** | 15.96 **7.45** 14.90 26.65 |
   | legB, free | **5.33** 21.16 19.56 18.91 |

   With hinges frozen the optimizer cannot retract a corner, so its only levers are arc shape and
   current redistribution — and it immediately drives one coil to 7.45 MA. legB then inherits that
   lopsided state and only changes WHICH coil is starved. This also matches the 12-coil finding that
   "the held-hinge bn does not predict the final basin" (cut 120 best held, worst final).
   **The test is one chain (~1.2 h): repair -> free stage directly, no legA**, compared against
   cut30's 8.7227e-3. Note the warm start is NOT implicated — fit and repair preserve the planar
   currents exactly, and the planar source is evenly loaded (14.3-17.6 MA), as is 12-coil arcB2
   (21.6/21.2/22.1). Starving appears only at 16 coils, only after legA.

1. **Cut-angle scan for arcB2 at 16 coils** — the single biggest uncertainty in the number above.
   `work/helios/arc16/run_chain.sh ANGLE` is resumable and takes any angle or per-coil list.
2. **arcB3 at 16 coils** — now strongly motivated, not optional: the scale argument says shorter
   arcs are where the headroom is, and it is where the 12-coil ranking already pointed.
3. Untouched from before: 2(b) convexity off, 2(c) total length budget, 2(d) residual spectrum.
