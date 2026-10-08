# Single-stage coil optimization: status and handoff (2026-09-28 → 2026-10-02)

Read this together with `coil_sandbox/CLAUDE.md` (machine rules: cgroup cap, one job at a time, confirm
run configs with the user before launching; the stage-2 recipe and its reasons). The next session will try
a **QI** example; section 7 lists what changes for QI.

## 1. The question and the short answer

The user wants to know whether some equilibria permit **simple coils** (target: **arcB2**, two planar
arcs per coil, 4 coils per half period). Stage-2 Pareto sweeps on fixed equilibria were unremarkable, so
this session tried **single stage**: the boundary and the coils optimized together, starting from a stage-2
result, with quasisymmetry (QS) allowed to degrade within a tolerance.

Answer so far:
* **precise_QA is rigid.** With QS kept useful (τ_QS ×10, QS 2–5× worse), single stage improves arcB2's
  ⟨|B·n|⟩/⟨B⟩ by only 8–18%. Loosening QS step by step shows how the gain is bought: QA gets its iota
  from the rotating, elongated bean, and that shaping puts the bean tips into the coil ripple. B·n falls
  only when the plasma is un-shaped, which costs QS and iota (QS off: −60%; QS off and iota relaxed: −73%).
* **precise_QH is not.** At the same τ_QS, arcB2 improves by **65%** (5.48e-3 → 1.92e-3), and QS ends only
  0.9–4× worse than precise_QH. QH gets its iota from helical axis excursion, not from elongated tips.
* **With general coils (FourierXYZ N=6), single stage adds almost nothing on QA** (−4%; QS even improves).
  On QH, xyzN6 stage 2 alone reaches 9.35e-4; the QH xyzN6 single stage (**H4**) was never run (section 6).
* For context: Jorge, Giuliani & Loizu (arXiv 2406.07830, `928_singlestage/`) quote the MAX |B·n|/B. Their
  planar-coil case is 1.4e-2 and their best single-stage case 6e-4 (non-planar coils, a simple nfp4
  configuration). Their 1e-4–1e-6 results are QUASR devices from direct Boozer optimization with Fourier
  order-16 coils, so they are not comparable. Our maxima: QA arcB2 7.5e-3 with QS off; QH arcB2 8.0e-3
  with QS kept.

## 2. Results (⟨|B·n|⟩/⟨B⟩ from the dense check `common.evaluate`; max in parentheses)

Single-stage settings unless noted: τ_QS ×10 (relative to the start's scale-invariant QS rms), volume
within ±2% of the start, A bounds, iota floor at about 0.95× the start's minimum, k = 1→4 boundary
continuation, 100 iterations per step, `--solve-tol 1e-10`.

| case | stage 2 | single stage | QS (Boozer) vs the original eq | folder |
|---|---|---|---|---|
| QA arcB2 M=3, L×1.0 | 1.96e-3 (1.09e-2) | – | – | `ss_qa/stage2` |
| QA arcB2 M=3, L×0.8 | 5.15e-3 (3.70e-2) | – | – | `ss_qa/stage2_L0p8` |
| QA arcB2 M=5, L×0.8 | 4.54e-3 (1.96e-2) | **4.18e-3** (1.66e-2), clean k=1→4 route | 1–3× worse | `ss_qa/stage2_L0p8_M5`, `ss_qa/arcB2_clean/ss` |
| QA arcB2, mixed route (k=1–2 at M=3, then M=5 from k=3) | – | **3.75e-3** (1.79e-2) | 2–5× worse | `ss_qa/ss_v2_qs10*` → `ss_v2_qs10_k4_M5` |
| QA xyzN6, L×0.8 | 2.30e-3 | **2.22e-3** (8.6e-3) | **1.5–3× better** | `ss_qa/xyz/stage2`, `ss_qa/xyz/ss` |
| QH arcB2 M=5, L×1.0 | 5.48e-3 (κ 2.5% over on the dense grid) | **1.92e-3** (8.0e-3) | 0.9–4× worse | `ss_qh/arcB2/stage2`, `ss_qh/arcB2/ss` (k≤3), `ss_qh/arcB2/ss_k4` |
| QH xyzN6, L×1.0 | **9.35e-4** (iteration-300 checkpoint, killed; κ 0.2% over) | **not run (H4)** | – | `ss_qh/xyz/stage2` (see `NOTE.txt`) |

Why the two QA arcB2 routes differ is not understood. The mixed route started its k=1–2 steps from M=3
stage-2 coils, which may have had more room to improve. Treat the clean route as the like-for-like comparison.

**The QA ladder** (`ss_qa/qsladder/SUMMARY.md`, figures, `view.html`), warm-started from the mixed-route
×10 k=4 result:

| τ_QS | ⟨B·n⟩/⟨B⟩ | max | Boozer QS at the edge | binding constraints |
|---|---|---|---|---|
| ×10 | 3.75e-3 | 1.79e-2 | 7.9e-4 | L, iota ≥ 0.40, volume |
| ×30 | 3.39e-3 | 1.39e-2 | 2.0e-3 | same |
| ×100 | 2.83e-3 | 1.20e-2 | 4.8e-3 | same |
| ×1000 (QS off) | 1.81e-3 | 7.5e-3 | 1.7e-2 | same |
| ×1000, iota ≥ 0.30 | 1.50e-3 | 6.0e-3 | 2.1e-2 | L, volume (iota settled at 0.36) |
| ×1000, iota ≥ 0.20 | 1.24e-3 | 5.6e-3 | 2.5e-2 | L, volume (iota 0.32) |

Mechanism: the changes are in low-order modes (|m|, |n| ≤ 2, `spectrum.png`). The bean tips retract,
which reduces elongation, and the plasma bulges outboard to keep its volume (`xsections.png`). This is not
corrugation at the coil scale: the ripple sits at n ≥ 8 per period, and k ≤ 4 cannot represent it. Up to
×100, the boundary moved most under the B·n hot spots (top-10% overlap 0.32–0.35 against 0.10 by chance,
p < 0.001). At the start, QS error and B·n hot spots do not coincide; single stage pulls the QS damage
onto them.

## 3. Drivers and tools (all in `coil_sandbox/`)

| file | what it is |
|---|---|
| `single_stage.py` | **The single-stage driver.** `proximal-lsq-exact` on [eq, coils], ONE weighted least-squares objective (no augmented Lagrangian), boundary continuation k (free modes \|m\|,\|n\| ≤ k; R₀₀ always fixed), and a `--dry` term table. Run `--help` for every flag. |
| `sandbox/bnormal.py` | `VacuumBoundaryBn`: B_coil·n·√g on the MOVING boundary, area-weighted (the same as QuadraticFlux), normalized by ⟨\|B\|⟩√(mean g) at build, or by `--bn-norm` |
| `sandbox/qs_scale_invariant.py` | `QuasisymmetryTwoTermScaleInvariant`: f_C/\|B\|³ pointwise. A port of DESC PR #2304, which is not in the vendored DESC. Verified invariant under Ψ×1.2. |
| `run_one.py` | Stage 2 (augmented Lagrangian, unchanged recipe). Used for every stage-2 baseline. |
| `pad_arcs.py` | Raises the arc M exactly by zero-padding. It must copy `arc_ref`, `rotmat` and `shift`; without that the curve moved 27 mm. |
| `view.py` | Now accepts `--h5 NAME=PATH@EQ` (a coilset on its own boundary, with a faint overlay of the other boundaries) and `--length-mult` (one multiplier for every coilset; use it with an UNSCALED bounds file). Names must not contain "=". |
| `ss_qa/qsladder/analyze.py` | Ladder diagnostics: summary table, binding constraints, signed displacement from the original boundary (nearest point), B·n and QS maps, co-location statistics with a shift null, boundary spectrum, viewer |
| `work/compare/*.sh` | The 2026-09-29 chains (`run_chain.sh`, `queue2.sh`, `queue3.sh`): step functions, fallbacks, logs |
| bounds | `bounds/precise_qa_sweep910.json` (sweep_910, R₀-scaled, absolute), `_L0p8.json` (length ×0.8), `bounds/precise_qh_gil.json` (Gil Table III stated set, a-scaled to QH). The QA and QH bound recipes differ, so cross-machine comparisons are approximate. |

### How `single_stage.py` weights things (`--dry` prints it all)
Each term gets weight 1/(τ·s). s is chosen so that a residual equal to τ in every row costs ½, which makes
the err/tol column √(2·cost).
* B·n and QS: τ applies to the quadrature-weighted rms. Defaults: `--tau-bn 5e-3`; `--tau-qs-rel 10`
  multiplies the START's QS rms, or pass `--tau-qs` as an absolute value.
* Coil and shape bounds (length, κ, κ_MS, d_pc, A, iota, volume) are fractional excursions with τ ≈ 1%
  (`--tau-V 0.005`). They are penalties, so expect overshoots of about 0.1%.
* The coil–coil penalty is divided by N² (pair mean), with `--tau-dcc 1e-6`.
* CoilCurvature's DESC quad weight is dθ, not √dθ (grid dependent); the s factor corrects it.
* The current SUM is fixed (B·n is blind to the current scale). Volume band `--vol-band 0.02` around V₀.
* **To continue a run, pass `--tau-qs`, `--V0` and `--bn-norm` explicitly** (they are printed at the top
  of every log). Otherwise they re-anchor to the new start.

### Typical commands
```bash
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
# stage 2 baseline
$CAP $PY run_one.py --eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0 --vacuum \
    --start work/ss_qh/start_arcB2_M5.h5 --out work/ss_qh/arcB2/stage2 --maxiter 400
# look at the terms first (cheap)
$CAP $PY single_stage.py --eq precise_QH --start work/ss_qh/arcB2/stage2/result.h5 --out work/x/dry \
    --bounds bounds/precise_qh_gil.json --length-mult 1.0 --helicity 1,4 --A-min 7.5 --A-max 8.5 \
    --iota-min 1.19 --tau-qs-rel 10 --dry
# single stage, k = 1..4
$CAP $PY single_stage.py <same flags without --dry> --out work/x/ss --kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10
```
Outputs: `eq_kK.h5`, `coils_kK.h5` and `result.json` after every k step. Each log reports the physics
(A, iota range, Boozer QS at 4 radii) and the dense coil check per step. Timing on the RTX 5070: QA k steps
take 6–17 min per 100 iterations; QH similar to slower. Host RSS ~5.5–7 GB.

## 4. Traps found (MEASURED this session)

1. **`VacuumBoundaryError` is not quadratic flux.** It has B² rows, which carry the tangential mismatch at
   about 2× B·n, and its B·n rows are weighted by g rather than √g. The optimizer traded B·n for B² and
   moved the error inboard. Use `VacuumBoundaryBn`.
2. **The proximal re-solve adds noise** (about 6e-4 on a cost of 1.16 at default tolerances), which collapses
   the trust region near the optimum. `--solve-tol 1e-10` fixes it.
3. **QS stiffness:** at τ_QS ×1 the two-term QS pins the boundary (curvature 1e3–1e5× B·n per mode).
   Near-null directions exist in mode combinations. Linear Gauss–Newton predictions fail beyond about
   3–10% of a step, so do not trust "recoverable" tables built from them.
4. **B·n alone shrinks the plasma** (the ripple decays inward). The stock QS normalization (B_scale³)
   accidentally resisted this. Use scale-invariant QS plus the volume band.
5. **`xtol` 1e-6 can end a k step after 2 iterations** with optimality still at 2.5e-2, because rejected
   steps shrink the radius. H2's k=4 rerun with `--xtol 1e-12` gave another −15%.
6. **κ on the dense grid can exceed the 301-node constraint grid by 1–3%** (curvature between nodes; QH
   arcB2 and QH xyzN6). Use a finer `--kappa-grid-n` in run_one or a finer `--coil-grid-n` if strict
   feasibility matters.
7. **FourierXYZ on the GPU:**
   * single stage needs Biot–Savart chunking (`--bs-chunk 100 --jac-chunk 20` worked for QA's 16 coils
     but NOT for QH's 32; try smaller);
   * `--src-n 100` (201 nodes) matches the default 801 nodes to 1e-15 for N=6;
   * stage 2 (run_one) crashed with CUDA_ERROR_ILLEGAL_ADDRESS on every attempt even with its bs chunk of
     10 (suspect the second-order Hessian; untested: `--so-chunk 4` or `--no-second-order`). CPU works
     (~12 s/it). arcB2 never failed on the GPU.
8. **Bookkeeping:** `common.evaluate` caches boundary data by `id(eq)`, and eq changes in place in single
   stage, so clear `S._BN_CACHE` (the driver does). In DESC `compute`, `basis="xyz"` converts EVERY vector
   output, normals included.
9. **Killing a stage-2 run** leaves no `result.h5`. Copy the last `ckpt_itNNNN.h5` (written every 50
   iterations) and note it, as done for H3.

## 5. Things the user decided or prefers
* Prefers kinked hard minima to C¹ penalties in weighted mode (d_pc uses `PlasmaCoilSetMinDistance`, eq not fixed).
* Corners at the hinges are fine. Prefers less wild coils (hence length ×0.8 on QA).
* Stop a stage-2 run once it levels off (the AL tail drags), keeping its checkpoint.
* Progress reports about every 10 minutes; stop runs that are clearly failing; save checkpoints for joint analysis.

## 6. Open items
* **H4** (QH xyzN6 single stage), from `work/ss_qh/xyz/stage2/result.h5` with the QH flags of
  `work/compare/queue3.sh`, plus `--src-n 100` and smaller chunks (e.g. `--bs-chunk 25 --jac-chunk 5`), on the GPU.
* Joint analysis across QA/QH × arcB2/xyzN6 × stage 2/single stage. Adapt `qsladder/analyze.py`
  (cross-sections, maps, spectrum) plus `view.py ...@EQ`.
* The QA arcB2 mixed-route vs clean-route gap (3.75e-3 vs 4.18e-3).
* Make `--bs-chunk` default on in `single_stage.py`; consider `--src-n 100` as the XYZ default.
* Optional: a QH ladder like the QA one; τ_QS ×1 on QH ("QS tight") was not tried.
* **Stage-2 comparison on the fixed QH single-stage boundary** (planar / arcB2 / FourierXYZ plus a viewer):
  a separate task with its own handoff, `work/ss_qh/QH_STAGE2_HANDOFF.md`.

## 7. For the QI session
* The driver's physics term is the QS two-term (`--helicity`). **QI is not QS**, so add a physics-term
  switch. Candidates in the vendored DESC: `Omnigenity` (needs an `OmnigenousField`; the proper QI
  target), `Isodynamicity`, `MirrorRatio`, `Elongation`, or `QuasisymmetryBoozer` with helicity (0, NFP)
  as a crude quasi-poloidal proxy. Keep the τ scheme: build s from that term's quad weights and check the
  `--dry` table.
* The driver refuses non-vacuum equilibria. The vendored examples include W7-X (QI-like); check p and
  current. The user's ConStellaration notes (memory) list QI boundaries, including finite-beta subsets, but
  no coils. A boundary-only input must be solved first.
* Needs: a bounds file (a-scaled Gil set via `paper_bounds`, or absolute), a start (`make_start.py`; check
  `linked` and d_cc, and remember the QH start needed shaped or blended normals), a stage-2 baseline,
  then single stage. Set the iota floor and A bounds relative to the start (we used 0.95× iota_min and
  roughly ±0.5 in A).
* Run `--dry` first and confirm the settings with the user before any long run.
