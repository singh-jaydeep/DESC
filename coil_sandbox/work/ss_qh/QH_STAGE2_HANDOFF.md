# precise_QH single-stage equilibrium: stage-2 coil comparison (handoff, 2026-10-02)

**Your task.** Take the equilibrium from our QH single-stage run and **hold it fixed**. Run stage-2 coil
optimization on it with three representations:

* **planar** (FourierPlanar N=7);
* **arcB2** (two planar arcs per coil, M=5; this refines the coils that came out of single stage);
* **FourierXYZ N=6** (general 3D coils).

**Deliverable:** one offline 3D viewer (`view.py`) showing all three on that boundary, with their
⟨|B·n|⟩/⟨B⟩ and constraint margins. Report the numbers in a small table alongside it.

Read first: `coil_sandbox/CLAUDE.md` (machine rules: cap every heavy job with the cgroup, run ONE at a
time, confirm run configs with the user before launching, report progress about every 10 min). Background:
`work/SINGLE_STAGE_HANDOFF.md` (how the single-stage result was made and why QH is interesting: arcB2
improved 65% there, against 8–18% on precise_QA).

## 1. Inputs (measured on 2026-10-02 against the fixed equilibrium)

**Equilibrium:** `work/ss_qh/arcB2/ss_k4/eq_k4.h5`
* vacuum (p = 0, current ~1e-10), NFP 4, stellarator symmetric, L/M/N 8/8/8;
* R0 1.0015 m, a 0.1227 m, A 8.165, V 0.2974 m³, iota 1.244–1.257;
* Boozer QS 4.4e-4 / 6.7e-4 / 1.5e-3 / 4.5e-3 at rho 0.25 / 0.5 / 0.75 / 1 (0.9–4× worse than
  precise_QH).

It is not a `desc.examples` name, so pass the path to `--eq`. **Always pass `--vacuum`**: the
plasma-field (virtual-casing) machinery is for finite beta and is unnecessary here.

**Bounds:** `bounds/precise_qh_gil.json`, the Gil et al. Table III stated set, a-scaled to precise_QH,
**absolute** values. Use `--length-mult 1.0`. Because the values are absolute, the bounds do not move
with this equilibrium's slightly smaller a.

| L per coil | κ | κ_MS | d_cc | d_pc |
|---|---|---|---|---|
| ≤ 2.926 m | ≤ 13.67 /m | ≤ 18.69 /m² | ≥ 0.0585 m | ≥ 0.1097 m |

**Starts** (4 unique coils, 32 in total, none linked; dense check on THIS equilibrium):

| rep | file | ⟨\|B·n\|⟩/⟨B⟩ | margins (L, κ, κ_MS ≤ 1; d_cc, d_pc ≥ 1) |
|---|---|---|---|
| arcB2 M=5 | `work/ss_qh/arcB2/ss_k4/coils_k4.h5` | **1.92e-3** (max 8.0e-3) | L 1.000 (about 0.03% over, a penalty overshoot), κ 0.997, κ_MS 0.994, d_cc 1.205, d_pc 1.133 |
| planar N=7 | `../sweep_qh/initial starts/start_planarN7.h5` (cold, shaped planar) | 8.7e-2 | L 0.663, κ 0.461, κ_MS 0.743, d_cc 1.988, d_pc 1.381, feasible |
| xyz N=6, **warm** | `work/ss_qh/xyz/stage2/result.h5` (stage 2 on the ORIGINAL precise_QH, iteration-300 checkpoint; see `NOTE.txt`) | 1.39e-2 | L 1.000, κ 1.002, κ_MS 1.000, d_cc 1.023, d_pc 1.210 (slightly over) |
| xyz N=6, cold | `work/ss_qh/start_xyzN6.h5` | 8.8e-2 | feasible |

Use the warm XYZ start; on precise_QH it reached 9.35e-4. The arcB2 coils are the single-stage optimum
for this boundary, so stage 2 should only polish them, and the augmented Lagrangian will remove the tiny
length overshoot.

## 2. How our stage-2 runs work (`run_one.py`, `--mode stage2`)

* **Objective:** `QuadraticFlux` (area-weighted B·n, vacuum) on `LinearGrid(M=25, N=25, NFP, sym)`. The
  equilibrium is fixed.
* **Constraints**, from the bounds file with backoffs applied (printed as `ENFORCED` at the top of the log):
  * `CoilLength` per coil;
  * `CoilCurvature` (max |κ|);
  * `CoilMeanSquaredCurvature`;
  * `CoilSetDistancePenalty` (all pairs, two-pass, **signed**, so linked pairs are penalized);
  * `PlasmaCoilSetDistancePenalty` (two-pass, C¹);
  * the coil-current SUM is fixed, because quadratic flux is blind to the current scale.
* **Backoffs:**
  * d_cc ×1.025 and d_pc ×1.02;
  * for arcs also κ ×0.995, κ_MS ×0.99, plus a segment-sagitta term on d_cc.
  * Verdicts always use the UNbacked-off bounds.
* **Solver:** `lsq-auglag` with `second_order="constraints"` (constraint Hessian), trust-radius cap 0.5,
  max_inner_iter 100, ctol 1e-4.
* **Grids:**
  * constraints on `LinearGrid(N=150)` (301 nodes);
  * Biot–Savart source: composite Gauss–Legendre, 24 nodes per arc, for arcs; `LinearGrid(N=--src-n)`
    for smooth coils;
  * d_pc plasma grid M=N=64 with symmetry.
* **Outputs in `--out`:**
  * `result.h5`, `result.json` (settings, bounds, start and result metrics, history) and `result.png`;
  * `ckpt_itNNNN.h5` every 50 iterations;
  * a dense check every 50 iterations in the log (`[it N] bn ... -> FEASIBLE/FAILS ...`).

Every option is a flag: `python run_one.py --help`.

## 3. Configs to use

```bash
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=work/ss_qh/arcB2/ss_k4/eq_k4.h5
COMMON="--eq $EQ --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 --maxiter 400"
OUT=work/ss_qh/fixed_ss_eq

# arcB2: refine the single-stage coils (GPU fine; ~15 min per 400 iterations, peak RSS ~7 GB)
$CAP $PY run_one.py $COMMON --start work/ss_qh/arcB2/ss_k4/coils_k4.h5 --out $OUT/arcB2 --kappa-grid-n 600

# planar N=7 (cold start; not run on QH this session)
$CAP $PY run_one.py $COMMON --start "../sweep_qh/initial starts/start_planarN7.h5" --out $OUT/planarN7 --src-n 100

# FourierXYZ N=6 (warm start)
$CAP $PY run_one.py $COMMON --start work/ss_qh/xyz/stage2/result.h5 --out $OUT/xyzN6 --src-n 100 --so-chunk 4
#   if it crashes on the GPU (see pitfall 1): add --device cpu  (~12 s/iteration, levels off by ~200 iterations)
```

Lay these out for the user and wait for the OK before launching (their standing preference). Run them one
after another, not in parallel.

## 4. Pitfalls (each seen this session)

1. **FourierXYZ stage 2 crashes on the GPU** with `CUDA_ERROR_ILLEGAL_ADDRESS`. It happened on every
   attempt on both QA and QH (before iteration 1 or around iteration 14), including with XLA autotuning
   off. arcB2 never crashed. The suspect is memory in the second-order constraint Hessian, since
   QuadraticFlux is already Biot–Savart-chunked (10). Try `--so-chunk 4` (or `2`), then
   `--no-second-order`, then `--device cpu`, which works. A GPU crash can also be the intermittent
   one-in-eight fault, so retry once before debugging. Planar coils are also smooth and use the uniform
   source grid, so they may hit the same problem.
2. **Source grid:** for FourierXYZ N=6, `--src-n 100` (201 nodes per coil) matches the default 801 nodes to
   1e-15 (MEASURED for both QA and QH coilsets), so it gives the same answer with about 4× less work.
   For planar N=7, check this the same way before relying on it: compute B on the boundary at
   `LinearGrid(N=100)` and `N=400` and compare. It should hold for band-limited curves.
3. **κ between nodes:** the dense check (401 points plus Gauss nodes) can read κ 1–3% above what the
   301-node constraint grid enforces. QH arcB2 stage 2 ended at κ 1.025, and QH xyz at 1.002. For a
   strictly feasible verdict use `--kappa-grid-n 600` (not tested at this value; it costs some Jacobian
   size, so chunk with `--coil-jac-chunk` if VRAM is short).
4. **The augmented-Lagrangian tail drags.** B·n typically levels off by iteration 200–300 (QH xyz:
   9.43e-4 at iteration 200, 9.35e-4 at 300). The user is happy to stop a run once B·n changes by less
   than about 1% between dense checks. **Killing a run leaves no `result.h5`**, so copy the last
   `ckpt_itNNNN.h5` to `result.h5` and write a short `NOTE.txt` saying so. The viewer and any later
   steps expect `result.h5`.
5. **The cold planar start is far from optimal** (8.7e-2), so expect the first ~50 iterations to be
   rough. On precise_QH, a 3-iteration arcB2 smoke run peaked at 5.4 GB RSS. QH stage 2 runs peaked at
   ~7 GB, fine under the 12 GB cap.
6. **QH B=2 arcs from circle-based starts were unusable** (the coil enters the plasma; see the sweep_qh
   notes). Do NOT rebuild arcB2 from circles; refine `coils_k4.h5`. If you ever need to raise arc M,
   use `pad_arcs.py`: it zero-pads and copies `arc_ref`, `rotmat` and `shift`; without that copy the
   curve moved 27 mm.
7. **`common.evaluate` caches boundary data by `id(eq)`.** Within one process, clear `S._BN_CACHE` before
   evaluating on a different equilibrium object (view.py does this).
8. **Length sits exactly on its bound in every run** (it is always the active constraint), so "FAILS L"
   at the 1e-5 level is solver noise, not a real violation. Report margins to 3 decimals.

## 5. The viewer (deliverable)

All three coilsets sit on the same fixed equilibrium, so a single `--eq` is enough:

```bash
$PY view.py $OUT/view.html --eq $EQ --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 \
    --h5 "planar N7=$OUT/planarN7/result.h5" \
    --h5 "arcB2 M5=$OUT/arcB2/result.h5" \
    --h5 "FourierXYZ N6=$OUT/xyzN6/result.h5" \
    --h5 "arcB2 single-stage start=work/ss_qh/arcB2/ss_k4/coils_k4.h5" \
    --title "precise_QH single-stage boundary: stage-2 coils, planar vs arcB2 vs FourierXYZ"
```

* The viewer is a self-contained HTML file. For each coilset it shows all 32 coils, the boundary, the
  closest coil–coil approach, the curvature peak of each unique coil, linked pairs, and the dense-check
  numbers against the bounds, with a coilset selector.
* **Labels must not contain "="**: the `NAME=PATH` parser splits on the first one. Use "N7", not "N=7".
* Pass `--length-mult 1.0` explicitly. Otherwise `view.py` reads each `result.json`'s multiplier and
  would double-apply it to a pre-scaled bounds file.
* Optional context: a coilset can carry its OWN boundary with `NAME=PATH@EQ`, drawn with a faint overlay
  of the other boundaries. For example, `--h5 "arcB2 on original precise_QH=work/ss_qh/arcB2/stage2/result.h5@precise_QH"`
  shows how the single-stage boundary differs from the original.
* Also make the numbers table. Per rep, give ⟨|B·n|⟩/⟨B⟩, its max, the five margins and the active
  constraints, from each `result.json` (`result`, `margins`, `active`) or by re-running
  `common.evaluate`. If you want a static image as well, the matplotlib snippets in
  `work/ss_qa/qsladder/analyze.py` show the pattern.

## 6. Reference numbers (to sanity-check your results)

| on precise_QH (original boundary) | stage 2 | after single stage (this boundary) |
|---|---|---|
| arcB2 M=5 | 5.48e-3 | 1.92e-3 (the boundary was co-optimized) |
| FourierXYZ N=6 | 9.35e-4 | – (you are measuring this) |

The question your viewer answers: on a boundary co-designed for arcB2, how close do planar coils and
refined arcB2 coils come to general 3D coils?
