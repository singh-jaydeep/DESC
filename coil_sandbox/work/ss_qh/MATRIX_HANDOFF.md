# precise_QH single stage per coil representation: the boundary × coils matrix (handoff, 2026-10-07)

**Your task in one paragraph.** We want to know whether a single-stage boundary designed *for* a coil
representation helps *that* representation more than a boundary designed for another one. Concretely: does
single stage aimed at arcB2 help arcB2 more than single stage aimed at planar coils (or at general 3D
coils) does? You will (1) compute converged stage-2 baselines for three representations on the original
precise_QH, (2) run single stage three times (equilibrium + coils together), once per representation, with
**identical settings and weights**, and (3) refine every representation's coils on every resulting boundary
with the new converging augmented-Lagrangian solver. The answer is read off a 4 × 3 table of B·n.

Read, in this order, before touching anything:
1. `coil_sandbox/CLAUDE.md` (machine rules, which DESC, the stage-2 recipe and why).
2. This file, completely.
3. Skim `work/SINGLE_STAGE_HANDOFF.md` (how `single_stage.py` works and its traps) and
   `work/ss_qh/fixed_ss_eq/RESULTS.md` (everything measured on the existing arcB2 single-stage boundary,
   including why the old stage-2 solver failed on planar coils and how the new one fixes it).
4. Skim `../devtools/coil_auglag/GUIDE.md` (the new solver and distance rows).

---

## 1. Orientation

### 1.1 Where things are

| what | where |
|---|---|
| sandbox root (run everything from here) | `/home/singh/Documents/DESC2/coil_sandbox` |
| DESC used | the main repo, the sandbox's parent `/home/singh/Documents/DESC2`, branch `js/coil-auglag`. `sandbox/boot.py` puts it first on `sys.path`; every log prints `DESC /home/singh/Documents/DESC2/desc` and `DEVICE requested=gpu obtained=gpu`. `SANDBOX_DESC=vendor` switches to the old `vendor/` snapshot (only to reproduce pre-2026-10-07 runs; do not use it here). |
| python | `/home/singh/miniforge3/envs/desc-env2/bin/python` (`desc-env2`; numpy 2.4.6, jax 0.9.2, adv-jax-math 1.3) |
| sanity check | `python check.py` → must end `SANDBOX OK` and list `lsq_auglag_composite`, `CoilSetDistanceRows`, `PlasmaCoilDistanceField` |
| stage 2 (new solver) | `run_al.py` + `sandbox/al_problem.py` (written 2026-10-07; section 4.1) |
| single stage | `single_stage.py` (penalty form, proximal `lsq-exact`; section 4.2) |
| arc repacking | `work/ss_qh/fixed_ss_eq/repack_arcs.py` (section 5, trap 1) |
| dense check (the verdict) | `sandbox/common.py: evaluate`, `margins`, `verdict`; `run_al.py` runs it at the end |
| viewer | `view.py` (section 4.4) |
| bounds | `bounds/precise_qh_gil.json` (absolute; L ≤ 2.92551 m, κ ≤ 13.6728 /m, κ_MS ≤ 18.6946 /m², d_cc ≥ 0.058510 m, d_pc ≥ 0.109707 m) |
| output root for this task | `work/ss_qh/matrix/` (create it; layout in section 3) |

### 1.2 Machine and user rules (non-negotiable)

* Laptop: WSL2, ~15 GB RAM, RTX 5070 8 GB. **Every heavy job under the cgroup cap, one at a time:**
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 <python> ...`. WSL dies outright if
  the VM runs out of memory.
* **Lay out every run's configuration and wait for the user's OK before launching** (standing preference).
  Pre-checks (`--dry`, `--maxiter 0`, small CPU checks) are fine without asking.
* Report progress about every 10 minutes while something runs: for `run_al.py` the per-outer line
  (inner its, optimality vs gtolk, violation vs ctolk) and the brute-force `true:` line; for
  `single_stage.py` the last solver rows and each `k=N physics` / `k=N coils` line.
* Stop runs that are clearly failing. Keep checkpoints. When you stop a run early, say so in a `NOTE.txt`.
* Measure only what changes a decision. State results plainly; say what was and was not verified.

### 1.3 The three representations

| name | class | DOF per coil | Biot–Savart source grid | gauge handled by `al_problem.py` |
|---|---|---|---|---|
| planar N7 | `FourierPlanarCoil`, N=7 | 21 | `LinearGrid(N=50)` (checked = N=100 to 5e-15 / 4.5e-11) | largest `normal` component pinned |
| arcB2 M5 | `PiecewisePlanarArcCoil`, B=2 arcs, M=5 | hinges, tilts, shape | Gauss–Legendre, 24 per arc (48/coil) | none needed (`arc_ref`, `rotmat`, `shift` fixed by default) |
| FourierXYZ N6 | `FourierXYZCoil`, N=6 | 39 | `LinearGrid(N=50)` (checked = N=100 to 4e-11) | phase pinned + relative speed rows within ±0.5 |

All: 4 unique coils per half period, 32 physical (NFP 4, stellarator symmetric), coil-current SUM fixed.

---

## 2. What is already established (do not redo)

On the **existing** arcB2 single-stage boundary `work/ss_qh/arcB2/ss_k4/eq_k4.h5` (made 2026-09-29 by
`single_stage.py` from old-solver arcB2 stage-2 coils), all three representations were refined with the
new solver on 2026-10-07 (`work/ss_qh/fixed_ss_eq/al_*`), all converged (optimality and violation < 1e-6):

| coils | ⟨\|B·n\|⟩/⟨B⟩ | max | ÷ XYZ | outer/inner its | wall |
|---|---|---|---|---|---|
| FourierXYZ N6 | 5.192e-4 | 2.12e-3 | 1.00 | 11/94 | 218 s |
| arcB2 M5 | 1.449e-3 | 6.35e-3 | 2.79 | 5/72 | 143 s |
| planar N7 (cold start) | 1.852e-2 | 7.77e-2 | 35.7 | 11/453 | 500 s |

(+ ~3 min field build each.) On the **original** precise_QH, old-solver stage 2 gave arcB2 5.48e-3 and XYZ
9.35e-4; planar was never run there; these will be redone with the new solver in step 1.

Because the arcB2 single stage will be **redone** from a new-solver baseline (user decision 2026-10-07),
`ss_k4` becomes a reference row, not part of the matrix.

Why the new solver matters: the old `lsq-auglag` (still used by `run_one.py`) puts inequality slacks in a
box handled by trust-region-reflective steps; weakly active rows (per-node κ, C¹ distance penalties) cut
almost every step short, and planar coils never got below an optimality of ~1e-3 and never became
κ-feasible. `lsq-auglag-composite` uses slack-free hinge rows and converges (details: `fixed_ss_eq/RESULTS.md`).
**Use `run_al.py` for every stage-2 run in this task. Do not use `run_one.py`.**

---

## 3. The experiment

### 3.1 The matrix you are filling

| boundary ↓ / coils → | planar N7 | arcB2 M5 | FourierXYZ N6 |
|---|---|---|---|
| B0: precise_QH (original) | step 1 | step 1 | step 1 |
| B_planar: single stage with planar coils | step 3 (diagonal: from step 2's coils) | step 3 | step 3 |
| B_arcB2: single stage with arcB2 coils (REDONE) | step 3 | step 3 (diagonal) | step 3 |
| B_xyz: single stage with XYZ coils (the old "H4") | step 3 | step 3 | step 3 (diagonal) |
| reference: old arcB2 boundary `ss_k4` | 1.852e-2 | 1.449e-3 | 5.192e-4 |

Every cell is a converged `run_al.py` result on that fixed boundary, judged by the dense check.

**How to read it** (state this in the deliverable):
* Down each column: is the representation best on its own boundary? E.g. is arcB2 lower on B_arcB2 than on
  B_planar and B_xyz? Is planar lower on B_planar than on B_arcB2?
* Improvement factor per cell: B·n / B·n on B0 (same column).
* If every column is best on its own row ("diagonal dominance"), targeting a representation helps that
  representation specifically. If one boundary is best for every column, single stage mostly makes the
  plasma generally coil-friendly.
* Next to each row, report the boundary's physics cost (Boozer QS at ρ = 0.25/0.5/0.75/1, A, R0, a, iota
  range, V/V0): a boundary that gave up more QS may be easier for everything for that reason alone.

### 3.2 Folder layout (create under `work/ss_qh/matrix/`)

```
matrix/
  starts/arcB2_stage2_old_coilset.h5     repacked old arcB2 stage-2 coils (step 1 start)
  base/{planarN7,arcB2,xyzN6}/           step 1: run_al.py on precise_QH
  ss/{planar,arcB2,xyz}/                 step 2: single_stage.py (coils_k1..4.h5, eq_k1..4.h5, result.json, run.log)
  on_planar/{planarN7,arcB2,xyzN6}/      step 3: run_al.py on ss/planar/eq_k4.h5
  on_arcB2/{planarN7,arcB2,xyzN6}/       step 3: run_al.py on ss/arcB2/eq_k4.h5
  on_xyz/{planarN7,arcB2,xyzN6}/         step 3: run_al.py on ss/xyz/eq_k4.h5
  RESULTS.md, matrix.md, matrix.json, view.html, scripts (*.sh)
```

### 3.3 Step 0: pre-checks (no OK needed)

```bash
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
M=work/ss_qh/matrix; mkdir -p $M/starts
$PY check.py                                                       # SANDBOX OK
# arcB2 old stage-2 coils are a MixedCoilSet: repack (trap 1); prints B.n 5.4822e-03 on precise_QH
JAX_PLATFORMS=cpu $PY work/ss_qh/fixed_ss_eq/repack_arcs.py work/ss_qh/arcB2/stage2/result.h5 \
    $M/starts/arcB2_stage2_old_coilset.h5 precise_QH
```

Then a build check (`--maxiter 0`) of each step-1 run (about 3 min each; reports rows, gap, margin,
memory, START dense and brute-force checks), e.g.:

```bash
$CAP $PY run_al.py --eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json \
    --start $M/starts/arcB2_stage2_old_coilset.h5 --out $M/base/arcB2_build --cc-gap coils --cc-k 40000 --maxiter 0
```

Check in each build log: `cc active rows at start N of K` (N must be < K, ideally < K/2), the coil-coil
`gap {'used', 'needed'}` in the `START true:` line (used must be ≥ needed/1.5; for arcs use `--cc-gap
coils`), `field {'max_error' ~1e-6, 'concave_radius' > 0.165 m}`, and peak `[mem]` (expect 4–6 GB).
`--maxiter 0` writes no `result.h5`; use a separate `*_build` folder.

### 3.4 Step 1: baselines on precise_QH (3 runs, ~10 min each, confirm configs first)

```bash
COMMON="--eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800"
$CAP $PY run_al.py $COMMON --start "../sweep_qh/initial starts/start_planarN7.h5" --out $M/base/planarN7 > $M/base/planarN7/run.log 2>&1
$CAP $PY run_al.py $COMMON --start $M/starts/arcB2_stage2_old_coilset.h5 --out $M/base/arcB2 --cc-gap coils --cc-k 40000 > $M/base/arcB2/run.log 2>&1
$CAP $PY run_al.py $COMMON --start work/ss_qh/xyz/stage2/result.h5 --out $M/base/xyzN6 --cc-k 20000 > $M/base/xyzN6/run.log 2>&1
```

(`mkdir -p` each `--out` first; `run_al.py` creates it but the shell redirect needs it.) Starts: planar
cold (built for precise_QH; B·n 8.65e-2 there), arcB2 and XYZ warm from their old-solver results. Each must
end `SOLVER \`gtol\` condition satisfied.`; check `RESULT ... linked 0`.

### 3.5 Step 2: single stage, once per representation (~1 h each; confirm configs first)

Identical settings for all three (the user asked for identical weights). These are the settings of the
original arcB2 single stage, plus `--xtol 1e-12` on every step (in the original only k = 4 had it; trap 6):

```bash
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --tau-qs-rel 10"
# dry first, for each (prints the term table; must show tau_qs 0.01695 and B.n normalization ~0.425)
$CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/planarN7/result.h5 --out $M/ss/planar --src-n 100 --dry
# then the runs
$CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/planarN7/result.h5 --out $M/ss/planar --src-n 100 > $M/ss/planar/run.log 2>&1
$CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/arcB2/result.h5    --out $M/ss/arcB2 > $M/ss/arcB2/run.log 2>&1
$CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/xyzN6/result.h5    --out $M/ss/xyz --src-n 100 --bs-chunk 25 --jac-chunk 5 > $M/ss/xyz/run.log 2>&1
```

Why the weights are identical across representations: `--tau-qs-rel 10` multiplies the QS rms of the START
EQUILIBRIUM (precise_QH, the same for all: 0.001695 → τ_QS 0.01695), and the frozen B·n normalization is
⟨|B|⟩·√(mean g) on the eval grid, a field-strength scale (0.42526 on precise_QH), not the start's B·n. A
`--dry` of planar on precise_QH (2026-10-07, main repo) printed exactly these two numbers, the same as the
original arcB2 run's log. Verify both lines in each run's log; if either differs, stop and ask.

Outputs per run: `coils_k{1..4}.h5`, `eq_k{1..4}.h5`, `result.json`, `run.log`. After each k the log prints
`k=N physics  A … R0 … a … iota [min, max]  QS(Boozer, |B_nonsym|/<B>) rho 0.25 … rho 1 …`,
`k=N coils  bn … (max …) L … kappa … kMS … d_cc … d_pc … linked …`, and `k=N solver … wall … min`.
Each k is 100 iterations and normally ends on `maxiter` (that is expected). Reference timings (old run,
arcB2): k = 1, 2, 4 took 11.9, 14.2, 16.2 min (k = 3 was not timed in the log); peak RSS 7.3 GB.

The XYZ single stage has never completed on this machine (trap 4). Order the runs planar → arcB2 → XYZ so
the two reliable ones are done first.

### 3.6 Step 3: every representation on every new boundary (9 runs, ~10 min each; confirm first)

For boundary `X` in {planar, arcB2, xyz} with `EQ=$M/ss/X/eq_k4.h5`:

* **Diagonal** (the representation that made the boundary): start from that single stage's own coils,
  `$M/ss/X/coils_k4.h5` (repack first if it is a MixedCoilSet; trap 1).
* **Off-diagonal**: start from that representation's step-1 baseline `$M/base/<rep>/result.h5`
  (its precise_QH optimum, moved onto the new boundary). The existing `fixed_ss_eq` runs did the same.

```bash
COMMONX="--eq $EQ --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800"
$CAP $PY run_al.py $COMMONX --start <start> --out $M/on_X/planarN7                                   # planar
$CAP $PY run_al.py $COMMONX --start <start> --out $M/on_X/arcB2 --cc-gap coils --cc-k 40000         # arcs
$CAP $PY run_al.py $COMMONX --start <start> --out $M/on_X/xyzN6 --cc-k 20000                        # XYZ
```

Do a `--maxiter 0` build check of at least the first run on each new boundary (the boundary changed, so
the plasma field is rebuilt and the reach check re-run; `concave_radius` must stay > 1.5 d_pc = 0.165 m).

---

## 4. Tool reference

### 4.1 `run_al.py` (stage 2, new solver)

```
python run_al.py --eq EQ --start COILS.h5 --bounds B.json --out DIR [--vacuum] [--maxiter 800]
    [--gtol 1e-6] [--ctol 1e-6] [--field-n 50] [--curv-n 300] [--row-n 256] [--cc-k 10000]
    [--cc-gap bounds|coils] [--options JSON] [--device gpu|cpu]
```

Problem (`sandbox/al_problem.py`): objective `QuadraticFlux` (vacuum, eval grid M = N = 25 sym); constraints
at their bounds with NO backoffs: `CoilLength` (0, L), `CoilCurvature` (−inf, κ) per node on 601 nodes,
`CoilMeanSquaredCurvature` (0, κ_MS), `CoilSetDistanceRows` (node rows, signed, gap), `PlasmaCoilDistanceField`
(spline distance field to the exact boundary, node margin), `FixSumCoilCurrent`, gauges (section 1.3).
Solver `Optimizer("lsq-auglag-composite")`, ftol = xtol = 0, defaults `hessian="secant"`, `finish_steps=5`.
`--cc-gap bounds`: gap = `node_gap_estimate(nodes, L, κ, d_cc, speed_ratio)` (2.0 planar, 1 + 0.5 XYZ) and the
field margin likewise from the bounds; `--cc-gap coils`: both computed from the coils at build (arcs).

Log lines (grep these): `DESC`, `DEVICE`, `  built <name> in <s>`, `BUILD … cc gap … pc node margin … cc active
rows at start N of K; field {…}`, `INFO {…}`, `START  bn … -> FEASIBLE/FAILS …`, `START  true: cc … pc … max L
… max |k| … linked … gap {used, needed}`, then per outer iteration
`[outer k] n inner its, opt X (gtolk Y, met|NOT met), violation V (ctolk W), T s`, one line per constraint
group (`mu min/median/max`, `max|y|`, `nonzero`, `max viol`), and `true: …`; at the end `SOLVER …`,
`START …`, `RESULT bn … L … kappa … kMS … d_cc … d_pc … linked … -> … active [...]`, `wrote …`.

Files: `outer.json` (per outer: groups, brute-force check, times), `x_outerNN.npy`, `x_last.npy`,
`result.h5` (coils), `result.json` (settings, bounds, info, solver, start/result metrics, margins, active).

Healthy run (all three on `ss_k4` looked like this): every inner solve `met`; violation falls 3–10× per
outer once μ settles; outer iterations with `0 inner its` are normal (the tolerance schedule resetting after
a μ increase); ends on `gtol` with optimality and violation < 1e-6. Unhealthy: repeated `NOT met`
(stalls), violation flat over several outers, μ climbing past ~1e6, `overflow` warnings for
`max_active_rows`.

### 4.2 `single_stage.py` (equilibrium + coils, penalty form)

See `work/SINGLE_STAGE_HANDOFF.md` section 3 and `--help`. Weighted least squares with tolerance weights
(each term's residual divided by its tolerance τ); proximal projection keeps the equilibrium solved;
boundary modes |m|, |n| ≤ k free at step k (k = kmin..kmax, `--maxiter` iterations per step). Physics
terms: B·n on the moving boundary (`sandbox/bnormal.py`, area-weighted), scale-invariant QS two-term
(`--helicity 1,4` for QH), aspect ratio band, volume band ±2% around the start, iota floor. Coil limits
are penalties (hinges) with run_one-style backoffs (d_cc ×1.025, d_pc ×1.02; arcs κ ×0.995, κ_MS ×0.99),
so the coils at the end are close to but not exactly feasible: **always refine with `run_al.py`
(step 3)**, never report single-stage coils as final. Continuing from a later step needs `--tau-qs`,
`--V0`, `--bn-norm` from the first run's log so nothing re-anchors (`work/compare/queue2.sh` shows how).

### 4.3 Dense check and verdicts

`common.evaluate(cs, eq, tree, vacuum=True)` → `bn` (⟨|B·n|⟩/⟨|B|⟩ on one field period, 72 × 96), `bn_max`,
per-coil L, κ (401 nodes + Gauss), κ_MS, d_cc (all pairs, 401 nodes), d_pc (full boundary), linked pairs.
`margins(m, P)`: value/bound (≤ 1 for L, κ, κ_MS; ≥ 1 for d_cc, d_pc). L and κ_MS reading 1.000 with
"FAILS" at the 1e-5 level is solver noise; report margins to 3–4 decimals.

### 4.4 Viewer and table

`view.py OUT.html --eq EQ --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 --h5 "NAME=PATH[@EQ]" …
--title "…" [--device cpu]`. `@EQ` draws a coilset on its OWN boundary (with a faint overlay of the
others); use it here, since the matrix has four boundaries. **Labels must not contain "="**. Always pass
`--length-mult 1.0` (the bounds file is absolute). Run it on CPU under the cap; it takes a few minutes.
`work/ss_qh/fixed_ss_eq/table.py` is the pattern for a markdown + JSON table from `common.evaluate`.

---

## 5. Traps (each one hit and measured; check for all of them)

1. **MixedCoilSet arcs.** Arc coilsets saved by the older tools are a `MixedCoilSet` (NFP 1, sym False) of
   four 1-coil `CoilSet(NFP=4, sym=True)`. `CoilSetDistanceRows` and `PlasmaCoilDistanceField` read NFP and
   sym from the top level only and crash (`IndexError: index 4 is out of bounds … size 4` in
   `CoilSetDistanceRows.build`). Fix: `repack_arcs.py IN OUT [EQ]`; it asserts the 32 curves match one to one
   (worst matched difference 0.0 m) and B·n is unchanged. Check every arc file's type before `run_al.py`
   (`type(load(p)).__name__`). The new `run_al.py` outputs are already plain `CoilSet`s; whether
   `single_stage.py`'s outputs are depends on the start (check).
2. **Arc node spacing is ~3× uneven.** The bounds-based gap (speed ratio 2) was 2.2× too small for arcs
   (2.33 mm used vs 5.23 mm needed). Always `--cc-gap coils --cc-k 40000` for arcs (gave ~1086 nodes on the ss_k4 start,
   gap ~1.06 mm, ~21k active rows, field margin ~1.0 mm). Watch `gap {used, needed}` in the `true:` lines.
3. **`max_active_rows` overflow.** XYZ used 9.4k rows at the start with the default 10000; use 20000. A
   warning names the objective if slots overflow; if you see it, rerun with a larger `--cc-k`.
4. **XYZ on the GPU.** The old XYZ single stage (H4) died twice on 2026-09-29 with
   `Failed to create cuFFT batched plan with scratch allocator` (VRAM), with and without
   `--bs-chunk 100 --jac-chunk 20`; a CPU attempt was started and killed. Try `--src-n 100 --bs-chunk 25
   --jac-chunk 5` on the GPU; if it fails before k = 1 finishes, try `--bs-chunk 10 --jac-chunk 2`, then
   `--device cpu` (slow: tell the user the estimate after the first 10 iterations before committing).
   The old `run_one.py` XYZ stage 2 also crashed with `CUDA_ERROR_ILLEGAL_ADDRESS` (one in eight runs;
   rerun once before debugging); `run_al.py` XYZ ran clean on the GPU.
5. **κ between nodes.** The optimizer puts curvature peaks between the 601 enforcement nodes: dense-check
   κ reads 1.0003–1.0019 on all three `fixed_ss_eq` results while the solver's own grid reads exactly the
   bound. Report it; do not "fix" it unasked. (A ~0.3% κ backoff would; the user has not decided.)
6. **`--xtol`.** Single-stage k = 4 stopped early on xtol 1e-6 in the old run; `--xtol 1e-12` gave another
   −15%. Use 1e-12 on every step of all three single stages.
7. **The devtools brute-force check's length is wrong for arcs** (uniform grid across corners: read
   2.932 m where the true value is 2.926 m). Trust `RESULT`/`common.evaluate` for arc lengths.
8. **Do not use `run_one.py` / `lsq-auglag` for stage 2.** It cannot converge planar coils (section 2).
9. **Equilibrium from single stage**: always pass `--vacuum`; the single-stage equilibria are vacuum.
10. **Log noise you can ignore**: `E1007 … cuda_executor.cc:1743] Could not get kernel mode driver version`,
    `xtile_compiler.cc … Fusion: …` blocks (XLA falls back; results are correct),
    `UserWarning: Unequal number of field periods for grid 1 and basis 4`.
11. **Transient `cuInit` failure** (`CUDA_ERROR_INVALID_VALUE`, JAX falls back to CPU, DESC aborts): happened
    once right after killing a run. Check `python -c "import jax; print(jax.devices())"` and relaunch.
12. **Shell hygiene.** `pkill -f <pattern>` matches your own shell if the pattern is in its command line
    (killed a launcher and the calling shell once): kill by PID. `pgrep -f "run_one.py.*X"` in a wait loop
    matches itself; anchor on the interpreter path (`^/home/singh/miniforge3/envs/desc-env2/bin/python`).
    Monitor regexes: iteration markers are space-padded (`\[it +[0-9]+\]`, `\[outer [0-9]+\]`).
13. **Killing a run** leaves no `result.h5`: for `run_al.py` use `x_last.npy`/`x_outerNN.npy` (full state;
    unpack with the problem's `obj.unpack_state`) and write `NOTE.txt`; for `single_stage.py` the last
    completed `coils_kN.h5`/`eq_kN.h5` are usable.
14. **Field build cost**: every `run_al.py` spends ~2.5 min building the plasma distance field (10 mm grid,
    spline error ~1e-6 m). It is rebuilt per run; that is expected.
15. **Memory**: `run_al.py` peaks 4–6 GB RSS (arcs with `--cc-gap coils` ~5.8 GB); single stage up to
    ~7.3 GB. Both fine under the 12 GB cap; never run two at once.

---

## 6. Deliverables

1. **`work/ss_qh/matrix/matrix.md` + `matrix.json`**: the 4 × 3 table of ⟨|B·n|⟩/⟨B⟩ (with max), plus the
   `ss_k4` reference row; for every cell also: margins (L, κ, κ_MS, d_cc, d_pc), linked count, solver
   outcome (outer/inner its, optimality, violation, wall time), and two ratios: ÷ the same row's XYZ, and
   ÷ the same column's B0 value (improvement from single stage).
2. **Boundary table**: for B0 and each single-stage boundary, Boozer QS at ρ = 0.25/0.5/0.75/1 (from the
   `k=4 physics` line; B0 from the `START physics` line), A, R0, a, iota range, V/V0, and the single-stage
   coils' own B·n before refinement (the `k=4 coils` line).
3. **The answer** in `RESULTS.md` (first section, plain language): is each representation best on its own
   boundary? By how much? Is there one boundary that is best for everything? How much QS did each boundary
   give up? State caveats (κ between nodes; single starts; one set of weights).
4. **Viewer** `work/ss_qh/matrix/view.html`: all 12 converged coilsets, each `@` its own boundary (labels
   like `arcB2 on B_planar`), title naming the experiment. Optionally per-boundary cross-section plots
   (pattern: `work/ss_qa/qsladder/analyze.py`; note it hard-codes QA helicity (1, 0); use (1, 4) for QH).
5. **Records**: the exact commands as `*.sh` scripts in `matrix/`; `NOTE.txt` for anything stopped or
   retried; update `coil_sandbox/CLAUDE.md` ("Current work") and the memory notes
   (`qh-fixed-ss-eq-stage2.md`, or a new one linked to it).

Rough budget: step 1 ≈ 35 min, step 2 ≈ 3–4 h (XYZ the uncertain part), step 3 ≈ 2 h, analysis ≈ 30 min.

---

## 7. Decisions already taken by the user, and open ones

Taken (2026-10-07): redo the arcB2 single stage from the new-solver baseline; identical single-stage settings
and weights for all three representations; new solver for all stage-2 runs; the existing `ss_k4` results
stay as a reference row.

Open (ask when they come up, with a recommendation):
* XYZ single stage on the CPU if the GPU fails (cost estimate first).
* Whether to add a ~0.3% κ backoff (currently none anywhere; keep it that way for comparability unless asked).
* Whether the off-diagonal starts should instead be the diagonal coils converted to the other
  representation (not planned; current protocol mirrors `fixed_ss_eq`).
