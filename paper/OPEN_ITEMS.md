# Open items for the experimental agent

The port (library in `desc/`, scripts in `paper/`) is complete and checked without long runs
(`PORT_STATUS.md`). What remains needs GPU runs or a research decision. Rules of the machine: one heavy
job at a time under `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0`, report progress,
confirm run settings with the user before launching, look at every coilset before calling it better.

## 1. Acceptance runs (PORT_CHECKLIST step 8), in this order

| id | what | command | compare against (poster) |
|---|---|---|---|
| 8a | design F, AL solver row | `paper/stage2.py --eq paper/inputs/references/designF/eq.h5 --bounds paper/inputs/bounds/fa_qa2_a0167.json --start paper/inputs/starts/F/F_z0_c25_p07.h5 --fix-hinge-z 0 --fold-max 60 --current-band 0.5 --vacuum --cc-gap bounds --row-n 512 --cc-k 30000 --fold-sides-from paper/inputs/starts/F/F10_fold_sides.json --maxiter 800 --out paper/runs/8a_F_AL` | gtol KKT, B.n 6.0e-5, folds 20/50/60/60/60, L and kappa active, unlinked. Differences by design: plasma rows replace the stock hard min; kappa on 1000 nodes per arc (`--curv-grid linear --curv-n 300` reproduces the poster's grid). B.n should not be worse |
| 8b | Jorge chain | `bash paper/jorge/chain.sh ladder A3 C` | R0 1.41e-2 / 17.5%, A3 6.43e-3 / 11.0%, C_bn004 5.52e-3 / 3.9%, C_bn002 5.30e-3 / 2.3%; every step unlinked, gtol |
| 8c | Jorge planar control | `bash paper/jorge/chain.sh P` | every AL subproblem converges (`outer.json` inner_converged), B.n ~3.34e-2, feasible. The regression test for the plasma rows |
| 8d | memory | 8b's C steps and an 8c cap step on the 8 GB GPU | must fit at `row_n 512` with the chunked scan; the poster OOMed |
| 8e | field lines | `bash paper/jorge/chain.sh trace`, then `paper/jorge/zoom_poincare.py` on C_bn002 | nested to start offset ~0.18 m |

## 2. Questions the runs must answer

1. **Plasma-coil gap is conservative.** The gap uses the boundary's global maximum principal curvature
   (design F 51 /m, Jorge 100 /m), so on Jorge the d_pc constraint is ~5% conservative at a 128 x 512 plasma
   grid (5.8 mm of 111 mm); the poster's C06 coils read as violating it. Options: accept and state it, refine
   the plasma grid further (cost: active rows, 24.6k on C06 at 128 x 512), or a gap from the LOCAL surface
   curvature near each row (a method change in `PlasmaCoilSetDistanceRows`). Decide after 8b/8c.
2. **Plasma-coil distance: hard min unless it stalls** (decided 2026-11-17). `stage2.py --pc-mode min` (default,
   upstream PlasmaCoilSetMinDistance, one row per coil, as the poster's `--dist hybrid`) everywhere it converges;
   `--pc-mode rows` (top-k rows: 128 x 128 half grid, 20k slots; Jorge 128 x 512, 40k) where it stalls, so far
   only the Jorge planar control. The rows cost ~4x per inner iteration on design F (the first 8a attempt).
3. **Inner Hessian.** The composite AL defaults to `{"hessian": "secant", "finish_steps": 5}` (every poster
   run). The poster session was testing how load-bearing secant is; if `gn` does as well, switch the
   default (`desc/optimize/aug_lagrangian_composite.py`) and state it in the paper.
4. **Curvature rows.** 1000 nodes per arc is ~3x the poster's curvature rows; check the cost in 8a.

## 3. Pipeline rebuilds (PORT_PLAN 9.1)

- **Design F** from `paper/inputs/near_axis/nc5_s3` (z = 0 from the start): `near_axis/to_desc.py` ->
  `free_boundary.py --repair` -> `single_stage.py --fix-hinge-z 0 --fold-max 60 --iota-min 0.36
  --repair-steps ...` (F's settings: `single_stage.py` docstring) -> `stage2.py` polish on the free-boundary
  eq -> `free_boundary.py --LMN 14` -> `metrics/make_metrics.py`. Compare with `inputs/references/designF`.
  Single stage at basis 12 ran on an A100 (40 GB); expect it not to fit 8 GB.
- **Design B** from `near_axis/o2w1_qa2s2_gg1e-1_w1e-2_w1x10` (iota 0.30 target, z = 0), same stages.
- Record every boundary repair (the scripts log it: `repairs` in result/score json, one REPAIR line each).
- **Single stage first real run**: the plasma rows now follow the moving boundary (the poster rebuilt them
  per k); the wiring was tested for 2 iterations at basis 4 only.

## 4. Experiments not yet set up

1. **precise_QA hinged at z = 0** (`compare/run.sh` uses level hinges per coil instead): snapping
   `start_arc`'s hinges to z = 0 moves them 15 cm and breaks feasibility. Needs a z = 0 lens start; the
   builder `paper/inputs/starts/F/make_z0_start.py` is a provenance copy on the sandbox imports, not yet a
   runnable script on this branch.
2. **Faces sweep fold cap**: `FOLD=deg bash paper/faces/run_sweep.sh`; the value is open (the QH arcs kept
   82-155 deg corners in RR-02). kappa_MS (in the RR-02 bounds) is not enforced in the port.
3. **Near-axis coil bounds as hard constraints** (PORT_PLAN 9 stage 1): `near_axis/run_na2.py` still
   penalizes them (`--wpen`); move to `lsq_auglag_composite`'s function API and compare one chain.
4. **Removability diagnostic** (poster `sandbox/demount.py`): not ported; port it if the paper reports
   demountability.

## 5. Housekeeping

- Full test suite on this branch (only the touched suites have run).
- Move `paper/` to `publications/<name>/` (upstream's convention, excluded from flake8) and update the
  `paper/...` paths in the scripts.
- `paper/lib/bdry_repair/_surface_fit_pr2337.py` is vendored from DESC PR #2337 (open): delete it and import
  `desc.geometry._surface_fit` once merged.
- Upstream PRs for the library pieces: the user decides later.
