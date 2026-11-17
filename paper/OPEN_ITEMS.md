# Experimental plan (open items after the port)

The port (library in `desc/`, scripts in `paper/`) is complete (`PORT_SUMMARY.md`, `PORT_STATUS.md`). What
remains are the paper's experiments and method decisions: they need GPU runs or the user's call. Rules of the machine: one heavy
job at a time under `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0`, report progress,
confirm run settings with the user before launching, look at every coilset before calling it better.

## 1. First experiments: the Jorge chain (also the end-to-end check of the Jorge-specific pieces)

Design F (acceptance run 8a) is done: `findings/8a_basins.md`. The port is correct; stage 2 from F's lens start
has two KKT optima 15x apart in B.n, chosen by the variable scaling. Expect the same kind of path dependence below,
so judge each step by KKT, feasibility on the dense check, no links and tidy coils, and treat B.n differences
from the poster as possible basin differences, not port errors.

| step | command | poster reference |
|---|---|---|
| Jorge ladder, A3, cap | `bash paper/jorge/chain.sh ladder A3 C` | R0 1.41e-2 / 17.5%, A3 6.43e-3 / 11.0%, C_bn004 5.52e-3 / 3.9%, C_bn002 5.30e-3 / 2.3% |
| planar control | `bash paper/jorge/chain.sh P` | every AL subproblem converges, B.n ~3.34e-2, feasible (plasma rows, where the hard minimum stalled) |
| memory | the C steps and a P cap step on the 8 GB GPU | must fit at `row_n 512` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` |
| field lines | `bash paper/jorge/chain.sh trace`, `paper/jorge/zoom_poincare.py` on C_bn002 | nested to start offset ~0.18 m |

Recommended: add the link guard to every chain step (`--exp-pair-rows signed --link-mu 1e8` in `BASE` of
`jorge/chain.sh`); two of six 8a runs without it linked a coil pair in one step. Whether the guard is part of
the paper's method is the user's decision.

## 2. Method questions

1. **Basins and variable scaling (8a).** `x_scale="jac"` takes column norms over all constraint rows, active or
   not, so rows with no constraint role steer the path. Options: report best-of-several starts/scalings with the
   spread (cheap, honest); a scaling from the objective and active rows only (reproducible; a method change);
   both. A small multistart on design F would show how many basins there are.
2. **Plasma-coil gap is conservative.** The rows' gap uses the boundary's global maximum principal curvature
   (design F 51 /m, Jorge 100 /m): ~5% of d_pc on Jorge at a 128 x 512 plasma grid. Only matters where the rows
   are used (`--pc-mode rows`). Options: accept and state it, refine the grid, or a gap from the local curvature.
3. **Plasma-coil distance: hard min unless it stalls** (decided 2026-11-17; `stage2.py --pc-mode`).
4. **Inner Hessian.** Default `{"hessian": "secant", "finish_steps": 5}` (every poster run). If `gn` does as well,
   switch the default and state it.
5. **Curvature rows.** 1000 nodes per arc (default) cost ~1.3x the poster's `N=300` grid on 8a (572 s vs 448 s).

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
- `paper/lib/bdry_repair/_surface_fit_pr2337.py` is vendored from DESC PR #2337 (open): delete it and import
  `desc.geometry._surface_fit` once merged.
- Upstream PRs for the library pieces: the user decides later.
