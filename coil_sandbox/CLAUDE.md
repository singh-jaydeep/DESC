# coil_sandbox: one-off coil-optimization trials on arbitrary equilibria

A portable copy of the DESC used for the precise_QH/QA coil sweeps, plus a few helper scripts.
Trials are run interactively with the user; there is no pipeline to follow.

## Which DESC (the main repo, since 2026-10-07)

The DESC used is the main repo, the parent of this sandbox (`DESC2/`, branch `js/coil-auglag`).
It has everything `vendor/desc` had (arcs, kappa_MS, the distance penalties, the modified
`lsq-auglag`) plus the slack-free hinge AL (`lsq-auglag-composite`), the composite solver
(`lsq-composite`) and the smooth distance rows (`CoilSetDistanceRows`,
`PlasmaCoilSetDistanceRows`, `PlasmaCoilDistanceField`); read `../devtools/coil_auglag/GUIDE.md`.
`vendor/` is kept only to reproduce runs made before 2026-10-07: set `SANDBOX_DESC=vendor`.
Another DESC may be installed on the machine, so every script must start with:

```python
import sys; sys.path.insert(0, "sandbox")   # path from the sandbox root
import boot; boot.setup(default="gpu")     # BEFORE anything imports desc
```

`boot.setup` puts the chosen DESC first on `sys.path` and selects the device. JAX fixes its
platform at first import, and a plain `import desc` picks CPU, so the device cannot be changed
afterwards. The device comes from `--device gpu|cpu` in argv, else `SANDBOX_DEVICE`, else the
default. Every run log should show `DESC .../DESC2/desc` and `DEVICE requested=gpu obtained=gpu`;
check those lines.

`python check.py` verifies the intended DESC is the one imported, and shows the device.
`tested_env.txt` lists the package versions this was tested with. numpy 2.5 breaks
interpax/jaxtyping, so use 2.4.x. The main repo also needs `adv-jax-math` (pinned 1.2;
1.3 is installed in desc-env2 and imports fine).

`run_one.py` still drives the OLD `lsq-auglag` with the C1 distance penalties: results match the
vendored ones until the recipe is ported to `lsq-auglag-composite`.

## Machine rules (the user's laptop: WSL2, ~15 GB RAM, RTX 5070 8 GB)

* Run every heavy job under a cgroup cap, and only ONE at a time:
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 python ...`
  WSL crashes outright if the VM runs out of memory; the cap makes a run fail on its own instead.
* Do not launch long runs without the user's go-ahead. Report progress every few minutes
  (the last solver rows: constraint violation, penalty parameter, multipliers).
* Measure only what changes a decision.

## Helpers

| script | what it does | state |
|---|---|---|
| `check.py` | vendored DESC in use? required features? device? | tested |
| `make_start.py` | circles (`initialize_modular_coils`, `--nc`, `--r-over-a`) → planar-N / FourierXYZ-N / arcs (B, M, chord fit) / polar arcs (`--rep polararc`); or arcs fitted to an existing coilset (`--from-coils`, hinges by `--hinges axis`/`--phase`, polar B=2 by `--cut-angle A` or `A0,A1,A2`); dense check; saves h5 + json | tested (Helios, 2026-09-16/17) |
| `run_one.py` | the precise_QH sweep recipe for any equilibrium (`--mode stage2`), or a constraint-only `--mode feasibility --margin 0.25 [--margin-skip L] [--anchor-weight 1]` that stops once the dense check passes; `--fix-hinges`; every knob is a flag (`--help`) | tested (Helios) |
| `view.py` | offline 3D HTML viewer (all coils, closest approach, curvature peaks, linked pairs, dense-check margins); `--h5 NAME=PATH@EQ` draws a coilset on its own boundary, `--length-mult` sets one multiplier for all | tested |
| `single_stage.py` | single stage (proximal-lsq-exact on eq + coils), tolerance-weighted LSQ, boundary continuation k, `--dry` term table; see `work/SINGLE_STAGE_HANDOFF.md` | tested (precise_QA/QH, 2026-09-28/29) |
| `pad_arcs.py` | raise an arc coilset's M exactly (zero-pad, copies arc_ref/rotmat/shift) | tested |
| `sandbox/bnormal.py`, `sandbox/qs_scale_invariant.py` | B.n on a moving boundary (area-weighted); scale-invariant QS two-term (DESC PR #2304 port) | verified |
| `sandbox/common.py` | equilibrium loading (examples name / .h5 / wout / input file), bounds (null = no bound), coil and hinge helpers, the dense check `evaluate` | used by the above |
| `sandbox/convexity.py` | `CoilSignedCurvature`: smooth convexity (signed in-plane curvature ≥ 0 per planar segment; planar, arc and polar-arc coils) | verified (values, FD Jacobian) |

**Finite beta is the default.** `QuadraticFlux` and the dense check include the plasma's own field (virtual
casing, source grid 2× DESC's default); pass `--vacuum` only for vacuum equilibria. An input-file
equilibrium is unsolved, so it needs `--vacuum`.

**Convexity** (bounds file `convex_excess`): `run_one --convex-form signed` (default) is smooth; the paper
form `total` (∫|κ|ds ≤ 2π + ε) stalls the solver on nearly straight legs.

**Current work (2026-10-08): arcB2 with every hinge on z = 0 (one demountable joint plane).** Write-up and viewer in
`work/ss_qh/joint/results/` (RESULTS.md), timeline in `work/ss_qh/joint/NOTE.txt`. `run_al.py --fix-hinge-z 0` and
`single_stage.py --fix-hinge-z 0` hold every hinge's z exactly (FixParameters; z = 0 is the only single plane under
stellarator symmetry). On precise_QH the joint plane costs 2.8× (fixed eq); single stage from near-converged states
leaves 4.3× vs free arcB2 on its own boundary, because precise_QH's helical-axis excursion is untouched. Next (awaiting
the user): WISTELL-A cold start (`work/ss_qh/joint/cold/`). Traps: arc starts with z = 0 hinges must be built
(`work/ss_qh/joint/make_z0_start.py`), not fitted, then repacked (`work/ss_qh/fixed_ss_eq/repack_arcs.py`);
free-hinge cold starts can link coils in one step, so pass `run_al.py --link-mu 1e8` (initial AL penalty on the
signed per-pair coil-coil rows only; main repo 31ed99491).

**New line (2026-10-09): near-axis search (`from_axis/`).** Giuliani-style joint axis + coil optimization with the
z = 0 two-face coil class built in; QA nfp 2/3 and QH nfp 5 give the best seeds. Read `from_axis/CLAUDE.md`,
`from_axis/REPORT.md`, `from_axis/PLAN.md`; results in `from_axis/viewer.html`.

**Done (2026-10-08): the precise_QH boundary × coils matrix.** Results, caveats and deviations are in
`work/ss_qh/matrix/RESULTS.md` (tables `matrix.md`, timeline `NOTE.txt`). Planar and arcB2 are best on their own
single-stage boundary (0.31× and 0.27× their precise_QH B·n). XYZ is best on the arcB2 boundary, because its own single
stage barely moved at τ_bn 5e-3. Plan and traps: `work/ss_qh/MATRIX_HANDOFF.md`. Stage 2 uses `run_al.py` +
`sandbox/al_problem.py`, not `run_one.py` (which cannot converge planar coils; `work/ss_qh/fixed_ss_eq/RESULTS.md`).
New traps from the matrix: arcs on precise_QH need `--cc-k 80000`. Single-stage k steps can stop early on `ftol` 1e-6;
a k=4 continuation with `--ftol 1e-12 --xtol 1e-12` and frozen anchors fixes it. The FourierXYZ single stage on the GPU
needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, because cuFFT plan memory falls outside JAX's preallocation. On an
autotuner failure, use `XLA_FLAGS=--xla_gpu_autotune_level=0`.

Earlier single-stage work: `work/SINGLE_STAGE_HANDOFF.md`: results (precise_QA rigid, precise_QH not;
arcB2 vs FourierXYZ), the `single_stage.py` driver and its tolerance weighting, measured traps, open items,
and what changes for a QI example (planned after the matrix).

Earlier work: Helios (`work/helios/HANDOFF.md`): transverse B=2 arcs can only hinge at the coil diameter;
polar arcs can hinge anywhere; the vendored polar fitter's tilt bug is fixed.

Equilibria from an input file are boundary-only and unsolved. DESC's `a` for them can be ~1%
off, so pass `--a` when the bounds must be exact.

Bounds default to Gil et al. PRE 114, 025202 Table III (reactor, a = 1.7 m), scaled by a/1.7,
with the half-period length split over the coils (`paper_bounds`). Override with
`--bounds file.json` (`"scale": "absolute"` for literal values).

## What the recipe is, and why (lessons from the sweeps)

* **Distances are C¹ penalties, not minimums.** Σ max(0, 1 − d/d_min)² per coil pair
  (`CoilSetDistancePenalty`) and per coil against the plasma (`PlasmaCoilSetDistancePenalty`).
  Hard minimums kink wherever the closest pair changes, and stalled runs with trust radii of 1e-4 to 1e-6.
* **The penalties are degenerate at the bound**: their gradient vanishes there, so multipliers
  grow and Gauss-Newton's dropped term dominates. `lsq-auglag(second_order="constraints")` adds that
  term back; it cleared a measured stall. Any non-negative C¹ penalty has this property; an
  unsquared hinge is kinked at the solution instead.
* **Coil-coil over ALL pairs** (`num_neighbors=None`). A 6-nearest-by-centroid list missed a
  self-image pair 9% inside d_min. The **two-pass** evaluation (`max_active_pairs`) makes all
  pairs cheap and is exact while the active-pair count stays under the capacity (a warning
  is raised otherwise).
* **`signed=True`** on the coil-coil rows is the topology guard: a linked pair reads
  N²(1 + d*/d_min)². An unsigned distance cannot see coils passing through each other; step vetoes
  were unreliable.
* **Backoffs**: the penalty settles ~1.7% inside d_min, so d_cc is enforced ×1.025 and d_pc ×1.02.
  Arcs also get κ ×0.995, κ_MS ×0.99, and the segment sagitta 1.4κ(L/N)²/8 on the enforcement grid.
  Verdicts are always against the unbacked-off bounds.
* **Grids**: constraints on `LinearGrid(N=150)`. Source grid: composite Gauss-Legendre per arc
  (arcs are C0), `LinearGrid(N=400)` for smooth coils. d_pc plasma grid M=N=64 with symmetry
  (half the surface; correct only for stellarator-symmetric coilsets).
* **Trust-radius cap 0.5.** If every accepted step sits on the cap with model ratio ≈ 1, try
  `--max-tr inf` (it helped planarN7/arcB2 on QH, and cost 48% more time on a case that never stalled).
* **The current constraint is on the SUM** (vacuum flux is degree-2 in the currents).
* **Starts matter.** Circles can be linked or nearly touching (precise_QH at QA's r/a). Check
  `linked` and d_cc in `make_start.py`'s output; a feasibility solve with a 25% margin gave good starts.
* **Memory**: on GPU, VRAM peaked at ~1.8 GB; host RSS was 4-6 GB and grows during a run. Levers:
  `--so-chunk`, `--dcc-k/--dpc-k` (the Hessian differentiates every slot), `--qf-chunk`,
  `--dcc-chunk/--dpc-chunk`.
* **Speed** (RTX 5070, 32 coils): ~7 s/it for arcs, ~11 s/it for FourierXYZ N=6, plus ~3-4 min of
  compile per run. The GPU is mostly idle, so much of the cost is host-side.
* **Intermittent GPU fault**: one run in eight died with `CUDA_ERROR_ILLEGAL_ADDRESS` and an
  identical rerun passed. Rerun once before debugging.

The full rationale and measurements are in the sweep bundle's `fixed_configs.md`, if it is
available (`sweep_qh/`). `vendor/qh_changes_vs_sweep_910.patch` and
`vendor/desc_changes_vs_master.patch` show exactly what differs from stock DESC.
