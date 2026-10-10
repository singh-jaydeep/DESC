# from_axis: near-axis search for z = 0-joint coil configurations (guide for agents)

Goal: find axes (then QS equilibria) that coils with **two planar faces per coil and every hinge on z = 0** can
produce, by the near-axis single-stage method of Giuliani et al. JCP 2022 (`singlestage_QS.pdf`). Read first:
`REPORT.md` (findings), `PLAN.md` (agreed rules, decisions, phases, status). Results to look at: `viewer.html`.

## Where things are

| path | what |
|---|---|
| `na/nae.py` | first-order near-axis QS in JAX (σ equation, B_QS, ∇B_QS, helicity); validated vs pyQSC (`na/test_nae.py`) |
| `na/folded.py` | the coil class (z0 or free hinges), full coil set by symmetry, polyline Biot-Savart, ∇B, curvature, length |
| `na/run_na2.py` | **the driver**: joint axis + η̄ + coil optimization with `lsq_composite` |
| `na/sweep1.sh`, `na/sweep1b.sh` | sweep 1 (5 axis types × 4 seeds), the template for sweeps |
| `na/summarize.py` | results table (`python summarize.py "runs2/s1_*.json"`) |
| `na/export_viewer.py`, `na/viewer_template.html` | offline viewer (`python export_viewer.py ../viewer.html "runs2/*.json"`) |
| `na/to_desc.py` | NAE → DESC equilibrium + coils → `PiecewisePlanarArcCoil` CoilSets, scored with `common.evaluate`. Its fixed-boundary solve is **not** the agreed scoring (see rules) |
| `na/runs2/` | current results: `<name>.json` (init/final numbers, args), `.npz` (parameters), `.log` (solver trace) |
| `na/runs_snapshot/` | prototype runs (`run_na.py`, scipy solver; superseded) |
| `na/pylib/qsc` | pyQSC 0.1.3, vendored (not installed in desc-env / desc-dev) |

## Running

```bash
cd coil_sandbox/from_axis/na
systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
  python run_na2.py --nfp 5 --nc 3 --axis0 '1,0.08;0,-0.06' --eta0 1.6 --iota 1.30 \
    --a 0.110 --dpc 0.0974 --kmax 15.45 --Lmax 2.60 --dcc 0.04 --emax 6 --maxiter 400 --seed 2 \
    --out runs2/NAME > runs2/NAME.log 2>&1
```

* **CPU**, ~1-1.5 s/iteration, ~1.4 GB peak at 32 coils; two runs in parallel fit under a 10 GB cap.
  DESC solves (to_desc, future free-boundary scoring) go on the **GPU**, one at a time.
* Bounds: your Gil set scaled by the minor radius a of the NAE surface used for clearance:
  d_pc = 0.885 a, κ_max = 1.699 / a, L_max = 23.6 a (precise_QH: a 0.124 → 0.11, 13.7, 2.93). d_cc 0.04 so far.
* `--seed -1` = deterministic start; `--seed k` randomizes axis amplitudes (×0.5-1.3), η̄ (×0.7-1.3), coil tilts,
  and redraws until the NAE elongation is under `--emax`.
* Units: B0 = 1, axis length 2π (R0 ≈ 1), currents in μ0/4π units (amps = I / 1e-7).
* Sign convention: use **negative Z1** (`--axis0 '1,R1;0,-Z1'`) for positive ι (pyQSC helicity −1 for QH).
* To choose seeds and ι targets for a new nfp, scan ι and elongation vs (R1, Z1, η̄) with `nae.NearAxis` first
  (one line per case; see REPORT §4.2 for the targets used).

## Solver: what to use and why

* Use the branch's **`desc.optimize.least_squares_composite.lsq_composite`** (`hessian="secant"`, `x_scale="jac"`),
  fed pre-hinge rows with per-row `lo`/`hi`/`tgt` and the `isb` flags. It keeps hinges exact in the model.
* Do **not** use scipy `least_squares` with squared-hinge residuals: it stalls when bounds are active (zero
  gradient at the bound), which is what killed the first constrained QH runs.
* Distances (clearance, coil-coil) must be **smooth**: `nearest()` is a soft-min over the 8 nearest nodes (picked
  without derivatives). A plain argmin distance is kinked and stalled every QH run in sweep 1 at ftol; rerun those.
* Evaluate B/∇B over axis points with `jax.lax.map`, not `vmap`: vmap made the Jacobian 5 GB at 32 coils (OOM kill).

## Rules (agreed with the user; details in PLAN.md)

0. **Use the branch's objectives and solvers wherever DESC objects exist** (`PlasmaCoilSetDistanceRows`,
   `CoilSetDistanceRows`, `lsq-composite` / `lsq-auglag-composite`; `../../devtools/coil_auglag/GUIDE.md`). Do not
   hand-roll distance rows. In the near-axis prototype (no DESC objects) port the same row scheme.

1. Coils are primary, vacuum; the NAE is only the on-axis target and the initial guess.
2. Scoring = vacuum **free-boundary** solve with the coils and no NAE term; NAE-vs-solved is a diagnostic; Poincaré
   check against fake nested surfaces. (Not built yet.)
3. Currents constrained: each in [0, 2× uniform]. Clearance is measured to the NAE surface at r = a, not the axis.
4. Long runs and Phase 3 refinement need the user's go-ahead; report progress.

## Traps learned

* `xargs` strips quotes: use `xargs -d '\n' -P 2 -I{} bash -c '{}'` (the axis string contains `;`).
* Forcing CPU prints a harmless `CUDA_ERROR_NO_DEVICE` traceback. Monitors must not treat it as a failure, and must
  watch for runs that end without a final line (OOM kills are silent in the logs; check
  `journalctl --user | grep -i oom`).
* Fixed-size starts can be wildly infeasible (QA coils 8 mm from the surface) and strand the solver in an infeasible
  corner. `run_na2.py` sizes each coil from the NAE cross-section; keep it that way for new cases.
* Helicity is computed from the starting axis and fixed during a run (it is an integer).
* `pkill -f <pattern>` can match your own shell command (exit 144); kill by PID. Likewise a script that waits with
  `pgrep -f "bash X.sh"` matches any monitor whose command line contains that text and waits forever: wait on a PID.
* `to_desc.py` must run with `boot.setup(default="gpu")` (it does): the CPU solve took 25+ minutes.
* The coils' far parts are nearly unconstrained by the on-axis objective (the paper's flat directions): expect
  length and curvature at their bounds until a regularizer is added (REPORT §6, §8).
