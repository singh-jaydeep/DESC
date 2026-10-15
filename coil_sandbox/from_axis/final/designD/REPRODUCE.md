# Design D: how it was made, and how to reproduce it

Every step from a cold start to `data/`, with the exact command, the commit it ran at, the machine, and the wall time.
The repository (`DESC`, branch `js/coil-auglag`) is canonical. `drivers/` holds frozen copies of each step's scripts
at that step's commit (`drivers/MANIFEST.md`), for reading and diffing.

**To rerun a step:** check out its commit in a worktree and run there, with the same working directory:
`git worktree add /tmp/desc_<c> <c>`. The `desc/` library did not change after `f977700a0` (2026-10-09), so every
step uses the same DESC; only the `coil_sandbox/` scripts differ between commits.

## Environment

| | |
|---|---|
| laptop | WSL2 (Linux 6.18), RTX 5070 Ti (16 GB), ~15 GB RAM; conda env `desc-dev`, Python 3.13.0 |
| packages | jax / jaxlib 0.7.2 (CUDA 12 plugin), numpy 2.2.6, scipy 1.15.3, interpax 0.3.12, equinox 0.13.2, h5py 3.15.1; full list `../environment_laptop.txt` |
| DESC | `desc 0.17.3+93.g835590a09` (repo checkout; `PYTHONPATH` / `boot.setup` puts it first) |
| Modal | A100-SXM4-40GB; image = `debian_slim(python 3.13)` + `drivers/3_single_stage/coil_sandbox/from_axis/cloud/requirements.txt` (pinned to the laptop's versions) + a copy of `desc/` and `coil_sandbox/` at the job's commit |
| memory caps (laptop) | every GPU job under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`; one GPU job at a time |

Modal jobs are submitted from `coil_sandbox/from_axis/cloud`: `FA_GPU=A100-40GB modal run --detach app.py --jobs <job>`
(or `--group <group>`). Each job writes to Volume `from-axis-runs:/<job>/` with a `meta.json` holding every step's argv,
return code, minutes, commit and GPU. Harvest with `modal volume get from-axis-runs <job>/<path> <local>`.

## The chain

```
cold start (seed 2) --1--> nc5_s1 -> nc5_s2 -> nc5_s3 -> nc5_s4            near-axis, 5 coils/half period   (CPU)
nc5_s4 --2--> fb_nc5lh_b12: DESC equilibrium + coils, free-boundary basis 12      (Modal)
       --3--> ss12_nc5lh_q15b: single stage k=1..6 -> coils_k6 (= data/coils.h5) + its free-boundary rescore  (Modal)
       --4--> fb14_D: free-boundary shell at basis 14 with coils_k6 -> eq_fb (= data/eq.h5)                   (Modal)
       --5--> this package: make_figures.py                                                              (laptop)
```

### 1. Near-axis chain (CPU, commit `4e2a72420`; `run_na2.py` identical to `49557cd7c`)

Working directory `coil_sandbox/from_axis/na`. Launched 2026-10-13 18:21 local as
`printf '4\n5\n6\n' | xargs -P 2 -I{} python ncchain.py {} > runs2/ncchain.out` (D is the 5-coil chain;
`python ncchain.py 5` alone is equivalent). `ncchain.py` builds each stage from the arguments of a reference run
(`runs2/<ref>.json`, `"args"`) and logs the exact line it executes (`runs2/ncchain.out`). Each line ran under
`systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0`, with output in `runs2/nc5_sN.log`:

```
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.15;0,-0.10' --eta0 0.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed 2 --out runs2/nc5_s1 --wlen 0.001 --wbend 0.001
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wlen 0.001 --wbend 0.001 --order 2 --w2 0.001 --wgg 0.01 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s1 --out runs2/nc5_s2
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s2 --out runs2/nc5_s3
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode free --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --hzmax 0.0334 --wlevel 0.0 --tiltmax 2.0 --wplanar 0.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s3 --out runs2/nc5_s4
```

| stage | what | nit / stop | wall time (CPU) |
|---|---|---|---|
| nc5_s1 | first-order near-axis QS + coils, every hinge on z = 0, cold start (seed 2) | 600 / maxiter | 760 s |
| nc5_s2 | second order, weak weights | 194 / gtol | 461 s |
| nc5_s3 | second order, final weights (z = 0 result) | 103 / gtol | 250 s |
| nc5_s4 | hinges freed to \|z\| <= 0.0334 (0.2 a), hinge-line tilt <= 2 deg | 538 / gtol | 1269 s |

Units: B0 = 1 T on axis, axis length 2 pi (R0 ~ 1 m), bounds = the Gil et al. reactor set scaled by the near-axis
a = 0.167 (`data/bounds.json`: d_pc 0.1478, kappa 10.17, L 3.941, d_cc 0.04, absolute).
**Reproduction check:** see "Reproduction check" below.

### 2. First free-boundary state (Modal job `fb_nc5lh_b12`, commit `da44eb5ec`, A100)

Working directory `coil_sandbox/from_axis/na` inside the image:
```
python score_fb.py runs2/nc5_s4 --LMN 12 --a 0.12 --shells 2,4,6,8 --maxiter 60 --repair --out /vol/fb_nc5lh_b12/fb   # 44.7 min
python tip.py /vol/fb_nc5lh_b12/fb/eq_fb.h5 /vol/fb_nc5lh_b12/fb/coils.h5 /vol/fb_nc5lh_b12/fb/tip.png --device gpu      # 0.5 min (cwd ss/)
```
`score_fb.py` turns the near-axis state into DESC objects (a fixed-boundary equilibrium on the near-axis surface at
r = 0.12, PiecewisePlanarArcCoil coils), then runs vacuum free-boundary shells (boundary modes up to 2, 4, 6, 8 free),
repairing the boundary before every shell after the first (`repair_lib.py` -> the `bdry_repair` package). Local copy: `coil_sandbox/from_axis/relaxed/nc5lh/`.

### 3. Single stage (Modal job `ss12_nc5lh_q15b`, commit `8aa3ed22f`, A100, started 2026-10-14 02:34 UTC)

Working directory `coil_sandbox/from_axis/ss`; full record `coil_sandbox/from_axis/ss/nc5lh/meta.json`.
```
python prep.py /vol/fb_nc5lh_b12/fb /vol/ss12_nc5lh_q15b/start --LMN 12                                       # 0.2 min
python ../../single_stage.py --eq /vol/ss12_nc5lh_q15b/start/eq0_b12.h5 --start /vol/ss12_nc5lh_q15b/start/coils0.h5 \
  --out /vol/ss12_nc5lh_q15b/ss --bounds ../../bounds/fa_qa2_a0167.json --hinge-zmax 0.0334 --hinge-tilt-max 2.0 \
  --dist rows --rows-select 1.6 --cc-rows 24000 --pc-rows 12000 --pc-gap 0.005 --plasma-m 40 --plasma-n 40 \
  --helicity 1,0 --qs-form boozer --qs-rho 0.25,0.5,0.75,1.0 --A-min 6.8 --A-max 7.6 --iota-min 0.35 --tau-bn 5e-4 \
  --tau-qs-rel 0.15 --length-mult 0.999 --src-gl 64 --force-chunk 400 --method lsq-composite --hessian secant \
  --solve-tol 1e-6 --kmin 1 --kmax 6 --maxiter 50 --track-steps --ckpt-every 1 --auto-resume --repair-steps        # 46.3 min
python fb_rescore.py /vol/ss12_nc5lh_q15b/ss/eq_k6.h5 /vol/ss12_nc5lh_q15b/ss/coils_k6.h5 /vol/ss12_nc5lh_q15b/fb --shells 4,8 --repair   # 9.7 min
python tip.py /vol/ss12_nc5lh_q15b/fb/eq_fb.h5 /vol/ss12_nc5lh_q15b/ss/coils_k6.h5 /vol/ss12_nc5lh_q15b/fb/tip.png --device gpu       # 0.4 min
```
The single stage optimizes the equilibrium (fixed-boundary basis 12, proximal) and the coils together: B.n, Boozer QS
at four surfaces (tolerance 15% relative), aspect ratio in [6.8, 7.6], iota >= 0.35, coil bounds via the branch's
`CoilSetDistanceRows` / `PlasmaCoilSetDistanceRows` and `lsq-composite`, hinge band and tilt as bounded terms; boundary
continuation k = 1..6. **The coils `ss/coils_k6.h5` are this package's `data/coils.h5`** (md5
`a84d134598c78d0a893102ec21b748d8`). Harvested to `coil_sandbox/from_axis/ss/nc5lh/`. The earlier job
`ss12_nc5lh_q15` is invalid (coil-coil rows overflowed) and is not part of this chain.

### 4. Final equilibrium: basis-14 free boundary (Modal job `fb14_D`, group `final_fb14`, commit `29f1c5e36`, A100)

Working directory `coil_sandbox/from_axis/ss`, 2026-10-14 18:04 local:
```
python fb_rescore.py nc5lh/fb/eq_fb.h5 nc5lh/ss/coils_k6.h5 /vol/fb14_D/fb --LMN 14 --shells 14 --maxiter 60 --solve-tol 1e-8 --chunk 400   # 39.1 min
python tip.py /vol/fb14_D/fb/eq_fb.h5 nc5lh/ss/coils_k6.h5 /vol/fb14_D/fb/tip.png --device gpu                                         # 0.4 min
```
Raises step 3's free-boundary state to L = M = N = 14 (fixed-boundary re-solve), then one vacuum free-boundary shell
with every boundary mode free and the coils fixed, inner force-balance tolerance 1e-8: B.n rms 9.7e-4 -> 1.6e-5
(stopped at maxiter, still improving; no "bad approximation" stall). **`/vol/fb14_D/fb/eq_fb.h5` is `data/eq.h5`**
(md5 `8cd5fe39b7f237f3c03a0a2171f8855c`; local copy `coil_sandbox/from_axis/ss/cloud_runs/fb14_D/`).

Why basis 14: the basis-12 vacuum equilibria of steps 2-3 do not reproduce the coils' interior field (iota off by
0.01-0.07, axis 1.5 mm off the traced axis, interior |B| error 0.2-0.5%, the size of the QS breaking). At basis 14
the traced iota matches the equilibrium to 6e-4 and the axis to 0.04 mm; re-solving this boundary at basis 16 changes
QS and eps_eff by 1-4% (`coil_sandbox/from_axis/final_work/NOTES.md`, sections 1a-1e).

### 5. This package (laptop, commit `835590a09` + this folder)

```
cd coil_sandbox/from_axis/final/designD
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python make_figures.py      # all stages, ~30 min
```
Stage times on the RTX 5070 Ti: eq 0.6 min, coils 0.5 min, trace (dense) 8-9 min, transport 0.9 min, figures 0.4 min.
QS uses the literature Boozer convention directly (the `qs_metrics.py` + `qs_booz_fix.py` pair of the working
scripts gives identical numbers).

## Reproduction check

**Step 1 rerun from scratch, 2026-10-15** (laptop CPU, a clean worktree at `4e2a72420`, `python ncchain.py 5` in
`coil_sandbox/from_axis/na`): all four stages reproduce the saved runs **bit for bit**. Every final scalar in the jsons
is equal, and the saved parameter vectors (`.npz` `p`) differ by exactly 0. Only the wall times differ: 592 / 348 /
195 / 1004 s against 760 / 461 / 250 / 1269 s, because the original ran two chains in parallel. The rerun's jsons and
its command log are in `repro_check/`.

Steps 2-4 (GPU) were not rerun: this session's Modal budget was used up, and the laptop needs ~2 h per basis-14
full-mode shell. `make_figures.py` (step 5) recomputes every metric from `data/` and reproduces the numbers the
working notes quoted to all printed digits: QS, B.n, coil margins, traced iota and Gamma_c.

Tolerance for reruns: GPU results need not match bit for bit (reduction order, XLA autotuning). A rerun of steps 2-4
should give B.n, QS, iota and coil margins within ~1% of `metrics.json` once converged; a near-axis rerun (step 1, CPU)
should reproduce its json's final numbers to the printed precision.

## Known gaps in the record

* None for D's chain: every command, commit and wall time above is from the job records (`meta.json`) or the logged
  command lines (`runs2/ncchain.out`). Detailed sourcing: `../PROVENANCE_DETAIL.md` (copy of `final_work/provenance_reconstruction.md`).
