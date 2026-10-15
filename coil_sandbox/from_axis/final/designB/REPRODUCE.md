# Design B: how it was made, and how to reproduce it

Every step from the near-axis reference state to `data/`, with the exact command, the commit it ran at, the machine,
and the wall time. The repository (`DESC`, branch `js/coil-auglag`) is canonical. `drivers/` holds frozen copies of
each step's scripts at that step's commit (`drivers/MANIFEST.md`), for reading and diffing.

**To rerun a step:** check out its commit in a worktree and run there, with the same working directory:
`git worktree add /tmp/desc_<c> <c>`. The `desc/` library did not change after `f977700a0` (2026-10-09), so every
step uses the same DESC; only the `coil_sandbox/` scripts differ between commits. **Several of B's early steps ran
from uncommitted working copies** (marked below, with the closest commit). B's chain is less clean than D's; the
last section lists every gap.

## Environment

| | |
|---|---|
| laptop | WSL2 (Linux 6.18), RTX 5070 Ti (16 GB), ~15 GB RAM; conda env `desc-dev`, Python 3.13.0 |
| packages | jax / jaxlib 0.7.2 (CUDA 12 plugin), numpy 2.2.6, scipy 1.15.3, interpax 0.3.12, equinox 0.13.2, h5py 3.15.1; full list `../environment_laptop.txt` |
| DESC | `desc 0.17.3+93.g835590a09` (repo checkout; `boot.setup` puts it first on `sys.path`) |
| Modal | A100-SXM4-40GB; image = `debian_slim(python 3.13)` + `drivers/3_single_stage/coil_sandbox/from_axis/cloud/requirements.txt` (pinned to the laptop's versions) + a copy of `desc/` and `coil_sandbox/` at the job's commit |
| memory caps (laptop) | every GPU job under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`; one GPU job at a time. Basis-14 full-mode free-boundary shells need `XLA_PYTHON_CLIENT_MEM_FRACTION=0.93` (the default 0.75 runs out of memory) |

Modal: submitted from `coil_sandbox/from_axis/cloud` with `FA_GPU=A100-40GB modal run --detach app.py --jobs <job>`;
output on Volume `from-axis-runs:/<job>/`, with `meta.json` (every step's argv, return code, minutes, commit, GPU).

## The chain

```
o2w1_qa2s2_gg1e-1_w1e-2_w1x10 (near-axis reference, saved state)
  --1--> li_qa2_iota0.30                                   near-axis, iota target 0.30, every hinge on z = 0 (CPU)
  --2--> _fb_a0.12_L12 (shells 2,4,6) --repair k6--> _L12_rep (shell 8)   free boundary, basis 12        (laptop GPU)
  --3--> ss12_li030_q15: single stage k=1..6 -> coils_k6 (polished in step 4 -> data/coils.h5)           (Modal)
  --3b-> fb_norepair_local (free-boundary shells 4,8) --bdry_repair--> repaired/eq.h5                   (laptop)
  --4--> basis-14 re-solve -> free-boundary shell (basis 14) -> coil polish x2 -> free-boundary shell -> data/eq.h5 (laptop)
  --5--> this package: make_figures.py                                                                  (laptop)
```

### 0. Near-axis reference state (not a step of this package; its saved state is the anchor)

B starts from `coil_sandbox/from_axis/na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10.{json,npz}` (committed). Its lineage is
`s1_qa2_z0_s2 -> reg_qa2s2_w0 -> _w1e-5 -> _w1e-4 -> _w1e-3 -> elong_qa2s2_e5 -> o2_qa2s2_gg1e-3_w1e-4 ->
o2_qa2s2_gg1e-2_w1e-3 -> o2w1_...` (each `runs2/<name>.json` stores its full `run_na2.py` arguments under `"args"`,
the parent under `args.init`; frozen launchers in `drivers/0_near_axis_reference/`). **The first run, `s1_qa2_z0_s2`,
ran before the first commit, with the old non-smooth nearest-node distance rows, so no commit reproduces it
exactly**; the later lineage runs ran from uncommitted code whose closest commit is `846ece7cb` (for `o2w1` probably
identical). Reproduce from the saved `.npz`, not from a cold start. Wall times: 212, 517, 87, 47, 184, 195, 414, 440,
654 s (CPU). Explicit commands: `../PROVENANCE_DETAIL.md` (copy of `final_work/provenance_reconstruction.md`), section 1.

### 1. Low-iota near-axis state (CPU; uncommitted, closest commit `e5b0a9ad2`)

Launched 2026-10-12 15:58 local with `./lowiota.sh` (from `na/`); the line it ran for iota 0.30, equivalent explicit
form from `runs2/li_qa2_iota0.30.json` `"args"`:
```
cd coil_sandbox/from_axis/na
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 \
  --mode z0 --iota 0.3 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 \
  --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --hzmax 0.0 --wlevel 0.0 --zmax 0.0 \
  --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 \
  --maxiter 600 --hessian secant --seed -1 --init runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10 --out runs2/li_qa2_iota0.30
```
208 iterations (ftol), 308 s. The working copy was `846ece7cb` plus `--hzmax`/`--wlevel`; `e5b0a9ad2` adds only
`--wplanar` (default 0, no effect).

### 2. Free boundary at basis 12 (laptop GPU; three sub-steps)

**2a.** 2026-10-12 18:45 local, `./fbscreen.sh li_qa2_iota0.30 ...` (from `na/`), which ran
```
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python score_fb.py runs2/li_qa2_iota0.30 \
  --LMN 12 --a 0.12 --shells 2,4,6,8 --maxiter 60 --out runs2/li_qa2_iota0.30_fb_a0.12_L12
```
with an uncommitted `score_fb.py` (two edits setting the ForceBalance chunk to 200; they became `--chunk`, default 200,
in `a322c7f22`, which is the equivalent code). Shells 2, 4, 6 finished at 686, 2019, 3747 s; the run was killed in
shell 8 (~4280 s) because the boundary creased (B.n max 13%). Output used: `eq_fb_k6.h5`. No `score.json`.

**2b.** 19:57 local, from `ss/`, the old fixed-label boundary refit (commit `89f5a39e4`), ~4 min (moved the surface
0.14 mm mean, 1.73 mm max):
```
D=../na/runs2/li_qa2_iota0.30_fb_a0.12_L12
python repair_eq.py $D/eq_fb_k6.h5 $D/coils.h5 $D/eq_fb_k6_repaired.h5
```

**2c.** 20:01 local, from `na/`, commit `a322c7f22`, 1268 s (shell 8: 38 iterations):
```
python score_fb.py runs2/li_qa2_iota0.30 --LMN 12 --a 0.12 --shells 8 --maxiter 60 \
  --resume runs2/li_qa2_iota0.30_fb_a0.12_L12/eq_fb_k6_repaired.h5 --out runs2/li_qa2_iota0.30_fb_a0.12_L12_rep
```
(The handoff's "shells 2-8 with `--repair`" is not what happened; `score_fb.py --repair` repairs before every shell.)

### 3. Single stage (Modal job `ss12_li030_q15`, commit `6993369dc`, A100, 2026-10-13 01:16 UTC)

Working directory `coil_sandbox/from_axis/ss`; full record `coil_sandbox/from_axis/ss/li030/meta.json`.
```
python prep.py ../na/runs2/li_qa2_iota0.30_fb_a0.12_L12_rep /vol/ss12_li030_q15/start --LMN 12                     # 0.3 min
python ../../single_stage.py --eq /vol/ss12_li030_q15/start/eq0_b12.h5 --start /vol/ss12_li030_q15/start/coils0.h5 \
  --out /vol/ss12_li030_q15/ss --bounds ../../bounds/fa_qa2_a0167.json --fix-hinge-z 0 --dist rows --rows-select 1.6 \
  --cc-rows 12000 --pc-rows 12000 --pc-gap 0.005 --plasma-m 40 --plasma-n 40 --helicity 1,0 --qs-form boozer \
  --qs-rho 0.25,0.5,0.75,1.0 --A-min 6.6 --A-max 7.6 --iota-min 0.22 --tau-bn 5e-4 --tau-qs-rel 0.15 --length-mult 0.999 \
  --src-gl 64 --force-chunk 400 --method lsq-composite --hessian secant --solve-tol 1e-6 --kmin 1 --kmax 6 --maxiter 50 \
  --track-steps --ckpt-every 1 --auto-resume --repair-steps                                                    # 55.9 min
python fb_rescore.py /vol/ss12_li030_q15/ss/eq_k6.h5 /vol/ss12_li030_q15/ss/coils_k6.h5 /vol/ss12_li030_q15/fb --shells 4,8 --repair   # 18.8 min (superseded)
```
Single stage = equilibrium (fixed boundary, basis 12, proximal) and coils optimized together: B.n, Boozer QS at four
surfaces, A in [6.6, 7.6], iota >= 0.22, coil bounds via `CoilSetDistanceRows` / `PlasmaCoilSetDistanceRows` and
`lsq-composite`, every hinge held at z = 0 exactly; boundary continuation k = 1..6. `--repair-steps` used the old
fixed-label `repair_lib.py` of this commit. Coils `ss/coils_k6.h5` -> polished in step 4 -> `data/coils.h5`. The
job's own free-boundary rescore (`fb_repair_modal/`) repaired after every shell and is superseded by 3b.

### 3b. Free-boundary rescore and repair (laptop)

2026-10-12 22:24 local, from `ss/`, commit `6993369dc` (shell 4 at 384 s, shell 8 at 1465 s):
```
R=cloud_runs/ss12_li030_q15
python fb_rescore.py $R/ss/eq_k6.h5 $R/ss/coils_k6.h5 $R/fb_norepair_local --shells 4,8      # -> ss/li030/fb_norepair_local
```
2026-10-13 09:57 local, from `ss/`, the `bdry_repair` package (ran uncommitted; committed 5 min later as `d5c8966e2`),
~4.5 min, defaults `--rule chordal --max-steps 8 --chunk 200`:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 python repair_eq.py li030/fb_norepair_local/eq_fb.h5 li030/ss/coils_k6.h5 <out>.h5   # -> ss/li030/repaired/eq.h5
```

### 4. Final state (laptop, 2026-10-14/15; code unchanged from the run times to `835590a09`)

Why: the basis-12 vacuum equilibria above do not reproduce the coils' interior field (iota off by up to 0.02, axis
5.8 mm off the traced axis for B), so B was moved to basis 14 and rescored there (`final_work/NOTES.md` 1a-1h).

**4a. Basis-14 re-solve of the repaired state** (from `ss/`, `final_work/res/run_B.sh`; `resladder.py`, first
committed in `29f1c5e36` with no later change; basis 14 took ~23 min):
```
python resladder.py li030/repaired/eq.h5 ../final_work/res/B_k6rep --lmn 12,14,16          # uses B_k6rep/eq_b14.h5
```
**4b. Free-boundary shell at basis 14** (`final_work/res/overnight_B.sh`, then stopped after the first shell by
`replan_B.sh`; 8519 s for 20 iterations):
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 python fb_rescore.py ../final_work/res/B_k6rep/eq_b14.h5 li030/ss/coils_k6.h5 \
  ../final_work/res/fb14_Bk6 --LMN 14 --shells 14 --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare
```
B.n rms 1.04e-3 -> 1.06e-5. Coils still 1.08 mm over the length bound (the single stage's length term is soft).
**k=6 was chosen over k=5** after giving k=5 the same treatment (`fb14_Bk5`: 4-8% worse QS at every radius).

**4c. Coil polish, two runs** (from `coil_sandbox/`, `run_al.py` = stage 2 with the equilibrium fixed, hard bounds
through `lsq-auglag-composite`, hinges at z = 0; `final_work/res/polish_B.sh`):
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 python run_al.py --eq from_axis/final_work/res/fb14_Bk6/eq_fb.h5 \
  --start from_axis/ss/li030/ss/coils_k6.h5 --bounds bounds/fa_qa2_a0167.json \
  --out from_axis/final_work/res/polish_Bk6/run --vacuum --fix-hinge-z 0 --maxiter 200                       # 17.8 min
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 python run_al.py --eq from_axis/final_work/res/fb14_Bk6/eq_fb.h5 \
  --start from_axis/final_work/res/polish_Bk6/run/result.h5 --bounds from_axis/final_work/res/polish_Bk6/bounds_backoff.json \
  --out from_axis/final_work/res/polish_Bk6/run2 --vacuum --fix-hinge-z 0 --maxiter 200                      # 17.1 min
```
Run 1 left kappa 0.07% over in the dense check (the constraint rows sit on a coarser node grid); run 2 enforces
L x 0.9999 and kappa x 0.995 (`bounds_backoff.json`) and is feasible against the original bounds. **`run2/result.h5`
is `data/coils.h5`.**

**4d. Free-boundary shell with the polished coils** (from `ss/`, 5083 s):
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 python fb_rescore.py ../final_work/res/fb14_Bk6/eq_fb.h5 \
  ../../from_axis/final_work/res/polish_Bk6/run2/result.h5 ../final_work/res/fb14_Bk6p --LMN 14 --shells 14 \
  --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare
```
B.n rms 1.4e-4 -> 8.7e-6. **`fb14_Bk6p/eq_fb.h5` is `data/eq.h5`.**

### 5. This package

```
cd coil_sandbox/from_axis/final/designB
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python make_figures.py      # all stages
```
Stage times on the RTX 5070 Ti: eq 0.9 min, coils 0.7 min, trace (dense) 9.4 min, transport 1.2 min, figures 0.3 min.

## Reproduction check

No step of B's chain was rerun. D's near-axis chain, rerun from scratch, reproduced bit for bit
(`../designD/REPRODUCE.md`), so the CPU near-axis driver is deterministic. B's own near-axis step 1 ran from
uncommitted code, so a rerun at `e5b0a9ad2` is expected to match but has not been checked. `make_figures.py` (step 5)
recomputes every metric from `data/` and reproduces the numbers the working notes quoted to all printed digits: QS,
B.n, coil margins, traced iota and Gamma_c.

Tolerance for reruns: GPU results need not match bit for bit. A rerun of a GPU step should give B.n, QS, iota and coil
margins within ~1% of `metrics.json` once converged.

## Known gaps in the record

1. `s1_qa2_z0_s2` (the root of the reference lineage) cannot be reproduced from any commit; start from the saved state.
2. The reference lineage runs and step 1 ran from uncommitted code (closest commits given; differences are flags not
   used or with no-op defaults).
3. Step 2a ran from an uncommitted `score_fb.py` (functionally `a322c7f22` with `--chunk 200`, not byte-identical); its
   fold check never ran and it has no `score.json`. `tip.py` used in 2c was not yet committed.
4. Step 3b's repair and step 4a ran a few minutes before their commits (`d5c8966e2`, `29f1c5e36`); no edits were
   found in between.
5. Step 4b was stopped by hand after its first shell (the replan); the polished coils were never put through a
   single stage, only through the fixed-equilibrium polish and a free-boundary shell.

Detailed sourcing of every line: `../PROVENANCE_DETAIL.md` (copy of `final_work/provenance_reconstruction.md`).
