# Provenance reconstruction: laptop steps of designs B and D (for REPRODUCE.md)

Written 2026-10-15. Read-only reconstruction; nothing was rerun.

**Sources.** `na/runs2/<name>.json` (`"args"`), the `.log` files, the scripts in `na/` and `ss/`, git history, and the Claude
session transcript `~/.claude/projects/-home-jay-Documents-desc-dev-DESC/59468c83-5694-4cd1-8df3-e0855fb2fcf6.jsonl`
(T = its tool_use timestamp, in UTC). Local time is EDT = UTC - 4, and file mtimes and commit dates are local. The
other transcript (`92e48db5…`, which ends 2026-10-12 14:47) contains none of these steps.

**Notes on the "args" field.** The `"init"` key at the top level of each json holds the *initial-state metrics*. The
parent run is `args.init`. The `"args"` field holds every argparse value, defaults included, so the explicit
`run_na2.py` commands below can carry more flags than were typed. They are equivalent because every flag has a value.
`run_na2.py` has no `store_true` flags. Unless a row says otherwise, all `run_na2.py` runs ran on CPU under
`systemd-run --user --scope -q -p MemoryMax=… -p MemorySwapMax=0`, and the `time_s` values come from each json's
`final`.

**DESC library.** `git log -- desc/` shows no change after `f977700a0` (2026-10-09 17:02). Only scripts under
`coil_sandbox/` changed during this work.

---

## 1. Near-axis reference lineage (QA nfp 2, z = 0). Working dir: `coil_sandbox/from_axis/na`

The lineage is longer than `FINALIZE_HANDOFF.md` says. Walking `args.init` gives:
`s1_qa2_z0_s2 → reg_qa2s2_w0 → reg_qa2s2_w1e-5 → reg_qa2s2_w1e-4 → reg_qa2s2_w1e-3 → elong_qa2s2_e5 →
o2_qa2s2_gg1e-3_w1e-4 → o2_qa2s2_gg1e-2_w1e-3 → o2w1_qa2s2_gg1e-1_w1e-2_w1x10`.

### 1a. What was actually launched (transcript)
| run(s) | launcher as typed (cwd `na/`) | T (UTC) | code state |
|---|---|---|---|
| `s1_qa2_z0_s2` | `nohup systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash sweep1.sh > runs2/sweep1.out 2>&1 &`. The line it ran is in `sweep1_cmds.txt` (below). | 2026-10-09T17:11:34 | **uncommitted** (before the first commit `15fa3e65a`, 17:02 local). It also predates the soft-min distances (added 18:52Z), so **no commit reproduces it exactly**. The closest is `15fa3e65a`, which has the soft-min distances. |
| `reg_qa2s2_w{0,1e-5,1e-4,1e-3}` | `nohup systemd-run … -p MemoryMax=10G … bash reg1.sh > runs2/reg1.out 2>&1 &` (`reg1.sh` as in git: `for w in 0 1e-5 1e-4 1e-3 … cont.py $prev $out --wlen $w --wbend $w --maxiter 600`) | 2026-10-09T18:52:46 | uncommitted; closest `15fa3e65a` |
| `elong_qa2s2_e5` | `nohup systemd-run … bash reg2.sh > runs2/reg2.out 2>&1 &` (`cont.py reg_qa2s2_w1e-3 … --emax 5 --wlen 1e-3 --wbend 1e-3 --maxiter 600`) | 2026-10-09T19:13:35 | uncommitted; closest `15fa3e65a` |
| `o2_qa2s2_gg1e-3_w1e-4`, `o2_qa2s2_gg1e-2_w1e-3` | `nohup systemd-run … -p MemoryMax=11G … bash o2.sh` (`TAG=qa2s2 chain runs2/elong_qa2s2_e5 qa2s2 1.20`; pairs `1e-3 1e-4`, `1e-2 1e-3`, …) | 2026-10-09T22:13:49 | uncommitted (`--order 2` was added 22:11Z); closest `846ece7cb` |
| `o2w1_qa2s2_gg1e-1_w1e-2_w1x10` | `nohup systemd-run … -p MemoryMax=8G … bash w1test.sh` (`cont.py runs2/o2_qa2s2_gg1e-2_w1e-3 $out --order 2 --wgg 1e-1 --w2 1e-2 --w1 10 --maxiter 600`) | 2026-10-10T00:11:05 | uncommitted (`--w1` was added 00:10:56Z). `846ece7cb` was committed at 02:36Z with no later edit to `run_na2.py` in between, so it is the closest commit and probably identical. |

`sweep1_cmds.txt` line for s1:
```
python run_na2.py --nfp 2 --nc 4 --axis0 '1,0.15;0,-0.10' --eta0 0.6 --iota 0.42 --a 0.167 --dpc 0.147795 --kmax 10.173652694610778 --Lmax 3.9412000000000003 --dcc 0.04 --emax 6 --maxiter 400 --seed 2 --out runs2/s1_qa2_z0_s2 > runs2/s1_qa2_z0_s2.log 2>&1
```

### 1b. Equivalent explicit commands (from each json's `args`)
| run | finished (local) | nit | time_s |
|---|---|---|---|
| s1_qa2_z0_s2 | 10-09 13:18 | 400 (maxiter) | 212 |
| reg_qa2s2_w0 | 10-09 15:01 | 600 (maxiter) | 517 |
| reg_qa2s2_w1e-5 | 15:03 | 102 | 87 |
| reg_qa2s2_w1e-4 | 15:04 | 54 | 47 |
| reg_qa2s2_w1e-3 | 15:07 | 219 | 184 |
| elong_qa2s2_e5 | 15:17 | 226 | 195 |
| o2_qa2s2_gg1e-3_w1e-4 | 18:20 | 211 | 414 |
| o2_qa2s2_gg1e-2_w1e-3 | 18:28 | 216 | 440 |
| o2w1_qa2s2_gg1e-1_w1e-2_w1x10 | 20:22 | 278 (gtol) | 654 |

```
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.15;0,-0.10' --eta0 0.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 6.0 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 400 --hessian secant --seed 2 --out runs2/s1_qa2_z0_s2
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 6.0 --wlen 0.0 --wbend 0.0 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/s1_qa2_z0_s2 --out runs2/reg_qa2s2_w0
python run_na2.py ... (same) --wlen 1e-05 --wbend 1e-05 ... --init runs2/reg_qa2s2_w0 --out runs2/reg_qa2s2_w1e-5
python run_na2.py ... (same) --wlen 0.0001 --wbend 0.0001 ... --init runs2/reg_qa2s2_w1e-5 --out runs2/reg_qa2s2_w1e-4
python run_na2.py ... (same) --wlen 0.001 --wbend 0.001 ... --init runs2/reg_qa2s2_w1e-4 --out runs2/reg_qa2s2_w1e-3
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wlen 0.001 --wbend 0.001 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/reg_qa2s2_w1e-3 --out runs2/elong_qa2s2_e5
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wlen 0.001 --wbend 0.001 --order 2 --w2 0.0001 --wgg 0.001 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/elong_qa2s2_e5 --out runs2/o2_qa2s2_gg1e-3_w1e-4
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wlen 0.001 --wbend 0.001 --order 2 --w2 0.001 --wgg 0.01 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/o2_qa2s2_gg1e-3_w1e-4 --out runs2/o2_qa2s2_gg1e-2_w1e-3
python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/o2_qa2s2_gg1e-2_w1e-3 --out runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10
```
In the three `reg_*` rows, "(same)" means the `reg_qa2s2_w0` line with only `--wlen`, `--wbend` and `--init`/`--out`
changed. Each json records the full line.

---

## 2. Design B, near-axis: `li_qa2_iota0.30`. Working dir: `na/`

* **Launched:** T 2026-10-12T19:58:46Z (15:58 local). The session wrote `lowiota.sh` and ran
  `chmod +x lowiota.sh; ./lowiota.sh`. The script `cd`s to `na/` itself.
* **What the script runs for I = 0.30** (it builds the line with `cont.py`; derived by hand from `cont.py`'s logic,
  not executed):
```
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c \
 "python run_na2.py --init runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10 --out runs2/li_qa2_iota0.30 --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --hessian secant --iota 0.30 --maxiter 600" > runs2/li_qa2_iota0.30.log 2>&1
```
* **Explicit equivalent** (from the json's `args`):
  `python run_na2.py --nfp 2 --nc 4 --K 5 --Kax 5 --mode z0 --iota 0.3 --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --hzmax 0.0 --wlevel 0.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed -1 --init runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10 --out runs2/li_qa2_iota0.30`
* **Result:** nit 208 (ftol), time_s **308** (CPU), finished 16:04 local.
* **Code state:** uncommitted. It was `846ece7cb` plus the `--hzmax`/`--wlevel` edit (19:06Z). The `--wplanar` edit at
  20:00:23Z came after this process had started. The closest commit is `e5b0a9ad2` (2026-10-12 17:54), which also has
  `--wplanar`; its default of 0 leaves the result unchanged.

---

## 3. Design B, free boundary at basis 12 (laptop GPU, RTX 5070 Ti). Working dir: `na/`

### 3a. `runs2/li_qa2_iota0.30_fb_a0.12_L12` (shells 2, 4, 6; killed during shell 8) — RECOVERED
* **Launched:** T 2026-10-12T22:45:09Z (18:45:09 local) with
  `./fbscreen.sh li_qa2_iota0.30 li_qa2_iota0.20 lv_qa2_h0.05`. A first launch at 22:44:14Z was killed and its output
  deleted 50 s later, after the patch described below. `fbscreen.sh` is committed in `a322c7f22` unchanged. For this
  run it executes:
```
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  python score_fb.py runs2/li_qa2_iota0.30 --LMN 12 --a 0.12 --shells 2,4,6,8 --maxiter 60 \
  --out runs2/li_qa2_iota0.30_fb_a0.12_L12 > runs2/li_qa2_iota0.30_fb_a0.12_L12.log 2>&1
```
* **Code state:** `score_fb.py` as of `d519b8955`/`e5b0a9ad2`, plus two uncommitted edits:
  - `ForceBalance(eq=eq, jac_chunk_size=200)` in the shell constraints (sed, 22:44:14Z);
  - an initial fixed-boundary `eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)), …)`
    (22:45:03Z).

  Both edits became `--chunk` (default 200) in `a322c7f22`. That commit also added a try/except around each shell and
  `shells_done.json` resume bookkeeping. Neither changes a run that succeeds. The equivalent today is the command above
  at `a322c7f22`. At `6993369dc` the only additional flag is `--repair`, which this run did not use.
* **Wall time** (log stamps): NAE fixed solve 75 s; shell k=2 at 686 s, k=4 at 2019 s, k=6 at 3747 s. The session
  killed the process at T 23:56:31Z (19:56 local, about 4280 s) during shell 8, because of a boundary crease (B.n max
  13%). The `jac.py` FOLD check in `fbscreen.sh` never ran, and there is no `score.json`. The usable output is
  `eq_fb_k6.h5`.

### 3b. Boundary repair of shell 6 → `…_L12/eq_fb_k6_repaired.h5` — RECOVERED (missing from the handoff's chain)
* T 2026-10-12T23:57:42Z (19:57:42 local). Working dir `coil_sandbox/from_axis/ss`:
```
D=../na/runs2/li_qa2_iota0.30_fb_a0.12_L12
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  python repair_eq.py $D/eq_fb_k6.h5 $D/coils.h5 $D/eq_fb_k6_repaired.h5 > $D/repair_k6.log 2>&1
```
* **Code:** `ss/repair_eq.py` was written at 23:57:09Z, then a sed removed `zeta=ze` from the `from_values` call. It
  was committed in `89f5a39e4` (20:34 local) with no edits found in between. This is the old equal-arclength,
  fixed-label refit with a continuation re-solve. It is not the later `bdry_repair` package, and today's `repair_eq.py`
  is a different, guarded wrapper.
* **Wall time:** about 4 min (log mtime 20:01:34 local). Surface moved: mean 0.14 mm, max 1.73 mm (`repair_k6.log`).

### 3c. `runs2/li_qa2_iota0.30_fb_a0.12_L12_rep` (shell 8, resumed from the repaired k=6) — RECOVERED
* T 2026-10-13T00:01:49Z (20:01:49 local). Working dir `na/`:
```
O=runs2/li_qa2_iota0.30_fb_a0.12_L12_rep
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python score_fb.py runs2/li_qa2_iota0.30 \
  --LMN 12 --a 0.12 --shells 8 --maxiter 60 \
  --resume runs2/li_qa2_iota0.30_fb_a0.12_L12/eq_fb_k6_repaired.h5 --out $O > $O.log 2>&1
cd ../ss && SANDBOX_DEVICE=cpu python tip.py ../na/$O/eq_fb.h5 ../na/$O/coils.h5 ../na/$O/tip_k8.png
```
* **Code:** `score_fb.py` at `a322c7f22` (committed 19:52 local; no edits to it before 00:26Z). There was no
  `--repair` flag yet.
* **Wall time:** `score.json` time_s **1268**. It redid the NAE fixed-boundary solve (61 s) and then shell 8 took
  about 1140 s (38 iterations).
* **The handoff's description needs correcting.** The handoff calls this "repaired/continued to `_rep` (shells 2-8)"
  and suggests `score_fb.py … --shells 2,4,6,8 --repair`. What actually happened is
  **3a (shells 2,4,6, no repair) → 3b (one repair of k=6 with the old `repair_eq.py`) → 3c (shell 8 only, resumed)**.
  `score_fb.py --repair` (added in `89f5a39e4`) repairs after every shell, so it is a different procedure.

### 3d. `ss/li030` start (Modal, not the laptop)
`prep.py` ran **on Modal** as step 0 of job `ss12_li030_q15` (`ss/li030/meta.json`): commit `6993369dc`, A100-40GB,
started 2026-10-13 01:15:58 (Modal clock, UTC), working dir `coil_sandbox/from_axis/ss`:
`python prep.py ../na/runs2/li_qa2_iota0.30_fb_a0.12_L12_rep /vol/ss12_li030_q15/start --LMN 12` (0.3 min). The
single stage took 55.9 min, the `fb_rescore.py … --shells 4,8 --repair` step 18.8 min and `tip.py` 0.6 min; the job
finished at 02:31:37. The full argv is in `meta.json`.

Submitted from `coil_sandbox/from_axis/cloud` at T 2026-10-13T01:15:48Z, right after commit `6993369dc`:
`FA_GPU=A100-40GB modal run --detach app.py --jobs ss12_li030_q15`. A watchdog with a 2026-10-12 23:15:58 local
deadline was started with `./watchdog.sh ap-q3M51tutZa8AYUGeiFhe2T <deadline> ss12_li030_q15`.

---

## 4. Design B, laptop steps after Modal

### 4a. `ss/li030/fb_norepair_local` — RECOVERED
* The checkpoints were pulled from `coil_sandbox/from_axis/ss` at T 2026-10-13T02:24:13Z with
  `modal volume get --force from-axis-runs ss12_li030_q15/ss/<f> .` (into `cloud_runs/ss12_li030_q15/ss`), for
  `eq_k5.h5 coils_k5.h5 eq_k6.h5 coils_k6.h5 raw_eq_k6.h5 result.json anchors.json`.
* T 2026-10-13T02:24:28Z (22:24 local). Working dir `ss/`:
```
R=cloud_runs/ss12_li030_q15
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  python fb_rescore.py $R/ss/eq_k6.h5 $R/ss/coils_k6.h5 $R/fb_norepair_local --shells 4,8 > $R/fb_norepair_local.log 2>&1
SANDBOX_DEVICE=gpu python tip.py $R/fb_norepair_local/eq_fb.h5 $R/ss/coils_k6.h5 $R/fb_norepair_local/tip.png --device gpu >> $R/fb_norepair_local.log 2>&1
```
  The defaults in force were `--maxiter 60 --chunk 200 --helicity 1,0 --bounds ../../bounds/fa_qa2_a0167.json`, with
  no `--repair`.
* **Code:** `6993369dc`. No edits to `fb_rescore.py` between that commit (01:15Z) and this run; the next edit is at
  02:52Z.
* **Wall time:** shell 4 done at 384 s, shell 8 at 1465 s, output written at 1503 s (log stamps).
* **Copied into `ss/li030/`** at T 2026-10-13T02:50:16Z (`cp -r … fb_norepair_local`, plus `ss`, `fb`→`fb_repair_modal`,
  `run.log`, `meta.json`). Committed in `220446924`. The paths in `score.json` still read `cloud_runs/ss12_li030_q15/…`.

### 4b. `ss/li030/repaired` — RECOVERED
* T 2026-10-13T13:57:01Z (09:57 local). Working dir `coil_sandbox/from_axis/ss`.
  `S=/tmp/claude-1000/-home-jay-Documents-desc-dev-DESC/59468c83-5694-4cd1-8df3-e0855fb2fcf6/scratchpad`:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 systemd-run --user --scope -q -p MemoryMax=11G -p MemorySwapMax=0 \
  python repair_eq.py li030/fb_norepair_local/eq_fb.h5 li030/ss/coils_k6.h5 $S/li030_new.h5 > $S/li030_new.log 2>&1
```
  The defaults in force were `--rule chordal --max-steps 8 --chunk 200`, without `--force`. The output was copied at
  14:02:18Z:

  | from | to |
  |---|---|
  | `$S/li030_new.h5` | `li030/repaired/eq.h5` |
  | `$S/li030_new_repair.json` | `li030/repaired/repair.json` |
  | `grep -E "^(boundary\|equilibrium\|REPAIR)" $S/li030_new.log` | `li030/repaired/report.txt` |

  The scratchpad log still exists, and its grep output is identical to `report.txt`.
* **Code:** `d5c8966e2` (committed 10:02 local, five minutes later). It contains the new `sandbox/bdry_repair`,
  `ss/repair_lib.py` (now a shim to it) and the new `ss/repair_eq.py`. The run used them uncommitted, and the session
  shows no edits to them in between.
* **Wall time:** about 4.5 min (started about 09:57:1x, output mtime 10:01:39 local).

---

## 5. Design D, near-axis chain: `nc5_s1 … nc5_s4`. Working dir: `na/`

* **Launched:** T 2026-10-13T22:21:43Z (18:21 local). The session wrote `ncchain.py`, which is identical to the
  version committed in `4e2a72420` 15 s later, and ran
  `printf '4\n5\n6\n' | xargs -P 2 -I{} python ncchain.py {} > runs2/ncchain.out 2>&1`. That ran the 4- and 5-coil
  chains in parallel, then the 6-coil chain. A summary table followed. **The exact command lines executed are logged
  in `na/runs2/ncchain.out`.**
* **Code:** `run_na2.py` at `49557cd7c` (`--tiltmax`; no edits between 17:29Z and 22:21Z). `ncchain.py` at
  `4e2a72420`.
* **Commands** (verbatim from `ncchain.out`; each was prefixed by
  `systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0`, with stdout to `runs2/nc5_sN.log`):
```
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --axis0 '1,0.15;0,-0.10' --eta0 0.6 --rcoil 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --seed 2 --out runs2/nc5_s1 --wlen 0.001 --wbend 0.001
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --wlen 0.001 --wbend 0.001 --order 2 --w2 0.001 --wgg 0.01 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s1 --out runs2/nc5_s2
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode z0 --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s2 --out runs2/nc5_s3
python run_na2.py --nfp 2 --nc 5 --K 5 --Kax 5 --mode free --iota 0.42 --a 0.167 --dpc 0.147795 --dcc 0.04 --Lmax 3.9412000000000003 --kmax 10.173652694610778 --imax 2.0 --emax 5.0 --hzmax 0.0334 --wlevel 0.0 --tiltmax 2.0 --wplanar 0.0 --zmax 0.0 --wlen 0.001 --wbend 0.001 --order 2 --w1 10.0 --w2 0.01 --wgg 0.1 --B2c0 1.2 --wpen 100.0 --nphi 41 --ntheta 48 --maxiter 600 --hessian secant --init runs2/nc5_s3 --out runs2/nc5_s4
```
  The parameters come from the reference runs: s1 from `s1_qa2_z0_s2` with `--emax 5 --wlen/--wbend 1e-3 --maxiter 600`,
  s2 from `o2_qa2s2_gg1e-2_w1e-3`, s3 from `o2w1_…`, and s4 from `lo_qa2_i42_h0.0334_t2`. The `lo_qa2_i42_h0.0334_t2`
  run itself, for `--init o2w1…`, is only the *source of the s4 arguments* and is not an ancestor of `nc5_s4`.

| stage | finished (local) | nit / status | time_s |
|---|---|---|---|
| nc5_s1 | 10-13 18:34 | 600, maxiter | 760 |
| nc5_s2 | 18:42 | 194, gtol | 461 |
| nc5_s3 | 18:47 | 103, gtol | 250 |
| nc5_s4 | 19:08 | 538, gtol | 1269 |

D's free-boundary score (`fb_nc5lh_b12`) and its single stage ran on Modal; they are already described in the handoff.

---

## 6. Not recovered, and caveats
1. **`s1_qa2_z0_s2` cannot be reproduced exactly from any commit.** It ran before the first commit, with the old
   nearest-node (kinked) distance rows. `15fa3e65a` has the soft-min distances, so rerunning the s1 command there
   gives a different cold-start optimum. The `.npz` state is the reproducible anchor, and REPRODUCE.md should start
   from it or state this.
2. The reference-lineage runs (`reg_*`, `elong_*`, `o2_*`, `o2w1_*`) and `li_qa2_iota0.30` ran from uncommitted code.
   The nearest commits are given above. For `o2w1` and later, the run-time code most likely equals the next commit.
3. The `li_qa2_iota0.30` command for `lowiota.sh` was rebuilt from `cont.py`'s logic. `cont.py` was not executed for
   this report because that tool call was not permitted. The json `args` agree with the rebuilt line.
4. The basis-12 laptop free boundary (3a) ran from a working copy between `d519b8955` and `a322c7f22`. As far as can be
   seen, it is functionally equal to `a322c7f22` with `--chunk 200`, but that commit is not byte-identical.
5. The FOLD (`jac.py`) step of `fbscreen.sh` never ran for li030, and the original L12 directory has no `score.json`.
