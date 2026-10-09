#!/bin/bash
# QUASR 1328095 with its OWN design: 5 coils/half period and bounds/quasr1328095_own.json (L 2.5/coil, d_pc 0.15,
# d_cc 0.10, kappa_MS 35, kappa 13.67). Start: level-chord lens (clear 2.0, QUASR's 5 currents), infeasible.
# level (--horizontal-hinges), then free warm-started from it. User OK 2026-10-08.
cd /home/singh/Documents/DESC2/coil_sandbox
Q=work/ss_qh/joint/quasr1328095
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq $Q/eq_q1328095.h5 --vacuum --bounds bounds/quasr1328095_own.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8"
run() { name=$1; start=$2; shift 2; mkdir -p $Q/nc5/$name; echo "=== start nc5/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $start --out $Q/nc5/$name "$@" > $Q/nc5/$name/run.log 2>&1; echo "=== end nc5/$name exit $? $(date +%T)"; sleep 10; }
run level $Q/nc5/starts/level_start_packed.h5 --horizontal-hinges
[ -f $Q/nc5/level/result.h5 ] && run free_warm $Q/nc5/level/result.h5
