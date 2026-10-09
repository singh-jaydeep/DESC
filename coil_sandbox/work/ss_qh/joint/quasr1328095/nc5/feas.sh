#!/bin/bash
# Is QUASR 1328095's own design (5 coils/hp, bounds/quasr1328095_own.json) feasible for arcB2 M5 at all?
# Constraint-only solves (--obj-weight 1e-6) from the same infeasible lens start as nc5/level, with the coil-coil row
# grid doubled (--row-n 512: nc5/level's node gap check read used 1.8 mm < needed 2.8 mm). F1 level hinges, F2 free.
cd /home/singh/Documents/DESC2/coil_sandbox
Q=work/ss_qh/joint/quasr1328095
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq $Q/eq_q1328095.h5 --vacuum --bounds bounds/quasr1328095_own.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 --obj-weight 1e-6 --row-n 512"
run() { name=$1; shift; mkdir -p $Q/nc5/$name; echo "=== start nc5/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $Q/nc5/starts/level_start_packed.h5 --out $Q/nc5/$name "$@" > $Q/nc5/$name/run.log 2>&1; echo "=== end nc5/$name exit $? $(date +%T)"; sleep 10; }
run feas_level --horizontal-hinges
run feas_free
