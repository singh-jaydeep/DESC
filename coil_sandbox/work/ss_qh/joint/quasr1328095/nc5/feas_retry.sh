#!/bin/bash
# Re-run the feasibility solves that crashed on the first --obj-weight version (setter checked dim_f before build).
cd /home/singh/Documents/DESC2/coil_sandbox
Q=work/ss_qh/joint/quasr1328095
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq $Q/eq_q1328095.h5 --vacuum --bounds bounds/quasr1328095_own.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 --obj-weight 1e-6 --row-n 512"
run() { name=$1; shift; [ -f $Q/nc5/$name/result.json ] && { echo "skip $name (has result)"; return; }; mv $Q/nc5/$name $Q/nc5/${name}_crash1 2>/dev/null; mkdir -p $Q/nc5/$name; echo "=== start nc5/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $Q/nc5/starts/level_start_packed.h5 --out $Q/nc5/$name "$@" > $Q/nc5/$name/run.log 2>&1; echo "=== end nc5/$name exit $? $(date +%T)"; sleep 10; }
run feas_level --horizontal-hinges
run feas_free
echo "RETRY DONE"
