#!/bin/bash
# Step 1: new-solver (lsq-auglag-composite) baselines on the original precise_QH, one at a time, capped.
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800"
run() { name=$1; shift; mkdir -p $M/base/$name; echo "=== start $name $(date +%T)"; $CAP $PY run_al.py $COMMON --out $M/base/$name "$@" > $M/base/$name/run.log 2>&1; echo "=== end $name exit $? $(date +%T)"; sleep 10; }
run planarN7 --start "../sweep_qh/initial starts/start_planarN7.h5"
run arcB2 --start $M/starts/arcB2_stage2_old_coilset.h5 --cc-gap coils --cc-k 80000   # 40000 overflowed: 35937 active at the build check
run xyzN6 --start work/ss_qh/xyz/stage2/result.h5 --cc-k 20000
echo STEP1 DONE
