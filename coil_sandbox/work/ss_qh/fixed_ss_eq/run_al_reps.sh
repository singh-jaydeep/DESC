#!/bin/bash
# arcB2 then FourierXYZ with lsq-auglag-composite (2026-10-07), one at a time, each under the cgroup cap.
cd /home/singh/Documents/DESC2/coil_sandbox
B=work/ss_qh/fixed_ss_eq
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq work/ss_qh/arcB2/ss_k4/eq_k4.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800"
run() { name=$1; shift; mkdir -p $B/$name; echo "=== start $name $(date +%T)"; $CAP $PY run_al.py $COMMON --out $B/$name "$@" > $B/$name/run.log 2>&1; echo "=== end $name exit $? $(date +%T)"; sleep 10; }
run al_arcB2 --start $B/start_arcB2_coilset.h5 --cc-gap coils --cc-k 40000
run al_xyzN6 --start work/ss_qh/xyz/stage2/result.h5 --cc-k 20000
