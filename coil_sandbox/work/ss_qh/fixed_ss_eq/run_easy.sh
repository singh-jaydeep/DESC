#!/bin/bash
# Planar optimality tests R1-R3 (2026-10-02), one at a time, each under the cgroup cap.
cd /home/singh/Documents/DESC2/coil_sandbox
OUT=work/ss_qh/fixed_ss_eq
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq work/ss_qh/arcB2/ss_k4/eq_k4.h5 --vacuum --length-mult 1.0 --maxiter 300 --src-n 100 --kappa-grid-n 600 --kappa-lb none --max-tr inf --diag-log diag.jsonl"
N7="../sweep_qh/initial starts/start_planarN7.h5"
run() { mkdir -p $OUT/$1; shift_name=$1; shift; echo "=== start $shift_name $(date +%T)"; $CAP $PY run_one.py $COMMON --out $OUT/$shift_name "$@" > $OUT/$shift_name/run.log 2>&1; echo "=== end $shift_name exit $? $(date +%T)"; sleep 10; }
run easy_R1 --start $OUT/start_planarN3.h5 --bounds bounds/precise_qh_gil.json
run easy_R2 --start "$N7" --bounds bounds/precise_qh_gil_kappa2x.json
run easy_R3 --start "$N7" --bounds bounds/precise_qh_gil_nokappa.json
