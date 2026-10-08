#!/bin/bash
# Step 0: --maxiter 0 build checks of the three step-1 runs on precise_QH (sequential, capped).
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
M=work/ss_qh/matrix
COMMON="--eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --maxiter 0"
mkdir -p $M/base/planarN7_build $M/base/arcB2_build $M/base/xyzN6_build
$CAP $PY run_al.py $COMMON --start "../sweep_qh/initial starts/start_planarN7.h5" --out $M/base/planarN7_build > $M/base/planarN7_build/run.log 2>&1
$CAP $PY run_al.py $COMMON --start $M/starts/arcB2_stage2_old_coilset.h5 --out $M/base/arcB2_build --cc-gap coils --cc-k 40000 > $M/base/arcB2_build/run.log 2>&1
$CAP $PY run_al.py $COMMON --start work/ss_qh/xyz/stage2/result.h5 --out $M/base/xyzN6_build --cc-k 20000 > $M/base/xyzN6_build/run.log 2>&1
echo STEP0 DONE
# step 0 rerun: arcB2 had 35937 of 40000 cc rows active at start; try 80000
# systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $PY run_al.py $COMMON --start $M/starts/arcB2_stage2_old_coilset.h5 --out $M/base/arcB2_build_k80k --cc-gap coils --cc-k 80000
