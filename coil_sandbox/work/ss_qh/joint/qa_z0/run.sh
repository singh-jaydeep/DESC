#!/bin/bash
# z = 0 joint plane on precise_QA (the z0 screen's best candidate with real iota: z = 0 chord >= 36% of every section).
# Bounds: bounds/precise_qa_sweep910_L0p8.json (the QA arcB2 baseline's), 4 coils/hp, arcB2 M5, fixed equilibrium.
#   z0        : --fix-hinge-z 0 from the z = 0 lens start
#   level     : --horizontal-hinges from the same start
#   free_warm : free hinges, warm from level
#   free_base : free hinges, warm from the old QA arcB2 stage 2 (run_one, 4.54e-3) -- the new-solver free baseline
cd /home/singh/Documents/DESC2/coil_sandbox
Z=work/ss_qh/joint/qa_z0
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq precise_QA --vacuum --bounds bounds/precise_qa_sweep910_L0p8.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8"
run() { name=$1; start=$2; shift 2; mkdir -p $Z/$name; echo "=== start qa/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $start --out $Z/$name "$@" > $Z/$name/run.log 2>&1; echo "=== end qa/$name exit $? $(date +%T)"; sleep 10; }
S=$Z/starts/z0_start_packed.h5
run z0 $S --fix-hinge-z 0
run level $S --horizontal-hinges
[ -f $Z/level/result.h5 ] && run free_warm $Z/level/result.h5
run free_base work/ss_qa/stage2_L0p8_M5/result.h5
echo "QA DONE"
