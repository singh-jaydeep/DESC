#!/bin/bash
# free_base rerun: the old QA arcB2 stage 2 (work/ss_qa/stage2_L0p8_M5, run_one, 4.54e-3) repacked to one CoilSet first.
cd /home/singh/Documents/DESC2/coil_sandbox
Z=work/ss_qh/joint/qa_z0
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
mkdir -p $Z/free_base; echo "=== start qa/free_base $(date +%T)"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $PY run_al.py --eq precise_QA --vacuum --bounds bounds/precise_qa_sweep910_L0p8.json \
  --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 --start $Z/starts/qa_stage2_L0p8_M5_packed.h5 --out $Z/free_base > $Z/free_base/run.log 2>&1
echo "=== end qa/free_base exit $? $(date +%T)"; echo "BASE DONE"
