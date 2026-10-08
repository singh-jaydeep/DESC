#!/bin/bash
# Q1 (QA xyzN6 stage 2) died of the intermittent CUDA_ERROR_ILLEGAL_ADDRESS at iteration 14 on 2026-09-29 14:05.
# Rerun Q1 and then Q2 once the main chain is done (one job at a time).
cd /home/singh/Documents/DESC2/coil_sandbox
until grep -q "^CHAIN done" work/compare/chain.log; do sleep 30; done
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
Q=work/ss_qa
mv $Q/xyz/stage2 $Q/xyz/stage2_cudafail_1405
mkdir -p $Q/xyz/stage2 $Q/xyz/ss
echo "CHAIN Q1-retry start $(date +%T)"
$CAP $PY run_one.py --eq precise_QA --bounds bounds/precise_qa_sweep910.json --length-mult 0.8 --vacuum \
  --start "sweep_910_copy/initial starts/start_xyzN6.h5" --out $Q/xyz/stage2 --maxiter 400 > $Q/xyz/stage2/run.log 2>&1
if [ -f $Q/xyz/stage2/result.h5 ]; then
  echo "CHAIN Q1-retry ok $(date +%T)"; echo "CHAIN Q2 start $(date +%T)"
  $CAP $PY single_stage.py --eq precise_QA --start $Q/xyz/stage2/result.h5 --out $Q/xyz/ss --length-mult 0.8 \
    --kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --tau-qs-rel 10 > $Q/xyz/ss/run.log 2>&1
  [ -f $Q/xyz/ss/coils_k4.h5 ] && echo "CHAIN Q2 ok $(date +%T)" || echo "CHAIN Q2 FAILED $(date +%T)"
else
  echo "CHAIN Q1-retry FAILED $(date +%T)"
fi
echo "CHAIN retry done $(date +%T)"
