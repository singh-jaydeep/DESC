#!/bin/bash
# tau_QS ladder, k=4 M=5, each rung warm-started from the previous one; then the analysis.
# tau_QS = mult x 2.046e-4 (the start's scale-invariant QS rms); V0, bn-norm anchored to the original start.
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
W=work/ss_qa
prev_eq=$W/ss_v2_qs10_k4_M5/eq_k4.h5; prev_cs=$W/ss_v2_qs10_k4_M5/coils_k4.h5
for m in 30 100 1000; do
  d=$W/qsladder/qs$m; mkdir -p $d
  tau=$(python3 -c "print($m * 2.046e-4)")
  echo "LADDER rung x$m (tau_qs $tau) from $prev_eq  $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY single_stage.py --eq $prev_eq --start $prev_cs \
      --out $d --length-mult 0.8 --tau-qs $tau --V0 0.60032 --bn-norm 0.48316 --kmin 4 --kmax 4 --maxiter 100 \
      --solve-tol 1e-10 > $d/run.log 2>&1
  rc=$?
  echo "LADDER rung x$m exit $rc  $(date +%T)"
  if [ $rc -ne 0 ] || [ ! -f $d/eq_k4.h5 ]; then echo "LADDER stopping: rung x$m failed"; break; fi
  prev_eq=$d/eq_k4.h5; prev_cs=$d/coils_k4.h5
done
echo "LADDER analysis  $(date +%T)"
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY $W/qsladder/analyze.py --out $W/qsladder > $W/qsladder/analyze.log 2>&1
echo "LADDER analysis exit $?  $(date +%T)"
echo "LADDER done"
