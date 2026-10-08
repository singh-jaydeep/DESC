#!/bin/bash
# iota-floor relaxation at tau_QS x1000 (k=4, M=5), chained from the x1000 rung: floor 0.30, then 0.20.
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
W=work/ss_qa
prev_eq=$W/qsladder/qs1000/eq_k4.h5; prev_cs=$W/qsladder/qs1000/coils_k4.h5
for fl in 30 20; do
  d=$W/qsladder/iota$fl; mkdir -p $d
  echo "IOTA rung floor 0.$fl from $prev_eq  $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY single_stage.py --eq $prev_eq --start $prev_cs \
      --out $d --length-mult 0.8 --tau-qs 0.2046 --V0 0.60032 --bn-norm 0.48316 --iota-min 0.$fl \
      --kmin 4 --kmax 4 --maxiter 100 --solve-tol 1e-10 > $d/run.log 2>&1
  rc=$?
  echo "IOTA rung floor 0.$fl exit $rc  $(date +%T)"
  if [ $rc -ne 0 ] || [ ! -f $d/eq_k4.h5 ]; then echo "IOTA stopping: floor 0.$fl failed"; break; fi
  prev_eq=$d/eq_k4.h5; prev_cs=$d/coils_k4.h5
done
echo "IOTA analysis  $(date +%T)"
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY $W/qsladder/analyze.py --out $W/qsladder > $W/qsladder/analyze.log 2>&1
echo "IOTA analysis exit $?  $(date +%T)"
echo "IOTA done"
