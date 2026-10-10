#!/bin/bash
# Control: is the xyz5 gain count or conductor? XYZ N6 nc4 with per-coil L x1.25 (same total length as xyz5), cold circles r/a 2.6.
# Also nc4 at the stated L from the same cold start (start-procedure control vs the warm matrix xyzX). Waits for queue2.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck; EQX=work/ss_qh/matrix/ss/xyz_k4b/eq_k4.h5
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CAPQ="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
until grep -q "QUEUE2 DONE" $N/queue2.log; do sleep 30; done
for v in "xyz4_L125 bounds/precise_qh_gil_nc4_L125.json" "xyz4_cold bounds/precise_qh_gil.json"; do
  set -- $v; mkdir -p $N/$1; echo "=== start $1 $(date +%T)"
  $CAP $PY run_al.py --eq $EQX --vacuum --bounds $2 --maxiter 800 --cc-k 30000 --link-mu 1e8 --start $N/starts/circ4_r2.6.h5 --out $N/$1 > $N/$1/run.log 2>&1
  echo "=== end $1 exit $? $(date +%T)"; sleep 5
  $CAPQ $PY $F/spectrum.py nc_$1 $N/$1/result.h5 $EQX > $F/spectrum/nc_$1.log 2>&1
done
echo "=== QUEUE3 DONE $(date +%T)"
