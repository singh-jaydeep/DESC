#!/bin/bash
# arc5 retry with the repacked start (make_start arcs are a MixedCoilSet; run_al needs CoilSet(NFP, sym)). Waits for queue3.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck; EQA=work/ss_qh/matrix/ss/arcB2_k4b/eq_k4.h5
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
until grep -q "QUEUE3 DONE" $N/queue3.log; do sleep 30; done
mkdir -p $N/arc5; echo "=== start arc5 $(date +%T)"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $PY run_al.py --eq $EQA --vacuum --bounds bounds/precise_qh_gil.json \
  --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 --start $N/starts/arcB2_r2.6_packed.h5 --out $N/arc5 > $N/arc5/run.log 2>&1
echo "=== end arc5 exit $? $(date +%T)"; sleep 5
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY $F/spectrum.py nc5_arc5 $N/arc5/result.h5 $EQA > $F/spectrum/nc5_arc5.log 2>&1
echo "=== QUEUE4 DONE $(date +%T)"
