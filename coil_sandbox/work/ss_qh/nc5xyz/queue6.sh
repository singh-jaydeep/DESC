#!/bin/bash
# Design-relevant: arcB2 M5, 5 coils/hp, LEVEL hinges, on the level example's boundary (cont/k4b), warm from arc5 (free, B_arcB2),
# cont/refine flags. Compare with cont/refine (nc4, 2.266e-3). Then proxy. Waits for queue5.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck; EQL=work/ss_qh/joint/cont/k4b/eq_k4.h5
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
until grep -q "QUEUE5 DONE" $N/queue5.log; do sleep 30; done
mkdir -p $N/arc5_level; echo "=== start arc5_level $(date +%T)"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $PY run_al.py --eq $EQL --vacuum --bounds bounds/precise_qh_gil.json \
  --maxiter 800 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8 --start $N/arc5/result.h5 --out $N/arc5_level > $N/arc5_level/run.log 2>&1
echo "=== end arc5_level exit $? $(date +%T)"; sleep 5
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 $PY $F/spectrum.py nc5_arc5_level $N/arc5_level/result.h5 $EQL > $F/spectrum/nc5_arc5_level.log 2>&1
echo "=== QUEUE6 DONE $(date +%T)"
