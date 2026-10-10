#!/bin/bash
# Does more coils cut the coil-field ripple floor? FourierXYZ N6, 5 coils/half period (40), on B_xyz (matrix ss/xyz_k4b),
# precise_QH bounds unchanged (per-coil L same: 25% more conductor than nc4). Cold start: circles r/a 2.6 (d_cc 0.47, d_pc 0.96
# of bounds, unlinked), link guard. Waits for fieldcheck/chain_xyz.sh (one GPU job at a time). Then the proxy-QS screen.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
until grep -q "chain_xyz done" $F/chain_xyz.log; do sleep 30; done
mkdir -p $N/xyz5; echo "=== start xyz5 $(date +%T)"
$CAP $PY run_al.py --eq work/ss_qh/matrix/ss/xyz_k4b/eq_k4.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 \
  --cc-k 30000 --link-mu 1e8 --start $N/starts/circ_r2.6.h5 --out $N/xyz5 > $N/xyz5/run.log 2>&1
echo "=== end xyz5 exit $? $(date +%T)"
sleep 10; $CAP $PY $F/spectrum.py nc5_xyz5 $N/xyz5/result.h5 work/ss_qh/matrix/ss/xyz_k4b/eq_k4.h5 > $F/spectrum/nc5_xyz5.log 2>&1
echo "=== NC5 DONE $(date +%T)"
