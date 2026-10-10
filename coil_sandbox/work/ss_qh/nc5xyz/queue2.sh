#!/bin/bash
# Overnight queue 2 (2026-10-10): is the nc5 gain coil count or conductor? does it carry to arcB2? confirm xyz5 with full pipeline.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck; EQX=work/ss_qh/matrix/ss/xyz_k4b/eq_k4.h5; EQA=work/ss_qh/matrix/ss/arcB2_k4b/eq_k4.h5
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CAPQ="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
# 1. same total conductor as nc4: per-coil L x 0.8
mkdir -p $N/xyz5_Lfix; echo "=== start xyz5_Lfix $(date +%T)"
$CAP $PY run_al.py --eq $EQX --vacuum --bounds bounds/precise_qh_gil_nc5_Lfix.json --maxiter 800 --cc-k 30000 --link-mu 1e8 \
  --start $N/starts/circ_r2.6.h5 --out $N/xyz5_Lfix > $N/xyz5_Lfix/run.log 2>&1; echo "=== end xyz5_Lfix exit $? $(date +%T)"
sleep 5; $CAPQ $PY $F/spectrum.py nc5_xyz5_Lfix $N/xyz5_Lfix/result.h5 $EQX > $F/spectrum/nc5_xyz5_Lfix.log 2>&1
# 2. full coil-field check of xyz5
sleep 5; mkdir -p $F/xyz5; echo "=== xyz5 trace $(date +%T)"
export XLA_FLAGS=--xla_gpu_autotune_level=0 NPER=400
$CAPQ $PY $F/fieldcheck.py xyz5 $N/xyz5/result.h5 $EQX $F/xyz5 trace > $F/xyz5/run.log 2>&1
sleep 5; $F/pipeline.sh xyz5 $N/xyz5/result.h5 $EQX
sleep 5; $CAPQ $PY $F/directcheck.py xyz5 $N/xyz5/result.h5 $EQX > $F/xyz5/directcheck.log 2>&1
sleep 5; $CAPQ $PY $F/ripple.py xyz5 $EQX > $F/xyz5/ripple.log 2>&1
sleep 5; $CAPQ $PY $F/figures.py xyz5 $EQX > $F/xyz5/figures.log 2>&1
unset XLA_FLAGS
# 3. arcB2 M5, 5 coils/hp, free hinges, on B_arcB2, cold (arcs from circles r/a 2.6), matrix arcB2 recipe + link guard
sleep 5; mkdir -p $N/arc5; echo "=== start arc5 $(date +%T)"
$CAP $PY run_al.py --eq $EQA --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 \
  --start $N/starts/arcB2_r2.6.h5 --out $N/arc5 > $N/arc5/run.log 2>&1; echo "=== end arc5 exit $? $(date +%T)"
sleep 5; $CAPQ $PY $F/spectrum.py nc5_arc5 $N/arc5/result.h5 $EQA > $F/spectrum/nc5_arc5.log 2>&1
echo "=== QUEUE2 DONE $(date +%T)"
