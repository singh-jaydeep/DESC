#!/bin/bash
# Full coil-field QS check for one case (each stage its own process: GPU memory does not carry over).
#   tracedsurf (fit traced rho0 0.95 line, deform target onto it, res 10) -> polish (res 12) -> modes
# Usage: pipeline.sh CASE COILS EQ_TARGET   (needs CASE/trace.npz + result.json from fieldcheck.py trace)
cd /home/singh/Documents/DESC2/coil_sandbox
F=work/ss_qh/joint/fieldcheck
PY="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 /home/singh/miniforge3/envs/desc-env2/bin/python"
export XLA_FLAGS=--xla_gpu_autotune_level=0
c=$1; coils=$2; eq=$3
echo "=== $c tracedsurf $(date +%T)"; $PY $F/tracedsurf.py $c $coils $eq 0.95 10 > $F/$c/tracedsurf.log 2>&1 || exit 1
sleep 5; echo "=== $c polish L12 $(date +%T)"; $PY $F/polish.py $c $eq $F/$c/eq_traced.h5 $F/$c/eq_traced_L12.h5 12 500 > $F/$c/polish_L12.log 2>&1 || exit 1
sleep 5; echo "=== $c modes $(date +%T)"; $PY $F/modes.py $c $eq $F/$c/eq_traced_L12.h5 0.95 > $F/$c/modes.log 2>&1 || exit 1
echo "=== $c done $(date +%T)"
