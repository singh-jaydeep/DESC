#!/bin/bash
# XYZ single stage retry (2026-10-08). The 25/5 attempt died on a cuFFT batched plan for the 301-node coil grid
# (301 nodes, batch 4); the 10/2 attempt on the 201-node source grid during setup (that size had passed in the
# first attempt). Size-independent -> GPU memory state: cuFFT plan memory is allocated outside JAX's preallocated
# pool. Fix 1: mem fraction 0.5. Fix 2: no preallocation at all. Neither changes the numerics.
# Then the XYZ k4b continuation with the same environment and flags.
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
until grep -q "STEP2B DONE" $M/step2b_chain.out; do sleep 20; done
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --tau-qs-rel 10"
export XLA_FLAGS=--xla_gpu_autotune_level=0
XF="--src-n 100 --bs-chunk 25 --jac-chunk 5"
ok=0
for envv in XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 XLA_PYTHON_CLIENT_PREALLOCATE=false; do
  export $envv; extra=""
  out=$M/ss/xyz; [ -d $out ] && mv $out ${out}_fail_$(date +%H%M%S); mkdir -p $out
  echo "=== start ss xyz [$envv] $(date +%T)"
  $CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/xyzN6/result.h5 --out $out $XF $extra > $out/run.log 2>&1
  echo "=== end ss xyz [$envv] exit $? $(date +%T)"
  [ -f $out/coils_k4.h5 ] && { ok=1; XEXTRA=""; break; }
  [ -f $out/coils_k1.h5 ] && { echo "=== xyz died after k=1; stopping"; break; }
  sleep 10
done
if [ $ok = 1 ]; then
  out=$M/ss/xyz_k4b; mkdir -p $out; echo "=== start k4b xyz $(date +%T)"
  $CAP $PY single_stage.py $QH $QHSS --eq $M/ss/xyz/eq_k4.h5 --start $M/ss/xyz/coils_k4.h5 --out $out \
      --tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526 --kmin 4 --kmax 4 --maxiter 100 \
      --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12 $XF $XEXTRA > $out/run.log 2>&1
  echo "=== end k4b xyz exit $? $(date +%T)"
fi
echo "STEP2C DONE ok=$ok $(date +%T)"
