#!/bin/bash
# Step 2 (XYZ) with fallbacks, then the k=4 continuation "k4b" for all three reps (2026-10-08 01:0x).
# k4b: one more k=4 step of 100 its from each rep's k=4 result, ftol=xtol=1e-12, anchors frozen (same as the
# original ss_k4 recipe, which reran k=4 with xtol 1e-12); the k4b boundaries are the matrix rows.
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --tau-qs-rel 10"
export XLA_FLAGS=--xla_gpu_autotune_level=0
# --- XYZ single stage: chunks 25/5, then 10/2 (handoff trap 4); CPU is decided separately
for ch in "25 5" "10 2"; do
  set -- $ch; out=$M/ss/xyz
  [ -d $out ] && mv $out ${out}_fail_$(date +%H%M%S)
  mkdir -p $out; echo "=== start ss xyz chunks $1/$2 $(date +%T)"
  $CAP $PY single_stage.py $QH $QHSS $SS --start $M/base/xyzN6/result.h5 --out $out --src-n 100 \
      --bs-chunk $1 --jac-chunk $2 > $out/run.log 2>&1
  echo "=== end ss xyz chunks $1/$2 exit $? $(date +%T)"
  [ -f $out/coils_k4.h5 ] && break
  [ -f $out/coils_k1.h5 ] && { echo "=== xyz died after k=1; not retrying with smaller chunks"; break; }
  sleep 10
done
# --- k4b for each rep that has a k=4 result
for rep in arcB2 planar xyz; do
  src=$M/ss/$rep; out=$M/ss/${rep}_k4b
  [ -f $src/coils_k4.h5 ] || { echo "=== skip k4b $rep (no coils_k4)"; continue; }
  REPF=""; [ $rep = planar ] && REPF="--src-n 100"; [ $rep = xyz ] && REPF="--src-n 100 --bs-chunk 25 --jac-chunk 5"
  mkdir -p $out; echo "=== start k4b $rep $(date +%T)"
  $CAP $PY single_stage.py $QH $QHSS --eq $src/eq_k4.h5 --start $src/coils_k4.h5 --out $out \
      --tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526 --kmin 4 --kmax 4 --maxiter 100 \
      --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12 $REPF > $out/run.log 2>&1
  echo "=== end k4b $rep exit $? $(date +%T)"; sleep 10
done
echo "STEP2B DONE $(date +%T)"
