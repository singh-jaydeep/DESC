#!/bin/bash
# Full coil-field check of the two best proxy-QS coilsets (screen 2026-10-09): trace -> traced-surface eq (res 12) -> modes
# -> direct Biot-Savart check -> eps_eff -> figures. One GPU process at a time, 12G cap.
#   xyzX: matrix on_xyz/xyzN6 on its own single-stage boundary B_xyz (proxy QS rho .5 3.8e-4)
#   xyzB: matrix base/xyzN6 on precise_QH (proxy 5.9e-4)
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix; F=work/ss_qh/joint/fieldcheck
PY="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 /home/singh/miniforge3/envs/desc-env2/bin/python"
export XLA_FLAGS=--xla_gpu_autotune_level=0 NPER=400
one() {
  c=$1; coils=$2; eq=$3; mkdir -p $F/$c
  echo "=== $c trace $(date +%T)"; $PY $F/fieldcheck.py $c $coils $eq $F/$c trace > $F/$c/run.log 2>&1 || { echo "FAIL trace $c"; return; }
  sleep 5; $F/pipeline.sh $c $coils $eq || { echo "FAIL pipeline $c"; return; }
  sleep 5; echo "=== $c direct $(date +%T)"; $PY $F/directcheck.py $c $coils $eq > $F/$c/directcheck.log 2>&1 || echo "FAIL direct $c"
  sleep 5; echo "=== $c ripple $(date +%T)"; $PY $F/ripple.py $c $eq > $F/$c/ripple.log 2>&1 || echo "FAIL ripple $c"
  sleep 5; echo "=== $c figures $(date +%T)"; $PY $F/figures.py $c $eq > $F/$c/figures.log 2>&1 || echo "FAIL figures $c"
}
one xyzX $M/on_xyz/xyzN6/result.h5 $M/ss/xyz_k4b/eq_k4.h5
one xyzB $M/base/xyzN6/result.h5 precise_QH
echo "=== chain_xyz done $(date +%T)"
