#!/bin/bash
# Fair re-score of the basis-8 k5 results at basis 12: re-solve the equilibrium at basis 12, then the --dry term table
# (same anchors and weights) with each run's own coils.
cd /home/jay/Documents/desc_dev/DESC/coil_sandbox
J=work/ss_qh/joint
D=$J/stall
PY=/home/jay/miniconda3/envs/desc-dev/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0
FLAGS="--bounds bounds/precise_qh_gil.json --length-mult 1.0 --fix-hinge-z 0 --helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19 --tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526 --kmin 5 --kmax 5"
mkdir -p $D/rescore12
while read name eqf coilf; do
  out=$D/rescore12/$name; mkdir -p $out
  [ -f $out/eq12.h5 ] || $CAP $PY $D/rescore12.py $eqf $out/eq12.h5 12 300 200 > $out/solve.log 2>&1
  grep RESCORE $out/solve.log
  $CAP $PY single_stage.py --eq $out/eq12.h5 --start $coilf $FLAGS --out $out --dry > $out/dry.log 2>&1
  echo "== $name"; grep -E "^START|^Vacuum|^QS two|^rotational|^total" $out/dry.log | cut -c1-200
done <<LIST
k4b_start   $J/ss_z0_k4b/eq_k4.h5            $J/ss_z0_k4b/coils_k4.h5
k5_orig     $J/ss_z0_k5/eq_k5.h5             $J/ss_z0_k5/coils_k5.h5
k5_tol6     $D/k5_tol6/eq_k5_resolved.h5     $D/k5_tol6/coils_k5.h5
k5_comp     $D/k5_comp_secant/eq_k5_resolved.h5  $D/k5_comp_secant/coils_k5.h5
k5_pincomp  $D/k5_pin_comp/eq_k5_resolved.h5 $D/k5_pin_comp/coils_k5.h5
k5_sn       $D/k5_sn/eq_k5.h5                $D/k5_sn/coils_k5.h5
LIST
