#!/bin/bash
# K5 stall diagnostics (2026-10-08, local box). Usage: run.sh NAME <extra flags>   (one job at a time, capped)
cd /home/jay/Documents/desc_dev/DESC/coil_sandbox
J=work/ss_qh/joint
PY=/home/jay/miniconda3/envs/desc-dev/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0
QH="--bounds bounds/precise_qh_gil.json --length-mult 1.0 --fix-hinge-z 0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
ANCH="--tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526"
K5="--kmin 5 --kmax 5 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12"
START=${START:-"--eq $J/ss_z0_k4b/eq_k4.h5 --start $J/ss_z0_k4b/coils_k4.h5"}
name=$1; shift
out=$J/stall/$name; mkdir -p $out
case $name in
  probe*) $CAP $PY $J/stall/probe.py --probe-out $out $START $QH $QHSS $ANCH $K5 --out $out "$@" > $out/run.log 2>&1 ;;
  *)      $CAP $PY single_stage.py $START $QH $QHSS $ANCH $K5 --out $out "$@" > $out/run.log 2>&1 ;;
esac
echo "exit $?"
