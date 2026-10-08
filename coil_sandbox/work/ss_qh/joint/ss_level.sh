#!/bin/bash
# Single stage with each arcB2 coil's hinges LEVEL at its own free height (sandbox/demount.HingeLevel), on precise_QH.
# Identical to ss_z0.sh (matrix arcB2 weights and anchors, k = 1..4 then k4b, then the converged refinement) except
# --horizontal-hinges instead of --fix-hinge-z 0 and the start: base_level (converged stage 2, B.n 8.768e-3).
# Usage: ss_level.sh dry | chain | k4b | refine | all      (one job at a time, capped)
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0
QH="--bounds bounds/precise_qh_gil.json --length-mult 1.0 --horizontal-hinges"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
ANCH="--tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12"
K4B="--kmin 4 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12"
go() { out=$1; shift; mkdir -p $out; echo "=== start $out $(date +%T)"; $CAP $PY "$@" --out $out > $out/run.log 2>&1; echo "=== end $out exit $? $(date +%T)"; sleep 10; }
dry()    { go $J/ss_level_dry single_stage.py --eq precise_QH --start $J/base_level/result.h5 $QH $QHSS $ANCH $SS --dry; }
chain()  { go $J/ss_level single_stage.py --eq precise_QH --start $J/base_level/result.h5 $QH $QHSS $ANCH $SS; }
k4b()    { go $J/ss_level_k4b single_stage.py --eq $J/ss_level/eq_k4.h5 --start $J/ss_level/coils_k4.h5 $QH $QHSS $ANCH $K4B; }
refine() { go $J/on_level/arcB2_level run_al.py --eq $J/ss_level_k4b/eq_k4.h5 --vacuum --bounds bounds/precise_qh_gil.json \
             --maxiter 800 --start $J/ss_level_k4b/coils_k4.h5 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8; }
case ${1:-all} in
  dry) dry ;; chain) chain ;; k4b) k4b ;; refine) refine ;;
  all) dry && chain && [ -f $J/ss_level/eq_k4.h5 ] && k4b && [ -f $J/ss_level_k4b/eq_k4.h5 ] && refine ;;
esac
