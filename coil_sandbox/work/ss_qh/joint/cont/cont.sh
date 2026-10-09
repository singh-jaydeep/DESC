#!/bin/bash
# Coil continuation alongside boundary-mode continuation (user, 2026-10-08), precise_QH, arcB2 hinges level per coil.
# Baseline: ss_level.sh (M5, length x1.0 at every k; refined 2.859e-3). Here (user chose the wider ramp):
#   stage 2  L x0.80 M3 (new M3 level-chord lens start)
#   k=1 L x0.80 M3 | k=2 x0.867 M4 | k=3 x0.933 M5 | k=4 x1.0 M5 | k4b x1.0 M5 | refine (run_al) x1.0 M5
# Everything else as ss_level.sh (weights, anchors, maxiter 100 per k, k4b ftol 1e-12). One step per process.
cd /home/singh/Documents/DESC2/coil_sandbox
C=work/ss_qh/joint/cont
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CPU="systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19 --bounds bounds/precise_qh_gil.json --horizontal-hinges"
ANCH="--tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526"
STEP="--maxiter 100 --solve-tol 1e-10 --xtol 1e-12"
go() { out=$1; shift; mkdir -p $out; echo "=== start $out $(date +%T)"; $CAP $PY "$@" --out $out > $out/run.log 2>&1; r=$?; echo "=== end $out exit $r $(date +%T)"; sleep 10; return $r; }
pad() { $CPU $PY pad_arcs.py $1 --M $2 --out $3 --eq $4 > $3.log 2>&1 && grep -E "max curve change" $3.log; }
ss() { k=$1; lm=$2; eq=$3; coils=$4; shift 4; go $C/k$k single_stage.py --eq $eq --start $coils --kmin $k --kmax $k --length-mult $lm $QHSS $ANCH $STEP "$@" && [ -f $C/k$k/eq_k$k.h5 ]; }
set -o pipefail
go $C/s2 run_al.py --eq precise_QH --vacuum --bounds bounds/precise_qh_gil_L080.json --maxiter 800 \
    --start $C/starts/level_M3_packed.h5 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8 || exit 1
ss 1 0.80 precise_QH $C/s2/result.h5 || exit 1
pad $C/k1/coils_k1.h5 4 $C/k1/coils_k1_M4.h5 $C/k1/eq_k1.h5 || exit 1
ss 2 0.8667 $C/k1/eq_k1.h5 $C/k1/coils_k1_M4.h5 || exit 1
pad $C/k2/coils_k2.h5 5 $C/k2/coils_k2_M5.h5 $C/k2/eq_k2.h5 || exit 1
ss 3 0.9333 $C/k2/eq_k2.h5 $C/k2/coils_k2_M5.h5 || exit 1
ss 4 1.0 $C/k3/eq_k3.h5 $C/k3/coils_k3.h5 || exit 1
mkdir -p $C/k4b; go $C/k4b single_stage.py --eq $C/k4/eq_k4.h5 --start $C/k4/coils_k4.h5 --kmin 4 --kmax 4 --length-mult 1.0 \
    $QHSS $ANCH $STEP --ftol 1e-12 && [ -f $C/k4b/eq_k4.h5 ] || exit 1
go $C/refine run_al.py --eq $C/k4b/eq_k4.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 \
    --start $C/k4b/coils_k4.h5 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8
