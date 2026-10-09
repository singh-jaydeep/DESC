#!/bin/bash
# Best performer (cont/: coil continuation, precise_QH, arcB2 hinges level per coil) at equilibrium resolution 12.
# User (2026-10-09): another session found single-stage stalls can come from the discretization and vanish at higher
# resolution; cont/k4b at L=M=N=8 stopped on xtol after 5 its (B.n 2.4436e-3 -> 2.4432e-3).
#   prep   upres.py: cont/k4b/eq_k4.h5 -> L=M=N=12, grids 24, fixed-boundary re-solve -> eq_k4b_L12.h5
#   k4b    = cont.sh k4b flags, from the L12 eq + cont/k4b coils
#   k5     k = 5 (boundary |m|,|n| <= 5 free), same flags
#   refine = cont.sh refine flags (run_al, fixed eq)
# Usage: run.sh k4b | k5 | refine | all.  CHUNK="--jac-chunk N --bs-chunk N" to rerun a step on memory trouble.
cd /home/singh/Documents/DESC2/coil_sandbox
H=work/ss_qh/joint/hires; C=work/ss_qh/joint/cont
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19 --bounds bounds/precise_qh_gil.json --horizontal-hinges"
ANCH="--tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526"
CHUNK=${CHUNK:-"--eq-jac-chunk 300"}
STEP="--maxiter ${MAXITER:-100} --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12 --length-mult 1.0"
go() { out=$1; shift; mkdir -p $out; echo "=== start $out $(date +%T)"; $CAP $PY "$@" --out $out > $out/run.log 2>&1; r=$?; echo "=== end $out exit $r $(date +%T)"; sleep 10; return $r; }
k4b()    { go $H/k4b single_stage.py --eq $H/eq_k4b_L12.h5 --start $C/k4b/coils_k4.h5 --kmin 4 --kmax 4 $QHSS $ANCH $STEP $CHUNK \
             && [ -f $H/k4b/eq_k4.h5 ]; }
k5()     { go $H/k5 single_stage.py --eq $H/k4b/eq_k4.h5 --start $H/k4b/coils_k4.h5 --kmin 5 --kmax 5 $QHSS $ANCH $STEP $CHUNK \
             && [ -f $H/k5/eq_k5.h5 ]; }
refine() { S=${FROM:-k5}; go $H/refine run_al.py --eq $H/$S/eq_${S%b}.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 \
             --start $H/$S/coils_${S%b}.h5 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8; }
set -o pipefail
case ${1:-all} in
  k4b) k4b ;; k5) k5 ;; refine) refine ;;
  all) k4b && k5 && refine ;;
  short) k4b && FROM=k4b refine ;;   # user 2026-10-09: MAXITER=30, no k5, refine from k4b
esac
