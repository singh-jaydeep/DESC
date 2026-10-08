#!/bin/bash
# Single stage with every arcB2 hinge held at z = 0 (one demountable joint plane), on precise_QH.
# Identical to the matrix arcB2 single stage (work/ss_qh/matrix/step2_ss.sh + step2b_chain.sh k4b) except:
#   --fix-hinge-z 0, the start (base_z0: converged joint-plane stage 2 on precise_QH), and the anchors passed
#   explicitly (tau_QS 0.01695, V0 0.30352, B.n norm 0.42526 = the matrix's), so the weights are identical.
# Then the converged stage-2 refinement on the new boundary (run_al.py, joint plane + link guard).
# Usage: ss_z0.sh dry | chain | k4b | refine | all      (one job at a time, capped)
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0   # the arcB2 single stage hit the autotuner failure once
QH="--bounds bounds/precise_qh_gil.json --length-mult 1.0 --fix-hinge-z 0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
ANCH="--tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12"
K4B="--kmin 4 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12"
go() { out=$1; shift; mkdir -p $out; echo "=== start $out $(date +%T)"; $CAP $PY "$@" --out $out > $out/run.log 2>&1; echo "=== end $out exit $? $(date +%T)"; sleep 10; }
dry()    { go $J/ss_z0_dry single_stage.py --eq precise_QH --start $J/base_z0/result.h5 $QH $QHSS $ANCH $SS --dry; }
chain()  { go $J/ss_z0 single_stage.py --eq precise_QH --start $J/base_z0/result.h5 $QH $QHSS $ANCH $SS; }
k4b()    { go $J/ss_z0_k4b single_stage.py --eq $J/ss_z0/eq_k4.h5 --start $J/ss_z0/coils_k4.h5 $QH $QHSS $ANCH $K4B; }
refine() { go $J/on_z0/arcB2_z0 run_al.py --eq $J/ss_z0_k4b/eq_k4.h5 --vacuum --bounds bounds/precise_qh_gil.json \
             --maxiter 800 --start $J/ss_z0_k4b/coils_k4.h5 --cc-gap coils --cc-k 80000 --fix-hinge-z 0 --link-mu 1e8; }
# k = 5 (2026-10-08, user's suggested next step): one more step from the k4b state with |m|, |n| <= 5 free
# (120 boundary modes), same flags as k4b; then its refinement.
K5="--kmin 5 --kmax 5 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --ftol 1e-12"
k5()      { go $J/ss_z0_k5 single_stage.py --eq $J/ss_z0_k4b/eq_k4.h5 --start $J/ss_z0_k4b/coils_k4.h5 $QH $QHSS $ANCH $K5; }
refine5() { go $J/on_z0_k5/arcB2_z0 run_al.py --eq $J/ss_z0_k5/eq_k5.h5 --vacuum --bounds bounds/precise_qh_gil.json \
              --maxiter 800 --start $J/ss_z0_k5/coils_k5.h5 --cc-gap coils --cc-k 80000 --fix-hinge-z 0 --link-mu 1e8; }
case ${1:-all} in
  k5) k5 ;; refine5) refine5 ;;
  all5) k5 && [ -f $J/ss_z0_k5/eq_k5.h5 ] && refine5 ;;
  dry) dry ;; chain) chain ;; k4b) k4b ;; refine) refine ;;
  all) chain && [ -f $J/ss_z0/eq_k4.h5 ] && k4b && [ -f $J/ss_z0_k4b/eq_k4.h5 ] && refine ;;
esac
