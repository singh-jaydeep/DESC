#!/bin/bash
# After the main chain: the two FourierXYZ stage-2 runs (Q1, H3) both died of CUDA_ERROR_ILLEGAL_ADDRESS on
# 2026-09-29 (Q1 at iteration 14, H3 before iteration 1) while all arcB2 runs passed. Retry each with XLA GPU
# autotuning off, then on CPU if that also fails; then their single-stage steps; then H2's k=4 with xtol 1e-12.
cd /home/singh/Documents/DESC2/coil_sandbox
until grep -q "^CHAIN done" work/compare/chain.log; do sleep 30; done
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --tau-qs-rel 10"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
Q=work/ss_qa; H=work/ss_qh
stage2() {  # name outdir start extra-args...  -> 0 if result.h5 written
  local name=$1 out=$2 start=$3; shift 3
  [ -d $out ] && mv $out ${out}_fail_$(date +%H%M%S)
  for mode in noautotune cpu; do
    mkdir -p $out
    echo "CHAIN $name ($mode) start $(date +%T)"
    if [ $mode = noautotune ]; then
      XLA_FLAGS=--xla_gpu_autotune_level=0 $CAP \
        $PY run_one.py "$@" --vacuum --start "$start" --out $out --maxiter 400 > $out/run.log 2>&1
    else
      $CAP $PY run_one.py "$@" --vacuum --start "$start" --out $out --maxiter 400 --device cpu > $out/run.log 2>&1
    fi
    if [ -f $out/result.h5 ]; then echo "CHAIN $name ($mode) ok $(date +%T)"; return 0; fi
    echo "CHAIN $name ($mode) FAILED $(date +%T)"; mv $out ${out}_fail_${mode}
  done
  return 1
}
ss() {  # name outdir args...
  local name=$1 out=$2; shift 2
  mkdir -p $out; echo "CHAIN $name start $(date +%T)"
  $CAP $PY single_stage.py "$@" --out $out > $out/run.log 2>&1
  [ -f $out/coils_k4.h5 ] && echo "CHAIN $name ok $(date +%T)" || echo "CHAIN $name FAILED $(date +%T)"
}
stage2 Q1 $Q/xyz/stage2 "sweep_910_copy/initial starts/start_xyzN6.h5" --eq precise_QA --bounds bounds/precise_qa_sweep910.json --length-mult 0.8 \
  && ss Q2 $Q/xyz/ss --eq precise_QA --start $Q/xyz/stage2/result.h5 --length-mult 0.8 $SS
stage2 H3 $H/xyz/stage2 $H/start_xyzN6.h5 $QH \
  && ss H4 $H/xyz/ss $QH $QHSS --start $H/xyz/stage2/result.h5 $SS
mkdir -p $H/arcB2/ss_k4; echo "CHAIN H2-k4 start $(date +%T)"
$CAP $PY single_stage.py $QH $QHSS --eq $H/arcB2/ss/eq_k3.h5 --start $H/arcB2/ss/coils_k3.h5 --out $H/arcB2/ss_k4 \
  --tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526 --kmin 4 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 > $H/arcB2/ss_k4/run.log 2>&1
[ -f $H/arcB2/ss_k4/coils_k4.h5 ] && echo "CHAIN H2-k4 ok $(date +%T)" || echo "CHAIN H2-k4 FAILED $(date +%T)"
echo "CHAIN all done $(date +%T)"
