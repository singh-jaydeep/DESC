#!/bin/bash
# After queue2: the FourierXYZ single-stage steps. Q2 died on the GPU (cuFFT plan: scratch allocation failed, i.e.
# likely VRAM exhaustion; the XYZ stage-2 runs died of illegal-address errors, also only for XYZ). Try the GPU with
# a 201-node source grid (--src-n 100; MEASURED identical to the default 801 nodes to 1e-15 for these N=6 coils,
# 4x less Biot-Savart memory), then small chunks as well, then CPU. H4 only if H3 produced a stage-2 result and H4 did not finish.
cd /home/singh/Documents/DESC2/coil_sandbox
until grep -q "^CHAIN all done" work/compare/queue2.log; do sleep 30; done
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --tau-qs-rel 10"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
Q=work/ss_qa; H=work/ss_qh
ss_retry() {  # name outdir args...
  local name=$1 out=$2; shift 2
  [ -d $out ] && mv $out ${out}_fail_$(date +%H%M%S)
  for mode in gpu-src100 gpu-src100-chunked cpu-src100; do
    mkdir -p $out; echo "CHAIN $name ($mode) start $(date +%T)"
    case $mode in
      gpu-src100)         $CAP $PY single_stage.py "$@" --out $out --src-n 100 > $out/run.log 2>&1 ;;
      gpu-src100-chunked) $CAP $PY single_stage.py "$@" --out $out --src-n 100 --bs-chunk 100 --jac-chunk 20 > $out/run.log 2>&1 ;;
      cpu-src100)         $CAP $PY single_stage.py "$@" --out $out --src-n 100 --device cpu > $out/run.log 2>&1 ;;
    esac
    if [ -f $out/coils_k4.h5 ]; then echo "CHAIN $name ($mode) ok $(date +%T)"; return 0; fi
    echo "CHAIN $name ($mode) FAILED $(date +%T)"; mv $out ${out}_fail_${mode}
  done
}
ss_retry Q2 $Q/xyz/ss --eq precise_QA --start $Q/xyz/stage2/result.h5 --length-mult 0.8 $SS
if [ -f $H/xyz/stage2/result.h5 ] && [ ! -f $H/xyz/ss/coils_k4.h5 ]; then
  ss_retry H4 $H/xyz/ss $QH $QHSS --start $H/xyz/stage2/result.h5 $SS
fi
echo "CHAIN queue3 done $(date +%T)"
