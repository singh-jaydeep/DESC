#!/bin/bash
# Sequential chain (2026-09-29): QA xyzN6 (Q1 stage 2, Q2 single stage), QH arcB2 (H1, H2), QH xyzN6 (H3, H4),
# QA arcB2 clean route (Q3). One job at a time under the cgroup cap. A failed step skips its dependents only.
# Stage 2 (run_one) checkpoints every 50 iterations; single stage saves eq_kK/coils_kK after every k step.
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
SS_COMMON="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --tau-qs-rel 10"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QH_SS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
QA_S2="--eq precise_QA --bounds bounds/precise_qa_sweep910.json --length-mult 0.8"
step() {  # name outdir expected_file command...
  local name=$1 out=$2 want=$3; shift 3
  mkdir -p $out
  echo "CHAIN $name start $(date +%T)"
  $CAP "$@" > $out/run.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f $out/$want ]; then echo "CHAIN $name ok $(date +%T)"; return 0; fi
  echo "CHAIN $name FAILED rc=$rc $(date +%T)"; return 1
}
Q=work/ss_qa; H=work/ss_qh
step Q1 $Q/xyz/stage2 result.h5 $PY run_one.py $QA_S2 --vacuum --start "sweep_910_copy/initial starts/start_xyzN6.h5" --out $Q/xyz/stage2 --maxiter 400 \
  && step Q2 $Q/xyz/ss coils_k4.h5 $PY single_stage.py --eq precise_QA --start $Q/xyz/stage2/result.h5 --out $Q/xyz/ss --length-mult 0.8 $SS_COMMON
step H1 $H/arcB2/stage2 result.h5 $PY run_one.py $QH --vacuum --start $H/start_arcB2_M5.h5 --out $H/arcB2/stage2 --maxiter 400 \
  && step H2 $H/arcB2/ss coils_k4.h5 $PY single_stage.py $QH $QH_SS --start $H/arcB2/stage2/result.h5 --out $H/arcB2/ss $SS_COMMON
step H3 $H/xyz/stage2 result.h5 $PY run_one.py $QH --vacuum --start $H/start_xyzN6.h5 --out $H/xyz/stage2 --maxiter 400 \
  && step H4 $H/xyz/ss coils_k4.h5 $PY single_stage.py $QH $QH_SS --start $H/xyz/stage2/result.h5 --out $H/xyz/ss $SS_COMMON
step Q3 $Q/arcB2_clean/ss coils_k4.h5 $PY single_stage.py --eq precise_QA --start $Q/stage2_L0p8_M5/result.h5 --out $Q/arcB2_clean/ss --length-mult 0.8 $SS_COMMON
echo "CHAIN done $(date +%T)"
