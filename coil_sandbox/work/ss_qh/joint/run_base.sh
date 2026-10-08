#!/bin/bash
# Joint-plane baseline on precise_QH (fixed equilibrium): arcB2 M5 with every hinge held at z = 0,
# plus the same cold start with hinges free, to separate the joint plane's cost from the cold start's.
# Settings match the matrix arcB2 runs (work/ss_qh/matrix/step1_base.sh) except the start and the flag.
# One run at a time, capped. Usage: run_base.sh [z0|free|both]
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 --cc-gap coils --cc-k 80000"
START=$J/starts/z0_start_packed.h5
run() { name=$1; shift; mkdir -p $J/$name; echo "=== start $name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $START --out $J/$name "$@" > $J/$name/run.log 2>&1; echo "=== end $name exit $? $(date +%T)"; sleep 10; }
case ${1:-both} in
  z0)   run base_z0 --fix-hinge-z 0 ;;
  free) run base_free ;;
  free_link) run base_free_link --link-mu 1e8 ;;
  level) START=$J/base_z0/result.h5 run base_level --horizontal-hinges --link-mu 1e8 ;;
  both) run base_z0 --fix-hinge-z 0; run base_free ;;
esac
# 2026-10-08, after base_free linked coils 0-1 in outer 1: same free run with the link guard (user's minimal fix):
#   run_base.sh free_link  ->  initial mu 1e8 on the 72 signed per-pair coil-coil rows only
# 2026-10-08 (user): per-coil horizontal hinges (each coil level at its own height) from the converged base_z0 coils:
#   run_base.sh level  ->  base_level; compare with base_z0 (1.801e-2) and base_free_link (6.421e-3)
