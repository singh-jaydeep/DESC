#!/bin/bash
# arcB2 M5 stage 2 on QUASR 1328095 (fixed equilibrium; nfp 4 QH, A 8, a 0.125, Psi rescaled to precise_QH's <|B|>),
# from one start: lens arches over a level chord at each cross-section's centre height (make_z0_start.py --chord-z centre).
#   free:  hinges free;            level: hinges level per coil (--horizontal-hinges). Both with the link guard.
# Settings as run_base.sh on precise_QH (bounds precise_qh_gil.json apply: a matches to 0.6%).
# Usage: run.sh free | level | both     (one job at a time, capped)
cd /home/singh/Documents/DESC2/coil_sandbox
Q=work/ss_qh/joint/quasr1328095
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
COMMON="--eq $Q/eq_q1328095.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8"
START=$Q/starts/level_start_packed.h5
run() { name=$1; shift; mkdir -p $Q/$name; echo "=== start $name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $START --out $Q/$name "$@" > $Q/$name/run.log 2>&1; echo "=== end $name exit $? $(date +%T)"; sleep 10; }
case ${1:-both} in
  free)  run free ;;
  level) run level --horizontal-hinges ;;
  both)  run free; run level --horizontal-hinges ;;
  free_warm) START=$Q/level/result.h5 run free_warm ;;
esac
# 2026-10-08: free (cold, lens start) fell into a poor basin in outer 1 (B.n 0.1005, hinges 0.5-0.75 m up, one 47 deg
# apart in phi), while level from the same start reached 5.64e-3. Fair free control: warm-started from the level result.
case done in
esac
