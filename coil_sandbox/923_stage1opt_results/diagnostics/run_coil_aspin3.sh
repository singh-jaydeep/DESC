#!/usr/bin/env bash
# MATCHED-QUALITY coil test (2026-09-23). See NOTES 9m.
#
# The 9l ladder's equilibria carried force-balance errors spanning 500x (3.3e-6, 2.3e-4,
# 1.75e-3), monotonically with eps_1, so "coils got worse" and "equilibrium got worse" are
# confounded there. `live_eps1_L1_lsqfb/eq_lo.h5` is the first eps_1-optimized equilibrium
# that is resolution-CONVERGED (L=8 -> L=12 gap 0.29 pts) and carries baseline-quality force
# balance (1.56e-4 vs the 9l legs' 2.3e-4 / 1.75e-3). eps_1 -27.21%.
#
# IDENTICAL coil problem to run_coil_validation.sh so the existing baseline runs are the
# control: same bounds file, --nc 4 (16 coils), r/a 2.5 cold start, MAXIT 200, --vacuum,
# --qf-chunk 5 (REQUIRED for xyz: 4/4 die without it).
set -u
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
B=bounds/precise_qa_planar.json
OUT=922_stage1opt_results/runs/coilval
MAXIT=${MAXIT:-200}
mkdir -p $OUT
tag=aspin3
EQPATH=922_stage1opt_results/runs/live_eps1_L1_aspin3/eq_lo.h5

for rep in planar xyz; do
  st=$OUT/start_${rep}_${tag}.h5
  rd=$OUT/${tag}_${rep}
  if [ ! -f "$st" ]; then
    echo "### start $tag $rep ($(date +%H:%M))"
    $PY make_start.py --eq "$EQPATH" --rep $rep --nc 4 \
        --r-over-a 2.5 --bounds $B --out "$st" --device cpu 2>&1 | tail -3
  fi
  if [ -f "$rd/result.h5" ]; then echo "### skip $tag $rep (done)"; continue; fi
  for i in $(seq 1 10); do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
    echo "    waiting for GPU to clear ($i)"; sleep 10
  done
  for attempt in 1 2 3; do
    echo "### stage2 $tag $rep attempt $attempt ($(date +%H:%M))"
    systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=0 \
      $PY run_one.py --eq "$EQPATH" --start "$st" --bounds $B --vacuum \
      --mode stage2 --maxiter $MAXIT --device gpu --qf-chunk 5 --out "$rd" \
      > $OUT/${tag}_${rep}.log 2>&1
    [ -f "$rd/result.h5" ] && break
    echo "    attempt $attempt failed: $(grep -oE 'CUDA_ERROR_[A-Z_]*|Traceback' $OUT/${tag}_${rep}.log | head -1)"
    sleep 15
  done
  grep -E "RESULT bn" $OUT/${tag}_${rep}.log | tail -1
done
echo "### MATCHED-QUALITY DONE $(date +%H:%M)"
