#!/usr/bin/env bash
# Does eps_1 predict what PLANAR coils can achieve?
#
# Three equilibria spanning eps_1 0.0676 -> 0.0507 -> 0.0450, each given an IDENTICAL coil
# problem (absolute bounds, same coil count, same r/a cold start, no warm starts, no shared
# lineage). Both planar and FourierXYZ are run on each, because the response variable is the
# RATIO bn_planar/bn_free -- bn_planar alone cannot distinguish "better for planar coils"
# from "easier for all coils", and these equilibria differ in QS, iota and minor radius.
#
# --vacuum is REQUIRED: precise_QA has p=0 and I=0, and the sandbox defaults to finite beta.
#
# --qf-chunk 5 is REQUIRED for FourierXYZ on this GPU. The default qf_chunk=25 dies with
# CUDA_ERROR_ILLEGAL_ADDRESS every single time for rep=xyz (4/4 attempts) while rep=planar
# always succeeds -- so it looks like CLAUDE.md's documented 1-in-8 random fault but is
# deterministic and representation-specific. Chunking is mathematically identical: at
# qf-chunk 5 the GPU reproduces the CPU result to 5 digits (bn 1.0566e-01 both ways).
set -u
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
B=bounds/precise_qa_planar.json
OUT=922_stage1opt_results/runs/coilval
MAXIT=${MAXIT:-200}
mkdir -p $OUT

declare -A EQ=(
  [baseline]="precise_QA"
  [leg1]="922_stage1opt_results/runs/loose_eps1_L1/eq_lo.h5"
  [leg2]="922_stage1opt_results/runs/loose_eps1_L2/eq_lo.h5"
)

for tag in baseline leg1 leg2; do
  for rep in planar xyz; do
    st=$OUT/start_${rep}_${tag}.h5
    rd=$OUT/${tag}_${rep}
    if [ ! -f "$st" ]; then
      echo "### start $tag $rep"
      $PY make_start.py --eq "${EQ[$tag]}" --rep $rep --nc 4 --r-over-a 2.5 \
          --bounds $B --out "$st" --device cpu 2>&1 | grep -E "value / bound" | tail -1
    fi
    if [ -f "$rd/result.h5" ]; then echo "### skip $tag $rep (done)"; continue; fi
    # Verify the GPU is actually free first: a leftover run_one from a previous kill was
    # still holding 734 MiB and made two consecutive runs die with CUDA_ERROR_ILLEGAL_ADDRESS,
    # which looks exactly like the documented 1-in-8 random fault but is not.
    for i in 1 2 3 4 5 6 7 8 9 10; do
      [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
      echo "    waiting for GPU to clear ($i)"; sleep 10
    done
    # CLAUDE.md: CUDA_ERROR_ILLEGAL_ADDRESS hits ~1 run in 8 and an identical rerun passes.
    # Observed here on a clean GPU after a successful run, so retry up to twice.
    for attempt in 1 2 3; do
      echo "### stage2 $tag $rep attempt $attempt ($(date +%H:%M))"
      systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=0 \
        $PY run_one.py --eq "${EQ[$tag]}" --start "$st" --bounds $B --vacuum \
        --mode stage2 --maxiter $MAXIT --device gpu --qf-chunk 5 --out "$rd" \
        > $OUT/${tag}_${rep}.log 2>&1
      if [ -f "$rd/result.h5" ]; then break; fi
      echo "    attempt $attempt failed: $(grep -o 'CUDA_ERROR_[A-Z_]*' $OUT/${tag}_${rep}.log | head -1)"
      sleep 15
    done
    grep -E "RESULT bn" $OUT/${tag}_${rep}.log | tail -1
  done
done
echo "### ALL DONE $(date +%H:%M)"
