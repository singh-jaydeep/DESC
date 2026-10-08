#!/usr/bin/env bash
# Why doesn't the 4th coil help arcB2 at 16 coils?  Constraint sensitivity, warm-restarted
# from the best 16-coil arcB2 result (cut30 legB, bn 8.7195e-3) with ONE bound changed.
#
# Motivation (measured, all five 16-coil arcB2 runs):
#   d_cc sits on its ENFORCED bound (1.243 m vs 1.2451) in ALL FIVE runs
#   d_pc sits on its enforced bound (1.222 m vs 1.224) in THREE of five, incl. the two best
#   kappa is only 0.32-0.47 of its bound, while FourierXYZ at 16 coils uses 1.00
# -> the restricted coils are SPACE-limited, not curvature-limited.
#
#   dpc10  d_pc 1.2 -> 1.0 m   (user's hypothesis: give them room toward the plasma)
#   dcc10  d_cc 1.2 -> 1.0 m   (the universally binding one; note the 12-coil version of this
#                               test bought nothing, but packing is far tighter at 16 coils)
#   L30    L 36 -> 30 m        (user's idea: shorter coils stay in their own area and stop
#                               crowding neighbours. Pure-length elasticity d(ln bn)/d(ln L)
#                               = -2.1 predicts 8.72e-3 -> 1.28e-2 if ONLY length matters, so
#                               anything better than 1.28e-2 means the freed space paid.)
#
# Serial, one GPU job at a time. Resumable: skips a label that already has result.h5.
set -u
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
START=work/helios/arc16/cut30/legB/result.h5
IT=${IT:-300}
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)

run_one_case() {
  local lab=$1 bnd=$2
  local D=work/helios/arc16/sens_$lab
  mkdir -p "$D"
  if [ -f "$D/result.h5" ]; then echo "[$lab] done already, skipping"; return; fi
  echo "=== [$lab] bounds/$bnd.json  $(date +%H:%M:%S) ==="
  "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds "bounds/$bnd.json" \
    --start "$START" --out "$D" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
    --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$D/run.log"; then
    echo "[$lab] CUDA fault, retrying once"; mv "$D/run.log" "$D/run.log.fail1"
    "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds "bounds/$bnd.json" \
      --start "$START" --out "$D" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
      --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  fi
  grep -E "^(START|RESULT)" "$D/run.log" || echo "[$lab] FAILED"
}

run_one_case dpc10 helios_dpc1.0
run_one_case dcc10 helios_dcc1.0_16
run_one_case L30   helios_L30
echo "=== SENSITIVITY DONE $(date +%H:%M:%S) ==="
for d in work/helios/arc16/sens_*/; do
  [ -f "$d/run.log" ] || continue
  printf "%-28s %s\n" "${d%/}" "$(grep '^RESULT' "$d/run.log" | tail -1)"
done
