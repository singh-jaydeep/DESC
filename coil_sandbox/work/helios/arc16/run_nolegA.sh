#!/usr/bin/env bash
# Is the 50-iteration hinges-HELD stage (leg A) itself creating the subdominant coil?
#
# Identical to the cut-30 chain except leg A is SKIPPED: repair -> free stage directly.
# Budget is 300 free iterations = the 50 (legA) + 250 (legB) that cut 30 got, so the ONLY
# difference from cut30 is whether the hinges were frozen for the first 50 iterations.
#
# Compare against: cut30 legB 8.7195e-3, final currents 5.33/21.16/19.56/18.91 (min/max 0.25).
# Question: do the hinges "emerge" differently when never frozen, and is a subdominant coil
# still there?  (HANDOFF next-step 0.)
set -u
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
BND=bounds/helios_kruger2026.json
D=work/helios/arc16/nolegA_cut30
IT=${IT:-300}
mkdir -p "$D"

systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
  "$PY" run_one.py --eq $EQ --bounds $BND \
  --start work/helios/arc16/cut30/repair/result.h5 --out "$D/free" \
  --maxiter "$IT" --check-every 25 --ckpt-every 50 --convex-form signed \
  --diag-log diag.jsonl > "$D/free.log" 2>&1

if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$D/free.log"; then
  echo "CUDA fault, retrying once"; mv "$D/free.log" "$D/free.log.fail1"
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
    "$PY" run_one.py --eq $EQ --bounds $BND \
    --start work/helios/arc16/cut30/repair/result.h5 --out "$D/free" \
    --maxiter "$IT" --check-every 25 --ckpt-every 50 --convex-form signed \
    --diag-log diag.jsonl > "$D/free.log" 2>&1
fi
grep -E "^(START|RESULT|SOLVER)" "$D/free.log"
