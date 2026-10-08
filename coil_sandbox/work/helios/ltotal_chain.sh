#!/usr/bin/env bash
# The cut30 chain re-run from its ORIGINAL fit with the total-length budget in force from the start
# (repair -> 50 iterations with hinges held -> free stage), so the optimizer never sees the per-coil
# cap. Compare with work/helios/clearance/ltotal108, which only refined the finished cut30 result.
set -u
cd "$(dirname "$0")/../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
EQ=sample_equilibria/helios_repro.h5
BND=bounds/helios_Ltotal108.json
D=work/helios/ltotal_chain
FREE_IT=${FREE_IT:-250}
mkdir -p $D

run() {
  local log=$1; shift
  "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$log"; then
    echo "CUDA fault in $log, retrying once"; mv "$log" "$log.fail1"
    "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  fi
  grep -q "^wrote" "$log" || { echo "FAILED: $log"; exit 1; }
}

echo "repair $(date +%H:%M)"
run $D/repair.log --eq $EQ --bounds $BND --start work/helios/scan_polar/cut30/fit.h5 --out $D/repair \
  --mode feasibility --anchor-weight 1.0 --margin 0.25 --margin-skip Ltot --maxiter 120 \
  --check-every 10 --ckpt-every 50 --convex-form signed

echo "stage 2, hinges held $(date +%H:%M)"
run $D/legA.log --eq $EQ --bounds $BND --start $D/repair/result.h5 --out $D/legA --fix-hinges \
  --maxiter 50 --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl

echo "stage 2, free $(date +%H:%M)"
run $D/legB.log --eq $EQ --bounds $BND --start $D/legA/result.h5 --out $D/legB \
  --maxiter $FREE_IT --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl
echo "done $(date +%H:%M)"
