#!/usr/bin/env bash
# Shadow price of the coil-coil bound (2026-09-17): restart two finished results with d_cc moved,
# everything else identical, and compare bn. Loser cut90 (pinned at 1.2) gets room; winner cut60
# (slack, 1.70 m) gets squeezed.
cd "$(dirname "$0")/../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
EQ=sample_equilibria/helios_repro.h5
run() {  # label start bounds out
  echo "[$1] start $(date +%H:%M)"
  "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds "$3" --start "$2" --out "$4" \
    --maxiter 250 --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl > "$4.log" 2>&1
  grep -q "^wrote" "$4.log" && echo "[$1] done $(date +%H:%M)" || echo "[$1] FAILED, see $4.log"
}
mkdir -p work/helios/clearance
run relax90 work/helios/scan_polar/cut90/legB/result.h5  bounds/helios_dcc1.0.json work/helios/clearance/relax90
run tight60 work/helios/scan_polar/cut60/legB/result.h5  bounds/helios_dcc1.5.json work/helios/clearance/tight60
echo "CLEARANCE TEST DONE"
