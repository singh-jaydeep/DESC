#!/usr/bin/env bash
# Mixed cut-angle design (2026-09-17): 10 polar arcB2 starts in which every coil pair sees every
# nonzero cut-angle difference (mod 180) exactly twice. With the six equal-angle runs (difference 0)
# this covers all differences. No exactly balanced 6-run design exists (Z6 has no complete mapping).
# Writes cutc0xA_c1xB_c2xC/fit.h5 (+ .json, fit_diag.png) for each; run each with run_chain.sh c0xA_c1xB_c2xC.
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
DESIGN="0,120,60 0,150,120 30,90,0 30,120,60 60,90,150 90,60,30 90,150,0 120,30,90 120,60,150 150,0,30"
for t in ${1:-$DESIGN}; do
  IFS=, read -r a b c <<< "$t"
  L=c0x${a}_c1x${b}_c2x${c}; D=work/helios/scan_polar/cut$L
  mkdir -p "$D"
  [ -f "$D/fit.h5" ] && { echo "[$L] fit exists, skipping"; continue; }
  "${CAP[@]}" "$PY" make_start.py --eq sample_equilibria/helios_repro.h5 --bounds bounds/helios_kruger2026.json \
    --from-coils work/helios/planarN7_final_it300.h5 --rep polararc --B 2 --M 5 --cut-angle "$t" \
    --out "$D/fit.h5" > "$D/fit.log" 2>&1 || { echo "[$L] make_start FAILED"; continue; }
  "$PY" work/helios/scan_polar/polar_diag.py "$D/fit_diag.png" "fit=$D/fit.h5" > "$D/fit_diag.log" 2>&1
  echo "[$L] $(grep -E '^  (fit max|L |B\.n|value)' "$D/fit.log" | tr -s ' ' | tr '\n' ';')"
done
