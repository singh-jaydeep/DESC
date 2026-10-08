#!/usr/bin/env bash
# Varied cut-angle scan for polar arcB2 on the 16-coil helios configuration.
#
# Design (work/helios/arc16/design.json): 6 runs x 4 unique coils, every COLUMN a permutation of
# {0,30,60,90,120,150}, so each coil tries every angle exactly once. Pair-difference coverage
# 30/36 -- each coil pair sees 5 of the 6 possible differences. A fully balanced design does not
# exist at order 6 (no complete mapping of Z6), the same obstruction the 12-coil mixed design hit.
#
# Uniform cut 30 (work/helios/arc16/cut30) is already done and serves as the difference-0 reference.
#
# Serial, one GPU job at a time (machine rule). RESUMABLE: skips any label that already has
# legB/result.h5, so it can be interrupted and restarted.
#
#   usage: ./run_scan.sh            run the whole design
#          ./run_scan.sh A,B,C,D    run one label only
set -u
cd "$(dirname "$0")"
DESIGN="60,150,60,60 0,0,120,30 90,30,150,0 30,60,30,90 120,90,90,150 150,120,0,120"
for T in ${1:-$DESIGN}; do
  LAB=${T//,/_}
  D="cut$LAB"
  if [ -f "$D/legB/result.h5" ]; then echo "[$LAB] done already, skipping"; continue; fi
  echo "=== [$LAB] start $(date +%H:%M:%S) ==="
  FREE_IT=250 ./run_chain.sh "$T" > "$D/chain.log" 2>&1
  rc=$?
  if [ -f "$D/legB/result.h5" ]; then
    echo "[$LAB] $(grep '^RESULT' "$D/legB.log" | tail -1)"
  else
    echo "[$LAB] FAILED (exit $rc); see $D/chain.log"
  fi
done
echo "=== SCAN DONE $(date +%H:%M:%S) ==="
echo "--- summary ---"
for d in cut*/; do
  [ -f "$d/legB.log" ] || continue
  printf "%-22s %s\n" "${d%/}" "$(grep '^RESULT' "$d/legB.log" | tail -1)"
done
