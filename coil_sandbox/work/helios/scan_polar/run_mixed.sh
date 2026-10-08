#!/usr/bin/env bash
# Queue for the mixed cut-angle design (build_mixed.sh), serial, one GPU job at a time.
# Resumable: a label whose legB/result.h5 exists is skipped; an interrupted label reruns from its fit.
cd "$(dirname "$0")"
DESIGN="0,120,60 0,150,120 30,90,0 30,120,60 60,90,150 90,60,30 90,150,0 120,30,90 120,60,150 150,0,30"
for t in ${1:-$DESIGN}; do
  IFS=, read -r a b c <<< "$t"; L=c0x${a}_c1x${b}_c2x${c}
  [ -f "cut$L/legB/result.h5" ] && { echo "$L: done already, skipping"; continue; }
  echo "$L: start $(date +%H:%M)"
  FREE_IT=250 ./run_chain.sh "$L" > "cut$L/chain.log" 2>&1; echo "$L: exit $? $(date +%H:%M)"
done
echo "MIXED DONE"
