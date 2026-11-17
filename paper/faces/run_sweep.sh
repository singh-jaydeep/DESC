#!/bin/bash
# Faces sweep (RR-02 style): field error vs number of planar faces per coil on precise_QA and precise_QH, stage 2 on
# paper/stage2.py from the starts in paper/inputs/starts/faces, then the one scoring function per run.
#
#   bash paper/faces/run_sweep.sh [CFG ...]          CFG in QA QH (default both)
#
# Classes: planar1 (FourierPlanar N 8), arc2 .. arc8 (B faces, M 5, free hinges), modular (FourierXYZ N 8, the
# smooth-coil floor). Seeds: QA 0-4 (r/a), QA 5-7 (hinge rotations, arcs only); QH 0-2 (r/a), QH 3-9 (rotations).
# FOLD=deg caps every arc coil's fold (the paper's corner control; RR-02 had none); SEEDS="0 1" / CLASSES="arc4"
# restrict the sweep. One job at a time under a 12 GB cap; a finished run (result.json present) is skipped.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export XLA_FLAGS=--xla_gpu_autotune_level=0
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
PY=${PY:-python}
IN=paper/inputs
OUT=${RUNS:-paper/runs/faces}
for cfg in ${*:-QA QH}; do
  if [ "$cfg" = QA ]; then seeds=${SEEDS:-0 1 2 3 4 5 6 7}; else seeds=${SEEDS:-0 1 2 3 4 5 6 7 8 9}; fi
  for s in $seeds; do
    rot=$( [ "$cfg" = QA ] && [ "$s" -ge 5 ] || { [ "$cfg" = QH ] && [ "$s" -ge 3 ]; } && echo 1 || echo 0)
    for cls in ${CLASSES:-planar1 arc2 arc3 arc4 arc5 arc6 arc7 arc8 modular}; do
      [[ $rot == 1 && ( $cls == planar1 || $cls == modular ) ]] && continue   # rotations change arcs only
      d=$OUT/${cfg}_s$s/$cls
      [ -f "$d/result.json" ] && continue
      mkdir -p "$d"
      x=""
      [[ -n "${FOLD:-}" && $cls == arc* ]] && x="--fold-max $FOLD"
      echo "=== $cfg s$s $cls start $(date +%T)"
      $CAP $PY paper/stage2.py --eq precise_$cfg --bounds $IN/bounds/rr02_precise_$cfg.json \
        --start $IN/starts/faces/${cfg}_s$s/$cls.h5 --vacuum --current-band 0.5 --cc-gap coils --cc-k 80000 \
        --maxiter 800 $x --out "$d" > "$d/stdout.log" 2>&1
      grep -E "^SOLVER|^RESULT" "$d/run.log" | cut -c1-250
      [ -f "$d/result.h5" ] && $CAP $PY paper/faces/score.py "$d/result.h5" precise_$cfg "$d/score.json" \
        > "$d/score.log" 2>&1
    done
  done
done
echo "=== faces sweep done $(date +%T)"
