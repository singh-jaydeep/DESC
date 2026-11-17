#!/bin/bash
# STATUS (2026-11-17): superseded by paper/EXPERIMENTS.md section 3, which governs: run on the REBUILT design F
# (fold-60 rung) with bounds gil_a0p12_nc5.json, cold starts on that equilibrium, 4 seeds per arm, the pQA set
# retired. The paths and bounds below are the poster-era ones (fa_qa2_a0167 is retired).
# Planar vs hinged vs free coils at equal bounds (PORT_PLAN 9.2), stage 2 on paper/stage2.py.
#
#   bash paper/compare/run.sh [F] [pQA]        (default both)
#
# F    design F's equilibrium and bounds: planar (FourierPlanar N 8, circles r/a 3.4), hinged (design F's class:
#      every hinge on z = 0, fold <= 60, from the z = 0 lens start), free (FourierXYZ N 8)
# pQA  precise_QA control (bounds_pqa: fa_qa2 scaled to precise_QA's a). Hinged = level hinges per coil
#      (--horizontal-hinges) + fold <= 60: snapping start_arc's hinges to z = 0 moves them 15 cm and makes the start
#      infeasible (d_cc 0.75, d_pc 0.71 of the bounds); a z = 0 arm needs a lens start like F's (make_z0_start.py,
#      in paper/inputs/starts/F for provenance, not yet ported as a runnable builder)
# Then each result is scored by paper/lib/dense_check (in run.log / result.json). One job at a time, 12 GB cap.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export XLA_FLAGS=--xla_gpu_autotune_level=0
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
PY=${PY:-python}
IN=paper/inputs
OUT=${RUNS:-paper/runs/compare}
COMMON="--vacuum --current-band 0.5 --maxiter 800 --cc-gap coils --cc-k 80000"

run() {  # run NAME EQ BOUNDS START "FLAGS"
  local d=$OUT/$1
  [ -f "$d/result.json" ] && return 0
  mkdir -p "$d"
  echo "=== $1 start $(date +%T)"
  $CAP $PY paper/stage2.py --eq "$2" --bounds "$3" --start "$4" $COMMON $5 --out "$d" > "$d/stdout.log" 2>&1
  grep -E "^SOLVER|^RESULT" "$d/run.log" | cut -c1-250
}

for set_ in ${*:-F pQA}; do
  if [ "$set_" = F ]; then
    EQ=$IN/references/designF/eq.h5; B=$IN/bounds/fa_qa2_a0167.json; S=$IN/starts/F
    run F/planar "$EQ" "$B" $S/F_planar_r3.4.h5 ""
    run F/hinged "$EQ" "$B" $S/F_z0_c25_p07.h5 "--fix-hinge-z 0 --fold-max 60 --fold-sides-from $S/F10_fold_sides.json"
    run F/free "$EQ" "$B" $S/F_xyz_r3.4.h5 ""
  else
    EQ=precise_QA; B=$IN/bounds/bounds_pqa.json; S=$IN/starts/compare_pqa
    run pQA/planar "$EQ" "$B" $S/start_planar.h5 ""
    run pQA/hinged "$EQ" "$B" $S/start_arc.h5 "--horizontal-hinges --fold-max 60"
    run pQA/free "$EQ" "$B" $S/start_xyz.h5 ""
  fi
done
echo "=== compare done $(date +%T)"
