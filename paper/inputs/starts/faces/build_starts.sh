#!/bin/bash
# RR-02 starts, built ONE PROCESS AT A TIME (cwd coil_sandbox, laptop CPU). Measured peak RSS: make_start 2.26 GB,
# flatten 1.48 GB. Never parallelize this: 10 at once took WSL down on 2026-10-26.
# Per (config, seed): the circles once (planar1.h5 = make_start --rep planar --N 8), then every other class converted
# FROM those circles with --from-coils (modular = to_FourierXYZ N 8; arcN = chord fit, B = N, M 5, hinges at equal
# parameter intervals from s = 0, or from s = f/B of a turn when the dir has hinge_phase.txt = f, the hinge
# rotation as a fraction of one face: the hinge-rotation seeds), arcs flattened to one CoilSet(NFP, sym) by flatten.py (exact, checked).
# Usage: bash from_axis/rr02/build_starts.sh [CFG_sSEED ...]    (default: every dir under from_axis/rr02/starts)
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/home/singh/miniforge3/envs/desc-env2/bin/python}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4
R=from_axis/rr02/starts
dirs=("$@"); [ ${#dirs[@]} -eq 0 ] && dirs=($(cd $R && ls -d QA_s* QH_s*))
for name in "${dirs[@]}"; do
  d=$R/$name; c=${name%%_s*}; r=$(cat $d/r_over_a.txt)
  f=$( [ -f $d/hinge_phase.txt ] && cat $d/hinge_phase.txt || echo 0 )
  CL=${CLASSES:-$( [ -f $d/hinge_phase.txt ] && echo "arc2 arc3 arc4 arc5 arc6 arc7 arc8" || echo "modular arc2 arc3 arc4 arc5 arc6 arc7 arc8" )}
  if [ ! -f $d/planar1.h5 ]; then
    $PY make_start.py --eq precise_$c --nc 4 --r-over-a $r --rep planar --N 8 --vacuum --device cpu \
      --bounds bounds/rr02_precise_$c.json --out $d/planar1.h5 > $d/planar1.log 2>&1 || echo "FAIL $name planar1"
  fi
  C="--eq precise_$c --vacuum --device cpu --bounds bounds/rr02_precise_$c.json --from-coils $d/planar1.h5"
  for cls in $CL; do
    [ -f $d/$cls.done ] && continue
    if [ $cls = modular ]; then
      $PY make_start.py $C --rep xyz --N 8 --out $d/modular.h5 > $d/modular.log 2>&1 && touch $d/modular.done || echo "FAIL $name modular"
    else
      B=${cls#arc}; ph=$(python3 -c "print($f / $B)")
      $PY make_start.py $C --rep arc --B $B --M 5 --hinges phase --phase $ph --out $d/${cls}_mixed.h5 > $d/$cls.log 2>&1 &&
      $PY from_axis/rr02/flatten.py $d/${cls}_mixed.h5 $d/$cls.h5 --eq precise_$c --device cpu > $d/${cls}_flatten.log 2>&1 &&
      touch $d/$cls.done || echo "FAIL $name $cls"
    fi
    echo "$(date +%H:%M:%S) $name $cls $( [ -f $d/$cls.done ] && echo ok || echo FAILED)"
  done
done
echo "build_starts done"
