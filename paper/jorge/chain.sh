#!/bin/bash
# Jorge 2022 QI (vacuum, NFP 1) stage-2 chain of the paper (PORT_PLAN 9.4), on paper/stage2.py.
#
#   bash paper/jorge/chain.sh [PHASE ...]      phases: ladder A3 C P trace (default: all, in order)
#
# Plasma-coil distance: the hard minimum (stage2.py --pc-mode min, as the poster's --dist hybrid) except in the
# planar control P, where it stalls and the smooth rows are used.
# ladder  level hinges near the axis (+-0.025 m), fold cap 10 -> 60 deg, from inputs/starts/jorge/J_arc90.h5;
#         F60 is R0 (poster: mean B.n 1.41e-2, max 17.5%)
# A3      tilt <= 20 deg, mid-height +-0.025 m, fold <= 60, from R0 (poster: 6.43e-3 / 11.0%)
# C       A3 + pointwise |B.n|/|B| cap 4% then 2% (poster C_bn004 5.52e-3 / 3.9%, C_bn002 5.30e-3 / 2.3%)
# P       planar control: FourierPlanar N 8 from J_planar_r3.6.h5, no cap (poster P0c 3.34e-2); P_CAPS="0.12 ..."
#         continues with a cap ladder
# trace   field lines of C_bn002, C_bn004, A3, R0 (paper/trace_lines.py)
#
# Every step: one job at a time under a 12 GB cap; stops the chain if a step ends linked or crashed, flags a step
# that does not converge. Run from the repository root. RUNS=... sets the output folder.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92}   # the B.n cap needs it on an 8 GB GPU
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
PY=${PY:-python}
IN=paper/inputs
EQ=$IN/equilibria/jorge_vac.h5
BND=$IN/bounds/qi_jorge2022.json
OUT=${RUNS:-paper/runs/jorge}
BASE="--eq $EQ --bounds $BND --current-band 0.5 --vacuum --maxiter 800 --cc-gap bounds --row-n 512 --cc-k 30000
--pc-plasma-mn 48,96 --pc-plasma-sym half --pc-row-n 128"
# the planar control stalls on the hard plasma-coil minimum (poster P0/P0b): smooth top-k rows there; the 128 x 512
# plasma grid keeps the rows' gap at 5.8 mm on Jorge's boundary (curvature up to 100 /m)
PROWS="--pc-mode rows --pc-plasma-mn ${PCMN:-128,512} --pc-k ${PCK:-40000}"
PHASES=${*:-ladder A3 C P trace}

run() {  # run NAME START "FLAGS": 0 converged and unlinked, 1 not converged, 2 crashed, 3 linked
  local n=$1 start=$2 x=$3 d=$OUT/$1
  mkdir -p "$d"
  echo "=== $n start $(date +%T) from $start"
  $CAP $PY paper/stage2.py $BASE --start "$start" $x --out "$d" > "$d/stdout.log" 2>&1
  grep -E "^SOLVER|^RESULT" "$d/run.log" | cut -c1-300
  $PY - "$d/result.json" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("=== no result.json (crashed?)"); sys.exit(2)
if r["result"]["linked"]:
    print("=== ended LINKED"); sys.exit(3)
if not r["solver"]["success"]:
    print("=== did NOT converge (flagged)"); sys.exit(1)
EOF
}

has() { [[ " $PHASES " == *" $1 "* ]]; }
LEVEL="--horizontal-hinges --hinge-zmax-rel 0.025"

if has ladder; then
  start=$IN/starts/jorge/J_arc90.h5; sides=""
  for f in 10 20 30 45 60; do
    run ladder/F$f "$start" "$LEVEL --fold-max $f $sides"; rc=$?
    [ $rc -ge 2 ] && { echo "=== ladder stopped at F$f"; exit 1; }
    start=$OUT/ladder/F$f/result.h5
    [ -z "$sides" ] && sides="--fold-sides-from $OUT/ladder/F$f/result.json"
  done
fi
SIDES="--fold-sides-from $OUT/ladder/F10/result.json"
R0=$OUT/ladder/F60/result.h5
TILT="--hinge-tilt-max 20 --hinge-zmax-rel 0.025 --fold-max 60 $SIDES"

if has A3; then
  run A3_tilt20 "$R0" "$TILT" || [ $? -eq 1 ] || exit 1
fi

if has C; then
  start=$OUT/A3_tilt20/result.h5
  for c in ${C_CAPS:-0.04 0.02}; do
    n=C_bn$(echo $c | tr -d .)
    run $n "$start" "$TILT --bn-max $c --bn-max-grid 48,96 --jac-chunk ${JC:-16}" || [ $? -eq 1 ] || exit 1
    start=$OUT/$n/result.h5
  done
fi

if has P; then
  start=$IN/starts/jorge/J_planar_r3.6.h5
  run P0 "$start" "--row-n ${PROWN:-384} $PROWS" || [ $? -eq 1 ] || exit 1
  start=$OUT/P0/result.h5
  for c in ${P_CAPS:-}; do
    n=P_bn$(echo $c | tr -d .)
    run $n "$start" "--row-n ${PROWN:-384} $PROWS --bn-max $c --bn-max-grid 48,96 --jac-chunk ${JC:-16}" || [ $? -eq 1 ] || exit 1
    start=$OUT/$n/result.h5
  done
fi

if has trace; then
  for n in ${TRACE:-C_bn002 C_bn004 A3_tilt20 ladder/F60}; do
    o=$OUT/traces/$(basename $n)
    mkdir -p "$o"
    echo "=== trace $n start $(date +%T)"
    $CAP $PY paper/trace_lines.py "$OUT/$n/result.h5" "$EQ" "$o" --ntransit 150 \
      --offsets 0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20 > "$o/run.log" 2>&1
    tail -3 "$o/run.log" | cut -c1-200
  done
fi
echo "=== jorge chain done $(date +%T)"
