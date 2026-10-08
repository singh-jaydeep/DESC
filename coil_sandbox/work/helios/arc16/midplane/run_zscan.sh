#!/usr/bin/env bash
# Z-SCAN: horizontal joint chords at heights other than the midplane.
#
# Stellarator symmetry sends a hinge at z to -z, so a single pinned height z* != 0 gives the
# machine TWO horizontal joint planes, +z* and -z*, with the 16 coils split 8/8 between them.
# Only z* = 0 gives one plane. The scan is therefore symmetric in z*: only z* >= 0 is useful.
#
# Cheap RANKING only: cold start + feasibility repair + free stage at L28/d1.0, no ladder.
# The midplane ladder's ratio to the free-hinge baseline held between 2.09x and 2.41x across
# all five rungs, so the cold rung orders the heights without paying for four more ladders.
# Compare against the z*=0 cold rung, bn 2.6724e-2 (work/helios/arc16/midplane/free).
#
# Serial, one GPU job at a time. RESUMABLE: skips a height that already has free/result.h5.
set -u
cd "$(dirname "$0")/../../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
D=work/helios/arc16/midplane
CIRC=$D/circles_M5.h5
COLD=bounds/helios_cold_L28_d10.json
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)

# the same circles the midplane ladder was built from: r/a 2.5, 4 unique, M=5
if [ ! -f "$CIRC" ]; then
  "$PY" make_start.py --eq $EQ --nc 4 --r-over-a 2.5 --rep polararc --B 2 --M 5 \
    --bounds $COLD --out "$CIRC" --device cpu > "$D/circles.log" 2>&1 \
    || { echo "circles FAILED"; tail -3 "$D/circles.log"; exit 1; }
fi

for Z in ${ZS:-0.75 1.50 2.25}; do
  T=$D/z$Z
  if [ -f "$T/free/result.h5" ]; then echo "[z=$Z] done already, skipping"; continue; fi
  mkdir -p "$T"
  echo "=== z* = $Z  $(date +%H:%M:%S) ==="
  HS=$("$PY" "$D/level_hinges.py" "$CIRC" "$Z" 2>/dev/null | tail -1)
  [ -n "$HS" ] || { echo "  no hinge placement at z=$Z"; exit 1; }
  "$PY" make_start.py --eq $EQ --from-coils "$CIRC" --rep polararc --B 2 --M 5 \
    --hinge-s "$HS" --bounds $COLD --out "$T/start.h5" --device cpu > "$T/start.log" 2>&1 \
    || { echo "  make_start FAILED"; tail -3 "$T/start.log"; exit 1; }
  grep -E "fit max deviation|value / bound" "$T/start.log"

  for STAGE in repair free; do
    [ -f "$T/$STAGE/result.h5" ] && continue
    if [ "$STAGE" = repair ]; then
      ARGS=(--start "$T/start.h5" --mode feasibility --anchor-weight 1.0 --margin 0.10
            --margin-skip L --maxiter 300)
    else
      ARGS=(--start "$T/repair/result.h5" --maxiter 300 --diag-log diag.jsonl)
    fi
    # --demount-margin IS needed here, unlike the midplane ladder. At z* = 0 each coil crosses
    # the plane at its extreme inboard/outboard points, far outside the plasma's shadow, so
    # removability came free. At z* = 0.75 the crossings move to less extreme radii: the first,
    # unconstrained run put a hinge 0.293 m INSIDE the shadow with 1.12% of an arc blocked.
    # Affordable here because the cold rung carries no corner constraint -- it was corner +
    # demount together that doubled the distinct Hessian block sets and hit the 12 GB cap.
    "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $COLD --out "$T/$STAGE" \
      --fix-hinge-z "$Z" --demount-margin 0.3 "${ARGS[@]}" \
      --check-every 25 --ckpt-every 50 --so-chunk 8 \
      --convex-form signed > "$T/$STAGE.log" 2>&1
    grep -E "^RESULT" "$T/$STAGE.log"
    [ -f "$T/$STAGE/result.h5" ] || {
      echo "  FAILED: no result.h5 in $T/$STAGE"
      grep -E "Killed|MemoryError|CUDA_ERROR|Traceback" "$T/$STAGE.log" | tail -3; exit 1; }
  done
  "$PY" - "$T/free/result.h5" <<'PYEOF'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S, demount as D
from desc.io import load
from corners import corner_angles
cs = load(sys.argv[1]); eq = load("sample_equilibria/helios_repro.h5")
I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
a = np.concatenate([corner_angles(c) for c in S.iter_unique(cs)])
rep = D.demount_report(cs, eq)
cl = min(min(r["hinge_clearance"]) for r in rep)
bf = max(x["blocked_frac"] for r in rep for x in r["arcs"])
print(f"   corners max {a.max():.1f}   currents min/max {I.min()/I.max():.2f}"
      f"   demount clearance {cl:+.3f} m, blocked {100*bf:.2f}%"
      f"  -> {'REMOVABLE' if all(r['ok'] for r in rep) else 'NOT REMOVABLE'}")
PYEOF
done
echo "=== Z-SCAN DONE $(date +%H:%M:%S) ==="
printf "%-10s %s\n" "z*=0" "$(grep -hE '^RESULT' $D/free.log | tail -1)"
for Z in ${ZS:-0.75 1.50 2.25}; do
  printf "%-10s %s\n" "z*=$Z" "$(grep -hE '^RESULT' $D/z$Z/free.log 2>/dev/null | tail -1)"
done
