#!/usr/bin/env bash
# B2: horizontal joint chords with the height left FREE, one plane per coil.
#
# Splits the restriction the midplane ladder measured into its two halves:
#   free hinges -> B2   is the price of HORIZONTALITY (chord perpendicular to z)
#   B2 -> z* = 0        is the price of a SHARED PLANE
# The z* scan already showed the shared plane's HEIGHT is worth only 3.2% over its useful
# range, so this decides whether the 2.23x is horizontality or the sharing.
#
# Starts from cold_M5.h5, exactly where the midplane ladder started: circles cut at z = 0, so
# the start already satisfies z_a = z_b and the linear constraint's projection moves nothing.
# The only difference from the z*=0 run is that the common height is now free to move.
#
# --demount-margin IS carried: with the height free the chords can drift off the midplane,
# and the z* scan showed removability stops being free as soon as they do.
set -u
cd "$(dirname "$0")/../../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
D=work/helios/arc16/midplane/level
COLD=bounds/helios_cold_L28_d10.json
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
mkdir -p "$D"

for STAGE in repair free; do
  [ -f "$D/$STAGE/result.h5" ] && { echo "[$STAGE] done already"; continue; }
  if [ "$STAGE" = repair ]; then
    ARGS=(--start work/helios/arc16/midplane/cold_M5.h5 --mode feasibility --anchor-weight 1.0
          --margin 0.10 --margin-skip L --maxiter 300)
  else
    ARGS=(--start "$D/repair/result.h5" --maxiter 300 --diag-log diag.jsonl)
  fi
  echo "=== level $STAGE  $(date +%H:%M:%S) ==="
  "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $COLD --out "$D/$STAGE" \
    --level-hinges --demount-margin 0.3 "${ARGS[@]}" \
    --check-every 25 --ckpt-every 50 --so-chunk 8 --convex-form signed > "$D/$STAGE.log" 2>&1
  grep -E "^(START|RESULT)|hinge level" "$D/$STAGE.log"
  [ -f "$D/$STAGE/result.h5" ] || {
    echo "  FAILED: no result.h5"; grep -E "Killed|Error|Traceback" "$D/$STAGE.log" | tail -3; exit 1; }
done

"$PY" - "$D/free/result.h5" <<'PYEOF'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S, demount as D
from desc.io import load
from corners import corner_angles
cs = load(sys.argv[1]); eq = load("sample_equilibria/helios_repro.h5")
I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
a = np.concatenate([corner_angles(c) for c in S.iter_unique(cs)])
rep = D.demount_report(cs, eq)
print(f"   corners max {a.max():.1f}   currents min/max {I.min()/I.max():.2f}")
for i, c in enumerate(S.iter_unique(cs)):
    h = np.asarray(c.hinges).reshape(-1, 3)
    print(f"   coil {i}: hinge z {np.round(h[:,2],4)}  (level to {abs(h[0,2]-h[1,2]):.2e} m)"
          f"  clearance {np.round(rep[i]['hinge_clearance'],3)}")
print(f"   -> {'REMOVABLE' if all(r['ok'] for r in rep) else 'NOT REMOVABLE'}")
PYEOF
echo "=== LEVEL DONE $(date +%H:%M:%S) ==="
