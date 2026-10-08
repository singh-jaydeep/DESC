#!/usr/bin/env bash
# DUAL LADDER: loosen the length cap and the corner limit TOGETHER, tight -> loose.
#
# Why both at once. The plain length ladder (28 -> 32 -> 34 -> 36) reached 7.7549e-3 with
# all four coils healthy, but it bought that partly by OPENING CORNERS: max hinge turn angle
# went 81.3 deg (cold L=28) -> 117.1 deg (grown L=36). Raising M to 10 bought another 11.3%
# and opened them further, to 144.5 deg. So field error and corner sharpness have been trading
# off the whole time, and no run has ever had a corner constraint.
#
# This ladder holds corners down while the coils grow, loosening the limit only as fast as the
# length, to see whether the chain ends somewhere other than the 117-deg basin.
#
#   rung   L (m)   corner (deg)
#     1     28        60
#     2     30        70
#     3     32        80
#     4     34        90
#     5     36       100
#
# Base: the cold L=28 coilset (currents min/max 0.90, corners max 81.3 deg, d_cc 1.412 m and
# d_pc 1.348 m -- both already clear the paper's 1.2 m, so it is feasible at paper clearances
# even though it was produced under relaxed ones). Rung 1 therefore starts feasible on
# everything EXCEPT the corner limit, which binds on 3 of 8 hinges.
#
# Corner constraint: sandbox/corners.py, per hinge (not a max over hinges). Validated against
# common.curve_metrics to 1e-4 deg and against finite differences to 1.9e-6.
#
# Serial, one GPU job at a time. RESUMABLE: skips a rung that already has result.h5.
set -u
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
IT=${IT:-400}
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
# "L:corner" per rung
RUNGS=${RUNGS:-"28:60 30:70 32:80 34:90 36:100"}
prev=${START:-work/helios/arc16/cold_L28_d10/free/result.h5}

for R in $RUNGS; do
  L=${R%%:*}; C=${R##*:}
  case "$L" in 36) BND=bounds/helios_L36_ladder.json;; *) BND=bounds/helios_L$L.json;; esac
  D=work/helios/arc16/dual_L${L}_c${C}
  mkdir -p "$D"
  if [ -f "$D/result.h5" ]; then echo "[L$L c$C] done already, skipping"; prev="$D/result.h5"; continue; fi
  echo "=== [L <= $L m, corner <= $C deg] from $prev  $(date +%H:%M:%S) ==="
  "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds "$BND" --start "$prev" --out "$D" \
    --corner-max "$C" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
    --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$D/run.log"; then
    echo "[L$L c$C] CUDA fault, retrying once"; mv "$D/run.log" "$D/run.log.fail1"
    "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds "$BND" --start "$prev" --out "$D" \
      --corner-max "$C" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
      --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  fi
  grep -E "^RESULT" "$D/run.log" || { echo "[L$L c$C] FAILED"; exit 1; }
  "$PY" - "$D/result.h5" "$C" <<'EOF'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
from corners import corner_angles
cs = load(sys.argv[1]); lim = float(sys.argv[2])
a = np.concatenate([corner_angles(c) for c in S.iter_unique(cs)])
I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
print(f"   corners {np.round(np.sort(a),1)}  max {a.max():.1f} / limit {lim:.0f}"
      f"  {'OK' if a.max() <= lim + 0.5 else 'OVER'}   currents min/max {I.min()/I.max():.2f}")
EOF
  prev="$D/result.h5"
done
echo "=== DUAL LADDER DONE $(date +%H:%M:%S) ==="
for d in work/helios/arc16/dual_L*/; do
  [ -f "$d/run.log" ] || continue
  printf "%-34s %s\n" "${d%/}" "$(grep '^RESULT' "$d/run.log" | tail -1)"
done
