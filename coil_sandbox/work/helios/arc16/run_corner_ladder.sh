#!/usr/bin/env bash
# Corner-limit ladder, TIGHT -> LOOSE, on the best 16-coil arcB2 (warm_L36_paper, 7.7549e-3).
#
# There has never been a corner constraint; hinge turn angles were only reported, and the
# grown L=36 optimum reaches 117 deg (vs ~100 deg at 12 coils, and 81 deg in the short-coil
# cold L=28 set -- the length ladder bought field error partly by opening corners).
#
# sandbox/corners.py adds CoilCornerAngle: cos(theta) at EVERY hinge, bounded below by
# cos(theta_max), so the bound is PER HINGE (a max over hinges would kink exactly where the
# binding hinge changes -- the failure mode that stalled the paper-form convexity constraint).
# Validated: values match common.curve_metrics 'corner' to 1e-4 deg, Jacobian matches finite
# differences to 1.9e-6 over hinge/shape/tilt parameters.
#
# Tight -> loose mirrors the length ladder that worked: start restricted, grow into freedom,
# and see whether the chain ends somewhere other than the 7.7549e-3 basin.
# Corners in that basin, sorted: 25.3 63.4 69.6 73.9 80.5 96.0 114.8 117.1 deg
#   -> 60 binds 6 of 8 hinges, 75 binds 4, 90 binds 3, 105 binds 2, 120 binds none.
#
# Serial, one GPU job at a time. RESUMABLE: skips a rung that already has result.h5.
set -u
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
BND=bounds/helios_L36_ladder.json
IT=${IT:-400}
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
RUNGS=${RUNGS:-"60 75 90 105"}
START=work/helios/arc16/warm_L36_paper/result.h5

prev="$START"
for C in $RUNGS; do
  D=work/helios/arc16/corner$C
  mkdir -p "$D"
  if [ -f "$D/result.h5" ]; then echo "[$C deg] done already, skipping"; prev="$D/result.h5"; continue; fi
  echo "=== [corner <= $C deg] from $prev  $(date +%H:%M:%S) ==="
  "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $BND --start "$prev" --out "$D" \
    --corner-max "$C" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
    --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$D/run.log"; then
    echo "[$C] CUDA fault, retrying once"; mv "$D/run.log" "$D/run.log.fail1"
    "${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $BND --start "$prev" --out "$D" \
      --corner-max "$C" --maxiter "$IT" --check-every 25 --ckpt-every 50 \
      --convex-form signed --diag-log diag.jsonl > "$D/run.log" 2>&1
  fi
  grep -E "^(START|RESULT)" "$D/run.log" || { echo "[$C] FAILED"; exit 1; }
  "$PY" - "$D/result.h5" "$C" <<'EOF'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
from corners import corner_angles
cs = load(sys.argv[1]); lim = float(sys.argv[2])
a = np.concatenate([corner_angles(c) for c in S.iter_unique(cs)])
I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
print(f"   corners {np.round(np.sort(a),1)}  max {a.max():.1f} (limit {lim})"
      f"  {'OK' if a.max() <= lim + 0.5 else 'OVER'}   currents min/max {I.min()/I.max():.2f}")
EOF
  prev="$D/result.h5"
done
echo "=== CORNER LADDER DONE $(date +%H:%M:%S) ==="
for d in work/helios/arc16/corner*/; do
  [ -f "$d/run.log" ] || continue
  printf "%-34s %s\n" "${d%/}" "$(grep '^RESULT' "$d/run.log" | tail -1)"
done
