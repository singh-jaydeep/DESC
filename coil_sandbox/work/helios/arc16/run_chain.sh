#!/usr/bin/env bash
# Polar arcB2 on the 16-coil helios configuration, from the planar N=7 16-coil optimum.
# Same chain as work/helios/scan_polar/run_chain.sh (12 coils), retargeted:
#   fit at a cut angle -> heavy-anchor repair (25% margin, length skipped)
#   -> stage 2 with hinges held (50 it) -> assert the hinges did not move
#   -> stage 2 free (FREE_IT iterations).
# usage: run_chain.sh CUT_ANGLE                  (one value, or "a,b,c,d" per unique coil)
#        FREE_IT=250 run_chain.sh 30
#        run_chain.sh 30 continue 150            (continue legB -> legC)
set -u
cd "$(dirname "$0")/../../.."
CUT=$1
LAB=${CUT//,/_}
D=work/helios/arc16/cut$LAB
FREE_IT=${FREE_IT:-250}
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
BND=bounds/helios_kruger2026.json
SRC=work/helios/planar16/stage2/result.h5
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
mkdir -p "$D"

run() {  # run once, retry once on the known intermittent CUDA fault
  local log=$1; shift
  "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$log"; then
    echo "[$LAB] CUDA fault in $log, retrying once"
    mv "$log" "$log.fail1"
    "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  fi
  grep -q "^wrote" "$log" || { echo "[$LAB] FAILED: $log"; exit 1; }
}

if [ "${2:-}" = "continue" ]; then
  echo "[$LAB] continue free stage for $3 iterations"
  run "$D/legC.log" --eq $EQ --bounds $BND --start "$D/legB/result.h5" --out "$D/legC" \
    --maxiter "$3" --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl
  echo "[$LAB] done"; exit 0
fi

if [ ! -f "$D/fit.h5" ]; then
  echo "[$LAB] fit"
  "$PY" make_start.py --eq $EQ --bounds $BND --from-coils "$SRC" --rep polararc --B 2 --M 5 \
    --cut-angle "$CUT" --out "$D/fit.h5" > "$D/fit.log" 2>&1 || { echo "[$LAB] make_start FAILED"; exit 1; }
  grep -E "fit max deviation|value / bound|^  L " "$D/fit.log"
  # Fit gate. The discriminator is DEVIATION, not convexity: all 17 twelve-coil fits that repaired
  # fine had deviation 136-261 mm and convexity excess 1027-1753x bound, while the spline fit that
  # could not be repaired was 630 mm at only 284x. (The HANDOFF's "convexity < 20x" gate is wrong --
  # no successful fit has ever been near it.) 0.35 m leaves headroom over the worst success.
  "$PY" - "$D/fit.log" <<'EOF' || { echo "[$LAB] FIT GATE FAILED -- not spending GPU on this start"; exit 2; }
import re, sys
t = open(sys.argv[1]).read()
dev = [float(x) for x in re.findall(r"([\d.]+) mm", t.split("fit max deviation")[1].split("\n")[0])]
cx = float(re.search(r"convex ([\d.e+-]+)", t.split("value / bound")[1]).group(1))
print(f"  GATE dev max {max(dev)/1e3:.3f} m (gate <0.35); convex {cx:.0f}x bound (reported, not gated)")
sys.exit(0 if max(dev) < 350 else 1)
EOF
fi

echo "[$LAB] repair"
run "$D/repair.log" --eq $EQ --bounds $BND --start "$D/fit.h5" --out "$D/repair" --mode feasibility \
  --anchor-weight 1.0 --margin 0.25 --margin-skip L --maxiter 120 --check-every 10 --ckpt-every 50 --convex-form signed

echo "[$LAB] stage 2, hinges held"
run "$D/legA.log" --eq $EQ --bounds $BND --start "$D/repair/result.h5" --out "$D/legA" --fix-hinges \
  --maxiter 50 --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl

"$PY" - "$D" <<'EOF' || { echo "[$LAB] hinges moved in leg A"; exit 1; }
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
d = sys.argv[1]
a = [np.asarray(c.hinges) for c in S.iter_unique(load(f"{d}/repair/result.h5"))]
b = [np.asarray(c.hinges) for c in S.iter_unique(load(f"{d}/legA/result.h5"))]
dm = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
print(f"hinge change over leg A: {dm:.2e} m")
sys.exit(0 if dm < 1e-6 else 1)
EOF

echo "[$LAB] stage 2, free"
run "$D/legB.log" --eq $EQ --bounds $BND --start "$D/legA/result.h5" --out "$D/legB" \
  --maxiter "$FREE_IT" --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl

"$PY" - "$D/legB/result.h5" <<'EOF2'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
m = [float(np.max(np.asarray(c.arc_ref_margin))) for c in S.iter_unique(load(sys.argv[1]))]
print(f"arc_ref margins {np.round(m, 3)}" + ("   WARNING: near degenerate" if max(m) > 0.9 else ""))
EOF2
echo "[$LAB] done"
