#!/usr/bin/env bash
# One hinge-placement start for polar arcB2 on helios_repro:
#   heavy-anchor repair (25% margin, length skipped) -> stage 2 with hinges held (50 it)
#   -> check the hinges did not move -> stage 2 free (FREE_IT iterations).
# usage: run_chain.sh CUT_ANGLE        (the fit must exist at work/helios/scan_polar/cut<ANGLE>/fit.h5)
#        FREE_IT=250 run_chain.sh ...     length of the free stage (default 250)
#        run_chain.sh CUT_ANGLE continue N   only continue the free stage from legB for N iterations (-> legC)
set -u
cd "$(dirname "$0")/../../.."
CUT=$1
D=work/helios/scan_polar/cut$CUT
FREE_IT=${FREE_IT:-250}
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
BND=bounds/helios_kruger2026.json
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)

run() {  # run once, retry once on the known intermittent CUDA fault
  local log=$1; shift
  "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$log"; then
    echo "[$CUT] CUDA fault in $log, retrying once"
    mv "$log" "$log.fail1"
    "${CAP[@]}" "$PY" run_one.py "$@" > "$log" 2>&1
  fi
  grep -q "^wrote" "$log" || { echo "[$CUT] FAILED: $log"; exit 1; }
}

margins() {  # frame-degeneracy margins after a stage (1 = degenerate)
  "$PY" - "$1" <<'EOF2'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
m = [float(np.max(np.asarray(c.arc_ref_margin))) for c in S.iter_unique(load(sys.argv[1]))]
print(f"arc_ref margins {np.round(m, 3)}" + ("   WARNING: near degenerate" if max(m) > 0.9 else ""))
EOF2
}

if [ "${2:-}" = "continue" ]; then
  echo "[$CUT] continue free stage for $3 iterations"
  run "$D/legC.log" --eq $EQ --bounds $BND --start "$D/legB/result.h5" --out "$D/legC" \
    --maxiter "$3" --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl
  margins "$D/legC/result.h5"
  echo "[$CUT] done"
  exit 0
fi

echo "[$CUT] repair"
run "$D/repair.log" --eq $EQ --bounds $BND --start "$D/fit.h5" --out "$D/repair" --mode feasibility \
  --anchor-weight 1.0 --margin 0.25 --margin-skip L --maxiter 120 --check-every 10 --ckpt-every 50 --convex-form signed

echo "[$CUT] stage 2, hinges held"
run "$D/legA.log" --eq $EQ --bounds $BND --start "$D/repair/result.h5" --out "$D/legA" --fix-hinges \
  --maxiter 50 --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl

"$PY" - "$D" <<'EOF' || { echo "[$CUT] hinges moved in leg A"; exit 1; }
import sys; sys.path.insert(0, "sandbox"); import os
os.environ["SANDBOX_DEVICE"] = "cpu"
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

echo "[$CUT] stage 2, free"
run "$D/legB.log" --eq $EQ --bounds $BND --start "$D/legA/result.h5" --out "$D/legB" \
  --maxiter "$FREE_IT" --check-every 25 --ckpt-every 25 --convex-form signed --diag-log diag.jsonl
margins "$D/legB/result.h5"
echo "[$CUT] done"
