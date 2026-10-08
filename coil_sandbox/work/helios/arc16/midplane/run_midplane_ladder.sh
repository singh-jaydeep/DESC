#!/usr/bin/env bash
# (B) THE MIDPLANE JOINT PLANE: every hinge held at z = 0, so all 16 coils are cut on one
# horizontal plane and each half lifts clear of the torus.
#
# Stellarator symmetry forces the choice of height: a unique coil's hinges at z* have images
# at -z*, so z* = 0 is the ONLY height that gives one joint plane for the whole machine.
#
# This mirrors run_cold.sh + run_dual_ladder.sh EXACTLY -- same circles (r/a 2.5), same M=5,
# same bounds files, same iteration budgets, same 0.10-margin repair -- so the only variable
# against dual_L36_c100 (bn 7.7913e-3) is --fix-hinge-z 0. The start differs only in WHERE the
# hinges are: at each circle's two z=0 crossings instead of a 30-degree cut angle. Circles are
# planar, so that split is exact (fit deviation 3.5 mm, same as the baseline's).
#
# --demount-margin 0.3 requires every hinge to sit 0.3 m outside the plasma's vertical shadow
# (sandbox/demount.py). It is NOT expected to bind -- the midplane starts sit 2.4-5.0 m clear --
# it is a guard, and every check line carries the exact ray-cast verdict. --demount-arcs is left
# OFF to save memory; if a check ever reports a blocked arc, turn it on and restart that rung.
#
# Serial, one GPU job at a time. RESUMABLE: skips any stage that already has result.h5.
set -u
cd "$(dirname "$0")/../../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
D=work/helios/arc16/midplane
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
# --demount-margin is deliberately NOT passed to the rungs. It never binds at the midplane
# (0.91 m of clearance against a 0.32 m enforced bound), but each distinct ACTIVE BLOCK SET
# gets its own second-order Hessian compile, and adding one more constrant roughly doubles the
# number of distinct sets; the accumulated compilations walked RSS to 12.19 GB and the cgroup
# cap killed L30/c70 twice. Removability is verified offline after every rung instead, by exact
# ray casting, which is what the verdict was always based on.
HINGE=(--fix-hinge-z 0)
IT=${IT:-400}
# The 12 GB cap killed L30/c70 at RSS 12.17 GB while compiling the second-order constraint
# Hessian, once the active block set grew to CoilLength + CoilCurvature + CoilSignedCurvature
# + CoilSetDistancePenalty. --so-chunk is the documented lever for exactly that step.
SO=${SO:-8}

run () {  # run <outdir> <logname> <args...>
  local out=$1 log=$2; shift 2
  if [ -f "$out/result.h5" ]; then echo "[$(basename "$out")] done already, skipping"; return 0; fi
  mkdir -p "$out"
  echo "=== $(basename "$out")  $(date +%H:%M:%S) ==="
  "${CAP[@]}" "$PY" run_one.py --eq $EQ "$@" --out "$out" > "$D/$log" 2>&1
  if grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$D/$log"; then
    echo "  CUDA fault, retrying once"; mv "$D/$log" "$D/$log.fail1"
    "${CAP[@]}" "$PY" run_one.py --eq $EQ "$@" --out "$out" > "$D/$log" 2>&1
  fi
  grep -E "^(START|RESULT)" "$D/$log"
  if [ ! -f "$out/result.h5" ]; then
    echo "  FAILED: no result.h5 -- see $D/$log"
    grep -E "Killed|MemoryError|CUDA_ERROR|Traceback" "$D/$log" | tail -3
    exit 1
  fi
}

COLD=bounds/helios_cold_L28_d10.json
run "$D/repair" repair.log --bounds $COLD --start "$D/cold_M5.h5" "${HINGE[@]}" \
    --mode feasibility --anchor-weight 1.0 --margin 0.10 --margin-skip L \
    --maxiter 300 --check-every 25 --ckpt-every 50 --convex-form signed
run "$D/free" free.log --bounds $COLD --start "$D/repair/result.h5" "${HINGE[@]}" \
    --maxiter 300 --check-every 25 --ckpt-every 50 --convex-form signed --diag-log diag.jsonl

prev="$D/free/result.h5"
# assign first, then loop over the UNQUOTED variable: ${RUNGS:-"a b c"} keeps the inner
# quotes, so the default expands as ONE word and the loop silently runs a single bogus rung
# (L=${R%%:*}=28, C=${R##*:}=100). That is what run_dual_ladder.sh does, for this reason.
RUNGS=${RUNGS:-"28:60 30:70 32:80 34:90 36:100"}
for R in $RUNGS; do
  L=${R%%:*}; C=${R##*:}
  case "$L" in 36) BND=bounds/helios_L36_ladder.json;; *) BND=bounds/helios_L$L.json;; esac
  out="$D/dual_L${L}_c${C}"
  run "$out" "dual_L${L}_c${C}.log" --bounds "$BND" --start "$prev" "${HINGE[@]}" \
      --corner-max "$C" --maxiter "$IT" --check-every 25 --ckpt-every 50 --so-chunk "$SO" \
      --convex-form signed --diag-log diag.jsonl
  "$PY" - "$out/result.h5" "$C" <<'PYEOF'
import sys, os; sys.path.insert(0, "sandbox"); os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
from corners import corner_angles
import demount as D
cs = load(sys.argv[1]); lim = float(sys.argv[2])
eq = load("sample_equilibria/helios_repro.h5")
rep = D.demount_report(cs, eq)
cl = min(min(r["hinge_clearance"]) for r in rep)
bf = max(x["blocked_frac"] for r in rep for x in r["arcs"])
a = np.concatenate([corner_angles(c) for c in S.iter_unique(cs)])
I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
z = np.concatenate([np.asarray(c.hinges).reshape(-1, 3)[:, 2] for c in S.iter_unique(cs)])
print(f"   corners max {a.max():.1f} / limit {lim:.0f}  {'OK' if a.max() <= lim + 0.5 else 'OVER'}"
      f"   currents min/max {I.min()/I.max():.2f}   max |hinge z| {np.abs(z).max():.2e} m\n"
      f"   demount: min hinge clearance {cl:+.3f} m, worst arc blocked {100*bf:.2f}%"
      f"  -> {'REMOVABLE' if all(r['ok'] for r in rep) else 'NOT REMOVABLE'}")
PYEOF
  prev="$out/result.h5"
done
echo "=== MIDPLANE LADDER DONE $(date +%H:%M:%S) ==="
for d in "$D"/dual_L*/; do
  [ -f "${d}result.json" ] || continue
  printf "%-46s %s\n" "${d%/}" "$(grep -hE '^RESULT' "$D/$(basename "${d%/}")".log | tail -1)"
done
