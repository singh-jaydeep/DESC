#!/usr/bin/env bash
# COLD start: polar arcB2 fitted to CIRCLES, never to the planar optimum.
#
# Every 16-coil arcB2 result so far inherits the planar optimum's shape AND its currents
# (make_start copies c.current; the fit and the heavy-anchor repair preserve them exactly).
# So "one coil ends subdominant" could be an accident of that lineage. This start shares
# nothing with it: equal currents, equal circular shapes, hinges from a plain cut angle.
#
# No leg A either -- the point is to let the hinges emerge without ever being frozen.
#   circles (r/a 2.5, 4 unique) -> polar arc B=2 fit -> feasibility repair -> free stage
#
# r/a 2.5 is the value the 16-coil FourierXYZ work used: no circular 16-coil start satisfies
# both clearances (d_cc >= 1.2 needs r/a <~ 2.4, d_pc >= 1.2 needs r/a >~ 2.6), so the
# feasibility solve has to shape it, exactly as it did there.
set -u
cd "$(dirname "$0")/../../.."
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
EQ=sample_equilibria/helios_repro.h5
BND=${BND:-bounds/helios_kruger2026.json}
D=work/helios/arc16/${LABEL:-cold_cut${CUT:-30}}
CUT=${CUT:-30}
RA=${RA:-2.5}
IT=${IT:-300}
CAP=(systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0)
mkdir -p "$D"

if [ ! -f "$D/fit.h5" ]; then
  echo "=== cold fit: circles r/a $RA, cut $CUT ==="
  "$PY" make_start.py --eq $EQ --bounds $BND --nc 4 --r-over-a "$RA" \
    --rep polararc --B 2 --M 5 --cut-angle "$CUT" --out "$D/fit.h5" > "$D/fit.log" 2>&1 \
    || { echo "make_start FAILED"; tail -5 "$D/fit.log"; exit 1; }
  grep -E "fit max deviation|^  L |value / bound|B\.n" "$D/fit.log"
fi

# 10% margin, not the usual 25%: at 16 coils no circular start satisfies both clearances at
# once, and the planar-16 repair never reached a 25% margin in 300 iterations (n_outer stuck
# at 1). A 25% target here means d_cc 1.5 m, which is likely unreachable; 10% means 1.32 m.
MARGIN=${MARGIN:-0.10}
echo "=== feasibility repair (${MARGIN} margin, length skipped) ==="
"${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $BND --start "$D/fit.h5" --out "$D/repair" \
  --mode feasibility --anchor-weight 1.0 --margin "$MARGIN" --margin-skip L \
  --maxiter 300 --check-every 25 --ckpt-every 50 --convex-form signed > "$D/repair.log" 2>&1
grep -E "^(START|RESULT)" "$D/repair.log"
[ -f "$D/repair/result.h5" ] || { echo "repair FAILED"; exit 1; }

echo "=== free stage, no leg A ==="
"${CAP[@]}" "$PY" run_one.py --eq $EQ --bounds $BND --start "$D/repair/result.h5" --out "$D/free" \
  --maxiter "$IT" --check-every 25 --ckpt-every 50 --convex-form signed \
  --diag-log diag.jsonl > "$D/free.log" 2>&1
grep -E "^(START|RESULT|SOLVER)" "$D/free.log"
