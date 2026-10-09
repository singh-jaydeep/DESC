#!/bin/bash
# arcB2 M5, fixed equilibrium, per device (2026-10-08, user: "build out more examples"):
#   prep  : eq (QUASR boundary or a prepared .h5), level-chord lens start (make_z0_start.py --chord-z centre), repack
#   level : run_al --horizontal-hinges --link-mu 1e8 from the lens start
#   free  : run_al --link-mu 1e8 warm-started from the level result (a cold free start fell into a poor basin on 1328095)
# Settings as quasr1328095/run.sh (precise_QH bounds: every device has a = 0.124-0.125 m).
# Usage: run.sh NAME EQSPEC   (EQSPEC = QUASR ID, or an .h5 path)      one job at a time, capped
cd /home/singh/Documents/DESC2/coil_sandbox
NAME=$1; SPEC=$2
D=work/ss_qh/joint/devices/$NAME
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CPU="systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0"
mkdir -p $D/starts
EQ=$D/eq.h5
if [[ "$SPEC" =~ ^[0-9]+$ ]]; then $CPU $PY work/ss_qh/joint/devices/prep_quasr.py $SPEC $EQ > $D/prep.log 2>&1 || exit 1
else cp "$SPEC" $EQ; fi
$CPU $PY work/ss_qh/joint/make_z0_start.py --eq $EQ --vacuum --bounds bounds/precise_qh_gil.json \
    --currents work/ss_qh/joint/base_level/result.h5 --power 0.7 --clear 2.0 --chord-z centre --out $D/starts/level_start.h5 >> $D/prep.log 2>&1 || exit 1
$CPU $PY work/ss_qh/fixed_ss_eq/repack_arcs.py $D/starts/level_start.h5 $D/starts/level_start_packed.h5 $EQ >> $D/prep.log 2>&1 || exit 1
grep -E "^QUASR|^coil|  L |B.n:|->" $D/prep.log
COMMON="--eq $EQ --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8"
run() { name=$1; start=$2; shift 2; mkdir -p $D/$name; echo "=== start $NAME/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $start --out $D/$name "$@" > $D/$name/run.log 2>&1; echo "=== end $NAME/$name exit $? $(date +%T)"; sleep 10; }
run level $D/starts/level_start_packed.h5 --horizontal-hinges
[ -f $D/level/result.h5 ] && run free_warm $D/level/result.h5
