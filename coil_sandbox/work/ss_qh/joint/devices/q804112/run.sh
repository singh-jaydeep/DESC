#!/bin/bash
# z = 0 test on the screen's best QA candidate with nonzero iota: QUASR 804112 (QA nfp 3, A 10, iota 0.2), boundary
# scaled to a = 0.1243 (exact, vacuum), precise_QH bounds, 4 coils/hp, arcB2 M5, fixed eq.
#   z0: --fix-hinge-z 0 from a z = 0 lens start | level: same start | free_warm: from level.
cd /home/singh/Documents/DESC2/coil_sandbox
D=work/ss_qh/joint/devices/q804112
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CPU="systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0"
mkdir -p $D/starts
$CPU $PY work/ss_qh/joint/make_z0_start.py --eq $D/eq.h5 --vacuum --bounds bounds/precise_qh_gil.json \
  --currents work/ss_qh/joint/base_level/result.h5 --power 0.7 --clear 2.0 --chord-z zero --out $D/starts/z0_start.h5 > $D/prep_start.log 2>&1 || exit 1
$CPU $PY work/ss_qh/fixed_ss_eq/repack_arcs.py $D/starts/z0_start.h5 $D/starts/z0_start_packed.h5 $D/eq.h5 >> $D/prep_start.log 2>&1 || exit 1
grep -E "^coil|^  L|B.n:" $D/prep_start.log
COMMON="--eq $D/eq.h5 --vacuum --bounds bounds/precise_qh_gil.json --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8"
run() { name=$1; start=$2; shift 2; mkdir -p $D/$name; echo "=== start q804112/$name $(date +%T)"; $CAP $PY run_al.py $COMMON --start $start --out $D/$name "$@" > $D/$name/run.log 2>&1; echo "=== end q804112/$name exit $? $(date +%T)"; sleep 10; }
run z0 $D/starts/z0_start_packed.h5 --fix-hinge-z 0
run level $D/starts/z0_start_packed.h5 --horizontal-hinges
[ -f $D/level/result.h5 ] && run free_warm $D/level/result.h5
echo "Q804112 DONE"
