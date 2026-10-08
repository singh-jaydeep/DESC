#!/bin/bash
# Usage dry: DRY=1 step2_ss.sh planar
# Step 2: single stage per representation, identical settings (handoff 3.5). Usage: step2_ss.sh planar|arcB2|xyz [extra flags]
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
QH="--eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0"
QHSS="--helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19"
SS="--kmin 1 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 --tau-qs-rel 10"
rep=$1; shift
case $rep in
  planar) START=$M/base/planarN7/result.h5; REPF="--src-n 100";;
  arcB2)  START=$M/base/arcB2/result.h5;    REPF="";;
  xyz)    START=$M/base/xyzN6/result.h5;    REPF="--src-n 100 --bs-chunk 25 --jac-chunk 5";;
esac
OUT=$M/ss/$rep${DRY:+_dry}; mkdir -p $OUT   # DRY=1 step2_ss.sh rep -> --dry into ss/rep_dry
echo "=== start ss $rep $(date +%T) $*"
$CAP $PY single_stage.py $QH $QHSS $SS --start $START --out $OUT $REPF ${DRY:+--dry} "$@" > $OUT/run.log 2>&1
echo "=== end ss $rep exit $? $(date +%T)"
