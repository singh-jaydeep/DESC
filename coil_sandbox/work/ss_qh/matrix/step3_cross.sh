#!/bin/bash
# Step 3: every representation refined on a single-stage boundary with run_al.py (handoff 3.6), one at a time, capped.
# Usage: step3_cross.sh planar|arcB2|xyz [maxiter, default 800; 0 = build check of the first run only]
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
X=$1; MI=${2:-800}
EQ=$M/ss/${X}_k4b/eq_k4.h5
COMMONX="--eq $EQ --vacuum --bounds bounds/precise_qh_gil.json --maxiter $MI"
start() { [ "$1" = "$X" ] && echo $M/ss/${X}_k4b/coils_k4.h5 || echo $M/base/$2/result.h5; }   # diagonal: own ss coils
run() { name=$1; shift; out=$M/on_$X/$name${SUF}; mkdir -p $out; echo "=== start on_$X/$name$SUF $(date +%T)"; $CAP $PY run_al.py $COMMONX --out $out "$@" > $out/run.log 2>&1; echo "=== end on_$X/$name$SUF exit $? $(date +%T)"; sleep 10; }
[ "$MI" = 0 ] && SUF=_build
run planarN7 --start $(start planar planarN7)
[ "$MI" = 0 ] && exit
run arcB2    --start $(start arcB2 arcB2) --cc-gap coils --cc-k 80000
run xyzN6    --start $(start xyz xyzN6) --cc-k 20000
echo "STEP3 $X DONE"
