#!/bin/bash
# Length price for arcB2 (fixed boundary, warm starts, everything else as the original refine): per-coil L x1.25.
#   arcL125_free : matrix on_arcB2/arcB2 recipe on B_arcB2, start = its result (B.n 1.484e-3)
#   arcL125_level: cont/refine recipe on cont/k4b (level hinges), start = cont/refine result (2.266e-3)
# then the full pipeline on xyz4_L125 (nc4, +25% length) for a traced comparison with xyz5. Waits for queue4.
cd /home/singh/Documents/DESC2/coil_sandbox
N=work/ss_qh/nc5xyz; F=work/ss_qh/joint/fieldcheck; J=work/ss_qh/joint; M=work/ss_qh/matrix
EQA=$M/ss/arcB2_k4b/eq_k4.h5; EQL=$J/cont/k4b/eq_k4.h5; EQX=$M/ss/xyz_k4b/eq_k4.h5; B=bounds/precise_qh_gil_nc4_L125.json
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
CAPQ="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0"
until grep -q "QUEUE4 DONE" $N/queue4.log; do sleep 30; done
mkdir -p $N/arcL125_free; echo "=== start arcL125_free $(date +%T)"
$CAP $PY run_al.py --eq $EQA --vacuum --bounds $B --maxiter 800 --cc-gap coils --cc-k 80000 \
  --start $M/on_arcB2/arcB2/result.h5 --out $N/arcL125_free > $N/arcL125_free/run.log 2>&1; echo "=== end arcL125_free exit $? $(date +%T)"
sleep 5; $CAPQ $PY $F/spectrum.py arcL125_free $N/arcL125_free/result.h5 $EQA > $F/spectrum/arcL125_free.log 2>&1
mkdir -p $N/arcL125_level; echo "=== start arcL125_level $(date +%T)"
$CAP $PY run_al.py --eq $EQL --vacuum --bounds $B --maxiter 800 --cc-gap coils --cc-k 80000 --horizontal-hinges --link-mu 1e8 \
  --start $J/cont/refine/result.h5 --out $N/arcL125_level > $N/arcL125_level/run.log 2>&1; echo "=== end arcL125_level exit $? $(date +%T)"
sleep 5; $CAPQ $PY $F/spectrum.py arcL125_level $N/arcL125_level/result.h5 $EQL > $F/spectrum/arcL125_level.log 2>&1
export XLA_FLAGS=--xla_gpu_autotune_level=0 NPER=400
c=xyz4L; mkdir -p $F/$c; echo "=== $c trace $(date +%T)"
$CAPQ $PY $F/fieldcheck.py $c $N/xyz4_L125/result.h5 $EQX $F/$c trace > $F/$c/run.log 2>&1
sleep 5; $F/pipeline.sh $c $N/xyz4_L125/result.h5 $EQX
sleep 5; $CAPQ $PY $F/directcheck.py $c $N/xyz4_L125/result.h5 $EQX > $F/$c/directcheck.log 2>&1
sleep 5; $CAPQ $PY $F/ripple.py $c $EQX > $F/$c/ripple.log 2>&1
sleep 5; $CAPQ $PY $F/figures.py $c $EQX > $F/$c/figures.log 2>&1
echo "=== QUEUE5 DONE $(date +%T)"
