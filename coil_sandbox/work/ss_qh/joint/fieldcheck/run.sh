#!/bin/bash
# Field quality of the best candidates (user, 2026-10-09): Poincare from the coils + vacuum free-boundary -> Boozer QS.
#   level : cont/refine coils on cont/k4b eq (res 8; refined B.n 2.2661e-3, the best level-hinge result)
#   z0    : on_z0_k5 refined coils on ss_z0_k5 eq (6.356e-3, the best z = 0 QH result)
#   free  : reference, free arcB2 (matrix on_arcB2) on its own single-stage eq B_arcB2 (1.484e-3)
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint; M=work/ss_qh/matrix; F=$J/fieldcheck
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
export XLA_FLAGS=--xla_gpu_autotune_level=0 NPER=${NPER:-400}
go() { n=$1; shift; mkdir -p $F/$n; echo "=== start $n $(date +%T)"; $CAP $PY $F/fieldcheck.py $n "$@" $F/$n trace freeb > $F/$n/run.log 2>&1; echo "=== end $n exit $? $(date +%T)"; sleep 5; }
go level $J/cont/refine/result.h5          $J/cont/k4b/eq_k4.h5
go z0    $J/on_z0_k5/arcB2_z0/result.h5    $J/ss_z0_k5/eq_k5.h5
go free  $M/on_arcB2/arcB2/result.h5       $M/ss/arcB2_k4b/eq_k4.h5
