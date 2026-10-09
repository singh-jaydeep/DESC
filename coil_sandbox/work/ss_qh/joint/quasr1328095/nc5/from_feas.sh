#!/bin/bash
# (a) follow-up: the B.n solve at QUASR 1328095's own design, level hinges, from the FEASIBLE point found by feas_level
# (constraint-only), with the doubled coil-coil grid (--row-n 512). Did nc5/level stall because of its infeasible start?
cd /home/singh/Documents/DESC2/coil_sandbox
Q=work/ss_qh/joint/quasr1328095
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
mkdir -p $Q/nc5/level_from_feas; echo "=== start nc5/level_from_feas $(date +%T)"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $PY run_al.py --eq $Q/eq_q1328095.h5 --vacuum --bounds bounds/quasr1328095_own.json \
  --maxiter 800 --cc-gap coils --cc-k 80000 --link-mu 1e8 --row-n 512 --horizontal-hinges \
  --start $Q/nc5/feas_level/result.h5 --out $Q/nc5/level_from_feas > $Q/nc5/level_from_feas/run.log 2>&1
echo "=== end nc5/level_from_feas exit $? $(date +%T)"; echo "FROMFEAS DONE"
