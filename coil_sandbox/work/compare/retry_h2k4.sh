#!/bin/bash
# H2's k=4 step stopped on xtol=1e-6 after 2 iterations (optimality 2.5e-2, not converged): rerun k=4 from H2's
# k=3 result with xtol 1e-12 and H2's anchors. Runs after the Q1/Q2 retry (one job at a time).
cd /home/singh/Documents/DESC2/coil_sandbox
until grep -q "^CHAIN retry done" work/compare/retry.log 2>/dev/null; do sleep 30; done
H=work/ss_qh/arcB2; mkdir -p $H/ss_k4
echo "CHAIN H2-k4 start $(date +%T)"
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 /home/singh/miniforge3/envs/desc-env2/bin/python single_stage.py \
  --eq precise_QH --bounds bounds/precise_qh_gil.json --length-mult 1.0 --helicity 1,4 --A-min 7.5 --A-max 8.5 --iota-min 1.19 \
  --eq $H/ss/eq_k3.h5 --start $H/ss/coils_k3.h5 --out $H/ss_k4 --tau-qs 0.01695 --V0 0.30352 --bn-norm 0.42526 \
  --kmin 4 --kmax 4 --maxiter 100 --solve-tol 1e-10 --xtol 1e-12 > $H/ss_k4/run.log 2>&1
[ -f $H/ss_k4/coils_k4.h5 ] && echo "CHAIN H2-k4 ok $(date +%T)" || echo "CHAIN H2-k4 FAILED $(date +%T)"
echo "CHAIN all done $(date +%T)"
