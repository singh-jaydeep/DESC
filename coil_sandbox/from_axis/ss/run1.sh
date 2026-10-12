#!/bin/bash
# single stage on the best QA nfp2 candidate: rows distances, composite secant, basis 12, hinges at z = 0
cd "$(dirname "$0")"
OUT=${OUT:-qa2best/ss1}
mkdir -p $OUT
exec systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python ../../single_stage.py \
  --eq ${EQ:-qa2best/eq0_b12.h5} --start ${CS:-qa2best/coils0.h5} --out $OUT \
  --bounds ../../bounds/fa_qa2_a0167.json --fix-hinge-z 0 --dist rows --rows-select 1.6 --cc-rows 12000 \
  --pc-rows 12000 --pc-gap 0.005 --plasma-m 40 --plasma-n 40 --helicity 1,0 --qs-form ${QSFORM:-boozer} --qs-rho ${QSRHO:-0.25,0.5,0.75,1.0} --A-min ${AMIN:-6.5} --A-max ${AMAX:-7.0} \
  --iota-min ${IOTAMIN:-0.30} --tau-bn ${TAUBN:-2e-3} --tau-qs-rel ${TAUQS:-0.5} --src-gl 64 --force-chunk 200 \
  --method lsq-composite --hessian secant --solve-tol 1e-6 --kmin ${KMIN:-1} --kmax ${KMAX:-6} \
  --maxiter ${MAXIT:-50} "$@" > $OUT/log.txt 2>&1
