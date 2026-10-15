#!/bin/bash
# basis ladder for B (k6 repaired free-boundary state, k5 single-stage state); run from ss/
O=../final_work/res
for job in "B_k6rep li030/repaired/eq.h5" "B_k5 li030/ss/eq_k5.h5"; do
  set -- $job
  echo "### $1 $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python resladder.py $2 $O/$1 --lmn 12,14,16 > $O/$1.log 2>&1
  echo "rc $? $(date +%T)"
done
echo ALLDONE
