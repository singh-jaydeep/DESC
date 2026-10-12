#!/bin/bash
# finish the basis-14 run: k = 5, 6 from ss7_b14 k=4, then the free-boundary rescore of the last state
cd "$(dirname "$0")"
OUT=qa2best/ss7b_b14 EQ=qa2best/ss7_b14/eq_k4.h5 CS=qa2best/ss7_b14/coils_k4.h5 KMIN=5 KMAX=6 MAXIT=60 TAUBN=5e-4 \
  ./run1.sh --tau-qs 1.5e-3 --V0 0.24924885650248618 --bn-norm 0.36262355746578623 --length-mult 0.999 --solve-tol 1e-8
d=ss7b_b14
k=$(ls qa2best/$d/eq_k*.h5 2>/dev/null | sed 's/.*eq_k\([0-9]*\).h5/\1/' | sort -n | tail -1)
if [ -z "$k" ]; then d=ss7_b14; k=4; fi
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python fb_rescore.py qa2best/$d/eq_k$k.h5 \
  qa2best/$d/coils_k$k.h5 qa2best/${d}_fb --shells 4,8,14 > qa2best/${d}_fb.log 2>&1
echo "CHAIN DONE: rescored $d k=$k"
grep -E "^k=[0-9]+ (physics|coils|solver)" qa2best/ss7b_b14/log.txt | cut -c1-330
grep -E "^START|^FREE BOUNDARY|^COILS|shell k=|Traceback" qa2best/${d}_fb.log | cut -c1-700
