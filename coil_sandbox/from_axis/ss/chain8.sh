#!/bin/bash
# after the basis-14 rescore of ss7_b14 k=4 (PID $1): k = 5, 6 at basis 14, then the free-boundary rescore of the result
cd "$(dirname "$0")"
while kill -0 $1 2>/dev/null; do sleep 30; done
OUT=qa2best/ss7b_b14 EQ=qa2best/ss7_b14/eq_k4.h5 CS=qa2best/ss7_b14/coils_k4.h5 KMIN=5 KMAX=6 MAXIT=60 TAUBN=5e-4 \
  ./run1.sh --tau-qs 1.5e-3 --V0 0.24924885650248618 --bn-norm 0.36262355746578623 --length-mult 0.999 --solve-tol 1e-8 --track-steps
k=$(ls qa2best/ss7b_b14/eq_k*.h5 2>/dev/null | sed 's/.*eq_k\([0-9]*\).h5/\1/' | sort -n | tail -1)
[ -z "$k" ] && { echo "CHAIN8: single stage produced no state"; tail -5 qa2best/ss7b_b14/log.txt; exit 1; }
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python fb_rescore.py qa2best/ss7b_b14/eq_k$k.h5 \
  qa2best/ss7b_b14/coils_k$k.h5 qa2best/ss7b_b14_fb --shells 4,8,14 > qa2best/ss7b_b14_fb.log 2>&1
echo "CHAIN8 DONE k=$k"
