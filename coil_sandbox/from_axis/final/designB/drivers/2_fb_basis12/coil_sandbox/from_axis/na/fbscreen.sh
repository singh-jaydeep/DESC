#!/bin/bash
# Free-boundary screen (rule 2) of near-axis candidates: basis 12, a = 0.12, shells 2,4,6,8 (the all-modes shell runs out
# of GPU memory), then the Jacobian-fold check. GPU, one at a time. The winner is to be confirmed at basis 14.
cd "$(dirname "$0")"
for n in "$@"; do
  out=runs2/${n}_fb_a0.12_L12
  [ -f $out/score.json ] || systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
    python score_fb.py runs2/$n --LMN 12 --a 0.12 --shells 2,4,6,8 --maxiter 60 --out $out > $out.log 2>&1
  (cd ../ss && python jac.py ../na/$out/eq_fb.h5 2>&1 | grep -E "rho [0-9.]+:" | sed "s#^#FOLD #") >> $out.log
  echo "SCREENED $n" >> runs2/fbscreen.log
done
echo ALLDONE >> runs2/fbscreen.log
