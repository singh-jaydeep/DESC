#!/bin/bash
# after replan_B.sh: dense resonance traces for B k6 and k5 (as D's 1f). Run from ss/.
O=../final_work
until grep -q "^ALLDONE" $O/res/overnight_B.out; do sleep 60; done
for t in Bk6:li030/ss/coils_k6.h5 Bk5:li030/ss/coils_k5.h5; do
  tag=${t%%:*}; c=${t#*:}; F=$O/res/fb14_$tag/eq_fb.h5; [ -f $F ] || continue
  echo "### dense $tag $(date +%T)" >> $O/res/overnight_B.out
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=NLINES=80 --setenv=NPER=2000 python iota_trace.py fb14_${tag}_dense $c $O/iota/fb14_${tag}_dense $F=fb14 > $O/iota/fb14_${tag}_dense.log 2>&1
  echo "dense rc $? $(date +%T)" >> $O/res/overnight_B.out
done
echo "DENSEDONE $(date +%T)" >> $O/res/overnight_B.out
