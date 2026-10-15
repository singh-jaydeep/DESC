#!/bin/bash
# overnight 2026-10-14/15: design B free-boundary at basis 14 (k6 and k5), checks, basis-16 re-solves. Run from ss/.
O=../final_work
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 "$@"; }
runfft() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 "$@"; }
while kill -0 1015660 2>/dev/null; do sleep 30; done   # the dense D trace (one GPU job at a time)
echo "### start $(date +%T)"
fb() {  # tag eq coils shells
  echo "### fb $1 $(date +%T)"
  run python fb_rescore.py $2 $3 $O/res/fb14_$1 --LMN 14 --shells $4 --maxiter 20 --solve-tol 1e-8 --chunk 200 > $O/res/fb14_$1.log 2>&1; echo "rc $? $(date +%T)"
}
post() {  # tag coils
  F=$O/res/fb14_$1/eq_fb.h5; [ -f $F ] || { echo "no $F"; return; }
  echo "### post $1 $(date +%T)"
  run python tip.py $F $2 $O/res/fb14_$1/tip.png --device gpu > $O/res/fb14_$1_tip.log 2>&1; echo "tip rc $?"
  run python iota_trace.py fb14_$1 $2 $O/iota/fb14_$1 $F=fb14 > $O/iota/fb14_$1.log 2>&1; echo "trace rc $?"
  run python interior_check.py $2 $F=fb14 --json $O/iota/interior_fb14_$1.json > $O/iota/interior_fb14_$1.log 2>&1; echo "interior rc $?"
  run python qs_metrics.py $O/res/fb14_$1_qs.json $1=$F > $O/res/fb14_$1_qs.log 2>&1 && run python qs_booz_fix.py $O/res/fb14_$1_qs.json $1=$F >> $O/res/fb14_$1_qs.log 2>&1; echo "qs rc $?"
  runfft python gamma_c.py $O/res/fb14_$1_gamma_c.json $1=$F > $O/res/fb14_$1_gamma_c.log 2>&1; echo "gamma_c rc $?"
}
fb Bk6 li030/repaired/eq.h5 li030/ss/coils_k6.h5 14,14,14
post Bk6 li030/ss/coils_k6.h5
fb Bk5 cloud_runs/fb14_Bk5/fb/eq_fb.h5 li030/ss/coils_k5.h5 8,14,14,14
post Bk5 li030/ss/coils_k5.h5
for t in Bk6 Bk5; do
  F=$O/res/fb14_$t/eq_fb.h5; [ -f $F ] || continue
  echo "### ladder16 $t $(date +%T)"; run python resladder.py $F $O/res/fb14_$t/b16 --lmn 16 > $O/res/fb14_$t/b16.log 2>&1; echo "rc $? $(date +%T)"
done
echo "ALLDONE $(date +%T)"
