#!/bin/bash
# overnight 2026-10-14/15: design B free-boundary at basis 14 (k6 and k5), checks, basis-16 re-solves. Run from ss/.
# try1 (20:53) ran out of GPU memory in the full-mode shell (6.25 GB request under JAX's default 75% pool):
# now MEM_FRACTION 0.93, start from states already at basis 14, and fall back to shell 12 if shell 14 still does not fit.
O=../final_work
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 "$@"; }
runfft() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 "$@"; }
echo "### start $(date +%T)"
fb() {  # tag eq coils shells fallback_shells
  echo "### fb $1 shells $4 $(date +%T)"
  run python fb_rescore.py $2 $3 $O/res/fb14_$1 --LMN 14 --shells $4 --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare > $O/res/fb14_$1.log 2>&1; echo "rc $? $(date +%T)"
  if grep -q "FAILED" $O/res/fb14_$1.log; then
    echo "### fb $1 FAILED in a shell: fallback shells $5 $(date +%T)"
    S=$2; [ -f $O/res/fb14_$1/eq_fb.h5 ] && { cp $O/res/fb14_$1/eq_fb.h5 $O/res/fb14_$1_partial.h5; S=$O/res/fb14_$1_partial.h5; }
    mv $O/res/fb14_$1 $O/res/fb14_$1_try; mv $O/res/fb14_$1.log $O/res/fb14_$1_try.log
    run python fb_rescore.py $S $3 $O/res/fb14_$1 --LMN 14 --shells $5 --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare > $O/res/fb14_$1.log 2>&1; echo "rc $? $(date +%T)"
  fi
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
fb Bk6 $O/res/B_k6rep/eq_b14.h5 li030/ss/coils_k6.h5 14,14,14 12,12,12
post Bk6 li030/ss/coils_k6.h5
fb Bk5 cloud_runs/fb14_Bk5/fb/eq_fb.h5 li030/ss/coils_k5.h5 8,14,14,14 8,12,12,12
post Bk5 li030/ss/coils_k5.h5
for t in Bk6 Bk5; do
  F=$O/res/fb14_$t/eq_fb.h5; [ -f $F ] || continue
  echo "### ladder16 $t $(date +%T)"; run python resladder.py $F $O/res/fb14_$t/b16 --lmn 16 > $O/res/fb14_$t/b16.log 2>&1; echo "rc $? $(date +%T)"
done
echo "ALLDONE $(date +%T)"
