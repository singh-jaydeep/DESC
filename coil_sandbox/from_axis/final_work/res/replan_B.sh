#!/bin/bash
# 23:30 replan: the laptop runs a basis-14 full-mode shell at ~3 h per 20 iterations, so give k6 and k5 ONE full-mode
# shell each by morning (equal effort for the k5/k6 decision) instead of three shells of k6 only. Run from ss/.
O=../final_work
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 "$@"; }
runfft() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 "$@"; }
log() { echo "$@" >> $O/res/overnight_B.out; }
until [ -f $O/res/fb14_Bk6/shells_done.json ]; do sleep 30; done   # k6 shell 1 saved
sleep 20
for p in $(pgrep -f "overnight_B.sh"); do kill $p; done
for p in $(pgrep -f "python fb_rescore.py"); do kill $p; done
sleep 10
log "### replan $(date +%T): k6 stopped after one full-mode shell; k5 gets shells 8,14"
post() {  # tag coils
  F=$O/res/fb14_$1/eq_fb.h5; [ -f $F ] || { log "no $F"; return; }
  log "### post $1 $(date +%T)"
  run python iota_trace.py fb14_$1 $2 $O/iota/fb14_$1 $F=fb14 > $O/iota/fb14_$1.log 2>&1; log "trace rc $?"
  run python interior_check.py $2 $F=fb14 --json $O/iota/interior_fb14_$1.json > $O/iota/interior_fb14_$1.log 2>&1; log "interior rc $?"
  run python qs_metrics.py $O/res/fb14_$1_qs.json $1=$F > $O/res/fb14_$1_qs.log 2>&1 && run python qs_booz_fix.py $O/res/fb14_$1_qs.json $1=$F >> $O/res/fb14_$1_qs.log 2>&1; log "qs rc $?"
  runfft python gamma_c.py $O/res/fb14_$1_gamma_c.json $1=$F > $O/res/fb14_$1_gamma_c.log 2>&1; log "gamma_c rc $?"
  run python tip.py $F $2 $O/res/fb14_$1/tip.png --device gpu > $O/res/fb14_$1_tip.log 2>&1; log "tip rc $?"
}
post Bk6 li030/ss/coils_k6.h5
log "### fb Bk5 shells 8,14 $(date +%T)"
run python fb_rescore.py cloud_runs/fb14_Bk5/fb/eq_fb.h5 li030/ss/coils_k5.h5 $O/res/fb14_Bk5 --LMN 14 --shells 8,14 --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare > $O/res/fb14_Bk5.log 2>&1; log "rc $? $(date +%T)"
post Bk5 li030/ss/coils_k5.h5
log "ALLDONE $(date +%T)"
