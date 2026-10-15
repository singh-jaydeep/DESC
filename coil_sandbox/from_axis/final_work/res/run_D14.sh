#!/bin/bash
# checks on D's basis-14 free-boundary state (Modal fb14_D); run from ss/
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 "$@"; }
echo "### trace $(date +%T)"; run python iota_trace.py D_fb14 nc5lh/ss/coils_k6.h5 ../final_work/iota/D_fb14 cloud_runs/fb14_D/fb/eq_fb.h5=fb14 > ../final_work/iota/D_fb14.log 2>&1; echo "rc $?"
echo "### interior $(date +%T)"; run python interior_check.py nc5lh/ss/coils_k6.h5 cloud_runs/fb14_D/fb/eq_fb.h5=fb14 --json ../final_work/iota/interior_D_fb14.json > ../final_work/iota/interior_D_fb14.log 2>&1; echo "rc $?"
echo "### qs $(date +%T)"; run python qs_metrics.py ../final_work/res/D_fb14_qs.json D_fb14=cloud_runs/fb14_D/fb/eq_fb.h5 > ../final_work/res/D_fb14_qs.log 2>&1 && run python qs_booz_fix.py ../final_work/res/D_fb14_qs.json D_fb14=cloud_runs/fb14_D/fb/eq_fb.h5 >> ../final_work/res/D_fb14_qs.log 2>&1; echo "rc $?"
echo "### ladder16 $(date +%T)"; run python resladder.py cloud_runs/fb14_D/fb/eq_fb.h5 ../final_work/res/D_fb14 --lmn 16 > ../final_work/res/D_fb14.log 2>&1; echo "rc $?"
echo "ALLDONE $(date +%T)"
