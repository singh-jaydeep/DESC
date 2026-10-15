#!/bin/bash
# 2026-10-15 ~03:30: B k6 coil-length polish (stage 2, eq fixed at its basis-14 free-boundary state, hinges on z = 0,
# hard length bound via lsq-auglag-composite), then a basis-14 free-boundary shell with the polished coils, then checks.
# Run from coil_sandbox/.
R=from_axis/final_work/res; I=from_axis/final_work/iota
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 "$@"; }
runfft() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 "$@"; }
echo "### polish $(date +%T)"
run python run_al.py --eq $R/fb14_Bk6/eq_fb.h5 --start $R/polish_Bk6/run/result.h5 --bounds $R/polish_Bk6/bounds_backoff.json \
  --out $R/polish_Bk6/run2 --vacuum --fix-hinge-z 0 --maxiter 200 > $R/polish_Bk6/run2.log 2>&1; echo "rc $? $(date +%T)"
C=$R/polish_Bk6/run2/result.h5; [ -f $C ] || { echo "no $C"; exit 1; }
cd from_axis/ss
echo "### fb $(date +%T)"
run python fb_rescore.py ../final_work/res/fb14_Bk6/eq_fb.h5 ../../$C ../final_work/res/fb14_Bk6p --LMN 14 --shells 14 --maxiter 20 --solve-tol 1e-8 --chunk 200 --no-poincare > ../final_work/res/fb14_Bk6p.log 2>&1; echo "rc $? $(date +%T)"
F=../final_work/res/fb14_Bk6p/eq_fb.h5; CC=../../$C; O=../final_work
echo "### post $(date +%T)"
run python iota_trace.py fb14_Bk6p $CC $O/iota/fb14_Bk6p $F=fb14 > $O/iota/fb14_Bk6p.log 2>&1; echo "trace rc $?"
run python interior_check.py $CC $F=fb14 --json $O/iota/interior_fb14_Bk6p.json > $O/iota/interior_fb14_Bk6p.log 2>&1; echo "interior rc $?"
run python qs_metrics.py $O/res/fb14_Bk6p_qs.json Bk6p=$F > $O/res/fb14_Bk6p_qs.log 2>&1 && run python qs_booz_fix.py $O/res/fb14_Bk6p_qs.json Bk6p=$F >> $O/res/fb14_Bk6p_qs.log 2>&1; echo "qs rc $?"
runfft python gamma_c.py $O/res/fb14_Bk6p_gamma_c.json Bk6p=$F > $O/res/fb14_Bk6p_gamma_c.log 2>&1; echo "gamma_c rc $?"
echo "POLISHDONE $(date +%T)"
