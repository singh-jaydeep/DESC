#!/bin/bash
# traced iota vs equilibria, all designs (run from ss/)
O=../final_work/iota
run() { echo "### $1 $(date +%T)"; systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python iota_trace.py "$@" > $O/$1.log 2>&1; echo "rc $? $(date +%T)"; }
run relaxed_nc5lh ../relaxed/nc5lh/coils.h5 $O/relaxed_nc5lh ../relaxed/nc5lh/eq_fb.h5=shell8 ../relaxed/nc5lh/fb_repaired/eq.h5=shell8_repaired
run relaxed_nc6z0 ../relaxed/nc6z0/coils.h5 $O/relaxed_nc6z0 ../relaxed/nc6z0/eq_fb.h5=shell8 ../relaxed/nc6z0/fb_repaired/eq.h5=shell8_repaired
run B_k6 li030/ss/coils_k6.h5 $O/B_k6 li030/fb_norepair_local/eq_fb.h5=fb li030/repaired/eq.h5=fb_repaired li030/ss/eq_k6.h5=ss_k6
run B_k5 li030/ss/coils_k5.h5 $O/B_k5 li030/ss/eq_k5.h5=ss_k5
run D_k6 nc5lh/ss/coils_k6.h5 $O/D_k6 nc5lh/fb/eq_fb.h5=fb nc5lh/ss/eq_k6.h5=ss_k6
echo ALLDONE
