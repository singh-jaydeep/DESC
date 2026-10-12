#!/bin/bash
# (a) per-coil level hinges: from the best z = 0 QA nfp2 state, each coil's hinges at one free height with |z| <= H.
# Same second-order weights. lvfree = no level constraint, |z| <= 0.2 (what full hinge freedom buys). CPU, one at a time.
cd "$(dirname "$0")"
src=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10
for H in 0.02 0.05 0.10 0.20; do
  out=runs2/lv_qa2_h$H
  [ -f $out.json ] || systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c \
    "$(python cont.py $src $out --mode free --wlevel 1e4 --hzmax $H --maxiter 600)" > $out.log 2>&1
done
out=runs2/lvfree_qa2_h0.20
[ -f $out.json ] || systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c \
  "$(python cont.py $src $out --mode free --hzmax 0.20 --maxiter 600)" > $out.log 2>&1
python - <<'PY'
import json, glob
keys = ('cost_field','max_dB','max_dG_rel','iota','B20_residual','max_dGG_rel','max_abs_Zaxis','max_abs_hinge_z','max_hinge_dz','max_elong','nit')
for f in ['runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10.json'] + sorted(glob.glob('runs2/lv*_qa2_h*.json')):
    d = json.load(open(f))['final']
    print(f.split('/')[-1][:-5], {k: (round(d[k], 4) if isinstance(d.get(k), float) else d.get(k)) for k in keys}, 'hz', d.get('hinge_z'))
PY
