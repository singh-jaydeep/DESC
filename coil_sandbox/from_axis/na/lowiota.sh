#!/bin/bash
# (b) screen: if the plasma supplied part of iota, how much easier is the coils' job? Same z = 0 state and weights,
# lower VACUUM iota target. CPU, one at a time.
cd "$(dirname "$0")"
src=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10
for I in 0.30 0.20 0.10; do
  out=runs2/li_qa2_iota$I
  [ -f $out.json ] || systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c \
    "$(python cont.py $src $out --iota $I --maxiter 600)" > $out.log 2>&1
  src=$out
done
python - <<'PY'
import json, glob
keys = ('cost_field','max_dB','max_dG_rel','max_dGG_rel','iota','B20_residual','max_abs_Zaxis','max_elong','nit')
for f in ['runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10.json'] + sorted(glob.glob('runs2/li_qa2_iota*.json'), reverse=True):
    d = json.load(open(f))['final']
    print(f.split('/')[-1][:-5], [(round(d[k], 4) if isinstance(d.get(k), float) else d.get(k)) for k in keys], d['status'][:30])
PY
