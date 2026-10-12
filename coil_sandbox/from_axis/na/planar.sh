#!/bin/bash
# Is the fold doing anything? Same QA nfp2 problem, weights and bounds, with each coil forced PLANAR
# (alpha_u + alpha_l = 0), reached by continuation in the planarity weight from the folded optimum.
#   pl_z0   : planar, hinges on z = 0 (the coil plane contains a horizontal chord in the midplane)
#   pl_free : planar, hinges free with |z| <= 0.2 (a general planar coil; compare lvfree_qa2_h0.20)
# usage: planar.sh [PID to wait for]     CPU, one at a time
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 $1 2>/dev/null; do sleep 20; done
run() { systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c "$1" > $2.log 2>&1; }
prev=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10
for w in 1 1e2 1e4; do
  out=runs2/pl_z0_qa2_w$w
  [ -f $out.json ] || run "$(python cont.py $prev $out --wplanar $w --maxiter 600)" $out
  prev=$out
done
prev=runs2/lvfree_qa2_h0.20
for w in 1 1e2 1e4; do
  out=runs2/pl_free_qa2_w$w
  [ -f $out.json ] || run "$(python cont.py $prev $out --wplanar $w --maxiter 600)" $out
  prev=$out
done
python - <<'PY'
import json
keys = ('cost_field','max_dB','max_dG_rel','max_dGG_rel','iota','B20_residual','max_abs_Zaxis','max_abs_hinge_z','max_elong','nit')
for f in ['o2w1_qa2s2_gg1e-1_w1e-2_w1x10','pl_z0_qa2_w1','pl_z0_qa2_w1e2','pl_z0_qa2_w1e4','lvfree_qa2_h0.20','pl_free_qa2_w1','pl_free_qa2_w1e2','pl_free_qa2_w1e4']:
    try: d = json.load(open('runs2/'+f+'.json'))['final']
    except Exception as e: print(f, 'MISSING'); continue
    print(f, [(round(d[k], 4) if isinstance(d.get(k), float) else d.get(k)) for k in keys], 'fold', d.get('fold_deg'), d['status'][:25])
PY
