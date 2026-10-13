#!/bin/bash
# Weakening the engineering bounds one at a time (QA nfp2, iota target 0.42), at two hinge settings:
#   z0 : every hinge on z = 0 (the best z = 0 state)       lh : hinges within 0.2 a of z = 0, tilt <= 2 degrees
# knobs (a = 0.167): plasma-coil distance 0.885 a -> 0.8, 0.7, 0.6 a; max curvature x1.15, x1.3; max length x1.15, x1.3;
#   coil-coil distance 0.04 -> 0.03, 0.02; coils per half period 4 -> 5, 6 (cold starts, with a cold 4-coil control).
# CPU, two at a time. Table: python loose2_table.py
cd "$(dirname "$0")"
Z0=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10; LH=runs2/lo_qa2_i42_h0.0334_t2
pre="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 bash -c"
{
for hs in z0:$Z0 lh:$LH; do
  S=${hs%%:*}; src=${hs#*:}
  for kv in dpc:0.1336 dpc:0.1169 dpc:0.1002 kmax:11.70 kmax:13.23 Lmax:4.532 Lmax:5.124 dcc:0.03 dcc:0.02; do
    k=${kv%%:*}; v=${kv#*:}; out=runs2/l2_qa2_${S}_${k}${v}
    [ -f $out.json ] && continue
    echo "$pre \"\$(python cont.py $src $out --$k $v --maxiter 600)\" > $out.log 2>&1"
  done
  for nc in 4 5 6; do
    out=runs2/l2_qa2_${S}_cold_nc$nc
    [ -f $out.json ] && continue
    echo "$pre \"\$(python cont.py $src $out --nc $nc --maxiter 600 | sed 's#--init [^ ]* ##') --axis0 '1,0.17;0,-0.13' --eta0 1.6 --rcoil 0.42 --seed -1\" > $out.log 2>&1"
  done
done
} | xargs -d '\n' -P 2 -I{} bash -c '{}'
python loose2_table.py
