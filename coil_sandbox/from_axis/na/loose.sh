#!/bin/bash
# Weakening the hinge requirement, one notch at a time (QA nfp2, second-order weights and bounds of the z = 0 best state).
#   height cap H  : both hinges of every coil within |z| <= H, H = 0.1, 0.2, 0.3 x a   (a = 0.167 in these units)
#   tilt          : level (each coil's hinge line horizontal), <= 2 degrees, or free inside the height cap
#   vacuum iota   : 0.30, 0.42, 0.50 (each from its own z = 0 state, so the gain is measured against the right baseline)
# CPU, two at a time, ~5 min each. Table: python loose_table.py
cd "$(dirname "$0")"
B42=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10; B30=runs2/li_qa2_iota0.30; B50=runs2/hi_qa2_iota0.50
run() { systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c "$1" > $2.log 2>&1; }
[ -f $B50.json ] || run "$(python cont.py $B42 $B50 --iota 0.50 --maxiter 600)" $B50
for it in 30:$B30 42:$B42 50:$B50; do
  I=${it%%:*}; src=${it#*:}
  for H in 0.0167 0.0334 0.0501; do
    for T in level 2 free; do
      out=runs2/lo_qa2_i${I}_h${H}_t$T
      [ -f $out.json ] && continue
      case $T in level) x="--wlevel 1e4";; free) x="";; *) x="--tiltmax $T";; esac
      echo "systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash -c \"\$(python cont.py $src $out --mode free --hzmax $H $x --maxiter 600)\" > $out.log 2>&1"
    done
  done
done | xargs -d '\n' -P 2 -I{} bash -c '{}'
python loose_table.py
