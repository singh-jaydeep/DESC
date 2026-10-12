#!/bin/bash
# waits for a PID, then preps the next candidates (GPU, one job at a time)
cd "$(dirname "$0")"
while kill -0 $1 2>/dev/null; do sleep 30; done
for pair in "o2_qh5s2_gg1e-1_w1e-2_fb_a0.08_L10 qh5" "s2_qa2_s0_z0.04_gg1e-1_w1e-2_w1x10_fb_a0.12_L10 qa2flat"; do
  set -- $pair
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python prep.py ../na/runs2/$1 $2 > $2.prep.log 2>&1
done
