#!/bin/bash
# after.sh PID DIR...: wait for PID, then free-boundary rescore each qa2best/DIR at its last k (one GPU job at a time)
cd "$(dirname "$0")"
P=$1; shift
while kill -0 $P 2>/dev/null; do sleep 30; done
for d in "$@"; do
  k=$(ls qa2best/$d/eq_k*.h5 | sed 's/.*eq_k\([0-9]*\).h5/\1/' | sort -n | tail -1)
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python fb_rescore.py qa2best/$d/eq_k$k.h5 \
    qa2best/$d/coils_k$k.h5 qa2best/${d}_fb > qa2best/${d}_fb.log 2>&1
  echo ALLDONE >> qa2best/${d}_fb.log
done
