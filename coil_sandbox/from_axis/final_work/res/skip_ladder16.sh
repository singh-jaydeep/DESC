#!/bin/bash
# stop overnight_B.sh when it reaches the basis-16 re-solves (not needed: D showed basis 14 vs 16 agree to 1-4%)
cd "$(dirname "$0")"
until grep -q "### ladder16" overnight_B.out 2>/dev/null; do sleep 30; pgrep -f overnight_B.sh >/dev/null || exit 0; done
for p in $(pgrep -f "overnight_B.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "python resladder.py"); do kill $p; done
echo "$(date +%T) ladder16 skipped by skip_ladder16.sh; chain stopped" >> overnight_B.out
