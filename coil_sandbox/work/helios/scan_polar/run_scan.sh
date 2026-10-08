#!/usr/bin/env bash
# The polar arcB2 hinge-placement scan queue (serial, one GPU job at a time).
cd "$(dirname "$0")"
./run_chain.sh 90 continue 150 > cut90/chain_continue.log 2>&1; echo "cut 90 continue: exit $?"
for c in 0 30 60 120 150; do
  FREE_IT=250 ./run_chain.sh $c > cut$c/chain.log 2>&1; echo "cut $c: exit $?"
done
echo "SCAN DONE"
