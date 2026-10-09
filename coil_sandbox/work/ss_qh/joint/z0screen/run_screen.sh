#!/bin/bash
# z = 0 joint-plane screen over QUASR groups (boundary only, CPU, 4G cap). Best 60 by QS per group.
cd /home/singh/Documents/DESC2/coil_sandbox
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CPU="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0"
O=work/ss_qh/joint/z0screen/screen.jsonl
for h in 1 0; do
  for nfp in 2 3 4 5; do
    [ $h = 0 ] && [ $nfp -gt 3 ] && continue
    for A in 6.67 8 10; do
      echo "=== helicity $h nfp $nfp A $A $(date +%T)"
      $CPU $PY work/ss_qh/joint/z0screen/screen.py --helicity $h --nfp $nfp --A $A --qs-max -3.0 --n 60 --out $O 2>&1 | grep -E "devices match|Error" 
    done
  done
done
echo "SCREEN DONE"
