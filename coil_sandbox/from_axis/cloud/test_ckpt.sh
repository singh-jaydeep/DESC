#!/bin/bash
# CPU test of single_stage.py --ckpt-every / --auto-resume on a tiny (basis 6) copy of the QA problem.
#  A: k=1, 2 iterations            -> trials_k1.txt, [ckpt] lines, eq_k1/coils_k1, anchors.json
#  B: k=2 started, KILLED once ckpt.json exists (a mid-step death)
#  C: --auto-resume                -> must pick up the k=2 checkpoint with the same anchors and finish k=2
cd "$(dirname "$0")/../ss"
# usage: test_ckpt.sh [DIR]   device from SANDBOX_DEVICE (default cpu; the CPU run needs > 8 GB of RAM: it was
# OOM-killed on the laptop next to a GPU job, so it is run on Modal: job ckpt_test)
T=${1:-cloud_runs/_local/ckpt_test}; rm -rf $T/ss $T/start; mkdir -p $T
export SANDBOX_DEVICE=${SANDBOX_DEVICE:-cpu}; DEV=$SANDBOX_DEVICE
if command -v systemd-run >/dev/null && [ -z "$FA_ROOT" ]; then
  run() { systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -E SANDBOX_DEVICE=$DEV "$@"; }
else
  run() { "$@"; }
fi
run python prep.py ../na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10_fb_a0.12_L10 $T/start --LMN 6 --device $DEV > $T/prep.log 2>&1 || { echo "PREP FAILED"; tail -3 $T/prep.log; exit 1; }
SS="python ../../single_stage.py --device $DEV --eq $T/start/eq0_b6.h5 --start $T/start/coils0.h5 --out $T/ss --bounds ../../bounds/fa_qa2_a0167.json --fix-hinge-z 0 --dist rows --rows-select 1.6 --cc-rows 3000 --pc-rows 3000 --coil-grid-n 60 --pc-gap 0.005 --plasma-m 16 --plasma-n 16 --eval-m 12 --eval-n 12 --helicity 1,0 --qs-form boozer --qs-rho 0.5,1.0 --A-min 6.5 --A-max 7.5 --iota-min 0.25 --tau-bn 2e-3 --tau-qs-rel 0.3 --src-gl 24 --force-chunk 200 --method lsq-composite --hessian secant --solve-tol 1e-6 --track-steps --ckpt-every 1 --auto-resume"
run $SS --kmin 1 --kmax 1 --maxiter 2 > $T/A.log 2>&1; echo "A rc $?"
ls $T/ss; cat $T/ss/anchors.json; echo; cat $T/ss/trials_k1.txt; grep -E "\[ckpt\]|AUTO-RESUME|^k=1 solver" $T/A.log
run $SS --kmin 1 --kmax 2 --maxiter 6 > $T/B.log 2>&1 &
for i in $(seq 1 240); do [ -f $T/ss/ckpt.json ] && grep -q '"k": 2' $T/ss/ckpt.json && break; sleep 5; done
cat $T/ss/ckpt.json; echo
ps -eo pid,args | grep "single_stage.py" | grep "$T/ss" | grep -v grep | awk '{print $1}' | xargs -r kill -9; sleep 3
echo "B killed; files:"; ls $T/ss; grep -E "AUTO-RESUME|anchors from|\[ckpt\]" $T/B.log
run $SS --kmin 1 --kmax 2 --maxiter 2 > $T/C.log 2>&1; echo "C rc $?"
grep -E "AUTO-RESUME|anchors from|\[ckpt\]|^k=2 (physics|solver)|^START physics" $T/C.log | cut -c1-250; ls $T/ss; cat $T/ss/trials_k2.txt | tail -12
python -c "
import json;r=json.load(open('$T/ss/result.json'));print('steps in result.json:',[s['k'] for s in r['steps']],'resumed entries:',len(r.get('resumed',[])),'step_log entries k2:',len(r['steps'][-1].get('step_log',[])))"
run $SS --kmin 1 --kmax 2 --maxiter 2 > $T/D.log 2>&1; echo "D (nothing left) rc $?"; grep AUTO-RESUME $T/D.log
grep -q "AUTO-RESUME: mid-step checkpoint k=2" $T/C.log && grep -q "anchors from" $T/C.log && [ -f $T/ss/eq_k2.h5 ] && grep -q "nothing to do" $T/D.log && echo "CKPT TEST PASSED" || { echo "CKPT TEST FAILED"; exit 1; }
