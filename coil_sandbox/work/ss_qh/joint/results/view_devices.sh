#!/bin/bash
# Viewer for the other equilibria (fixed eq): level vs free arcB2 per device, 1328095 at its own design, precise_QA z = 0.
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=${MEM:-8G} -p MemorySwapMax=0"
H=()
add() { [ -f "$2" ] && H+=(--h5 "$1=$2@$3"); }
Q=$J/quasr1328095
add "q1328095 LEVEL"                 $Q/level/result.h5                 $Q/eq_q1328095.h5
add "q1328095 free (warm)"           $Q/free_warm/result.h5             $Q/eq_q1328095.h5
add "q1328095 free (cold, collapsed)" $Q/free/result.h5                $Q/eq_q1328095.h5
add "q1328095 nc5 own bounds LEVEL (stalled)" $Q/nc5/level/result.h5    $Q/eq_q1328095.h5
add "q1328095 nc5 own bounds free (warm, stalled)" $Q/nc5/free_warm/result.h5 $Q/eq_q1328095.h5
add "q1328095 nc5 own bounds free feasibility" $Q/nc5/feas_free/result.h5 $Q/eq_q1328095.h5
add "q1328095 nc5 own bounds LEVEL feasibility" $Q/nc5/feas_level/result.h5 $Q/eq_q1328095.h5
add "q1328095 nc5 own bounds LEVEL from feasible" $Q/nc5/level_from_feas/result.h5 $Q/eq_q1328095.h5
for d in q1632037 q1632468 q1775228 wistell; do
  add "$d LEVEL"        $J/devices/$d/level/result.h5      $J/devices/$d/eq.h5
  add "$d free (warm)"  $J/devices/$d/free_warm/result.h5  $J/devices/$d/eq.h5
done
add "q804112 z0"             $J/devices/q804112/z0/result.h5        $J/devices/q804112/eq.h5
add "q804112 LEVEL"          $J/devices/q804112/level/result.h5     $J/devices/q804112/eq.h5
add "q804112 free (warm)"    $J/devices/q804112/free_warm/result.h5 $J/devices/q804112/eq.h5
add "precise_QA z0"           $J/qa_z0/z0/result.h5         precise_QA
add "precise_QA LEVEL"        $J/qa_z0/level/result.h5      precise_QA
add "precise_QA free (warm)"  $J/qa_z0/free_warm/result.h5  precise_QA
add "precise_QA free (base)"  $J/qa_z0/free_base/result.h5  precise_QA
$CAP $PY view.py $J/results/view_devices.html --eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 \
    --device cpu --title "arcB2 level vs free hinges on other equilibria (fixed eq); margins vs precise_QH bounds" "${H[@]}"
