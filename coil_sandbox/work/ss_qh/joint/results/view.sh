#!/bin/bash
# Viewer for the joint-plane study: every converged arcB2 coilset, each on its own boundary (@EQ). CPU.
# Rerun after new results land (missing files are skipped). MEM=5G by default (a GPU run may be going).
cd /home/singh/Documents/DESC2/coil_sandbox
J=work/ss_qh/joint
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=${MEM:-5G} -p MemorySwapMax=0"
H=()
add() { [ -f "$2" ] && H+=(--h5 "$1=$2@$3"); }
add "free arcB2, warm (matrix) on precise_QH"       $M/base/arcB2/result.h5            precise_QH
add "free arcB2, cold, NO guard (LINKED, failed) on precise_QH" $J/base_free/result.h5      precise_QH
add "free arcB2, cold + link guard on precise_QH"   $J/base_free_link/result.h5        precise_QH
add "JOINT z0 on precise_QH"                       $J/base_z0/result.h5               precise_QH
add "free arcB2 on B_arcB2 (matrix ss)"             $M/on_arcB2/arcB2/result.h5        $M/ss/arcB2_k4b/eq_k4.h5
add "JOINT z0 single-stage coils on B_z0 (k4b)"    $J/ss_z0_k4b/coils_k4.h5           $J/ss_z0_k4b/eq_k4.h5
add "JOINT z0 refined on B_z0 (k4b)"               $J/on_z0/arcB2_z0/result.h5        $J/ss_z0_k4b/eq_k4.h5
add "JOINT z0 single-stage coils on B_z0 (k5)"     $J/ss_z0_k5/coils_k5.h5            $J/ss_z0_k5/eq_k5.h5
add "JOINT z0 refined on B_z0 (k5)"                $J/on_z0_k5/arcB2_z0/result.h5     $J/ss_z0_k5/eq_k5.h5
add "LEVEL per coil on precise_QH"                  $J/base_level/result.h5            precise_QH
add "LEVEL single-stage coils on B_level (k4b)"     $J/ss_level_k4b/coils_k4.h5        $J/ss_level_k4b/eq_k4.h5
add "LEVEL refined on B_level (k4b)"                $J/on_level/arcB2_level/result.h5  $J/ss_level_k4b/eq_k4.h5
add "LEVEL+CONT stage-2 start (L x0.80, M3) on precise_QH" $J/cont/s2/result.h5      precise_QH
add "LEVEL+CONT single-stage coils on B_cont (k4b)" $J/cont/k4b/coils_k4.h5              $J/cont/k4b/eq_k4.h5
add "LEVEL+CONT refined on B_cont"                 $J/cont/refine/result.h5              $J/cont/k4b/eq_k4.h5
add "LEVEL+CONT RES 12 single-stage coils (k4b, 30 its)" $J/hires/k4b/coils_k4.h5     $J/hires/k4b/eq_k4.h5
add "LEVEL+CONT RES 12 refined"                    $J/hires/refine/result.h5          $J/hires/k4b/eq_k4.h5
$CAP $PY view.py $J/results/view.html --eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 \
    --device cpu --title "precise_QH, arcB2 M5 with hinges on z = 0 (joint plane) or level per coil: baselines, single stage, coil continuation, res 12" "${H[@]}"
