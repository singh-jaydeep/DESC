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
add "free arcB2, cold + link guard on precise_QH"   $J/base_free_link/result.h5        precise_QH
add "JOINT z0 on precise_QH"                       $J/base_z0/result.h5               precise_QH
add "free arcB2 on B_arcB2 (matrix ss)"             $M/on_arcB2/arcB2/result.h5        $M/ss/arcB2_k4b/eq_k4.h5
add "JOINT z0 single-stage coils on B_z0 (k4b)"    $J/ss_z0_k4b/coils_k4.h5           $J/ss_z0_k4b/eq_k4.h5
add "JOINT z0 refined on B_z0 (k4b)"               $J/on_z0/arcB2_z0/result.h5        $J/ss_z0_k4b/eq_k4.h5
add "JOINT z0 single-stage coils on B_z0 (k5)"     $J/ss_z0_k5/coils_k5.h5            $J/ss_z0_k5/eq_k5.h5
add "JOINT z0 refined on B_z0 (k5)"                $J/on_z0_k5/arcB2_z0/result.h5     $J/ss_z0_k5/eq_k5.h5
$CAP $PY view.py $J/results/view.html --eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 \
    --device cpu --title "precise_QH, arcB2 M5 with every hinge on z = 0 (joint plane): baselines and single stage" "${H[@]}"
