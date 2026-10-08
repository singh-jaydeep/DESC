#!/bin/bash
# Viewer: all 12 converged coilsets, each on its own boundary (@EQ). CPU, under the cap.
cd /home/singh/Documents/DESC2/coil_sandbox
M=work/ss_qh/matrix
PY=/home/singh/miniforge3/envs/desc-env2/bin/python
CAP="systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0"
declare -A EQ=([B0]=precise_QH [B_planar]=$M/ss/planar_k4b/eq_k4.h5 [B_arcB2]=$M/ss/arcB2_k4b/eq_k4.h5 [B_xyz]=$M/ss/xyz_k4b/eq_k4.h5)
declare -A DIR=([B0]=$M/base [B_planar]=$M/on_planar [B_arcB2]=$M/on_arcB2 [B_xyz]=$M/on_xyz)
H=()
for b in B0 B_planar B_arcB2 B_xyz; do
  for rep in planarN7 arcB2 xyzN6; do
    [ -f ${DIR[$b]}/$rep/result.h5 ] && H+=(--h5 "$rep on $b=${DIR[$b]}/$rep/result.h5@${EQ[$b]}")
  done
done
$CAP $PY view.py $M/view.html --eq precise_QH --vacuum --bounds bounds/precise_qh_gil.json --length-mult 1.0 \
    --device cpu --title "precise_QH boundary x coils matrix: planar N7 / arcB2 M5 / FourierXYZ N6 on B0 and three single-stage boundaries" "${H[@]}"
