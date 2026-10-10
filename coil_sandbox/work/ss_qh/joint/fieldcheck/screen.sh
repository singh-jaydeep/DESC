#!/bin/bash
# proxy-QS screen of every QH coilset (spectrum.py), one GPU process at a time
cd /home/singh/Documents/DESC2/coil_sandbox
PY="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 /home/singh/miniforge3/envs/desc-env2/bin/python"
W=work/ss_qh; F=$W/joint/fieldcheck
while read name coils eq; do
  [ -z "$name" ] && continue
  [ -s $F/spectrum/$name.json ] && continue
  echo "=== $name $(date +%H:%M)"
  $PY $F/spectrum.py $name $coils $eq > $F/spectrum/$name.log 2>&1 || echo "FAILED $name"
done <<LIST
m_base_arcB2 $W/matrix/base/arcB2/result.h5 precise_QH
m_base_planar $W/matrix/base/planarN7/result.h5 precise_QH
m_base_xyz $W/matrix/base/xyzN6/result.h5 precise_QH
m_onA_arcB2 $W/matrix/on_arcB2/arcB2/result.h5 $W/matrix/ss/arcB2_k4b/eq_k4.h5
m_onA_planar $W/matrix/on_arcB2/planarN7/result.h5 $W/matrix/ss/arcB2_k4b/eq_k4.h5
m_onA_xyz $W/matrix/on_arcB2/xyzN6/result.h5 $W/matrix/ss/arcB2_k4b/eq_k4.h5
m_onP_arcB2 $W/matrix/on_planar/arcB2/result.h5 $W/matrix/ss/planar_k4b/eq_k4.h5
m_onP_planar $W/matrix/on_planar/planarN7/result.h5 $W/matrix/ss/planar_k4b/eq_k4.h5
m_onP_xyz $W/matrix/on_planar/xyzN6/result.h5 $W/matrix/ss/planar_k4b/eq_k4.h5
m_onX_arcB2 $W/matrix/on_xyz/arcB2/result.h5 $W/matrix/ss/xyz_k4b/eq_k4.h5
m_onX_planar $W/matrix/on_xyz/planarN7/result.h5 $W/matrix/ss/xyz_k4b/eq_k4.h5
m_onX_xyz $W/matrix/on_xyz/xyzN6/result.h5 $W/matrix/ss/xyz_k4b/eq_k4.h5
f_arcB2 $W/fixed_ss_eq/al_arcB2/result.h5 $W/arcB2/ss_k4/eq_k4.h5
f_xyz $W/fixed_ss_eq/al_xyzN6/result.h5 $W/arcB2/ss_k4/eq_k4.h5
j_base_level $W/joint/base_level/result.h5 precise_QH
j_base_z0 $W/joint/base_z0/result.h5 precise_QH
j_base_free_link $W/joint/base_free_link/result.h5 precise_QH
j_cont $W/joint/cont/refine/result.h5 $W/joint/cont/k4b/eq_k4.h5
j_hires $W/joint/hires/refine/result.h5 $W/joint/hires/k4b/eq_k4.h5
j_on_level $W/joint/on_level/arcB2_level/result.h5 $W/joint/ss_level_k4b/eq_k4.h5
j_on_z0 $W/joint/on_z0/arcB2_z0/result.h5 $W/joint/ss_z0_k4b/eq_k4.h5
d_q1328095_free $W/joint/quasr1328095/free_warm/result.h5 $W/joint/quasr1328095/eq_q1328095.h5
d_q1328095_level $W/joint/quasr1328095/level/result.h5 $W/joint/quasr1328095/eq_q1328095.h5
d_q1328095_nc5_free $W/joint/quasr1328095/nc5/free_warm/result.h5 $W/joint/quasr1328095/eq_q1328095.h5
d_q1328095_nc5_level $W/joint/quasr1328095/nc5/level/result.h5 $W/joint/quasr1328095/eq_q1328095.h5
d_q1632037_free $W/joint/devices/q1632037/free_warm/result.h5 $W/joint/devices/q1632037/eq.h5
d_q1632468_free $W/joint/devices/q1632468/free_warm/result.h5 $W/joint/devices/q1632468/eq.h5
d_q1775228_free $W/joint/devices/q1775228/free_warm/result.h5 $W/joint/devices/q1775228/eq.h5
d_wistell_free $W/joint/devices/wistell/free_warm/result.h5 $W/joint/devices/wistell/eq.h5
LIST
echo "=== screen done $(date +%H:%M)"
