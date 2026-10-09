#!/bin/bash
# The device batch, sequential. WISTELL-A uses the prepared mirrored+scaled eq (../cold/eq_wistell.h5).
cd /home/singh/Documents/DESC2/coil_sandbox
R=work/ss_qh/joint/devices/run.sh
bash $R q1632037 1632037
bash $R q1632468 1632468
bash $R q1775228 1775228
bash $R wistell work/ss_qh/joint/cold/eq_wistell.h5
echo "BATCH DONE"
