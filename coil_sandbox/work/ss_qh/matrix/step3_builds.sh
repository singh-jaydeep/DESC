#!/bin/bash
# Step 3 pre-check: --maxiter 0 build of planarN7 on each k4b boundary (field rebuilt, concave_radius must be > 0.165 m).
cd /home/singh/Documents/DESC2/coil_sandbox/work/ss_qh/matrix
until grep -q "STEP2C DONE" step2c_xyz.out; do sleep 20; done
for X in planar arcB2 xyz; do [ -f ss/${X}_k4b/eq_k4.h5 ] && ./step3_cross.sh $X 0; done
echo "STEP3 BUILDS DONE $(date +%T)"
