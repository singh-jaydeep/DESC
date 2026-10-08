#!/bin/bash
# Step 3: the nine run_al.py refinements, boundary by boundary, after the build checks.
cd /home/singh/Documents/DESC2/coil_sandbox/work/ss_qh/matrix
until grep -q "STEP3 BUILDS DONE" step3_builds.out; do sleep 20; done
for X in planar arcB2 xyz; do ./step3_cross.sh $X; done
echo "STEP3 ALL DONE $(date +%T)"
