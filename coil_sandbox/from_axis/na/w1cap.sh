# Capped near-planar QA nfp2 (abs(Z) <= 0.04, s0): step 3 again with stiff first order (w1 = 10), then free boundary.
while [ "$(ps -eo args | grep -c '^python -u score_fb.py')" -ge 1 ]; do sleep 30; done
out=runs2/s2_qa2_s0_z0.04_gg1e-1_w1e-2_w1x10
eval "$(python cont.py runs2/s2_qa2_s0_z0.04_gg1e-2_w1e-3 $out --order 2 --wgg 1e-1 --w2 1e-2 --w1 10 --maxiter 600)" > $out.log 2>&1
python -u score_fb.py $out --a 0.12 --out ${out}_fb_a0.12_L10 > runs2/fb_o2w1cap_qa2.log 2>&1
