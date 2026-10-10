# Does a stiff first order (w1 = 10) keep near-axis QS while the second-order step improves the edge? QA nfp2 s2.
out=runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10
eval "$(python cont.py runs2/o2_qa2s2_gg1e-2_w1e-3 $out --order 2 --wgg 1e-1 --w2 1e-2 --w1 10 --maxiter 600)" > $out.log 2>&1
python -u score_fb.py $out --a 0.12 --out ${out}_fb_a0.12_L10 > runs2/fb_o2w1_qa2.log 2>&1
