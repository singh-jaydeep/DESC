# Phase 2 runs 1 and 2: QA nfp2 s2 at a smaller radius (inside the clean region), then QH nfp5 s2 at its design a.
python -u score_fb.py runs2/elong_qa2s2_e5 --a 0.12 --out runs2/elong_qa2s2_e5_fb_a0.12_L10 > runs2/fb_qa2_a012.log 2>&1
python -u score_fb.py runs2/elong_qh5s2_e5 --out runs2/elong_qh5s2_e5_fb_L10 > runs2/fb_qh5.log 2>&1
