# Pass 1: QH nfp4, z0, constrained currents, NAE-surface clearance, nc = 4, 5, 6 (deterministic start)
for nc in 4 5 6; do
  python run_na2.py --nfp 4 --nc $nc --dcc 0.04 --maxiter 600 --out runs2/qh_z0_nc${nc} > runs2/qh_z0_nc${nc}.log 2>&1
done
