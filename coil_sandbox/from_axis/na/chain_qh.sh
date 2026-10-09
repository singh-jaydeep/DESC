set -x
C="--nfp 4 --nc 4 --iota 1.25 --rcoil 0.42 --dpa 0.28 --Lmax 2.93 --kmax 13.7 --dcc 0.0585 --eta0 1.6 --welong 1 --emax 5"
python run_na.py $C --mode z0 --axis0 "1,0.17;0,-0.13" --maxfev 1500 --out runs/qh_z0
python run_na.py $C --mode free --axis0 "1,0.17;0,-0.13" --maxfev 1500 --out runs/qh_free
C2="--nfp 4 --nc 4 --iota 1.25 --rcoil 0.42 --dpa 0.28 --Lmax 2.93 --kmax 13.7 --dcc 0.0585 --eta0 1.6 --welong 10 --emax 4 --imax 2"
python run_na.py $C2 --mode z0 --axis0 "1,0.17;0,-0.13" --maxfev 2500 --out runs/qh_z0_c
python run_na.py $C2 --mode z0 --init runs/qh_z0 --maxfev 2500 --out runs/qh_z0_c_warm
