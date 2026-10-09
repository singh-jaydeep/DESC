set -x
C="--nfp 2 --nc 4 --iota 0.42 --rcoil 0.5 --dpa 0.35 --Lmax 3.6 --kmax 10 --dcc 0.08"
python run_na.py $C --mode z0 --init runs/qa_z0 --maxfev 1500 --out runs/qa_z0b
python run_na.py $C --mode free --axis0 "1,0.15;0,0.1" --maxfev 1700 --out runs/qa_free
python run_na.py $C --mode free --init runs/qa_z0b --maxfev 1000 --out runs/qa_z0b_to_free
