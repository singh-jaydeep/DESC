#!/bin/bash
# Stage A chain for one (case, seed[, zmax]): first order (smooth distances) -> length+bending regularizer ->
# second-order ladder (wgg, w2). Usage: bash chainA.sh CASE SEED      (from na/, CPU)
case=$1; seed=$2; zmax=${3:-0}
case $case in
  qa2) base="--nfp 2 --nc 4 --axis0 '1,0.15;0,-0.10' --eta0 0.6 --iota 0.42 --a 0.167" ;;
  qa3) base="--nfp 3 --nc 4 --axis0 '1,0.05;0,-0.04' --eta0 0.9 --iota 0.40 --a 0.143" ;;
  qh4) base="--nfp 4 --nc 4 --axis0 '1,0.13;0,-0.10' --eta0 1.6 --iota 1.25 --a 0.124" ;;
  qh5) base="--nfp 5 --nc 3 --axis0 '1,0.08;0,-0.06' --eta0 1.6 --iota 1.30 --a 0.110" ;;
  *) echo "unknown case $case"; exit 1 ;;
esac
a=$(echo "$base" | sed 's/.*--a //')
bnd="--dpc $(python -c "print(0.885*$a)") --kmax $(python -c "print(1.699/$a)") --Lmax $(python -c "print(23.6*$a)") --dcc 0.04 --emax 5"
tag=s2_${case}_s${seed}
if [ "$zmax" != "0" ]; then tag=${tag}_z${zmax}; bnd="$bnd --zmax $zmax"; fi
o1=runs2/${tag}_o1
[ -f $o1.json ] || eval "python run_na2.py $base $bnd --maxiter 400 --seed $seed --out $o1" > $o1.log 2>&1
reg=runs2/${tag}_reg
[ -f $reg.json ] || eval "$(python cont.py $o1 $reg --wlen 1e-3 --wbend 1e-3 --maxiter 400)" > $reg.log 2>&1
prev=$reg
# --w1 10 keeps first order stiff (QA nfp2 s2 step 3: near-axis QS 6.7e-3 -> 3.2e-3, edge 2.8e-2 -> 2.3e-2).
# the (1, 1e-1) step overpowered the first-order rows on QA nfp2 s2 (field mismatch 11%): stop at (1e-1, 1e-2)
for pair in "1e-3 1e-4" "1e-2 1e-3" "1e-1 1e-2"; do
  set -- $pair; out=runs2/${tag}_gg$1_w$2
  [ -f $out.json ] || eval "$(python cont.py $prev $out --order 2 --wgg $1 --w2 $2 --w1 10 --maxiter 600)" > $out.log 2>&1
  prev=$out
done
