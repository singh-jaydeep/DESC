# Sweep 1: breadth of axis types, z0 hinges, currents in [0, 2x], NAE-surface clearance, 4 seeds each.
# Bounds: Gil set scaled by a (precise_QH: a 0.124 -> d_pc 0.11, kappa 13.7, L 2.93); d_cc relaxed to 0.04.
# case nfp nc axis eta iota a
cat > /tmp/sweep1_cases.txt <<'C'
qa2 2 4 1,0.15;0,-0.10 0.6 0.42 0.167
qa3 3 4 1,0.05;0,-0.04 0.9 0.40 0.143
qh3 3 4 1,0.15;0,-0.13 1.2 1.10 0.143
qh4 4 4 1,0.13;0,-0.10 1.6 1.25 0.124
qh5 5 3 1,0.08;0,-0.06 1.6 1.30 0.110
C
while read case nfp nc axis eta iota a; do
  for seed in 0 1 2 3; do
    dpc=$(python -c "print(0.885*$a)"); km=$(python -c "print(1.699/$a)"); Lm=$(python -c "print(23.6*$a)")
    echo "python run_na2.py --nfp $nfp --nc $nc --axis0 '$axis' --eta0 $eta --iota $iota --a $a --dpc $dpc --kmax $km --Lmax $Lm --dcc 0.04 --emax 6 --maxiter 400 --seed $seed --out runs2/s1_${case}_z0_s${seed} > runs2/s1_${case}_z0_s${seed}.log 2>&1"
  done
done < /tmp/sweep1_cases.txt > sweep1_cmds.txt
xargs -d '\n' -P 2 -I{} bash -c '{}' < sweep1_cmds.txt
