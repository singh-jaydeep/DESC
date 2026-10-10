# Second-order continuation: ramp the coil grad grad B weight (wgg) and the B20 QS weight (w2) together,
# continued from the elongation <= 5 seeds (length + bending regularizer kept at 1e-3).
chain() {  # $1 start run, $2 tag (also TAG), $3 initial B2c
  prev=$1; extra="--B2c0 $3"
  for pair in "1e-3 1e-4" "1e-2 1e-3" "1e-1 1e-2" "1 1e-1"; do
    set -- $pair; out=runs2/o2_${TAG}_gg$1_w$2
    eval "$(python cont.py $prev $out --order 2 --wgg $1 --w2 $2 $extra --maxiter 600)" > $out.log 2>&1
    prev=$out; extra=""
  done
}
TAG=qa2s2 chain runs2/elong_qa2s2_e5 qa2s2 1.20 &
TAG=qh5s2 chain runs2/elong_qh5s2_e5 qh5s2 0.50 &
wait
