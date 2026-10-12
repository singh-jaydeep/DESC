# Elongation-cap ladder, continued from the w = 1e-3 regularized states (length + bending weight kept at 1e-3).
chain() {  # $1 start run, $2 tag
  prev=$1
  for e in 5 4; do
    out=runs2/elong_$2_e$e
    eval "$(python cont.py $prev $out --emax $e --wlen 1e-3 --wbend 1e-3 --maxiter 600)" > $out.log 2>&1
    prev=$out
  done
}
chain runs2/reg_qa2s2_w1e-3 qa2s2 &
chain runs2/reg_qh5s2_w1e-3 qh5s2 &
wait
