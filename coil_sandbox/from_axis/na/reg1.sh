# Regularizer ladder (length + bending, equal weights) continued from the two chosen seeds; w = 0 first:
# an unregularized continuation with the smooth (soft-min) distances = the baseline the ladder is compared to.
chain() {  # $1 seed run, $2 tag
  prev=$1
  for w in 0 1e-5 1e-4 1e-3; do
    out=runs2/reg_$2_w$w
    eval "$(python cont.py $prev $out --wlen $w --wbend $w --maxiter 600)" > $out.log 2>&1
    prev=$out
  done
}
chain runs2/s1_qa2_z0_s2 qa2s2 &
chain runs2/s1_qh5_z0_s2 qh5s2 &
wait
