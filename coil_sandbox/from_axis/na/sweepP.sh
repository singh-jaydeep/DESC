# Planar-axis search (axis |Z| capped) through the second-order ladder, 2 chains at a time; unconstrained baselines last.
# (sweepA.sh never ran: its "pgrep -f" wait matched a monitor's own command line.)
printf "%s\n" "qa2 0 0.04" "qh5 0 0.03" "qa3 0 0.02" "qh4 0 0.04" "qa2 1 0.04" "qh5 1 0.03" "qa3 1 0.02" "qh4 1 0.04" "qa2 0" "qh5 0" \
  | xargs -d '\n' -P 2 -I{} bash -c 'bash chainA.sh {}'
