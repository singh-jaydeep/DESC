# Rest of the planar sweep. A scheduler loop instead of xargs: GNU xargs stops for good once a child is killed by a
# signal, so killing one bad chain ended the whole queue. Keeps 2 chains running (counts live chainA.sh processes).
for job in "qa3 0 0.02" "qh4 0 0.04" "qa2 1 0.04" "qh5 1 0.03" "qa3 1 0.02" "qh4 1 0.04" "qa2 0" "qh5 0"; do
  while [ "$(ps -eo args | grep -c '^bash chainA.sh')" -ge 2 ]; do sleep 20; done
  nohup bash chainA.sh $job > /dev/null 2>&1 &
  sleep 5
done
wait
