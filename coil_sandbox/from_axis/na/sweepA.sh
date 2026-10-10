# Stage A sweep: 4 axis types x 2 seeds, 2 chains at a time; waits for the current o2 ladder to finish.
while pgrep -f "bash o2.sh" > /dev/null; do sleep 30; done
printf "%s\n" "qa2 0" "qh5 0" "qa3 0" "qh4 0" "qa2 1" "qh5 1" "qa3 1" "qh4 1" | xargs -d '\n' -P 2 -I{} bash -c 'bash chainA.sh {}'
