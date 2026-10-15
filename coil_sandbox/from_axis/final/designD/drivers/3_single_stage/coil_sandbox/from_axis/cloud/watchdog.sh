#!/bin/bash
# watchdog.sh APP_ID DEADLINE_EPOCH JOB : stop a Modal app at the deadline unless its job finished first.
# Runs on this machine (it must stay awake); the job's per-iteration checkpoints make the stop lossless.
APP=$1; DL=$2; J=$3; LOG=$(dirname "$0")/watchdog_$J.log
echo "$(date) watching $APP ($J), hard stop at $(date -d @$DL)" >> $LOG
while [ $(date +%s) -lt $DL ]; do
  modal volume ls from-axis-runs $J 2>/dev/null | grep -q "DONE" && { echo "$(date) job finished on its own" >> $LOG; exit 0; }
  sleep 60
done
modal app stop -y $APP >> $LOG 2>&1
sleep 10
n=$(modal container list --json 2>/dev/null | python -c "import json,sys; print(sum(c['app_id']=='$APP' for c in json.load(sys.stdin)))")
echo "$(date) DEADLINE: stopped $APP; its active containers now: $n" >> $LOG
