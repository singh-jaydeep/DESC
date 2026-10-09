# Sweep 1, remaining cases (QH nfp 4, 5) after the OOM kill; run_na2.py now evaluates the axis points with lax.map
xargs -d '\n' -P 2 -I{} bash -c '{}' < sweep1b_cmds.txt
