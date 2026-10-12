#!/bin/bash
# cold-start check of the planar result: full chain (first order -> regularizer -> second-order ladder) with the coils
# planar from the start. plc = general planar (free hinges |z| <= 0.2); CPU, one at a time.
cd "$(dirname "$0")"
for seed in 0 2; do
  EXTRA="--mode free --hzmax 0.2 --wplanar 1e4" PTAG=plc systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 bash chainP.sh qa2 $seed
done
python - <<'PY'
import json, glob
keys = ('cost_field','max_dB','max_dG_rel','max_dGG_rel','iota','B20_residual','max_abs_Zaxis','max_elong','nit')
for f in sorted(glob.glob('runs2/plc_qa2_s*.json')):
    d = json.load(open(f))['final']
    print(f.split('/')[-1][:-5], [(round(d[k], 4) if isinstance(d.get(k), float) else d.get(k)) for k in keys], 'fold', d.get('fold_deg'), d['status'][:25])
PY
