#!/usr/bin/env bash
# Time tr_method="qr" against tr_method="qr-struct" on real DESC equilibrium
# solves, using YOUR laptop GPU and YOUR desc-env.
#
# My sandbox cannot see /dev/nvidia*, so this leg has to run from your shell.
#
#   conda activate desc-env
#   bash run_laptop_gpu.sh
#
# Writes solve_bench_gpu.json next to the script. Runtime is roughly 20-40 min
# for the default four cases (each solve is a cold process, by design).
set -euo pipefail

WT=/home/singh/Documents/DESC/.worktrees/alpha-loop
cd "$WT"

echo "=== branch ==="
git log --oneline -1
echo

echo "=== GPU visible to jax? ==="
python - <<'EOF'
import jax
devs = jax.devices()
print("jax", jax.__version__, "->", devs)
if devs[0].platform != "gpu":
    raise SystemExit(
        "\nNo GPU backend. This env needs a CUDA-enabled jaxlib:\n"
        "  pip install -U 'jax[cuda12]'\n"
        "(check `nvidia-smi` works first). Aborting so we don't\n"
        "accidentally publish CPU numbers as GPU numbers."
    )
EOF
echo

echo "=== is the branch DESC the one being imported? ==="
python -c "
import desc, sys
print('desc      :', desc.__file__)
print('version   :', desc.__version__)
from desc.optimize.tr_subproblems import trust_region_step_exact_qr_struct
print('qr-struct : present')
if '$WT' not in desc.__file__:
    sys.exit('desc is being imported from elsewhere — run: pip install -e $WT')
"
echo

echo "=== running (each solve in a fresh process) ==="
export DESC_BENCH_DEVICE=gpu
export DESC_BENCH_MAXITER=25
export DESC_BENCH_REPS=2
python branch_experiments/solve_bench.py \
  '[["HELIOTRON",6,0.01],["precise_QA",6,0.01],["W7-X",6,0.01],["HELIOTRON",8,0.01]]'

echo
echo "=== done: solve_bench_gpu.json ==="
ls -la solve_bench_gpu.json 2>/dev/null || echo "(check output above for errors)"
