# coil_sandbox

A portable copy of the DESC used for the precise_QH/QA coil sweeps (`vendor/desc`), with small
helpers for one-off coil trials on any equilibrium. Copy the folder anywhere.

```bash
pip install -r vendor/requirements.txt    # then pin to tested_env.txt (numpy 2.4.x!); jax[cuda12] for GPU
python check.py                           # the vendored DESC is used, and which device

python make_start.py --eq precise_QA --rep arc --B 5 --out work/qa/start.h5
python run_one.py   --eq precise_QA --start work/qa/start.h5 --out work/qa/run1
python view.py work/qa/view.html --eq precise_QA --h5 start=work/qa/start.h5 --h5 run=work/qa/run1/result.h5
```

`--eq` takes a `desc.examples` name, a DESC `.h5`, a `wout_*.nc`, or a VMEC/DESC input file.
Your own scripts: start with `sys.path.insert(0, "sandbox"); import boot; boot.setup()` before
importing `desc`. `CLAUDE.md` has the details, the machine rules, and the lessons behind the recipe.
