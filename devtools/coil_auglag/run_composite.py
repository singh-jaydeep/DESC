"""Run Optimizer("lsq-composite") on a stage-2 coil problem.

usage: python run_composite.py TAG KW_JSON [OPTIONS_JSON] [--maxiter N] [--start X]
KW_JSON goes to the builder (as in mkstage.py); OPTIONS_JSON to the solver, default
{"hessian": "secant", "finish_steps": 8}. --start takes a full state (*_x.npy).
Writes TAG.json, TAG.h5, TAG_x.npy and TAG_hist.json.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402

from desc.optimize import Optimizer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("kw")
p.add_argument("options", nargs="?", default='{"hessian": "secant", "finish_steps": 8}')
p.add_argument("--maxiter", type=int, default=1000)
p.add_argument("--gtol", type=float, default=1e-10)
p.add_argument("--start", default=None)
a = p.parse_args()
kw = json.loads(a.kw)
eq, c0, obj, cons = build(**kw)
if a.start:  # objective built on the initial coils so normalizations match
    obj.build(verbose=0)
    # constraints build on the loaded state: setting params normalizes planar normals
    c0.params_dict = obj.unpack_state(np.load(a.start), False)[0]
json.dump(kw, open(f"{a.tag}.json", "w"))
(opt,), res = Optimizer("lsq-composite").optimize(
    c0,
    objective=obj,
    constraints=cons,
    maxiter=a.maxiter,
    verbose=3,
    ftol=0,
    xtol=0,
    gtol=a.gtol,
    options=json.loads(a.options),
    copy=True,
)
opt.save(f"{a.tag}.h5")
np.save(f"{a.tag}_x.npy", np.asarray(res["allx"][-1]))
json.dump(res["step_history"], open(f"{a.tag}_hist.json", "w"))
if res.get("step_log"):
    json.dump(res["step_log"], open(f"{a.tag}_steps.json", "w"), indent=0)
for h in res["step_history"]:
    if (
        h["iteration"] % 10 == 0
        or h is res["step_history"][-1]
        or h["hessian"] == "exact"
    ):
        print(
            f"{h['iteration']:4d} cost={h['cost']:.12e} opt={h['optimality']:.3e} "
            f"lam={h['damping']:.2e} {h['hessian']}{'+S' if h['model_S'] else ''} "
            f"active={h['active_hinges']}"
        )
