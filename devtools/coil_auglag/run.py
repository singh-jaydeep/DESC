"""Run lsqtr on the stage-2 coil problem with the chosen fixes."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402

from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--maxiter", type=int, default=150)
p.add_argument("--arclen", default=None)
p.add_argument("--arclen_w", type=float, default=None)
p.add_argument("--cc", default=None)
p.add_argument("--cc_alpha", type=float, default=None)
p.add_argument("--cc_bound", type=float, default=None)
p.add_argument("--cc_w", type=float, default=None)
p.add_argument("--pair_N", type=int, default=None)
p.add_argument("--curv", default=None)
p.add_argument("--curv_target", type=float, default=None)
p.add_argument("--curv_w", type=float, default=None)
p.add_argument("--case", default="qa")
p.add_argument("--kw", default="{}", help="extra builder kwargs as JSON")
p.add_argument("--phase", action="store_true")
p.add_argument("--start", default=None)
p.add_argument("--method", default="lsq-exact")
a = p.parse_args()
kw = {
    k: v
    for k, v in vars(a).items()
    if k not in ("tag", "maxiter", "start", "method", "phase", "kw") and v is not None
}
kw.update(json.loads(a.kw))
eq, c0, obj, cons = build(**kw)
if (
    a.start
):  # build on c0 so normalizations match a fresh run, then copy in start params
    obj.build(verbose=0)
    for c in cons:
        c.build(verbose=0)
    c0.params_dict = load(a.start).params_dict
if a.phase:  # pin the parameter phase of each coil at its current value
    from common import phase_tangent

    from desc.objectives import FixParameters

    modes = c0[0].X_basis.modes[:, 2]
    fix = []
    for q in c0.params_dict:
        t = phase_tangent(q, modes)
        best = max(
            ((k, i) for k in t for i in np.where(np.abs(modes) == 1)[0]),
            key=lambda ki: abs(t[ki[0]][ki[1]]),
        )
        fix.append({best[0]: np.array([best[1]])})
    print("phase fix:", fix)
    cons = cons + (FixParameters(c0, fix),)
json.dump(kw, open(f"{a.tag}.json", "w"))
(opt,), res = Optimizer(a.method).optimize(
    c0,
    objective=obj,
    constraints=cons,
    maxiter=a.maxiter,
    verbose=3,
    xtol=1e-16,
    ftol=1e-16,
    gtol=1e-12,
    copy=True,
)
opt.save(f"{a.tag}.h5")
np.save(f"{a.tag}_allx.npy", np.array([np.asarray(v) for v in res["allx"]]))
