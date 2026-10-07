"""Run Optimizer("lsq-auglag-composite") on a stage-2 coil problem in AL mode.

usage: python run_al.py TAG KW_JSON [OPTIONS_JSON] [--maxiter N] [--start X]
KW_JSON goes to the builder with al=True (distance "node", pair_N None unless
given). OPTIONS_JSON goes to the solver. --start takes a full state (*_x.npy).
Writes TAG.json; TAG_x.npy after every accepted step; after every outer iteration k,
TAG_outer.json (summary), TAG_outer{k}.npz (state, row ids, multipliers and penalties
before and after the update) and TAG_steps{k}.json (every attempted inner step);
TAG.h5 at the end.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402
from check import distance_objectives, feasibility  # noqa: E402

from desc.optimize import Optimizer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("kw")
p.add_argument("options", nargs="?", default="{}")
p.add_argument("--maxiter", type=int, default=2000)
p.add_argument("--gtol", type=float, default=1e-6)
p.add_argument("--ctol", type=float, default=1e-6)
p.add_argument("--start", default=None)
a = p.parse_args()
kw = {"distance": "node", "pair_N": None, **json.loads(a.kw), "al": True}
eq, c0, obj, cons = build(**kw)
obj.build(verbose=0)
nonlinear = [c for c in cons if not c.linear]
for c in nonlinear:
    if not c.built:
        c.build(verbose=0)
if a.start:  # everything built on the initial coils so normalizations match
    c0.params_dict = obj.unpack_state(np.load(a.start), False)[0]
json.dump(kw, open(f"{a.tag}.json", "w"))
sizes = [getattr(c, "num_row_ids", c.dim_f) for c in nonlinear]
offsets = np.concatenate([[0], np.cumsum(sizes)])
cc, pc = distance_objectives(obj, cons)
check_coils = c0.copy()
outer_log = []


def group_of(ids):
    return np.searchsorted(offsets, ids, side="right") - 1


def report(h):
    """One line per outer iteration, plus per-group detail."""
    x = np.asarray(h["x"])
    np.save(f"{a.tag}_x.npy", x)
    k = h["outer"]
    np.savez(
        f"{a.tag}_outer{k}.npz",
        x=x,
        **{q: np.asarray(h[q]) for q in ("ids", "y", "mu", "y_next", "mu_next")},
        violation=np.asarray(h["violation"]),
    )
    json.dump(h["step_log"], open(f"{a.tag}_steps{k}.json", "w"), indent=0)
    params = obj.unpack_state(x, False)[0]
    params = [{k: np.asarray(v) for k, v in q.items()} for q in params]
    true = feasibility(check_coils, cc, pc, params)
    ids, y, mu, viol = (np.asarray(h[k]) for k in ("ids", "y", "mu", "violation"))
    used = ids >= 0
    grp = np.where(used, group_of(ids), -1)
    groups = {}
    for k, c in enumerate(nonlinear):
        m = grp == k
        if not np.any(m):
            continue
        groups[c.name] = dict(
            rows=int(m.sum()),
            mu=[float(np.min(mu[m])), float(np.median(mu[m])), float(np.max(mu[m]))],
            max_y=float(np.abs(y[m]).max()),
            nonzero_y=int(np.sum(y[m] != 0)),
            max_violation=float(viol[m].max()),
        )
    # signed rows of linked pairs: ids after the candidates of the coil-coil rows
    k_cc = nonlinear.index(cc)
    signed = {}
    for pair in true["linked"]:
        sid = offsets[k_cc] + cc._num_candidates + cc._pairs.index(tuple(pair))
        hit = ids == sid
        signed[str(tuple(pair))] = float(y[hit][0]) if np.any(hit) else None
    entry = dict(
        outer=h["outer"],
        inner_iterations=h["inner_iterations"],
        optimality=h["optimality"],
        gtolk=h["gtolk"],
        inner_converged=h["inner_converged"],
        constr_violation=h["constr_violation"],
        ctolk=h["ctolk"],
        groups=groups,
        true=dict(true, linked=[list(q) for q in true["linked"]]),
        signed_y=signed,
    )
    outer_log.append(entry)
    json.dump(outer_log, open(f"{a.tag}_outer.json", "w"), indent=1)
    print(
        f"outer {h['outer']}: {h['inner_iterations']} inner its, opt "
        f"{h['optimality']:.2e} (gtolk {h['gtolk']:.1e}, "
        f"{'met' if h['inner_converged'] else 'NOT met'}), violation "
        f"{h['constr_violation']:.2e} (ctolk {h['ctolk']:.1e})",
        flush=True,
    )
    for name, g in groups.items():
        print(
            f"   {name[:26]:26s} rows {g['rows']:6d}  mu {g['mu'][0]:.1e}/"
            f"{g['mu'][1]:.1e}/{g['mu'][2]:.1e}  max|y| {g['max_y']:.2e} "
            f"(nonzero {g['nonzero_y']})  max viol {g['max_violation']:.2e}",
            flush=True,
        )
    print(
        f"   true: cc {true['cc']:.5f} m, pc {true['pc']:.5f} m, max length "
        f"{max(true['length']):.4f}, max |k| {max(true['curvature']):.3f}, linked "
        f"{true['linked']}, signed-row y {signed}",
        flush=True,
    )


def save_x(x):
    np.save(f"{a.tag}_x.npy", np.asarray(x))
    return False


options = {
    "outer_callback": report,
    "callback": save_x,
    "track_steps": True,
    **json.loads(a.options),
}
(opt,), res = Optimizer("lsq-auglag-composite").optimize(
    c0,
    objective=obj,
    constraints=cons,
    maxiter=a.maxiter,
    verbose=1,
    ftol=0,
    xtol=0,
    gtol=a.gtol,
    ctol=a.ctol,
    options=options,
    copy=True,
)
opt.save(f"{a.tag}.h5")
np.save(f"{a.tag}_x.npy", np.asarray(res["allx"][-1]))
print("done", a.tag, res["message"])
