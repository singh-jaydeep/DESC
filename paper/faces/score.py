"""Score of one faces-sweep coilset: the one function for every row of the table.

    python paper/faces/score.py COILS.h5 precise_QA OUT.json

The dense check (paper/lib/dense_check.evaluate, vacuum, fixed boundary) against
paper/inputs/bounds/rr02_<cfg>.json, plus B.n on the same grid: |B.n| on one field
period, 72 (phi) x 96 (theta), unweighted, coil field only; bn_mean = mean|B.n| /
mean|B|, bn_max = max|B.n| / mean|B|, bn_rms = sqrt(mean (B.n)^2) / mean|B|.
dof_per_coil counts one coil's optimizable parameters without rotmat, shift and
arc_ref, current included.
Arc coils also get fold_deg_max (ArcFold's plane angle) and corner_deg, the largest
turning angle between the arc tangents at a hinge (RR-02's corner measure,
`turning_angles`).
Ported from coil_sandbox/from_axis/rr02/score.py (js/qa-poster).
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
)
import device  # noqa: E402

device.setup(default="cpu")

import dense_check as D  # noqa: E402, I001
import numpy as np  # noqa: E402

from desc.coils import coil_source_grid  # noqa: E402
from desc.io import load  # noqa: E402

path, cfg, out = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
bounds = os.path.join(here, "..", "inputs", "bounds", f"rr02_{cfg}.json")
eq, cs = D.load_equilibrium(cfg), load(path)
leaves = D.leaf_coils(cs)
P = D.load_bounds(bounds, eq, len(leaves))
m = D.evaluate(cs, eq, vacuum=True)
_, x, n, Bp = D._BN[(id(eq), True)]
B = np.asarray(
    cs.compute_magnetic_field(x, basis="rpz", source_grid=coil_source_grid(cs))
)
Bn = np.abs(np.sum((B + Bp) * n, axis=1))
mB = float(np.mean(np.linalg.norm(B + Bp, axis=1)))
mean, mx, rms = Bn.mean() / mB, Bn.max() / mB, np.sqrt(np.mean(Bn**2)) / mB
assert abs(mean / m["bn"] - 1) < 1e-12 and abs(mx / m["bn_max"] - 1) < 1e-12
c0 = leaves[0]
skip = ("rotmat", "shift", "arc_ref")
dof = int(
    sum(
        np.size(v)
        for k, v in c0.params_dict.items()
        if k in c0.optimizable_params and k not in skip
    )
)
mg = D.margins(m, P)
rec = dict(
    coils=path,
    config=cfg,
    coil_type=type(c0).__name__,
    n_unique=len(leaves),
    n_coils_total=m["n_coils"],
    dof_per_coil=dof,
    bn_mean=float(mean),
    bn_rms=float(rms),
    bn_max=float(mx),
    L_max_m=m["L"],
    L_total_unique_m=float(sum(m["L_per_coil"])),
    L_over_bound=mg.get("L"),
    kappa_over_bound=mg.get("kappa"),
    dcc_over_bound=mg.get("dcc"),
    dpc_over_bound=mg.get("dpc"),
    d_cc_m=m["dcc"],
    d_pc_m=m["dpc"],
    linked=m["linked"],
    verdict=D.verdict(m, P),
    feasible=D.verdict(m, P) == "FEASIBLE",
    bounds=P,
)
if hasattr(c0, "B"):
    from desc.objectives._piecewise_planar import fold_angles, turning_angles

    rec["fold_deg_max"] = float(np.max(fold_angles(cs)))
    # RR-02's corner measure: largest turning angle between arc tangents at a hinge
    rec["corner_deg"] = float(np.max(turning_angles(cs)))
    rec["corner_deg_per_coil"] = turning_angles(cs).tolist()
json.dump(rec, open(out, "w"), indent=1, default=str)
print(
    f"SCORE {path}: bn mean {mean:.4e} rms {rms:.4e} max {mx:.4e}  {D.report(m, P)}"
    f"  linked {m['linked']} -> {rec['verdict']}  dof/coil {dof}",
    flush=True,
)
