"""Feasibility and topology report for coilsets, independent of the objectives' rows.

usage: python check.py KW_JSON FILE.h5|X_x.npy|init [...]
Closest coil-coil and plasma-coil distances by brute force on fine polylines
(stellarator reflections included), linked coil pairs, lengths and curvature.
"""

import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402

from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import (  # noqa: E402
    _closed_polyline,
    _point_segment_distance,
    _segment_segment_distance,
    _unique_neighbour_pairs,
)

kw = json.loads(sys.argv[1])
_, cs, obj, _ = build(**kw)
obj.build(verbose=0)
cc, pc = obj.objectives[1], obj.objectives[2]
fine = LinearGrid(N=1000)
for f in sys.argv[2:]:
    if f.endswith("_x.npy"):
        p = obj.unpack_state(jnp.asarray(np.load(f)), False)[0]
        p = [{k: np.asarray(v) for k, v in q.items()} for q in p]
    else:
        p = load(f).params_dict if f != "init" else cs.params_dict
    cs.params_dict = p
    x = np.asarray(cs._compute_position(grid=fine, basis="xyz"))
    pairs = _unique_neighbour_pairs(
        x.mean(axis=1), np.arange(len(cs)), x.shape[0] - 1, cs.NFP, cs.sym
    )
    poly = [_closed_polyline(jnp.asarray(c)) for c in x]
    dcc = min(
        float(np.asarray(_segment_segment_distance(*poly[a], *poly[b])).min())
        for a, b in pairs
    )
    Q = jnp.asarray(pc.constants["plasma_coords"])
    dpc = min(
        float(
            np.asarray(
                _point_segment_distance(Q[:, None], *[v[None] for v in poly[k]])
            ).min()
        )
        for k in range(len(cs) * (int(cs.sym) + 1))
    )
    g = LinearGrid(N=200)
    L = [float(c.compute("length", grid=g)["length"]) for c in cs]
    kmax = [float(c.compute("|curvature|", grid=g)["|curvature|"].max()) for c in cs]
    linked = cc.linked_pairs(p) if hasattr(cc, "linked_pairs") else "n/a"
    print(
        f"{f}: min coil-coil {dcc:.5f} m, min plasma-coil {dpc:.5f} m, linked {linked}"
    )
    print(f"   lengths {np.round(L, 4)}, max |curvature| {np.round(kmax, 3)}")
