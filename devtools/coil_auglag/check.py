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
from desc.objectives import (  # noqa: E402
    CoilSetDistanceRows,
    CoilSetMinDistance,
    PlasmaCoilSetDistanceRows,
    PlasmaCoilSetMinDistance,
)
from desc.objectives._coils import (  # noqa: E402
    _closed_polyline,
    _point_segment_distance,
    _segment_segment_distance,
    _unique_neighbour_pairs,
)


def distance_objectives(obj, cons):
    """The built coil-coil and plasma-coil distance objectives of a problem."""
    found = {}
    for o in list(obj.objectives) + list(cons):
        if isinstance(o, (CoilSetDistanceRows, CoilSetMinDistance)):
            found["cc"] = o
        if isinstance(o, (PlasmaCoilSetDistanceRows, PlasmaCoilSetMinDistance)):
            found["pc"] = o
    for o in found.values():
        if not o.built:
            o.build(verbose=0)
    return found["cc"], found["pc"]


def feasibility(cs, cc, pc, params):
    """Brute-force distances, linked pairs, lengths and curvature at coil params."""
    cs.params_dict = params
    x = np.asarray(cs._compute_position(grid=LinearGrid(N=1000), basis="xyz"))
    pairs = _unique_neighbour_pairs(
        x.mean(axis=1), np.arange(len(cs)), x.shape[0] - 1, cs.NFP, cs.sym
    )
    poly = [_closed_polyline(jnp.asarray(c)) for c in x]
    dcc = min(
        float(np.asarray(_segment_segment_distance(*poly[a], *poly[b])).min())
        for a, b in pairs
    )
    Q = jnp.asarray(pc._constants["plasma_coords"])
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
    linked = cc.linked_pairs(params) if hasattr(cc, "linked_pairs") else "n/a"
    return dict(cc=dcc, pc=dpc, linked=linked, length=L, curvature=kmax)


def load_params(obj, f, cs):
    """Coil params from a full state file, a saved coilset, or the initial coils."""
    if f.endswith("_x.npy"):
        p = obj.unpack_state(jnp.asarray(np.load(f)), False)[0]
        return [{k: np.asarray(v) for k, v in q.items()} for q in p]
    return load(f).params_dict if f != "init" else cs.params_dict


if __name__ == "__main__":
    kw = json.loads(sys.argv[1])
    _, cs, obj, cons = build(**kw)
    obj.build(verbose=0)
    cc, pc = distance_objectives(obj, cons)
    for f in sys.argv[2:]:
        r = feasibility(cs, cc, pc, load_params(obj, f, cs))
        print(
            f"{f}: min coil-coil {r['cc']:.5f} m, min plasma-coil {r['pc']:.5f} m, "
            f"linked {r['linked']}"
        )
        print(
            f"   lengths {np.round(r['length'], 4)}, "
            f"max |curvature| {np.round(r['curvature'], 3)}"
        )
