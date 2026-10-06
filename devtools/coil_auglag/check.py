"""Feasibility and topology report for coilsets, independent of the objectives' rows.

usage: python check.py KW_JSON FILE.h5|X_x.npy|init [...]
Closest coil-coil distance by brute force on fine polylines and plasma-coil distance
from their vertices to the exact boundary (stellarator reflections included), linked coil pairs, lengths and curvature. For
node-mode coil-coil rows, also the gap the coils need at that state against the gap used.
"""

import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402

from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    CoilSetDistanceRows,
    CoilSetMinDistance,
    PlasmaCoilDistanceField,
    PlasmaCoilSetDistanceRows,
    PlasmaCoilSetMinDistance,
)
from desc.objectives._coils import (  # noqa: E402
    _closed_polyline,
    _coil_gap_data,
    _lower_bound_meters,
    _node_gap,
    _segment_segment_distance,
    _surface_distance,
    _unique_neighbour_pairs,
)


def distance_objectives(obj, cons):
    """The built coil-coil and plasma-coil distance objectives of a problem."""
    found = {}
    for o in list(obj.objectives) + list(cons):
        if isinstance(o, (CoilSetDistanceRows, CoilSetMinDistance)):
            found["cc"] = o
        if isinstance(
            o,
            (
                PlasmaCoilDistanceField,
                PlasmaCoilSetDistanceRows,
                PlasmaCoilSetMinDistance,
            ),
        ):
            found["pc"] = o
    for o in found.values():
        if not o.built:
            o.build(verbose=0)
    return found["cc"], found["pc"]


def surface_distance(eq, X):
    """Smallest distance from points X to the exact plasma boundary (Newton)."""
    from scipy.spatial import cKDTree

    surf = getattr(eq, "surface", eq)
    cloud = LinearGrid(M=max(64, 8 * surf.M), N=max(64, 8 * surf.N * surf.NFP), NFP=1)
    cloud = Grid(cloud.nodes, sort=False)
    pts = np.asarray(surf.compute(["x"], grid=cloud, basis="xyz")["x"])
    return float(_surface_distance(surf, X, cloud.nodes[:, 1:], cKDTree(pts)).min())


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
    dpc = surface_distance(pc._eq, x[: len(cs) * (int(cs.sym) + 1)].reshape(-1, 3))
    g = LinearGrid(N=200)
    L = [float(c.compute("length", grid=g)["length"]) for c in cs]
    kmax = [float(c.compute("|curvature|", grid=g)["|curvature|"].max()) for c in cs]
    linked = cc.linked_pairs(params) if hasattr(cc, "linked_pairs") else "n/a"
    gap = None
    if getattr(cc, "_distance", None) == "node":
        h, k = _coil_gap_data(cs, cc._constants["grid"].nodes[:, 2])
        lo = _lower_bound_meters(cc)
        need = cc._gap_backoff * _node_gap(h.max(), h.max(), k.max(), k.max(), lo)
        gap = dict(used=float(np.max(cc._constants["gap"])), needed=float(need))
    return dict(cc=dcc, pc=dpc, linked=linked, length=L, curvature=kmax, gap=gap)


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
    init = cs.copy()  # feasibility() overwrites cs
    for f in sys.argv[2:]:
        r = feasibility(cs, cc, pc, load_params(obj, f, init))
        print(
            f"{f}: min coil-coil {r['cc']:.5f} m, min plasma-coil {r['pc']:.5f} m, "
            f"linked {r['linked']}"
        )
        print(
            f"   lengths {np.round(r['length'], 4)}, "
            f"max |curvature| {np.round(r['curvature'], 3)}"
        )
        if r["gap"]:
            used, need = r["gap"]["used"], r["gap"]["needed"]
            short = used < 0.98 * need
            print(
                f"   coil-coil gap used {1e3 * used:.3f} mm, needed here "
                f"{1e3 * need:.3f} mm"
                + (" ** SHORT by more than 2% **" if short else "")
            )
