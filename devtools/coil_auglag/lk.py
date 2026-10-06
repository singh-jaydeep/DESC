"""Linked coil pairs (|Lk| > 0.5) for unique coils vs all coils.

usage: python lk.py FILE.h5 [...]   or   python lk.py TAG ALLX.npy (every iterate)
"""

import sys

import numpy as np

from desc.coils import _linking_number
from desc.grid import LinearGrid
from desc.io import load


def linked(cs, params=None, N=50):
    """Return [(i, j, Lk)] for unique coil i, any coil j > i, with |Lk| > 0.5."""
    g = LinearGrid(N=N)
    dx = np.asarray(g.spacing[:, 2])
    x, xs = cs._compute_position(params=params, grid=g, dx1=True, basis="xyz")
    x, xs = np.asarray(x), np.asarray(xs)
    out = []
    for i in range(len(cs)):
        for j in range(i + 1, x.shape[0]):
            lk = _linking_number(x[i], x[j], xs[i].copy(), xs[j].copy(), dx, dx)
            lk = float(lk) / (4 * np.pi)
            if abs(lk) > 0.5:
                out.append((i, j, round(lk, 2)))
    return out


if __name__ == "__main__":
    if sys.argv[1].endswith(".h5"):
        for f in sys.argv[1:]:
            print(f, linked(load(f)))
    else:
        import json

        import jax.numpy as jnp
        from cases import build

        kw = json.load(open(f"{sys.argv[1]}.json"))
        _, cs, obj, _ = build(**kw)
        obj.build(verbose=0)
        for k, x in enumerate(np.atleast_2d(np.load(sys.argv[2]))):
            pd = obj.unpack_state(jnp.asarray(x), False)[0]
            print(k, linked(cs, [{a: np.asarray(b) for a, b in p.items()} for p in pd]))
