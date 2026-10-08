"""Repack a MixedCoilSet of per-coil symmetric CoilSets (NFP, sym) into one CoilSet(NFP, sym).

The new distance rows (CoilSetDistanceRows / PlasmaCoilDistanceField, main repo) read NFP and sym from
the top-level coilset only, so the nested form gives them 4 coils instead of 32. Same coils, same geometry.
usage: python repack_arcs.py IN.h5 OUT.h5 [EQ]   (EQ for the B.n check; default the fixed QH single-stage eq)
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.coils import CoilSet
from desc.grid import LinearGrid
from desc.io import load

src, dst = sys.argv[1:3]
eq_spec = sys.argv[3] if len(sys.argv) > 3 else "work/ss_qh/arcB2/ss_k4/eq_k4.h5"
mix = load(src)
nfp, sym = {(m.NFP, m.sym) for m in mix}.pop()
assert all(len(m) == 1 and (m.NFP, m.sym) == (nfp, sym) for m in mix), "members must be 1-coil CoilSets with one symmetry"
cs = CoilSet(*[m[0].copy() for m in mix], NFP=nfp, sym=sym, check_intersection=False)
g = LinearGrid(N=200)
# the two forms list the 32 physical coils in different orders: match each curve to its nearest counterpart
A, B = np.asarray(mix._compute_position(grid=g)), np.asarray(cs._compute_position(grid=g))
D = np.array([[np.abs(p - q).max() for q in B] for p in A])
assert len(set(D.argmin(1))) == len(A), "curves do not match one to one"
d = D.min(1).max()
eq, _ = S.load_equilibrium(eq_spec)
bn = [S.bn_stats(c, eq, S.source_grid(c)[0], vacuum=True)["mean"] for c in (mix, cs)]
print(f"{type(mix).__name__} -> {type(cs).__name__}(NFP={cs.NFP}, sym={cs.sym}, {len(cs)} coils); "
      f"worst matched curve difference {d:.1e} m (order differs); bn {bn[0]:.10e} -> {bn[1]:.10e}")
cs.save(dst)
