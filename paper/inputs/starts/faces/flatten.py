"""RR-02: flatten make_start.py's arc output to ONE CoilSet(NFP, sym) of leaf coils, and check it is the same coil set.

    python from_axis/rr02/flatten.py IN_mixed.h5 OUT.h5 --eq precise_QA      (cwd coil_sandbox, CPU)

make_start.py writes arcs as a MixedCoilSet of one-coil symmetric CoilSets; CoilSetDistanceRows indexes the
independent coils of a flat CoilSet and crashes on that container (planar_cmp/pqa/hinged_D_crash_mixedstart).
This is a container change only: the leaf coils (and their frozen arc_ref / rotmat / shift) are carried over
unchanged. Checks: every physical coil's positions on a fine grid, and the field at the plasma boundary, match the
input to round-off; the flat set has the expected physical coil count.
"""
import sys

sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import CoilSet  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--eq", required=True)
ap.add_argument("--device", default=None)
a = ap.parse_args()

eq, _ = S.load_equilibrium(a.eq)
mixed = load(a.src)
leaves = list(S.iter_unique(mixed))
inner = [c for c in mixed]
nfp = {int(getattr(c, "NFP", 1)) for c in inner}
sym = {bool(getattr(c, "sym", False)) for c in inner}
if len(nfp) != 1 or len(sym) != 1:
    raise SystemExit(f"inner coil sets disagree on NFP/sym: {nfp} {sym}")
flat = CoilSet(*leaves, NFP=nfp.pop(), sym=sym.pop())

g = LinearGrid(N=400)
p0 = np.asarray(mixed._compute_position(grid=g))  # (physical coils, nodes, 3); the two containers order the
p1 = np.asarray(flat._compute_position(grid=g))   # symmetry images differently, so match coils as a set
dx = (max(min(float(np.abs(q - p).max()) for q in p0) for p in p1) if p0.shape == p1.shape else np.inf)
p1 = p1.reshape(-1, 3)
xb = np.asarray(eq.compute("x", grid=LinearGrid(M=12, N=12, NFP=eq.NFP))["x"])
src = S.source_grid(flat)[0]
B0 = np.asarray(mixed.compute_magnetic_field(xb, basis="rpz", source_grid=src))
B1 = np.asarray(flat.compute_magnetic_field(xb, basis="rpz", source_grid=src))
dB = float(np.abs(B0 - B1).max() / np.abs(B0).max())
exp = S.expected_coils(flat, eq)
print(f"flatten {a.src} -> {a.out}: {len(leaves)} leaf coils ({type(leaves[0]).__name__}), NFP {flat.NFP} sym {flat.sym}, "
      f"{p1.shape[0] // g.num_nodes} physical (expected {exp}); max |dx| {dx:.2e} m, max |dB|/|B| {dB:.2e}")
if not (dx < 1e-12 and dB < 1e-12 and p1.shape[0] // g.num_nodes == exp):
    raise SystemExit("FLATTEN CHECK FAILED")
flat.save(a.out)
print(f"wrote {a.out}")
