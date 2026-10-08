"""For each unique coil: WHICH coil is it closest to, and is that its own symmetry image?

The user's reading of the 3D viewer is that the subdominant coil is physically boxed in --
adjacent coils' hinges bend toward it -- and that a coil and its own mirror image can hinge
straight at each other, which would leave the boxed coil free to move only on the outboard
side. This measures that: for every physical coil, the nearest other physical coil, whether
it is the same unique coil (a self-image pair), and the gap.
"""
import sys, os
os.chdir("/home/singh/Documents/DESC2/coil_sandbox"); sys.path.insert(0, "sandbox")
os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np
from scipy.spatial import cKDTree
import common as S
from desc.grid import LinearGrid
from desc.io import load
from desc.objectives._coils import _independent_coil_indices

for path, lab in [(p, l) for l, p in [
        ("baseline cut30", "work/helios/arc16/cut30/legB/result.h5"),
        ("d_cc>=1.0",      "work/helios/arc16/sens_dcc10/result.h5"),
        ("L<=30",          "work/helios/arc16/sens_L30/result.h5"),
        ("cold (L36)",     "work/helios/arc16/cold_cut30/free/result.h5")] if os.path.exists(p)]:
    cs = load(path)
    pos = np.asarray(cs._compute_position(grid=LinearGrid(N=400)))
    n = pos.shape[0]
    uniq = [int(i) for i in _independent_coil_indices(cs)]
    nper = n // len(uniq)
    owner = [i // nper for i in range(n)]          # which unique coil each physical coil is
    I = [float(c.current) / 1e6 for c in S.iter_unique(cs)]
    trees = [cKDTree(p) for p in pos]
    print(f"\n=== {lab} ===  currents {[round(v,1) for v in I]}")
    # per unique coil, take its first physical copy
    for u in range(len(uniq)):
        i = owner.index(u)
        d = [(trees[j].query(pos[i])[0].min(), j) for j in range(n) if j != i]
        d.sort()
        gap, j = d[0]
        same = "SELF-IMAGE" if owner[j] == u else f"coil {owner[j]}"
        gap2, j2 = d[1]
        same2 = "self-image" if owner[j2] == u else f"coil {owner[j2]}"
        print(f"  coil {u} (I {I[u]:5.1f} MA): nearest {gap:.3f} m -> {same:<11} | "
              f"2nd {gap2:.3f} m -> {same2}")
