"""Print the --hinge-s string that cuts each unique coil on the horizontal plane z = Z.

    python level_hinges.py COILSET.h5 Z

Each coil crosses a horizontal plane exactly twice, so this is the B=2 hinge placement for
a horizontal joint chord at height Z. Circles are planar, so cutting a circular start this
way is exact whatever Z is (the polar-arc fit deviation stays at 3.5 mm).
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, "sandbox")
os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.grid import Grid
from desc.io import load

cs, Z = load(sys.argv[1]), float(sys.argv[2])
N = 200001
groups = []
for c in S.iter_unique(cs):
    s = np.linspace(0, 2 * np.pi, N)
    x = np.asarray(c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1),
                                            spacing=np.ones((N, 3)), sort=False,
                                            jitable=False), basis="xyz")["x"])
    f = x[:, 2] - Z
    k = np.where(f[:-1] * f[1:] < 0)[0]
    if len(k) != 2:
        raise SystemExit(f"coil crosses z={Z} {len(k)} times, expected 2 "
                         f"(its z range is {x[:,2].min():.2f}..{x[:,2].max():.2f} m)")
    groups.append([s[j] - f[j] * (s[j + 1] - s[j]) / (f[j + 1] - f[j]) for j in k])
print(";".join(",".join(f"{v:.6f}" for v in g) for g in groups))
