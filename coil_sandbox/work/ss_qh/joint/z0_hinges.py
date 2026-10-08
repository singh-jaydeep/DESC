"""Hinge parameters at each unique coil's z = 0 crossings, for make_start.py --hinge-s.

    python work/ss_qh/joint/z0_hinges.py COILS.h5        (from the sandbox root, CPU)

Prints one line per unique coil (crossings, their R and phi, and the coil's extent along the
chord between them, which a transverse arc cannot exceed) and finally the --hinge-s string.
A B=2 joint plane at z = 0 needs exactly two crossings per coil.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.io import load
from desc.grid import LinearGrid

cs = load(sys.argv[1])
groups = []
for i, c in enumerate(S.iter_unique(cs)):
    g = LinearGrid(N=4000)
    s = np.asarray(g.nodes[:, 2])
    x = np.asarray(c.compute("x", grid=g, basis="xyz")["x"])
    z = x[:, 2]
    k = np.where(np.sign(z) != np.sign(np.roll(z, -1)))[0]
    if len(k) != 2:
        print(f"coil {i}: {len(k)} crossings of z = 0 -- not a two-piece joint plane")
    sc = []
    for j in k:
        j1 = (j + 1) % len(s)
        s1 = s[j1] + (2 * np.pi if j1 == 0 else 0)
        sc.append(float(np.mod(s[j] + (s1 - s[j]) * z[j] / (z[j] - z[j1]), 2 * np.pi)))
    P = np.array([x[j] for j in k])
    R, phi = np.hypot(P[:, 0], P[:, 1]), np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    e = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
    t = (x - P[0]) @ e  # coordinate along the chord; a transverse arc stays within [0, |chord|]
    over = max(-t.min(), t.max() - np.linalg.norm(P[1] - P[0]))
    print(f"coil {i}: s {np.round(sc, 4)}  R {np.round(R, 3)}  phi {np.round(phi, 1)}  chord {np.linalg.norm(P[1]-P[0]):.3f} m"
          f"  coil overhangs the chord ends by {over * 1e3:.1f} mm  (z range [{z.min():.3f}, {z.max():.3f}])")
    groups.append(",".join(f"{v:.6f}" for v in sorted(sc)))
print("--hinge-s '" + ";".join(groups) + "'")
