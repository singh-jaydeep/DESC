"""Sort final polar arcB2 coils into the known per-coil layouts (by coil shape distance).

    python work/helios/scan_polar/classify.py [LABEL ...]      (default: every cut*/ folder with a final)

Coils are clustered per coil index by their symmetric max nearest-point distance (a coil joins the first
cluster whose representative is within TOL). Named layouts: coil 0 A (cut60) A0 (cut0) C (cut120) D (cut150);
coil 1 X (cut30) Y (cut60); coil 2 P (cut60) Q (cut90). Others get c<i>new<k> with the run that found them.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import common as S  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

P = "work/helios/scan_polar/"
TOL = 0.30  # m
G = LinearGrid(N=400)


def final(d):
    for leg in ("legC", "legB"):
        f = f"{P}{d}/{leg}/result.h5"
        if os.path.exists(f):
            return f
    return None


def xyz(c):
    return np.asarray(c.compute("x", grid=G, basis="xyz")["x"])


def dist(a, b):
    return max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max())


labels = sys.argv[1:] or sorted((os.path.basename(d)[3:] for d in glob.glob(P + "cut*") if final(os.path.basename(d))),
                                 key=lambda l: (not l.isdigit(), int(l) if l.isdigit() else 0, l))
NAMES = {0: {"cut60": "A", "cut0": "A0", "cut120": "C", "cut150": "D"},
         1: {"cut30": "X", "cut60": "Y"}, 2: {"cut60": "P", "cut90": "Q"}}
runs = []
for lab in labels:
    f = final("cut" + lab)
    if f is None:
        print(f"{lab:22s} no final yet")
        continue
    runs.append((lab, f, [xyz(c) for c in S.iter_unique(load(f))]))

# greedy clustering per coil: a coil joins the first cluster whose representative is within TOL,
# named layouts (NAMES) are seeded first so their names stay fixed
tags = {lab: [None] * 3 for lab, _, _ in runs}
for i in range(3):
    clusters = []  # (name, representative xyz)
    order = sorted(runs, key=lambda r: ("cut" + r[0]) not in NAMES[i])
    n_new = 0
    for lab, f, xs in order:
        ds = [(dist(xs[i], rep), name) for name, rep in clusters]
        d, name = min(ds) if ds else (np.inf, None)
        if d < TOL:
            tags[lab][i] = f"{name}({d * 100:.0f}cm)"
        else:
            name = NAMES[i].get("cut" + lab)
            if name is None:
                n_new += 1
                name = f"c{i}new{n_new}"
            clusters.append((name, xs[i]))
            near = f", {d:.1f} m from {min(ds)[1]}" if ds else ""
            tags[lab][i] = name + ("" if "new" not in name else f"[{lab}{near}]")

for lab, f, xs in runs:
    r = json.load(open(os.path.splitext(f)[0] + ".json"))
    m = r["result"]
    worst_L = max(c["L"] for c in m["per_coil"]) - r["bounds"]["L"]
    print(f"{lab:22s} bn {m['bn']:.4e}  L+{worst_L * 1e6:.0f}um  convex {m['convex'] * 1e3:.2f}mrad  "
          f"d_cc {m['d_cc'] / r['bounds']['dcc']:.3f}  d_pc {m['d_pc'] / r['bounds']['dpc']:.3f}  "
          + "  ".join(f"c{i} {t}" for i, t in enumerate(tags[lab])))
