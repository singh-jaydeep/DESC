"""Which polar-arc modes actually carry amplitude, and WHERE the extra detail sits.

The user's question: raising M, are the additional modes useful, and are they useful
specifically where the coil runs closest to the plasma -- or does convexity kill them?

Each arc is r(phi) = |chord|/2 + sum_m a_m sin(m pi phi), m = 1..M. Every mode VANISHES
at phi = 0 and 1, i.e. at the hinges, so no mode can act at a corner at any M (this is
the structural point already in HANDOFF: a B=2 polar coil has no local degree of freedom
at a hinge). They can only act mid-arc.

Reports, per unique coil:
  * |a_m| for every mode, so the high-m tail is visible
  * the share of sum |a_m| carried by the modes ABOVE the old M
  * where the coil is closest to the plasma, as a phi within its arc, and the value of
    the new (m > oldM) part of r there vs its own RMS -- i.e. whether the new detail is
    concentrated at the close-approach point or spread evenly

    python work/helios/arc16/mode_use.py NEW.h5 OLD_M [OLD.h5]
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

new_path = sys.argv[1]
oldM = int(sys.argv[2])
old_path = sys.argv[3] if len(sys.argv) > 3 else None

cs = load(new_path)
eq = load("sample_equilibria/helios_repro.h5")
uniq = list(S.iter_unique(cs))
B, M = int(uniq[0].B), int(uniq[0].M)
print(f"{new_path}: B={B} M={M} (old M={oldM})\n")

# plasma surface points, for the closest-approach location
tree = S.surface_tree(eq)

old_uniq = list(S.iter_unique(load(old_path))) if old_path else [None] * len(uniq)

for i, (c, c0) in enumerate(zip(uniq, old_uniq)):
    a = np.asarray(c.shape).reshape(B, M)
    print(f"coil {i}:")
    for b in range(B):
        amp = np.abs(a[b])
        tail = amp[oldM:].sum() / max(1e-30, amp.sum())
        bars = " ".join(f"{v:.3f}" for v in amp)
        print(f"  arc {b}: |a_m| = {bars}")
        print(f"          modes > {oldM} carry {tail*100:5.1f}% of total |a|, "
              f"max new |a_m| = {amp[oldM:].max():.4f} m")
    # where is this coil closest to the plasma, and is the new detail there?
    ns = 2000
    s = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    from desc.grid import Grid
    x = np.asarray(c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False,
                                            jitable=False), basis="xyz")["x"])
    d = tree.query(x)[0]
    j = int(np.argmin(d))
    arc = int(s[j] // (2 * np.pi / B))
    phi = (s[j] - arc * 2 * np.pi / B) / (2 * np.pi / B)
    m = np.arange(1, M + 1)
    r_new = lambda p: float(np.sum(a[arc, oldM:] * np.sin(np.pi * m[oldM:] * p)))
    pg = np.linspace(0, 1, 401)
    prof = np.array([r_new(p) for p in pg])
    rms = float(np.sqrt(np.mean(prof ** 2)))
    print(f"  closest to plasma: {d.min():.3f} m at arc {arc}, phi {phi:.3f}")
    print(f"  new-mode displacement there {r_new(phi)*1e3:+7.1f} mm  "
          f"(profile RMS {rms*1e3:.1f} mm, peak {np.abs(prof).max()*1e3:.1f} mm)"
          + ("   <-- concentrated at close approach" if abs(r_new(phi)) > 1.5 * rms else ""))
    print()
