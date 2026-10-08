"""Is the spline coilset's field converged, and comparable with the arc coilset's?

Spline coils use a DIFFERENT Biot-Savart path from every other coil class:
`SplineXYZCoil._compute_A_or_B` joins the source-grid points with STRAIGHT segments and
uses Hanson-Hirshman (`coils.py: biot_savart_hh`), converging ~quadratically, while arc
coils use direct quadrature with the curve's own tangents on a composite-GL grid,
converging spectrally. So a bn difference between the two representations is only
meaningful once the polyline is converged well below it.

Two tests:

  A. SAME GEOMETRY, BOTH PATHS. Convert the arc optimum at 256 knots (0.005 mm from the
     source), then evaluate bn for the arc coilset on GL grids and for the spline
     coilset on uniform grids. They must agree in the limit; the gap at a given N is the
     discretization error we would otherwise mistake for a representation difference.

  B. THE ACTUAL RESULT. bn of the stage-2 spline result against its own grid refinement,
     to see whether the reported number is converged at the 801 points used in the run.

    python work/helios/spline/bs_convergence.py [RESULT.h5]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import CoilSet, MixedCoilSet, SplineXYZCoil  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402

ARC = "work/helios/polarB2_M5_cut30_final.h5"
RESULT = sys.argv[1] if len(sys.argv) > 1 else "work/helios/spline/stage2_n32/result.h5"

eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
arc = load(ARC)


def sg(s):
    return Grid(np.stack([0 * s, 0 * s, np.mod(s, 2 * np.pi)], 1), sort=False, jitable=False)


def to_spline(cs, n, method="cubic2"):
    coils, k = [], np.linspace(0, 2 * np.pi, n, endpoint=False)
    for c in S.iter_unique(cs):
        B = int(c.B)
        x = np.asarray(c.compute("x", grid=sg(k), basis="xyz")["x"])
        coils.append(SplineXYZCoil.from_values(float(c.current), x, knots=k, method=method,
                                               basis="xyz", break_indices=np.arange(B) * (n // B)))
    return MixedCoilSet(*[CoilSet(c, NFP=eq.NFP, sym=eq.sym) for c in coils])


def bn(cs, grid):
    return S.bn_stats(cs, eq, grid, vacuum=False)["mean"]


print(f"A. same geometry, both Biot-Savart paths  ({ARC} -> 256-knot spline)\n")
spl = to_spline(arc, 256)
print("   arc coilset, composite-GL quadrature (its native path):")
ref = None
for m in (8, 16, 24, 48, 96):
    v = bn(arc, S.gl_grid(S.arc_breaks(list(S.iter_unique(arc))[0]), m))
    print(f"     GL m={m:3d}/arc ({2 * m:5d} nodes)   bn {v:.8e}" +
          ("" if ref is None else f"   d(prev) {v - ref:+.2e}"))
    ref = v
print("\n   spline coilset, Hanson-Hirshman polyline:")
for N in (100, 200, 400, 800, 1600, 3200):
    v = bn(spl, LinearGrid(N=N))
    print(f"     LinearGrid(N={N:4d}) ({2 * N + 1:5d} pts)  bn {v:.8e}   vs arc GL {v - ref:+.2e}"
          f"  ({100 * (v - ref) / ref:+.3f}%)")

if os.path.exists(RESULT):
    print(f"\nB. the stage-2 spline result, grid refinement  ({RESULT})\n")
    res = load(RESULT)
    prev = None
    for N in (400, 800, 1600, 3200, 6400):
        v = bn(res, LinearGrid(N=N))
        d = "" if prev is None else f"   d(prev) {v - prev:+.2e}"
        star = "   <-- the grid the run used" if N == 400 else ""
        print(f"     LinearGrid(N={N:4d}) ({2 * N + 1:5d} pts)  bn {v:.8e}{d}{star}")
        prev = v
else:
    print(f"\nB. skipped: {RESULT} not written yet")
