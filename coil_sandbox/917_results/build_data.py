"""Extract surface, B.n heatmap, coils and metrics for every coilset -> viewer_data.json.

B.n is the TOTAL normal field (coils + the plasma's own field by virtual casing, since
Helios is finite beta) on the full boundary, normalized by <|B|> of the total field --
the same `bn` the sandbox reports, so the numbers here match the run records.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import json  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import _independent_coil_indices  # noqa: E402

SETS = [
    ("planar_N7",              "Planar N=7",            "FourierPlanarCoil, the paper-style planar baseline"),
    ("arcB2_transverse_M5",    "Transverse arc B=2",    "PiecewisePlanarArcCoil, 2 arcs, M=5"),
    ("polarB2_M5_cut60",       "Polar arc B=2, cut 60",  "PolarPlanarArcCoil, 2 arcs, M=5"),
    ("polarB2_M5_cut30",       "Polar arc B=2, cut 30",  "PolarPlanarArcCoil, best B=2"),
    ("arcB3_M5",               "Arc B=3",               "PiecewisePlanarArcCoil, 3 arcs, M=5"),
    ("spline_n32_from_cut30",  "Spline (C0 breaks)",    "SplineXYZCoil, 32 knots, C0 at the 2 hinges, refined from cut 30"),
    ("polarB2_cut30_L37",      "Polar B=2, L<=37 m",    "cut 30 re-optimized with a 37 m length cap (different bound)"),
    ("fourierXYZ_N10_ceiling",  "FourierXYZ N=10 (ceiling)",
     "unrestricted 3D shapes, convexity dropped - what 12 coils at this standoff can do at all"),
    ("fourierXYZ_16coils",      "FourierXYZ, 16 coils",
     "4 unique instead of 3: unrestricted 3D, 16 coils x 36 m. More conductor as well as more coils"),
    ("planar_N7_16coils",       "Planar N=7, 16 coils",
     "FourierPlanarCoil, 4 unique, 16 x 36 m - the planar baseline at the higher coil count"),
    ("polarB2_M5_cut30_16coils", "Polar arc B=2, 16 coils",
     "PolarPlanarArcCoil B=2 M=5, 4 unique, fitted at cut 30 to the 16-coil planar optimum"),
]
NTHETA, NZETA = 96, 192     # boundary mesh for the heatmap
NC = 240                    # points per coil

eq, _ = S.load_equilibrium("917_results/helios_repro.h5")
from desc.integrals.singularities import compute_B_plasma  # noqa: E402

# Virtual casing needs eval_grid.NFP == source_grid.NFP, so it can only be evaluated on
# ONE field period. Everything is field-period periodic in rpz -- R, Z, the rpz normal
# and the rpz components of B_plasma all repeat -- so compute one period and replicate
# it round the torus, shifting only phi. The eval grid must also be at least as fine as
# the virtual-casing source grid, or its interpolator silently truncates.
src = S.vc_source_grid(eq)
NZP = max(NZETA // eq.NFP, src.num_zeta)
NTH = max(NTHETA, src.num_theta)
grid = LinearGrid(rho=np.array([1.0]), theta=NTH, zeta=NZP, NFP=eq.NFP, sym=False)
d = eq.compute(["R", "phi", "Z", "n_rho"], grid=grid)
R1, phi1, Z1 = (np.asarray(d[k]) for k in ("R", "phi", "Z"))
n1 = np.asarray(d["n_rho"])

print(f"virtual casing on one field period ({NZP} x {NTH}) ...", flush=True)
Bp1 = np.asarray(compute_B_plasma(eq, grid, src, chunk_size=64))

# replicate round the torus: phi -> phi + 2 pi k / NFP, rpz components unchanged
R = np.tile(R1, eq.NFP)
Z = np.tile(Z1, eq.NFP)
phi = np.concatenate([phi1 + 2 * np.pi * k / eq.NFP for k in range(eq.NFP)])
normal = np.tile(n1, (eq.NFP, 1))
Bp = np.tile(Bp1, (eq.NFP, 1))
rpz = np.stack([R, phi, Z], axis=1)
xyz = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1)
NZETA, NTHETA = NZP * eq.NFP, NTH     # the mesh the viewer will draw

out = {"ntheta": int(NTHETA), "nzeta": int(NZETA),
       "xyz": np.round(xyz, 4).ravel().tolist(),
       "B_axis": 6.056, "coilsets": []}

for key, label, note in SETS:
    path = f"917_results/{key}.h5"
    cs = load(path)
    src = S.source_grid(cs)[0]
    B = np.asarray(cs.compute_magnetic_field(rpz, basis="rpz", source_grid=src)) + Bp
    Bn = np.sum(B * normal, axis=1)
    Bmag = np.linalg.norm(B, axis=1)
    mB = float(np.mean(Bmag))
    s = np.linspace(0, 2 * np.pi, NC, endpoint=False)
    pos = np.asarray(cs._compute_position(
        grid=Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False), basis="xyz"))
    pos = np.concatenate([pos, pos[:, :1]], axis=1)          # close each loop
    uniq = [int(i) for i in _independent_coil_indices(cs)]
    nper = pos.shape[0] // len(uniq)                          # copies per unique coil
    owner = [i // nper for i in range(pos.shape[0])]
    m = S.evaluate(cs, eq, vacuum=False)
    rec = dict(key=key, label=label, note=note,
               bn_mean=float(np.mean(np.abs(Bn)) / mB), bn_max=float(np.max(np.abs(Bn)) / mB),
               bn_mean_T=float(np.mean(np.abs(Bn))), bn_max_T=float(np.max(np.abs(Bn))),
               n_coils=int(pos.shape[0]), n_unique=len(uniq), owner=owner,
               L_max=m["L"], kappa_max=m["kappa"], d_cc=m["d_cc"], d_pc=m["d_pc"],
               currents=[float(c.current) / 1e6 for c in S.iter_unique(cs)],
               bn=np.round(Bn / mB, 5).tolist(),
               coils=[np.round(p, 3).ravel().tolist() for p in pos])
    out["coilsets"].append(rec)
    print(f"  {label:<26} bn {rec['bn_mean']:.4e}  max {rec['bn_max']:.4e}  "
          f"<|B.n|>/B_axis {rec['bn_mean_T']/out['B_axis']*100:.3f}%  "
          f"({rec['n_coils']} coils, {rec['n_unique']} unique)", flush=True)

with open("917_results/viewer_data.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"\nwrote 917_results/viewer_data.json "
      f"({os.path.getsize('917_results/viewer_data.json')/2**20:.2f} MiB)")
