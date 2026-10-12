"""Coil conversion check: folded (JAX) coils -> DESC PiecewisePlanarArcCoil, field compared at the axis."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; boot.setup(default="cpu")
sys.path.insert(0, HERE)
import numpy as np, jax.numpy as jnp
import folded
folded.NPT = 1280
from folded import Layout, coil_arcs, full_coilset, biot_savart
from nae import NearAxis
from desc.coils import CoilSet, MixedCoilSet, PiecewisePlanarArcCoil
from desc.grid import LinearGrid
import common as S

run = sys.argv[1]
Mfit = int(sys.argv[2]) if len(sys.argv) > 2 else None
z = np.load(run + ".npz"); a = json.load(open(run + ".json"))["args"]
L = Layout(int(z["nfp"]), int(z["nc"]), int(z["K"]), int(z["Kax"]), str(z["mode"]))
rc, zs, eta, C = L.split(jnp.asarray(z["p"])[:L.n])
q = NearAxis(L.nfp, 41).solve(rc, zs, eta, float(z["hel"]), iota0=a["iota"])
pts_ax = np.asarray(q["x"])[::4]
X, I = full_coilset(C, L)
Bm = np.array([np.asarray(biot_savart(jnp.asarray(p), X, I)) for p in pts_ax])
from convert import to_desc_coils
cs, arcs = to_desc_coils(C, L)
for name, sg in [("LinearGrid(N=400)", LinearGrid(N=400)), ("source_grid (GL per arc)", S.source_grid(cs)[0])]:
    Bd = np.asarray(cs.compute_magnetic_field(pts_ax, basis="xyz", source_grid=sg)) * 1e-7 / 1e-7
    print(f"{name:26s} max |B_desc - B_mine| / |B| = {np.max(np.linalg.norm(Bd - Bm, axis=1) / np.linalg.norm(Bm, axis=1)):.2e}")
# geometry: DESC arcs vs my curve
g = np.asarray(cs.coils[0].coils[0].compute("x", grid=LinearGrid(N=2000), basis="xyz")["x"]) if hasattr(cs.coils[0], "coils") else None
from scipy.spatial import cKDTree
for i, arc in enumerate(arcs):
    (xu, _, _), (xl, _, _), _ = coil_arcs(C[i], L)
    mine = np.concatenate([np.asarray(xu), np.asarray(xl)])
    dsc = np.asarray(arc.compute("x", grid=LinearGrid(N=4000), basis="xyz")["x"])
    d1 = cKDTree(mine).query(dsc)[0].max(); d2 = cKDTree(dsc).query(mine)[0].max()
    print(f"coil {i}: max distance DESC arc <-> mine {max(d1, d2) * 1e3:.3f} mm; DESC hinge |z| {np.abs(np.asarray(arc.hinges).reshape(-1, 3)[:, 2]).max():.1e}")
