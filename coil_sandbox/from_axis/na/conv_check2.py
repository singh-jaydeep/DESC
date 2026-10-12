import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; boot.setup(default="cpu")
sys.path.insert(0, HERE)
import numpy as np, jax.numpy as jnp
import folded
folded.NPT = 400
from folded import Layout, coil_arcs, full_coilset, biot_savart
from nae import NearAxis
from desc.grid import LinearGrid
import common as S
from convert import to_desc_coils
run = sys.argv[1]
z = np.load(run + ".npz"); a = json.load(open(run + ".json"))["args"]
L = Layout(int(z["nfp"]), int(z["nc"]), int(z["K"]), int(z["Kax"]), str(z["mode"]))
rc, zs, eta, C = L.split(jnp.asarray(z["p"])[:L.n])
cs, arcs = to_desc_coils(C, L)
t = np.linspace(0, 1, 401)
for i, arc in enumerate(arcs):
    (xu, _, _), (xl, _, _), _ = coil_arcs(C[i], L)
    s = np.concatenate([t[:-1] * np.pi, np.pi + t[:-1] * np.pi])  # arc 0: s in [0, pi), arc 1: [pi, 2pi)
    g = LinearGrid(x=s) if False else None
    from desc.grid import Grid
    gr = Grid(np.stack([np.zeros_like(s), np.zeros_like(s), s], 1))
    xd = np.asarray(arc.compute("x", grid=gr, basis="xyz")["x"])
    mine = np.concatenate([np.asarray(xu)[:-1], np.asarray(xl)[:-1]])
    print(f"coil {i}: pointwise max |x_desc - x_mine| = {np.abs(xd - mine).max():.2e} m")
q = NearAxis(L.nfp, 41).solve(rc, zs, eta, float(z["hel"]), iota0=a["iota"])
pts = np.asarray(q["x"])[::4]
X, I = full_coilset(C, L)
Bm = np.array([np.asarray(biot_savart(jnp.asarray(p), X, I)) for p in pts])
for gl in [24, 64]:
    sg = S.source_grid(cs, gl_m=gl)[0]
    Bd = np.asarray(cs.compute_magnetic_field(pts, basis="xyz", source_grid=sg))
    print(f"GL {gl}: max |dB|/|B| {np.max(np.linalg.norm(Bd - Bm, axis=1) / np.linalg.norm(Bm, axis=1)):.2e}")
print("n coils desc", len(cs.compute_magnetic_field.__self__.coils) if False else sum(len(c) for c in cs.coils), " mine", X.shape[0])
