"""Spline distance field vs Newton-exact distance to precise_QH's boundary."""

import time

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.spatial import cKDTree

import desc.examples
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid

# curvature sign convention on a circular torus (R0=10, a=1): poloidal curvature 1/a
t = Equilibrium(M=2, N=0).surface.compute(
    ["curvature_k1_rho", "curvature_k2_rho"], grid=LinearGrid(M=4, N=0)
)
print(
    "circular torus a=1: k1",
    np.round(t["curvature_k1_rho"][:3], 3),
    "k2",
    np.round(t["curvature_k2_rho"][:3], 3),
)

eq = desc.examples.get("precise_QH")
surf = eq.surface
NFP = eq.NFP
g = LinearGrid(M=150, N=150, NFP=1, sym=False)
d = surf.compute(["x"], grid=g, basis="xyz")
tree = cKDTree(np.asarray(d["x"]))
TZ = g.nodes[:, 1:]
keys = ["x", "e_theta", "e_zeta", "e_theta_t", "e_theta_z", "e_zeta_z"]


def exact(X, its=8, chunk=50000):
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        x = X[s : s + chunk]
        tz = TZ[tree.query(x, workers=-1)[1]].copy()
        for _ in range(its):
            nodes = np.column_stack([np.ones(len(x)), tz])
            q = surf.compute(keys, grid=Grid(nodes, sort=False), basis="xyz")
            f = np.asarray(q["x"]) - x
            St, Sz = np.asarray(q["e_theta"]), np.asarray(q["e_zeta"])
            g1, g2 = np.sum(f * St, -1), np.sum(f * Sz, -1)
            a = np.sum(St * St, -1) + np.sum(f * np.asarray(q["e_theta_t"]), -1)
            b = np.sum(St * Sz, -1) + np.sum(f * np.asarray(q["e_theta_z"]), -1)
            c = np.sum(Sz * Sz, -1) + np.sum(f * np.asarray(q["e_zeta_z"]), -1)
            det = a * c - b * b
            tz[:, 0] -= (c * g1 - b * g2) / det
            tz[:, 1] -= (a * g2 - b * g1) / det
        out[s : s + chunk] = np.linalg.norm(f, axis=-1)
        res = np.max(np.abs([g1, g2]))
    return out, res


X = np.asarray(surf.compute(["R", "Z"], grid=g)["R"]), np.asarray(
    surf.compute(["R", "Z"], grid=g)["Z"]
)
pad = 0.3
R0, R1, Z0, Z1 = X[0].min() - pad, X[0].max() + pad, X[1].min() - pad, X[1].max() + pad
dphi = 2 * np.pi / NFP
rng = np.random.default_rng(0)
n = 300000
P = np.stack(
    [rng.uniform(R0, R1, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(Z0, Z1, n)], -1
)
cyl = lambda P: np.stack(
    [P[..., 0] * np.cos(P[..., 1]), P[..., 0] * np.sin(P[..., 1]), P[..., 2]], -1
)
dc = tree.query(cyl(P), workers=-1)[0]
P = P[(dc > 0.07) & (dc < 0.17)]
de, res = exact(cyl(P))
m = (de > 0.08) & (de < 0.16)
P, de = P[m], de[m]
print(f"test pts {len(de)}, Newton stationarity residual {res:.1e}")
for h in [0.02, 0.01]:
    t0 = time.time()
    nR, nZ = int(np.ceil((R1 - R0) / h)) + 1, int(np.ceil((Z1 - Z0) / h)) + 1
    nP = int(np.ceil(dphi * X[0].max() / h))
    rr, pp, zz = (
        np.linspace(R0, R1, nR),
        np.arange(nP) * dphi / nP,
        np.linspace(Z0, Z1, nZ),
    )
    G = cyl(np.stack(np.meshgrid(rr, pp, zz, indexing="ij"), -1)).reshape(-1, 3)
    D = tree.query(G, workers=-1)[0]
    band = (D > 0.08 - 3 * h) & (D < 0.16 + 3 * h)
    D[band] = exact(G[band])[0]
    C = spline_filter(D.reshape(nR, nP, nZ), order=3, mode="grid-wrap")
    idx = np.stack(
        [
            (P[:, 0] - R0) / (rr[1] - rr[0]),
            (P[:, 1] % dphi) / (dphi / nP),
            (P[:, 2] - Z0) / (zz[1] - zz[0]),
        ]
    )
    err = map_coordinates(C, idx, order=3, mode="grid-wrap", prefilter=False) - de
    print(
        f"h={1e3*h:.0f} mm: {D.size/1e6:.2f}M nodes, {band.sum()/1e6:.2f}M refined, {time.time()-t0:.0f} s; "
        f"err +{1e3*err.max():.4f} / {1e3*err.min():.4f} mm, p99 |err| {1e3*np.quantile(np.abs(err), 0.99):.4f} mm"
    )
