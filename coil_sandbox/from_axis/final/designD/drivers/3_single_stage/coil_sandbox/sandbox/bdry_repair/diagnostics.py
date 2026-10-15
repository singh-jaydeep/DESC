"""Measurements of a boundary (or equilibrium): is its poloidal angle healthy, and how far apart are two surfaces."""
import numpy as np
from desc.grid import LinearGrid


def boundary_health(obj, NFP):
    """(smallest area element / median, largest principal curvature [1/m]) of the rho = 1 surface."""
    g = LinearGrid(rho=np.array([1.0]), M=64, N=64, NFP=NFP, sym=False)
    d = obj.compute(["|e_theta x e_zeta|", "curvature_k1_rho", "curvature_k2_rho"], grid=g)
    ar = np.asarray(d["|e_theta x e_zeta|"])
    k = np.maximum(np.abs(np.asarray(d["curvature_k1_rho"])), np.abs(np.asarray(d["curvature_k2_rho"])))
    return float(ar.min() / np.median(ar)), float(k.max())


def jac_min(eq, rhos=(0.8, 0.95, 1.0)):
    """sqrt(g) min / median on each of the given surfaces (negative = the coordinates fold there)."""
    out = []
    for rho in rhos:
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        s = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
        out.append(float(s.min() / np.median(s)))
    return out


def sections(obj, NFP, nt, nz):
    """R, Z of the rho = 1 surface, shape (nt, nz), on uniform theta and zeta over one field period."""
    t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z = np.linspace(0, 2 * np.pi / NFP, nz, endpoint=False)
    g = LinearGrid(rho=np.array([1.0]), theta=t, zeta=z, NFP=NFP)
    d = obj.compute(["R", "Z"], grid=g)
    return np.asarray(g.meshgrid_reshape(d["R"], "rtz")[0]), np.asarray(g.meshgrid_reshape(d["Z"], "rtz")[0]), z


def _seg_dist(P, Q):
    """Distance from each point of P (n, 2) to the closed polyline Q (m, 2)."""
    A, AB = Q, np.roll(Q, -1, 0) - Q
    AP = P[:, None] - A[None]
    s = np.clip((AP * AB[None]).sum(-1) / np.maximum((AB**2).sum(-1), 1e-300)[None], 0, 1)
    return np.linalg.norm(AP - s[..., None] * AB[None], axis=-1).min(1)


def displacement(old, new, NFP, nz=24, nt=3000, every=5):
    """Distance between two surfaces that share the cylindrical toroidal angle, measured both ways in nz planes.
    Independent of how either is parametrized. Returns dict(rms, p99, max) in mm."""
    Ro, Zo, _ = sections(old, NFP, nt, nz)
    Rn, Zn, _ = sections(new, NFP, nt, nz)
    d = []
    for j in range(nz):
        o, n = np.c_[Ro[:, j], Zo[:, j]], np.c_[Rn[:, j], Zn[:, j]]
        d += [_seg_dist(n[::every], o), _seg_dist(o[::every], n)]
    d = 1e3 * np.concatenate(d)
    return dict(rms=float(np.sqrt(np.mean(d**2))), p99=float(np.percentile(d, 99)), max=float(d.max()))
