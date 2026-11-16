"""Equilibria, bounds and the dense check of a coilset against its engineering bounds.

The dense check is independent of the optimization's own grids: lengths by composite
Gauss-Legendre, curvature on 2000 nodes per arc, distances between dense polylines and
the full plasma boundary, linking numbers, and B.n on one field period.
"""

import itertools
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from desc.coils import CoilSet, coil_curvature_grid, coil_quad_grid, coil_source_grid
from desc.grid import LinearGrid

# Gil et al., PRE 114, 025202, Table III (reactor with a = 1.7 m)
REACTOR_A = 1.7
REACTOR = dict(length_per_half_period=160.0, kappa=1.0, d_cc=0.8, d_pc=1.5)
UPPER, LOWER = ("L", "kappa"), ("dcc", "dpc")


def load_equilibrium(spec):
    """An Equilibrium from a desc.examples name or a DESC .h5 (last of a family)."""
    from desc.equilibrium import EquilibriaFamily

    if os.path.exists(spec):
        from desc.io import load

        eq = load(spec)
        return eq[-1] if isinstance(eq, EquilibriaFamily) else eq
    from desc.examples import get

    return get(spec)


def leaf_coils(coilset):
    """The independent coils of a (nested) coilset."""
    out = []
    for c in coilset:
        out += leaf_coils(c) if isinstance(c, CoilSet) else [c]
    return out


def load_bounds(path, eq, n_coils, a=None):
    """Bounds dict(L, kappa, dcc, dpc, a, f) from a bounds file.

    ``"scale": "absolute"`` values are used as given (m, 1/m; "L" per coil). Otherwise
    (``"reactor"``, the default) values are the reactor set above, scaled by
    f = a / 1.7, with the half-period length split over ``n_coils``. null drops a
    bound. Keys starting with "_" are comments. kappa_MS and other keys are reported
    as not enforced.
    """
    a = float(a) if a else float(eq.compute("a")["a"])
    f = a / REACTOR_A
    r = dict(REACTOR)
    user = {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}
    scale = user.pop("scale", "reactor")
    r.update(user)

    def sc(v, p):
        return None if v is None else (v if scale == "absolute" else v * f**p)

    L = r["L"] if "L" in r else r["length_per_half_period"] / n_coils
    out = dict(
        L=sc(L, 1), kappa=sc(r["kappa"], -1), dcc=sc(r["d_cc"], 1), dpc=sc(r["d_pc"], 1)
    )
    known = {"L", "length_per_half_period", "kappa", "d_cc", "d_pc"}
    ignored = sorted(k for k, v in user.items() if k not in known and v is not None)
    return dict(out, a=a, f=f, scale=scale, not_enforced=ignored)


def linked_pairs(pos):
    """1-based pairs with |Gauss linking integral| > 0.5 between closed polylines."""
    out = []
    for i, j in itertools.combinations(range(pos.shape[0]), 2):
        p, q = pos[i], pos[j]
        dp, dq = np.roll(p, -1, 0) - p, np.roll(q, -1, 0) - q
        r = (p + 0.5 * dp)[:, None, :] - (q + 0.5 * dq)[None, :, :]
        cr = np.cross(dp[:, None, :], dq[None, :, :])
        lk = np.sum(
            np.einsum("ijk,ijk->ij", r, cr) / np.linalg.norm(r, axis=-1) ** 3
        ) / (4 * np.pi)
        if abs(lk) > 0.5:
            out.append([i + 1, j + 1, round(float(lk), 3)])
    return out


def surface_tree(eq, ntheta=256, nzeta=1024):
    """KD-tree of the full plasma boundary."""
    grid = LinearGrid(
        rho=np.array([1.0]),
        theta=np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi, nzeta, endpoint=False),
        NFP=1,
    )
    return cKDTree(np.asarray(eq.compute("x", grid=grid, basis="xyz")["x"]))


_BN = {}


def bn_stats(coilset, eq, vacuum, nphi=72, ntheta=96):
    """|B.n| on one field period: mean and max over <|B|>, and the max in tesla.

    B is the coil field plus, unless ``vacuum``, the plasma field by virtual casing on
    twice DESC's default source grid.
    """
    key = (id(eq), vacuum)
    if key not in _BN:
        src = LinearGrid(
            rho=np.array([1.0]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False
        )
        grid = LinearGrid(
            rho=np.array([1.0]),
            theta=max(ntheta, src.num_theta),
            zeta=max(nphi, src.num_zeta),
            NFP=eq.NFP,
            sym=False,
        )
        d = eq.compute(["R", "phi", "Z", "n_rho"], grid=grid)
        x = np.stack([d["R"], d["phi"], d["Z"]], axis=1)
        if vacuum:
            Bp = np.zeros_like(x)
        else:
            from desc.integrals.singularities import compute_B_plasma

            Bp = np.asarray(compute_B_plasma(eq, grid, src, chunk_size=64))
        _BN[key] = (eq, x, np.asarray(d["n_rho"]), Bp)
    _, x, n, Bp = _BN[key]
    B = np.asarray(
        coilset.compute_magnetic_field(
            x, basis="rpz", source_grid=coil_source_grid(coilset)
        )
    )
    B = B + Bp
    Bn = np.abs(np.sum(B * n, axis=1))
    mB = float(np.mean(np.linalg.norm(B, axis=1)))
    return dict(
        mean=float(Bn.mean()) / mB, max=float(Bn.max()) / mB, max_T=float(Bn.max())
    )


def evaluate(coilset, eq, tree=None, vacuum=True):
    """Dense metrics: per-coil length and |kappa|, d_cc, d_pc, links, B.n."""
    tree = tree if tree is not None else surface_tree(eq)
    L, k = [], []
    for c in leaf_coils(coilset):
        q = c.compute("length", grid=coil_quad_grid(c, n=48), override_grid=False)
        L.append(float(q["length"]))
        kg = coil_curvature_grid(c, n=2000)
        k.append(float(np.max(c.compute("|curvature|", grid=kg)["|curvature|"])))
    pos = np.asarray(coilset._compute_position(grid=LinearGrid(N=1000), basis="xyz"))
    trees = [cKDTree(p) for p in pos]
    d_cc = min(
        trees[i].query(pos[j])[0].min()
        for i, j in itertools.combinations(range(len(pos)), 2)
    )
    d_pc = min(tree.query(p)[0].min() for p in pos)
    coarse = coilset._compute_position(grid=LinearGrid(N=64), basis="xyz")
    links = linked_pairs(np.asarray(coarse))
    bn = bn_stats(coilset, eq, vacuum)
    return dict(
        L=max(L),
        kappa=max(k),
        L_per_coil=L,
        kappa_per_coil=k,
        dcc=float(d_cc),
        dpc=float(d_pc),
        linked=len(links),
        linked_pairs=links,
        bn=bn["mean"],
        bn_max=bn["max"],
        bn_max_T=bn["max_T"],
        n_coils=int(pos.shape[0]),
    )


def margins(m, b):
    """Value over bound for every set bound: <= 1 for L, kappa; >= 1 for dcc, dpc."""
    return {k: m[k] / b[k] for k in UPPER + LOWER if b.get(k) is not None}


def verdict(m, b):
    """FEASIBLE, or FAILS with the violated bounds."""
    g = margins(m, b)
    bad = [k for k, v in g.items() if (v < 1 if k in LOWER else v > 1)]
    bad += ["linked"] if m["linked"] else []
    return "FEASIBLE" if not bad else "FAILS " + ",".join(bad)


def report(m, b):
    """One line of margins."""
    return "  ".join(f"{k} {v:.4f}" for k, v in margins(m, b).items())
