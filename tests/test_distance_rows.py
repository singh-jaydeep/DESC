"""Tests for the coil-coil and plasma-coil distance rows."""

import numpy as np
import pytest

from desc.coils import CoilSet, FourierPlanarCoil, PiecewisePlanarArcCoil
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    CoilLength,
    CoilSetDistanceRows,
    FixCoilCurrent,
    FixParameters,
    ObjectiveFunction,
    PlasmaCoilSetDistanceRows,
)
from desc.objectives._coils import _coil_node_grid
from desc.optimize import Optimizer


def _arc_coils():
    """Two arc coils per half period, symmetric, close to each other."""
    phi = np.array([0.25, 0.7])
    coils = []
    for k, p in enumerate(phi):
        c, s = np.cos(p), np.sin(p)
        h = np.array([[2.6 * c, 2.6 * s, 0.0], [3.4 * c, 3.4 * s, 0.0]])
        coils.append(
            PiecewisePlanarArcCoil(
                1e5, h, tilts=[np.pi / 2 + 0.1 * k, np.pi / 2], shape=[[0.6], [0.5]]
            )
        )
    return CoilSet(coils, NFP=2, sym=True)


def _spline_coils():
    """The arc coils converted to splines with a break at every hinge."""
    arcs = _arc_coils()
    s = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    return arcs.to_SplineXYZ(knots=s, break_indices=[0, 30])


def _dense_min(xa, xb):
    return np.min(np.linalg.norm(xa[:, None] - xb[None], axis=-1))


@pytest.mark.unit
@pytest.mark.parametrize("make", [_arc_coils, _spline_coils], ids=["arcs", "spline"])
def test_coil_rows_lower_bound_true_distance(make):
    """Every pair's smallest row is below the brute-force distance of the curves."""
    coils = make()
    obj = CoilSetDistanceRows(
        coils,
        select_distance=0.45,
        bounds=(0.2, np.inf),
        normalize=False,
        grid=_coil_node_grid(coils, 201),  # gap computed from the coils
        max_active_rows=20000,
    )
    obj.build(verbose=0)
    f = np.asarray(obj.compute(coils.params_dict))
    ids = np.asarray(obj.compute_row_ids(coils.params_dict))
    used = ids >= 0
    assert 0 < used.sum() < obj.dim_f
    np.testing.assert_allclose(f[~used], 0.45)  # unused slots read select_distance
    of = ObjectiveFunction(obj)
    of.build(verbose=0)
    J = np.asarray(of.jac_unscaled(of.x(coils)))
    assert np.all(np.isfinite(J)) and np.any(J[used] != 0)
    assert not np.any(J[~used])  # unused slots are constants

    dense = np.asarray(coils._compute_position(grid=LinearGrid(N=4000), basis="xyz"))
    N = obj._n_nodes
    pair = ids[used] // (N * N)
    for p, (a, b) in enumerate(obj._pairs):
        rows = f[used][pair == p]
        if rows.size:
            assert rows.min() <= _dense_min(dense[a], dense[b])


@pytest.mark.unit
def test_coil_rows_ids_decode_and_chunking():
    """Row ids decode to the node pair they hold; chunked scan equals unchunked."""
    coils = _arc_coils()
    kw = dict(
        select_distance=0.45,
        bounds=(0.2, np.inf),
        normalize=False,
        gap=0.01,
        max_active_rows=20000,
    )
    grid = _coil_node_grid(coils, 101)
    ref = CoilSetDistanceRows(coils, grid=grid, **kw)
    small = CoilSetDistanceRows(coils, grid=grid, scan_block_size=500, **kw)
    ref.build(verbose=0)
    small.build(verbose=0)
    assert small._block_rows < ref._block_rows  # several blocks per pair
    p = coils.params_dict
    f, ids = np.asarray(ref.compute(p)), np.asarray(ref.compute_row_ids(p))
    np.testing.assert_array_equal(np.asarray(small.compute(p)), f)
    np.testing.assert_array_equal(np.asarray(small.compute_row_ids(p)), ids)

    x = np.asarray(coils._compute_position(grid=grid, basis="xyz"))
    N = ref._n_nodes
    for row, cid in zip(f[ids >= 0], ids[ids >= 0]):
        k, i, j = cid // (N * N), (cid // N) % N, cid % N
        a, b = ref._pairs[k]
        np.testing.assert_allclose(row, np.linalg.norm(x[a, i] - x[b, j]) - 0.01)
    assert np.all(np.diff(ids[ids >= 0]) > 0)  # in id order


@pytest.mark.unit
def test_coil_rows_keep_and_overflow():
    """Kept candidates stay selected; too few slots warn."""
    coils = _arc_coils()
    obj = CoilSetDistanceRows(
        coils,
        select_distance=0.3,
        bounds=(0.2, np.inf),
        normalize=False,
        gap=0.0,
        max_active_rows=20000,
    )
    obj.build(verbose=0)
    p = coils.params_dict
    ids = np.asarray(obj.compute_row_ids(p))
    far = np.setdiff1d(np.arange(0, obj.num_row_ids, 997), ids)[:3]
    obj.keep_rows(far)
    kept = np.asarray(obj.compute_row_ids(p))
    assert set(far) <= set(kept) and set(ids[ids >= 0]) <= set(kept)

    small = CoilSetDistanceRows(
        coils,
        select_distance=0.3,
        bounds=(0.2, np.inf),
        normalize=False,
        gap=0.0,
        max_active_rows=2,
    )
    small.build(verbose=0)
    with pytest.warns(UserWarning, match="Raise max_active_rows"):
        small.compute(p)


@pytest.mark.unit
def test_plasma_rows_fixed_and_moving_boundary():
    """Plasma rows: lower bound on the true distance; eq_fixed=False moves with eq."""
    surf = FourierRZToroidalSurface(
        R_lmn=[3.5, 1.0],
        Z_lmn=[-1.0],
        modes_R=[[0, 0], [1, 0]],
        modes_Z=[[-1, 0]],
        NFP=4,
    )
    coils = CoilSet(
        FourierPlanarCoil(r_n=1.6, center=[3.5, 0, 0], normal=[0, 1, 0]),
        NFP=4,
        sym=True,
    )
    kw = dict(
        select_distance=0.9,
        bounds=(0.4, np.inf),
        plasma_grid=LinearGrid(M=16, N=16, NFP=surf.NFP),
        normalize=False,
    )
    fixed = PlasmaCoilSetDistanceRows(surf, coils, eq_fixed=True, **kw)
    moving = PlasmaCoilSetDistanceRows(surf, coils, **kw)
    fixed.build(verbose=0)
    moving.build(verbose=0)
    assert moving.things == [surf, coils] and fixed.things == [coils]
    f = np.asarray(fixed.compute(coils.params_dict))
    np.testing.assert_array_equal(
        np.asarray(moving.compute(surf.params_dict, coils.params_dict)), f
    )
    used = np.asarray(fixed.compute_row_ids(coils.params_dict)) >= 0
    assert used.any()
    # true distance: dense surface and coil
    s = surf.compute("x", grid=LinearGrid(M=80, N=80), basis="xyz")["x"]
    x = np.asarray(coils._compute_position(grid=LinearGrid(N=400), basis="xyz"))
    assert f[used].min() <= _dense_min(np.asarray(s), x.reshape(-1, 3))

    obj = ObjectiveFunction(moving)
    obj.build(verbose=0)
    J = np.asarray(obj.jac_unscaled(obj.x(surf, coils)))
    n_surf = surf.dim_x
    assert np.any(J[used, :n_surf] != 0)  # rows depend on the boundary


@pytest.mark.unit
def test_coil_rows_with_auglag():
    """Two coplanar circles grow toward each other and stop at the distance bound."""
    coils = CoilSet(
        FourierPlanarCoil(r_n=2, basis="xyz", center=[0, 0, 0], normal=[0, 0, 1]),
        FourierPlanarCoil(r_n=2, basis="xyz", center=[10, 0, 0], normal=[0, 0, 1]),
    )
    # select_distance above the starting gap of 6: a row that only appears inside a
    # narrower band can be stepped over in one step, and crossed circles cannot be
    # separated by small steps
    rows = CoilSetDistanceRows(
        coils,
        select_distance=7.0,
        bounds=(1.0, np.inf),
        normalize=False,
        gap=0.0,
        max_active_rows=5000,
    )
    obj = ObjectiveFunction(CoilLength(coils, target=2 * np.pi * 6))
    cons = (
        rows,
        FixCoilCurrent(coils),
        FixParameters(coils, [{"center": True, "normal": True}] * 2),
    )
    (out,), res = Optimizer("lsq-auglag-composite").optimize(
        coils, obj, cons, maxiter=300, gtol=1e-8, ctol=1e-8, verbose=0, copy=True
    )
    assert res.success
    r = np.array([c.r_n[0] for c in out])
    # radii would grow to 6, but the gap between the circles stops at 1 (to within
    # the node discretization, as the gap is 0 here)
    np.testing.assert_allclose(r.sum(), 9, rtol=1e-2)
    assert np.all(r > 4.4)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*Raise max_active_rows")
def test_coil_rows_overflow_keeps_closest():
    """With too few slots, the rows kept are the closest candidates."""
    coils = _arc_coils()
    grid = _coil_node_grid(coils, 101)
    kw = dict(select_distance=0.45, bounds=(0.2, np.inf), normalize=False, gap=0.0)
    full = CoilSetDistanceRows(coils, grid=grid, max_active_rows=20000, **kw)
    small = CoilSetDistanceRows(
        coils, grid=grid, max_active_rows=50, scan_block_size=500, **kw
    )
    full.build(verbose=0)
    small.build(verbose=0)
    p = coils.params_dict
    f = np.asarray(full.compute(p))[np.asarray(full.compute_row_ids(p)) >= 0]
    assert f.size > 50
    g = np.asarray(small.compute(p))
    ids = np.asarray(small.compute_row_ids(p))
    assert np.all(ids >= 0) and np.all(np.diff(ids) > 0)  # full, in id order
    np.testing.assert_allclose(np.sort(g), np.sort(f)[:50])
    # kept rows stay selected even when far
    far = np.setdiff1d(np.asarray(full.compute_row_ids(p)), ids)[-3:]
    small.keep_rows(far)
    kept = np.asarray(small.compute_row_ids(p))
    assert set(far) <= set(kept)
