"""Tests for piecewise-planar arc curves and coils, and splines with breaks."""

import numpy as np
import pytest

from desc.coils import CoilSet, PiecewisePlanarArcCoil, PolarPlanarArcCoil
from desc.geometry import (
    FourierPlanarCurve,
    PiecewisePlanarArcCurve,
    PolarPlanarArcCurve,
    SplineXYZCurve,
)
from desc.grid import Grid, LinearGrid
from desc.io import load
from desc.objectives import CoilLength, FixCoilCurrent, ObjectiveFunction
from desc.optimize import Optimizer

HINGES = np.array(
    [[1.0, 0.9, 0.1], [-1.1, 1.0, -0.1], [-1.0, -1.0, 0.2], [1.0, -1.1, 0]]
)
TILTS = np.array([0.4, -0.7, 1.0, 0.3])
SHAPE = np.array([[0.3, 0.05], [0.2, -0.04], [0.35, 0.02], [0.25, 0.0]])


def _curves():
    h = HINGES + [5, 0, 0]
    return [
        PiecewisePlanarArcCurve(h, tilts=TILTS, shape=SHAPE),
        PolarPlanarArcCurve(h, tilts=TILTS, shape=0.1 * SHAPE),
    ]


def _at(curve, name, s):
    """Evaluate at curve parameters s, in the given order (wrapped into [0, 2pi))."""
    s = np.mod(np.atleast_1d(s), 2 * np.pi)
    z = np.zeros_like(s)
    grid = Grid(np.column_stack([z, z, s]), sort=False, jitable=True)
    return curve.compute(name, grid=grid, basis="xyz")[name]


@pytest.mark.unit
@pytest.mark.parametrize("curve", _curves(), ids=["transverse", "polar"])
def test_arc_derivatives_finite_difference(curve):
    """x_s, x_ss, x_sss match central differences away from the hinges."""
    B = 4
    # interior points of every arc, away from the hinges at s = 2 pi i / B
    s = (np.arange(B)[:, None] + np.array([0.2, 0.5, 0.8])).ravel() * 2 * np.pi / B
    h = 1e-5
    for f, df in [("x", "x_s"), ("x_s", "x_ss"), ("x_ss", "x_sss")]:
        fd = (_at(curve, f, s + h) - _at(curve, f, s - h)) / (2 * h)
        exact = _at(curve, df, s)
        np.testing.assert_allclose(
            fd, exact, rtol=1e-5, atol=1e-5 * np.abs(exact).max()
        )


@pytest.mark.unit
@pytest.mark.parametrize("curve", _curves(), ids=["transverse", "polar"])
def test_arc_c0_hinges_and_planar_arcs(curve):
    """Positions are continuous at the hinges, tangents jump, every arc is planar."""
    B = 4
    hinges = np.asarray(curve.hinges).reshape(B, 3)
    s_h = 2 * np.pi * np.arange(B) / B
    eps = 1e-9
    left = _at(curve, "x", s_h - eps)
    right = _at(curve, "x", s_h + eps)
    np.testing.assert_allclose(_at(curve, "x", s_h), hinges, atol=1e-12)
    np.testing.assert_allclose(left, hinges, atol=1e-7)
    np.testing.assert_allclose(right, hinges, atol=1e-7)

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    t_left = unit(_at(curve, "x_s", s_h - eps))
    t_right = unit(_at(curve, "x_s", s_h + eps))
    assert np.all(np.linalg.norm(t_left - t_right, axis=1) > 1e-2)  # corners

    for i in range(B):
        s = 2 * np.pi * (i + np.linspace(0, 1, 30)) / B
        x = _at(curve, "x", s)
        sv = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
        assert sv[2] < 1e-12 * sv[0]


@pytest.mark.unit
def test_transverse_arc_from_values_round_trip():
    """from_values recovers the parameters of a sampled arc curve."""
    curve = _curves()[0]
    s = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    x = _at(curve, "x", s)
    fit = PiecewisePlanarArcCurve.from_values(x, B=4, M=2, knots=s, fit_method="chord")
    np.testing.assert_allclose(fit.hinges, curve.hinges, atol=1e-12)
    np.testing.assert_allclose(_at(fit, "x", s), x, atol=1e-6)


@pytest.mark.unit
def test_arc_coilset_save_load(tmp_path):
    """A CoilSet of arc coils survives save/load, including the frozen arc_ref."""
    c1 = PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 0], tilts=TILTS, shape=SHAPE)
    c2 = PolarPlanarArcCoil(2e5, HINGES + [5, 0, 1], tilts=TILTS, shape=0.1 * SHAPE)
    for coil in (c1, c2):
        coils = CoilSet(coil, NFP=2, sym=True)
        path = tmp_path / f"{type(coil).__name__}.h5"
        coils.save(path)
        loaded = load(path)
        assert type(loaded[0]) is type(coil)
        np.testing.assert_allclose(loaded[0].arc_ref, coil.arc_ref)
        grid = LinearGrid(N=40)
        np.testing.assert_allclose(
            loaded[0].compute("x", grid=grid)["x"], coil.compute("x", grid=grid)["x"]
        )
        pt = np.array([[5.0, 0.2, 0.1]])
        np.testing.assert_allclose(
            loaded.compute_magnetic_field(pt), coils.compute_magnetic_field(pt)
        )


@pytest.mark.unit
def test_arc_ref_fixed_during_optimization():
    """arc_ref is a parameterization constant: auto-fixed, never moved by a solve."""
    coils = CoilSet(
        PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 0], tilts=TILTS, shape=SHAPE),
        PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 3], tilts=-TILTS, shape=SHAPE),
    )
    L0 = np.array([c.compute("length")["length"] for c in coils])
    ref0 = [np.array(c.arc_ref) for c in coils]
    obj = ObjectiveFunction(CoilLength(coils, target=1.2 * L0))
    (out,), _ = Optimizer("lsq-exact").optimize(
        coils, obj, (FixCoilCurrent(coils),), maxiter=20, verbose=0, copy=True
    )
    for c, c0, L, ref in zip(out, coils, L0, ref0):
        np.testing.assert_allclose(c.compute("length")["length"], 1.2 * L, rtol=1e-4)
        np.testing.assert_array_equal(c.arc_ref, ref)
        assert np.any(np.asarray(c.hinges) != np.asarray(c0.hinges))


@pytest.mark.unit
def test_spline_with_breaks_is_c0():
    """A spline with break indices is continuous at the breaks but has corners."""
    # a square sampled along its edges: corners at knots 0, 10, 20, 30
    t = np.linspace(0, 1, 10, endpoint=False)
    corners = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [1, 1, 0]])
    x = np.concatenate(
        [corners[i] + t[:, None] * (corners[i + 1] - corners[i]) for i in range(4)]
    )
    knots = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    c = SplineXYZCurve.from_values(x, knots=knots, break_indices=[0, 10, 20, 30])
    s_b = knots[[10, 20, 30]]
    eps = 1e-8
    np.testing.assert_allclose(_at(c, "x", s_b), corners[1:4], atol=1e-12)
    np.testing.assert_allclose(_at(c, "x", s_b - eps), corners[1:4], atol=1e-6)
    np.testing.assert_allclose(_at(c, "x", s_b + eps), corners[1:4], atol=1e-6)
    # the edges stay straight: no overshoot from interpolating across a corner
    s = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    np.testing.assert_allclose(
        np.max(np.abs(_at(c, "x", s)[:, :2]), axis=1), 1, atol=1e-12
    )
    # the default periodic spline rounds the corners instead
    smooth = SplineXYZCurve.from_values(x, knots=knots)
    assert np.max(np.abs(_at(smooth, "x", s)[:, :2])) > 1 + 1e-3


@pytest.mark.unit
def test_spline_break_indices_must_start_at_zero():
    """A first break after knot 0 would leave part of the curve uncovered."""
    s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    x = np.stack([8 * np.cos(s), 8 * np.sin(s), np.zeros_like(s)], axis=1)
    SplineXYZCurve.from_values(x, knots=s, break_indices=[0, 10, 20, 30])
    with pytest.raises(ValueError, match="break_indices must include 0"):
        SplineXYZCurve.from_values(x, knots=s, break_indices=[5, 15, 25, 35])


@pytest.mark.unit
def test_polar_planar_arc_signed_curvature_convention():
    """Signed curvature uses the rpz center, so it is right away from phi = 0."""
    fp = FourierPlanarCurve(center=[0, 3, 0], normal=[1, 0, 0], r_n=1.0)
    grid = LinearGrid(N=20)
    curve = PolarPlanarArcCurve.from_values(
        fp.compute("x", grid=grid, basis="xyz")["x"], B=4, M=2
    )
    data = curve.compute(
        ["x", "frenet_normal", "curvature", "|curvature|"], grid=grid, basis="xyz"
    )
    center = np.asarray(curve.params_dict["hinges"]).reshape(-1, 3).mean(axis=0)
    sgn = np.sign(np.sum((center - data["x"]) * data["frenet_normal"], axis=1))
    np.testing.assert_allclose(data["curvature"], sgn * data["|curvature|"])


@pytest.mark.unit
def test_polar_planar_arc_from_values_round_trip():
    """from_values recovers tilts of either sign (arcs reoriented to bulge +perp)."""
    hinges = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) + [5, 0, 0]
    curve = PolarPlanarArcCurve(
        hinges, tilts=[0.4, -0.7, 1.0, 0.3], shape=np.zeros(4), B=4, M=1
    )
    grid = LinearGrid(N=40)
    x = curve.compute("x", grid=grid, basis="xyz")["x"]
    fit = PolarPlanarArcCurve.from_values(x, B=4, M=1)
    # tilts with the right sign (the shape fit itself is only approximate)
    np.testing.assert_allclose(fit.params_dict["tilts"], curve.tilts, atol=0.03)
    np.testing.assert_allclose(
        fit.compute("x", grid=grid, basis="xyz")["x"], x, atol=0.1
    )


@pytest.mark.unit
def test_coil_quad_grid_length():
    """Composite Gauss-Legendre per arc gives arc lengths to quadrature accuracy."""
    from scipy.integrate import quad

    from desc.coils import coil_quad_grid, coil_source_grid

    # straight arcs: the length is the perimeter of the hinge polygon
    poly = PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 0], shape=np.zeros((4, 1)))
    perimeter = np.sum(np.linalg.norm(np.roll(HINGES, -1, 0) - HINGES, axis=1))
    for n in (4, 24):
        L = poly.compute("length", grid=coil_quad_grid(poly, n=n), override_grid=False)
        np.testing.assert_allclose(L["length"], perimeter, rtol=1e-13)
    # curved arcs: against adaptive quadrature of |x_s| arc by arc
    coil = PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 0], tilts=TILTS, shape=SHAPE)

    def speed(s):
        return np.linalg.norm(_at(coil, "x_s", s)[0])

    ref = sum(
        quad(speed, 2 * np.pi * i / 4, 2 * np.pi * (i + 1) / 4, epsabs=1e-13)[0]
        for i in range(4)
    )
    grid = coil_quad_grid(CoilSet(coil), n=24)
    L = coil.compute("length", grid=grid, override_grid=False)["length"]
    np.testing.assert_allclose(L, ref, rtol=1e-9)
    assert grid.num_nodes == 4 * 24
    assert coil_source_grid(coil).num_nodes == 4 * 24
    # smooth coils and splines
    assert coil_quad_grid(FourierPlanarCurve()).num_nodes == LinearGrid(N=400).num_nodes


@pytest.mark.unit
def test_coil_curvature_grid():
    """The curvature grid has n nodes strictly inside every arc, none on a hinge."""
    from desc.coils import coil_curvature_grid

    coil = PiecewisePlanarArcCoil(1e5, HINGES + [5, 0, 0], tilts=TILTS, shape=SHAPE)
    grid = coil_curvature_grid(CoilSet(coil), n=50)
    s = np.asarray(grid.nodes[:, 2])
    assert s.size == 4 * 50
    arc = np.floor(s / (np.pi / 2))
    np.testing.assert_array_equal(np.bincount(arc.astype(int)), [50] * 4)
    hinge_dist = np.abs(s[:, None] - np.pi / 2 * np.arange(5)[None]).min(axis=1)
    assert hinge_dist.min() > 0.9 * (np.pi / 2) / 100  # half a spacing
    np.testing.assert_allclose(np.sum(grid.spacing[:, 2]), 2 * np.pi)
    # usable as an objective grid: the constraint reads its node values
    from desc.objectives import CoilCurvature

    obj = CoilCurvature(CoilSet(coil), grid=grid, normalize=False)
    obj.build(verbose=0)
    f = obj.compute(CoilSet(coil).params_dict)
    np.testing.assert_allclose(f, coil.compute("|curvature|", grid=grid)["|curvature|"])
