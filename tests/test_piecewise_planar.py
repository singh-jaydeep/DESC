"""Tests for the piecewise-planar coil constraints, CoilCurrentBand and B.n rows."""

import numpy as np
import pytest

from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    PiecewisePlanarArcCoil,
    PolarPlanarArcCoil,
)
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.magnetic_fields import VerticalMagneticField
from desc.objectives import (
    ArcFold,
    CoilCurrentBand,
    CoilLength,
    FixCoilCurrent,
    FixHingeHeight,
    HingeLevel,
    HingeMidHeightRelative,
    HingeTilt,
    HingeTurningAngle,
    ObjectiveFunction,
    PointwiseBn,
    PolarArcRadius,
    VacuumBoundaryBn,
    VacuumBoundaryError,
)
from desc.objectives._piecewise_planar import axis_coeffs, fold_angles, turning_angles
from desc.optimize import Optimizer


def _value(obj, *things):
    o = ObjectiveFunction(obj)
    o.build(verbose=0)
    return np.asarray(o.compute_unscaled(o.x(*things)))


def _circle_arcs(B=2, M=3, z=0.0):
    """A planar circle of radius 1 around (3, 0, z), normal along y, as arcs."""
    s = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    x = np.stack([3 + np.cos(s), np.zeros_like(s), z + np.sin(s)], axis=1)
    return PiecewisePlanarArcCoil.from_values(
        1e5, x, B=B, M=M, knots=s, fit_method="chord"
    )


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.3, 1.0])
def test_arc_fold_tilted_arc(alpha):
    """A planar coil has fold 0; tilting one arc by alpha gives fold alpha."""
    coil = _circle_arcs()
    np.testing.assert_allclose(_value(ArcFold(coil), coil), 1, atol=1e-10)
    fold = ArcFold(coil)
    fold.build(verbose=0)
    sides = fold.sides
    coil.tilts = np.asarray(coil.tilts) + np.array([0.0, alpha])
    np.testing.assert_allclose(
        _value(ArcFold(coil, sides=sides), coil), np.cos(alpha), atol=1e-10
    )
    np.testing.assert_allclose(
        fold_angles(CoilSet(coil), sides=sides, deg=False), alpha, atol=1e-10
    )


@pytest.mark.unit
def test_arc_fold_jacobian_finite_difference():
    """Jacobian of ArcFold matches central differences."""
    coil = _circle_arcs(B=3, M=2)
    coil.tilts = np.asarray(coil.tilts) + np.array([0.2, -0.3, 0.4])
    o = ObjectiveFunction(ArcFold(coil))
    o.build(verbose=0)
    x = np.asarray(o.x(coil))
    J = np.asarray(o.jac_unscaled(x))
    h = 1e-6
    fd = np.stack(
        [
            (o.compute_unscaled(x + h * e) - o.compute_unscaled(x - h * e)) / (2 * h)
            for e in np.eye(x.size)
        ],
        axis=1,
    )
    np.testing.assert_allclose(J, fd, atol=1e-7)


@pytest.mark.unit
@pytest.mark.parametrize("B", [3, 4, 6])
def test_turning_angle_polygon(B):
    """Straight arcs on a regular planar polygon turn by 2 pi / B at every hinge."""
    ang = 2 * np.pi * np.arange(B) / B
    H = np.stack([3 + np.cos(ang), np.zeros(B), np.sin(ang)], axis=1)
    coil = PiecewisePlanarArcCoil(1e5, H, shape=np.zeros((B, 2)))
    np.testing.assert_allclose(
        _value(HingeTurningAngle(coil), coil), np.cos(2 * np.pi / B), atol=1e-12
    )
    np.testing.assert_allclose(
        turning_angles(CoilSet(coil), deg=False), 2 * np.pi / B, atol=1e-12
    )


@pytest.mark.unit
def test_turning_angle_polar_circle():
    """Two semicircles on one chord: a circle (no corner) or folded back (cusps)."""
    H = np.array([[4.0, 0, 0], [2.0, 0, 0]])
    # opposite sides of the chord, in one plane: a circle
    coil = PolarPlanarArcCoil(1e5, H, tilts=[0, np.pi], shape=np.zeros((2, 1)))
    np.testing.assert_allclose(_value(HingeTurningAngle(coil), coil), 1, atol=1e-12)
    # the same side: the second arc retraces the first, a cusp at both hinges
    coil = PolarPlanarArcCoil(1e5, H, tilts=np.zeros(2), shape=np.zeros((2, 1)))
    np.testing.assert_allclose(_value(HingeTurningAngle(coil), coil), -1, atol=1e-12)


def _sampled_turning(coil, eps=1e-7):
    """Turning angles from sampled tangents either side of each hinge (RR-02's way).

    The tangents come from the curve itself, in the lab frame.
    """
    from desc.grid import Grid

    B = int(coil.B)
    brk = 2 * np.pi * np.arange(B) / B
    s = np.concatenate([[(b - eps) % (2 * np.pi), b + eps] for b in brk])
    g = Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False)
    t = np.asarray(coil.compute("x_s", grid=g, basis="xyz")["x_s"])
    t = t / np.linalg.norm(t, axis=1)[:, None]
    return np.arccos(np.clip(np.sum(t[0::2] * t[1::2], axis=1), -1, 1))


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PiecewisePlanarArcCoil, PolarPlanarArcCoil])
@pytest.mark.parametrize("B", [2, 3, 5])
def test_turning_angle_matches_sampled_tangents(cls, B):
    """Closed form equals the angle between sampled tangents, under a rigid rotation."""
    rng = np.random.default_rng(B)
    ang = 2 * np.pi * np.arange(B) / B + 0.2 * rng.standard_normal(B)
    H = np.stack(
        [3 + np.cos(ang), 0.3 * rng.standard_normal(B), 1.2 * np.sin(ang)], axis=1
    )
    coil = cls(
        1e5,
        H,
        tilts=0.5 * rng.standard_normal(B),
        shape=0.15 * rng.standard_normal((B, 3)),
    )
    ref = np.asarray(turning_angles(coil, deg=False, per_hinge=True))[0]
    np.testing.assert_allclose(_sampled_turning(coil), ref, atol=1e-6)
    # a rigid rotation moves the coil, not its corners
    th = 0.7
    coil.rotmat = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    ).ravel()
    np.testing.assert_allclose(_sampled_turning(coil), ref, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PiecewisePlanarArcCoil, PolarPlanarArcCoil])
def test_turning_angle_jacobian_finite_difference(cls):
    """Jacobian of HingeTurningAngle matches central differences."""
    rng = np.random.default_rng(0)
    coil = _circle_arcs(B=3, M=2)
    coil = cls(
        1e5,
        coil.hinges,
        tilts=np.asarray(coil.tilts) + np.array([0.2, -0.3, 0.4]),
        shape=0.1 * rng.standard_normal((3, 2)),
    )
    o = ObjectiveFunction(HingeTurningAngle(CoilSet(coil, coil.copy())))
    o.build(verbose=0)
    x = np.asarray(o.x(*o.things))
    J = np.asarray(o.jac_unscaled(x))
    h = 1e-6
    fd = np.stack(
        [
            (o.compute_unscaled(x + h * e) - o.compute_unscaled(x - h * e)) / (2 * h)
            for e in np.eye(x.size)
        ],
        axis=1,
    )
    assert J.shape == (6, x.size)
    np.testing.assert_allclose(J, fd, atol=1e-7)


@pytest.mark.unit
def test_hinge_level_and_tilt():
    """Hinge height differences and chord elevations by hand."""
    H = np.array([[4.0, 0, 0.1], [3.0, 1.0, 0.4], [2.0, 0, -0.2]])
    coil = PiecewisePlanarArcCoil(1e5, H, shape=np.full((3, 1), 0.1))
    coils = CoilSet(coil, coil.copy())
    np.testing.assert_allclose(
        _value(HingeLevel(coils), coils), [0.3, -0.3, 0.3, -0.3], atol=1e-14
    )
    d = np.roll(H, -1, axis=0) - H
    sin = d[:, 2] / np.linalg.norm(d, axis=1)
    np.testing.assert_allclose(_value(HingeTilt(coils), coils), np.tile(sin, 2))
    # B = 2: one chord, one row per coil
    coil2 = PiecewisePlanarArcCoil(1e5, H[:2], shape=np.full((2, 1), 0.1))
    np.testing.assert_allclose(_value(HingeTilt(coil2), coil2), sin[:1])
    # a leaf coil with a rigid transform is refused
    moved = coil.copy()
    moved.shift = [0, 0, 1.0]
    with pytest.raises(NotImplementedError, match="rigid transform"):
        _value(HingeTilt(moved), moved)


@pytest.mark.unit
def test_hinge_mid_height_relative():
    """The axis height series reproduces eq.axis; rows are (z_mean - Z_axis) / z_max."""
    eq = get("precise_QH")
    phi = np.linspace(0, 2 * np.pi, 13)
    k, zk, is_cos = axis_coeffs(eq)
    Z = np.sum(
        zk * np.where(is_cos, np.cos(k * phi[:, None]), np.sin(k * phi[:, None])), 1
    )
    ref = eq.axis.compute("Z", grid=LinearGrid(zeta=phi[:-1]))["Z"]
    np.testing.assert_allclose(Z[:-1], ref, atol=1e-12)
    assert np.ptp(Z) > 0.01  # a helical axis

    p = 0.4  # toroidal angle of the hinge midpoint
    H = np.array(
        [[1.2 * np.cos(p), 1.2 * np.sin(p), 0.05], [0.8, 0.8 * np.tan(p), -0.01]]
    )
    H[1] = 2 * np.array([np.cos(p), np.sin(p), 0.0]) - H[0]  # midpoint at angle p
    H[1, 2] = -0.01
    coil = PiecewisePlanarArcCoil(1e5, H, shape=np.full((2, 1), 0.1))
    mid = H.mean(axis=0)
    z_axis = np.sum(zk * np.where(is_cos, np.cos(k * p), np.sin(k * p)))
    np.testing.assert_allclose(
        _value(HingeMidHeightRelative(coil, eq, 0.05), coil),
        (mid[2] - z_axis) / 0.05,
        atol=1e-12,
    )


@pytest.mark.unit
def test_fix_hinge_height():
    """Hinge heights stay fixed while the hinges move in x and y."""
    coils = CoilSet(_circle_arcs(), _circle_arcs(z=0.5))
    for c in coils:
        h = np.array(c.hinges).reshape(-1, 3)
        h[:, 2] = 0.0
        c.hinges = h.reshape(-1)
    con = FixHingeHeight(coils, height=0.0)
    obj = ObjectiveFunction(CoilLength(coils, target=1.3 * 2 * np.pi))
    (out,), _ = Optimizer("lsq-exact").optimize(
        coils, obj, (con, FixCoilCurrent(coils)), maxiter=30, verbose=0, copy=True
    )
    for c, c0 in zip(out, coils):
        h = np.asarray(c.hinges).reshape(-1, 3)
        np.testing.assert_allclose(h[:, 2], 0, atol=1e-14)
        assert not np.allclose(h, np.asarray(c0.hinges).reshape(-1, 3))
        np.testing.assert_allclose(c.compute("length")["length"], 1.3 * 2 * np.pi)
    with pytest.raises(ValueError, match="arc coils"):
        FixHingeHeight(CoilSet(FourierPlanarCoil()))


@pytest.mark.unit
def test_polar_arc_radius():
    """Relative radius is 1 + (2 / |chord|) sum_m a_m sin(m pi phi)."""
    H = np.array([[4.0, 0, 0], [2.0, 0, 0]])
    a = np.array([[0.3, 0.1], [-0.2, 0.05]])
    coil = PolarPlanarArcCoil(1e5, H, shape=a)
    K = 8
    phi = (np.arange(K) + 0.5) / K
    s = np.sin(np.pi * np.outer(phi, [1, 2]))
    expected = 1 + (a @ s.T) / 1.0  # |chord| / 2 = 1
    np.testing.assert_allclose(
        _value(PolarArcRadius(coil, K=K), coil), expected.ravel(), atol=1e-14
    )


@pytest.mark.unit
def test_coil_current_band():
    """I_i over the average at build."""
    coils = CoilSet(
        FourierPlanarCoil(current=1e5),
        FourierPlanarCoil(current=2e5, center=[10, 2, 0]),
        FourierPlanarCoil(current=3e5, center=[10, -2, 0]),
    )
    np.testing.assert_allclose(_value(CoilCurrentBand(coils), coils), [0.5, 1, 1.5])


@pytest.mark.unit
def test_pointwise_bn_vertical_field():
    """A vertical field on a circular torus: B.n / |B| is the z component of n."""
    surf = FourierRZToroidalSurface(
        R_lmn=[3.0, 1.0], Z_lmn=[-1.0], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    grid = LinearGrid(M=6, N=0)
    field = VerticalMagneticField(B0=0.7)
    n = surf.compute("n_rho", grid=grid, basis="xyz")["n_rho"]
    np.testing.assert_allclose(
        _value(PointwiseBn(surf, field, grid), field), n[:, 2], atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:Pressure is non-zero")
@pytest.mark.filterwarnings("ignore:Current is non-zero")
def test_vacuum_boundary_bn():
    """Rows equal the B.n rows of VacuumBoundaryError divided by sqrt(g)."""
    eq = get("DSHAPE")
    coils = CoilSet(
        FourierPlanarCoil(current=1e6, center=[3.5, 0, 0], normal=[0, 1, 0], r_n=2.2),
        NFP=12,
        sym=True,
    )
    grid = LinearGrid(M=4, N=4, NFP=eq.NFP)
    ours = VacuumBoundaryBn(eq, coils, eval_grid=grid, normalize=False)
    ref = VacuumBoundaryError(eq, coils, eval_grid=grid, normalize=False)
    f, f_ref = _value(ours, eq, coils), _value(ref, eq, coils)
    g = eq.compute("|e_theta x e_zeta|", grid=grid)["|e_theta x e_zeta|"]
    np.testing.assert_allclose(f, f_ref[: f.size] / np.sqrt(g), rtol=1e-10)
    assert ours.dim_f == grid.num_nodes
