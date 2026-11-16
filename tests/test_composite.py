"""Tests for the composite solvers and the row-identity interface they read."""

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.coils import CoilSet, FourierPlanarCoil
from desc.objectives import CoilCurvature, CoilLength, FixCoilCurrent, ObjectiveFunction
from desc.optimize import LinearConstraintProjection
from desc.optimize.aug_lagrangian_composite import _RowValues, lsq_auglag_composite
from desc.optimize.least_squares_composite import lsq_composite, paired_decrease


def _coilset():
    coil = FourierPlanarCoil(r_n=[2, 0.3], modes=[0, 2], basis="rpz")
    return CoilSet.linspaced_linear(
        coil, n=3, displacement=[0, 6, 0], check_intersection=False
    )


@pytest.mark.unit
def test_row_ids_and_scaled_bounds():
    """Default row ids are row positions; scaled bounds split bounds from targets."""
    coils = _coilset()
    length = CoilLength(coils, bounds=(10, 14), normalize=False)
    curv = CoilCurvature(coils, target=0.5, normalize=False)
    obj = ObjectiveFunction([length, curv])
    obj.build(verbose=0)
    x = obj.x(coils)

    ids = obj.row_ids(x)
    np.testing.assert_array_equal(ids, np.arange(obj.dim_f))

    lo, hi, tgt, isb = obj.scaled_bounds()
    nb = length.dim_f
    np.testing.assert_array_equal(isb, np.arange(obj.dim_f) < nb)
    np.testing.assert_allclose(lo[:nb], 10)
    np.testing.assert_allclose(hi[:nb], 14)
    assert np.all(np.isneginf(lo[nb:])) and np.all(np.isposinf(hi[nb:]))
    np.testing.assert_allclose(tgt[:nb], 0)
    np.testing.assert_allclose(tgt[nb:], obj.target_scaled[nb:])  # in scaled units

    # the same through a linear constraint projection
    lcp = LinearConstraintProjection(obj, ObjectiveFunction(FixCoilCurrent(coils)))
    lcp.build(verbose=0)
    xr = lcp.project(x)
    np.testing.assert_array_equal(lcp.row_ids(xr), np.arange(obj.dim_f))
    for a, b in zip(lcp.scaled_bounds(), (lo, hi, tgt, isb)):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- lsq_composite


def _jac(f):
    return jax.jit(jax.jacfwd(f))


@pytest.mark.unit
@pytest.mark.parametrize("hessian", ["gn", "secant"])
def test_lsq_composite_linear_hinge(hessian):
    """Two target rows plus one active bound row: closed-form minimizer.

    r = (x0 - 2, x1 - 1, max(0, x0 + x1 - 2)); the hinge is active at the solution.
    """

    def fun(x):
        return jnp.array([x[0], x[1], x[0] + x[1]])

    lo = np.array([-np.inf, -np.inf, -np.inf])
    hi = np.array([np.inf, np.inf, 2.0])
    tgt = np.array([2.0, 1.0, 0.0])
    isb = np.array([False, False, True])
    res = lsq_composite(
        fun,
        np.zeros(2),
        _jac(fun),
        lo,
        hi,
        tgt,
        isb,
        gtol=1e-12,
        ftol=0,
        xtol=0,
        verbose=0,
        options={"hessian": hessian},
    )
    assert res.success
    np.testing.assert_allclose(res.x, [5 / 3, 2 / 3], atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("hessian", ["gn", "secant"])
def test_lsq_composite_nonlinear(hessian):
    """Rosenbrock rows with a hinge on x0 > 0.5: closed-form minimizer.

    The first row vanishes on x1 = x0^2, where the cost is
    ((x0 - 1)^2 + (x0 - 0.5)^2) / 2, minimized at x0 = 0.75.
    """

    def fun(x):
        return jnp.array([10 * (x[1] - x[0] ** 2), x[0], x[0]])

    lo = np.array([-np.inf, -np.inf, -np.inf])
    hi = np.array([np.inf, np.inf, 0.5])
    tgt = np.array([0.0, 1.0, 0.0])
    isb = np.array([False, False, True])
    res = lsq_composite(
        fun,
        np.array([-1.0, 1.0]),
        _jac(fun),
        lo,
        hi,
        tgt,
        isb,
        gtol=1e-10,
        ftol=0,
        xtol=0,
        verbose=0,
        options={"hessian": hessian},
    )
    assert res.success
    np.testing.assert_allclose(res.x, [0.75, 0.5625], atol=1e-9)


@pytest.mark.unit
def test_paired_decrease():
    """Pairing by id equals the difference of the costs, rows in any order."""
    rng = np.random.default_rng(0)
    r, r_new = rng.normal(size=6), rng.normal(size=6)
    ids = np.array([0, 1, 2, 3, -1, 5])
    perm = np.array([3, 0, 5, 1, 4, 2])
    ids_new = ids[perm]  # same rows, new order, and the -1 slot never pairs
    d = paired_decrease(jnp.array(r), jnp.array(ids), jnp.array(r_new), ids_new)
    np.testing.assert_allclose(d, 0.5 * (r @ r - r_new @ r_new), rtol=1e-12)


# --------------------------------------------------------- lsq_auglag_composite


def _toy_al(con, con_row_ids=None, lb=-np.inf, ub=2.0):
    # minimize |x - (2, 1)|^2 / 2
    def fun(x):
        return x

    return lsq_auglag_composite(
        fun,
        np.zeros(2),
        _jac(fun),
        -np.inf,
        np.inf,
        np.array([2.0, 1.0]),
        np.zeros(2, dtype=bool),
        con,
        _jac(con),
        lb,
        ub,
        con_row_ids=con_row_ids,
        gtol=1e-8,
        ctol=1e-8,
        verbose=0,
    )


@pytest.mark.unit
def test_lsq_auglag_composite_kkt():
    """An active inequality and an equality: KKT point and multipliers."""

    def con(x):
        return jnp.array([x[0] + x[1], x[0] - x[1]])

    # x0 + x1 <= 2 (active), x0 - x1 = 1.2 -> x = (1.6, 0.4), and
    # (2, 1) - x = y1 (1, 1) + y2 (1, -1) -> y1 = 0.5, y2 = -0.1
    res = _toy_al(con, lb=np.array([-np.inf, 1.2]), ub=np.array([2.0, 1.2]))
    assert res.success
    np.testing.assert_allclose(res.x, [1.6, 0.4], atol=1e-7)
    np.testing.assert_allclose(np.abs(res.y), [0.5, 0.1], atol=1e-6)
    assert res.constr_violation < 1e-8


@pytest.mark.unit
def test_lsq_auglag_composite_reordered_rows():
    """Rows that change order between evaluations keep their multipliers by id."""
    # c <= (2, 1.2, 2): x = (1.2, 0.8), y = (0.2, 0.6, 0), the third row inactive
    A = jnp.array([[1.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    ub = np.array([2.0, 1.2, 2.0])

    def con_fixed(x):
        return A @ x

    # rows sorted by their slack, as a top-k selection would return them; every
    # row then needs the same bound, so compare c - ub against 0
    def con_sorted(x):
        return jnp.sort(A @ x - ub)[::-1]

    def ids_sorted(x):
        return np.argsort(-(np.asarray(A @ x) - ub), kind="stable")

    ref = _toy_al(con_fixed, ub=ub)
    res = _toy_al(con_sorted, con_row_ids=ids_sorted, ub=0.0)
    assert ref.success and res.success
    np.testing.assert_allclose(ref.x, [1.2, 0.8], atol=1e-7)
    np.testing.assert_allclose(ref.y, [0.2, 0.6, 0.0], atol=1e-6)
    np.testing.assert_allclose(res.x, ref.x, atol=1e-7)
    y = np.zeros(3)
    y[res.con_ids] = res.y
    np.testing.assert_allclose(y, ref.y, atol=1e-6)


@pytest.mark.unit
def test_row_values():
    """_RowValues: defaults for unseen ids, and the floor raises every value."""
    v = _RowValues(0.0)
    v.set([4, 7, -1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v.get([7, 4, 5, -1]), [2.0, 1.0, 0.0, 0.0])
    v.set([4], [0.0])  # set back to the default: dropped
    np.testing.assert_array_equal(v.keys, [7])
    mu = _RowValues(10.0, floor=True)
    mu.set([1, 2], [100.0, 5.0])  # 5 < floor is not kept
    np.testing.assert_array_equal(mu.get([1, 2]), [100.0, 10.0])
    mu.default = 200.0
    np.testing.assert_array_equal(mu.get([1, 2]), [200.0, 200.0])
