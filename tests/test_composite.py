"""Tests for the composite solvers and the row-identity interface they read."""

import numpy as np
import pytest

from desc.coils import CoilSet, FourierPlanarCoil
from desc.objectives import CoilCurvature, CoilLength, FixCoilCurrent, ObjectiveFunction
from desc.optimize import LinearConstraintProjection


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
