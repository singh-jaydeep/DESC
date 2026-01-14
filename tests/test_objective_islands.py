"""Tests for magnetic island penalty objective."""

import warnings

import numpy as np
import pytest

from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import MagneticIslandPenalty, ObjectiveFunction
from desc.objectives._islands import (
    _find_all_rationals_up_to_denominator,
    _find_surfaces_for_rational_jax,
)
from desc.profiles import PowerSeriesProfile


def _make_test_eq():
    """Create a test equilibrium, suppressing expected warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eq = Equilibrium(
            M=2,
            N=1,
            L=2,
            iota=PowerSeriesProfile([0.4, 0.1]),  # iota from 0.4 to 0.5
        )
    return eq


class TestMagneticIslandHelpers:
    """Test helper functions for magnetic island objective."""

    @pytest.mark.unit
    def test_find_all_rationals_basic(self):
        """Test finding all rationals up to a denominator."""
        rationals = _find_all_rationals_up_to_denominator(5, iota_range=(0, 1))

        # Should include basic fractions like 0/1, 1/2, 1/3, 2/3, 1/4, 3/4, 1/5, etc.
        assert (0, 1) in rationals
        assert (1, 2) in rationals
        assert (1, 3) in rationals
        assert (2, 3) in rationals

        # Check that all returned rationals are in lowest terms
        from math import gcd

        for p, q in rationals:
            assert gcd(abs(p), q) == 1

    @pytest.mark.unit
    def test_find_all_rationals_ordering(self):
        """Test that rationals are ordered by denominator (most rational first)."""
        rationals = _find_all_rationals_up_to_denominator(10, iota_range=(0, 1))

        # Should start with lowest denominators
        denominators = [q for p, q in rationals]
        # First few should have small denominators
        assert denominators[0] <= 2
        assert denominators[1] <= 2

    @pytest.mark.unit
    def test_find_surfaces_monotonic_iota(self):
        """Test finding surfaces for a monotonic iota profile."""
        # Linear iota from 0.3 to 0.7
        iota_profile = jnp.array(0.3 + 0.4 * np.linspace(0, 1, 100))

        # Find surface where iota = 0.5
        indices, valid = _find_surfaces_for_rational_jax(iota_profile, 0.5, 2)

        assert len(indices) == 2
        assert len(valid) == 2
        # First surface should be valid and near index 50 (rho = 0.5)
        assert valid[0]
        assert 45 <= indices[0] <= 55

    @pytest.mark.unit
    def test_find_surfaces_reversed_shear(self):
        """Test finding surfaces for a non-monotonic iota profile (reversed shear)."""
        rho = np.linspace(0, 1, 100)
        # Parabolic iota: max at rho=0.5, iota = 0.3 + 0.4*rho - 0.4*rho^2
        # At rho=0: iota=0.3, at rho=0.5: iota=0.4, at rho=1: iota=0.3
        iota_profile = jnp.array(0.3 + 0.4 * rho - 0.4 * rho**2)

        # Find surfaces where iota = 0.35 (should have two crossings)
        indices, valid = _find_surfaces_for_rational_jax(iota_profile, 0.35, 3)

        assert len(indices) == 3
        # Should find at least 2 valid surfaces for reversed shear profile
        num_valid = jnp.sum(valid)
        assert num_valid >= 2

    @pytest.mark.unit
    def test_find_surfaces_out_of_range(self):
        """Test behavior when target iota is outside the range."""
        # Linear iota from 0.3 to 0.7
        iota_profile = jnp.array(0.3 + 0.4 * np.linspace(0, 1, 100))

        # Target far outside range should return invalid surfaces
        indices, valid = _find_surfaces_for_rational_jax(iota_profile, 1.5, 2)

        # Surfaces should be marked as invalid since target is far from range
        assert not jnp.any(valid)


class TestMagneticIslandPenalty:
    """Test MagneticIslandPenalty objective."""

    @pytest.mark.unit
    def test_build(self):
        """Test that the objective builds correctly."""
        eq = _make_test_eq()

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=5,
            num_surfaces=2,
            max_surfaces_per_rational=1,
            r_max=2,
        )
        obj.build(verbose=0)

        assert obj.built
        assert obj.dim_f > 0

    @pytest.mark.unit
    def test_compute(self):
        """Test that compute returns correct shape."""
        eq = _make_test_eq()

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=5,
            num_surfaces=2,
            max_surfaces_per_rational=1,
            r_max=2,
        )
        obj.build(verbose=0)

        result = obj.compute(eq.params_dict)

        assert result.shape == (obj.dim_f,)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.unit
    def test_in_objective_function(self):
        """Test that objective works within ObjectiveFunction."""
        eq = _make_test_eq()

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=5,
            num_surfaces=2,
        )

        objective = ObjectiveFunction(obj)
        objective.build(verbose=0)

        # Should be able to compute scalar value
        scalar = objective.compute_scalar(objective.x(eq))
        assert jnp.isfinite(scalar)

    @pytest.mark.unit
    def test_properties(self):
        """Test objective properties."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            eq = Equilibrium(M=2, N=1, L=2)

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=8,
            num_surfaces=3,
            r_max=4,
        )

        assert obj.max_denominator == 8
        assert obj.num_surfaces == 3
        assert obj.r_max == 4

        # Test setters
        obj.max_denominator = 10
        assert obj.max_denominator == 10
        assert not obj.built  # Should require rebuild

    @pytest.mark.unit
    def test_normalization(self):
        """Test that normalization works correctly."""
        eq = _make_test_eq()

        obj_normalized = MagneticIslandPenalty(eq, normalize=True)
        obj_normalized.build(verbose=0)

        obj_unnormalized = MagneticIslandPenalty(eq, normalize=False)
        obj_unnormalized.build(verbose=0)

        # Both should compute without error
        result_norm = obj_normalized.compute(eq.params_dict)
        result_unnorm = obj_unnormalized.compute(eq.params_dict)

        assert jnp.all(jnp.isfinite(result_norm))
        assert jnp.all(jnp.isfinite(result_unnorm))

    @pytest.mark.unit
    def test_dynamic_rational_filtering(self):
        """Test that rationals are dynamically filtered based on current iota range."""
        eq = _make_test_eq()  # iota from 0.4 to 0.5

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=10,
            num_surfaces=3,
            max_surfaces_per_rational=1,
            r_max=2,
        )
        obj.build(verbose=0)

        # Check that many rationals are precomputed (not just those in initial range)
        assert obj._num_candidate_rationals > 10  # Should have many candidates

        # Compute should still work - rationals outside range are masked
        result = obj.compute(eq.params_dict)
        assert result.shape == (obj.dim_f,)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.unit
    def test_multiple_surfaces_per_rational(self):
        """Test that multiple surfaces are found for each rational."""
        eq = _make_test_eq()

        # Test with max_surfaces_per_rational > 1
        obj = MagneticIslandPenalty(
            eq,
            max_denominator=5,
            num_surfaces=2,
            max_surfaces_per_rational=3,
            r_max=2,
        )
        obj.build(verbose=0)

        # Output dimension should account for multiple surfaces
        num_modes_per_surface = 2 * 2  # 2 * r_max
        expected_dim = 2 * 3 * num_modes_per_surface * 2  # surfaces * per_rational * modes * real/imag
        assert obj.dim_f == expected_dim

        result = obj.compute(eq.params_dict)
        assert result.shape == (obj.dim_f,)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.unit
    def test_reversed_shear_profile(self):
        """Test behavior with a reversed shear iota profile."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Parabolic iota profile: peaks in the middle (reversed shear)
            # iota = 0.3 + 0.2*rho - 0.15*rho^2 gives max around rho=0.67
            eq = Equilibrium(
                M=2,
                N=1,
                L=2,
                iota=PowerSeriesProfile([0.3, 0.2, -0.15]),
            )

        obj = MagneticIslandPenalty(
            eq,
            max_denominator=5,
            num_surfaces=2,
            max_surfaces_per_rational=2,  # Allow finding 2 surfaces per rational
            r_max=2,
        )
        obj.build(verbose=0)

        result = obj.compute(eq.params_dict)
        assert result.shape == (obj.dim_f,)
        assert jnp.all(jnp.isfinite(result))


class TestDivJPerpCompute:
    """Test the div(J_perp) compute function."""

    @pytest.mark.unit
    def test_div_J_perp_computes(self):
        """Test that div(J_perp) can be computed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            eq = Equilibrium(M=2, N=1, L=2)
        # Use grid that avoids axis to prevent singularity
        grid = LinearGrid(rho=np.array([0.5]), M=4, N=2, NFP=eq.NFP)

        data = eq.compute("div(J_perp)", grid=grid)

        assert "div(J_perp)" in data
        assert data["div(J_perp)"].shape == (grid.num_nodes,)
        assert jnp.all(jnp.isfinite(data["div(J_perp)"]))

    @pytest.mark.unit
    def test_div_J_perp_units(self):
        """Test that div(J_perp) has correct units in data_index."""
        from desc.compute import data_index

        parameterization = "desc.equilibrium.equilibrium.Equilibrium"
        info = data_index[parameterization]["div(J_perp)"]

        assert info["units"] == "A \\cdot m^{-4}"
        assert info["coordinates"] == "rtz"
