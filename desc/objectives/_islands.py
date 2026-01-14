"""Objectives for penalizing magnetic island formation."""

import warnings

import numpy as np

from desc.backend import jnp, vmap
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, errorif, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


def _find_all_rationals_up_to_denominator(max_denominator, iota_range=(-5, 5)):
    """Find all rational numbers up to a given denominator.

    Generates all rationals p/q with q <= max_denominator in a given range,
    sorted by "rationality" (lower denominators first).

    Parameters
    ----------
    max_denominator : int
        Maximum denominator q in p/q to consider
    iota_range : tuple
        (min, max) range of iota values to include rationals from

    Returns
    -------
    rationals : list of tuple
        List of (p, q) tuples representing p/q rationals, sorted by order (q)
    """
    from math import gcd

    iota_min, iota_max = iota_range
    candidates = []

    for q in range(1, max_denominator + 1):
        # Determine p range based on iota_range
        p_min = int(np.floor(iota_min * q))
        p_max = int(np.ceil(iota_max * q))
        for p in range(p_min, p_max + 1):
            val = p / q
            if iota_min <= val <= iota_max:
                # Check if this is in lowest terms
                if gcd(abs(p), q) == 1:
                    candidates.append((p, q, val))

    # Sort by denominator (lower = more rational), then by abs(numerator)
    candidates.sort(key=lambda x: (x[1], abs(x[0])))

    # Remove duplicates (same value)
    seen_values = set()
    unique_candidates = []
    for p, q, val in candidates:
        val_rounded = round(val, 10)
        if val_rounded not in seen_values:
            seen_values.add(val_rounded)
            unique_candidates.append((p, q))

    return unique_candidates


def _find_surfaces_for_rational_jax(iota_profile, iota_target, max_surfaces):
    """Find rho indices where iota equals target value (JAX-compatible).

    Finds local minima of |iota - target| to identify distinct surfaces
    where the rational is crossed. Returns fixed-size arrays with masking.

    Parameters
    ----------
    iota_profile : jnp.ndarray
        1D array of iota values on the rho grid
    iota_target : float
        Target iota value (the rational p/q)
    max_surfaces : int
        Maximum number of surfaces to return

    Returns
    -------
    surface_indices : jnp.ndarray
        Array of shape (max_surfaces,) with rho indices of surfaces.
        Padded with 0 if fewer surfaces exist.
    valid_mask : jnp.ndarray
        Boolean array of shape (max_surfaces,) indicating which surfaces are valid.
    """
    abs_diff = jnp.abs(iota_profile - iota_target)
    n = len(iota_profile)

    # Find local minima: points where abs_diff[i] <= neighbors
    # Use <= to handle flat regions at crossings
    is_local_min_interior = (abs_diff[1:-1] <= abs_diff[:-2]) & (
        abs_diff[1:-1] <= abs_diff[2:]
    )
    # Pad to match original length (boundaries can't be interior local minima)
    is_local_min = jnp.concatenate(
        [jnp.array([False]), is_local_min_interior, jnp.array([False])]
    )

    # Also include boundary points if they're close to target
    # (handles case where rational is at edge of iota range)
    is_local_min = is_local_min.at[0].set(abs_diff[0] <= abs_diff[1])
    is_local_min = is_local_min.at[-1].set(abs_diff[-1] <= abs_diff[-2])

    # Create scores: local minima get their abs_diff, others get infinity
    scores = jnp.where(is_local_min, abs_diff, jnp.inf)

    # Get indices sorted by score (smallest abs_diff first)
    sorted_indices = jnp.argsort(scores)

    # Take top max_surfaces indices
    surface_indices = sorted_indices[:max_surfaces]

    # Validity mask: surface is valid if it's a local minimum with finite score
    # and the target is reasonably close to the iota range
    iota_range = jnp.max(iota_profile) - jnp.min(iota_profile)
    threshold = 0.5 * jnp.maximum(iota_range, 0.01)  # avoid zero threshold
    valid_mask = scores[surface_indices] < threshold

    return surface_indices, valid_mask


class MagneticIslandPenalty(_Objective):
    """Penalize magnetic island formation by targeting resonant Fourier modes.

    This objective computes div(J_perp) on rational surfaces and extracts the
    resonant Fourier modes that would drive magnetic island formation. Since DESC
    assumes nested flux surfaces, islands cannot form directly, but this objective
    penalizes the configurations that would lead to island formation if the nested
    surface constraint were relaxed.

    On a rational surface with iota = p/q (in lowest terms), the resonant modes
    are (m, n) = (r*q, -r*p) for integer r. These modes correspond to perturbations
    that don't average to zero along field lines and can drive island formation.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    max_denominator : int, optional
        Maximum denominator q in p/q rationals to consider. Lower values focus
        on the most dangerous (lowest order) rational surfaces. Default = 10.
    num_surfaces : int, optional
        Number of rational surfaces to penalize. Default = 5.
    max_surfaces_per_rational : int, optional
        Maximum number of surfaces per rational (for reversed shear profiles
        with multiple surfaces at the same iota). Default = 2.
    M_island : int, optional
        Poloidal resolution for Fourier transform. Default = eq.M_grid.
    N_island : int, optional
        Toroidal resolution for Fourier transform. Default = eq.N_grid.
    r_max : int, optional
        Maximum |r| for extracting resonant modes (r*q, -r*p). Default = 3.
    num_rho_interp : int, optional
        Number of points for iota interpolation grid. Default = 100.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "rtz"
    _units = "(A/m^4)"
    _print_value_fmt = "Magnetic island penalty: "
    _io_attrs_ = _Objective._io_attrs_ + [
        "_max_denominator",
        "_num_surfaces",
        "_max_surfaces_per_rational",
        "_M_island",
        "_N_island",
        "_r_max",
        "_num_rho_interp",
    ]
    # Mark grid dimensions as static to prevent JAX from tracing them during JIT
    _static_attrs = _Objective._static_attrs + [
        "_num_theta",
        "_num_zeta",
        "_num_rho_grid",
        "_NFP",
        "_max_denominator",
        "_num_surfaces",
        "_max_surfaces_per_rational",
        "_r_max",
        "_num_candidate_rationals",
    ]

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None, ## ONLY IMPLEMENTED FOR OBJECTIVES INVARIANT UNDER PADDING
        deriv_mode="auto",
        grid=None,
        max_denominator=10,
        num_surfaces=5,
        max_surfaces_per_rational=2,
        M_island=None,
        N_island=None,
        r_max=3,
        num_rho_interp=100,
        name="magnetic island",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._max_denominator = max_denominator
        self._num_surfaces = num_surfaces
        self._max_surfaces_per_rational = max_surfaces_per_rational
        self._M_island = M_island
        self._N_island = N_island
        self._r_max = r_max
        self._num_rho_interp = num_rho_interp
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]

        # Set default resolutions
        M_island = self._M_island or eq.M_grid
        N_island = self._N_island or eq.N_grid

        # Create base 2D grid for (theta, zeta) evaluation
        # We'll set rho values dynamically in compute
        if self._grid is None:
            # LinearGrid with single rho value (will be replaced dynamically)
            # Use enough points for FFT: need 2*M+1 and 2*N+1 at minimum
            num_theta = 2 * M_island + 1
            num_zeta = 2 * N_island + 1
            grid = LinearGrid(
                rho=np.array([0.5]),  # placeholder
                M=num_theta // 2,
                N=num_zeta // 2,
                NFP=eq.NFP,
                sym=False,
            )
        else:
            grid = self._grid
            num_theta = grid.num_theta
            num_zeta = grid.num_zeta

        errorif(grid.sym, ValueError, "MagneticIslandPenalty grid must be non-symmetric")

        # Store grid parameters
        self._num_theta = num_theta
        self._num_zeta = num_zeta
        self._M_island = M_island
        self._N_island = N_island

        # Data keys needed for div(J_perp) computation
        self._data_keys = ["div(J_perp)", "iota"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        # Create dense rho grid for iota interpolation
        # Start at small positive rho to avoid axis singularity in div(J_perp)
        rho_interp = np.linspace(0.01, 1, self._num_rho_interp)

        # Create a 3D grid for computing div(J_perp) on all candidate surfaces
        # We precompute for a fixed set of rho values and select during compute
        # Maximum number of surfaces = num_surfaces * max_surfaces_per_rational
        max_total_surfaces = self._num_surfaces * self._max_surfaces_per_rational

        # Create grid with multiple rho surfaces for precomputation
        full_grid = LinearGrid(
            rho=rho_interp,
            M=num_theta // 2,
            N=num_zeta // 2,
            NFP=eq.NFP,
            sym=False,
        )

        profiles = get_profiles(self._data_keys, obj=eq, grid=full_grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=full_grid)

        # Store grid dimensions as instance attributes for JAX JIT compatibility
        # (constants dict values become traced and can't be used in reshape)
        self._num_rho_grid = len(rho_interp)
        self._NFP = eq.NFP

        # Precompute ALL candidate rationals up to max_denominator
        # These are filtered dynamically at compute time based on current iota range
        # Use a generous iota range to capture all possible rationals
        all_rationals = _find_all_rationals_up_to_denominator(
            self._max_denominator, iota_range=(-5, 5)
        )

        if verbose > 0:
            print(
                f"  Precomputed {len(all_rationals)} candidate rationals "
                f"(up to denominator {self._max_denominator})"
            )

        # Precompute FFT mode indices for each rational
        # For rational p/q, resonant modes are (m, n) = (r*q, -r*p*NFP) for r != 0
        rational_iota_vals = []
        fft_mode_indices_m = []
        fft_mode_indices_n = []
        rational_p_vals = []
        rational_q_vals = []

        for p, q in all_rationals:
            rational_iota_vals.append(p / q)
            rational_p_vals.append(p)
            rational_q_vals.append(q)

            m_indices = []
            n_indices = []
            for r in range(-self._r_max, self._r_max + 1):
                if r == 0:
                    continue
                m = r * q
                n = -r * p * eq.NFP
                # Wrap indices to FFT array bounds
                m_idx = m % num_theta
                n_idx = n % num_zeta
                m_indices.append(m_idx)
                n_indices.append(n_idx)

            fft_mode_indices_m.append(m_indices)
            fft_mode_indices_n.append(n_indices)

        # Store as arrays for JAX compatibility
        self._num_candidate_rationals = len(all_rationals)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "rho_interp": jnp.array(rho_interp),
            "rational_iota_vals": jnp.array(rational_iota_vals),
            "rational_p_vals": jnp.array(rational_p_vals),
            "rational_q_vals": jnp.array(rational_q_vals),
            "fft_mode_indices_m": jnp.array(fft_mode_indices_m),
            "fft_mode_indices_n": jnp.array(fft_mode_indices_n),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        # Output dimension:
        # - num_surfaces rationals (dynamically selected from candidates)
        # - max_surfaces_per_rational surfaces per rational (for reversed shear)
        # - 2 * r_max modes per surface
        # - 2 values (real/imag) per mode
        num_modes_per_surface = 2 * self._r_max
        self._dim_f = (
            self._num_surfaces
            * self._max_surfaces_per_rational
            * num_modes_per_surface
            * 2
        )

        # Set quad_weights explicitly since our output is FFT coefficients, not grid values
        # Use uniform weights since all Fourier modes should be weighted equally
        self._constants["quad_weights"] = jnp.ones(self._dim_f)

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # Normalize by J/V ~ current density scale
            self._normalization = scales["J"] / scales["V"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute magnetic island penalty.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Resonant Fourier mode coefficients of div(J_perp) on rational surfaces.

        """
        if constants is None:
            constants = self.constants

        # Compute iota and div(J_perp) on the full interpolation grid
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )

        # Use precomputed rational values and FFT indices from constants
        rational_iota_vals = constants["rational_iota_vals"]
        rational_q_vals = constants["rational_q_vals"]
        fft_mode_indices_m = constants["fft_mode_indices_m"]
        fft_mode_indices_n = constants["fft_mode_indices_n"]

        # Use instance attributes for grid dimensions (JAX JIT compatible)
        num_theta = self._num_theta
        num_zeta = self._num_zeta
        num_rho = self._num_rho_grid
        num_surfaces = self._num_surfaces
        max_surfaces_per_rational = self._max_surfaces_per_rational
        num_modes_per_surface = 2 * self._r_max

        # Get iota values - compress to get one per rho
        # iota is constant on flux surfaces
        iota_full = data["iota"]
        # Reshape to (num_rho, num_theta, num_zeta) and take first point per surface
        iota_3d = iota_full.reshape((num_rho, num_theta, num_zeta))
        iota_profile = iota_3d[:, 0, 0]

        # Get div(J_perp) reshaped for FFT
        div_J_perp_full = data["div(J_perp)"]
        div_J_perp_3d = div_J_perp_full.reshape((num_rho, num_theta, num_zeta))

        # Dynamic filtering: determine which rationals are in current iota range
        iota_min = jnp.min(iota_profile)
        iota_max = jnp.max(iota_profile)

        # A rational is "in range" if it falls within [iota_min, iota_max]
        in_range_mask = (rational_iota_vals >= iota_min) & (
            rational_iota_vals <= iota_max
        )

        # Create a score for sorting: in-range rationals get their denominator,
        # out-of-range get a large value (so they sort to the end)
        # This way we select the lowest-order rationals that are in range
        sort_score = jnp.where(in_range_mask, rational_q_vals, 1e6)

        # Get indices of top num_surfaces rationals (sorted by score)
        # Since rationals are already sorted by denominator in constants,
        # we can use argsort on the score to get in-range ones first
        sorted_rational_indices = jnp.argsort(sort_score)
        selected_rational_indices = sorted_rational_indices[:num_surfaces]

        # Gather the data for selected rationals (vectorized indexing)
        selected_iota_targets = rational_iota_vals[selected_rational_indices]
        selected_in_range = in_range_mask[selected_rational_indices]
        selected_m_indices = fft_mode_indices_m[selected_rational_indices]
        selected_n_indices = fft_mode_indices_n[selected_rational_indices]

        # Vectorized surface finding for all selected rationals at once
        # vmap over the rational dimension
        def find_surfaces_single(iota_target):
            return _find_surfaces_for_rational_jax(
                iota_profile, iota_target, max_surfaces_per_rational
            )

        # Shape: (num_surfaces, max_surfaces_per_rational)
        all_surface_indices, all_valid_masks = vmap(find_surfaces_single)(
            selected_iota_targets
        )

        # Combined validity: rational in range AND surface is valid
        # Shape: (num_surfaces, max_surfaces_per_rational)
        combined_valid = selected_in_range[:, None] & all_valid_masks

        # Vectorized FFT and mode extraction
        # For each (rational, surface) pair, we need to:
        # 1. Get the rho index
        # 2. Extract the 2D surface data
        # 3. Compute FFT
        # 4. Extract the resonant modes

        def process_single_surface(rho_idx, m_indices, n_indices, is_valid):
            """Process a single surface: FFT and extract modes."""
            # Get div(J_perp) on this surface
            div_J_2d = div_J_perp_3d[rho_idx, :, :]

            # Compute 2D FFT
            fft_result = jnp.fft.fft2(div_J_2d)

            # Extract all resonant modes at once using advanced indexing
            # m_indices and n_indices have shape (num_modes_per_surface,)
            coeffs = fft_result[m_indices, n_indices]

            # Apply validity mask: zero out if surface is invalid
            coeffs = jnp.where(is_valid, coeffs, 0.0 + 0.0j)

            # Stack real and imaginary parts: shape (num_modes_per_surface * 2,)
            return jnp.stack([jnp.real(coeffs), jnp.imag(coeffs)], axis=-1).flatten()

        def process_rational(surface_indices, m_indices, n_indices, valid_mask):
            """Process all surfaces for a single rational."""
            # vmap over the max_surfaces_per_rational dimension
            return vmap(
                lambda rho_idx, is_valid: process_single_surface(
                    rho_idx, m_indices, n_indices, is_valid
                )
            )(surface_indices, valid_mask)

        # vmap over all selected rationals
        # all_surface_indices: (num_surfaces, max_surfaces_per_rational)
        # selected_m_indices: (num_surfaces, num_modes_per_surface)
        # combined_valid: (num_surfaces, max_surfaces_per_rational)
        all_modes = vmap(process_rational)(
            all_surface_indices, selected_m_indices, selected_n_indices, combined_valid
        )

        # Shape: (num_surfaces, max_surfaces_per_rational, num_modes_per_surface * 2)
        # Flatten to 1D for output
        return all_modes.flatten()

    @property
    def max_denominator(self):
        """int: Maximum denominator q in p/q rationals."""
        return self._max_denominator

    @max_denominator.setter
    def max_denominator(self, value):
        self._max_denominator = int(value)
        self._built = False

    @property
    def num_surfaces(self):
        """int: Number of rational surfaces to penalize."""
        return self._num_surfaces

    @num_surfaces.setter
    def num_surfaces(self, value):
        self._num_surfaces = int(value)
        self._built = False

    @property
    def r_max(self):
        """int: Maximum |r| for resonant mode extraction."""
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        self._r_max = int(value)
        self._built = False
