import numbers
import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jax, jnp
from desc.backend import tree_broadcast as jax_tree_broadcast
from desc.backend import tree_flatten, tree_leaves, tree_map, tree_unflatten
from desc.batching import vmap_chunked
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, _Grid
from desc.integrals import compute_B_plasma
from desc.utils import (
    Timer,
    broadcast_tree,
    copy_rpz_periods,
    errorif,
    rpz2xyz,
    safediv,
    safenorm,
    setdefault,
    warnif,
)

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import softmax, softmin


class _CoilObjective(_Objective):
    """Base class for calculating coil objectives.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil for which the data keys will be optimized.
    data_keys : list of str
        data keys that will be optimized when this class is inherited.
    grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate at.
        If a list, must have the same structure as coil.
    target : float, list, optional
        Target values for the coil objective.
        If a float, target is applied to all coils.
        If a list, must have the same structure as coil.
    bounds : tuple, optional
        Upper and lower bounds for the coil objective.
        If used, should consist of a tuple (-,-), with
        each entry a float or list satisfying requirements
        of target. Cannot be used with target.
    weight: float, list, optional
        Weight for the coil objective during optimization.
        Default is a uniform weight. If a list, must have
        the same structure as coil, and consist of non-negative
        floats. Set weight to zero to exclude given coils from optimization.

    Subclasses must define a static attribute "_broadcast_input." Equals
    "coil" if the objective returns a single scalar per coil, and "node"
    if it returns a scalar at every grid point. It is case-insensitive.
    To be compatible with masking, compute function should apply the mask
    self._coilset_tree["objective_mask"] before returning data.
    """

    __doc__ = __doc__.rstrip() + collect_docs(coil=True)
    _static_attrs = _Objective._static_attrs + ["_coilset_tree", "_broadcast_input"]

    def __init__(
        self,
        coil,
        data_keys,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name=None,
        jac_chunk_size=None,
    ):
        self._grid = grid
        self._data_keys = data_keys
        self._normalize = normalize

        super().__init__(
            things=[coil],
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

    def build(self, use_jit=True, verbose=1):  # noqa:C901
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        # local import to avoid circular import
        from desc.coils import CoilSet, MixedCoilSet, _Coil

        def _is_single_coil(c):
            return isinstance(c, _Coil) and not isinstance(c, CoilSet)

        def _prune_coilset_tree(coilset):
            """Remove extra members from CoilSets (but not MixedCoilSets)."""
            if isinstance(coilset, list) or isinstance(coilset, MixedCoilSet):
                return [_prune_coilset_tree(c) for c in coilset]
            elif isinstance(coilset, CoilSet):
                # CoilSet only uses a single grid/transform for all coils
                return _prune_coilset_tree(coilset.coils[0])
            else:
                return coilset  # single coil

        def _build_coilset_tree():
            """Unpacks the input coilset, builds coilset tree and mask.

            Sets self._coilset_tree, a dict. self._coilset_tree["coils"] contains
            a nested list of 0s representing individual coils and the coilsets
            to which they belong. Similarly, self._coilset_tree["nodes"] lists
            the grid nodes associated with each coil. self._coilset_tree["coilset_mask"]
            contains the indices in [0,self._num_coils-1] for which the corresponding
            weight is positive. self._coilset_tree["objective_mask"] contains the
            indices in [0,self._dim_f-1] for which the corresponding weight is
            positive. If all weights are positive (i.e. no masking needed), contains
            default slice(None).
            """
            # Local import to avoid circular import
            from desc.coils import CoilSet, MixedCoilSet, _Coil

            tol = 1e-12

            def expand(t, idx=0):
                if isinstance(t, MixedCoilSet):
                    return expand(t.coils, idx)
                if isinstance(t, CoilSet):
                    return (
                        [0] * len(t.coils),
                        [grid[idx].num_nodes] * len(t.coils),
                        idx + len(t.coils),
                    )
                if isinstance(t, _Coil):
                    return 0, grid[idx].num_nodes, idx + 1
                if isinstance(t, list):
                    l_coils = []
                    l_nodes = []
                    idx_curr = idx
                    for i in range(len(t)):
                        a_coils, a_nodes, idx_curr = expand(t[i], idx_curr)
                        l_coils += [a_coils]
                        l_nodes += [a_nodes]
                    return l_coils, l_nodes, idx_curr
                return t, idx

            tree = expand(coil)
            self._coilset_tree = {
                "coils": tree[0],
                "nodes": tree[1],
                "coilset_mask": np.arange(self._num_coils),
                "objective_mask": slice(None),
            }
            if np.any(
                np.isclose([w for w in tree_leaves(self._weight)], 0.0, atol=tol)
            ):
                coilset_mask = self._coilset_broadcast(self._weight, target="coil")
                objective_mask = self._coilset_broadcast(
                    self._weight, self._broadcast_input
                )
                self._coilset_tree["coilset_mask"] = np.nonzero(coilset_mask > tol)[0]
                self._coilset_tree["objective_mask"] = np.nonzero(objective_mask > tol)[
                    0
                ]

        coil = self.things[0]
        grid = self._grid

        # get individual coils from coilset
        coils, structure = tree_flatten(coil, is_leaf=_is_single_coil)
        for c in coils:
            errorif(
                not isinstance(c, _Coil),
                TypeError,
                f"Expected object of type Coil, got {type(c)}",
            )
        self._num_coils = len(coils)

        # map grid to list of length coils
        if grid is None:
            grid = []
            for c in coils:
                grid.append(LinearGrid(N=2 * c.N * getattr(c, "NFP", 1) + 5))
        if isinstance(grid, numbers.Integral):
            grid = LinearGrid(N=self._grid)
        if isinstance(grid, _Grid):
            grid = [grid] * self._num_coils
        if isinstance(grid, list):
            grid = tree_leaves(grid, is_leaf=lambda g: isinstance(g, _Grid))

        errorif(
            len(grid) != len(coils),
            ValueError,
            "grid input must be broadcastable to the coil structure.",
        )
        errorif(
            np.any([g.num_rho > 1 or g.num_theta > 1 for g in grid]),
            ValueError,
            "Only use toroidal resolution for coil grids.",
        )

        _build_coilset_tree()
        quad_weights = np.sqrt(
            np.concatenate([g.spacing[:, 2] for g in grid])[
                self._coilset_tree["objective_mask"]
            ]
        )

        if self._broadcast_input.lower() == "node":
            grid_nodes_unmasked = [
                grid[i].num_nodes for i in self._coilset_tree["coilset_mask"]
            ]
            self._dim_f = np.sum(grid_nodes_unmasked)
        else:
            coils_unmasked = np.ones(self._num_coils)[
                self._coilset_tree["coilset_mask"]
            ]
            self._dim_f = len(coils_unmasked)

        # map grid to the same structure as coil and then remove unnecessary members
        grid = tree_unflatten(structure, grid)
        grid = _prune_coilset_tree(grid)
        coil = _prune_coilset_tree(coil)

        self._weight = self._coilset_broadcast(self._weight, self._broadcast_input)
        if self._bounds is not None:
            self._bounds = (
                self._coilset_broadcast(self._bounds[0], self._broadcast_input),
                self._coilset_broadcast(self._bounds[1], self._broadcast_input),
            )
        elif self._target is not None:
            self._target = self._coilset_broadcast(self._target, self._broadcast_input)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        transforms = tree_map(
            lambda c, g: get_transforms(self._data_keys, obj=c, grid=g),
            coil,
            grid,
            is_leaf=lambda x: _is_single_coil(x) or isinstance(x, _Grid),
        )

        self._grid = grid
        self._constants = {"transforms": transforms, "quad_weights": quad_weights}

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._scales = [compute_scaling_factors(coil) for coil in coils]

    def compute(self, params, constants=None):
        """Compute data of coil for given data key.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : float or array of floats
            Coil objective value(s).

        """
        constants = self._get_deprecated_constants(constants)

        coil = self.things[0]
        data = coil.compute(
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            grid=self._grid,
            # Use the grid this objective was BUILT on. `Curve.compute` otherwise
            # recomputes any 0d quantity (`coordinates == ""`) on its own
            # `LinearGrid(N=2*N+5)`, because its escape hatch requires
            # `isinstance(grid, LinearGrid)` -- so an explicitly chosen non-LinearGrid
            # is silently discarded, and the result no longer matches the transforms
            # built here. `length` is the only 0d key any _CoilObjective requests, so
            # this changes CoilLength alone.
            #
            # It matters: on a C0 piecewise-arc coil, a composite-GL grid aligned to
            # the hinges gives the length to 9e-8 with 120 nodes, while a uniform
            # LinearGrid converges O(h) -- 7e-5 at 12801 nodes -- AND its error jitters
            # as the hinges move across fixed nodes, which is non-smooth in the
            # optimization variables and collapses the trust region.
            override_grid=False,
        )
        return data

    @_Objective.bounds.setter
    def bounds(self, bounds):
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        if bounds is not None:
            self._bounds = (
                self._coilset_broadcast(bounds[0], self._broadcast_input),
                self._coilset_broadcast(bounds[1], self._broadcast_input),
            )
        else:
            self._bounds = None
        self._check_dimensions()

    @_Objective.target.setter
    def target(self, target):
        if target is not None:
            self._target = self._coilset_broadcast(target, self._broadcast_input)
        else:
            self._target = None
        self._check_dimensions()

    @_Objective.weight.setter
    def weight(self, weight):
        assert np.all(np.asarray(tree_leaves(weight)) >= 0)
        self._weight = weight
        # objective should be rebuilt to account for masking
        self._built = False

    def _coilset_broadcast(self, x, target="coil"):
        """Broadcast an array to dimensions consistent with "target".

        Parameters
        ----------
        x : float or list[float]
            Must be broadcastable to the structure of self._things[0].
        target: str, optional
            Optional string taking values "coil" or "node". Defaults to "coil".

        Returns
        -------
        arr: float or list[float]
            Float inputs are returned unchanged, and list inputs are
            expanded to size self._dim_f.
        """
        target = target.lower()
        assert target in ["node", "coil"]

        if isinstance(x, (np.ndarray, jnp.ndarray)):
            x = x.tolist()

        # No need to broadcast if input is a scalar
        arr_flat = tree_leaves(x)
        if len(arr_flat) == 1:
            return np.atleast_1d(arr_flat[0])

        arr = jax_tree_broadcast(x, self._coilset_tree["coils"])
        if target == "node":
            arr = tree_map(lambda a, b: [a] * b, arr, self._coilset_tree["nodes"])
        arr, _ = tree_flatten(arr)
        return np.asarray(arr)[self._coilset_tree["objective_mask"]]


class CoilLength(_CoilObjective):
    """Coil length.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=2*np.pi``.",
        bounds_default="``target=2*np.pi``.",
        coil=True,
    )

    _scalar = False  # Not always a scalar, if a coilset is passed in
    _units = "(m)"
    _print_value_fmt = "Coil length: "
    _broadcast_input = "coil"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil length",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 2 * np.pi

        super().__init__(
            coil,
            ["length"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1

        if self._normalize:
            self._normalization = np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil length.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Coil length.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array([dat["length"] for dat in data])
        return out[self._coilset_tree["objective_mask"]]


class CoilCurvature(_CoilObjective):
    """Coil curvature.

    Targets the magnitude of the local curvature at each grid node for each coil --
    the reciprocal of the local radius of curvature. Larger values mean a more tightly
    bent coil; values near 0 indicate straighter sections. Bounding it above is the
    usual way to impose a minimum bend radius.

    This uses the unsigned ``|curvature|``. The signed ``curvature`` quantity carries a
    convex/concave convention set by ``sign(dot(r, frenet_normal))``, which is a step
    function: it jumps by twice the full curvature magnitude wherever the curve's
    normal turns perpendicular to the line back to its center. That locus is unrelated
    to where the curvature vanishes and moves when distant parts of the curve move, so
    signed curvature is not usable as an optimization target or bound -- a trust region
    cannot shrink past the discontinuity.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``
    signed : bool, optional
        Use the signed ``curvature`` instead of ``|curvature|`` (master's behavior).
        Default False; see above for why signed curvature should not be bounded.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0,1).``",
        bounds_default="``bounds=(0,1).``",
        coil=True,
    )

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil curvature: "
    _broadcast_input = "node"
    _static_attrs = _CoilObjective._static_attrs + ["_key"]

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil curvature",
        jac_chunk_size=None,
        signed=False,
    ):
        if target is None and bounds is None:
            bounds = (0, 1)
        self._key = "curvature" if signed else "|curvature|"

        super().__init__(
            coil,
            [self._key],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil curvature.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            1D array of coil curvature values.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.concatenate([dat[self._key] for dat in data])
        return out[self._coilset_tree["objective_mask"]]


class CoilTorsion(_CoilObjective):
    """Coil torsion.

    Targets the local torsion value at each grid node for each coil. Indicative of how
    non-planar the coil is (a torsion value of 0 means the coil is perfectly planar).

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil torsion: "
    _broadcast_input = "node"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil torsion",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coil,
            ["torsion"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil torsion.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Coil torsion.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.concatenate([dat["torsion"] for dat in data])
        return out[self._coilset_tree["objective_mask"]]


class CoilCurrentLength(CoilLength):
    """Coil current length.

    Targets the coil current length, i.e. current * length for each coil.
    Useful for approximating HTS cost.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _scalar = False
    _units = "(A*m)"
    _print_value_fmt = "Coil current length: "
    _broadcast_input = "coil"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil current length",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coil,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1

        if self._normalize:
            mean_length = np.mean([scale["a"] for scale in self._scales])
            params = tree_leaves(
                self.things[0].params_dict, is_leaf=lambda x: isinstance(x, dict)
            )
            mean_current = np.mean([np.abs(param["current"]) for param in params])
            mean_current = np.max((mean_current, 1))
            self._normalization = mean_current * mean_length

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil current length (current * length).

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats

        """
        lengths = super().compute(params, constants=constants)
        params = tree_leaves(params, is_leaf=lambda x: isinstance(x, dict))
        currents = jnp.concatenate([param["current"] for param in params])
        out = jnp.atleast_1d(lengths * currents[self._coilset_tree["objective_mask"]])
        return out


class CoilIntegratedCurvature(_CoilObjective):
    """Coil integrated curvature.

    If a curve is convex, then the following condition must be true: ∫ κ ||∂ₛx|| ds = 2π
    where κ is the scalar (unsigned) curvature, ∂ₛx is tangent to the curve,
    and s is the curve parameter.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5, endpoint=True)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=2*np.pi``.",
        bounds_default="``target=2*np.pi``.",
        coil=True,
    )

    _scalar = False  # not always a scalar, if a coilset is passed in
    _units = "(dimensionless)"
    _print_value_fmt = "Integrated curvature: "
    _broadcast_input = "coil"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil integrated curvature",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 2 * np.pi
        super().__init__(
            coil,
            ["ds", "x_s", "curvature"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute integrated curvature.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Integrated curvature.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array(
            [
                jnp.sum(
                    jnp.abs(dat["curvature"]) * safenorm(dat["x_s"], axis=1) * dat["ds"]
                )
                for dat in data
            ]
        )
        return out[self._coilset_tree["objective_mask"]]


class CoilMeanSquaredCurvature(_CoilObjective):
    """Arclength-averaged squared curvature, one value per coil.

    (1/L) ∫ κ² dl, with dl = ‖∂ₛx‖ ds and L = ∫ dl.

    This is a DIFFERENT constraint from ``CoilCurvature``, which bounds |κ| at every
    grid node INDEPENDENTLY and is therefore set by the single tightest point on a
    curve. Bounding the pointwise maximum limits the minimum bend RADIUS -- a winding
    limit; bounding this quantity limits total bending -- a stress/energy limit. They do
    not track each other: measured on a piecewise-planar-arc campaign, one coilset sat
    at 0.98 of its pointwise bound while exceeding its mean-square bound by 8%.

    Note ``κ²`` is smooth wherever ``|κ|`` is not. ``CoilCurvature`` uses
    ``|curvature|``
    and its gradient is undefined where a curve goes momentarily straight, which on a
    piecewise-planar arc happens at EVERY hinge (the sine-series shape basis forces
    w = w'' = 0 there). Squaring removes that kink, so this objective needs no
    corresponding safeguard.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5, endpoint=True)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0, 1)``.",
        bounds_default="``bounds=(0, 1)``.",
        coil=True,
    )

    _scalar = False
    _units = "(m^-2)"
    _print_value_fmt = "Coil mean squared curvature: "
    _broadcast_input = "Coil"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil mean squared curvature",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, 1)
        super().__init__(
            coil,
            ["ds", "x_s", "curvature"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1
        if self._normalize:
            # the quantity is 1/length^2, so the scale is 1/a^2, not 1/a
            self._normalization = (
                1 / np.mean([scale["a"] for scale in self._scales]) ** 2
            )

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute mean squared curvature.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Arclength-averaged squared curvature, one entry per coil.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array(
            [
                jnp.sum(
                    dat["curvature"] ** 2 * safenorm(dat["x_s"], axis=1) * dat["ds"]
                )
                / jnp.sum(safenorm(dat["x_s"], axis=1) * dat["ds"])
                for dat in data
            ]
        )
        return out[self._coilset_tree["objective_mask"]]


def _coilset_symmetry_perms(centroids, NFP, sym):
    """Permutations of the physical coil rows induced by the coilset symmetry group.

    Generators are rotation by ``2*pi/NFP`` about z and, if ``sym``, the stellarator
    reflection (x, y, z) -> (x, -y, -z). Each is turned into a row permutation by
    matching coil centroids, then the group is closed under composition. If any
    generator fails to map the centroids onto themselves (a coilset that is not
    actually symmetric), only the identity is returned.
    """
    n = centroids.shape[0]
    tol = 1e-6 * max(float(np.max(np.linalg.norm(centroids, axis=-1))), 1e-12)

    def as_perm(mapped):
        d = np.linalg.norm(mapped[:, None, :] - centroids[None, :, :], axis=-1)
        p = np.argmin(d, axis=1)
        ok = np.all(d[np.arange(n), p] <= tol) and np.unique(p).size == n
        return p if ok else None

    gens = []
    if NFP > 1:
        a = 2 * np.pi / NFP
        rot = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        gens.append(as_perm(centroids @ rot.T))
    if sym:
        gens.append(as_perm(centroids * np.array([1.0, -1.0, -1.0])))
    if any(g is None for g in gens):
        return [np.arange(n)]
    group = {tuple(range(n))}
    frontier = [np.arange(n)]
    while frontier:
        new = []
        for p in frontier:
            for g in gens:
                q = g[p]
                if tuple(q) not in group:
                    group.add(tuple(q))
                    new.append(q)
        frontier = new
    return [np.array(t) for t in sorted(group)]


def _unique_neighbour_pairs(centroids, indices, n_neighbors, NFP, sym):
    """Fixed, symmetry-deduplicated (coil, neighbour) pairs for ``per_pair_unique``.

    Each independent coil ``i`` in ``indices`` gets its ``n_neighbors`` nearest
    physical coils by centroid. The candidate rows are the pairs (i, j) with j in that
    set. Two candidates describe the same distance if one maps to the other (in either
    order) under the symmetry group; of each such orbit, the lexicographically smallest
    candidate is kept.

    Returns
    -------
    pairs : list of (int, int)
        Physical row indices into ``_compute_position``, first entry independent.
    """
    n = centroids.shape[0]
    cd = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(cd, np.inf)
    nbrs = {
        int(i): sorted(int(j) for j in np.argsort(cd[i])[: min(n_neighbors, n - 1)])
        for i in indices
    }
    group = _coilset_symmetry_perms(centroids, NFP, sym)
    pairs = []
    for i in nbrs:
        for j in nbrs[i]:
            orbit = set()
            for g in group:
                orbit.add((int(g[i]), int(g[j])))
                orbit.add((int(g[j]), int(g[i])))
            canonical = min((a, b) for a, b in orbit if a in nbrs and b in nbrs[a])
            if canonical == (i, j):
                pairs.append((i, j))
    return pairs


def _independent_coil_indices(coilset):
    """Rows of ``_compute_position`` holding independent (non-symmetry-copy) coils.

    ``CoilSet._compute_position`` emits the ``len(coilset)`` independent coils first,
    then their stellarator reflections, then tiles the whole block over field periods.
    A symmetric ``CoilSet`` is invariant under that group, so every copy of a given
    coil has an isometric neighbourhood *within the coilset*, and any quantity that
    depends only on coilset geometry is identical across copies.

    Only valid for quantities whose "other object" is the coilset itself. Distances to
    an external target (e.g. a discretized plasma surface) are invariant under the
    field-period rotation but not necessarily under the reflection.

    Returns
    -------
    idx : ndarray of int
        Indices into the ``num_coils`` rows of ``_compute_position``.

    """
    from desc.coils import CoilSet, MixedCoilSet

    if isinstance(coilset, MixedCoilSet):  # checked first: subclasses CoilSet
        idx, offset = [], 0
        for coil in coilset:
            idx.append(_independent_coil_indices(coil) + offset)
            offset += coil.num_coils
        return np.concatenate(idx) if idx else np.array([], dtype=int)
    if isinstance(coilset, CoilSet):
        return np.arange(len(coilset))
    return np.array([0])  # a single coil


def _field_period_independent_indices(coilset, target_nfp=None):
    """Rows of ``_compute_position`` that are not field-period copies of another row.

    ``CoilSet._compute_position`` tiles a base block -- the independent coils plus,
    if ``sym``, their stellarator reflections -- over ``NFP`` field periods. A target
    invariant under rotation by ``2*pi/NFP`` sees identical distances from every tile,
    so only the base block carries information.

    Unlike `_independent_coil_indices` this does NOT deduplicate the reflections:
    those are redundant only if the *target* is reflection symmetric too, which is not
    guaranteed for an external object such as a plasma surface.

    Parameters
    ----------
    coilset : CoilSet or MixedCoilSet or _Coil
        Coilset whose position rows are being indexed.
    target_nfp : int, optional
        Field periodicity of the target the coils are measured against. The tiles are
        equivalent only if this is a multiple of the coilset ``NFP``; when it is not,
        every row is returned so no information is discarded.

    Returns
    -------
    idx : ndarray of int
        Indices into the ``num_coils`` rows of ``_compute_position``.

    """
    from desc.coils import CoilSet, MixedCoilSet

    if isinstance(coilset, MixedCoilSet):  # checked first: subclasses CoilSet
        idx, offset = [], 0
        for coil in coilset:
            idx.append(_field_period_independent_indices(coil, target_nfp) + offset)
            offset += coil.num_coils
        return np.concatenate(idx) if idx else np.array([], dtype=int)
    if isinstance(coilset, CoilSet):
        if target_nfp is not None and coilset.NFP > 1 and target_nfp % coilset.NFP:
            return np.arange(coilset.num_coils)  # tiles are not equivalent
        return np.arange(len(coilset) * (int(coilset.sym) + 1))
    return np.array([0])  # a single coil


def _closed_polyline(x):
    """Points on a closed curve -> (segment starts, segment directions).

    ``x`` must be ordered along the curve; the final segment closes the loop.
    """
    return x, jnp.roll(x, -1, axis=-2) - x


def _segment_segment_distance(p1, d1, p2, d2):
    """Minimum distance between every pair of line segments.

    Segments are ``p1 + s d1`` and ``p2 + t d2`` for ``s, t`` in [0, 1]. Standard
    clamped closest-point solve: solve the unconstrained least squares for ``s``,
    clamp to [0,1], re-solve ``t``, clamp, re-solve ``s``, clamp.

    Parameters
    ----------
    p1, d1 : ndarray, shape(m, 3)
        Starts and directions of the first set of segments.
    p2, d2 : ndarray, shape(n, 3)
        Starts and directions of the second set of segments.

    Returns
    -------
    dist : ndarray, shape(m, n)

    """
    r = p1[:, None, :] - p2[None, :, :]  # shape(m,n,3)
    a = jnp.sum(d1 * d1, axis=-1)[:, None]  # shape(m,1)
    e = jnp.sum(d2 * d2, axis=-1)[None, :]  # shape(1,n)
    b = d1 @ d2.T  # shape(m,n)
    c = jnp.einsum("mk,mnk->mn", d1, r)
    f = jnp.einsum("nk,mnk->mn", d2, r)
    # safediv, not a bare divide: parallel segments make `a e - b b` vanish and a
    # repeated node makes `a` or `e` vanish. A plain jnp.where would still
    # differentiate the dead branch and hand back a nan gradient.
    s = jnp.clip(safediv(b * f - c * e, a * e - b * b), 0.0, 1.0)
    t = jnp.clip(safediv(b * s + f, e), 0.0, 1.0)
    s = jnp.clip(safediv(b * t - c, a), 0.0, 1.0)
    # (p1 + s d1) - (p2 + t d2) == r + s d1 - t d2; formed directly so neither array
    # of closest points is materialized. safenorm because two touching coils give
    # exactly zero here, where d||v||/dv is 0/0.
    return safenorm(
        r + s[..., None] * d1[:, None, :] - t[..., None] * d2[None, :, :], axis=-1
    )


def _segment_segment_distance_elementwise(p1, d1, p2, d2):
    """``_segment_segment_distance`` for matched rows.

    Segment k of set 1 against segment k of set 2. All inputs shape(K, 3); returns
    shape(K,).
    """
    r = p1 - p2
    a = jnp.sum(d1 * d1, axis=-1)
    e = jnp.sum(d2 * d2, axis=-1)
    b = jnp.sum(d1 * d2, axis=-1)
    c = jnp.sum(d1 * r, axis=-1)
    f = jnp.sum(d2 * r, axis=-1)
    s = jnp.clip(safediv(b * f - c * e, a * e - b * b), 0.0, 1.0)
    t = jnp.clip(safediv(b * s + f, e), 0.0, 1.0)
    s = jnp.clip(safediv(b * t - c, a), 0.0, 1.0)
    return safenorm(r + s[..., None] * d1 - t[..., None] * d2, axis=-1)


def _warn_cc_active_overflow(count, cap):
    count = int(np.asarray(count))
    if count > cap:
        warnings.warn(
            f"CoilSetDistancePenalty: {count} segment pairs are within d_min but "
            f"max_active_pairs={cap}. The pairs past the first {cap} were dropped, so "
            "the value and its derivatives are inexact; raise max_active_pairs.",
            UserWarning,
        )


class CoilSetMinDistance(_Objective):
    """Target the minimum distance between coils in a coilset.

    Will yield one value per coil in the coilset, which is the minimum distance to
    another coil in that coilset. With ``pair_mode="per_pair"`` it instead yields one
    value per (coil, neighbour) pair; see that parameter.

    Parameters
    ----------
    coil : CoilSet
        Coil(s) that are to be optimized.
    grid : Grid, list, optional
        Collocation grid used to discretize each coil. Defaults to the default grid
        for the given coil-type, see ``coils.py`` and ``curve.py`` for more details.
        If a list, must have the same structure as coils.
    use_softmin: bool, optional
        Use softmin or hard min. Softmin is a smooth approximation to the actual minimum
        distance that may give smoother gradients, at the expense of being slightly more
        expensive and only an approximate minimum.
    softmin_alpha: float, optional
        Parameter used for softmin. The larger ``softmin_alpha``, the closer the
        softmin approximates the hardmin. softmin -> hardmin as
        ``softmin_alpha`` -> infinity.
    dist_chunk_size : int > 0, optional
        When computing distances, how many coils to consider at once. Default is all
        coils, which is generally the fastest but requires the most memory. If there are
        a large number of coils, or if the resolution is very high, setting this to a
        small value will reduce peak memory usage at the cost of slightly increased
        runtime.
    num_neighbors : int, optional
        Limit the pairwise distance computation to the num_neighbors nearest neighbors
        per coil, determined by centroid distance. This is helpful for reducing memory
        usage when you have hundreds of small coils, with a recommended value of about
        num_neighbors = 20. Default value of None or num_neighbors >= num_coils - 1
        computes the full pairwise distances.
    pair_mode : {"per_coil", "per_pair"}, optional
        Whether to collapse each coil's neighbours into a single row.

        ``"per_coil"`` (default, and the historical behaviour) returns one value per
        coil: the minimum distance to ANY other coil. That outer minimum is kinked
        wherever two neighbours are equidistant, so when two pairs are near-tied at
        the bound the optimizer is asked to stand on a ridge. The argmin then swaps
        between them inside a single trust-region step, the Gauss-Newton model built
        at one argmin mispredicts, the step is rejected and the radius shrinks.

        ``"per_pair"`` returns one value per (coil, neighbour) pair, so no minimum is
        taken across neighbours and that ridge does not exist. Rows whose pairs are
        comfortably separated are inactive and contribute exactly zero, exactly as
        with any other bounds block, so the extra rows cost little. ``dim_f`` grows
        from ``n_coils`` to ``n_coils * n_neighbours``. The minimum WITHIN a pair
        (over nodes or segments) is kept: there the closest point is generically
        unique, which is the well-behaved case.

        Measured motivation at ``B=2`` (see ``c0/CONVENTIONS.md`` 4.1.1): two coil
        pairs 8.3e-05 apart straddling the bound, one violating by 5.0e-05 and one
        satisfied by 3.3e-05, with the augmented-Lagrangian optimality oscillating
        rather than decreasing and ``gtolk`` never firing.

        ``"per_pair_unique"`` is ``"per_pair"`` with two changes that make it cheap
        enough for production:

        1. The neighbour sets are FIXED at build time: for each independent coil, its
           ``num_neighbors`` nearest coils by centroid (all others if None), chosen
           among ALL physical coils, so field-period and stellarator copies count
           (coil 0's neighbours can include the last coil). ``"per_pair"`` re-picks the
           neighbours every call and orders its rows by distance, so two neighbours
           swapping rank swap row VALUES -- a discontinuity. Fixed sets give every row
           a fixed pair. A pair that becomes close without being in the set is not
           seen; the dense feasibility check is the backstop.
        2. Rows are deduplicated over the coilset's symmetry group (rotation by
           ``2*pi/NFP`` and, if ``sym``, the stellarator reflection), found by
           matching coil centroids. A pair and all its images, in either order, have
           the same distance, so one row per orbit is kept. Without this, e.g. the
           (coil 0, coil 1) and (coil 1, coil 0) rows are identical, which doubles
           that pair's weight and makes its multiplier non-unique.
           If the centroids do not match under the group (a non-symmetric coilset),
           only the (k, j)/(j, k) duplicates are removed.

        ``dist_chunk_size`` is still in units of coils: pairs are evaluated
        ``dist_chunk_size * num_neighbors`` at a time, the same memory as
        ``"per_pair"``.
    distance_method : {"point", "segment"}, optional
        How to measure the distance between two coils. ``"point"`` (default) takes the
        minimum over distances between grid POINTS, so the closest approach must land
        on a node and the result is biased high by roughly the node spacing.
        ``"segment"`` takes the minimum over distances between the line SEGMENTS
        joining consecutive points, which lets the closest approach fall anywhere on
        the curve and converges as O(h^2) instead.

        Prefer ``"segment"`` whenever the coils may come much closer together than the
        node spacing. Point sampling cannot represent that case at all: MEASURED on a
        coilset whose coils actually intersect, ``"point"`` reported 6.4e-3 m of
        clearance on a 301-node grid where ``"segment"`` reported 9.0e-6 m, a factor of
        710. Worse, the point-sampled value moves the WRONG WAY under a step that
        genuinely separates the coils -- its sampling error shrinks faster than the true
        distance grows -- so it manufactures a barrier that blocks the optimizer from
        repairing the geometry. ``"segment"`` is also cheaper for equal accuracy: it
        beats point sampling here on a grid 60x coarser.

        Requires grid nodes ordered along the curve (as ``LinearGrid`` gives).

        Peak memory is roughly twice ``"point"`` at the same grid, and note that
        ``dist_chunk_size`` chunks over COILS, not nodes, so it does not bound the
        per-coil ``(n_other, num_nodes, num_nodes, 3)`` temporaries. Reach for a
        COARSER grid with ``"segment"``, not a finer one -- it is more accurate there
        anyway, and a fine grid will exhaust memory (MEASURED: 8001 nodes x 15 other
        coils asks for 53 GB).

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
        coil=True,
    )

    _static_attrs = _Objective._static_attrs + [
        "_use_softmin",
        "_dist_chunk_size",
        "_num_neighbors",
        "_n_other",
        "_pairs",
        "_coil_indices",
        "_distance_method",
        "_pair_mode",
        "_signed",
    ]

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Minimum coil-coil distance: "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil-coil minimum distance",
        jac_chunk_size=None,
        use_softmin=False,
        softmin_alpha=1.0,
        dist_chunk_size=None,
        num_neighbors=None,
        distance_method="point",
        pair_mode="per_coil",
        signed=False,
        link_grid=None,
    ):
        from desc.coils import CoilSet

        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._grid = grid
        self._use_softmin = use_softmin
        self._softmin_alpha = softmin_alpha
        self._dist_chunk_size = dist_chunk_size
        self._num_neighbors = num_neighbors
        self._distance_method = distance_method
        self._pair_mode = pair_mode
        self._signed = signed
        self._link_grid = link_grid
        errorif(
            pair_mode not in ("per_coil", "per_pair", "per_pair_unique"),
            ValueError,
            'pair_mode must be "per_coil", "per_pair" or "per_pair_unique", '
            f"got {pair_mode}",
        )
        errorif(
            pair_mode != "per_coil" and signed,
            ValueError,
            'signed=True is only implemented for pair_mode="per_coil"',
        )
        errorif(
            distance_method not in ("point", "segment"),
            ValueError,
            f'distance_method must be "point" or "segment", got {distance_method}',
        )
        errorif(
            not isinstance(coil, CoilSet),
            ValueError,
            "coil must be of type CoilSet, not an individual Coil",
        )
        super().__init__(
            things=coil,
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
        coilset = self.things[0]
        grid = self._grid or None

        # Every symmetry copy of a coil has an isometric neighbourhood, so evaluating
        # all `num_coils` rows returns each distinct value `num_coils // len(coilset)`
        # times. That duplication is not free: it multiplies this objective's Gram
        # contribution (and so its effective weight) by the multiplicity, skews the
        # `x_scale="auto"` column norms, and costs the same factor in compute. Keep one
        # representative per independent coil.
        self._coil_indices = _independent_coil_indices(coilset)
        # `num_coils` is the full PHYSICAL count that `_compute_position` emits
        # (symmetry copies included), which is what `body` iterates over. It is not
        # `len(coilset)` nor the leaf count -- at nc=4, NFP=2, sym=True those are 4
        # while `_compute_position` returns 16.
        _num_coils = coilset.num_coils
        _n_other = _num_coils - 1
        if self._num_neighbors is not None and self._num_neighbors < _num_coils - 1:
            _n_other = self._num_neighbors
        self._n_other = _n_other
        self._dim_f = self._coil_indices.size * (
            _n_other if self._pair_mode == "per_pair" else 1
        )
        if self._pair_mode == "per_pair_unique":
            pts0 = np.asarray(
                coilset._compute_position(
                    params=coilset.params_dict, grid=grid, basis="xyz"
                )
            )
            self._pairs = tuple(
                _unique_neighbour_pairs(
                    np.mean(pts0, axis=1),
                    self._coil_indices,
                    _n_other,
                    getattr(coilset, "NFP", 1),
                    bool(getattr(coilset, "sym", False)),
                )
            )
            self._dim_f = len(self._pairs)
        if self._signed:
            # Grid for the linking number ONLY. Memory in `_compute_linking_number` is
            # the (ncoil, ncol, nnode, nnode, 3) broadcast, i.e. O(N^2): 16x16 at 801
            # nodes is 3.9 GB and OOMs inside an AD trace, while 201 nodes is ~250 MB.
            # MEASURED on a linked coilset: the SIGN is identical at N=50/100/150/200
            # and to the N=400 reference, returning a clean integer 1.00000 -- we need
            # the sign only to +/-0.5, not the value, so a coarse grid is exact here.
            self._constants_link_grid = (
                self._link_grid if self._link_grid is not None else LinearGrid(N=100)
            )
        self._constants = {"coilset": coilset, "grid": grid, "quad_weights": 1.0}
        if self._pair_mode == "per_pair_unique":
            self._constants["pair_a"] = np.array([a for a, _ in self._pairs], dtype=int)
            self._constants["pair_b"] = np.array([b for _, b in self._pairs], dtype=int)

        if self._normalize:
            coils = tree_leaves(coilset, is_leaf=lambda x: not hasattr(x, "__len__"))
            scales = [compute_scaling_factors(coil)["a"] for coil in coils]
            self._normalization = np.mean(scales)  # mean length of coils

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute minimum distances between coils.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Minimum distance to another coil for each coil in the coilset.

        """
        constants = self._get_deprecated_constants(constants)
        pts = constants["coilset"]._compute_position(
            params=params, grid=constants["grid"], basis="xyz"
        )  # pts.shape = (num_coils, num_nodes, 3)
        # all physical coils remain the *targets*; only the set we evaluate from
        # is reduced, so this is the full count, not dim_f
        num_coils = pts.shape[0]

        if self._pair_mode == "per_pair_unique":
            pair_a, pair_b = constants["pair_a"], constants["pair_b"]

            def pair_body(p):
                c1, c2 = pts[pair_a[p]], pts[pair_b[p]]
                if self._distance_method == "segment":
                    p1, d1 = _closed_polyline(c1)
                    p2, d2 = _closed_polyline(c2)
                    dist = _segment_segment_distance(p1, d1, p2, d2)
                else:
                    dist = safenorm(c1[:, None] - c2[None, :], axis=-1)
                if self._use_softmin:
                    return softmin(dist, self._softmin_alpha)
                return jnp.min(dist)

            chunk = (
                None
                if self._dist_chunk_size is None
                else self._dist_chunk_size * self._n_other
            )
            return vmap_chunked(pair_body, chunk_size=chunk)(jnp.arange(pair_a.size))

        if self._num_neighbors is not None and self._num_neighbors < num_coils - 1:
            # only consider nearest neighbors
            centroids = jnp.mean(pts, axis=1)  # (num_coils, 3)
            centroid_dists = safenorm(centroids[:, None] - centroids[None, :], axis=-1)
            centroid_dists = centroid_dists.at[jnp.diag_indices(num_coils)].set(jnp.inf)

            def get_other_pts(k):
                neighbors_idx = jax.lax.stop_gradient(
                    jnp.argsort(centroid_dists[k])[: self._num_neighbors]
                )
                return pts[neighbors_idx]

        else:  # consider all other coils

            def get_other_pts(k):
                return jnp.delete(pts, k, axis=0, assume_unique_indices=True)

        def body(k):
            # Entry i, j, n of dist holds the distance from the jth point (or
            # segment) on the kth coil to the nth point (or segment) on the ith
            # coil, giving shape (ncoils, num_nodes, num_nodes).
            coil_pts = pts[k]
            other_pts = get_other_pts(k)
            if self._distance_method == "segment":
                p1, d1 = _closed_polyline(coil_pts)

                def one(other):
                    p2, d2 = _closed_polyline(other)
                    return _segment_segment_distance(p1, d1, p2, d2)

                dist = jax.vmap(one)(other_pts)
            else:
                dist = safenorm(coil_pts[None, :, None] - other_pts[:, None], axis=-1)
            if self._pair_mode == "per_pair":
                # One row per (coil, neighbour) instead of one row per coil. The
                # collapsed form takes a min ACROSS neighbours, and that min is kinked
                # wherever two neighbours are equidistant -- a ridge the optimizer has
                # to sit on whenever two pairs are near-tied at the bound. Measured at
                # B=2 (CONVENTIONS 4.1.1): two pairs 8.3e-05 apart straddling the
                # bound, the argmin swapping inside the solver's own step length, and
                # optimality oscillating instead of decreasing. Keeping the neighbours
                # as separate rows removes that ridge; the surviving min is over nodes
                # WITHIN one pair, where the closest point is generically unique.
                if self._use_softmin:
                    return jax.vmap(lambda d: softmin(d, self._softmin_alpha))(dist)
                return jnp.min(dist, axis=(1, 2))
            if self._signed:
                # SIGNED distance: negate the pairs this coil is LINKED with, so a
                # linked pair goes negative, wins the min, and reads as a violation
                # that GROWS as the coils separate on the wrong side.
                #
                # This exists because the unsigned minimum is blind to a crossing.
                # MEASURED on planar_cert_trace, pair (1,2): d went 85.7 -> 10.9 ->
                # 5.0 mm through the crossing and then REBOUNDED to 28, 38, 48, 55,
                # 62, 69, 75 mm -- so after passing through, the constraint reports
                # the geometry as steadily improving and the AL ratifies a linked
                # coilset. Signed, those same iterates read -28 ... -75 mm: monotone,
                # increasingly violated, pointing back at the crossing.
                #
                # `sign` is STOP_GRADIENT'd. That is exact, not an approximation: the
                # sign is locally constant so its derivative is 0 almost everywhere,
                # and AT a crossing d = 0 makes the product differentiable anyway. It
                # also keeps the Gauss integral out of the derivative graph entirely,
                # which is what makes this usable at all -- differentiating it gives
                # 2.9e+12 at contact (CONVENTIONS 8i). MEASURED with stop_gradient on
                # a LINKED coilset: max|jac entry| = 3.99, same order as unsigned.
                #
                # A flip can only happen where the curves intersect, i.e. where d = 0,
                # so the induced jump in `sign*d` is 2d ~ 0 -- the discontinuity
                # cancels exactly where it would occur.
                lk = constants["coilset"]._compute_linking_number(
                    params=params, grid=self._constants_link_grid
                )
                sgn = jax.lax.stop_gradient(jnp.where(jnp.abs(lk[k]) > 0.5, -1.0, 1.0))
                sgn = jnp.delete(sgn, k, assume_unique_indices=True)
                per_other = (
                    jax.vmap(lambda d: softmin(d, self._softmin_alpha))(dist)
                    if self._use_softmin
                    else jnp.min(dist, axis=(1, 2))
                )
                return jnp.min(sgn * per_other)
            if self._use_softmin:
                return softmin(dist, self._softmin_alpha)
            return jnp.min(dist)

        k = self._coil_indices
        min_dist_per_coil = vmap_chunked(body, chunk_size=self._dist_chunk_size)(k)
        return min_dist_per_coil.reshape(-1)


class CoilSetDistancePenalty(_Objective):
    """Smooth threshold penalty on coil-coil distance: one C^1 row per coil pair.

    For each coil pair (a, b) this returns

        g_ab = sum_{i, j} max(0, 1 - d_ij / d_min)**2

    over all segment pairs (or node pairs) i on coil a, j on coil b, where d_ij is their
    distance. g_ab >= 0 everywhere and g_ab = 0 exactly when every segment pair is at
    least ``d_min`` apart, so the constraint is ``g <= 0``: the default ``bounds=(-inf,
    0)``. This is the form of SIMSOPT's ``CurveCurveDistance`` penalty (without its
    arclength weights), used by Gil et al., PRE 114, 025202 (2026) as a threshold
    constraint.

    Why not a minimum. ``CoilSetMinDistance`` returns a hard minimum over segment pairs
    (and, per coil, over neighbours). A minimum is kinked wherever two of its arguments
    tie, and a coil optimization drives the iterate ONTO such ties: coils pressed
    against the bound touch along a nearly parallel stretch, where two separate approach
    points keep trading places. MEASURED on precise_QH xyzN6
    (qh_feas_work/qh_kink_jump2.log): pair (0, 1) had two approaches 7 segments apart
    within 0.2 mm, the directional-derivative jump was O(1) down to h = 1e-7, and the
    augmented-Lagrangian trust radius collapsed to 1e-6 with optimality alternating
    between two values. A sum of ``max(0, .)**2`` terms has no minimum to switch: it is
    C^1 in the coil parameters (the segment-segment distance is C^1; with
    ``distance_method="point"`` the distances are C^inf and the row is still C^1).

    Scale and degeneracy. g is dimensionless. A single contact penetrating by delta
    gives g ~ n * (delta / d_min)**2, with n the number of segment pairs inside the
    contact (1-4 at small delta, more for a long parallel contact), so ``ctol=1e-4``
    corresponds to roughly 0.5-1% penetration. The price of C^1 is that the gradient
    vanishes at the boundary: the constraint is degenerate there, the multiplier grows
    as the penetration shrinks, and the solution sits very slightly violated. Adjudicate
    feasibility with a hard-min distance check and back off ``d_min`` if needed.

    Rows. As ``CoilSetMinDistance(pair_mode="per_pair_unique")``: for each independent
    coil, its ``num_neighbors`` nearest coils by centroid among ALL physical coils
    (field-period and stellarator copies included), fixed at build, one row per pair
    orbit of the coilset symmetry group.

    Use ``num_neighbors=None`` (every coil) for anything but a quick look. Nearest by
    CENTROID is a heuristic that shaped coils defeat. MEASURED on precise_QH arcB2
    (qh_feas_work/qh_nbr_all.log): with 6 neighbours, the closest pair -- two symmetry
    copies of one coil bending toward each other across a symmetry plane, 9% inside
    d_min -- was in no row, the penalty summed 3.4e-5, and the solver ended at d_cc
    0.938 of its bound without ever seeing the violation. With every coil the row reads
    5.4e-2.

    Two-pass evaluation (``max_active_pairs`` set). Every coil pair is then affordable:
    all segment-pair distances are computed with no derivative (``stop_gradient``), the
    ones within ``d_min`` are selected across ALL rows into one list padded to
    ``max_active_pairs``, and only those are recomputed with derivatives and summed back
    into their rows. Exact (value, gradient, Hessian) whenever at most
    ``max_active_pairs`` segment pairs are active in total; beyond that the pairs past
    the first ``max_active_pairs`` are dropped and a warning is raised.
    ``active_pair_count``
    measures the count. Without it, every segment pair of every row is differentiated:
    MEASURED on arcB2 with every coil (118 rows), a 4.2 s Jacobian.

    Signed rows (``signed=True``), the analogue of ``CoilSetMinDistance(signed=True)``.
    An unsigned distance is blind to two coils passing THROUGH each other: after the
    crossing it rebounds and reports clearance while the coilset is linked (MEASURED on
    precise_QA, see ``CoilSetMinDistance``), and on precise_QA unguarded warm starts
    linked 2 of 2 times against 0 of 3 with the signed minimum. Here a pair whose
    linking number is nonzero (|Lk| > 0.5, computed per row on ``link_grid`` under
    stop_gradient, so it only selects the branch) drops its repulsive sum and reads

        N**2 * (1 + d*/d_min)**2,     d* = min over segment pairs (argmin from pass 1)

    It grows as the pair separates on the wrong side, and its gradient pulls the closest
    points together, back toward the crossing. The factor N**2 makes un-linking always
    LOWER the row: every unsigned term is at most 1, so an unlinked row never exceeds
    N**2, while a linked row is at least N**2. A plain sign flip inside the sum cannot
    promise that, and an un-crossing step that raises the merit would be rejected. The
    jump at the crossing is therefore deliberate and points the right way.

    Parameters
    ----------
    coil : CoilSet
        Coil(s) that are to be optimized.
    d_min : float
        Distance threshold [m]. Segment pairs closer than this are penalized.
    grid : Grid, list, optional
        Collocation grid used to discretize each coil (nodes ordered along the curve).
    dist_chunk_size : int > 0, optional
        Pairs are evaluated ``dist_chunk_size * num_neighbors`` at a time. Default: all.
    num_neighbors : int, optional
        Neighbours per independent coil, by centroid distance. Default None = all
        others.
    distance_method : {"segment", "point"}, optional
        Distance between segments (default; resolves approaches between nodes) or nodes.
    max_active_pairs : int, optional
        Capacity of the two-pass selection, in segment pairs summed over ALL rows.
        Default None: no selection, every segment pair is differentiated.
    signed : bool, optional
        Topology guard (requires ``max_active_pairs``). A LINKED pair's row becomes
        ``N**2 * (1 + d*/d_min)**2``, with N the nodes per coil and d* the pair's
        closest segment distance, instead of its repulsive sum. See "Signed rows" above.
    link_grid : Grid, optional
        Grid for the linking numbers of ``signed``. Default ``LinearGrid(N=100)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-np.inf, 0)``.",
        bounds_default="``bounds=(-np.inf, 0)``.",
        coil=True,
    )

    _static_attrs = _Objective._static_attrs + [
        "_dist_chunk_size",
        "_num_neighbors",
        "_n_other",
        "_pairs",
        "_coil_indices",
        "_distance_method",
        "_max_active_pairs",
        "_n_nodes",
        "_signed",
    ]

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil-coil distance penalty: "

    def __init__(
        self,
        coil,
        d_min,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil-coil distance penalty",
        jac_chunk_size=None,
        dist_chunk_size=None,
        num_neighbors=None,
        distance_method="segment",
        max_active_pairs=None,
        signed=False,
        link_grid=None,
    ):
        from desc.coils import CoilSet

        if target is None and bounds is None:
            bounds = (-np.inf, 0.0)
        self._d_min = d_min
        self._grid = grid
        self._dist_chunk_size = dist_chunk_size
        self._num_neighbors = num_neighbors
        self._distance_method = distance_method
        errorif(
            max_active_pairs is not None and int(max_active_pairs) < 1,
            ValueError,
            "max_active_pairs must be None or a positive integer, "
            f"got {max_active_pairs}",
        )
        self._max_active_pairs = (
            None if max_active_pairs is None else int(max_active_pairs)
        )
        errorif(
            signed and max_active_pairs is None,
            ValueError,
            "signed=True is only implemented for the two-pass evaluation "
            "(max_active_pairs set)",
        )
        self._signed = bool(signed)
        self._link_grid = link_grid
        errorif(
            distance_method not in ("point", "segment"),
            ValueError,
            f'distance_method must be "point" or "segment", got {distance_method}',
        )
        errorif(not d_min > 0, ValueError, f"d_min must be positive, got {d_min}")
        errorif(
            not isinstance(coil, CoilSet),
            ValueError,
            "coil must be of type CoilSet, not an individual Coil",
        )
        super().__init__(
            things=coil,
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
        coilset = self.things[0]
        grid = self._grid or None
        self._coil_indices = _independent_coil_indices(coilset)
        num_coils = coilset.num_coils
        self._n_other = (
            num_coils - 1
            if self._num_neighbors is None
            else min(self._num_neighbors, num_coils - 1)
        )
        pts0 = np.asarray(
            coilset._compute_position(
                params=coilset.params_dict, grid=grid, basis="xyz"
            )
        )
        self._pairs = tuple(
            _unique_neighbour_pairs(
                np.mean(pts0, axis=1),
                self._coil_indices,
                self._n_other,
                getattr(coilset, "NFP", 1),
                bool(getattr(coilset, "sym", False)),
            )
        )
        self._dim_f = len(self._pairs)
        self._n_nodes = int(pts0.shape[1])
        if self._max_active_pairs is not None:
            self._max_active_pairs = min(
                self._max_active_pairs, self._dim_f * self._n_nodes**2
            )
        self._constants = {
            "coilset": coilset,
            "grid": grid,
            "quad_weights": 1.0,
            "pair_a": np.array([a for a, _ in self._pairs], dtype=int),
            "pair_b": np.array([b for _, b in self._pairs], dtype=int),
        }
        if self._signed:
            lg = self._link_grid if self._link_grid is not None else LinearGrid(N=100)
            self._constants["link_grid"] = lg
            self._constants["link_dx"] = np.asarray(lg.spacing[:, 2])
        if self._normalize:
            self._normalization = 1.0  # already dimensionless

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the distance penalty of each coil pair.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            sum over segment (or node) pairs of max(0, 1 - d/d_min)**2, per coil pair.

        """
        constants = self._get_deprecated_constants(constants)
        pts = constants["coilset"]._compute_position(
            params=params, grid=constants["grid"], basis="xyz"
        )
        pair_a, pair_b = constants["pair_a"], constants["pair_b"]
        if self._max_active_pairs is not None:
            return self._compute_two_pass(
                pts, jnp.asarray(pair_a), jnp.asarray(pair_b), params, constants
            )

        def pair_body(p):
            c1, c2 = pts[pair_a[p]], pts[pair_b[p]]
            if self._distance_method == "segment":
                p1, d1 = _closed_polyline(c1)
                p2, d2 = _closed_polyline(c2)
                dist = _segment_segment_distance(p1, d1, p2, d2)
            else:
                dist = safenorm(c1[:, None] - c2[None, :], axis=-1)
            return jnp.sum(jnp.maximum(0.0, 1.0 - dist / self._d_min) ** 2)

        chunk = (
            None
            if self._dist_chunk_size is None
            else self._dist_chunk_size * self._n_other
        )
        return vmap_chunked(pair_body, chunk_size=chunk)(jnp.arange(pair_a.size))

    def _row_distances(self, pts, pair_a, pair_b, p):
        """All segment-pair (or node-pair) distances of row p, shape(N, N)."""
        c1, c2 = pts[pair_a[p]], pts[pair_b[p]]
        if self._distance_method == "segment":
            p1, d1 = _closed_polyline(c1)
            p2, d2 = _closed_polyline(c2)
            return _segment_segment_distance(p1, d1, p2, d2)
        return safenorm(c1[:, None] - c2[None, :], axis=-1)

    def _chunk(self):
        return (
            None
            if self._dist_chunk_size is None
            else self._dist_chunk_size * self._n_other
        )

    def _pass1(self, pts, pair_a, pair_b):
        """First pass, no derivatives: segment pairs within d_min, and row argmins.

        The mask is flattened over (row, i, j); the argmin is the flat index i*N + j of
        each row's closest segment pair.
        """
        P = jax.lax.stop_gradient(pts)

        def row(p):
            d = self._row_distances(P, pair_a, pair_b, p)
            return d < self._d_min, jnp.argmin(d)

        mask, amin = vmap_chunked(row, chunk_size=self._chunk())(
            jnp.arange(pair_a.size)
        )
        return mask.reshape(-1), amin

    def _linked_rows(self, params, constants, pair_a, pair_b):
        """Whether each row's pair is linked (|linking number| > 0.5), no derivative."""
        from desc.coils import _linking_number

        x, x_s = constants["coilset"]._compute_position(
            params, constants["link_grid"], dx1=True, basis="xyz"
        )
        x, x_s = jax.lax.stop_gradient(x), jax.lax.stop_gradient(x_s)
        dx = jnp.asarray(constants["link_dx"])
        lk = vmap_chunked(
            lambda p: _linking_number(
                x[pair_a[p]], x[pair_b[p]], x_s[pair_a[p]], x_s[pair_b[p]], dx, dx
            ),
            chunk_size=self._chunk(),
        )(jnp.arange(pair_a.size))
        return jnp.abs(lk / (4 * jnp.pi)) > 0.5

    def _seg_dist(self, pts, a, b, i, j):
        """Distances of matched (coil a node i, coil b node j) pairs, differentiable."""
        N = self._n_nodes
        if self._distance_method == "segment":
            p1 = pts[a, i]
            p2 = pts[b, j]
            d1 = pts[a, (i + 1) % N] - p1
            d2 = pts[b, (j + 1) % N] - p2
            return _segment_segment_distance_elementwise(p1, d1, p2, d2)
        return safenorm(pts[a, i] - pts[b, j], axis=-1)

    def _compute_two_pass(self, pts, pair_a, pair_b, params, constants):
        N, K = self._n_nodes, self._max_active_pairs
        active, amin = self._pass1(pts, pair_a, pair_b)
        if self._signed:
            linked = self._linked_rows(params, constants, pair_a, pair_b)
            # a linked row keeps no repulsive terms (they would oppose un-crossing)
            active = active & ~jnp.repeat(linked, N * N)
        count = jnp.sum(active)
        (idx,) = jnp.nonzero(active, size=K, fill_value=0)
        jax.debug.callback(lambda n: _warn_cc_active_overflow(n, K), count)
        # second pass: only the selected segment pairs, with derivatives; slots past
        # `count` are padding (index 0) and are masked out
        row = idx // (N * N)
        dist = self._seg_dist(pts, pair_a[row], pair_b[row], (idx // N) % N, idx % N)
        valid = jnp.arange(K) < count
        term = jnp.where(valid, jnp.maximum(0.0, 1.0 - dist / self._d_min) ** 2, 0.0)
        out = jnp.zeros(pair_a.size).at[row].add(term)
        if self._signed:
            dstar = self._seg_dist(pts, pair_a, pair_b, amin // N, amin % N)
            out = out + jnp.where(linked, N**2 * (1.0 + dstar / self._d_min) ** 2, 0.0)
        return out

    def active_pair_count(self, params=None):
        """Segment pairs within ``d_min`` over all rows, evaluated eagerly.

        Use it to size ``max_active_pairs``.
        """
        coilset = self.things[0]
        params = coilset.params_dict if params is None else params
        pts = coilset._compute_position(
            params=params, grid=self._constants["grid"], basis="xyz"
        )
        pair_a = jnp.asarray(self._constants["pair_a"])
        pair_b = jnp.asarray(self._constants["pair_b"])
        return int(jnp.sum(self._pass1(pts, pair_a, pair_b)[0]))


class PlasmaCoilSetDistanceBound(_Objective):
    """Target the distance between the plasma and coilset.

    Will yield one or two values per coil in the coilset, depending on the mode
    variable, which is the minimum and/or maximum distance from that coil to the
    plasma boundary surface. If ``max`` or ``min`` mode is selected, only one value
    is returned. If ``bound`` mode is selected, two values are returned per coil,
    which are the minimum and maximum distance from the coil to the plasma boundary
    surface. The minima for all coils are returned first, then the maxima in a
    flattened array as applicable.

    NOTE: By default, assumes the plasma boundary is not fixed and its coordinates are
    computed at every iteration, for example if the equilibrium is changing in a
    single-stage optimization.
    If the plasma boundary is fixed, set eq_fixed=True to precompute the last closed
    flux surface coordinates and improve the efficiency of the calculation.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium (or FourierRZToroidalSurface) that will be optimized
        to satisfy the Objective.
    coil : CoilSet
        Coil(s) that are to be optimized.
    mode: string, optional
        One of ``bound``, ``min``, or ``max`` for bounding both min and max plasma-coil
        distance or targeting only min or max plasma-coil distance.
        Defaults to ``bound``.
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    coil_grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate coilset geometry at.
        Defaults to the default grid for the given coil-type, see ``coils.py``
        and ``curve.py`` for more details.
        If a list, must have the same structure as coils.
    eq_fixed: bool, optional
        Whether the equilibrium is fixed or not. If True, the last closed flux surface
        is fixed and its coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [coil] only.
        If False, the surface coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    coils_fixed: bool, optional
        Whether the coils are fixed or not. If True, the coils
        are fixed and their coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [eq] only.
        If False, the coil coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    use_softmin: bool, optional
        Use softmin (softmax) or hard min (max). Softmin is a smooth approximation to
        the actual minimum distance that may give smoother gradients, at the expense of
        being slightly more expensive and only an approximate extremum.
    softmin_alpha: float, optional
        Parameter used for softmin. The larger ``softmin_alpha``, the closer the
        softmin (softmax) approximates the hardmin (hardmax). softmin -> hardmin as
        ``softmin_alpha`` -> infinity.
    dist_chunk_size : int > 0, optional
        When computing distances, how many coils to consider at once. Default is all
        coils, which is generally the fastest but requires the most memory. If there are
        a large number of coils, or if the resolution is very high, setting this to a
        small value will reduce peak memory usage at the cost of slightly increased
        runtime.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0,1)``.",
        bounds_default="``bounds=(0,1)``.",
        coil=True,
    )
    _static_attrs = _Objective._static_attrs + [
        "_mode",
        "_eq_fixed",
        "_coils_fixed",
        "_use_softmin",
        "_dist_chunk_size",
        "_coil_indices",
    ]

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Plasma-coil distance: "

    def __init__(
        self,
        eq,
        coil,
        mode="bound",
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        plasma_grid=None,
        coil_grid=None,
        eq_fixed=False,
        coils_fixed=False,
        name="plasma-coil distance",
        jac_chunk_size=None,
        use_softmin=False,
        softmin_alpha=1.0,
        dist_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, 1)
        errorif(
            mode not in ["bound", "max", "min"],
            ValueError,
            "mode must be 'bound', 'max', or 'min'",
        )
        self._eq = eq
        self._coil = coil
        self._mode = mode
        self._plasma_grid = plasma_grid
        self._coil_grid = coil_grid
        self._eq_fixed = eq_fixed
        self._coils_fixed = coils_fixed
        self._use_softmin = use_softmin
        self._softmin_alpha = softmin_alpha
        self._dist_chunk_size = dist_chunk_size
        errorif(eq_fixed and coils_fixed, ValueError, "Cannot fix both eq and coil")
        things = []
        if not eq_fixed:
            things.append(eq)
        if not coils_fixed:
            things.append(coil)
        super().__init__(
            things=things,
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
        if self._eq_fixed:
            eq = self._eq
            coil = self.things[0]
        elif self._coils_fixed:
            eq = self.things[0]
            coil = self._coil
        else:
            eq = self.things[0]
            coil = self.things[1]
        default_M = 2 * eq.M if not hasattr(eq, "_M_grid") else eq.M_grid
        default_N = 2 * eq.N if not hasattr(eq, "_M_grid") else eq.N_grid
        plasma_grid = self._plasma_grid or LinearGrid(
            M=default_M, N=default_N, NFP=eq.NFP
        )
        coil_grid = self._coil_grid or None
        warnif(
            not np.allclose(plasma_grid.nodes[:, 0], 1),
            UserWarning,
            "Plasma/Surface grid includes interior points, should be rho=1.",
        )

        # The field-period copies of a coil are all the same distance from an
        # NFP-periodic plasma cloud, so evaluating every row returns each distinct
        # value NFP times -- inflating this objective's effective weight and costing
        # the same factor in compute. Keep one representative per tile. The
        # stellarator reflections are NOT deduplicated: they are only redundant if the
        # plasma cloud is reflection symmetric, which is not guaranteed.
        self._coil_indices = _field_period_independent_indices(coil, plasma_grid.NFP)
        if self._mode == "bound":
            self._dim_f = 2 * self._coil_indices.size
        else:  # min or max mode
            self._dim_f = self._coil_indices.size
        self._data_keys = ["R", "phi", "Z"]

        eq_profiles = get_profiles(self._data_keys, obj=eq, grid=plasma_grid)
        eq_transforms = get_transforms(self._data_keys, obj=eq, grid=plasma_grid)

        self._constants = {
            "eq": eq,
            "coil": coil,
            "coil_grid": coil_grid,
            "eq_profiles": eq_profiles,
            "eq_transforms": eq_transforms,
            "quad_weights": 1.0,
        }

        if self._eq_fixed:
            # precompute the equilibrium surface coordinates
            data = compute_fun(
                eq,
                self._data_keys,
                params=eq.params_dict,
                transforms=eq_transforms,
                profiles=eq_profiles,
            )
            rpz = jnp.array([data["R"], data["phi"], data["Z"]]).T
            rpz = copy_rpz_periods(rpz, plasma_grid.NFP)
            plasma_pts = rpz2xyz(rpz)
            self._constants["plasma_coords"] = plasma_pts
        if self._coils_fixed:
            coils_pts = coil._compute_position(params=coil.params_dict, grid=coil_grid)
            self._constants["coil_coords"] = coils_pts

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute minimum/maximum distance between coils and the plasma/surface.

        Parameters
        ----------
        params_1 : dict
            Dictionary of coilset degrees of freedom, eg ``CoilSet.params_dict`` if
            self._coils_fixed is False, else is the equilibrium or surface degrees of
            freedom
        params_2 : dict
            Dictionary of equilibrium or surface degrees of freedom,
            eg ``Equilibrium.params_dict``
            Only required if ``self._eq_fixed = False``.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Minimum/maximum distance from coil to surface for each coil in the coilset.

        """
        constants = self._get_deprecated_constants(constants)
        if self._eq_fixed:
            coils_params = params_1
        elif self._coils_fixed:
            eq_params = params_1
        else:
            eq_params = params_1
            coils_params = params_2

        # coil pts; shape(ncoils,coils_grid.num_nodes,3)
        if self._coils_fixed:
            coils_pts = constants["coil_coords"]
        else:
            coils_pts = constants["coil"]._compute_position(
                params=coils_params, grid=constants["coil_grid"]
            )

        # plasma pts; shape(plasma_grid.num_nodes,3)
        if self._eq_fixed:
            plasma_pts = constants["plasma_coords"]
        else:
            data = compute_fun(
                constants["eq"],
                self._data_keys,
                params=eq_params,
                transforms=constants["eq_transforms"],
                profiles=constants["eq_profiles"],
            )
            rpz = jnp.array([data["R"], data["phi"], data["Z"]]).T
            rpz = copy_rpz_periods(rpz, constants["eq_transforms"]["grid"].NFP)
            plasma_pts = rpz2xyz(rpz)

        def body(k):
            # dist btwn all pts; shape(ncoils,plasma_grid.num_nodes,coil_grid.num_nodes)
            dist = safenorm(coils_pts[k][None, :, :] - plasma_pts[:, None, :], axis=-1)
            if self._use_softmin:
                # minimum over plasma points, then max over coil points
                max = softmax(
                    softmin(dist, self._softmin_alpha, axis=0), self._softmin_alpha
                )
                # minimum over all points
                min = softmin(dist, self._softmin_alpha)
            else:
                max = jnp.max(jnp.min(dist, axis=0))
                min = jnp.min(dist)

            if self._mode == "max":
                return max
            if self._mode == "min":
                return min
            return jnp.array([min, max])

        k = self._coil_indices

        extreme_dist_per_coil = vmap_chunked(body, chunk_size=self._dist_chunk_size)(k)

        # if mode is bound, flatten the output
        extreme_dist_per_coil = extreme_dist_per_coil.flatten()

        return extreme_dist_per_coil


class PlasmaCoilSetMinDistance(PlasmaCoilSetDistanceBound):
    """Target the minimum distance between the plasma and coilset.

    Will yield one value per coil in the coilset, which is the minimum distance from
    that coil to the plasma boundary surface.

    NOTE: By default, assumes the plasma boundary is not fixed and its coordinates are
    computed at every iteration, for example if the equilibrium is changing in a
    single-stage optimization.
    If the plasma boundary is fixed, set eq_fixed=True to precompute the last closed
    flux surface coordinates and improve the efficiency of the calculation.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium (or FourierRZToroidalSurface) that will be optimized
        to satisfy the Objective.
    coil : CoilSet
        Coil(s) that are to be optimized.
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    coil_grid : Grid, list, optional
        Collocation grid containing the nodes to evaluate coilset geometry at.
        Defaults to the default grid for the given coil-type, see ``coils.py``
        and ``curve.py`` for more details.
        If a list, must have the same structure as coils.
    eq_fixed: bool, optional
        Whether the equilibrium is fixed or not. If True, the last closed flux surface
        is fixed and its coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [coil] only.
        If False, the surface coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    coils_fixed: bool, optional
        Whether the coils are fixed or not. If True, the coils
        are fixed and their coordinates are precomputed, which saves on computation time
        during optimization, and self.things = [eq] only.
        If False, the coil coordinates are computed at every iteration.
        False by default, so that self.things = [coil, eq].
    use_softmin: bool, optional
        Use softmin or hard min. Softmin is a smooth approximation to the actual minimum
        distance that may give smoother gradients, at the expense of being slightly more
        expensive and only an approximate minimum.
    softmin_alpha: float, optional
        Parameter used for softmin. The larger ``softmin_alpha``, the closer the
        softmin approximates the hardmin. softmin -> hardmin as
        ``softmin_alpha`` -> infinity.
    dist_chunk_size : int > 0, optional
        When computing distances, how many coils to consider at once. Default is all
        coils, which is generally the fastest but requires the most memory. If there are
        a large number of coils, or if the resolution is very high, setting this to a
        small value will reduce peak memory usage at the cost of slightly increased
        runtime.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(1,np.inf)``.",
        bounds_default="``bounds=(1,np.inf)``.",
        coil=True,
    )

    _static_attrs = PlasmaCoilSetDistanceBound._static_attrs
    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Minimum plasma-coil distance: "

    def __init__(
        self,
        eq,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        plasma_grid=None,
        coil_grid=None,
        eq_fixed=False,
        coils_fixed=False,
        name="plasma-coil minimum distance",
        jac_chunk_size=None,
        use_softmin=False,
        softmin_alpha=1.0,
        dist_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
        super().__init__(
            eq=eq,
            coil=coil,
            mode="min",
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=eq_fixed,
            coils_fixed=coils_fixed,
            name=name,
            jac_chunk_size=jac_chunk_size,
            use_softmin=use_softmin,
            softmin_alpha=softmin_alpha,
            dist_chunk_size=dist_chunk_size,
        )


def _point_segment_distance(q, p, d):
    """Distance from points ``q`` to segments ``p + s d``, s in [0, 1] (broadcasting).

    The squared distance to a segment is C^1 in all arguments (envelope theorem: the
    closest point is the projection onto a convex set), so the distance is C^1 wherever
    it is positive.
    """
    w = q - p
    s = jnp.clip(jnp.sum(w * d, axis=-1) / jnp.sum(d * d, axis=-1), 0.0, 1.0)
    return safenorm(w - s[..., None] * d, axis=-1)


def _warn_active_overflow(counts, cap):
    counts = np.asarray(counts)
    if counts.size and counts.max() > cap:
        warnings.warn(
            f"PlasmaCoilSetDistancePenalty: {int(counts.max())} pairs of one coil are "
            f"within d_min but max_active_pairs={cap}. The pairs past the first {cap} "
            "were dropped, so the value and its derivatives are inexact; raise "
            "max_active_pairs.",
            UserWarning,
        )


class PlasmaCoilSetDistancePenalty(_Objective):
    """Smooth threshold penalty on plasma-coil distance: one C^1 row per coil.

    For each field-period-independent coil k this returns

        g_k = sum_{i, j} max(0, 1 - d_ij / d_min)**2

    over coil segments (or nodes) i and plasma surface points j, where d_ij is their
    distance. g_k >= 0 everywhere and g_k = 0 exactly when every pair is at least
    ``d_min`` apart, so the constraint is ``g <= 0``: the default ``bounds=(-inf, 0)``.

    Why not a minimum. ``PlasmaCoilSetMinDistance`` returns min_{i,j} d_ij, which is
    kinked wherever the closest pair changes: along the coil, or from one plasma grid
    node to the next. MEASURED on precise_QH planarN7 (qh_feas_work/qh_split_planarN7,
    lsq_auglag ``diag_block_split_from``): with that bound active, all 24 poor
    trust-region steps (ratio < 0.9) had 100% of their model error in the minimum's
    block, appearing partway along the step, and the trust radius shrank to 1e-4. A sum
    of ``max(0, .)**2`` terms has no minimum to switch and is C^1. As for
    ``CoilSetDistancePenalty``, its gradient vanishes at the boundary, so the constraint
    is degenerate there; that is what ``lsq_auglag(second_order="constraints")``
    addresses.

    Two-pass evaluation. Only pairs closer than ``d_min`` contribute, but
    differentiating through the full (plasma points x coil segments) array costs memory
    in proportion to its size: the constraint Hessian of the hard minimum needed ~23 GB
    for 33k plasma points (MEASURED). So each evaluation first computes every distance
    with no derivative (``stop_gradient``) and keeps the indices of each coil's pairs
    within ``d_min``, in a list padded to ``max_active_pairs``; it then recomputes only
    those distances, with derivatives. The selection is piecewise constant and every
    dropped pair contributes exactly zero, so the value, gradient and Hessian are exact
    whenever at most ``max_active_pairs`` pairs of a coil are within ``d_min``. Beyond
    that, the active pairs past the first ``max_active_pairs`` (in index order) are
    dropped and a warning is raised; ``active_pair_counts`` measures the counts.
    MEASURED on precise_QH planarN7 with d_pc at 0.76 of its bound: at most 2240 active
    pairs per coil, of 4.3M.

    Rows and symmetry. As ``PlasmaCoilSetMinDistance``: one row per coil of
    ``_field_period_independent_indices`` (stellarator reflections kept), with the
    plasma points copied over all field periods. A ``sym=True`` plasma grid covers theta
    in [0, pi] only. For a stellarator-symmetric coilset, the rows of a coil and of its
    reflection together still see the whole surface; otherwise pass a ``sym=False``
    grid.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Plasma boundary. It is held fixed: only ``eq_fixed=True`` is supported.
    coil : CoilSet
        Coil(s) that are to be optimized.
    d_min : float
        Distance threshold [m]. Pairs closer than this are penalized.
    plasma_grid : Grid, optional
        Surface grid (rho=1). Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid,
        NFP=eq.NFP)``.
    coil_grid : Grid, optional
        Collocation grid used to discretize each coil (nodes ordered along the curve).
    eq_fixed : bool
        Must be True.
    max_active_pairs : int, optional
        Capacity of the selection per coil. Default 20000. The value and Jacobian barely
        depend on it, but a dense Hessian (``lsq_auglag`` ``second_order``)
        differentiates
        every slot, padding included. MEASURED (8 rows, 4.3M pairs per coil, chunk 16):
        K 20000 -> 0.9 s / 2.3 GB peak, 100000 -> 3.4 s / 3.5 GB, 400000 -> 13.5 s / 8.9
        GB.
    dist_chunk_size : int > 0, optional
        Coils handled at once in the selection pass (the full distance array of one coil
        is plasma points x coil nodes). Default: all.
    distance_method : {"segment", "point"}, optional
        Distance from each plasma point to coil segments (default; resolves approaches
        between coil nodes) or to coil nodes.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-np.inf, 0)``.",
        bounds_default="``bounds=(-np.inf, 0)``.",
        coil=True,
    )

    _static_attrs = _Objective._static_attrs + [
        "_dist_chunk_size",
        "_coil_indices",
        "_distance_method",
        "_max_active_pairs",
        "_n_coil_nodes",
    ]

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Plasma-coil distance penalty: "

    def __init__(
        self,
        eq,
        coil,
        d_min,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        plasma_grid=None,
        coil_grid=None,
        eq_fixed=True,
        name="plasma-coil distance penalty",
        jac_chunk_size=None,
        max_active_pairs=20000,
        dist_chunk_size=None,
        distance_method="segment",
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, 0.0)
        errorif(
            not eq_fixed,
            ValueError,
            "PlasmaCoilSetDistancePenalty supports a fixed plasma boundary only",
        )
        errorif(not d_min > 0, ValueError, f"d_min must be positive, got {d_min}")
        errorif(
            distance_method not in ("point", "segment"),
            ValueError,
            f'distance_method must be "point" or "segment", got {distance_method}',
        )
        errorif(
            int(max_active_pairs) < 1,
            ValueError,
            f"max_active_pairs must be a positive integer, got {max_active_pairs}",
        )
        self._eq = eq
        self._d_min = d_min
        self._plasma_grid = plasma_grid
        self._coil_grid = coil_grid
        self._max_active_pairs = int(max_active_pairs)
        self._dist_chunk_size = dist_chunk_size
        self._distance_method = distance_method
        super().__init__(
            things=coil,
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
        eq = self._eq
        coil = self.things[0]
        default_M = 2 * eq.M if not hasattr(eq, "_M_grid") else eq.M_grid
        default_N = 2 * eq.N if not hasattr(eq, "_M_grid") else eq.N_grid
        plasma_grid = self._plasma_grid or LinearGrid(
            M=default_M, N=default_N, NFP=eq.NFP
        )
        warnif(
            not np.allclose(plasma_grid.nodes[:, 0], 1),
            UserWarning,
            "Plasma/Surface grid includes interior points, should be rho=1.",
        )
        coil_grid = self._coil_grid or None
        self._coil_indices = _field_period_independent_indices(coil, plasma_grid.NFP)
        self._dim_f = self._coil_indices.size

        keys = ["R", "phi", "Z"]
        transforms = get_transforms(keys, obj=eq, grid=plasma_grid)
        profiles = get_profiles(keys, obj=eq, grid=plasma_grid)
        data = compute_fun(
            eq, keys, params=eq.params_dict, transforms=transforms, profiles=profiles
        )
        rpz = jnp.array([data["R"], data["phi"], data["Z"]]).T
        plasma_pts = rpz2xyz(copy_rpz_periods(rpz, plasma_grid.NFP))

        pts0 = coil._compute_position(params=coil.params_dict, grid=coil_grid)
        self._n_coil_nodes = int(pts0.shape[1])
        self._max_active_pairs = min(
            self._max_active_pairs, int(plasma_pts.shape[0]) * self._n_coil_nodes
        )
        self._constants = {
            "coil": coil,
            "coil_grid": coil_grid,
            "plasma_coords": plasma_pts,
            "quad_weights": 1.0,
        }
        if self._normalize:
            self._normalization = 1.0  # already dimensionless

        super().build(use_jit=use_jit, verbose=verbose)

    def _all_distances(self, coil_pts, plasma_pts):
        """Distances from every plasma point to one coil, flattened (points x nodes)."""
        if self._distance_method == "segment":
            p, d = _closed_polyline(coil_pts)
            dist = _point_segment_distance(plasma_pts[:, None, :], p[None], d[None])
        else:
            dist = safenorm(plasma_pts[:, None, :] - coil_pts[None], axis=-1)
        return dist.reshape(-1)

    def _select(self, coil_pts, plasma_pts):
        """First pass, no derivatives: one coil's pairs within d_min, and their count.

        The indices are padded to max_active_pairs.

        A fixed-size ``nonzero`` of the mask, not ``top_k`` of the distances: MEASURED
        on 4.3M pairs, top_k(K=20000) took 0.72 s per coil against 0.03 s.
        """
        dist = self._all_distances(jax.lax.stop_gradient(coil_pts), plasma_pts)
        active = dist < self._d_min
        (idx,) = jnp.nonzero(active, size=self._max_active_pairs, fill_value=0)
        return idx, jnp.sum(active)

    def compute(self, params, constants=None):
        """Compute the plasma-coil distance penalty of each coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            sum over (coil segment or node, plasma point) pairs of
            max(0, 1 - d/d_min)**2, per coil.

        """
        constants = self._get_deprecated_constants(constants)
        pts = constants["coil"]._compute_position(
            params=params, grid=constants["coil_grid"]
        )[self._coil_indices]
        plasma_pts = constants["plasma_coords"]

        idx, counts = vmap_chunked(
            lambda c: self._select(c, plasma_pts), chunk_size=self._dist_chunk_size
        )(pts)
        cap = self._max_active_pairs
        jax.debug.callback(lambda n: _warn_active_overflow(n, cap), counts)

        def penalty(c, idx, n):
            # second pass: only the selected pairs, with derivatives; slots past the n
            # active pairs are padding (index 0) and are masked out
            q = plasma_pts[idx // self._n_coil_nodes]
            i = idx % self._n_coil_nodes
            if self._distance_method == "segment":
                p, d = _closed_polyline(c)
                dist = _point_segment_distance(q, p[i], d[i])
            else:
                dist = safenorm(q - c[i], axis=-1)
            valid = jnp.arange(cap) < n
            return jnp.sum(
                jnp.where(valid, jnp.maximum(0.0, 1.0 - dist / self._d_min) ** 2, 0.0)
            )

        return jax.vmap(penalty)(pts, idx, counts)

    def active_pair_counts(self, params=None):
        """Pairs within ``d_min`` for each row's coil, evaluated eagerly.

        Use it to size ``max_active_pairs``.
        """
        coil = self.things[0]
        params = coil.params_dict if params is None else params
        pts = coil._compute_position(params=params, grid=self._constants["coil_grid"])
        plasma_pts = self._constants["plasma_coords"]
        return np.array(
            [
                int(jnp.sum(self._all_distances(pts[k], plasma_pts) < self._d_min))
                for k in self._coil_indices
            ]
        )


class CoilArclengthVariance(_CoilObjective):
    """Variance of ||dx/ds|| along the curve.

    This objective is meant to combat any issues corresponding to non-uniqueness of
    the representation of a curve, in that the same physical curve can be represented
    by different parametrizations by changing the curve parameter [1]_. Note that this
    objective has no effect for ``FourierRZCoil`` and ``FourierPlanarCoil`` which have a
    single unique parameterization (the objective will always return 0 for these types).

    References
    ----------
    .. [1] Wechsung, et al. "Precise stellarator quasi-symmetry can be achieved
       with electromagnetic coils." PNAS (2022)

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``.", coil=True
    )

    _scalar = False  # Not always a scalar, if a coilset is passed in
    _units = "(m^2)"
    _print_value_fmt = "Coil Arclength Variance: "
    _broadcast_input = "coil"

    def __init__(
        self,
        coils,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil arclength variance",
    ):
        if target is None and bounds is None:
            target = 0

        super().__init__(
            coils,
            ["x_s"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
            name=name,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1

        coilset = self.things[0]
        # local import to avoid circular import
        from desc.coils import (
            CoilSet,
            FourierXYCoil,
            FourierXYZCoil,
            SplineXYZCoil,
            _Coil,
        )

        def _is_single_coil(c):
            return isinstance(c, _Coil) and not isinstance(c, CoilSet)

        coils = tree_leaves(coilset, is_leaf=_is_single_coil)
        self._constants["mask"] = np.array(
            [
                int(isinstance(coil, (FourierXYZCoil, SplineXYZCoil, FourierXYCoil)))
                for coil in coils
            ]
        )

        if self._normalize:
            self._normalization = np.mean([scale["a"] ** 2 for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil arclength variance.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : float or array of floats
            Coil arclength variance.
        """
        data = super().compute(params, constants=constants)
        constants = self._get_deprecated_constants(constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.array([jnp.var(jnp.linalg.norm(dat["x_s"], axis=1)) for dat in data])
        return (out * constants["mask"])[self._coilset_tree["objective_mask"]]


class CoilArclengthResidual(CoilArclengthVariance):
    """Coil arclength residuals, one per grid node.

    Each node of each coil gives ``|x_s|_i - mean(|x_s|)``, with quadrature weight
    ``sqrt(ds / 2 pi)``, so the weighted sum of squares over a coil is that coil's
    arclength variance (``CoilArclengthVariance``) at any grid resolution.
    Prefer this form with least-squares optimizers: passing the variance as a single
    residual squares it again, so the cost is quartic in the deviation and its
    Gauss-Newton Jacobian vanishes at the target.

    As for ``CoilArclengthVariance``, only coils without a unique parameterization
    (FourierXYZ, SplineXYZ, FourierXY) contribute; the rows of other coils are zero.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _units = "(m)"
    _print_value_fmt = "Coil arclength residual: "
    _broadcast_input = "node"

    def __init__(
        self,
        coils,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil arclength residual",
    ):
        super().__init__(
            coils,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
            name=name,
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
        # the variance build replaces the per-node quadrature weights with 1
        _CoilObjective.build(self, use_jit=use_jit, verbose=verbose)
        quad_weights = self._constants["quad_weights"] / np.sqrt(2 * np.pi)
        super().build(use_jit=use_jit, verbose=verbose)
        self._constants["quad_weights"] = quad_weights
        if self._normalize:  # residuals have units of length, not length^2
            self._normalization = np.mean([scale["a"] for scale in self._scales])
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil arclength residuals.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Arclength residuals at each node of each coil.
        """
        data = _CoilObjective.compute(self, params, constants=constants)
        constants = self._get_deprecated_constants(constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = []
        for k, dat in enumerate(data):
            sp = jnp.linalg.norm(dat["x_s"], axis=1)
            out.append(constants["mask"][k] * (sp - jnp.mean(sp)))
        return jnp.concatenate(out)[self._coilset_tree["objective_mask"]]


class QuadraticFlux(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n. The equilibrium is kept fixed while the
    field is unfixed.

    Note: This objective is intended for coil optimization. For finding the surface
    that minimizes the normal field error, use the SurfaceQuadraticFlux objective.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error
        will be minimized. The equilibrium is kept fixed during the optimization
        with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
    source_grid : Grid, optional
        Collocation grid containing the nodes for plasma source terms.
        Default grid is detailed in the docs for ``compute_B_plasma``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid,
        NFP=eq.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    vacuum : bool
        If true, B_plasma (the contribution to the normal field on the boundary from the
        plasma currents) is set to zero.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation for B_plasma into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_B_plasma_chunk_size",
        "_bs_chunk_size",
        "_vacuum",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        name="Quadratic flux",
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        **kwargs,
    ):
        from desc.geometry import FourierRZToroidalSurface

        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = [field] if not isinstance(field, list) else field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)
        errorif(
            isinstance(eq, FourierRZToroidalSurface),
            TypeError,
            "Detected FourierRZToroidalSurface object "
            "if attempting to find a QFM surface, please use "
            "SurfaceQuadraticFlux objective instead.",
        )
        super().__init__(
            things=self._field,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        from desc.magnetic_fields import SumMagneticField

        eq = self._eq

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        eval_data = compute_fun(
            eq,
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        # pre-compute B_plasma because we are assuming eq is fixed
        Bplasma = (
            jnp.zeros(eval_grid.num_nodes)
            if self._vacuum
            else compute_B_plasma(
                eq,
                eval_grid,
                self._source_grid,
                normal_only=True,
                chunk_size=self._B_plasma_chunk_size,
            )
        )

        self._constants = {
            "field": SumMagneticField(self._field),
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_plasma": Bplasma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *field_params, constants=None):
        """Compute normal field error on boundary.

        Parameters
        ----------
        field_params : dict
            Dictionary of the external field's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Bnorm from B_ext and B_plasma

        """
        constants = self._get_deprecated_constants(constants)

        # B_plasma from equilibrium precomputed
        eval_data = constants["eval_data"]
        B_plasma = constants["B_plasma"]

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_ext = constants["field"].compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
            chunk_size=self._bs_chunk_size,
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = (B_ext + B_plasma) * jnp.sqrt(eval_data["|e_theta x e_zeta|"])
        return f


class SurfaceQuadraticFlux(_Objective):
    """Target B*n = 0 on a surface.

    Used to find a quadratic-flux-minimizing (QFM) surface, so a
    `FourierRZToroidalSurface` should be passed to the objective.
    Should always be used along with a ``ToroidalFlux`` or ``Volume`` objective to
    ensure that the resulting QFM surface has the desired amount of
    flux enclosed and avoid trivial solutions.

    Note: This objective can be used with ``field_fixed=True`` to find the QFM surface
    by fixing the coils, however the surface is always free to change. For coil
    optimization, use the ``QuadraticFlux`` objective.

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + ["_bs_chunk_size", "_field_fixed"]

    _scalar = False
    _linear = False
    _print_value_fmt = "QFM surface normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        surface,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        eval_grid=None,
        field_grid=None,
        name="Surface Quadratic Flux",
        field_fixed=False,
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0
        self._eval_grid = eval_grid
        self._surface = surface
        self._field = field
        self._field_grid = field_grid
        self._field_fixed = field_fixed
        self._bs_chunk_size = bs_chunk_size

        things = [surface]
        if not field_fixed:
            things += [field]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        surface = self._surface

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=2 * surface.M,
                N=2 * surface.N,
                NFP=surface.NFP,
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=surface, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=surface, grid=eval_grid)
        eval_data = compute_fun(
            surface,
            self._data_keys,
            params=surface.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        self._constants = {
            "field": self._field,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(surface)
            Bscale = 1.0  # surface has no inherent B scale
            self._normalization = Bscale * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute normal field on surface.

        Parameters
        ----------
        params_1 : dict
            Dictionary of the surface's degrees of freedom.
        params_2 : dict
            Dictionary of the external field's degrees of freedom, only provided if
            if field_fixed=False.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Bnorm on the QFM surface from the external field

        """
        constants = self._get_deprecated_constants(constants)
        field_params = params_2 if not self._field_fixed else None
        surf_params = params_1

        eval_data = compute_fun(
            self._surface,
            self._data_keys,
            surf_params,
            constants["eval_transforms"],
            constants["eval_profiles"],
        )
        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        if field_params is None:
            field_params = constants["field"].params_dict
        B_ext = constants["field"].compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
            chunk_size=self._bs_chunk_size,
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
        f = B_ext * jnp.sqrt(eval_data["|e_theta x e_zeta|"])
        return f


class ToroidalFlux(_Objective):
    """Target the toroidal flux in an equilibrium from a magnetic field.

    This objective is needed when performing stage-two coil optimization on
    a vacuum equilibrium, to avoid the trivial solution of minimizing Bn
    by making the coil currents zero. Instead, this objective ensures
    the coils create the necessary toroidal flux for the equilibrium field.

    Will try to use the vector potential method to calculate the toroidal flux
    (Φ = ∮ 𝐀 ⋅ 𝐝𝐥 over the perimeter of a constant zeta plane)
    instead of the brute force method using the magnetic field
    (Φ = ∯ 𝐁 ⋅ 𝐝𝐒 over a constant zeta XS). The vector potential method
    is much more efficient, however not every ``MagneticField`` object
    has a vector potential available to compute, so in those cases
    the magnetic field method is used.

    Parameters
    ----------
    eq : Equilibrium or FourierRZToroidalSurface
        Equilibrium (or QFM surface) for which the toroidal flux will be calculated.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).
    field_fixed : bool
        Whether to fix the field's DOFs during the optimization.
    eq_fixed : bool
        Whether to fix the equilibrium (or QFM surface) DOFs
        during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default=(
            "``target=eq.Psi`` if an Equilibrium is passed,"
            + " or ``target=1.0`` if a surface."
        ),
        bounds_default=(
            "``target=eq.Psi`` if an Equilibrium is passed,"
            + " or ``target=1.0`` if a surface."
        ),
        loss_detail=" Note: has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_eq_fixed",
        "_field_fixed",
        "_use_vector_potential",
    ]

    _coordinates = "rtz"
    _units = "(Wb)"
    _print_value_fmt = "Toroidal Flux: "

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        field_grid=None,
        eval_grid=None,
        name="toroidal-flux",
        field_fixed=False,
        eq_fixed=False,
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 1.0 if not hasattr(eq, "Psi") else eq.Psi
        self._field = field
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field_fixed = field_fixed
        self._eq_fixed = eq_fixed
        self._bs_chunk_size = bs_chunk_size
        errorif(
            eq_fixed and field_fixed,
            ValueError,
            "Cannot have both `field_fixed=True` and `eq_fixed=True`",
        )
        things = []
        if not eq_fixed:
            things += [eq]
        if not field_fixed:
            things += [field]
        super().__init__(
            things=things,
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
        from desc.geometry import FourierRZToroidalSurface

        eq = self._eq
        self._use_vector_potential = True
        try:
            self._field.compute_magnetic_vector_potential(
                [0, 0, 0], chunk_size=self._bs_chunk_size
            )
        except (NotImplementedError, ValueError) as e:
            self._use_vector_potential = False
            errorif(
                isinstance(eq, FourierRZToroidalSurface)
                and not self._use_vector_potential,
                ValueError,
                "Targeting a QFM surface requires the vector potential to be "
                "calculated from the field, however the field cannot calculate "
                f"the vector potential, encountered error {e}",
            )
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid if not self._use_vector_potential else 0,
                M=eq.M_grid if hasattr(eq, "M_grid") else 3 * eq.M,
                zeta=jnp.array(0.0),
                NFP=eq.NFP,
            )
            self._eval_grid = eval_grid
        eval_grid = self._eval_grid

        errorif(
            not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]),
            ValueError,
            "Evaluation grid should be at constant zeta",
        )
        if self._normalize:
            self._normalization = 1.0 if not hasattr(eq, "Psi") else eq.Psi
        if not isinstance(eq, FourierRZToroidalSurface):
            # ensure vacuum eq, as is unneeded for finite beta
            pres = np.max(np.abs(eq.compute("p")["p"]))
            curr = np.max(np.abs(eq.compute("current")["current"]))
            warnif(
                pres > 1e-8,
                UserWarning,
                f"Pressure appears to be non-zero (max {pres} Pa), "
                + "this objective is unneeded at finite beta.",
            )
            warnif(
                curr > 1e-8,
                UserWarning,
                f"Current appears to be non-zero (max {curr} A), "
                + "this objective is unneeded at finite beta.",
            )

        # eval_grid.num_nodes for quad flux cost
        self._dim_f = 1
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        data_keys = ["R", "phi", "Z"]
        if self._use_vector_potential:
            data_keys += ["e_theta"]
        else:
            data_keys += ["|e_rho x e_theta|", "n_zeta"]
        self._data_keys = data_keys
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)
        data = compute_fun(
            eq,
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        self._constants = {
            "plasma_coords": plasma_coords,
            "equil_data": data,
            "quad_weights": 1.0,
            "field": self._field,
            "field_grid": self._field_grid,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "eval_grid": eval_grid,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        params_1 : dict
            Dictionary of the external field's degrees of freedom, or the surface's
            degrees of freedom if qfm_surface=True.
        params_2 : dict
            Dictionary of the external field's degrees of freedom, if qfm_surface=True.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : float
            Toroidal flux from coils and external field

        """
        constants = self._get_deprecated_constants(constants)
        field_params = params_2 if not self._eq_fixed else params_1
        field_params = (
            constants["field"].params_dict if self._field_fixed else field_params
        )
        surf_params = params_1 if not self._eq_fixed else None

        if not self._eq_fixed:
            data = compute_fun(
                self._eq,
                self._data_keys,
                surf_params,
                constants["eval_transforms"],
                constants["eval_profiles"],
            )
            plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T
        else:
            data = constants["equil_data"]
            plasma_coords = constants["plasma_coords"]

        grid = constants["eval_grid"]

        if self._use_vector_potential:
            A = constants["field"].compute_magnetic_vector_potential(
                plasma_coords,
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
                chunk_size=self._bs_chunk_size,
            )

            A_dot_e_theta = jnp.sum(A * data["e_theta"], axis=1)
            Psi = jnp.sum(grid.spacing[:, 1] * A_dot_e_theta)
        else:
            B = constants["field"].compute_magnetic_field(
                plasma_coords,
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
                chunk_size=self._bs_chunk_size,
            )

            B_dot_n_zeta = jnp.sum(B * data["n_zeta"], axis=1)
            Psi = jnp.sum(
                grid.spacing[:, 0]
                * grid.spacing[:, 1]
                * data["|e_rho x e_theta|"]
                * B_dot_n_zeta
            )

        return Psi


class LinkingCurrentConsistency(_Objective):
    """Target the self-consistent poloidal linking current between the plasma and coils.

    A self-consistent coil + plasma configuration must have the sum of the signed
    currents in the coils that poloidally link the plasma equal to the total poloidal
    current required to be linked by the plasma according to the loop integral of its
    toroidal magnetic field, given by `G(rho=1)`. This objective computes the difference
    between these two quantities, such that a value of zero means the coils create the
    correct net poloidal current.

    Assumes the coil topology does not change (ie the linking number with the plasma
    is fixed).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    coil : CoilSet
        Coil(s) that are to be optimized.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma current at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)``.
    eq_fixed : bool
        Whether the equilibrium is assumed fixed (should be true for stage 2, false
        for single stage).
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + ["_eq_fixed"]

    _scalar = True
    _units = "(A)"
    _print_value_fmt = "Linking current error: "

    def __init__(
        self,
        eq,
        coil,
        *,
        grid=None,
        eq_fixed=False,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="linking current",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._eq_fixed = eq_fixed
        self._linear = eq_fixed
        self._eq = eq
        self._coil = coil

        super().__init__(
            things=[coil] if eq_fixed else [coil, eq],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
            name=name,
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
        eq = self._eq
        coil = self._coil
        grid = self._grid or LinearGrid(
            M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        warnif(
            not np.allclose(grid.nodes[:, 0], 1),
            UserWarning,
            "grid includes interior points, should be rho=1.",
        )

        self._dim_f = 1
        self._data_keys = ["G"]

        all_params = tree_map(lambda dim: np.arange(dim), coil.dimensions)
        current_params = tree_map(lambda idx: {"current": idx}, True)
        # indices of coil currents
        self._indices = tree_leaves(broadcast_tree(current_params, all_params))
        self._num_coils = coil.num_coils

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        # compute linking number of coils with plasma. To do this we add a fake "coil"
        # along the magnetic axis and compute the linking number of that coilset
        from desc.coils import FourierRZCoil, MixedCoilSet

        axis_coil = FourierRZCoil(
            1.0,
            eq.axis.R_n,
            eq.axis.Z_n,
            eq.axis.R_basis.modes[:, 2],
            eq.axis.Z_basis.modes[:, 2],
            eq.axis.NFP,
        )
        dummy_coilset = MixedCoilSet(axis_coil, coil, check_intersection=False)
        # linking number for coils with axis
        link = np.round(dummy_coilset._compute_linking_number())[0, 1:]

        self._constants = {
            "quad_weights": 1.0,
            "link": link,
        }

        if self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,
                params=eq.params_dict,
                transforms=transforms,
                profiles=profiles,
            )
            eq_linking_current = 2 * jnp.pi * data["G"][0] / mu_0
            self._constants["eq_linking_current"] = eq_linking_current
        else:
            self._constants["profiles"] = profiles
            self._constants["transforms"] = transforms

        if self._normalize:
            params = tree_leaves(
                coil.params_dict, is_leaf=lambda x: isinstance(x, dict)
            )
            self._normalization = np.sum([np.abs(param["current"]) for param in params])

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, coil_params, eq_params=None, constants=None):
        """Compute linking current error.

        Parameters
        ----------
        coil_params : dict
            Dictionary of coilset degrees of freedom, eg ``CoilSet.params_dict``
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg ``Equilibrium.params_dict``
            Only required if eq_fixed=False.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            Linking current error.

        """
        constants = self._get_deprecated_constants(constants)
        if self._eq_fixed:
            eq_linking_current = constants["eq_linking_current"]
        else:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,
                params=eq_params,
                transforms=constants["transforms"],
                profiles=constants["profiles"],
            )
            eq_linking_current = 2 * jnp.pi * data["G"][0] / mu_0

        coil_currents = jnp.concatenate(
            [
                jnp.atleast_1d(param[idx])
                for param, idx in zip(tree_leaves(coil_params), self._indices)
            ]
        )
        coil_currents = self.things[0]._all_currents(coil_currents)
        coil_linking_current = jnp.sum(constants["link"] * coil_currents)
        return eq_linking_current - coil_linking_current


class CoilSetLinkingNumber(_Objective):
    """Prevents coils from becoming interlinked.

    The linking number of 2 curves is (approximately) 0 if they are not linked, and
    (approximately) +/-1 if they are (with the sign indicating the helicity of the
    linking).

    This objective returns a single value for each coil in the coilset, with that number
    being the sum of the absolute value of the linking numbers of that coil with every
    other coil in the coilset, approximating the number of other coils that are linked

    Parameters
    ----------
    coil : CoilSet
        Coil(s) that are to be optimized.
    grid : Grid, list, optional
        Collocation grid used to discretize each coil. Defaults to
        ``LinearGrid(N=50)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        coil=True,
    )

    _static_attrs = _Objective._static_attrs + ["_coil_indices"]

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil linking number: "

    def __init__(
        self,
        coil,
        grid=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="coil-coil linking number",
    ):
        from desc.coils import CoilSet

        if target is None and bounds is None:
            target = 0
        self._grid = grid
        errorif(
            not isinstance(coil, CoilSet),
            ValueError,
            "coil must be of type CoilSet, not an individual Coil",
        )
        super().__init__(
            things=coil,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
            name=name,
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
        coilset = self.things[0]
        grid = self._grid or LinearGrid(N=50)

        # linking number depends only on coilset geometry, and the coilset maps to
        # itself under its own symmetry group, so every copy of a coil links the set
        # identically. Report one value per independent coil.
        self._coil_indices = _independent_coil_indices(coilset)
        self._dim_f = self._coil_indices.size
        self._constants = {"coilset": coilset, "grid": grid, "quad_weights": 1.0}

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute linking numbers between coils.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : array of floats
            For each coil, the sum of the absolute value of the linking numbers between
            that coil and every other coil in the coilset, which approximates the
            number of coils linked with that coil.

        """
        constants = self._get_deprecated_constants(constants)
        link = constants["coilset"]._compute_linking_number(
            params=params, grid=constants["grid"], indices=self._coil_indices
        )
        # `link[i, k]` is the linking number of coil i with the representative coil
        # `_coil_indices[k]`, so the `i == _coil_indices[k]` entries are each coil's
        # Gauss integral with ITSELF -- its writhe. That is nonzero for any non-planar
        # coil (it converges to a finite value under grid refinement rather than to 0)
        # and says nothing about whether coils are interlinked. Leaving it in makes
        # `target=0` unsatisfiable and turns this objective into a penalty on coil
        # non-planarity. Mask it out so the result is what the docstring says: the sum
        # over every OTHER coil.
        link = link.at[self._coil_indices, jnp.arange(self._dim_f)].set(0.0)

        # the diagonal entries of "link" should be excluded
        mask = ~jnp.eye(self._dim_f, dtype=bool)
        return jnp.abs(link).sum(axis=0, where=mask)


class SurfaceCurrentRegularization(_Objective):
    """Target the surface current magnitude.

    If ``regularization="K"``:

    compute::

        w * ||K|| * sqrt(||e_theta x e_zeta||)

    where K is the winding surface current density, w is the
    regularization parameter (the weight on this objective),
    and ||e_theta x e_zeta|| is the magnitude of the surface normal i.e. the
    surface jacobian ||e_theta x e_zeta||

    This is intended to be used with a surface current::

        K = n x ∇ Φ

    i.e. a CurrentPotentialField

    If ``regularization="Phi"``:

    compute::

        w * |Φ| * sqrt(||e_theta x e_zeta||)

    If ``regularization="sqrt(Phi)"``:

    compute::

        w * sqrt(|Φ|) * sqrt(||e_theta x e_zeta||)

    Intended to be used with a QuadraticFlux objective, to form
    a problem similar to the REGCOIL algorithm described in [1]_ (if used with a
    ``FourierCurrentPotentialField``, is equivalent to the ``simple``
    regularization of the ``solve_regularized_surface_current`` method).

    References
    ----------
    .. [1] Landreman, Matt. "An improved current potential method for fast computation
      of stellarator coil shapes." Nuclear Fusion (2017).

    Parameters
    ----------
    surface_current_field : CurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    regularization : str, optional
        Regularization method. One of {'K', 'Phi', 'sqrt(Phi)'}. Default = 'K'.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with its ``field_grid`` matching this ``source_grid``, this replicates the
        REGCOIL algorithm described in [1]_ .

    """

    weight_str = (
        "weight : {float, ndarray}, optional"
        "\n\tWeighting to apply to the Objective, relative to other Objectives."
        "\n\tMust be broadcastable to to ``Objective.dim_f``"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        overwrite={"weight": weight_str},
    )
    _static_attrs = _Objective._static_attrs + ["_regularization"]
    _coordinates = "tz"
    _print_value_fmt = "Surface Current Regularization: "

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        regularization="K",
        source_grid=None,
        name="surface-current-regularization",
    ):
        from desc.magnetic_fields import (
            CurrentPotentialField,
            FourierCurrentPotentialField,
        )

        errorif(
            regularization not in ["K", "Phi", "sqrt(Phi)"],
            ValueError,
            "regularization must be one of ['K', 'Phi', 'sqrt(Phi)'], "
            + f"got {regularization}.",
        )
        if target is None and bounds is None:
            target = 0
        assert isinstance(
            surface_current_field, (CurrentPotentialField, FourierCurrentPotentialField)
        ), (
            "surface_current_field must be a CurrentPotentialField or "
            + f"FourierCurrentPotentialField, instead got {type(surface_current_field)}"
        )
        self._regularization = regularization
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid
        self._units = (
            "(A)"
            if self._regularization == "K"
            else "(A*m)" if self._regularization == "Phi" else "(sqrt(A)*m)"
        )

        super().__init__(
            things=[surface_current_field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
            name=name,
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
        from desc.magnetic_fields import FourierCurrentPotentialField

        surface_current_field = self.things[0]
        if isinstance(surface_current_field, FourierCurrentPotentialField):
            M_Phi = surface_current_field._M_Phi
            N_Phi = surface_current_field._N_Phi
        else:
            M_Phi = surface_current_field.M
            N_Phi = surface_current_field.N

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=3 * M_Phi + 1,
                N=3 * N_Phi + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._data_keys = ["Phi", "K", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_transforms = get_transforms(
            self._data_keys,
            obj=surface_current_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
        )
        if self._normalize:
            if isinstance(surface_current_field, FourierCurrentPotentialField):
                self._normalization = np.max(
                    [abs(surface_current_field.I) + abs(surface_current_field.G), 1]
                )
            else:  # it does not have I,G bc is CurrentPotentialField
                Phi = surface_current_field.compute("Phi", grid=source_grid)["Phi"]
                self._normalization = np.max([np.mean(np.abs(Phi)), 1])

        self._constants = {
            "surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params=None, constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            The surface current density magnitude on the source surface.

        """
        constants = self._get_deprecated_constants(constants)

        surface_data = compute_fun(
            self._surface_current_field,
            self._data_keys,
            params=surface_params,
            transforms=constants["surface_transforms"],
            profiles={},
        )

        if self._regularization == "K":
            K = safenorm(surface_data["K"], axis=-1)
        elif self._regularization == "Phi":
            K = jnp.abs(surface_data["Phi"])
        elif self._regularization == "sqrt(Phi)":
            K = jnp.sqrt(jnp.abs(surface_data["Phi"]))
        return K * jnp.sqrt(surface_data["|e_theta x e_zeta|"])
