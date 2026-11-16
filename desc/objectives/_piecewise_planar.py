"""Constraints for piecewise-planar arc coils, coil currents and pointwise B.n.

The arc objectives read ``hinges[:, 2]`` as the lab height, which is valid only for leaf
coils that carry no rigid transform (``rotmat`` is the identity and ``shift`` zero); a
symmetric ``CoilSet`` applies its symmetry itself, so its leaves satisfy this. It is
checked at build.
"""

import numpy as np

from desc.backend import jnp
from desc.utils import errorif

from .linear_objectives import FixParameters
from .objective_funs import _Objective, collect_docs


def _leaf_coils(coilset):
    """Every leaf coil of a (nested) coilset, depth first: the independent coils."""
    from desc.coils import CoilSet, MixedCoilSet

    out = []
    for c in coilset if isinstance(coilset, (CoilSet, MixedCoilSet)) else [coilset]:
        out += _leaf_coils(c) if isinstance(c, (CoilSet, MixedCoilSet)) else [c]
    return out


def _leaf_params(tree):
    """The per-coil parameter dicts, in the order of ``_leaf_coils``."""
    if isinstance(tree, dict):
        return [tree]
    out = []
    for t in tree:
        out += _leaf_params(t)
    return out


def _is_polar(coil):
    from desc.coils import PolarPlanarArcCoil

    return isinstance(coil, PolarPlanarArcCoil)


def _arc_coils(coilset, name, same_B=True):
    """Leaf coils of coilset, checked: arc coils without rigid transforms."""
    from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

    coils = _leaf_coils(coilset)
    kinds = {type(c) for c in coils}
    errorif(
        not kinds <= {PiecewisePlanarArcCoil, PolarPlanarArcCoil},
        ValueError,
        f"{name} needs arc coils, got {kinds}.",
    )
    errorif(
        same_B and len({int(c.B) for c in coils}) != 1,
        ValueError,
        f"{name} needs the same number of arcs B on every coil.",
    )
    for c in coils:
        errorif(
            not (
                np.allclose(np.asarray(c.rotmat).reshape(3, 3), np.eye(3))
                and np.allclose(np.asarray(c.shift), 0.0)
            ),
            NotImplementedError,
            f"{name} reads hinges[:, 2] as the lab height, but a leaf coil carries a "
            "rigid transform (rotmat/shift).",
        )
    return coils


# ------------------------------------------------------------------------- folds


def side_signs(params, B, polar):
    """Side of its chord each arc bulges to, +1 or -1 per arc (+1 for polar arcs).

    For a transverse arc ``x = H + t chord + w(t) perp`` it is the sign of the mean
    displacement ``int w dt = sum_m a_m (1 - cos(m pi)) / (m pi)``.
    """
    if polar:
        return np.ones(B)
    a = np.asarray(params["shape"]).reshape(B, -1)
    m = np.arange(1, a.shape[1] + 1)
    mean_w = a @ ((1 - np.cos(m * np.pi)) / (m * np.pi))
    return np.where(mean_w >= 0, 1.0, -1.0)


def fold_cos(params, B, polar, sides):
    """cos(fold) between consecutive arc planes of one coil: (1,) for B = 2, else (B,).

    The fold is the angle between the oriented plane normals
    ``n_i = s_i (e_i x perp_i)`` of consecutive arcs, with ``e_i`` the chord direction,
    ``perp_i`` the in-plane normal and ``s_i`` from ``side_signs``.
    """
    from desc.compute._curve import _ppa_arc_frame, _ppolar_arc_frame

    frame = _ppolar_arc_frame if polar else _ppa_arc_frame
    out = frame(
        jnp.asarray(params["hinges"]).reshape(B, 3),
        jnp.asarray(params["tilts"]).reshape(B),
        B,
        jnp.asarray(params["arc_ref"]),
    )
    n = jnp.cross(out[1], out[2]) * jnp.asarray(sides)[:, None]
    c = jnp.sum(n * jnp.roll(n, -1, axis=0), axis=1)
    return c[:1] if B == 2 else c


def fold_angles(coilset, sides=None, deg=True):
    """Largest fold of every independent arc coil, for reporting.

    Parameters
    ----------
    coilset : CoilSet
        Arc coils.
    sides : array-like, shape(num_coils, B), optional
        Arc orientations, e.g. ``ArcFold.sides``. Default from each coil's own state.
    deg : bool
        Degrees (default) or radians.

    Returns
    -------
    fold : ndarray, shape(num_coils,)

    """
    coils = _leaf_coils(coilset)
    out = []
    for k, c in enumerate(coils):
        B, polar = int(c.B), _is_polar(c)
        s = side_signs(c.params_dict, B, polar) if sides is None else sides[k]
        cos = np.asarray(fold_cos(c.params_dict, B, polar, s))
        out.append(np.arccos(np.clip(cos, -1, 1)).max())
    out = np.array(out)
    return np.degrees(out) if deg else out


class ArcFold(_Objective):
    """cos of the fold angle between the planes of consecutive arcs of each coil.

    The fold is 0 for a planar coil, 90 degrees for perpendicular arcs and 180 degrees
    for one arc folded back onto the next. Bound it below, ``bounds=(cos(fold_max),
    np.inf)``, for fold <= fold_max. One row per coil for B = 2 (both arcs share one
    chord), B rows otherwise. Works for both arc classes.

    The plane normals are oriented by the side of its chord each arc bulges to
    (``sides``), frozen at build so the rows stay smooth. A transverse arc that later
    crosses its own chord would read 180 - fold; ``fold_angles`` with the default
    sides recomputes the orientation from a state.

    Parameters
    ----------
    coil : CoilSet
        Arc coils of one class, all with the same B.
    sides : array-like, shape(num_coils, B), optional
        Orientation (+1 or -1) of every arc. Default from the coils at build. Pass the
        orientation of an earlier stage when continuing from it: an arc whose mean
        displacement has changed sign would otherwise flip its fold to 180 - fold.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-1, np.inf)``, i.e. no constraint.",
        bounds_default="``bounds=(-1, np.inf)``, i.e. no constraint.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Arc fold cos(angle): "
    _static_attrs = _Objective._static_attrs + ["_B", "_polar"]

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="arc fold",
        jac_chunk_size=None,
        sides=None,
    ):
        if target is None and bounds is None:
            bounds = (-1.0, np.inf)
        self._sides = None if sides is None else np.asarray(sides, dtype=float)
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        coils = _arc_coils(self.things[0], type(self).__name__)
        errorif(
            len({_is_polar(c) for c in coils}) != 1,
            ValueError,
            "ArcFold needs arc coils of one class.",
        )
        self._B, self._polar = int(coils[0].B), _is_polar(coils[0])
        sides = np.array(
            [side_signs(c.params_dict, self._B, self._polar) for c in coils]
        )
        if self._sides is not None:
            errorif(
                self._sides.shape != sides.shape,
                ValueError,
                f"sides must have shape {sides.shape}, got {self._sides.shape}.",
            )
            sides = self._sides
        self.sides = sides
        self._dim_f = len(coils) * (1 if self._B == 2 else self._B)
        self._constants = {"quad_weights": 1.0, "sides": jnp.asarray(sides)}
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute cos(fold) of every coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            cos(fold), coil by coil.

        """
        constants = self._get_deprecated_constants(constants)
        return jnp.concatenate(
            [
                fold_cos(p, self._B, self._polar, constants["sides"][k])
                for k, p in enumerate(_leaf_params(params))
            ]
        )


# ------------------------------------------------------------------------ hinges


class FixHingeHeight(FixParameters):
    """Fixes the height (z) of every hinge of every arc coil: a joint plane.

    With ``height=0`` every hinge lies on z = 0, the only single plane under
    stellarator symmetry (which maps z to -z). Linear: eliminated exactly by the
    optimizer, while the hinges' x and y stay free.

    Parameters
    ----------
    coil : CoilSet
        Arc coils, without rigid transforms on the leaf coils.
    height : float, optional
        Height (m) to hold every hinge at. Default the current heights.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=height``, or the current heights.",
        bounds_default="``target=height``, or the current heights.",
        coil=True,
    )

    _units = "(m)"
    _print_value_fmt = "Fixed hinge height error: "

    def __init__(
        self,
        coil,
        height=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name="fixed hinge height",
    ):
        coils = _arc_coils(coil, "FixHingeHeight", same_B=False)
        if target is None and bounds is None and height is not None:
            target = np.full(sum(int(c.B) for c in coils), float(height))

        def z_mask(t):
            if isinstance(t, dict):
                B = len(np.asarray(t["hinges"])) // 3
                return {
                    k: (np.arange(B) * 3 + 2 if k == "hinges" else False) for k in t
                }
            return [z_mask(q) for q in t]

        super().__init__(
            thing=coil,
            params=z_mask(coil.params_dict),
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )


class HingeLevel(_Objective):
    """Every hinge of a coil at the same height, the height left free.

    B - 1 residuals ``z_b - z_0`` per coil; use ``target=0`` (the default). Each coil's
    hinge chord is then horizontal, at its own height, so the machine has one joint
    plane per coil. Linear in the hinges: with a target the optimizer eliminates it
    exactly, without a multiplier.

    Parameters
    ----------
    coil : CoilSet
        Arc coils, all with the same B, without rigid transforms on the leaf coils.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _linear = True
    _fixed = False
    _units = "(m)"
    _print_value_fmt = "Hinge level: "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="hinge level",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0.0
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        coils = _arc_coils(self.things[0], type(self).__name__)
        self._dim_f = len(coils) * (int(coils[0].B) - 1)
        self._constants = {"quad_weights": 1.0}
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute z_b - z_0 at every hinge past the first, coil by coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Hinge height differences (m).

        """
        out = []
        for p in _leaf_params(params):
            z = jnp.asarray(p["hinges"]).reshape(-1, 3)[:, 2]
            out.append(z[1:] - z[0])
        return jnp.concatenate(out)


def axis_coeffs(eq):
    """Fourier series of the magnetic axis height of eq as arrays ``(k, z_k, is_cos)``.

    ``Z_axis(phi) = sum_k z_k (cos(k phi) if is_cos else sin(k phi))``, with phi the
    cylindrical angle.
    """
    ax = eq.axis
    errorif(
        not (
            np.allclose(np.asarray(ax.rotmat).reshape(3, 3), np.eye(3))
            and np.allclose(np.asarray(ax.shift), 0.0)
        ),
        NotImplementedError,
        "The axis curve carries a rigid transform.",
    )
    n = np.asarray(ax.Z_basis.modes[:, 2])
    return (
        np.abs(n) * float(ax.NFP),
        np.asarray(ax.Z_n, dtype=float),
        (n >= 0).astype(float),
    )


def _axis_height(phi, k, zk, is_cos):
    a = k * jnp.asarray(phi)[..., None]
    return jnp.sum(zk * (is_cos * jnp.cos(a) + (1 - is_cos) * jnp.sin(a)), axis=-1)


class HingeMidHeightRelative(_Objective):
    """Height of each coil's hinge midpoint above the magnetic axis, over z_max.

    Per coil ``(mean hinge z - Z_axis(phi_c)) / z_max``, with ``phi_c`` the toroidal
    angle of the mean hinge position and ``Z_axis`` the height of the axis of the
    (fixed) equilibrium. Bound with ``(-1, 1)``: the hinge chord stays within
    ``+-z_max`` of the axis. On a planar axis this is the plain hinge height.

    Parameters
    ----------
    coil : CoilSet
        Arc coils, without rigid transforms on the leaf coils.
    eq : Equilibrium
        Equilibrium whose magnetic axis gives the reference height (held fixed).
    z_max : float
        Allowed excursion (m).

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-1, 1)``.",
        bounds_default="``bounds=(-1, 1)``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Hinge height above axis / z_max: "

    def __init__(
        self,
        coil,
        eq,
        z_max,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="hinge height above axis",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (-1.0, 1.0)
        self._eq = eq
        self._z_max = float(z_max)
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        self._dim_f = len(_arc_coils(self.things[0], "HingeMidHeightRelative", False))
        k, zk, is_cos = axis_coeffs(self._eq)
        self._constants = {
            "quad_weights": 1.0,
            "axis": (jnp.asarray(k), jnp.asarray(zk), jnp.asarray(is_cos)),
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the hinge midpoint height above the axis over z_max, per coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            One value per coil (dimensionless).

        """
        constants = self._get_deprecated_constants(constants)
        out = []
        for p in _leaf_params(params):
            m = jnp.mean(jnp.asarray(p["hinges"]).reshape(-1, 3), axis=0)
            z0 = _axis_height(jnp.arctan2(m[1], m[0]), *constants["axis"])
            out.append((m[2] - z0)[None])
        return jnp.concatenate(out) / self._z_max


def tilt_sin(params):
    """sin(elevation) of the chords between consecutive hinges of one coil.

    (1,) for B = 2 (both chords are one line), else (B,).
    """
    H = jnp.asarray(params["hinges"]).reshape(-1, 3)
    d = jnp.roll(H, -1, axis=0) - H
    s = d[:, 2] / jnp.linalg.norm(d, axis=1)
    return s[:1] if H.shape[0] == 2 else s


class HingeTilt(_Objective):
    """sin of the elevation of the chord between consecutive hinges of each coil.

    0 for a horizontal hinge line, +-1 for a vertical one. Bound with
    ``(-sin(tilt_max), sin(tilt_max))``. One row per coil for B = 2, B rows otherwise.

    Parameters
    ----------
    coil : CoilSet
        Arc coils, without rigid transforms on the leaf coils.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-1, 1)``, i.e. no constraint.",
        bounds_default="``bounds=(-1, 1)``, i.e. no constraint.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Hinge chord sin(elevation): "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="hinge tilt",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (-1.0, 1.0)
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        coils = _arc_coils(self.things[0], "HingeTilt", same_B=False)
        self._dim_f = sum(1 if int(c.B) == 2 else int(c.B) for c in coils)
        self._constants = {"quad_weights": 1.0}
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute sin(elevation) of every hinge chord, coil by coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            sin(elevation) of each chord.

        """
        return jnp.concatenate([tilt_sin(p) for p in _leaf_params(params)])


# ------------------------------------------------------------------- polar arcs


def relative_radius(params, B, M, phi):
    """Polar radius over half the chord at nodes phi of every arc, shape (B, len(phi)).

    ``rho(phi) = 1 + (2 / |chord|) sum_m a_m sin(m pi phi)``.
    """
    H = jnp.asarray(params["hinges"]).reshape(B, 3)
    half = jnp.linalg.norm(jnp.roll(H, -1, axis=0) - H, axis=1) / 2
    a = jnp.asarray(params["shape"]).reshape(B, M)
    s = jnp.sin(jnp.pi * jnp.outer(jnp.asarray(phi), jnp.arange(1, M + 1)))
    return 1 + (a @ s.T) / half[:, None]


class PolarArcRadius(_Objective):
    """Relative polar radius r / (|chord|/2) at K interior nodes of every polar arc.

    A ``PolarPlanarArcCoil`` arc is the polar graph ``r(theta)`` about its chord
    midpoint, and is a valid coil only while ``r > 0``: where r crosses zero the arc
    passes through its pole to the far side of the chord. Bound below by
    ``rho_min > 0``. One row per node (not a minimum over nodes, which would be
    kinked). ``rho = 1`` is the circular arc; the hinges (rho = 1 identically) are left
    out.

    Parameters
    ----------
    coil : CoilSet
        Polar arc coils, all with the same B and M.
    K : int
        Nodes per arc, uniform in the polar angle. Default 64.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0, np.inf)``.",
        bounds_default="``bounds=(0, np.inf)``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Polar arc relative radius: "
    _static_attrs = _Objective._static_attrs + ["_B", "_M"]

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="polar arc radius",
        jac_chunk_size=None,
        K=64,
    ):
        if target is None and bounds is None:
            bounds = (0.0, np.inf)
        self._K = int(K)
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        from desc.coils import PolarPlanarArcCoil

        coils = _leaf_coils(self.things[0])
        errorif(
            not all(isinstance(c, PolarPlanarArcCoil) for c in coils),
            ValueError,
            "PolarArcRadius needs polar arc coils.",
        )
        BM = {(int(c.B), int(c.M)) for c in coils}
        errorif(len(BM) != 1, ValueError, "PolarArcRadius needs one (B, M).")
        self._B, self._M = BM.pop()
        self._dim_f = len(coils) * self._B * self._K
        self._constants = {
            "quad_weights": 1.0,
            "phi": jnp.asarray((np.arange(self._K) + 0.5) / self._K),
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute rho at every node of every arc, ordered (coil, arc, node).

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Relative radius (dimensionless).

        """
        constants = self._get_deprecated_constants(constants)
        return jnp.concatenate(
            [
                relative_radius(p, self._B, self._M, constants["phi"]).reshape(-1)
                for p in _leaf_params(params)
            ]
        )


# ---------------------------------------------------------------------- currents


class CoilCurrentBand(_Objective):
    """Each independent coil's current over the average current at build.

    Bound with ``(1 - f, 1 + f)``. Stage 2 usually fixes only the sum of the coil
    currents, so the optimizer may otherwise switch a coil off or reverse it. With the
    sum fixed the average is a constant, so the rows are linear in the currents; with
    bounds they are inequality constraints.

    Parameters
    ----------
    coil : CoilSet
        Coils to optimize.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(0.5, 1.5)``.",
        bounds_default="``bounds=(0.5, 1.5)``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil current / average: "

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="coil current band",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0.5, 1.5)
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

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        cur = np.array([float(c.current) for c in _leaf_coils(self.things[0])])
        errorif(cur.mean() == 0, ValueError, "The average coil current is zero.")
        self._dim_f = cur.size
        self._constants = {"quad_weights": 1.0, "I_avg": float(cur.mean())}
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute I_i / I_avg for every independent coil.

        Parameters
        ----------
        params : dict
            Dictionary of coilset degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Current fractions (dimensionless).

        """
        constants = self._get_deprecated_constants(constants)
        cur = [jnp.atleast_1d(p["current"]) for p in _leaf_params(params)]
        return jnp.concatenate(cur) / constants["I_avg"]


# -------------------------------------------------------------------- B.n on LCFS


class PointwiseBn(_Objective):
    """B.n / |B| of an external field at every node of a fixed plasma boundary.

    ``QuadraticFlux`` is an area integral and lets localized peaks stand; bounding this
    objective, ``bounds=(-c, c)``, caps the field-line angle off the surface at every
    node. Vacuum only: the plasma's own field is not included. The rows are
    dimensionless and unweighted.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium whose boundary is used (held fixed).
    field : MagneticField
        External field, e.g. a CoilSet, to optimize.
    eval_grid : Grid
        Nodes on the boundary (rho = 1).
    field_grid : Grid, optional
        Grid to discretize the field source, see ``field.compute_magnetic_field``.
    bs_chunk_size : int, optional
        Chunk size of the Biot-Savart evaluation points.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Pointwise B.n/|B|: "
    _static_attrs = _Objective._static_attrs + ["_field_grid", "_bs_chunk_size"]

    def __init__(
        self,
        eq,
        field,
        eval_grid,
        field_grid=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        bs_chunk_size=None,
        jac_chunk_size=None,
        name="pointwise B.n/|B|",
    ):
        if target is None and bounds is None:
            target = 0
        self._eq, self._eval_grid = eq, eval_grid
        self._field_grid, self._bs_chunk_size = field_grid, bs_chunk_size
        super().__init__(
            things=[field],
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
        errorif(
            not np.allclose(self._eval_grid.nodes[:, 0], 1),
            ValueError,
            "eval_grid must lie on the boundary (rho = 1).",
        )
        surf = getattr(self._eq, "surface", self._eq)
        d = surf.compute(["R", "phi", "Z", "n_rho"], grid=self._eval_grid, basis="rpz")
        x = np.stack([d["R"], d["phi"], d["Z"]], axis=1)
        self._constants = {
            "x": jnp.asarray(x),
            "n": jnp.asarray(d["n_rho"]),
            "quad_weights": 1.0,
        }
        self._dim_f = x.shape[0]
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute B.n / |B| at every node of the boundary.

        Parameters
        ----------
        params : dict
            Dictionary of field degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self._constants. (Deprecated)

        Returns
        -------
        f : ndarray
            B.n / |B| (dimensionless).

        """
        constants = self._get_deprecated_constants(constants)
        B = self.things[0].compute_magnetic_field(
            constants["x"],
            source_grid=self._field_grid,
            basis="rpz",
            params=params,
            chunk_size=self._bs_chunk_size,
        )
        return jnp.sum(B * constants["n"], axis=-1) / jnp.linalg.norm(B, axis=-1)
