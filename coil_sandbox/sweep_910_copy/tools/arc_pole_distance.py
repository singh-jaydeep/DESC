"""Keep PolarPlanarArc arcs away from their polar pole (a coordinate-validity bound).

PolarPlanarArcCurve writes each arc as a polar graph about its chord midpoint::

    r(phi) = |chord|/2 + sum_m a[i, m] sin((m+1) pi phi),    phi in [0, 1]

The class docstring requires ``r > 0`` -- the arc must not reach the pole -- and warns
that conditioning degrades as arcs get shallow, with a circular-arc floor of
``min r / (|chord|/2) = tan(beta/2)`` where ``beta = 90deg / (B/2)``. At B=5 that floor
is ``tan(18deg) = 0.325``.

This is a validity condition of the COORDINATE SYSTEM, not physics. Under weighted-sum
optimization it was enforced implicitly -- penalizing curvature keeps arcs round, and
round arcs keep ``r`` clear of the pole -- so it never needed stating. An augmented
Lagrangian rides the curvature bound instead of staying inside it, and shallow arcs are
both high-curvature and small-``r``, so it walks somewhere the penalty method never
went. Measured: a solve that stalled with the trust region collapsed to 3e-15 while
optimality was still 1.5 had driven ``min r/(|chord|/2)`` to 0.3055, below the floor;
the run that succeeded from a different start sat at 0.3727 and climbed to 0.4684.

Moving from penalties to constraints removes implicit regularization, so anything the
penalties were quietly protecting has to be written down. This writes that one down.

Usage::

    obj9 = ArcPoleDistance(coilset, bounds=(0.4, np.inf), normalize=False)

Cheap: closed form in ``hinges`` and ``shape``, no Biot-Savart, one value per arc.
"""

import numpy as np

from desc.backend import jnp
from desc.objectives.objective_funs import _Objective
from desc.utils import errorif


class ArcPoleDistance(_Objective):
    """Minimum polar radius of each arc, normalized by its chord half-length.

    Returns ``min_phi r(phi) / (|chord|/2)`` for every arc of every unique coil. The
    value is 1.0 for a semicircular arc (B=2), and approaches 0 as an arc flattens
    toward its chord, where the polar parameterization becomes singular.

    Parameters
    ----------
    coil : CoilSet
        Coilset of PolarPlanarArcCoil (or PiecewisePlanarArcCoil) to constrain.
    n_phi : int
        Number of points per arc at which to sample ``r(phi)``. The minimum of a
        sine series is not available in closed form, so it is sampled; ``r`` is
        smooth and low order, so this converges quickly. Default 65.
    target, bounds, weight, normalize, normalize_target, name
        See ``_Objective``. Give ``bounds=(r_floor, np.inf)``; ``normalize=False``
        is appropriate since the quantity is already dimensionless.

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Arc pole distance: "
    _static_attrs = _Objective._static_attrs + ["_B", "_M", "_n_coils"]

    def __init__(
        self,
        coil,
        n_phi=65,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="arc pole distance",
    ):
        from desc.coils import CoilSet

        errorif(
            not isinstance(coil, CoilSet),
            ValueError,
            "coil must be a CoilSet of arc coils",
        )
        if target is None and bounds is None:
            bounds = (0.4, np.inf)
        self._n_phi = int(n_phi)
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
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        coilset = self.things[0]
        coils = coilset.coils
        errorif(
            not all(hasattr(c, "_B") and hasattr(c, "_M") for c in coils),
            ValueError,
            "every coil must be a polar/piecewise planar arc coil (needs _B, _M)",
        )
        errorif(
            len({(c._B, c._M) for c in coils}) != 1,
            ValueError,
            "all coils must share the same (B, M)",
        )
        self._B = int(coils[0]._B)
        self._M = int(coils[0]._M)
        self._n_coils = len(coils)
        self._dim_f = self._n_coils * self._B

        phi = jnp.linspace(0.0, 1.0, self._n_phi)
        # basis[m, k] = sin((m+1) pi phi_k), shared by every arc
        basis = jnp.sin(jnp.arange(1, self._M + 1)[:, None] * jnp.pi * phi[None, :])
        self._constants = {"basis": basis, "quad_weights": 1.0}
        self._normalization = 1.0
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute min polar radius / (chord half-length) for each arc.

        Parameters
        ----------
        params : list of dict
            Coilset degrees of freedom, one dict per unique coil.
        constants : dict
            Deprecated; the sine basis is taken from ``self._constants``.

        Returns
        -------
        f : ndarray, shape(n_coils * B,)
            ``min_phi r(phi) / (|chord|/2)`` per arc.

        """
        constants = self._get_deprecated_constants(constants)
        basis = constants["basis"]
        B, M = self._B, self._M
        params = params if isinstance(params, (list, tuple)) else [params]

        out = []
        for p in params:
            H = jnp.asarray(p["hinges"]).reshape(B, 3)
            A = jnp.asarray(p["shape"]).reshape(B, M)
            chords = jnp.roll(H, -1, axis=0) - H  # H[i+1] - H[i], closed
            half = jnp.linalg.norm(chords, axis=1) / 2.0  # |chord|/2 per arc
            # r(phi) = half + A @ basis   ->  (B, n_phi)
            r = half[:, None] + A @ basis
            out.append(jnp.min(r, axis=1) / half)
        return jnp.concatenate(out)
