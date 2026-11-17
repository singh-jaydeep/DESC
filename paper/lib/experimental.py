"""EXPERIMENTAL objectives for diagnosing solver behaviour; not the paper method.

``PairClosestRows`` reproduces the per-pair rows the js/coil-auglag branch appended to
``CoilSetDistanceRows`` (with ``signed=False``): for each coil pair, the distance of its
closest node pair (no gap), bounded below by 0. They can never be active, but their
Jacobians enter the solver's ``x_scale="jac"`` column norms (acceptance run 8a,
2026-11-17), which changes the optimization path. With ``signed=True`` they are the
branch's link guard: a linked pair (|linking number| > 0.5) reads minus its distance,
violating the lower bound 0; with ``lsq_auglag_composite``'s ``initial_penalty_rows``
a large penalty on them makes the merit function reject any step that links a pair.
"""

import numpy as np

from desc.backend import jax, jnp
from desc.batching import vmap_chunked
from desc.objectives.objective_funs import _Objective
from desc.utils import safenorm


class PairClosestRows(_Objective):
    """Per coil pair, the distance of its closest node pair; bounds (0, inf).

    Parameters
    ----------
    cc : CoilSetDistanceRows
        A built coil-coil row objective: its coilset, node grid, pairs and
        normalization are reused, so the rows match the branch's per-pair rows.

    """

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Coil pair closest nodes: "
    _static_attrs = _Objective._static_attrs + ["_pairs", "_signed"]

    def __init__(self, cc, signed=False, name="coil pair closest nodes"):
        self._cc = cc
        self._signed = bool(signed)
        super().__init__(
            things=cc.things[0],
            target=None,
            bounds=(0.0, np.inf),
            normalize=cc._normalize,
            normalize_target=cc._normalize_target,
            name=name,
            jac_chunk_size=cc._jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        cc = self._cc
        self._pairs = cc._pairs
        self._dim_f = len(self._pairs)
        if self._normalize:
            self._normalization = cc._normalization
        from desc.grid import LinearGrid

        link_grid = LinearGrid(N=100)
        self._constants = {
            "coilset": cc._constants["coilset"],
            "grid": cc._constants["grid"],
            "link_grid": link_grid,
            "link_dx": np.asarray(link_grid.spacing[:, 2]),
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Closest node-pair distance of every pair."""
        constants = self._get_deprecated_constants(constants)
        pts = constants["coilset"]._compute_position(
            params=params, grid=constants["grid"], basis="xyz"
        )
        pts0 = jax.lax.stop_gradient(pts)
        a, b = (jnp.array(v) for v in zip(*self._pairs))
        N = pts.shape[1]

        def closest(p):  # no derivative: which node pair is closest
            d = jnp.linalg.norm(pts0[a[p], :, None] - pts0[b[p], None], axis=-1)
            return jnp.argmin(d)

        ij = vmap_chunked(closest, chunk_size=1)(jnp.arange(a.size))
        d = safenorm(pts[a, ij // N] - pts[b, ij % N], axis=-1)
        if self._signed:
            d = jnp.where(self.linked(params, constants), -d, d)
        return d

    def linked(self, params, constants=None):
        """Whether each pair is linked (|linking number| > 0.5), no derivative."""
        from desc.coils import _linking_number

        constants = self._get_deprecated_constants(constants)
        x, xs = constants["coilset"]._compute_position(
            params=params, grid=constants["link_grid"], dx1=True, basis="xyz"
        )
        x, xs = jax.lax.stop_gradient(x), jax.lax.stop_gradient(xs)
        dx = jnp.asarray(constants["link_dx"])
        a, b = (jnp.array(v) for v in zip(*self._pairs))
        lk = vmap_chunked(
            lambda p: _linking_number(x[a[p]], x[b[p]], xs[a[p]], xs[b[p]], dx, dx),
            chunk_size=8,
        )(jnp.arange(a.size))
        return jnp.abs(lk / (4 * jnp.pi)) > 0.5
