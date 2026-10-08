"""A smooth per-hinge corner constraint for piecewise-planar arc coils.

An arc coil is C0 at its hinges: the tangent jumps. The turn angle there,

    theta_b = angle between the incoming and outgoing unit tangents at hinge b,

is what a winding machine has to go round, and nothing in the recipe has ever bounded it.
It is only REPORTED (`common.curve_metrics`, as the max over hinges). On helios the
16-coil arcB2 optima reach 117 deg, up from ~100 deg at 12 coils.

This objective returns cos(theta_b) at EVERY hinge of every coil, so

    theta_b <= theta_max   for all b    <=>    cos(theta_b) >= cos(theta_max),

i.e. bound the returned values BELOW by cos(theta_max). Per hinge, not a max over hinges:
max() is kinked exactly where the binding hinge changes, which is the failure mode that
made the paper-form convexity constraint stall the solver (see convexity.py).

cos theta = t_in . t_out is smooth in the parameters wherever the tangents are nonzero,
which they are on any non-degenerate arc.

The one-sided tangents are evaluated a fixed EPS in the curve parameter either side of
each hinge. The offset is a constant, so the objective stays smooth in the coil
parameters -- it is a fixed quadrature choice, not a branch. EPS = 1e-6 of a turn is far
below any shape scale here and matches what `curve_metrics` already uses for reporting.
"""

import numpy as np

from desc.backend import jnp, tree_leaves
from desc.grid import Grid
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective
from desc.utils import dot, rpz2xyz_vec, safenorm

EPS = 1e-6


def _hinge_params(coil):
    """Parameter values of a coil's C0 hinges."""
    if hasattr(coil, "B"):
        B = int(coil.B)
        return np.arange(B) * 2.0 * np.pi / B
    raise ValueError(f"{type(coil).__name__} has no hinges")


def corner_grid(coil):
    """Grid of 2B nodes: EPS before and EPS after each hinge, interleaved (in, out)."""
    h = _hinge_params(coil)
    s = np.concatenate([[(b - EPS) % (2 * np.pi), (b + EPS) % (2 * np.pi)] for b in h])
    nodes = np.stack([0 * s, 0 * s, s], axis=1)
    # spacing must be given for a custom grid; these are POINTWISE residuals, not an
    # integral, so uniform weights are the right choice (the values are used as-is).
    return Grid(nodes, spacing=np.ones_like(nodes), sort=False, jitable=False)


def corner_angles(coil, deg=True):
    """Turn angle at every hinge (numpy), for reporting."""
    g = corner_grid(coil)
    d = coil.compute(["x", "x_s"], grid=g)
    t = np.asarray(rpz2xyz_vec(d["x_s"], phi=np.asarray(d["x"])[:, 1]))
    t = t / np.linalg.norm(t, axis=1)[:, None]
    c = np.clip(np.einsum("ij,ij->i", t[0::2], t[1::2]), -1, 1)
    a = np.arccos(c)
    return np.degrees(a) if deg else a


class CoilCornerAngle(_CoilObjective):
    """cos of the tangent turn angle at every hinge of every arc coil.

    Bound BELOW by cos(theta_max) to impose theta <= theta_max at every hinge.

    Parameters
    ----------
    coil : CoilSet or Coil
        Arc coils (all with the same B).
    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil hinge corner cos(angle): "
    _broadcast_input = "Node"

    def __init__(self, coil, target=None, bounds=None, weight=1, normalize=False,
                 normalize_target=False, loss_function=None, deriv_mode="auto",
                 name="coil corner angle", jac_chunk_size=None):
        if target is None and bounds is None:
            bounds = (-1.0, np.inf)   # no constraint by default
        from desc.coils import CoilSet, MixedCoilSet

        leaves = _leaf_coils(coil)
        grid = corner_grid(leaves[0])
        super().__init__(coil, ["x", "x_s"], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode, grid=grid,
                         name=name, jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

        super().build(use_jit=use_jit, verbose=verbose)
        coils = _leaf_coils(self.things[0])
        kinds = {type(c) for c in coils}
        if not kinds <= {PiecewisePlanarArcCoil, PolarPlanarArcCoil}:
            raise ValueError(f"CoilCornerAngle needs arc coils; got {kinds}")
        nb = {int(c.B) for c in coils}
        if len(nb) != 1:
            raise ValueError(f"CoilCornerAngle needs one B for all coils; got {nb}")
        self._normalization = 1.0
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """cos(turn angle) at every hinge of every coil, 1D."""
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = []
        for d in data:
            t = rpz2xyz_vec(d["x_s"], phi=d["x"][:, 1])
            t = t / safenorm(t, axis=-1, keepdims=True)
            # The framework allocates one residual per GRID NODE (2B per coil), while a
            # turn angle is one number per HINGE (B per coil). Emit each hinge's value at
            # both of its nodes so the shapes line up. The duplicate row is the same
            # constraint twice, which the augmented Lagrangian absorbs into its multiplier.
            out.append(jnp.repeat(dot(t[0::2], t[1::2]), 2))
        return jnp.concatenate(out)


def _leaf_coils(coilset):
    from desc.coils import CoilSet, MixedCoilSet

    out = []
    for c in (coilset if isinstance(coilset, (CoilSet, MixedCoilSet)) else [coilset]):
        out += _leaf_coils(c) if isinstance(c, (CoilSet, MixedCoilSet)) else [c]
    return out
