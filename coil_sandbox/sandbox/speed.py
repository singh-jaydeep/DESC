"""A lower bound on the parameterization speed |x_s|, for spline coils.

Why this and not a finer curvature grid. A spline's parameterization is not unique, and
the optimizer exploits that: kappa = |x_s x x_ss| / |x_s|^3, so slowing |x_s| at a point
BETWEEN constraint nodes manufactures curvature that the constraint cannot see and the
field objective is happy to buy. MEASURED on a cold spline run (32 knots, 2 C0 breaks):
kappa read 0.90 on the 301-node enforcement grid and 3.45 on a fine grid, with the peak
sitting exactly at |x_s| = 0.116 of its mean.

Two fixes were tried first and are worse:
  * A FINER CURVATURE GRID. 1601 nodes (~50/span) does not fit an 8 GB card: the
    second-order constraint Hessian OOMs under the pooled allocator, runs ~20x slower
    under `platform`, and hangs under `cuda_async`.
  * `CoilArclengthVariance`. It penalises Var(|x_s|) over the WHOLE coil, so a single
    narrow dip barely registers. Measured at weight 1.0: |x_s| min/mean improved
    0.396 -> 0.549 and 0.364 -> 0.481 on two coils but only 0.151 -> 0.171 on the third,
    which still hid 1.61x of its curvature.

Bounding |x_s| directly is better conditioned than bounding kappa, because collocation
error enters LINEARLY rather than cubed: the same grid that under-read kappa by 3.8x
under-read |x_s| by only 1.56x.

The residual is |x_s| divided by that coil's own mean speed, so it is dimensionless and
independent of the coil's length and of the curve parameter's scale. Bound it below by
`alpha`. Note this restricts the PARAMETERIZATION, not the geometry -- any curve can be
reparameterized to satisfy it -- but with a fixed knot vector the spline cannot always
do so, and a tight alpha therefore does remove shapes. Choose it loosely.
"""

import numpy as np

from desc.backend import jnp, tree_leaves
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective
from desc.utils import safenorm


class CoilSpeedFloor(_CoilObjective):
    """|x_s| at every node, divided by that coil's mean |x_s|. Bound below by alpha.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) to be optimized.
    alpha : float
        Lower bound on |x_s| / mean(|x_s|). 0 disables (the bound is vacuous).
    grid : Grid
        Collocation grid, shared by all coils.
    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil speed / mean speed: "
    _broadcast_input = "Node"

    def __init__(self, coil, alpha=0.3, target=None, bounds=None, weight=1, normalize=False,
                 normalize_target=False, loss_function=None, deriv_mode="auto", grid=None,
                 name="coil speed floor", jac_chunk_size=None):
        if target is None and bounds is None:
            bounds = (float(alpha), np.inf)
        super().__init__(coil, ["x_s"], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode, grid=grid,
                         name=name, jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        super().build(use_jit=use_jit, verbose=verbose)
        self._normalization = 1.0  # already dimensionless
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """|x_s| / mean(|x_s|) at every node of every coil, 1D."""
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = []
        for d in data:
            sp = safenorm(d["x_s"], axis=-1)
            out.append(sp / jnp.mean(sp))
        return jnp.concatenate(out)[self._coilset_tree["coilset_mask"]]
