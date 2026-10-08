"""A smooth convexity constraint for planar coils and for piecewise-planar arc coils.

Kruger et al. (Fusion Eng. Des. 230, 115893, Eq. 5) measure convexity with the total curvature
int |kappa| ds - 2 pi, which is 0 exactly when a planar curve is convex. On a grid that sum is
sum_i |kappa_i| w_i, which has a kink wherever any node's curvature changes sign. An optimized
encircling coil has a nearly straight leg with kappa ~ 0 at many nodes, so the kinks sit right
where the optimum is. On helios_repro (planarN7, 2026-09-16) the trust radius collapsed to
2e-5 there, with nodes 5e-7 to 1e-5 from a sign flip.

This objective instead returns the SIGNED in-plane curvature at every node,

    kappa_s = (x_s x x_ss) . n_hat / |x_s|^3,

where n_hat is the unit normal of the plane the node lies in, oriented by the traversal:

    n_hat ~ S = sum over the segment of (x - p) x x_s ds,   p = the segment's centroid.

A segment is the whole coil for a planar coil, and one arc for a PiecewisePlanarArcCoil.
Every term of S is a cross product of two vectors in the segment's plane, so S lies exactly
along the plane normal whatever the quadrature; for a closed planar curve it is twice the
vector area. Its sign follows the traversal around p, so a convex coil (or a convex arc that
bulges away from its chord) has kappa_s >= 0 everywhere; a circle gives +1/r either way round.
Planar coil: convex iff kappa_s >= 0 at every point. Arc: the arc closed by its chord is convex
iff kappa_s >= 0 along the arc. So bound it below by 0. Smooth wherever the segment is not a
straight line. The C0 corners at arc hinges are not constrained here.

`convexity_excess` = int (|kappa_s| - kappa_s) ds = 2 int max(0, -kappa_s) ds, the concave
turning. For a planar coil it equals the paper's int |kappa| ds - 2 pi (as int kappa_s ds = 2 pi).
"""

import numpy as np

from desc.backend import jnp, tree_leaves
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective
from desc.utils import cross, dot, rpz2xyz, rpz2xyz_vec, safenorm


def _breaks(coil):
    """Parameter values of the coil's C0 corners (arc hinges, or spline break knots)."""
    if hasattr(coil, "B"):
        B = int(coil.B)
        return np.arange(B) * 2.0 * np.pi / B
    iv = np.asarray(getattr(coil, "intervals", [[]]))
    if iv.size == 0:
        return np.zeros(0)
    return np.sort(np.asarray(coil.knots)[np.asarray(iv)[:, 0].astype(int)])


def segment_ids(coil, grid):
    """Segment index of every node, and the number of segments.

    A segment is one planar piece: the whole coil for a FourierPlanarCoil, one arc for
    an arc coil, one C0 interval for a spline coil with break points."""
    s = np.asarray(grid.nodes[:, 2])
    brk = _breaks(coil)
    if brk.size == 0:
        return np.zeros(s.size, dtype=int), 1
    # wrap: nodes before the first break belong to the LAST segment
    return (np.searchsorted(brk, s, side="right") - 1) % brk.size, int(brk.size)


def signed_inplane_curvature(x_rpz, xs_rpz, xss_rpz, ds, seg, nseg):
    """kappa_s at every node from rpz position and derivative data (see module docstring).
    `seg`: segment index of every node (numpy ints), `nseg`: number of segments."""
    phi = x_rpz[:, 1]
    x = rpz2xyz(x_rpz)
    xs = rpz2xyz_vec(xs_rpz, phi=phi)
    xss = rpz2xyz_vec(xss_rpz, phi=phi)
    W = jnp.asarray(np.eye(nseg)[seg]) * ds[:, None]  # (nodes, segments): quadrature weight in its segment
    p = (W.T @ x) / jnp.sum(W, axis=0)[:, None]  # segment centroids
    S = jnp.sum(W[:, :, None] * cross(x[:, None, :] - p[None, :, :], xs[:, None, :]), axis=0)
    n_hat = S / safenorm(S, axis=-1, keepdims=True)
    return dot(cross(xs, xss), n_hat[seg]) / safenorm(xs, axis=-1) ** 3


def convexity_excess(coil, n=1000):
    """int (|kappa_s| - kappa_s) ds on a uniform grid (numpy float): 0 iff every segment is convex."""
    from desc.grid import LinearGrid

    g = LinearGrid(N=n)
    seg, B = segment_ids(coil, g)
    d = coil.compute(["x", "x_s", "x_ss", "ds"], grid=g)
    k = np.asarray(signed_inplane_curvature(d["x"], d["x_s"], d["x_ss"], d["ds"], seg, B))
    sp = np.linalg.norm(np.asarray(d["x_s"]), axis=1)
    return float(np.sum((np.abs(k) - k) * sp * np.asarray(d["ds"]))), float(k.min())


class CoilSignedCurvature(_CoilObjective):
    """Signed in-plane curvature at every grid node (bound it below by 0), for planar coils
    (FourierPlanarCoil) or piecewise-planar arc coils (per arc).

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) to be optimized; all of one type (the same B for arcs).
    grid : LinearGrid
        Collocation grid, the same for every coil. Its node weights are used as ds.
    """

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Signed in-plane curvature: "
    _broadcast_input = "Node"
    _static_attrs = _CoilObjective._static_attrs + ["_seg", "_nseg"]

    def __init__(self, coil, target=None, bounds=None, weight=1, normalize=True, normalize_target=True,
                 loss_function=None, deriv_mode="auto", grid=None, name="coil signed curvature",
                 jac_chunk_size=None):
        if target is None and bounds is None:
            bounds = (0, np.inf)
        super().__init__(coil, ["x", "x_s", "x_ss", "ds"], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target, loss_function=loss_function,
                         deriv_mode=deriv_mode, grid=grid, name=name, jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        from desc.coils import (
            FourierPlanarCoil, PiecewisePlanarArcCoil, PolarPlanarArcCoil, SplineXYZCoil)

        super().build(use_jit=use_jit, verbose=verbose)
        coils = _leaf_coils(self.things[0])
        kinds = {type(c) for c in coils}
        nseg = {int(_breaks(c).size) for c in coils}
        if not kinds <= {FourierPlanarCoil, PiecewisePlanarArcCoil, PolarPlanarArcCoil,
                         SplineXYZCoil} or len(nseg) != 1:
            raise ValueError(f"CoilSignedCurvature needs planar, planar-arc or spline coils with the same "
                             f"number of C0 segments; got {kinds} with segment counts {nseg}")
        grids = tree_leaves(self._grid, is_leaf=lambda g: hasattr(g, "nodes"))
        if any(not np.array_equal(g.nodes, grids[0].nodes) for g in grids):
            raise ValueError("CoilSignedCurvature needs one grid shared by all coils")
        seg, self._nseg = segment_ids(coils[0], grids[0])
        self._seg = tuple(int(i) for i in seg)  # static (hashable)
        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Signed in-plane curvature at every node of every coil, 1D."""
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        out = jnp.concatenate([signed_inplane_curvature(d["x"], d["x_s"], d["x_ss"], d["ds"], np.asarray(self._seg),
                                                        self._nseg) for d in data])
        return out[self._coilset_tree["coilset_mask"]]


def _leaf_coils(coilset):
    from desc.coils import CoilSet, MixedCoilSet

    out = []
    for c in (coilset if isinstance(coilset, (CoilSet, MixedCoilSet)) else [coilset]):
        out += _leaf_coils(c) if isinstance(c, (CoilSet, MixedCoilSet)) else [c]
    return out
