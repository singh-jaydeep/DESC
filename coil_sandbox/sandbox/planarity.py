"""Exact per-segment planarity for spline coils with C0 breaks.

A spline interpolant is LINEAR in its knot values and reproduces constants, and with
``break_indices`` each interval's interpolant sees only that interval's knots (plus the
two break values, which lie in the interval's plane as well). So for any query s in
segment i,

    x(s) = sum_j c_j(s) P_j   with   sum_j c_j(s) = 1,   j over segment i's knots,

and if every P_j satisfies (P_j - a).n = 0 then so does x(s). Hence

    a spline segment is planar  <=>  that segment's knots are coplanar.

MEASURED on `work/helios/polarB2_M5_cut30_final.h5` converted at 64 knots: coplanar
knots give a curve planar to 1e-14 m, and pushing one interior knot 1/10/100 mm off its
plane moves the curve 0.93/9.30/93.0 mm out of plane while the neighbouring segment
stays at 1e-14 m. So planarity is an exact algebraic condition on the DEGREES OF
FREEDOM: no grid, no quadrature, no third derivatives, and the segments are decoupled.

That is worth having over a torsion penalty. torsion = (x' x x").x'"/|x' x x"|^2 has a
denominator that vanishes wherever the curve is locally straight -- which is exactly the
nearly straight legs these encircling coils have, and the same kappa ~ 0 legs that made
the paper-form convexity stall the solver (see sandbox/convexity.py).

Each segment's plane is fixed by three of its own knots (first, middle, last), and the
residual is the SIGNED distance of every other knot from that plane. Signed, so the
Jacobian does not vanish at the solution -- unlike a squared or absolute out-of-plane
distance, which is degenerate at its own bound in the way `coil_sandbox/CLAUDE.md`
warns about. Full rank: K - 3 rows for a segment of K knots, with the plane's own three
degrees of freedom carried by the anchors.
"""

import numpy as np

from desc.backend import jnp, tree_leaves
from desc.objectives.objective_funs import _Objective
from desc.utils import cross, dot, errorif, safenorm


def segment_knots(coil):
    """List of knot-index lists, one per C0 segment, each including both break knots."""
    iv = np.asarray(coil.intervals)
    n = int(np.asarray(coil.knots).size)
    if iv.size == 0:  # no breaks: one closed segment
        return [list(range(n)) + [0]]
    b0 = int(iv[0][1])  # smallest break index; shift so segments are contiguous
    edges = np.concatenate([np.sort((iv[:, 0] - b0) % n), [n]]).astype(int)
    return [
        [int((b0 + p) % n) for p in range(lo, hi + 1)]
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def planarity_rows(coil):
    """Index arrays (a0, a1, a2, j) -- one row per constrained knot, over all segments.

    Rows j are every knot of a segment other than its three anchors; the plane is fixed
    by the anchors a0 (first), a1 (middle), a2 (last) of that same segment.
    """
    a0, a1, a2, jj = [], [], [], []
    for idx in segment_knots(coil):
        anchors = (idx[0], idx[len(idx) // 2], idx[-1])
        errorif(
            len(set(anchors)) != 3,
            ValueError,
            f"a C0 segment needs at least 3 distinct knots to fix a plane, got {idx}",
        )
        for j in idx:
            if j not in anchors:
                a0.append(anchors[0])
                a1.append(anchors[1])
                a2.append(anchors[2])
                jj.append(j)
    return tuple(np.asarray(v, dtype=int) for v in (a0, a1, a2, jj))


def out_of_plane(P, rows):
    """Signed distance of each constrained knot from its segment's anchor plane."""
    a0, a1, a2, jj = rows
    a = P[a0]
    N = cross(P[a1] - a, P[a2] - a)
    return dot(P[jj] - a, N / safenorm(N, axis=-1, keepdims=True))


def planarity_excess(coil):
    """Max |out-of-plane distance| over the constrained knots, in metres (numpy float)."""
    P = np.stack([np.asarray(coil.X), np.asarray(coil.Y), np.asarray(coil.Z)], axis=1)
    return float(np.abs(np.asarray(out_of_plane(P, planarity_rows(coil)))).max())


class CoilSetPlanarity(_Objective):
    """Signed out-of-plane distance of every spline knot from its own segment's plane.

    Target 0 makes each C0 segment exactly planar; a bounds pair instead allows a
    tolerance, which is the useful form if you want the augmented Lagrangian to explore
    off the planar manifold before clamping onto it.

    Parameters
    ----------
    coil : CoilSet or MixedCoilSet
        Coils to constrain. Every leaf coil must be a SplineXYZCoil.
    """

    _scalar = False
    _units = "(m)"
    _print_value_fmt = "Coil out-of-plane distance: "
    _static_attrs = _Objective._static_attrs + ["_rows"]

    def __init__(self, coil, target=None, bounds=None, weight=1, normalize=True,
                 normalize_target=True, loss_function=None, deriv_mode="auto",
                 name="coil planarity", jac_chunk_size=None):
        if target is None and bounds is None:
            target = 0
        super().__init__(things=coil, target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode,
                         jac_chunk_size=jac_chunk_size, name=name)

    def build(self, use_jit=True, verbose=1):
        """Precompute the per-coil index arrays (static) and the normalization."""
        from desc.coils import SplineXYZCoil

        coils = _leaf_coils(self.things[0])
        errorif(
            not all(isinstance(c, SplineXYZCoil) for c in coils),
            ValueError,
            f"CoilSetPlanarity needs SplineXYZCoil coils, got {[type(c).__name__ for c in coils]}",
        )
        # tuples so the arrays stay hashable/static across jit
        self._rows = tuple(
            tuple(tuple(int(i) for i in v) for v in planarity_rows(c)) for c in coils
        )
        self._dim_f = sum(len(r[3]) for r in self._rows)
        if self._normalize:  # a coil "radius", so the residual reads as a fraction
            self._normalization = float(np.mean([
                np.linalg.norm(
                    np.stack([c.X, c.Y, c.Z], 1) - np.stack([c.X, c.Y, c.Z], 1).mean(0),
                    axis=1,
                ).mean()
                for c in coils
            ]))
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Signed out-of-plane distance of every constrained knot, 1D."""
        leaves = tree_leaves(params, is_leaf=lambda x: isinstance(x, dict))
        return jnp.concatenate([
            out_of_plane(
                jnp.stack([p["X"], p["Y"], p["Z"]], axis=1),
                tuple(np.asarray(v) for v in rows),
            )
            for p, rows in zip(leaves, self._rows)
        ])


def _leaf_coils(coilset):
    from desc.coils import CoilSet, MixedCoilSet

    out = []
    for c in (coilset if isinstance(coilset, (CoilSet, MixedCoilSet)) else [coilset]):
        out += _leaf_coils(c) if isinstance(c, (CoilSet, MixedCoilSet)) else [c]
    return out
