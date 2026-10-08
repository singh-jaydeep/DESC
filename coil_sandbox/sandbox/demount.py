"""Vertical demountability: can each piece of a hinged coil be lifted off the torus?

A rigid piece translates to infinity along a direction d without touching the plasma iff
no point of it lies in the plasma's SHADOW along d,

    S_d = { p : p + t d is inside the plasma for some t > 0 }.

For d = +/- z that shadow has a very cheap description. At a fixed toroidal angle the
plasma cross-section is simply connected, so the radii it covers form an INTERVAL: the
vertical line at (R, phi) meets the plasma iff

    R_in(phi) <= R <= R_out(phi),    R_out = max_theta R,   R_in = min_theta R,

and inside that annulus the line is inside the plasma between z_bot(R, phi) and
z_top(R, phi). So p = (R, phi, z) is BLOCKED going up iff it is inside the annulus and
below z_top, and blocked going down iff inside and above z_bot. Nothing here refers to a
chord, a cutting plane or a "width", all of which are ambiguous for a torus (a vertical
plane through two hinges generally cuts it in TWO closed curves, near side and far side).

The plasma is FIXED during stage 2, so R_in, R_out, z_top, z_bot are constants of the
problem: computed once here, interpolated at run time. No plasma quantity is ever
differentiated and no coil-plasma pair list is formed, so this costs far less than either
distance penalty.

Two objectives:

  HingeShadowClearance  one residual per hinge, in metres: how far OUTSIDE the annulus
                        the hinge sits. Hinges are PARAMETERS, not computed geometry, so
                        this reads params["hinges"] directly -- no grid, no transforms,
                        and none of the per-node _dim_f broadcasting that corners.py has
                        to work around. Bound BELOW by the margin you want.

  CoilDemountPenalty    one residual per grid node: a C1 penalty, zero unless the node is
                        blocked in its own arc's removal direction. The hinge condition is
                        necessary but NOT sufficient -- an arc can dip back under the
                        plasma between two perfectly legal hinges. Measured on the 16-coil
                        M=10 optimum: coil 3's hinge is free while 0.4% of its arc is not.

Two deliberate choices, both for the solver's sake:

* "Outside the annulus" is R <= R_in OR R >= R_out, which as a max() is kinked halfway
  across the annulus. Which side a hinge is on is fixed topology, so the side is assigned
  ONCE at build from the starting coilset and the residual is the single smooth one-sided
  expression. Same reasoning as the per-hinge corner angle in corners.py, and the opposite
  of the paper-form convexity that stalled the solver.

* In the penalty the radial factor is NOT clipped while the vertical one is. A blocked
  node escapes either radially (out of the annulus) or vertically (past the plasma in its
  own removal direction); a node running UNDER the plasma has no vertical escape, so the
  radial gradient has to survive at any depth. The vertical factor is only a switch.

`demount_report` is the independent, approximation-free check (explicit ray casting
against the boundary polygon at each point's own azimuth) used to validate the objectives
and to give the verdict on a finished coilset.
"""

import numpy as np
from interpax import interp1d, interp2d

from desc.backend import jnp, tree_leaves
from desc.grid import LinearGrid
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective


def _hull_chains(r, z):
    """Upper and lower convex chains of a cross-section, left to right in R.

    Andrew's monotone chain. Returns (upper, lower), each (k, 2) of (R, z) with R
    increasing, so np.interp on them is the hull's top and bottom as functions of R.
    """
    p = np.stack([np.asarray(r), np.asarray(z)], axis=1)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2:
                a, b = h[-1] - h[-2], q - h[-2]
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                h.pop()
            h.append(q)
        return np.array(h)

    lower = half(p)
    upper = half(p[::-1])[::-1]
    assert upper[:, 1].max() >= lower[:, 1].max(), "hull chains came out swapped"
    return upper, lower


def _leaf_coils(coilset):
    from desc.coils import CoilSet, MixedCoilSet

    out = []
    for c in (coilset if isinstance(coilset, (CoilSet, MixedCoilSet)) else [coilset]):
        out += _leaf_coils(c) if isinstance(c, (CoilSet, MixedCoilSet)) else [c]
    return out


def _leaf_params(tree):
    """The per-coil parameter dicts, depth first -- the order _leaf_coils uses."""
    if isinstance(tree, dict):
        return [tree]
    out = []
    for t in tree:
        out += _leaf_params(t)
    return out


def _hinges_xyz(p):
    """Hinge points of one coil, in the lab frame."""
    h = jnp.asarray(p["hinges"]).reshape(-1, 3)
    rot = jnp.asarray(p["rotmat"]).reshape(3, 3)
    return h @ rot.T + jnp.asarray(p["shift"]).reshape(3)


def _cyl(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return jnp.hypot(x, y), jnp.arctan2(y, x), z


def _smoothstep(t):
    """0 for t <= 0, 1 for t >= 1, C1 in between."""
    s = jnp.clip(t, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def annulus(C, phi):
    """R_in(phi), R_out(phi) from the shadow arrays."""
    return (interp1d(phi, C["phi"], C["Rin"], period=2 * np.pi),
            interp1d(phi, C["phi"], C["Rout"], period=2 * np.pi))


def depth(C, R, phi):
    """Metres inside the shadow annulus; <= 0 outside it, smooth across both edges.

    u(1-u)(R_out-R_in) with u the normalised radius: it equals the distance to the near
    edge to first order, is exactly zero on both edges, and is a polynomial in R (no
    min(), so no kink down the middle of the annulus).
    """
    Rin, Rout = annulus(C, phi)
    w = Rout - Rin
    u = (R - Rin) / w
    return u * (1.0 - u) * w


def z_extreme(C, R, phi, sign):
    """Top (sign > 0) or bottom (sign < 0) of the hull on the vertical line at (R, phi).

    The cosine stretch undoes the square-root turning point at the edges of the annulus.
    Its derivative blows up there, but it is always multiplied by a radial factor that
    vanishes at the same place, so the product stays differentiable.
    """
    Rin, Rout = annulus(C, phi)
    u = jnp.clip((R - Rin) / (Rout - Rin), 1e-10, 1.0 - 1e-10)
    vq = jnp.arccos(1.0 - 2.0 * u) / jnp.pi
    zt = interp2d(phi, vq, C["phi"], C["v"], C["ZT"], period=(2 * np.pi, None))
    zb = interp2d(phi, vq, C["phi"], C["v"], C["ZB"], period=(2 * np.pi, None))
    return jnp.where(sign > 0, zt, zb)


def blockage(C, R, phi, z, sign, delta, eps):
    """0 where the point can leave along sign*z; grows with how badly it cannot."""
    radial = jnp.maximum(depth(C, R, phi), 0.0) / delta
    gap = sign * (z_extreme(C, R, phi, sign) - z)
    return radial * _smoothstep(gap / eps)


class PlasmaShadow:
    """The shadow the fixed plasma boundary casts along +/- z.

    Parameters
    ----------
    eq : Equilibrium
        Fixed plasma. Only its boundary is used.
    nzeta, ntheta : int
        Boundary sampling for the precompute (full torus, no symmetry).
    nu : int
        Radial samples of z_top / z_bot across the annulus. They are placed on a cosine
        grid: z_top has a square-root turning point at each edge of the annulus (the
        vertical line becomes tangent to the surface there), and a cosine grid resolves
        that the way a uniform grid cannot.

    z_top and z_bot are taken from the CONVEX HULL of each cross-section, not from the
    cross-section itself. The true z_top is discontinuous -- a shaped cross-section
    overhangs, so the topmost ray crossing jumps as the ray moves past the overhang
    (measured on helios_repro: a 2.70 m jump, and no grid removes it). The hull's upper
    chain is continuous and concave (largest step 0.054 m over 2001 samples of the same
    azimuth), and it CONTAINS the plasma, so using it can only over-report blockage: a
    coilset feasible under this model is genuinely removable. What it costs is freedom
    inside a concave bay of the cross-section, which an encircling coil wrapped around
    the whole cross-section does not use -- verify with `demount_report`, which ray casts
    against the true boundary.
    """

    def __init__(self, eq, nzeta=512, ntheta=512, nu=96):
        g = LinearGrid(rho=np.array([1.0]), theta=ntheta, zeta=nzeta, NFP=1, sym=False)
        x = np.asarray(eq.compute("x", grid=g, basis="rpz")["x"])
        R = x[:, 0].reshape(nzeta, ntheta)
        Z = x[:, 2].reshape(nzeta, ntheta)
        phi = np.asarray(g.nodes[:, 2]).reshape(nzeta, ntheta)
        assert np.ptp(phi, axis=1).max() < 1e-12, "grid is not zeta-major"
        phi = phi[:, 0]

        Rin, Rout = R.min(axis=1), R.max(axis=1)
        v = np.linspace(0.0, 1.0, nu)
        u = (1.0 - np.cos(np.pi * v)) / 2.0
        ZT = np.zeros((nzeta, nu))
        ZB = np.zeros((nzeta, nu))
        for i in range(nzeta):
            up, lo = _hull_chains(R[i], Z[i])
            Rq = Rin[i] + u * (Rout[i] - Rin[i])
            ZT[i] = np.interp(Rq, up[:, 0], up[:, 1])
            ZB[i] = np.interp(Rq, lo[:, 0], lo[:, 1])

        self.phi = jnp.asarray(phi)
        self.v = jnp.asarray(v)
        self.Rin = jnp.asarray(Rin)
        self.Rout = jnp.asarray(Rout)
        self.ZT = jnp.asarray(ZT)
        self.ZB = jnp.asarray(ZB)
        self._np = dict(phi=phi, Rin=Rin, Rout=Rout, R=R, Z=Z, nzeta=nzeta)

    @property
    def arrays(self):
        """The shadow as a pytree of plain arrays.

        The objectives put THIS in their `_constants`, not the object: constants are
        traced, and a Python object there fails at the first jit with "Error interpreting
        argument to ObjectiveFunction.x as an abstract array".
        """
        return {"phi": self.phi, "v": self.v, "Rin": self.Rin, "Rout": self.Rout,
                "ZT": self.ZT, "ZB": self.ZB}

    def annulus(self, phi):
        """R_in(phi), R_out(phi)."""
        return annulus(self.arrays, phi)

    def depth(self, R, phi):
        """Metres inside the shadow annulus; <= 0 outside it, smooth across both edges.

        u(1-u)(R_out-R_in) with u the normalised radius: it equals the distance to the
        near edge to first order, is exactly zero on both edges, and is a polynomial in R
        (no min(), so no kink down the middle of the annulus).
        """
        return depth(self.arrays, R, phi)

    def z_extreme(self, R, phi, sign):
        """z_top where sign > 0, z_bot where sign < 0."""
        return z_extreme(self.arrays, R, phi, sign)

    def blockage(self, R, phi, z, sign, delta, eps):
        """0 where the point can leave along sign*z; grows with how badly it cannot."""
        return blockage(self.arrays, R, phi, z, sign, delta, eps)

    def clearance_np(self, R, phi):
        """Signed metres OUTSIDE the annulus (numpy, exact at the sample azimuths)."""
        i = np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi) * self._np["nzeta"]).astype(int)
        i = np.mod(i, self._np["nzeta"])
        return np.maximum(self._np["Rin"][i] - R, R - self._np["Rout"][i])

    def blocked_np(self, R, phi, z, sign):
        """Exact ray cast: is the point blocked moving along sign*z? (numpy, no model.)"""
        nz = self._np["nzeta"]
        Rg, Zg = self._np["R"], self._np["Z"]
        out = np.zeros(np.shape(R), dtype=bool)
        idx = np.mod(np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi) * nz).astype(int), nz)
        for k in range(np.size(R)):
            i = idx.flat[k]
            r, zz = Rg[i], Zg[i]
            r2, z2 = np.roll(r, -1), np.roll(zz, -1)
            Rq = np.ravel(R)[k]
            m = (r - Rq) * (r2 - Rq) < 0.0
            if not m.any():
                continue
            t = (Rq - r[m]) / (r2[m] - r[m])
            zc = zz[m] + t * (z2[m] - zz[m])
            zq = np.ravel(z)[k]
            out.flat[k] = (zc.max() > zq) if np.ravel(sign)[k] > 0 else (zc.min() < zq)
        return out


def arc_directions(coilset, shadow=None):
    """Which way each arc comes off: +1 lifts in +z, -1 lowers in -z.

    Fixed topology, decided once from the starting geometry: an arc leaves along +z if its
    midpoint is above the chord joining its two hinges.
    """
    from desc.grid import Grid

    out = []
    for c in _leaf_coils(coilset):
        B = int(c.B)
        h = np.asarray(c.params_dict["hinges"]).reshape(-1, 3)
        s = (np.arange(B) + 0.5) * 2 * np.pi / B
        mid = np.asarray(c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], axis=1),
                                                  spacing=np.ones((B, 3)), sort=False,
                                                  jitable=False), basis="xyz")["x"])
        chord = 0.5 * (h[np.arange(B), 2] + h[(np.arange(B) + 1) % B, 2])
        out.append(np.where(mid[:, 2] >= chord, 1.0, -1.0))
    return np.stack(out)


def hinge_sides(coilset, shadow):
    """+1 if a hinge escapes outboard (R >= R_out), -1 if inboard (R <= R_in)."""
    out = []
    for c in _leaf_coils(coilset):
        h = np.asarray(c.params_dict["hinges"]).reshape(-1, 3)
        R = np.hypot(h[:, 0], h[:, 1])
        phi = np.arctan2(h[:, 1], h[:, 0])
        i = np.mod(np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi) * shadow._np["nzeta"]
                            ).astype(int), shadow._np["nzeta"])
        mid = 0.5 * (shadow._np["Rin"][i] + shadow._np["Rout"][i])
        out.append(np.where(R >= mid, 1.0, -1.0))
    return np.stack(out)


class HingeShadowClearance(_Objective):
    """Signed distance of every hinge OUTSIDE the plasma's vertical shadow, in metres.

    Bound below by the margin you want: ``bounds=(margin, np.inf)``.

    Parameters
    ----------
    coil : CoilSet or Coil
        Arc coils (all with the same B).
    eq : Equilibrium
        Fixed plasma, used only to build the shadow.
    shadow : PlasmaShadow, optional
        Reuse a shadow already built for this equilibrium.
    sides : ndarray, optional
        (n_coils, B) of +/-1, overriding the outboard/inboard assignment taken from the
        starting geometry. Pass this to hold an assignment across a warm restart.
    """

    _scalar = False
    _coordinates = ""
    _units = "(m)"
    _print_value_fmt = "Hinge shadow clearance: "
    # DESC flattens an objective's attributes as a pytree, so anything that is not an
    # array has to be declared static or the first jit dies in ObjectiveFunction.x.
    _static_attrs = _Objective._static_attrs + ["_shadow", "_eq", "_sides", "_B"]

    def __init__(self, coil, eq, shadow=None, sides=None, target=None, bounds=None,
                 weight=1, normalize=False, normalize_target=False, loss_function=None,
                 deriv_mode="auto", name="hinge shadow clearance", jac_chunk_size=None):
        if target is None and bounds is None:
            bounds = (0.0, np.inf)
        self._eq = eq
        self._shadow = shadow
        self._sides = sides
        super().__init__(things=[coil], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode, name=name,
                         jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

        coils = _leaf_coils(self.things[0])
        kinds = {type(c) for c in coils}
        if not kinds <= {PiecewisePlanarArcCoil, PolarPlanarArcCoil}:
            raise ValueError(f"HingeShadowClearance needs arc coils; got {kinds}")
        nb = {int(c.B) for c in coils}
        if len(nb) != 1:
            raise ValueError(f"HingeShadowClearance needs one B for all coils; got {nb}")
        for c in coils:
            rot = np.asarray(c.params_dict["rotmat"]).reshape(3, 3)
            if not (np.allclose(rot, np.eye(3))
                    and np.allclose(np.asarray(c.params_dict["shift"]), 0.0)):
                raise NotImplementedError(
                    "HingeShadowClearance assumes leaf coils carry no rigid transform; "
                    "this one does, and the lab-frame convention has never been tested.")
        if self._shadow is None:
            self._shadow = PlasmaShadow(self._eq)
        if self._sides is None:
            self._sides = hinge_sides(self.things[0], self._shadow)
        self._B = nb.pop()
        self._dim_f = len(coils) * self._B
        self._normalization = 1.0
        self._constants = {"shadow": self._shadow.arrays,
                           "side": jnp.asarray(np.ravel(self._sides)),
                           "quad_weights": jnp.ones(self._dim_f)}
        if verbose > 0:
            n_out = int(np.sum(np.ravel(self._sides) > 0))
            print(f"  hinge shadow: {self._dim_f} hinges, {n_out} outboard, "
                  f"{self._dim_f - n_out} inboard")
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Metres outside the shadow annulus, one per hinge."""
        if constants is None:
            constants = self.constants
        C, side = constants["shadow"], constants["side"]
        xyz = jnp.concatenate([_hinges_xyz(p) for p in _leaf_params(params)])
        R, phi, _ = _cyl(xyz)
        Rin, Rout = annulus(C, phi)
        return jnp.where(side > 0, R - Rout, Rin - R)


class CoilDemountPenalty(_CoilObjective):
    """Per-node C1 penalty for a coil node that cannot leave along its arc's direction.

    Zero on the whole feasible set, so use ``target=0``.

    Parameters
    ----------
    coil : CoilSet or Coil
        Arc coils (all with the same B).
    eq : Equilibrium
        Fixed plasma, used only to build the shadow.
    delta : float
        Radial scale, in metres: the penalty reads depth/delta, so delta sets how hard a
        node inside the annulus is pushed out relative to the other constraints.
    eps : float
        Vertical softening, in metres, of the "is there plasma in the way" switch.
    directions : ndarray, optional
        (n_coils, B) of +/-1 overriding the removal direction per arc.
    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Coil demount blockage: "
    _broadcast_input = "Node"
    _static_attrs = _CoilObjective._static_attrs + [
        "_shadow", "_eq", "_directions", "_delta", "_eps"]

    def __init__(self, coil, eq, shadow=None, delta=0.5, eps=0.25, directions=None,
                 target=None, bounds=None, weight=1, normalize=False,
                 normalize_target=False, loss_function=None, deriv_mode="auto", grid=None,
                 name="coil demount", jac_chunk_size=None):
        if target is None and bounds is None:
            target = 0.0
        self._eq = eq
        self._shadow = shadow
        self._delta = float(delta)
        self._eps = float(eps)
        self._directions = directions
        super().__init__(coil, ["x"], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode,
                         grid=grid if grid is not None else LinearGrid(N=150),
                         name=name, jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

        super().build(use_jit=use_jit, verbose=verbose)
        coils = _leaf_coils(self.things[0])
        kinds = {type(c) for c in coils}
        if not kinds <= {PiecewisePlanarArcCoil, PolarPlanarArcCoil}:
            raise ValueError(f"CoilDemountPenalty needs arc coils; got {kinds}")
        B = int(coils[0].B)
        if self._shadow is None:
            self._shadow = PlasmaShadow(self._eq)
        if self._directions is None:
            self._directions = arc_directions(self.things[0])
        # one sign per grid node, from the arc that node belongs to
        grid = self._constants["transforms"][0]["grid"]
        s = np.asarray(grid.nodes[:, 2])
        arc = np.clip((s / (2 * np.pi / B)).astype(int), 0, B - 1)
        self._sign = jnp.asarray(np.stack([d[arc] for d in self._directions]))
        self._constants["shadow"] = self._shadow.arrays
        self._constants["sign"] = self._sign
        self._constants["delta"] = jnp.asarray(self._delta)
        self._constants["eps"] = jnp.asarray(self._eps)
        self._normalization = 1.0
        if verbose > 0:
            up = int(np.sum(np.asarray(self._directions) > 0))
            print(f"  demount penalty: {grid.num_nodes} nodes/coil, {up} arcs lift +z, "
                  f"{np.size(self._directions) - up} lower -z, "
                  f"delta {self._delta} m, eps {self._eps} m")
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Blockage at every node of every coil, 1D."""
        if constants is None:
            constants = self.constants
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        C, sign = constants["shadow"], constants["sign"]
        out = []
        for i, d in enumerate(data):
            x = d["x"]
            out.append(blockage(C, x[:, 0], x[:, 1], x[:, 2], sign[i],
                                constants["delta"], constants["eps"]))
        return jnp.concatenate(out)


def demount_report(coilset, eq, shadow=None, directions=None, n=801, margin=0.0):
    """Exact, model-free verdict: per arc, is it blocked, and where.

    Explicit ray casting against the boundary polygon at each sampled point's own azimuth
    -- no interpolation, no smoothing, nothing shared with the objectives above. This is
    what validates them and what judges a finished run.

    Returns a list, one dict per unique coil, with per-arc blocked fractions, the worst
    depth into the shadow annulus, and the hinge clearances.
    """
    from desc.grid import Grid

    if shadow is None:
        shadow = PlasmaShadow(eq)
    coils = _leaf_coils(coilset)
    if directions is None:
        directions = arc_directions(coilset)
    out = []
    for i, c in enumerate(coils):
        B = int(c.B)
        h = np.asarray(c.params_dict["hinges"]).reshape(-1, 3)
        Rh = np.hypot(h[:, 0], h[:, 1])
        phih = np.arctan2(h[:, 1], h[:, 0])
        rec = {"hinge_clearance": [float(v) for v in shadow.clearance_np(Rh, phih)],
               "hinge_R": [float(v) for v in Rh], "hinge_z": [float(v) for v in h[:, 2]],
               "arcs": []}
        for b in range(B):
            s = np.linspace(b * 2 * np.pi / B, (b + 1) * 2 * np.pi / B, n)
            x = np.asarray(c.compute("x", grid=Grid(np.stack([0 * s, 0 * s, s], axis=1),
                                                    spacing=np.ones((n, 3)), sort=False,
                                                    jitable=False), basis="xyz")["x"])
            R = np.hypot(x[:, 0], x[:, 1])
            phi = np.arctan2(x[:, 1], x[:, 0])
            sgn = np.full(n, directions[i][b])
            bad = shadow.blocked_np(R, phi, x[:, 2], sgn)
            cl = shadow.clearance_np(R, phi)
            rec["arcs"].append({"dir": float(directions[i][b]),
                                "blocked_frac": float(bad.mean()),
                                "worst_depth": float(max(0.0, -cl.min())),
                                "blocked_depth": float(max(0.0, -cl[bad].min())
                                                       if bad.any() else 0.0)})
        rec["ok"] = (all(a["blocked_frac"] == 0.0 for a in rec["arcs"])
                     and min(rec["hinge_clearance"]) >= margin)
        out.append(rec)
    return out


class HingeLevel(_Objective):
    """Every hinge of a coil at the SAME height, with that height left free.

    B-1 residuals per coil, z_b - z_0. Use ``target=0``.

    This is the weaker half of a demountable joint plane. `--fix-hinge-z` pins the height as
    well; stellarator symmetry sends a hinge at z to -z, so a single pinned height z* != 0
    gives the machine TWO joint planes (+z* and -z*) and only z* = 0 gives one. This objective
    pins nothing: each coil's chord is horizontal but each may sit at its own height, so the
    machine has one plane per coil. The gap between the two measures what the COMMON PLANE
    costs, as distinct from what HORIZONTALITY costs.

    It is LINEAR in the optimization parameters -- hinges are parameters and this is a
    difference of two of them -- so with a target (not bounds) `factorize_linear_constraints`
    eliminates it exactly: no augmented-Lagrangian multiplier and no second-order Hessian
    block. That is worth having here, since carrying one extra NONLINEAR constraint through
    the ladder doubled the number of distinct active block sets and walked RSS into the 12 GB
    cap twice.

    Linearity relies on the leaf coils carrying no rigid transform, since `rotmat` is itself a
    parameter and `rotmat @ hinges` would be bilinear. Checked at build. In this workflow the
    CoilSet applies the symmetry itself and the leaves stay at identity -- verified on both
    the M10_c100 and the midplane lineages after optimization.
    """

    _scalar = False
    _coordinates = ""
    _units = "(m)"
    _print_value_fmt = "Hinge level: "
    _linear = True
    _fixed = False  # a linear combination of parameters, not individual ones
    _static_attrs = _Objective._static_attrs + ["_B"]

    def __init__(self, coil, target=None, bounds=None, weight=1, normalize=False,
                 normalize_target=False, loss_function=None, deriv_mode="auto",
                 name="hinge level", jac_chunk_size=None):
        if target is None and bounds is None:
            target = 0.0
        super().__init__(things=[coil], target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode, name=name,
                         jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        from desc.coils import PiecewisePlanarArcCoil, PolarPlanarArcCoil

        coils = _leaf_coils(self.things[0])
        kinds = {type(c) for c in coils}
        if not kinds <= {PiecewisePlanarArcCoil, PolarPlanarArcCoil}:
            raise ValueError(f"HingeLevel needs arc coils; got {kinds}")
        nb = {int(c.B) for c in coils}
        if len(nb) != 1:
            raise ValueError(f"HingeLevel needs one B for all coils; got {nb}")
        for c in coils:
            rot = np.asarray(c.params_dict["rotmat"]).reshape(3, 3)
            if not (np.allclose(rot, np.eye(3))
                    and np.allclose(np.asarray(c.params_dict["shift"]), 0.0)):
                raise NotImplementedError(
                    "HingeLevel reads the hinge z components directly, which is only the lab "
                    "height when the leaf coil carries no rigid transform; this one does, and "
                    "rotmat is itself a parameter, so the constraint would not be linear.")
        self._B = nb.pop()
        self._dim_f = len(coils) * (self._B - 1)
        self._normalization = 1.0
        self._constants = {"quad_weights": jnp.ones(self._dim_f)}
        if verbose > 0:
            print(f"  hinge level: {self._dim_f} residuals ({len(coils)} coils, B={self._B}), "
                  "linear -- factorized out, not an AL constraint")
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """z_b - z_0 at every hinge past the first, per coil."""
        out = []
        for p in _leaf_params(params):
            z = jnp.asarray(p["hinges"]).reshape(-1, 3)[:, 2]
            out.append(z[1:] - z[0])
        return jnp.concatenate(out)
