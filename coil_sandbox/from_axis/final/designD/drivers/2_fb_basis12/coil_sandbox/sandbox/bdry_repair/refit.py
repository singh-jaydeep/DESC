"""Re-parametrize a boundary: same surface (as nearly as the resolution allows), well-behaved poloidal angle.

Two stages, each usable alone:

1. `relabelled_points`: points of the old surface at uniform values of a new poloidal label, per zeta plane. The label
   rule is one of PR 2337's ("chordal" = equal arclength, "curvature" = kappa^(1/3) ds, "centripetal", "uniform").
   A linear fit to these labels is the old repair. It is clean but moves the surface by the truncation error of that
   particular angle (2.6-4.2 mm for the elongation-5 iota 0.42 design at M = N = 12).
2. `guarded_fit`: PR 2337's joint fit of labels and coefficients, started from those labels and advanced ONE
   Levenberg-Marquardt step at a time, keeping a step only while the angle stays healthy. Unguarded, the joint fit
   returns to the creased angle, because that is the one parametrization that reproduces the old surface, kink
   included, exactly; a few guarded steps remove most of the truncation error and stop before the crease returns.
"""
import numpy as np
from desc.backend import jnp
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid

from . import _surface_fit_pr2337 as sf
from .diagnostics import boundary_health, displacement


def relabelled_points(obj, NFP, rule="chordal", nt=160, nz=80, nf=4001):
    """(coords (nt*nz, 3) in R, phi, Z; label u; zeta). theta_old = 0 stays u = 0, which keeps stellarator symmetry."""
    tf = np.linspace(0, 2 * np.pi, nf)
    z = np.linspace(0, 2 * np.pi / NFP, nz, endpoint=False)
    g = LinearGrid(rho=np.array([1.0]), theta=tf[:-1], zeta=z, NFP=NFP)
    d = obj.compute(["R", "Z"], grid=g)
    R = np.asarray(g.meshgrid_reshape(d["R"], "rtz")[0])
    Z = np.asarray(g.meshgrid_reshape(d["Z"], "rtz")[0])
    xyz = np.stack([R * np.cos(z), R * np.sin(z), Z], -1)                         # (nf-1, nz, 3)
    inc = np.asarray(sf._spacing(jnp.asarray(np.swapaxes(xyz, 0, 1)), rule)).T    # label increment i -> i+1
    tau = np.concatenate([np.zeros((1, nz)), np.cumsum(inc, 0)], 0)
    tau = 2 * np.pi * tau / tau[-1]
    u = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    th = np.stack([np.interp(u, tau[:, j], tf) for j in range(nz)], 1)            # old theta at each new label
    zz = np.broadcast_to(z, th.shape).ravel()
    nodes = np.stack([np.ones(nt * nz), th.ravel(), zz], 1)
    d = obj.compute(["R", "Z"], grid=Grid(nodes, sort=False, jitable=True, NFP=NFP))
    coords = np.stack([np.asarray(d["R"]), zz, np.asarray(d["Z"])], 1)
    return coords, np.broadcast_to(u[:, None], th.shape).ravel(), zz


def _surface(bases, x, NFP, sym):
    R_lmn, _, Z_lmn = sf._unpack(bases, x)
    return FourierRZToroidalSurface(R_lmn, Z_lmn, bases[0].modes[:, 1:], bases[2].modes[:, 1:], NFP, bool(sym))


def guarded_fit(old, coords, u, zeta, M, N, NFP, sym, max_steps=8, area_keep=0.85, kappa_grow=1.5, min_gain=0.03,
                say=None):
    """Fit at (M, N); labels start at u and take up to max_steps joint steps (0 = plain linear fit at fixed labels).

    A step is kept while, relative to the fixed-label fit, the smallest area element stays above area_keep x and the
    largest curvature below kappa_grow x, and it stops once a step lowers the rms distance to the old surface by less
    than min_gain (relative). Returns (surface, history); history[i] = dict(step, area, kappa, rms, p99, max, kept).
    """
    bases = sf._fourier_bases(M, N, 0, 0, NFP, sym, False)
    data = sf._prepare_data(coords, zeta)
    t, z = jnp.asarray(u), jnp.asarray(zeta)
    best, hist, base = None, [], None
    for step in range(max_steps + 1):
        x, t, z, _ = sf._optimize_labels(bases, data, t, z, 0 if step == 0 else 1, 0.0, 0.0)
        surf = _surface(bases, x, NFP, sym)
        area, kappa = boundary_health(surf, NFP)
        rec = dict(step=step, area=area, kappa=kappa, **displacement(old, surf, NFP))
        base = base or rec
        rec["kept"] = bool(step == 0 or (area >= area_keep * base["area"] and kappa <= kappa_grow * base["kappa"]
                                         and rec["rms"] < hist[-1]["rms"]))
        hist.append(rec)
        if say:
            say(f"  refit step {step}: moved rms {rec['rms']:.4f} p99 {rec['p99']:.3f} max {rec['max']:.3f} mm, area "
                f"min/median {area:.3f}, max curvature {kappa:.0f} /m{'' if rec['kept'] else '  (rejected)'}")
        if not rec["kept"]:
            break
        gain = 1.0 if step == 0 else 1 - rec["rms"] / hist[-2]["rms"]
        best = surf
        if gain < min_gain:
            break
    return best, hist


def reparametrize(obj, M=None, N=None, NFP=None, sym=None, rule="chordal", max_steps=8, nt=None, nz=None, say=None,
                  **guards):
    """Drop-in refit: a new FourierRZToroidalSurface for the rho = 1 surface of obj (an Equilibrium or a surface whose
    toroidal angle is the cylindrical one), same shape, healthy poloidal angle. Returns (surface, history).

    M, N, NFP, sym default to obj's. max_steps = 0 gives the plain fixed-label fit; guards go to guarded_fit.
    history[-1] (or the last entry with kept=True) says how far the surface moved [mm] and how healthy the angle is.
    """
    M, N = M or obj.M, N or obj.N
    NFP, sym = NFP or obj.NFP, obj.sym if sym is None else sym
    nt = nt or max(160, 10 * M)
    nz = nz or max(80, 6 * N)
    coords, u, zeta = relabelled_points(obj, NFP, rule, nt, nz)
    return guarded_fit(obj, coords, u, zeta, M, N, NFP, sym, max_steps=max_steps, say=say, **guards)
