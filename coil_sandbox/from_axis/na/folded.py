"""Two-face folded coils (transverse arcB2) + Giuliani-style near-axis single-stage objective, in JAX.

Coil i (one of nc unique coils in the half period): two hinges Ha (inboard), Hb (outboard).
  z0 mode : hinges at z = 0, given by (R, phi) each.
  free    : hinges at (R, phi, z) each (any chord; the arcB2 baseline).
Upper arc  x(t) = Ha + t c + w_u(t) u_u,  lower arc x(t) = Hb - t c + w_l(t) u_l,  t in [0, 1],
w(t) = |c| sum_k a_k sin(k pi t); u_u, u_l are unit vectors perpendicular to the chord c, at tilt
angles alpha_u, alpha_l from the "vertical" (the component of z perpendicular to c). Each arc is planar
(it lies in span(c, u)), so each coil is exactly two planar faces joined at the two hinges.
Full coil set: stellarator image (x, -y, -z) with current -I, then rotations by 2 pi j / nfp.
Currents are in units of mu0 / (4 pi) (so B = sum I dl x r / |r|^3).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

NPT = 80  # segments per arc


class Layout:
    """Index bookkeeping for the flat parameter vector."""

    def __init__(self, nfp, nc, K, Kax, mode="z0"):
        self.nfp, self.nc, self.K, self.Kax, self.mode = nfp, nc, K, Kax, mode
        self.nh = 4 if mode == "z0" else 6
        self.per = self.nh + 2 + 2 * K + 1
        self.n_axis = 2 * Kax + 1 + 1  # rc[0..Kax], zs[1..Kax], etabar
        self.n = self.n_axis + nc * self.per

    def split(self, p):
        Kax = self.Kax
        rc = p[: Kax + 1]
        zs = jnp.concatenate([jnp.zeros(1), p[Kax + 1: 2 * Kax + 1]])
        eta = p[2 * Kax + 1]
        C = p[self.n_axis:].reshape(self.nc, self.per)
        return rc, zs, eta, C

    def join(self, rc, zs, eta, C):
        return jnp.concatenate([rc, zs[1:], jnp.atleast_1d(eta), jnp.ravel(C)])


def _cyl(R, ph, z):
    return jnp.stack([R * jnp.cos(ph), R * jnp.sin(ph), z])


def coil_arcs(row, L: Layout):
    """Returns (upper, lower) arc point arrays (NPT+1, 3), their first/second t-derivatives, current."""
    K, nh = L.K, L.nh
    if L.mode == "z0":
        Ha = _cyl(row[0], row[1], 0.0)
        Hb = _cyl(row[2], row[3], 0.0)
    else:
        Ha = _cyl(row[0], row[1], row[2])
        Hb = _cyl(row[3], row[4], row[5])
    al_u, al_l = row[nh], row[nh + 1]
    au = row[nh + 2: nh + 2 + K]
    alw = row[nh + 2 + K: nh + 2 + 2 * K]
    I = row[nh + 2 + 2 * K]
    c = Hb - Ha
    Lc = jnp.linalg.norm(c)
    ch = c / Lc
    ez = jnp.array([0.0, 0.0, 1.0])
    v = ez - jnp.dot(ez, ch) * ch
    v = v / jnp.linalg.norm(v)
    h = jnp.cross(ch, v)
    uu = jnp.cos(al_u) * v + jnp.sin(al_u) * h
    ul = -jnp.cos(al_l) * v + jnp.sin(al_l) * h
    t = jnp.linspace(0, 1, NPT + 1)
    k = jnp.arange(1, K + 1)
    S = jnp.sin(jnp.pi * k[None] * t[:, None])
    Cs = jnp.cos(jnp.pi * k[None] * t[:, None]) * (jnp.pi * k)[None]
    S2 = -S * ((jnp.pi * k) ** 2)[None]

    def arc(H, sgn, a, u):
        w, w1, w2 = Lc * S @ a, Lc * Cs @ a, Lc * S2 @ a
        x = H[None] + sgn * t[:, None] * c[None] + w[:, None] * u[None]
        x1 = sgn * c[None] + w1[:, None] * u[None]
        x2 = w2[:, None] * u[None]
        return x, x1, x2

    return arc(Ha, 1.0, au, uu), arc(Hb, -1.0, alw, ul), I


def full_coilset(C, L: Layout):
    """Polyline segments of every coil: (ncoil_total, 2*NPT, 3) start points, end points, currents."""
    xs, Is = [], []
    for i in range(L.nc):
        (xu, _, _), (xl, _, _), I = coil_arcs(C[i], L)
        x = jnp.concatenate([xu, xl[1:]], 0)  # closed loop, last point == first
        xs.append(x)
        Is.append(I)
    xs = jnp.stack(xs)  # (nc, 2NPT+1, 3)
    Is = jnp.stack(Is)
    img = xs * jnp.array([1.0, -1.0, -1.0])
    base = jnp.concatenate([xs, img], 0)
    Ib = jnp.concatenate([Is, -Is])
    allx, allI = [], []
    for j in range(L.nfp):
        a = 2 * jnp.pi * j / L.nfp
        Rm = jnp.array([[jnp.cos(a), -jnp.sin(a), 0], [jnp.sin(a), jnp.cos(a), 0], [0, 0, 1.0]])
        allx.append(base @ Rm.T)
        allI.append(Ib)
    return jnp.concatenate(allx), jnp.concatenate(allI)


def biot_savart(r, X, I):
    """B at point r from closed polylines X (n, m+1, 3) with currents I (n,). Exact for straight segments."""
    a = r[None, None] - X[:, :-1]
    b = r[None, None] - X[:, 1:]
    na = jnp.linalg.norm(a, axis=-1)
    nb = jnp.linalg.norm(b, axis=-1)
    f = (na + nb) / (na * nb * (na * nb + jnp.sum(a * b, -1)))
    Bseg = jnp.cross(a, b) * f[..., None]
    return jnp.sum(I[:, None, None] * Bseg, (0, 1))


def B_and_gradB(r, X, I):
    B = biot_savart(r, X, I)
    G = jax.jacfwd(biot_savart)(r, X, I)  # G[j, i] = d_i B_j
    return B, G.T  # return G[i, j] = d_i B_j


def arc_curvature(x1, x2):
    cr = jnp.cross(x1, x2)
    return jnp.sqrt(jnp.sum(cr**2, -1) + 1e-30) / jnp.linalg.norm(x1, axis=-1) ** 3


def coil_length(row, L):
    (xu, _, _), (xl, _, _), _ = coil_arcs(row, L)
    x = jnp.concatenate([xu, xl[1:]], 0)
    return jnp.sum(jnp.linalg.norm(jnp.diff(x, axis=0), axis=-1))
