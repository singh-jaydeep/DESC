"""First-order Garren-Boozer near-axis QS field in JAX (mirrors pyQSC r1, sG = spsi = 1, I2 = 0).

Given axis Fourier coefficients (R_k cos(k N phi), Z_k sin(k N phi)) and eta_bar, solve the sigma
equation for (sigma, iota) and return the on-axis field B_QS = B0 t and its gradient tensor
G[i, j] = d_i B_j in Cartesian components, as in Giuliani et al. JCP 2022 eq. (2)-(3).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def spectral_diff_matrix(n, xmax):
    # same as pyQSC's (n points on [0, xmax), periodic)
    h = 2 * np.pi / n
    kk = np.arange(1, n)
    n1, n2 = (n // 2, n // 2 - 1) if n % 2 == 0 else ((n - 1) // 2, (n - 1) // 2)
    if n % 2 == 0:
        topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, -np.flip(topc[0:n1])))
    else:
        topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, np.flip(topc[0:n1])))
    col1 = np.concatenate(([0], 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    from scipy.linalg import toeplitz
    return 2 * np.pi / xmax * toeplitz(col1, row1)


def axis_geometry(rc, zs, nfp, phi):
    """rc[0..K], zs[0..K] (zs[0] unused). Returns position and derivatives in Cartesian."""
    k = jnp.arange(len(rc)) * nfp
    c, s = jnp.cos(k[None] * phi[:, None]), jnp.sin(k[None] * phi[:, None])
    R = c @ rc
    Rp = -(s * k) @ rc
    Rpp = -(c * k**2) @ rc
    Rppp = (s * k**3) @ rc
    Z = s @ zs
    Zp = (c * k) @ zs
    Zpp = -(s * k**2) @ zs
    Zppp = -(c * k**3) @ zs
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    # derivatives of (R cos phi, R sin phi, Z) wrt phi (Leibniz)
    x0 = jnp.stack([R * cp, R * sp, Z], 1)
    x1 = jnp.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], 1)
    x2 = jnp.stack([Rpp * cp - 2 * Rp * sp - R * cp, Rpp * sp + 2 * Rp * cp - R * sp, Zpp], 1)
    x3 = jnp.stack([Rppp * cp - 3 * Rpp * sp - 3 * Rp * cp + R * sp,
                    Rppp * sp + 3 * Rpp * cp - 3 * Rp * sp - R * cp, Zppp], 1)
    return x0, x1, x2, x3, R, Z


def frenet(x1, x2, x3):
    dl = jnp.linalg.norm(x1, axis=1)
    cr = jnp.cross(x1, x2)
    ncr = jnp.linalg.norm(cr, axis=1)
    kappa = ncr / dl**3
    tors = jnp.sum(cr * x3, 1) / ncr**2
    t = x1 / dl[:, None]
    b = cr / ncr[:, None]
    n = jnp.cross(b, t)
    return dl, kappa, tors, t, n, b


def helicity(n_cart, phi, nfp):
    """pyQSC's quadrant count (numpy, non-differentiable; sG = spsi = 1)."""
    n_cart = np.asarray(n_cart)
    phi = np.asarray(phi)
    nR = n_cart[:, 0] * np.cos(phi) + n_cart[:, 1] * np.sin(phi)
    nz = n_cart[:, 2]
    q = np.where(nR >= 0, np.where(nz >= 0, 1, 4), np.where(nz >= 0, 2, 3))
    q = np.append(q, q[0])
    cnt = 0
    for j in range(len(phi)):
        if q[j] == 4 and q[j + 1] == 1:
            cnt += 1
        elif q[j] == 1 and q[j + 1] == 4:
            cnt -= 1
        else:
            cnt += q[j + 1] - q[j]
    return cnt / 4


class NearAxis:
    def __init__(self, nfp, nphi=61, B0=1.0):
        self.nfp, self.nphi, self.B0 = nfp, nphi, B0
        self.phi = jnp.asarray(np.arange(nphi) * 2 * np.pi / nfp / nphi)
        self.D = jnp.asarray(spectral_diff_matrix(nphi, 2 * np.pi / nfp))

    def geometry(self, rc, zs):
        x0, x1, x2, x3, R, Z = axis_geometry(rc, zs, self.nfp, self.phi)
        dl, kappa, tors, t, n, b = frenet(x1, x2, x3)
        return dict(x=x0, dl=dl, kappa=kappa, tors=tors, t=t, n=n, b=b, R=R, Z=Z)

    def solve(self, rc, zs, etabar, hel, iota0=0.4, sigma0=None):
        """Returns dict with x, t, n, b, B (nphi,3), G (nphi,3,3) with G[q,i,j] = d_i B_j, iota, ..."""
        g = self.geometry(rc, zs)
        B0 = self.B0
        abs_G0 = jnp.mean(g["dl"]) * B0  # |G0| = B0 L / (2 pi)
        dl_dvarphi = abs_G0 / B0
        Dv = self.D / (B0 / abs_G0 * g["dl"])[:, None]
        e2k2 = etabar**2 / g["kappa"] ** 2
        hN = hel * self.nfp

        def res(x):
            sigma = x.at[0].set(0.0)
            iota = x[0]
            return Dv @ sigma + (iota + hN) * (e2k2**2 + 1 + sigma**2) - 2 * e2k2 * (-g["tors"]) * abs_G0 / B0

        def newton(x):
            def body(i, x):
                J = jax.jacfwd(res)(x)
                return x - jnp.linalg.solve(J, res(x))
            return jax.lax.fori_loop(0, 30, body, x)

        xg = jnp.zeros(self.nphi).at[0].set(iota0)
        xs = jax.lax.stop_gradient(newton(jax.lax.stop_gradient(xg)))
        # one Newton step with gradients flowing = exact implicit-function derivative at the root
        J = jax.jacfwd(res)(xs)
        x = xs - jnp.linalg.solve(J, res(x=xs))
        iota = x[0]
        sigma = x.at[0].set(0.0)
        iotaN = iota + hN
        X1c = etabar / g["kappa"]
        Y1s = g["kappa"] / etabar
        Y1c = g["kappa"] * sigma / etabar
        dX1c, dY1s, dY1c = Dv @ X1c, Dv @ Y1s, Dv @ Y1c
        f = B0 / dl_dvarphi
        tn = B0 * g["kappa"]
        bb = f * (X1c * dY1s - iotaN * X1c * Y1c)
        nn = f * (dX1c * Y1s + iotaN * X1c * Y1c)
        bn = f * (-dl_dvarphi * g["tors"] - iotaN * X1c**2)
        nb = f * (dY1c * Y1s - dY1s * Y1c + dl_dvarphi * g["tors"] + iotaN * (Y1s**2 + Y1c**2))
        t, n, b = g["t"], g["n"], g["b"]
        o = lambda u, v: u[:, :, None] * v[:, None, :]
        G = (nn[:, None, None] * o(n, n) + bn[:, None, None] * o(b, n) + nb[:, None, None] * o(n, b)
             + bb[:, None, None] * o(b, b) + tn[:, None, None] * (o(t, n) + o(n, t)))
        B = B0 * t
        resid = jnp.max(jnp.abs(res(x)))
        return dict(g, B=B, G=G, iota=iota, sigma=sigma, X1c=X1c, Y1s=Y1s, Y1c=Y1c, resid=resid,
                    elong_terms=(X1c, Y1s, Y1c), Dv=Dv, abs_G0=abs_G0, iotaN=iotaN, etabar=etabar,
                    dX1c=dX1c, dY1s=dY1s, dY1c=dY1c, B0=B0)


def elongation(X1c, Y1s, Y1c):
    p = X1c**2 + Y1s**2 + Y1c**2
    q = -X1c * Y1s
    return (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))
