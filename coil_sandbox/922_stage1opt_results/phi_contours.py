"""Proxy modular coils from the current potential on a flux surface.

The sheet current that reproduces the equilibrium field inside the boundary and zero
outside is K = (1/mu0) n x B (DESC's `K_vc`). Because J lies in the flux surface,
grad_S . K = -n . J = 0 exactly, so K has a stream function Phi:

    sqrt(g) K^theta = -dPhi/dzeta ,    sqrt(g) K^zeta = +dPhi/dtheta

with sqrt(g) = |e_theta x e_zeta|. Phi is multivalued,

    Phi = (G zeta + I theta) / 2pi + Phi~(theta, zeta)

with G the net poloidal current (what the modular coils must supply) and I the net
toroidal current (the plasma's own). Level sets of Phi do not close poloidally when
I != 0 -- you cannot replace a current-carrying plasma with modular coils -- so the
proxy contours

    Phi_mod = Phi - I theta / 2pi

Contours taken at equal increments of Phi carry equal current, and since
|grad_S Phi| = |K| their perpendicular spacing is dPhi/|K|: the curves crowd where the
current is large, which is the importance weighting we want, for free.

Known bias: the sheet carries the plasma's toroidal current while real coils do not, so
the proxy is contaminated at O(I/G) (0.019 on Helios). `t0_checks.py` measures it.

Every contour advances 2pi in theta and returns in zeta, which is exactly a
`FourierRZSurfaceCurve` with secular_theta=1, secular_zeta=0 (DESC PR #2285).
"""

import numpy as np

from desc.coils import FourierRZSurfaceCoil
from desc.geometry import FourierRZSurfaceCurve
from desc.grid import LinearGrid

MU_0 = 4e-7 * np.pi


class CurrentPotential:
    """Phi on the rho=1 surface, sampled on a uniform (theta, zeta) torus grid."""

    def __init__(self, theta, zeta, Phi_tilde_mn, G, I, sqrtg_Ktheta, sqrtg_Kzeta,
                 absK, residual, source):
        self.theta, self.zeta = theta, zeta          # 1-D node positions
        self.Phi_tilde_mn = Phi_tilde_mn             # complex FFT coeffs, [Mtheta, Nzeta]
        self.G, self.I = float(G), float(I)
        self.sqrtg_Ktheta = sqrtg_Ktheta             # = -dPhi/dzeta   (sign test lives here)
        self.sqrtg_Kzeta = sqrtg_Kzeta               # = +dPhi/dtheta
        self.absK = absK
        # Least-squares consistency of the stream function. This is NOT discretization
        # error (it is resolution-independent); it measures the equilibrium's own
        # J . n at rho = 1, since grad_S . K = -n . J. On helios_repro it is 1.1e-3
        # against an rms sqrt(g)|n.J|/scale of 2.3e-3. Treat it as an equilibrium-quality
        # diagnostic: a large value means the proxy's premise is shaky for that input.
        self.residual = float(residual)
        self.source = source

    # -- diagnostics ------------------------------------------------------------
    @property
    def monotonic(self):
        """Phi_mod strictly monotone in zeta <=> sqrt(g) K^theta never changes sign.

        This is exactly the condition for every contour to be a single-valued zeta(theta),
        i.e. a genuine modular curve rather than a saddle/windowpane."""
        s = self.sqrtg_Ktheta
        return bool(np.all(s > 0) or np.all(s < 0))

    def summary(self):
        s = self.sqrtg_Ktheta
        return dict(G=self.G, I=self.I, I_over_G=self.I / self.G,
                    residual=self.residual, monotonic=self.monotonic,
                    dPhi_dzeta_min=float(-s.max()), dPhi_dzeta_max=float(-s.min()),
                    absK_min=float(self.absK.min()), absK_max=float(self.absK.max()),
                    absK_ratio=float(self.absK.max() / self.absK.min()),
                    Phi_tilde_rms=float(np.sqrt(np.mean(self.eval_tilde() ** 2))))

    # -- evaluation -------------------------------------------------------------
    def eval_tilde(self, n_theta=None, n_zeta=None):
        """Phi~ on a (possibly finer) uniform grid, by zero-padded inverse FFT."""
        M, N = self.Phi_tilde_mn.shape
        n_theta, n_zeta = n_theta or M, n_zeta or N
        if (n_theta, n_zeta) == (M, N):
            return np.real(np.fft.ifft2(self.Phi_tilde_mn))
        pad = np.zeros((n_theta, n_zeta), complex)
        m = np.fft.fftfreq(M, 1 / M).astype(int)
        n = np.fft.fftfreq(N, 1 / N).astype(int)
        # place each coefficient at its own frequency in the larger array
        pad[np.ix_(m % n_theta, n % n_zeta)] = self.Phi_tilde_mn
        return np.real(np.fft.ifft2(pad)) * (n_theta * n_zeta) / (M * N)

    def Phi_mod(self, n_theta=None, n_zeta=None):
        """Phi with the toroidal-current secular term stripped, on a uniform grid."""
        pt = self.eval_tilde(n_theta, n_zeta)
        nz = pt.shape[1]
        z = 2 * np.pi * np.arange(nz) / nz
        return pt + self.G * z[None, :] / (2 * np.pi)


def current_potential(eq, M=128, N=256, vacuum_field=None):
    """Build Phi on the rho=1 surface of `eq`.

    M, N are theta and zeta samples over the FULL torus, so Fourier frequencies are
    plain integers and no field-period bookkeeping is needed.
    """
    theta = 2 * np.pi * np.arange(M) / M
    zeta = 2 * np.pi * np.arange(N) / N
    grid = LinearGrid(rho=np.array([1.0]), theta=theta, zeta=zeta, NFP=1, sym=False)
    keys = ["K_vc", "e_theta", "e_zeta", "|e_theta x e_zeta|"]
    d = eq.compute(keys, grid=grid)

    def g(k):
        return np.asarray(d[k])

    K = g("K_vc") if vacuum_field is None else vacuum_field
    et, ez, sg = g("e_theta"), g("e_zeta"), g("|e_theta x e_zeta|")
    # contravariant components from the 2x2 surface metric
    gtt = np.einsum("ij,ij->i", et, et)
    gtz = np.einsum("ij,ij->i", et, ez)
    gzz = np.einsum("ij,ij->i", ez, ez)
    bt = np.einsum("ij,ij->i", K, et)
    bz = np.einsum("ij,ij->i", K, ez)
    det = gtt * gzz - gtz**2
    Kt = (gzz * bt - gtz * bz) / det
    Kz = (gtt * bz - gtz * bt) / det

    # rebuild the (theta, zeta) map from the node coordinates, so nothing depends on
    # LinearGrid's internal ordering
    shape = (M, N)
    nt = np.rint(np.asarray(grid.nodes)[:, 1] / (2 * np.pi) * M).astype(int) % M
    nz = np.rint(np.asarray(grid.nodes)[:, 2] / (2 * np.pi) * N).astype(int) % N

    def to2d(v):
        out = np.full(shape, np.nan)
        out[nt, nz] = v
        assert not np.isnan(out).any(), "grid nodes do not tile the (theta, zeta) torus"
        return out

    sgKt = to2d(sg * Kt)       # = -dPhi/dzeta
    sgKz = to2d(sg * Kz)       # = +dPhi/dtheta
    absK = to2d(np.linalg.norm(K, axis=1))

    Phi_t = sgKz
    Phi_z = -sgKt
    I = 2 * np.pi * Phi_t.mean()
    G = 2 * np.pi * Phi_z.mean()

    # least-squares Poisson solve for the single-valued part
    a_t = Phi_t - I / (2 * np.pi)
    a_z = Phi_z - G / (2 * np.pi)
    At, Az = np.fft.fft2(a_t), np.fft.fft2(a_z)
    m = np.fft.fftfreq(M, 1 / M).astype(int)[:, None]
    n = np.fft.fftfreq(N, 1 / N).astype(int)[None, :]
    den = m**2 + n**2
    den[0, 0] = 1
    Phi_mn = -1j * (m * At + n * Az) / den
    Phi_mn[0, 0] = 0.0

    # consistency: does one scalar really reproduce BOTH derivative components?
    rt = np.real(np.fft.ifft2(1j * m * Phi_mn)) - a_t
    rz = np.real(np.fft.ifft2(1j * n * Phi_mn)) - a_z
    # normalise on the FULL derivative scale, not the fluctuating part: in the pure-TF
    # limit a_t and a_z are identically zero and a fluctuation-relative residual would be
    # round-off divided by round-off.
    scale = max(np.abs(Phi_t).max(), np.abs(Phi_z).max(), 1e-300)
    residual = max(np.abs(rt).max(), np.abs(rz).max()) / scale

    return CurrentPotential(theta, zeta, Phi_mn, G, I, sgKt, sgKz, absK, residual,
                            source=getattr(eq, "name", "") or repr(type(eq).__name__))


def modular_contours(pot, n_coils, phase=0.0, n_theta=256, n_zeta_fine=1024):
    """Contours of Phi_mod at `n_coils` equal increments of G -> equal current per coil.

    Returns a list of (theta, zeta) arrays, each of length n_theta, with theta uniform on
    [0, 2pi). `phase` in [0, 1) slides the whole ladder by that fraction of a gap.
    """
    if not pot.monotonic:
        raise ValueError(
            "Phi_mod is not monotone in zeta: contours are not single-valued zeta(theta), "
            "so this equilibrium is outside the modular-proxy class. Inspect "
            "pot.summary()['dPhi_dzeta_min/max']."
        )
    F = pot.Phi_mod(n_theta, n_zeta_fine)            # [n_theta, n_zeta_fine]
    zf = 2 * np.pi * np.arange(n_zeta_fine) / n_zeta_fine
    G = pot.G
    # extend by one period so every level is bracketed for every theta
    Fx = np.concatenate([F, F[:, :1] + G], axis=1)
    zx = np.concatenate([zf, [2 * np.pi]])
    if G < 0:                                        # make the table increasing
        Fx, G_ = -Fx, -G
    else:
        G_ = G
    out = []
    for k in range(n_coils):
        c = (k + phase) * G_ / n_coils
        zeta = np.empty(n_theta)
        for j in range(n_theta):
            f = Fx[j]
            cj = f[0] + np.mod(c - f[0], G_)         # unique level in this theta's window
            zeta[j] = np.interp(cj, f, zx)
        out.append((2 * np.pi * np.arange(n_theta) / n_theta, np.mod(zeta, 2 * np.pi)))
    return out


def to_surface_curve(theta, zeta, eq=None, surface=None, N_fourier=32, name=""):
    """Fit (theta(s), zeta(s)) with theta(s) = s to a FourierRZSurfaceCurve.

    The contour is modular: theta advances by 2pi and zeta returns, i.e.
    secular_theta = 1, secular_zeta = 0.
    """
    assert (eq is None) != (surface is None), "pass exactly one of eq, surface"
    z = np.unwrap(zeta)
    z = z - np.round((z[-1] - z[0]) / (2 * np.pi)) * 2 * np.pi * np.arange(len(z)) / len(z)
    Z = np.fft.rfft(z) / len(z)
    modes, coefs = [0], [np.real(Z[0])]
    for n in range(1, N_fourier + 1):
        modes += [n, -n]
        coefs += [2 * np.real(Z[n]), -2 * np.imag(Z[n])]
    kw = dict(equilibrium=eq) if eq is not None else dict(surface=surface)
    return FourierRZSurfaceCurve(
        secular_theta=1, secular_zeta=0,
        theta_n=[0.0], modes_theta=[0],
        zeta_n=np.array(coefs), modes_zeta=np.array(modes),
        NFP=1, name=name, **kw)


def proxy_coilset(pot, eq, n_coils, phase=0.0, N_fourier=32, n_theta=256):
    """The full proxy: n_coils equal-current contours as FourierRZSurfaceCoil objects."""
    cur = pot.G / n_coils
    coils = []
    for k, (th, ze) in enumerate(modular_contours(pot, n_coils, phase, n_theta)):
        c = to_surface_curve(th, ze, eq=eq, N_fourier=N_fourier, name=f"proxy{k}")
        coils.append(FourierRZSurfaceCoil(
            current=cur, equilibrium=eq,
            secular_theta=1, secular_zeta=0,
            theta_n=c.theta_n, modes_theta=c.theta_basis.modes[:, 2],
            zeta_n=c.zeta_n, modes_zeta=c.zeta_basis.modes[:, 2],
            NFP=1, name=f"proxy{k}"))
    return coils
