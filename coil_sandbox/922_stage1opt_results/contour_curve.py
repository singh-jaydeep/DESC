"""Proxy contours computed INSIDE the objective, so they are streamlines by construction.

The v0 driver held each curve's (theta, zeta) fixed, which lets the optimizer improve C on a
curve that is no longer a streamline. Constraining `x_s . B = 0` would fix that but needs a
two-thing objective and an AL equality whose violation contaminates everything upstream.

Instead: given the equilibrium parameters, build the contours here.

    K_vc, metric  ->  Phi_theta = sqrt(g) K^zeta,  Phi_zeta = -sqrt(g) K^theta
                  ->  G, I and the single-valued Phi~ by FFT
                  ->  zeta_k(theta) by Newton on Phi_mod = const   (monotone in zeta)
                  ->  x(theta, zeta) exactly from the surface Fourier basis
                  ->  x_s along the contour from dzeta/dtheta = -Phi_theta / Phi_zeta
                  ->  eps_1, eps_2, C

Everything is jnp and differentiable. The curves are contours of Phi at equal increments, so
they are exact streamlines AND equal-current by construction -- which also restores the
current weighting that NOTES section 3 got for free and section 5e had to patch with pins.
"""

import numpy as np

import jax
import jax.numpy as jnp


def _fourier_eval(coef, modes, theta, zeta, NFP):
    """Evaluate a DESC double-Fourier series at arbitrary (theta, zeta).

    modes are (l, m, n) rows; positive m -> cos(m theta), negative m -> sin(|m| theta),
    and likewise n with the NFP*zeta argument, matching DoubleFourierSeries.
    """
    m = modes[:, 1][None, :]
    n = modes[:, 2][None, :]
    th = theta[:, None]
    ze = zeta[:, None]
    tt = jnp.where(m >= 0, jnp.cos(jnp.abs(m) * th), jnp.sin(jnp.abs(m) * th))
    zz = jnp.where(n >= 0, jnp.cos(jnp.abs(n) * NFP * ze), jnp.sin(jnp.abs(n) * NFP * ze))
    return (tt * zz) @ coef


def _fourier_eval_dtheta(coef, modes, theta, zeta, NFP):
    m = modes[:, 1][None, :]
    n = modes[:, 2][None, :]
    th = theta[:, None]
    ze = zeta[:, None]
    am = jnp.abs(m)
    dtt = jnp.where(m >= 0, -am * jnp.sin(am * th), am * jnp.cos(am * th))
    zz = jnp.where(n >= 0, jnp.cos(jnp.abs(n) * NFP * ze), jnp.sin(jnp.abs(n) * NFP * ze))
    return (dtt * zz) @ coef


def _fourier_eval_dzeta(coef, modes, theta, zeta, NFP):
    m = modes[:, 1][None, :]
    n = modes[:, 2][None, :]
    th = theta[:, None]
    ze = zeta[:, None]
    an = jnp.abs(n)
    tt = jnp.where(m >= 0, jnp.cos(jnp.abs(m) * th), jnp.sin(jnp.abs(m) * th))
    dzz = jnp.where(n >= 0, -an * NFP * jnp.sin(an * NFP * ze),
                    an * NFP * jnp.cos(an * NFP * ze))
    return (tt * dzz) @ coef


def potential_from_grid(K_vc, e_theta, e_zeta, sqrtg, M, N):
    """Phi_theta, Phi_zeta, G, I and Phi~ coefficients from data on an M x N torus grid.

    Inputs are flat arrays ordered so that reshape(M, N) gives [theta, zeta].
    """
    gtt = jnp.sum(e_theta * e_theta, axis=1)
    gtz = jnp.sum(e_theta * e_zeta, axis=1)
    gzz = jnp.sum(e_zeta * e_zeta, axis=1)
    bt = jnp.sum(K_vc * e_theta, axis=1)
    bz = jnp.sum(K_vc * e_zeta, axis=1)
    det = gtt * gzz - gtz**2
    Kt = (gzz * bt - gtz * bz) / det
    Kz = (gtt * bz - gtz * bt) / det
    Phi_t = (sqrtg * Kz).reshape(M, N)
    Phi_z = (-sqrtg * Kt).reshape(M, N)
    I = 2 * jnp.pi * jnp.mean(Phi_t)
    G = 2 * jnp.pi * jnp.mean(Phi_z)
    At = jnp.fft.fft2(Phi_t - I / (2 * jnp.pi))
    Az = jnp.fft.fft2(Phi_z - G / (2 * jnp.pi))
    mm = jnp.fft.fftfreq(M, 1 / M)[:, None]
    nn = jnp.fft.fftfreq(N, 1 / N)[None, :]
    den = mm**2 + nn**2
    den = den.at[0, 0].set(1.0)
    Phi_mn = -1j * (mm * At + nn * Az) / den
    Phi_mn = Phi_mn.at[0, 0].set(0.0)
    return Phi_t, Phi_z, G, I, Phi_mn, mm, nn


def truncation_index(M, N, Mt, Nt):
    """STATIC mode selection for Phi~: |m| <= Mt, |n| <= Nt. Computed once at build.

    The dense (n_nodes, M, N) intermediate is the memory hog: at n=240, M=48, N=96 a
    FORWARD-mode Jacobian over 1462 equilibrium DOF tried to allocate **207 GB**.
    Truncating, and differentiating in reverse mode, both attack that. This must be
    numpy at build time -- doing it inside the traced chain raises
    TracerArrayConversionError, because mm/nn are traced there.
    """
    mm = np.fft.fftfreq(M, 1 / M)
    nn = np.fft.fftfreq(N, 1 / N)
    ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    sel = (np.abs(mm)[:, None] <= Mt) & (np.abs(nn)[None, :] <= Nt)
    ti, tj = ii[sel], jj[sel]
    return (jnp.asarray(ti), jnp.asarray(tj),
            jnp.asarray(mm[ti]), jnp.asarray(nn[tj]))


def truncate(Phi_mn, trunc, M, N):
    ti, tj, mf, nf = trunc
    return Phi_mn[ti, tj] / (M * N), mf, nf


def _tilde_and_dz(cf, mf, nf, theta, zeta):
    """Phi~ and dPhi~/dzeta at arbitrary (theta, zeta), from truncated coefficients."""
    ph = jnp.exp(1j * (mf[None, :] * theta[:, None] + nf[None, :] * zeta[:, None]))
    val = jnp.real(ph @ cf)
    dz = jnp.real(ph @ (1j * nf * cf))
    return val, dz


def tilde_derivs(cf, mf, nf, theta, zeta):
    """Phi~, dPhi~/dtheta, dPhi~/dzeta at arbitrary (theta, zeta)."""
    ph = jnp.exp(1j * (mf[None, :] * theta[:, None] + nf[None, :] * zeta[:, None]))
    return (jnp.real(ph @ cf), jnp.real(ph @ (1j * mf * cf)), jnp.real(ph @ (1j * nf * cf)))


def make_Phi_derivs(cf, mf, nf, G, I):
    """Closure giving (Phi_theta, Phi_zeta) at arbitrary (theta, zeta)."""
    def fn(theta, zeta):
        _, dt, dz = tilde_derivs(cf, mf, nf, theta, zeta)
        return I / (2 * jnp.pi) + dt, G / (2 * jnp.pi) + dz
    return fn


def contour_zeta(cf, mf, nf, G, theta, level, newton=6):
    """Solve Phi_mod(theta, zeta) = level for zeta. Phi_mod = G zeta/2pi + Phi~.

    Phi_mod is strictly monotone in zeta whenever sqrt(g) K^theta keeps one sign, which is
    the modular-proxy admissibility condition, so Newton from the linear guess converges.
    """
    zeta = 2 * jnp.pi * level / G * jnp.ones_like(theta)
    for _ in range(newton):
        t, dz = _tilde_and_dz(cf, mf, nf, theta, zeta)
        f = G * zeta / (2 * jnp.pi) + t - level
        df = G / (2 * jnp.pi) + dz
        zeta = zeta - f / df
    return zeta


def contour_residual(cf, mf, nf, G, theta, zeta, level):
    """|Phi_mod(theta, zeta) - level| / |G|: did Newton actually converge?"""
    t, _ = _tilde_and_dz(cf, mf, nf, theta, zeta)
    return jnp.max(jnp.abs(G * zeta / (2 * jnp.pi) + t - level)) / jnp.abs(G)


def contour_points(Rc, Rm, Zc, Zm, NFP, Phi_t_fn, theta, zeta):
    """x and dx/dtheta along a contour, in xyz.

    Along Phi_mod = const, dzeta/dtheta = -Phi_theta / Phi_zeta, so the tangent is exact.
    """
    R = _fourier_eval(Rc, Rm, theta, zeta, NFP)
    Z = _fourier_eval(Zc, Zm, theta, zeta, NFP)
    R_t = _fourier_eval_dtheta(Rc, Rm, theta, zeta, NFP)
    R_z = _fourier_eval_dzeta(Rc, Rm, theta, zeta, NFP)
    Z_t = _fourier_eval_dtheta(Zc, Zm, theta, zeta, NFP)
    Z_z = _fourier_eval_dzeta(Zc, Zm, theta, zeta, NFP)
    Pt, Pz = Phi_t_fn(theta, zeta)
    dzdt = -Pt / Pz
    dR = R_t + R_z * dzdt
    dZ = Z_t + Z_z * dzdt
    phi = zeta
    dphi = dzdt
    x = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=1)
    dx = jnp.stack([dR * jnp.cos(phi) - R * jnp.sin(phi) * dphi,
                    dR * jnp.sin(phi) + R * jnp.cos(phi) * dphi,
                    dZ], axis=1)
    return x, dx


# ---------------------------------------------------------------------------
# the DESC objective
# ---------------------------------------------------------------------------
from desc.compute import get_profiles, get_transforms  # noqa: E402
from desc.compute.utils import _compute as compute_fun  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives.objective_funs import _Objective  # noqa: E402
from desc.utils import errorif  # noqa: E402

DATA_KEYS = ["K_vc", "e_theta", "e_zeta", "|e_theta x e_zeta|"]


def _curves_from_params(params, const, M, N, NFP, K, newton, phase, diagnostics=False):
    """The whole differentiable chain: eq params -> (x, dx/dtheta) for each contour.

    M, N, NFP, K, newton, phase are passed SEPARATELY and must be static Python ints /
    floats. Putting them in `const` makes jit trace them (the objective passes _constants
    as a jit argument) and reshape then fails on a traced shape -- the same trap
    CLAUDE.md records for the spline `intervals` fix.
    """
    data = compute_fun("desc.equilibrium.equilibrium.Equilibrium", DATA_KEYS,
                       params=params, transforms=const["transforms"],
                       profiles=const["profiles"])
    perm = const["perm"]
    Pt, Pz, G, I, Pmn, mm, nn = potential_from_grid(
        data["K_vc"][perm], data["e_theta"][perm], data["e_zeta"][perm],
        data["|e_theta x e_zeta|"][perm], M, N)
    cf, mf, nf = truncate(Pmn, const["trunc"], M, N)
    fn = make_Phi_derivs(cf, mf, nf, G, I)
    th = const["theta"]
    levels = (jnp.arange(K) + phase) * G / (K * NFP)

    # vmap over contours. A Python `for k in range(K)` UNROLLS the traced graph K times,
    # each holding `newton` inlined Newton steps over n nodes: at K=16, n=480 that is ~96
    # inlined solves and XLA spent >7 minutes compiling before the run even started.
    def _one(level):
        ze = contour_zeta(cf, mf, nf, G, th, level, newton=newton)
        x, dx = contour_points(params["Rb_lmn"], const["Rm"], params["Zb_lmn"], const["Zm"],
                               NFP, fn, th, ze)
        return x, jnp.linalg.norm(dx, axis=1), ze

    xs, ws, zes = jax.vmap(_one)(levels)
    out = (xs, ws)
    diag = {}
    if diagnostics:
        diag["newton_residual"] = [
            float(contour_residual(cf, mf, nf, G, th, zes[k], levels[k])) for k in range(K)]
    if diagnostics:
        # Phi_mod is monotone in zeta iff sqrt(g) K^theta keeps one sign; that is exactly
        # the condition for every contour to be a single-valued zeta(theta)
        diag["dPhi_dzeta_min"] = float(jnp.min(Pz))
        diag["dPhi_dzeta_max"] = float(jnp.max(Pz))
        diag["monotonic"] = bool(diag["dPhi_dzeta_min"] * diag["dPhi_dzeta_max"] > 0)
        diag["G"], diag["I"] = float(G), float(I)
        tot = float(jnp.sum(jnp.abs(Pmn)))
        kept = float(jnp.sum(jnp.abs(cf))) * (M * N)
        diag["mode_truncation_kept_fraction"] = kept / max(tot, 1e-300)
        return out, diag
    return out


def _eps_one(x, w, masks, nrms, n0):
    """eps for ONE curve from stacked frozen arrays. vmap this over curves."""
    def seg(m, nrm):
        ww = w * m
        Wt = jnp.sum(ww)
        D = x - jnp.sum(ww[:, None] * x, axis=0) / Wt
        return jnp.sum(ww * (D @ nrm) ** 2)

    tot = jnp.sum(jax.vmap(seg)(masks, nrms))
    Wa = jnp.sum(w)
    D = x - jnp.sum(w[:, None] * x, axis=0) / Wa
    Mx = (D * w[:, None]).T @ D / Wa
    l0 = n0 @ Mx @ n0
    tr = jnp.trace(Mx)
    l1l2 = 0.5 * (tr**2 - jnp.trace(Mx @ Mx)) - l0 * (tr - l0)
    return jnp.sqrt(tot / Wa) / jnp.sqrt(jnp.sqrt(jnp.clip(l1l2, 1e-30)))


_eps_all = jax.vmap(_eps_one, in_axes=(0, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# eps_1 with a LIVE best-fit plane (no frozen normal)
# ---------------------------------------------------------------------------
# The frozen normal exists because `eigvalsh` has a meaningless gradient on a DEGENERATE
# eigenspace -- but that argument is about lambda_1 vs lambda_2, the IN-PLANE pair, which
# is exactly equal for a near-circular curve. It does NOT apply to lambda_0: by
# construction lambda_0/lambda_1 ~ eps_1^2, measured 0.0074..0.0159 on precise_QA, so
# lambda_0 is simple by a factor of 60-130 and is a smooth function of the points.
#
# Freezing it is not free. eps_1 is invariant under RIGIDLY ROTATING a curve -- same curve,
# same planarity -- but the frozen surrogate is not, and it prices tilt as non-planarity:
# measured on a baseline contour, a rigid tilt of 2 deg inflates it 1.08x, 4 deg 1.28x,
# 8 deg 1.89x, against a whole-ladder signal of 0.60x. So the optimizer cannot reach a
# planar basin whose plane is oriented differently, which is a restriction on the search
# that nothing in the physics justifies.
#
# The derivative below is first-order perturbation theory, d lambda_0 / dM = v0 v0^T,
# which involves NO division by an eigenvalue difference and so is safe even when the
# in-plane pair is exactly degenerate. It is the same expression the frozen path computes
# -- with v0 refreshed at every evaluation instead of held from build time. Cost: one
# symmetric 3x3 eigendecomposition per contour.


@jax.custom_jvp
def _lam_min(Mx):
    """Smallest eigenvalue of a symmetric 3x3."""
    return jnp.linalg.eigvalsh(Mx)[0]


@_lam_min.defjvp
def _lam_min_jvp(primals, tangents):
    (Mx,), (dMx,) = primals, tangents
    lam, vec = jnp.linalg.eigh(Mx)
    v = vec[:, 0]
    return lam[0], v @ dMx @ v


def _eps1_live(x, w):
    """eps_1 = sqrt(lambda_0) / (lambda_1 lambda_2)^(1/4), all live.

    The denominator still goes through the matrix invariants rather than through
    individual eigenvalues, so lambda_1 and lambda_2 are never separated and the
    in-plane degeneracy is still never differentiated.
    """
    W = jnp.sum(w)
    D = x - jnp.sum(w[:, None] * x, axis=0) / W
    Mx = (D * w[:, None]).T @ D / W
    lam0 = _lam_min(Mx)
    tr = jnp.trace(Mx)
    l1l2 = 0.5 * (tr**2 - jnp.trace(Mx @ Mx)) - lam0 * (tr - lam0)
    return jnp.sqrt(jnp.clip(lam0, 1e-30)) / jnp.sqrt(jnp.sqrt(jnp.clip(l1l2, 1e-30)))


_eps1_live_all = jax.vmap(_eps1_live, in_axes=(0, 0))


class ContourClamshell(_Objective):
    """Clamshell ratio C = eps_2 / (eps_1/4) on proxy coils built INSIDE the objective.

    The proxy coils are contours of the current potential on rho = 1, at equal increments
    of Phi. They are therefore exact streamlines AND equal-current by construction, so no
    streamline constraint and no pin are needed, and the optimizer cannot move a curve off
    its streamline to cheat the metric.

    quantity : "C" (the clamshell ratio, dim_f = K) or "eps1" (dim_f = K).
    Lower C is more clamshell-able; C = 1 is the generic B^-2 decay, C in [0, 4].

    The hinge segmentation and the best-fit plane normals are FROZEN at build time
    (Danskin). Rebuild the objective to refresh them. EXCEPTION: quantity="eps1" uses a
    LIVE best-fit plane by default (B = 1 has no segmentation to freeze, and lambda_0 is
    simple), so it is exact and rotation-invariant rather than an upper bound. Pass
    frozen_normal=True for the old behaviour.
    """

    _scalar = False
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Clamshell "
    _coordinates = ""
    _static_attrs = _Objective._static_attrs + ["_K", "_M", "_N", "_n", "_newton",
                                                "_phase", "_quantity", "_stride", "_NFP",
                                                "_Mt", "_Nt", "_frozen_normal"]

    def __init__(self, eq, K=4, M=64, N=128, n=240, newton=6, phase=0.0, stride=4,
                 Mt=16, Nt=16, deriv_mode="rev", frozen_normal=False,
                 quantity="C", target=None, bounds=None, weight=1, normalize=False,
                 normalize_target=False, loss_function=None,
                 name="clamshell", jac_chunk_size=None):
        errorif(quantity not in ("C", "eps1"), ValueError, "quantity must be C or eps1")
        self._K, self._M, self._N, self._n = K, M, N, n
        self._newton, self._phase, self._quantity, self._stride = newton, phase, quantity, stride
        self._Mt, self._Nt = Mt, Nt
        self._frozen_normal = bool(frozen_normal)
        if target is None and bounds is None:
            target = 0.0
        super().__init__(things=eq, target=target, bounds=bounds, weight=weight,
                         normalize=normalize, normalize_target=normalize_target,
                         loss_function=loss_function, deriv_mode=deriv_mode, name=name,
                         jac_chunk_size=jac_chunk_size)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays, and freeze the hinge segmentation."""
        from clamshell import best_breaks, freeze  # local: needs boot.setup first

        eq = self.things[0]
        M, N = self._M, self._N
        theta = 2 * np.pi * np.arange(M) / M
        zeta = 2 * np.pi * np.arange(N) / N
        grid = LinearGrid(rho=np.array([1.0]), theta=theta, zeta=zeta, NFP=1, sym=False)
        nodes = np.asarray(grid.nodes)
        nt = np.rint(nodes[:, 1] / (2 * np.pi) * M).astype(int) % M
        nz = np.rint(nodes[:, 2] / (2 * np.pi) * N).astype(int) % N
        perm = np.lexsort((nz, nt))
        errorif(not np.array_equal(np.sort(nt[perm] * N + nz[perm]), np.arange(M * N)),
                ValueError, "grid nodes do not tile the (theta, zeta) torus exactly")

        self._constants = {
            "transforms": get_transforms(DATA_KEYS, obj=eq, grid=grid),
            "profiles": get_profiles(DATA_KEYS, obj=eq, grid=grid),
            "perm": jnp.asarray(perm),
            "Rm": jnp.asarray(eq.surface.R_basis.modes),
            "Zm": jnp.asarray(eq.surface.Z_basis.modes),
            "theta": jnp.asarray(2 * np.pi * np.arange(self._n) / self._n),
            "trunc": truncation_index(M, N, self._Mt, self._Nt),
        }
        self._NFP = int(eq.NFP)
        # freeze the segmentation and plane normals at the current geometry
        (xs, ws), diag = self._chain(eq.params_dict, self._constants, diagnostics=True)
        errorif(not diag["monotonic"], ValueError,
                f"Phi_mod is not monotone in zeta (dPhi/dzeta spans "
                f"{diag['dPhi_dzeta_min']:.3e} .. {diag['dPhi_dzeta_max']:.3e}): the "
                f"contours are not single-valued zeta(theta), so this equilibrium is "
                f"outside the modular-proxy class.")
        errorif(max(diag["newton_residual"]) > 1e-10, ValueError,
                f"contour Newton did not converge (residual "
                f"{max(diag['newton_residual']):.2e}); raise `newton`.")
        fz1, fz2 = [], []
        for k in range(self._K):
            X, W = np.asarray(xs[k]), np.asarray(ws[k])
            fz1.append(freeze(X, W, best_breaks(X, W, 1, stride=self._stride)[1]))
            fz2.append(freeze(X, W, best_breaks(X, W, 2, stride=self._stride)[1]))
        for tag, fzs in (("fz1", fz1), ("fz2", fz2)):
            self._constants[tag] = tuple(
                jnp.asarray(np.stack([f[key] for f in fzs])) for key in
                ("masks", "normals", "n0"))
        self._gap_seg = min(float(f["gap_seg"]) for f in fz2)
        self._dim_f = self._K
        if verbose > 0:
            print(f"  {self.name}: K={self._K} contours, worst frozen segment "
                  f"lambda_1/lambda_2 = {self._gap_seg:.3f}")
        super().build(use_jit=use_jit, verbose=verbose)

    def _chain(self, params, constants, diagnostics=False):
        return _curves_from_params(params, constants, self._M, self._N, self._NFP,
                                   self._K, self._newton, self._phase, diagnostics)

    def diagnostics(self, eq=None):
        """Health check: monotonicity, Newton convergence, frozen-segment conditioning."""
        eq = eq or self.things[0]
        _, d = self._chain(eq.params_dict, self._constants, diagnostics=True)
        d["worst_newton_residual"] = max(d.pop("newton_residual"))
        d["worst_frozen_gap_seg"] = self._gap_seg
        return d

    def compute(self, params, constants=None):
        """Compute C (or eps_1) for each proxy contour."""
        if constants is None:
            constants = self.constants
        xs, ws = self._chain(params, constants)
        if self._quantity == "eps1" and not self._frozen_normal:
            return _eps1_live_all(xs, ws)
        e1 = _eps_all(xs, ws, *constants["fz1"])
        if self._quantity == "eps1":
            return e1
        return 4.0 * _eps_all(xs, ws, *constants["fz2"]) / e1
