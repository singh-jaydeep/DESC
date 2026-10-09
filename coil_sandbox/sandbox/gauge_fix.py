"""`LambdaCondensation`: a soft penalty that pins DESC's poloidal-angle (theta) gauge.

DESC leaves theta free: relabeling theta -> theta + omega(rho, theta, zeta) leaves force balance unchanged
(up to truncation), and moves lambda by ~ -omega. The equilibrium re-solve therefore drifts along a very
soft mode, and gauge-dependent objectives (QS two-term on the computational measure; iota through its
quadrature) jump between re-solves (work/ss_qh/joint/stall/DIAGNOSIS.md).

Rows: sqrt(weight) * (1 + m^2 + n^2)^(p/2) * L_lmn. Any positive-definite norm of lambda is strictly convex
along the gauge orbit (d lambda = -omega), so a small weight makes the minimizer unique without visibly
changing force balance. p > 0 penalizes spectral width (smooth lambda) rather than lambda itself, which
pulls less hard toward straight-field-line theta (lambda = 0) than p = 0.

ref=True anchors the penalty at the lambda of the equilibrium at build: rows w (L_lmn - L_lmn,ref). Its
gradient is then zero at the anchor, so it stiffens the gauge modes without pulling the solution toward
lambda = 0 (no distortion at the anchor; physical modes with singular value s >> weight move by ~(weight/s)^2).

It is linear in L_lmn but must NOT be classified as a linear constraint (that would force lambda = 0,
i.e. FixThetaSFL), so `_linear = False`: in a proximal optimization it joins ForceBalance in the
equilibrium sub-problem.
"""

import numpy as np

from desc.backend import jnp
from desc.objectives.objective_funs import _Objective


class LambdaCondensation(_Objective):
    """Spectrally weighted L2 norm of the lambda coefficients (theta-gauge pin)."""

    _units = "(rad)"
    _print_value_fmt = "lambda condensation: "
    _coordinates = ""
    _linear = False
    _equilibrium = True  # admitted by ProximalProjection into the equilibrium sub-problem
    _scalar = False

    def __init__(self, eq, weight=1e-4, power=1.0, ref=False, name="lambda condensation"):
        self._power = power
        self._ref = bool(ref)
        super().__init__(things=eq, target=0, weight=weight, normalize=False, name=name)

    def build(self, use_jit=True, verbose=1):
        eq = self.things[0]
        modes = np.asarray(eq.L_basis.modes)
        w = (1.0 + modes[:, 1] ** 2 + modes[:, 2] ** 2) ** (self._power / 2)
        self._dim_f = int(modes.shape[0])
        L0 = jnp.asarray(eq.L_lmn) if self._ref else jnp.zeros(self._dim_f)
        self._constants = {"w": jnp.asarray(w), "L0": L0, "quad_weights": 1.0}
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        constants = self._get_deprecated_constants(constants)
        return (params["L_lmn"] - constants["L0"]) * constants["w"]


class TangentialPin(_Objective):
    """Penalize the poloidal relabeling relative to an anchor: rows (dR R_t + dZ Z_t) / |X_t|^2.

    At fixed computational (rho, theta, zeta) the displacement from the anchor, dX = (dR, dZ) in the
    R-Z plane (zeta = phi is fixed in DESC), splits into a part along the anchor's X_theta and a part normal to
    the cross-section. A relabeling theta -> theta + omega moves points by dX = -omega X_theta, so the rows are
    exactly -omega (radians). Any change of the flux-surface SHAPES can be written as a normal displacement, and
    lambda is left free, so this pins only the gauge (to first order), unlike LambdaCondensation, which also
    resists lambda's physical (straight-field-line) response and admits non-equilibria.

    Rows are divided by sqrt(number of nodes) (rms over interior surfaces rho in (0, 1); rho = 1 is fixed by
    the boundary anyway). Anchored at the equilibrium's state at build.
    """

    _units = "(rad)"
    _print_value_fmt = "tangential pin: "
    _coordinates = ""
    _linear = False
    _equilibrium = True  # admitted by ProximalProjection into the equilibrium sub-problem
    _scalar = False

    def __init__(self, eq, weight=1e-2, n_rho=8, name="tangential pin"):
        self._n_rho = n_rho
        super().__init__(things=eq, target=0, weight=weight, normalize=False, name=name)

    def build(self, use_jit=True, verbose=1):
        from desc.compute import get_profiles, get_transforms
        from desc.grid import LinearGrid

        eq = self.things[0]
        grid = LinearGrid(rho=np.linspace(0.1, 0.95, self._n_rho), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP,
                          sym=False)
        keys = ["R", "Z"]
        d = eq.compute(["R", "Z", "R_t", "Z_t"], grid=grid)
        Rt, Zt = np.asarray(d["R_t"]), np.asarray(d["Z_t"])
        n2 = Rt**2 + Zt**2
        self._dim_f = int(grid.num_nodes)
        s = 1 / np.sqrt(grid.num_nodes)
        self._constants = {
            "transforms": get_transforms(keys, obj=eq, grid=grid),
            "profiles": get_profiles(keys, obj=eq, grid=grid),
            "R0": jnp.asarray(d["R"]), "Z0": jnp.asarray(d["Z"]),
            "tR": jnp.asarray(s * Rt / n2), "tZ": jnp.asarray(s * Zt / n2),
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        from desc.compute.utils import _compute as compute_fun

        constants = self._get_deprecated_constants(constants)
        data = compute_fun("desc.equilibrium.equilibrium.Equilibrium", ["R", "Z"], params=params,
                           transforms=constants["transforms"], profiles=constants["profiles"])
        return (data["R"] - constants["R0"]) * constants["tR"] + (data["Z"] - constants["Z0"]) * constants["tZ"]


class GaugeCondition(_Objective):
    """Equal-arclength poloidal angle as a gauge condition (spectral condensation with width ∮|X_theta|^2).

    A relabeling theta -> theta + omega moves the representation along a gauge orbit
    (dR = -omega R_t, dZ = -omega Z_t, dlambda ~ -omega (1 + lambda_t)) without changing the physics. A gauge
    condition must (1) be reachable from ANY configuration by a relabeling alone, so no physical state is excluded,
    and (2) fix the orbit point uniquely. The width W = ∮ (R_t^2 + Z_t^2) dtheta has first variation
    dW = 2 ∮ omega (R_t R_tt + Z_t Z_tt) dtheta, so stationarity along the orbit is d/dtheta |X_t|^2 = 0
    (equal arclength), which every cross-section admits (1) and which leaves only a theta shift c(rho, zeta); the
    shift moves lambda's m = 0 part by -c, so lambda_{m=0} = 0 fixes it (2).

    Discrete form, one row per gauge degree of freedom (the lambda basis, which is the space of relabelings):
      m != 0 modes phi_k:  <phi_k, (R_t R_tt + Z_t Z_tt)> / a^2   (Galerkin, quadrature grid)
      m == 0 modes:        L_lmn
    Unlike LambdaCondensation/TangentialPin (anchored penalties that also resist physical changes and admitted
    non-equilibria), this is satisfiable by relabeling alone, so it can be enforced hard without pulling physics.
    """

    _units = "(~)"
    _print_value_fmt = "gauge condition: "
    _coordinates = ""
    _linear = False
    _equilibrium = True  # admitted by ProximalProjection into the equilibrium sub-problem
    _scalar = False

    def __init__(self, eq, weight=1.0, name="gauge condition"):
        super().__init__(things=eq, target=0, weight=weight, normalize=False, name=name)

    def build(self, use_jit=True, verbose=1):
        from desc.compute import get_profiles, get_transforms
        from desc.grid import QuadratureGrid

        eq = self.things[0]
        grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        keys = ["R_t", "Z_t", "R_tt", "Z_tt"]
        modes = np.asarray(eq.L_basis.modes)
        m_nz = modes[:, 1] != 0
        Phi = np.asarray(eq.L_basis.evaluate(grid))[:, m_nz]  # nodes x gauge modes
        w = np.asarray(grid.weights)
        a = float(eq.compute("a")["a"])
        PW = (Phi * w[:, None]).T / (np.sum(w) * a**2)
        PW = PW / np.sqrt(np.mean(np.sum(Phi**2 * w[:, None], axis=0) / np.sum(w)))  # rows ~ rms-normalized
        self._dim_f = int(m_nz.sum() + (~m_nz).sum())
        self._constants = {
            "transforms": get_transforms(keys, obj=eq, grid=grid),
            "profiles": get_profiles(keys, obj=eq, grid=grid),
            "PW": jnp.asarray(PW), "m0": jnp.asarray(np.where(~m_nz)[0]), "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        from desc.compute.utils import _compute as compute_fun

        constants = self._get_deprecated_constants(constants)
        d = compute_fun("desc.equilibrium.equilibrium.Equilibrium", ["R_t", "Z_t", "R_tt", "Z_tt"], params=params,
                        transforms=constants["transforms"], profiles=constants["profiles"])
        h = d["R_t"] * d["R_tt"] + d["Z_t"] * d["Z_tt"]
        return jnp.concatenate([constants["PW"] @ h, params["L_lmn"][constants["m0"]]])
