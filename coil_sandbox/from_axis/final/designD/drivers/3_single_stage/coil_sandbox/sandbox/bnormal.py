"""`VacuumBoundaryBn`: quadratic flux on a MOVING boundary (the B.n rows of VacuumBoundaryError).

`QuadraticFlux` holds the equilibrium fixed, so it cannot be used when the boundary is
optimized. `VacuumBoundaryError(eq, field)` varies both, but its residual is two blocks:
sqrt(g) B_coil.n and sqrt(g) (B_in^2 - B_coil^2). Linearized about the equilibrium field, the
second is ~2 B.dB, i.e. ~2 dB_t / B normalized -- the TANGENTIAL mismatch, which is naturally
~2x the normal one and so carries ~4x the cost. MEASURED on precise_QA arcB2 (single stage,
k=1): the B^2 block was 2.7x the B.n block, and the optimizer cut it 25% while B.n rose 9%.

This keeps only the B.n block, and RE-WEIGHTS it: rows B_coil.n sqrt(g) / (<|B|> sqrt(mean g)) with
quadrature weights sqrt(d theta d zeta), so the cost is the AREA integral of (B.n / B)^2, as in
QuadraticFlux (which the stage-2 baseline used). VacuumBoundaryError's rows are B.n g, a g^2
weight that leans on the outboard side. MEASURED (precise_QA arcB2, k=1, B.n g rows): the g^2
measure fell 4% while the area-weighted rms rose 4% and the unweighted mean 12% -- the error
was moved to small-g (inboard) points. The normalization is <|B|> sqrt(mean g) on the eval grid at build (fixed thereafter), so the
rms of the rows is the area-weighted rms of B.n / <|B|> -- the tolerance applies to exactly that.
Eq and coils are both free. Vacuum only (no plasma field). It is blind
to the overall current scale (I -> 0 is a free minimizer), so hold the coil current sum fixed
(FixSumCoilCurrent), exactly as stage 2 does.
"""

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.objectives import VacuumBoundaryError
from desc.objectives.objective_funs import _Objective


class VacuumBoundaryBn(VacuumBoundaryError):
    """B_coil.n on the (moving) plasma boundary: the B.n block of VacuumBoundaryError."""

    _print_value_fmt = "Boundary normal field error: "

    def __init__(self, *args, name="Vacuum B.n error", norm=None, **kwargs):
        self._fixed_norm = norm  # pass the START value to keep the scale fixed across rebuilds
        super().__init__(*args, name=name, **kwargs)

    def build(self, use_jit=True, verbose=1):
        import numpy as np

        super().build(use_jit=use_jit, verbose=verbose)
        n = self._dim_f // 2
        self._dim_f = n
        self._constants["quad_weights"] = self._constants["quad_weights"][:n]
        if self._normalize and self._fixed_norm is not None:
            self._normalization = float(self._fixed_norm)
        elif self._normalize:
            # <|B|> (area-weighted) x sqrt(mean g), from the equilibrium AT BUILD, so that the
            # quadrature-weighted rms of the rows IS the area-weighted rms of B.n / <|B|>
            grid = self._constants["transforms"]["grid"]
            d = self.things[0].compute(["|B|", "|e_theta x e_zeta|"], grid=grid)
            w = np.asarray(grid.weights)
            g = np.asarray(d["|e_theta x e_zeta|"])
            Bmean = np.sum(w * g * np.asarray(d["|B|"])) / np.sum(w * g)
            self._normalization = float(Bmean * np.sqrt(np.sum(w * g) / np.sum(w)))
        if self._target is not None and jnp.size(self._target) > 1:
            self._target = self._target[:n]

    def compute(self, eq_params, *field_params, constants=None):
        """B_coil.n sqrt(|e_theta x e_zeta|) on the boundary [T m]."""
        if field_params == ():
            field_params = None
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            ["R", "phi", "Z", "n_rho", "|e_theta x e_zeta|"],
            params=eq_params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        x = jnp.array([data["R"], data["phi"], data["Z"]]).T
        B = constants["field"].compute_magnetic_field(
            x, source_grid=self._field_grid, basis="rpz", params=field_params,
            chunk_size=self._bs_chunk_size,
        )
        return jnp.sum(B * data["n_rho"], axis=-1) * jnp.sqrt(data["|e_theta x e_zeta|"])

    def print_value(self, args, args0=None, **kwargs):
        return _Objective.print_value(self, args, args0, **kwargs)
