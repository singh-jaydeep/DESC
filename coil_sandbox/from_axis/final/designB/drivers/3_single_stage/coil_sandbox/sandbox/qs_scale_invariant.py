"""`QuasisymmetryTwoTermScaleInvariant`: the two-term QS residual f_C / |B|^3, pointwise.

A port of the `scale_invariant=True` option of DESC PR #2304 (merged 2026-09-21, not in the
vendored DESC): `f_C_normalized = f_C / |B|**3` with the LOCAL field strength, which is the
Landreman & Paul (2022) definition. The stock objective divides by a constant B_scale^3 fixed
at build, so its value moves with the field strength: f_C ~ B^3 at fixed shape.

Why it matters here. At fixed Psi, resizing the plasma changes |B| ~ 1/a^2. The PR's motivation
is inflation (weaker |B| lowers the stock residual). In our single-stage runs the plasma SHRANK
instead (precise_QA, tau_QS x10, k=2: a -5%, <|B|> +11%), which raises the stock residual by
~35% from scale alone -- an accidental brake on shrinking, not a QS measure. This version
removes the scale dependence both ways; plasma size must then be held by its own constraint.

Dimensionless, so DESC's normalization is off. Same grid and quadrature weights as the stock
objective.
"""

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.objectives import QuasisymmetryTwoTerm


class QuasisymmetryTwoTermScaleInvariant(QuasisymmetryTwoTerm):
    """Two-term QS error f_C / |B|^3 at each node (dimensionless)."""

    _units = "(dimensionless)"

    def __init__(self, eq, *args, name="QS two-term (scale-invariant)", **kwargs):
        kwargs["normalize"] = False
        super().__init__(eq, *args, name=name, **kwargs)

    def build(self, use_jit=True, verbose=1):
        super().build(use_jit=use_jit, verbose=verbose)
        eq = self.things[0]
        grid = self._constants["transforms"]["grid"]
        self._data_keys = ["f_C", "|B|"]
        self._constants["profiles"] = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._constants["transforms"] = get_transforms(self._data_keys, obj=eq, grid=grid)

    def compute(self, params, constants=None):
        """f_C / |B|^3 at each node."""
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            helicity=constants["helicity"],
        )
        return data["f_C"] / data["|B|"] ** 3
