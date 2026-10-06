"""Functions for minimization and wrappers for scipy methods."""

from . import _desc_wrappers, _scipy_wrappers
from ._constraint_wrappers import LinearConstraintProjection, ProximalProjection
from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_composite import lsq_auglag_composite
from .aug_lagrangian_ls import lsq_auglag
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .least_squares_composite import lsq_composite
from .optimizer import Optimizer, optimizers, register_optimizer
from .stochastic import sgd
