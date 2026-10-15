"""Boundary repair for free-boundary / single-stage runs. Entry point: repair(eq). Pieces:

  diagnostics  boundary_health, jac_min, displacement (parametrization-independent distance between two surfaces)
  refit        relabelled_points, guarded_fit, reparametrize (label rules and joint label fit from DESC PR 2337)
  interior     how the inside is rebuilt for the new boundary (continuation)
  core         needs_repair (gate), repair (gate -> refit -> interior -> verify, with fallback)
"""
from .core import HEALTHY, needs_repair, repair
from .diagnostics import boundary_health, displacement, jac_min
from .refit import guarded_fit, relabelled_points, reparametrize

__all__ = ["HEALTHY", "needs_repair", "repair", "boundary_health", "displacement", "jac_min", "guarded_fit",
           "relabelled_points", "reparametrize"]
