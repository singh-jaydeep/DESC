"""Rebuild the inside of an equilibrium for a given boundary. Strategies share the signature f(eq, surface, chunk,
verbose) -> Equilibrium with eq's resolution, flux and profiles."""
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium


def _blank(eq, surface):
    return Equilibrium(Psi=float(eq.Psi), NFP=eq.NFP, sym=eq.sym, L=eq.L, M=eq.M, N=eq.N, L_grid=eq.L_grid,
                       M_grid=eq.M_grid, N_grid=eq.N_grid, surface=surface, pressure=eq.pressure, current=eq.current,
                       iota=eq.iota)


def continuation(eq, surface, chunk=200, verbose=0):
    """From scratch: the interior keeps no memory of the old (possibly folded) flux surfaces."""
    # the continuation's perturbation steps build a dense Jacobian (11 GB at L = M = N = 12): hand back whatever the
    # refit and any earlier reports left compiled or cached on the device first
    import gc
    import jax
    gc.collect()
    jax.clear_caches()
    return solve_continuation_automatic(_blank(eq, surface), verbose=verbose, jac_chunk_size=chunk)[-1]


INTERIOR = {"continuation": continuation}
