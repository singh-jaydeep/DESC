"""Reduced problem size (m, n) of the constraint-projected least-squares system.

n here is ``LinearConstraintProjection._dim_x_reduced`` -- the dimension lsqtr
actually works in, and the n that sets the cost of the alpha loop. Note that
``lcp.dim_x`` is the FULL objective dimension, which is a larger number.
"""

import json

from .common import MAX_CPU_CONTAINERS, app, cpu_image


@app.function(
    image=cpu_image,
    cpu=8,
    memory=65536,
    timeout=3600,
    max_containers=MAX_CPU_CONTAINERS,
)
def probe(name: str, L: int):
    """Build the projected problem and report its reduced dimensions."""
    from desc.examples import get
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )
    from desc.optimize._constraint_wrappers import LinearConstraintProjection

    eq = get(name)
    eq.change_resolution(L=L, M=L, N=L, L_grid=2 * L, M_grid=2 * L, N_grid=2 * L)
    obj = ObjectiveFunction(ForceBalance(eq))
    cons = get_fixed_boundary_constraints(eq)
    lcp = LinearConstraintProjection(obj, ObjectiveFunction(cons))
    lcp.build(verbose=0)
    m, n = int(lcp.dim_f), int(lcp._dim_x_reduced)
    gb = lambda b: b / 1024**3  # noqa: E731
    return dict(
        name=name,
        L=L,
        m=m,
        n=n,
        n_full=int(lcp.dim_x),
        m_over_n=m / n,
        J_GB=gb(8 * m * n),
        R_GB=gb(8 * n * n),
        stack_GB=gb(16 * n * n),
    )


@app.local_entrypoint()
def main(cases: str = ""):
    """Report reduced sizes for each (case, resolution)."""
    default = [
        (nm, L) for nm in ("precise_QA", "HELIOTRON", "W7-X") for L in (12, 16, 20, 25)
    ]
    todo = json.loads(cases) if cases else default
    rows = []
    for r in probe.starmap(todo, order_outputs=True, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"  FAILED: {type(r).__name__}: {str(r)[:160]}", flush=True)
            continue
        rows.append(r)
        print(
            f"{r['name']:11s} L={r['L']:2d}  m={r['m']:7d}  n={r['n']:6d}  "
            f"m/n={r['m_over_n']:5.2f} | J={r['J_GB']:7.2f} GB  R={r['R_GB']:6.2f} GB "
            f" stack={r['stack_GB']:6.2f} GB",
            flush=True,
        )
    with open("modal_bench/shapes.json", "w") as fh:
        json.dump(rows, fh, indent=2)
