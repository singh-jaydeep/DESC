"""Local: perturb the boundary the way DESC intends, then solve.

Replacing ``eq.surface`` and re-solving does NOT give a small perturbation: the
interior coefficients are left untouched, so ``LinearConstraintProjection`` has
to jump them onto the new boundary in one discontinuous step. Measured at L=12,
a nominal 1% boundary change that way gives a starting cost of 9.4e24 against
1.5e-9 unperturbed -- 33 orders of magnitude.

``eq.perturb`` instead solves the linearised problem to move the INTERIOR
consistently with the boundary change, and by default weights the perturbation
by (mode number)**2, so high-order modes are not over-driven.

Usage:  python perturb_sweep_local.py <pert> <maxiter> <chunk> [device] [res] [order]
"""

import json
import os
import sys
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

# desc.set_device must run before anything imports jax, so the imports below
# cannot be hoisted -- hence the E402 suppressions.
import desc  # noqa: E402

desc.set_device(sys.argv[4] if len(sys.argv) > 4 else "gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from desc.examples import get  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)

PERT = float(sys.argv[1])
MAXITER = int(sys.argv[2])
CHUNK = int(sys.argv[3])
RES = int(sys.argv[5]) if len(sys.argv) > 5 else 12
ORDER = int(sys.argv[6]) if len(sys.argv) > 6 else 2


def diagnose(eq, tag):
    """Physics diagnostics for one equilibrium."""
    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    lg = LinearGrid(L=20, M=0, N=0, NFP=eq.NFP, axis=False)
    d = eq.compute(["|F|", "<|F|>_vol", "sqrt(g)", "V", "R0/a"], grid=grid)
    io = np.asarray(eq.compute(["iota"], grid=lg)["iota"])
    order = np.argsort(np.asarray(lg.nodes[:, 0]))
    sg = np.asarray(d["sqrt(g)"])
    return dict(
        tag=tag,
        force_vol=float(np.asarray(d["<|F|>_vol"])),
        force_max=float(np.abs(np.asarray(d["|F|"])).max()),
        volume=float(np.asarray(d["V"])),
        aspect=float(np.asarray(d["R0/a"])),
        sqrtg_min=float(sg.min()),
        sqrtg_max=float(sg.max()),
        folded=bool(sg.min() * sg.max() <= 0),
        iota_axis=float(io[order][0]),
        iota_edge=float(io[order][-1]),
    )


def show(d):
    """Print one diagnostics line."""
    print(
        f"  [{d['tag']:<16}] <|F|>={d['force_vol']:>12.4e}  "
        f"max|F|={d['force_max']:>11.4e}  V={d['volume']:.6f}  "
        f"R0/a={d['aspect']:.4f}  folded={d['folded']}  "
        f"iota {d['iota_axis']:.4f}->{d['iota_edge']:.4f}",
        flush=True,
    )


eq = get("precise_QA")
eq.change_resolution(
    L=RES, M=RES, N=RES, L_grid=2 * RES, M_grid=2 * RES, N_grid=2 * RES
)
ref = diagnose(eq, "reference")
show(ref)

# Boundary deltas, each mode scaled by its own magnitude so ||dRb||/||Rb|| = PERT
rng = np.random.default_rng(0)
Rb, Zb = np.asarray(eq.Rb_lmn), np.asarray(eq.Zb_lmn)
dRb = PERT * rng.standard_normal(Rb.size) * np.abs(Rb)
dZb = PERT * rng.standard_normal(Zb.size) * np.abs(Zb)
print(
    f"  ||dRb||/||Rb|| = {np.linalg.norm(dRb)/np.linalg.norm(Rb):.4f}   "
    f"||dZb||/||Zb|| = {np.linalg.norm(dZb)/np.linalg.norm(Zb):.4f}",
    flush=True,
)

t0 = time.perf_counter()
eqp = eq.perturb(
    deltas={"Rb_lmn": dRb, "Zb_lmn": dZb},
    objective=ObjectiveFunction(ForceBalance(eq), jac_chunk_size=CHUNK),
    constraints=get_fixed_boundary_constraints(eq),
    order=ORDER,
    verbose=2,
    copy=True,
)
t_perturb = time.perf_counter() - t0
start = diagnose(eqp, "after perturb")
show(start)

obj = ObjectiveFunction(ForceBalance(eqp), jac_chunk_size=CHUNK)
cons = get_fixed_boundary_constraints(eqp)
t0 = time.perf_counter()
eq_out, res = eqp.solve(
    objective=obj,
    constraints=cons,
    maxiter=MAXITER,
    options={"tr_method": "qr"},
    verbose=1,
    copy=True,
)
wall = time.perf_counter() - t0
solved = diagnose(eq_out, "solved")
show(solved)

dev = jax.local_devices()[0]
st = dev.memory_stats() or {}
print(
    "JSONRESULT "
    + json.dumps(
        dict(
            pert=PERT,
            maxiter=MAXITER,
            chunk=CHUNK,
            res=RES,
            order=ORDER,
            perturb_s=t_perturb,
            solve_s=wall,
            device=str(dev.device_kind),
            peak_GB=st.get("peak_bytes_in_use", 0) / 1024**3,
            n_iter=int(res.nit),
            cost=float(res.cost),
            optimality=float(res.optimality),
            message=str(res.message),
            reference=ref,
            start=start,
            solved=solved,
        )
    ),
    flush=True,
)
