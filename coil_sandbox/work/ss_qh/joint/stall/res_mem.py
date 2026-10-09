"""Does a basis-RES equilibrium solve fit in GPU memory with an explicit ForceBalance jac_chunk_size?

    python work/ss_qh/joint/stall/res_mem.py EQ.h5 RES GRID_MULT CHUNK [MAXITER]

Raises EQ to L = M = N = RES (grids GRID_MULT x RES), runs MAXITER GN iterations with ForceBalance
(deriv_mode batched, jac_chunk_size CHUNK), and reports dim_x, rows, the peak device memory and the wall time.
"""

import os
import sys
import time

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402

path, res, gm, chunk = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 5
save_to = sys.argv[6] if len(sys.argv) > 6 else None
dev = jax.devices()[0]


def peak():
    st = dev.memory_stats() or {}
    return st.get("peak_bytes_in_use", 0) / 1e9, st.get("bytes_limit", 0) / 1e9


eq, _ = S.load_equilibrium(path)
g = int(round(gm * res))
eq.change_resolution(L=res, M=res, N=res, L_grid=g, M_grid=g, N_grid=g)
fb = O.ForceBalance(eq)
obj = O.ObjectiveFunction((fb,), deriv_mode="batched", jac_chunk_size=chunk)
t0 = time.time()
obj.build(verbose=0)
print(f"RES {res} grid {g} chunk {chunk}: dim_x {eq.dim_x}, rows {obj.dim_f}, dense J {obj.dim_f * eq.dim_x * 8 / 1e9:.2f} GB; "
      f"build {time.time() - t0:.0f} s; peak {peak()[0]:.2f} / limit {peak()[1]:.2f} GB", flush=True)
t0 = time.time()
_, res_ = eq.solve(objective=obj, constraints=O.get_fixed_boundary_constraints(eq), ftol=1e-10, xtol=1e-10,
                   gtol=1e-10, maxiter=maxiter, verbose=0)
wall = time.time() - t0
fc = 0.5 * float(np.sum(np.asarray(obj.compute_scaled_error(obj.x(eq))) ** 2))
print(f"   solve {int(res_['nit'])} its, {wall:.0f} s ({wall / max(int(res_['nit']), 1):.1f} s/it incl. compile); "
      f"force cost {fc:.4e}; peak {peak()[0]:.2f} GB", flush=True)
if save_to:
    eq.save(save_to)
    print(f"   saved {save_to}")
