"""Re-solve an equilibrium at basis RES (grids 2 RES) with a chunked ForceBalance, and save it, for a fair re-score.

    python work/ss_qh/joint/stall/rescore12.py IN.h5 OUT.h5 [RES] [MAXITER] [CHUNK]
"""

import os
import sys
import time

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import numpy as np  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
res = int(sys.argv[3]) if len(sys.argv) > 3 else 12
maxiter = int(sys.argv[4]) if len(sys.argv) > 4 else 300
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 200
eq, _ = S.load_equilibrium(src)
if eq.M != res:
    eq.change_resolution(L=res, M=res, N=res, L_grid=2 * res, M_grid=2 * res, N_grid=2 * res)
obj = O.ObjectiveFunction((O.ForceBalance(eq),), deriv_mode="batched", jac_chunk_size=chunk)
obj.build(verbose=0)
t0 = time.time()
_, r = eq.solve(objective=obj, constraints=O.get_fixed_boundary_constraints(eq), ftol=1e-10, xtol=1e-10, gtol=1e-10,
                maxiter=maxiter, verbose=0)
fc = 0.5 * float(np.sum(np.asarray(obj.compute_scaled_error(obj.x(eq))) ** 2))
eq.save(dst)
print(f"RESCORE {src} -> basis {res}: {int(r['nit'])} its {time.time() - t0:.0f} s, force {fc:.4e}; saved {dst}", flush=True)
