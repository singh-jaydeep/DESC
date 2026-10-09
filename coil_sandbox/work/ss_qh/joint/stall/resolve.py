"""Re-solve an equilibrium tightly (fixed boundary) and save it, to re-score a loose-solve result.

    python work/ss_qh/joint/stall/resolve.py IN.h5 OUT.h5 [TOL] [MAXITER]
"""

import os
import sys

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(SB, "sandbox"))
import boot  # noqa: E402

boot.setup(default="gpu")

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
maxiter = int(sys.argv[4]) if len(sys.argv) > 4 else 200
eq, _ = S.load_equilibrium(src)
eq.solve(objective="force", constraints=O.get_fixed_boundary_constraints(eq), ftol=tol, xtol=tol, gtol=tol,
         maxiter=maxiter, verbose=3)
eq.save(dst)
print(f"saved {dst}")
