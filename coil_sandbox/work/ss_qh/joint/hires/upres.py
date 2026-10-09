"""Raise a solved fixed-boundary equilibrium to L=M=N=RES (grids 2x), re-solve, and compare physics with the input
using single_stage's own Boozer QS/iota summary.  Usage: upres.py IN.h5 OUT.h5 RES"""
import os
import sys
import time

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")  # before any desc import, or JAX stays on the CPU
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402
import single_stage as SS  # noqa: E402

src, out, res = sys.argv[1], sys.argv[2], int(sys.argv[3])


def report(e, tag):
    p = SS.Physics(e, "1,4")()
    qg = QuadratureGrid(L=e.L_grid, M=e.M_grid, N=e.N_grid, NFP=e.NFP)
    f = e.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=qg)
    w = np.abs(f["sqrt(g)"]) * qg.weights
    fe = np.sum(f["|F|"] * w) / np.sum(f["|grad(|B|^2)|/2mu0"] * w)
    print(f"{tag} L,M,N {e.L},{e.M},{e.N} grid {e.L_grid},{e.M_grid},{e.N_grid}  V {p['V']:.5f}  "
          f"{SS.fmt_phys(p)}  |F|/|gradB^2/2mu0| {fe:.2e}", flush=True)


eq0 = desc.io.load(src)
report(eq0, "in ")
eq = eq0.copy()
eq.change_resolution(L=res, M=res, N=res, L_grid=2 * res, M_grid=2 * res, N_grid=2 * res)
t = time.time()
eq.solve(ftol=1e-10, xtol=1e-12, gtol=1e-10, maxiter=300, verbose=1)
print(f"solve wall {(time.time() - t) / 60:.1f} min", flush=True)
report(eq, "out")
eq.save(out)
print("wrote", out)
