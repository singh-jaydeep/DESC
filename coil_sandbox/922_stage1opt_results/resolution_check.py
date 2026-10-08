"""Does the eps_1 gain survive at higher spectral resolution?

The optimized boundary is far more shaped than precise_QA (|dR|max 17.8% of a) and at
L=M=N=8 force balance cannot get below 1.75e-3 there -- re-solving does not help, so it is
a RESOLUTION limit, not a convergence one. If eps_1 moves when L,M,N rise, part of the
-41.5% is an under-resolution artefact rather than real geometry.
"""
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
from contour_curve import ContourClamshell  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--res", type=int, nargs="+", default=[8, 10, 12])
_p.add_argument("--chunk", type=int, default=None, help="jac_chunk_size for the re-solve")
_p.add_argument("--tr-method", default="qr",
                help="KEEP qr (the DESC default). 'cho' forms J^T J, which SQUARES the "
                     "condition number -- unacceptable when the point of the run is an "
                     "accurately converged equilibrium. Memory must come from chunking, "
                     "not from degrading the factorization.")
_p.add_argument("--maxiter", type=int, default=100)
ARGS = _p.parse_args()

CKW = dict(K=8, M=32, N=64, n=360, Mt=16, Nt=16)


def measure(eq):
    o = ObjectiveFunction(ForceBalance(eq=eq))
    o.build(verbose=0)
    f = float(np.abs(np.asarray(o.compute_scaled_error(o.x(eq)))).max())
    del o
    o = ContourClamshell(eq, quantity="eps1", **CKW)
    o.build(verbose=0)
    e = float(np.asarray(o.compute(eq.params_dict)).mean())
    del o
    o = ContourClamshell(eq, quantity="C", **CKW)
    o.build(verbose=0)
    c = float(np.asarray(o.compute(eq.params_dict)).mean())
    del o
    gc.collect()
    return f, e, c


eq0 = get("precise_QA")
f0, E0, C0 = measure(eq0)
print(f"baseline precise_QA  L{eq0.L}M{eq0.M}N{eq0.N}: |force|max {f0:.3e}  "
      f"eps_1 {E0:.5f}  C {C0:.4f}", flush=True)
del eq0
gc.collect()

for L in ARGS.res:
    eq = load("922_stage1opt_results/runs/loose_eps1_L2/eq_lo.h5")
    if L != eq.L:
        eq.change_resolution(L=L, M=L, N=L, L_grid=2 * L, M_grid=2 * L, N_grid=2 * L)
    t0 = time.time()
    # chunk the re-solve's Jacobian and use the cheap trust-region factorization;
    # the default path OOM'd at 10 GB for L=12
    # NOTE (repeat offence, see NOTES 9c item 4): ObjectiveFunction-level jac_chunk_size
    # is only valid with deriv_mode="batched"; otherwise it raises.
    fobj = ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                             jac_chunk_size=ARGS.chunk)
    eq.solve(objective=fobj, maxiter=ARGS.maxiter, verbose=0,
             options={"tr_method": ARGS.tr_method})
    del fobj
    gc.collect()
    f, e, c = measure(eq)
    print(f"  leg2 at L=M=N={L:2d} (chunk={ARGS.chunk}, tr={ARGS.tr_method}): "
          f"|force|max {f:.3e}  eps_1 {e:.5f} "
          f"({(e/E0-1)*100:+.2f}% vs baseline)  C {c:.4f} ({(c/C0-1)*100:+.2f}%)"
          f"   [{time.time()-t0:.0f}s]", flush=True)
    del eq
    gc.collect()
