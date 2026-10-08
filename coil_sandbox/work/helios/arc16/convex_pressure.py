"""How hard is each coil pressing against the convexity floor?

convexity_excess ~ 0 only says the coil IS convex. Because the constraint is enforced
POINTWISE (kappa_s >= 0 at every node), a coil that WANTS to be concave sits with kappa_s
pinned at ~0 over a stretch -- flat/straight sections -- and the integral excess still reads 0.
So the measure of how binding convexity is, is the FRACTION OF ARCLENGTH at kappa_s ~ 0.
"""
import sys, os
os.chdir("/home/singh/Documents/DESC2/coil_sandbox"); sys.path.insert(0, "sandbox")
os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.grid import LinearGrid
from desc.io import load
from convexity import CoilSignedCurvature

def pressure(path, lab):
    cs = load(path)
    obj = CoilSignedCurvature(cs, grid=LinearGrid(N=400))
    obj.build(verbose=0)
    ks = np.asarray(obj.compute(*cs.params_dict if isinstance(cs.params_dict, tuple) else (cs.params_dict,)))
    n = len(list(S.iter_unique(cs)))
    ks = ks.reshape(n, -1) if ks.size % n == 0 else ks[None]
    print(f"\n=== {lab} ===")
    for i, row in enumerate(ks):
        kmax = np.abs(row).max()
        for tol in (0.02, 0.05):
            frac = float((np.abs(row) < tol * kmax).mean())
            if tol == 0.02: f2 = frac
            else: f5 = frac
        print(f"  coil {i}: kappa_s min {row.min():+.4f}  max {row.max():.4f}  "
              f"|kappa_s| < 2% of peak on {f2*100:5.1f}% of nodes, < 5% on {f5*100:5.1f}%")

for lab, p in [("L32 paper (even currents)", "work/helios/arc16/warm_L32_paper/result.h5"),
               ("baseline cut30 (sacrificed)", "work/helios/arc16/cut30/legB/result.h5"),
               ("cold L28 (uncrowded)", "work/helios/arc16/cold_L28_d10/free/result.h5")]:
    if os.path.exists(p):
        try: pressure(p, lab)
        except Exception as e: print(f"\n=== {lab} ===\n  FAILED: {type(e).__name__}: {e}")
