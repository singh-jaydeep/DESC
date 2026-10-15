"""Single-stage start from a Phase 2 free-boundary result: one CoilSet(NFP, sym) of the arcs, equilibrium raised to
basis LMN and re-solved (fixed boundary = the free-boundary surface).

    python prep.py ../na/runs2/NAME_fb_a0.12_L10 OUTDIR [--LMN 12]     (from from_axis/ss, GPU)
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.coils import CoilSet
from desc.objectives import ForceBalance, ObjectiveFunction

ap = argparse.ArgumentParser()
ap.add_argument("fb")
ap.add_argument("out")
ap.add_argument("--LMN", type=int, default=12)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
mc = load(os.path.join(a.fb, "coils.h5"))
arcs = [c[0] for c in mc]
nfp = mc[0].NFP
cs = CoilSet(*arcs, NFP=nfp, sym=True)
cs.save(os.path.join(a.out, "coils0.h5"))
print("coils:", len(arcs), "unique, NFP", nfp, "currents", [float(c.current) for c in arcs])
eq = load(os.path.join(a.fb, "eq_fb.h5"))
f0 = ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)); f0.build(verbose=0)
print("L,M,N", eq.L, eq.M, eq.N, "force rms before", float(np.sqrt(np.mean(f0.compute_scaled_error(f0.x(eq)) ** 2))))
if a.LMN != eq.L:
    eq.change_resolution(L=a.LMN, M=a.LMN, N=a.LMN, L_grid=2 * a.LMN, M_grid=2 * a.LMN, N_grid=2 * a.LMN)
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)), verbose=1, maxiter=200,
             ftol=1e-10, xtol=1e-12, gtol=1e-10)
eq.save(os.path.join(a.out, f"eq0_b{a.LMN}.h5"))
print("saved", a.out)
