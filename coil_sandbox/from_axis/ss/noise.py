"""Re-solve noise: perturb interior (not boundary) coefficients by eps, re-solve at tol, measure Boozer QS spread.
    python noise.py EQ.h5 --LMN 12,14 --eps 1e-5 --tol 1e-6 --n 4"""
import sys, argparse, time
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction, QuasisymmetryBoozer
ap = argparse.ArgumentParser()
ap.add_argument("eq"); ap.add_argument("--LMN", default="12,14"); ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--n", type=int, default=4)
ap.add_argument("--device", default="gpu")
a = ap.parse_args()
rng = np.random.default_rng(0)
for LMN in [int(v) for v in a.LMN.split(",")]:
    base = load(a.eq)
    if base.L != LMN:
        base.change_resolution(L=LMN, M=LMN, N=LMN, L_grid=2 * LMN, M_grid=2 * LMN, N_grid=2 * LMN)
    base.solve(objective=ObjectiveFunction(ForceBalance(base, jac_chunk_size=200)), verbose=0, maxiter=300,
               ftol=1e-10, xtol=1e-12, gtol=1e-10)
    qs_objs = []
    for r in (0.25, 0.5, 0.75, 1.0):
        o = QuasisymmetryBoozer(base, helicity=(1, 0), normalize=False, grid=LinearGrid(
            rho=np.array([r]), M=2 * base.M_grid, N=2 * base.N_grid, NFP=base.NFP, sym=False))
        o.build(verbose=0); qs_objs.append(o)
    def qs(eq):
        return np.array([float(np.linalg.norm(o.compute_unscaled(eq.params_dict))) for o in qs_objs])
    q0 = qs(base)
    vals = []
    for i in range(a.n):
        eq = base.copy()
        for c in ("R_lmn", "Z_lmn", "L_lmn"):
            v = np.array(getattr(eq, c)); v = v + a.eps * rng.standard_normal(v.shape) * (np.abs(v) > 0)
            setattr(eq, c, v)
        t = time.time()
        eq.solve(objective=ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)), verbose=0, maxiter=100,
                 ftol=a.tol, xtol=a.tol, gtol=a.tol)
        vals.append(qs(eq))
        print(f"basis {LMN} trial {i}: rel dQS per surface {np.round((vals[-1] - q0) / q0, 6).tolist()} ({time.time() - t:.0f}s)", flush=True)
    vals = np.array(vals)
    print(f"basis {LMN}: QS {q0.round(6).tolist()}  rel spread (std/mean) {np.round(vals.std(0) / q0, 6).tolist()}", flush=True)
