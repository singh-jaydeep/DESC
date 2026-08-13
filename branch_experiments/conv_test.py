"""Do the two methods agree at CONVERGENCE, or only mid-trajectory?

The profile/bench runs stop at maxiter, so their 'final cost' is a mid-trajectory
point of a nonlinear solve -- and a ~1e-12 step difference can be amplified into a
visibly different iterate. If the two flags are numerically equivalent, they should
agree once the solve actually CONVERGES (gtol/ftol satisfied), even if intermediate
iterates differ. Runs each method in a fresh process, to convergence.
"""
import sys, os, json
sys.path.insert(0, ".")
os.environ.setdefault("DESC_BENCH_DEVICE","cpu")
import numpy as np, desc
desc.set_device(os.environ["DESC_BENCH_DEVICE"])
from desc.examples import get
from desc.objectives import ForceBalance, ObjectiveFunction, get_fixed_boundary_constraints

def solve(meth, name="HELIOTRON", L=6, pert=0.01, maxiter=200):
    eq = get(name)
    eq.change_resolution(L=L, M=L, N=eq.N, L_grid=2*L, M_grid=2*L, N_grid=2*eq.N)
    rng = np.random.default_rng(0)
    eq.R_lmn = eq.R_lmn + pert*rng.standard_normal(eq.R_lmn.size)*np.abs(eq.R_lmn).max()
    eq.Z_lmn = eq.Z_lmn + pert*rng.standard_normal(eq.Z_lmn.size)*np.abs(eq.Z_lmn).max()
    obj = ObjectiveFunction(ForceBalance(eq)); cons = get_fixed_boundary_constraints(eq)
    out = eq.solve(objective=obj, constraints=cons, ftol=1e-12, xtol=1e-12, gtol=1e-12,
                   maxiter=maxiter, options={"tr_method": meth}, verbose=0, copy=True)
    res = out[1] if isinstance(out,(tuple,list)) else out
    return dict(meth=meth, cost=float(res.cost), nit=int(res.nit),
                msg=str(getattr(res,"message",""))[:80],
                success=bool(getattr(res,"success",False)))

if __name__ == "__main__":
    r = solve(sys.argv[1])
    print("CONV " + json.dumps(r), flush=True)
