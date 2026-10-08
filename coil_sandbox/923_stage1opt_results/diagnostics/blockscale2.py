"""Per-term Jacobian block scale w.r.t. the FULL equilibrium state.

Valid for comparing blocks: ProximalProjection applies the same linear map to the COLUMNS of
every block, so relative row scaling between terms is preserved.
"""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
sys.path.insert(0, "922_stage1opt_results")
import numpy as np
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import AspectRatio, ForceBalance, GenericObjective, ObjectiveFunction
from contour_curve import ContourClamshell

eq = get("precise_QA"); NFP = int(eq.NFP)
CKW = dict(K=8, M=32, N=64, n=360, Mt=16, Nt=16)
oE = ContourClamshell(eq, quantity="eps1", **CKW); oE.build(verbose=0)
E0 = np.asarray(oE.compute(eq.params_dict)); T = 0.60*E0.max(); w_eps = 1.0/(E0.max()-T)
gi = LinearGrid(rho=np.array([0.,.25,.5,.75,1.]), M=eq.M, N=eq.N, NFP=NFP)
_io = GenericObjective("iota", thing=eq, grid=gi); _io.build(verbose=0)
iota0 = np.asarray(_io.compute(*_io.xs(eq)))
gL = LinearGrid(rho=np.array([1.0]), M=24, N=24, NFP=NFP, sym=False)
_lg = GenericObjective("L_grad(B)", thing=eq, grid=gL); _lg.build(verbose=0)
L0 = np.asarray(_lg.compute(*_lg.xs(eq)))

terms = [
  ("clamshell eps1", ContourClamshell(eq, quantity="eps1",
      bounds=(np.zeros_like(E0), np.full_like(E0,T)), weight=w_eps, **CKW)),
  ("aspect ratio",   AspectRatio(eq=eq, bounds=(5.99,6.01), weight=10/0.01)),
  ("iota",           GenericObjective("iota", thing=eq, grid=gi, weight=10/0.05,
                                      bounds=(iota0-0.05, iota0+0.05))),
  ("L_gradB floor",  GenericObjective("L_grad(B)", thing=eq, grid=gL, weight=5.0,
                                      bounds=(float(L0.min()), np.inf))),
  ("force w=500",    ForceBalance(eq=eq, weight=500.0)),
  ("force w=1 (ref)",ForceBalance(eq=eq, weight=1.0)),
]
print(f"\n{'term':17s} {'dim_f':>6s} {'||r||':>11s} {'||J||_F':>11s} {'max row':>11s} "
      f"{'max col':>11s}")
out=[]
for name, o in terms:
    of = ObjectiveFunction(o, deriv_mode="batched", jac_chunk_size=200)
    of.build(verbose=0)
    x = of.x(eq)
    r = np.asarray(of.compute_scaled_error(x))
    J = np.asarray(of.jac_scaled_error(x))
    out.append((name, o.dim_f, r, J))
    print(f"{name:17s} {o.dim_f:6d} {np.linalg.norm(r):11.3e} {np.linalg.norm(J):11.3e} "
          f"{np.linalg.norm(J,axis=1).max():11.3e} {np.linalg.norm(J,axis=0).max():11.3e}")
act = [o for o in out if not o[0].startswith("force w=1")]
tot = sum(np.sum(J**2) for _,_,_,J in act)
print(f"\nshare of total ||J||^2 in the ACTUAL objective:")
for name,_,_,J in act:
    print(f"  {name:17s} {100*np.sum(J**2)/tot:7.3f}%")
