"""Stage-1 clamshell optimization on a vacuum equilibrium.  See SPEC_stage1.md.

v0 SCOPE, and it is a real restriction: the proxy curves' `zeta_n` are FIXED. They are
attached to the equilibrium (SurfaceCurveConsistency ties their surface params to eq's),
so C responds to the boundary moving under a curve at FIXED (theta, zeta) -- it is NOT
re-solved to stay a streamline. The streamline constraint of SPEC §3c needs a two-thing
objective (B comes from eq, the tangent from the curve) which is not built yet.

So v0 answers "does this run at a reasonable pace with the full equilibrium DOF in the
augmented Lagrangian", which is the open question, and it does NOT answer the science
question. Do not read a C change from v0 as evidence about clamshell-ability.

    python 922_stage1opt_results/stage1_clamshell.py --arm lo --maxiter 25
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["lo", "mid", "hi"], default="lo")
p.add_argument("--K", type=int, default=4, help="proxy curves over one field period")
p.add_argument("--maxiter", type=int, default=25)
p.add_argument("--mmax", type=int, default=2, help="free boundary modes |m|,|n| <= mmax")
p.add_argument("--ns", type=int, default=12, help="curve zeta Fourier resolution")
p.add_argument("--nodes", type=int, default=240, help="nodes per curve")
p.add_argument("--chunk", type=int, default=100, help="jac_chunk_size")
p.add_argument("--out", default="922_stage1opt_results/runs/v0")
p.add_argument("--device", default="gpu")
args = p.parse_args()
boot.setup(default=args.device)

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import phi_contours as P  # noqa: E402
from clamshell import best_breaks, freeze  # noqa: E402
from desc.examples import get  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixParameters,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    ObjectiveFromUser,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    SurfaceCurveConsistency,
)
from desc.optimize import Optimizer  # noqa: E402
from desc.utils import rpz2xyz  # noqa: E402
import jax.numpy as jnp  # noqa: E402

os.makedirs(args.out, exist_ok=True)
TARGET = {"lo": 0.0, "hi": 4.0}          # "mid" filled from the baseline below

# ---------------------------------------------------------------- equilibrium
eq = get("precise_QA")
NFP = eq.NFP
print(f"precise_QA  L{eq.L} M{eq.M} N{eq.N} NFP{NFP} sym={eq.sym}  dim_x={eq.dim_x}", flush=True)

# ---------------------------------------------------------------- proxy curves
pot = P.current_potential(eq, M=96, N=192)
s = pot.summary()
print(f"potential: G {s['G']:.4e}  I {s['I']:.3e}  I/G {s['I_over_G']:+.2e}  "
      f"residual {s['residual']:.2e}  monotonic {s['monotonic']}", flush=True)
cons = P.modular_contours(pot, args.K * NFP, n_theta=4 * args.nodes)[: args.K]
curves = [P.to_surface_curve(t, z, eq=eq, N_fourier=args.ns, name=f"proxy{k}")
          for k, (t, z) in enumerate(cons)]

sgrid = Grid(np.stack([np.zeros(args.nodes), np.zeros(args.nodes),
                       np.linspace(0, 2 * np.pi, args.nodes, endpoint=False)], 1),
             sort=False, jitable=False)
CKW = {"secular_theta": 1, "secular_zeta": 0}


def _pts(c):
    d = c.compute(["x", "x_s"], grid=sgrid, basis="xyz")
    return np.asarray(d["x"]), np.linalg.norm(np.asarray(d["x_s"]), axis=1)


FROZEN = []
for c in curves:
    X, W = _pts(c)
    FROZEN.append((freeze(X, W, best_breaks(X, W, 1, stride=4)[1]),
                   freeze(X, W, best_breaks(X, W, 2, stride=4)[1])))


def _eps_expr(x, w, fz):
    """eps from frozen masks/normals; identical algebra to clamshell.eps_from_frozen."""
    masks, nrms = jnp.asarray(fz["masks"]), jnp.asarray(fz["normals"])
    tot = 0.0
    for k in range(masks.shape[0]):
        ww = w * masks[k]
        Wt = jnp.sum(ww)
        D = x - jnp.sum(ww[:, None] * x, axis=0) / Wt
        tot = tot + jnp.sum(ww * (D @ nrms[k]) ** 2)
    Wa = jnp.sum(w)
    D = x - jnp.sum(w[:, None] * x, axis=0) / Wa
    M = (D * w[:, None]).T @ D / Wa
    l0 = jnp.asarray(fz["n0"]) @ M @ jnp.asarray(fz["n0"])
    tr = jnp.trace(M)
    l1l2 = 0.5 * (tr**2 - jnp.trace(M @ M)) - l0 * (tr - l0)
    return jnp.sqrt(tot / Wa) / jnp.sqrt(jnp.sqrt(jnp.clip(l1l2, 1e-30)))


def make_C(fz1, fz2):
    def fun(grid, data):
        x = rpz2xyz(data["x"])                 # compute_fun returns rpz
        w = jnp.linalg.norm(data["x_s"], axis=-1)
        return jnp.atleast_1d(4.0 * _eps_expr(x, w, fz2) / _eps_expr(x, w, fz1))
    return fun


def make_eps1(fz1):
    def fun(grid, data):
        x = rpz2xyz(data["x"])
        w = jnp.linalg.norm(data["x_s"], axis=-1)
        return jnp.atleast_1d(_eps_expr(x, w, fz1))
    return fun


# baseline values
C0, E10 = [], []
for c, (f1, f2) in zip(curves, FROZEN):
    o = ObjectiveFromUser(fun=make_C(f1, f2), thing=c, grid=sgrid, compute_kwargs=CKW)
    o.build(verbose=0)
    C0.append(float(np.asarray(o.compute(*o.xs(c)))[0]))
    o1 = ObjectiveFromUser(fun=make_eps1(f1), thing=c, grid=sgrid, compute_kwargs=CKW)
    o1.build(verbose=0)
    E10.append(float(np.asarray(o1.compute(*o1.xs(c)))[0]))
print(f"baseline C   {np.array2string(np.array(C0), precision=4)}  mean {np.mean(C0):.4f}", flush=True)
print(f"baseline e1  {np.array2string(np.array(E10), precision=4)}", flush=True)

# ---------------------------------------------------------------- objective
tgt = C0 if args.arm == "mid" else [TARGET[args.arm]] * args.K
objs = []
for k, (c, (f1, f2)) in enumerate(zip(curves, FROZEN)):
    objs.append(ObjectiveFromUser(fun=make_C(f1, f2), thing=c, grid=sgrid,
                                  compute_kwargs=CKW, target=tgt[k], weight=1.0,
                                  normalize=False, name=f"C{k}",
                                  jac_chunk_size=args.chunk))
    objs.append(ObjectiveFromUser(fun=make_eps1(f1), thing=c, grid=sgrid,
                                  compute_kwargs=CKW, target=E10[k], weight=10.0,
                                  normalize=False, name=f"eps1_{k}",
                                  jac_chunk_size=args.chunk))
# chunking goes per sub-objective: with several things DESC picks deriv_mode="blocked",
# and ObjectiveFunction-level jac_chunk_size is only valid for "batched"
objective = ObjectiveFunction(objs)

# ---------------------------------------------------------------- constraints
gq = LinearGrid(M=eq.M, N=eq.N, NFP=NFP, rho=np.linspace(0.1, 1.0, 5), sym=True)
qs0 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=gq)
qs0.build(verbose=0)
f_qs = np.asarray(qs0.compute_scaled_error(*qs0.xs(eq)))
qs_rms = float(np.sqrt(np.mean(f_qs**2)))
gi = LinearGrid(rho=np.array([0.0, 0.25, 0.5, 0.75, 1.0]), M=eq.M, N=eq.N, NFP=NFP)
# read the baseline THROUGH the objective, so target always matches its dim_f
_io = GenericObjective("iota", thing=eq, grid=gi)
_io.build(verbose=0)
iota0 = np.asarray(_io.compute(*_io.xs(eq)))
print(f"baseline QS rms {qs_rms:.4e}   iota {iota0.min():.5f}..{iota0.max():.5f}", flush=True)

mR = eq.surface.R_basis.modes
mZ = eq.surface.Z_basis.modes
freeR = (np.abs(mR[:, 1]) <= args.mmax) & (np.abs(mR[:, 2]) <= args.mmax) & \
        ~((mR[:, 1] == 0) & (mR[:, 2] == 0))
freeZ = (np.abs(mZ[:, 1]) <= args.mmax) & (np.abs(mZ[:, 2]) <= args.mmax)
print(f"free boundary DOF: R {freeR.sum()} + Z {freeZ.sum()} = {freeR.sum()+freeZ.sum()}", flush=True)

constraints = [
    ForceBalance(eq=eq, jac_chunk_size=args.chunk),
    FixBoundaryR(eq=eq, modes=mR[~freeR]),
    FixBoundaryZ(eq=eq, modes=mZ[~freeZ]),
    FixPressure(eq=eq),
    FixCurrent(eq=eq),
    FixPsi(eq=eq),
    AspectRatio(eq=eq, target=6.0, weight=10.0),
    GenericObjective("iota", thing=eq, grid=gi, target=iota0, weight=10.0),
    QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=gq, bounds=(-qs_rms, qs_rms)),
]
for c in curves:
    constraints += [SurfaceCurveConsistency(eq, c),
                    FixParameters(c, {"theta_n": True, "zeta_n": True,
                                      "rotmat": True, "shift": True})]

# ---------------------------------------------------------------- run
opt = Optimizer("lsq-auglag")
t0 = time.time()
(eq_new, *rest), result = opt.optimize(
    (eq, *curves), objective, constraints, maxiter=args.maxiter, verbose=3,
    options={"second_order": "constraints"}, copy=True)
dt = time.time() - t0
print(f"\nwall clock {dt:.1f} s for {args.maxiter} iterations "
      f"-> {dt/max(args.maxiter,1):.2f} s/iteration", flush=True)

eq_new.save(os.path.join(args.out, f"eq_{args.arm}.h5"))
print("saved", os.path.join(args.out, f"eq_{args.arm}.h5"), flush=True)
