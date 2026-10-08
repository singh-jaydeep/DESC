"""Is the force-balance degradation CONVERGENCE (re-solve stopped short) or RESOLUTION?
The NOTES 9k test: re-solve the saved equilibrium and see if |force| moves."""
import os, sys, time
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="gpu")
sys.path.insert(0, "922_stage1opt_results")
import numpy as np, jax.numpy as jnp
from desc.io import load
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid
import contour_curve as CC

K, M, N, n, Mt, Nt = 8, 32, 64, 360, 16, 16
def constants(eq):
    th = 2*np.pi*np.arange(M)/M; ze = 2*np.pi*np.arange(N)/N
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nd = np.asarray(g.nodes)
    nt = np.rint(nd[:,1]/(2*np.pi)*M).astype(int) % M
    nz = np.rint(nd[:,2]/(2*np.pi)*N).astype(int) % N
    return dict(transforms=get_transforms(CC.DATA_KEYS, obj=eq, grid=g),
                profiles=get_profiles(CC.DATA_KEYS, obj=eq, grid=g),
                perm=jnp.asarray(np.lexsort((nz, nt))),
                Rm=jnp.asarray(eq.surface.R_basis.modes),
                Zm=jnp.asarray(eq.surface.Z_basis.modes),
                theta=jnp.asarray(2*np.pi*np.arange(n)/n),
                trunc=CC.truncation_index(M, N, Mt, Nt))

def report(eq, tag):
    o = ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched", jac_chunk_size=100)
    o.build(verbose=0)
    f = np.abs(o.compute_scaled_error(o.x(eq))).max()
    x, w = CC._curves_from_params(eq.params_dict, constants(eq), M, N, int(eq.NFP), K, 6, 0.0)
    e = float(np.asarray(CC._eps1_live_all(x, w)).mean())
    a = float(eq.compute("a")["a"]); A = float(eq.compute("R0/a")["R0/a"])
    print(f"  {tag:32s} |force|max {f:.4e}   eps_1 {e:.6f}   a {a:.5f}  A {A:.4f}", flush=True)
    return f, e

eq = load("922_stage1opt_results/runs/live_eps1_L1_lsq/eq_lo.h5")
print("proximal-lsq-exact result (L=M=N=8):", flush=True)
f0, e0 = report(eq, "as saved")

t0 = time.time()
eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                     jac_chunk_size=100), maxiter=100, verbose=0)
print(f"  [re-solved at L=8 in {time.time()-t0:.0f}s]", flush=True)
f1, e1 = report(eq, "after eq.solve(maxiter=100)")

print(f"\n  -> |force| changed by {(f1/f0-1)*100:+.1f}%   eps_1 changed by {(e1/e0-1)*100:+.2f}%")
print("     (NOTES 9k: on leg1/leg2 the re-solve changed NOTHING -> resolution, not convergence)")

t0 = time.time()
eq.change_resolution(L=12, M=12, N=12, L_grid=24, M_grid=24, N_grid=24)
eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                     jac_chunk_size=100), maxiter=100, verbose=0)
print(f"\n  [raised to L=M=N=12 and re-solved in {time.time()-t0:.0f}s]", flush=True)
f2, e2 = report(eq, "at L=M=N=12")
print(f"\n  -> at L=12: |force| {f2:.4e} ({f2/f0:.3f}x the L=8 value), "
      f"eps_1 {e2:.6f} = {(e2/0.067633-1)*100:+.2f}% vs baseline (L=8 said -39.55%)")
