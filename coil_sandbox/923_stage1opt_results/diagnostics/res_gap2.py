"""eps_1 at L=8 and L=12 for each saved equilibrium given on the command line."""
import os, sys
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
BASE = 0.067633

def measure(eq):
    th = 2*np.pi*np.arange(M)/M; ze = 2*np.pi*np.arange(N)/N
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nd = np.asarray(g.nodes)
    nt = np.rint(nd[:,1]/(2*np.pi)*M).astype(int) % M
    nz = np.rint(nd[:,2]/(2*np.pi)*N).astype(int) % N
    c = dict(transforms=get_transforms(CC.DATA_KEYS, obj=eq, grid=g),
             profiles=get_profiles(CC.DATA_KEYS, obj=eq, grid=g),
             perm=jnp.asarray(np.lexsort((nz, nt))),
             Rm=jnp.asarray(eq.surface.R_basis.modes),
             Zm=jnp.asarray(eq.surface.Z_basis.modes),
             theta=jnp.asarray(2*np.pi*np.arange(n)/n),
             trunc=CC.truncation_index(M, N, Mt, Nt))
    x, w = CC._curves_from_params(eq.params_dict, c, M, N, int(eq.NFP), K, 6, 0.0)
    e = float(np.asarray(CC._eps1_live_all(x, w)).mean())
    o = ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched", jac_chunk_size=100)
    o.build(verbose=0)
    return e, float(np.abs(o.compute_scaled_error(o.x(eq))).max())

print(f"{'run':16s} {'L=8 eps1':>18s} {'|f|':>10s} {'L=12 eps1':>18s} {'|f|':>10s} {'GAP':>8s} {'A':>7s}")
for path in sys.argv[1:]:
    tag = os.path.basename(os.path.dirname(path))
    try:
        eq = load(path)
    except Exception as ex:
        print(f"{tag:16s} could not load: {ex}"); continue
    e8, f8 = measure(eq); A = float(eq.compute("R0/a")["R0/a"])
    eq.change_resolution(L=12, M=12, N=12, L_grid=24, M_grid=24, N_grid=24)
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched",
                                         jac_chunk_size=100), maxiter=100, verbose=0)
    e12, f12 = measure(eq)
    p8, p12 = (e8/BASE-1)*100, (e12/BASE-1)*100
    print(f"{tag:16s} {e8:.6f} ({p8:+7.2f}%) {f8:10.3e} {e12:.6f} ({p12:+7.2f}%) {f12:10.3e} "
          f"{abs(p12-p8):7.2f}p {A:7.4f}", flush=True)
