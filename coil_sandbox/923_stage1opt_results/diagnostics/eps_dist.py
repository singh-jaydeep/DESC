"""Per-contour eps_1 across the foliation: who actually moved?"""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
sys.path.insert(0, "922_stage1opt_results")
import numpy as np, jax.numpy as jnp
from desc.io import load
from desc.examples import get
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid
import contour_curve as CC

K, M, N, n, Mt = 24, 32, 64, 360, 16
def eps_of(eq):
    th = 2*np.pi*np.arange(M)/M; ze = 2*np.pi*np.arange(N)/N
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nd = np.asarray(g.nodes)
    nt = np.rint(nd[:,1]/(2*np.pi)*M).astype(int) % M
    nz = np.rint(nd[:,2]/(2*np.pi)*N).astype(int) % N
    c = dict(transforms=get_transforms(CC.DATA_KEYS, obj=eq, grid=g),
             profiles=get_profiles(CC.DATA_KEYS, obj=eq, grid=g),
             perm=jnp.asarray(np.lexsort((nz, nt))),
             Rm=jnp.asarray(eq.surface.R_basis.modes), Zm=jnp.asarray(eq.surface.Z_basis.modes),
             theta=jnp.asarray(2*np.pi*np.arange(n)/n),
             trunc=CC.truncation_index(M, N, Mt, Mt))
    x, w = CC._curves_from_params(eq.params_dict, c, M, N, int(eq.NFP), K, 6, 0.0)
    return np.asarray(CC._eps1_live_all(x, w))

B = eps_of(get("precise_QA"))
o = np.argsort(B)                      # order contours by BASELINE eps_1, easiest first
print(f"{'':10s} {'min':>8s} {'p25':>8s} {'median':>8s} {'p75':>8s} {'MAX':>8s} {'mean':>8s} {'spread':>8s}")
def row(tag, e):
    print(f"{tag:10s} {e.min():8.5f} {np.percentile(e,25):8.5f} {np.median(e):8.5f} "
          f"{np.percentile(e,75):8.5f} {e.max():8.5f} {e.mean():8.5f} {e.max()/e.min():8.2f}x")
row("baseline", B)
for tag, p in [("aspin", "live_eps1_L1_aspin"), ("lsqfb", "live_eps1_L1_lsqfb"),
               ("aspin3", "live_eps1_L1_aspin3"), ("worst1", "worst1"), ("worst3", "worst3")]:
    E = eps_of(load(f"922_stage1opt_results/runs/{p}/eq_lo.h5"))
    row(tag, E)
    q = (E[o]/B[o]-1)*100
    print(f"{'':10s} change by baseline rank, easiest->hardest quartile means: "
          + "  ".join(f"{q[i*K//4:(i+1)*K//4].mean():+6.1f}%" for i in range(4)))
