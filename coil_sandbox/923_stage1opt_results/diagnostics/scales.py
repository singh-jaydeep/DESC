"""Pick --lgradb-weight by measurement, and show what --target-mode worst actually asks."""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
sys.path.insert(0, "922_stage1opt_results")
import numpy as np, jax.numpy as jnp
from desc.examples import get
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import GenericObjective
from desc.compute import get_profiles, get_transforms
import contour_curve as CC

K, M, N, n, Mt = 8, 32, 64, 360, 16
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
             theta=jnp.asarray(2*np.pi*np.arange(n)/n), trunc=CC.truncation_index(M, N, Mt, Mt))
    x, w = CC._curves_from_params(eq.params_dict, c, M, N, int(eq.NFP), K, 6, 0.0)
    return np.asarray(CC._eps1_live_all(x, w))

eq0 = get("precise_QA")
E0 = eps_of(eq0)
print(f"baseline eps_1 (K={K}): {np.array2string(np.sort(E0), precision=5)}")
print(f"  min {E0.min():.5f}  max {E0.max():.5f}\n")
print("--target-mode worst: bound T = frac * max(baseline), weight = 1/(max - T)")
print(f"  {'frac':>5s} {'T':>9s} {'vs current min':>15s} {'#above':>7s} {'start cost':>11s}")
for f in (0.50, 0.60, 0.70, 0.80, 0.90):
    T = f*E0.max(); w = 1/max(E0.max()-T, 1e-12); r = w*np.maximum(0, E0-T)
    print(f"  {f:5.2f} {T:9.5f} {T/E0.min():14.3f}x {int((E0>T).sum()):6d} {0.5*np.sum(r**2):11.4f}")

gL = LinearGrid(rho=np.array([1.0]), M=24, N=24, NFP=eq0.NFP, sym=False)
o = GenericObjective("L_grad(B)", thing=eq0, grid=gL); o.build(verbose=0)
L0 = np.asarray(o.compute(*o.xs(eq0)))
floor = 1.0*L0.min()*1.01
print(f"\nL_gradB floor = 1.00 x original min x 1.01 backoff = {floor:.5f} m  (dim_f {L0.size})")
print(f"  {'equilibrium':12s} {'min':>9s} {'#below':>7s} {'sum viol^2':>12s} "
      f"{'w for cost 4':>13s} {'w for cost 40':>14s}")
for tag, eq in [("baseline", eq0),
                ("aspin", load("922_stage1opt_results/runs/live_eps1_L1_aspin/eq_lo.h5")),
                ("aspin3", load("922_stage1opt_results/runs/live_eps1_L1_aspin3/eq_lo.h5"))]:
    oo = GenericObjective("L_grad(B)", thing=eq, grid=gL); oo.build(verbose=0)
    L = np.asarray(oo.compute(*oo.xs(eq)))
    v = np.maximum(0.0, floor - L); ss = float(np.sum(v**2))
    w4 = np.sqrt(2*4.0/ss) if ss > 0 else float("inf")
    w40 = np.sqrt(2*40.0/ss) if ss > 0 else float("inf")
    print(f"  {tag:12s} {L.min():9.5f} {int((L<floor).sum()):6d} {ss:12.4e} "
          f"{w4:13.1f} {w40:14.1f}")
