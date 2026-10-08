"""Is the L_gradB change local or global? Ratio field vs baseline at matched (theta, zeta)."""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
from desc.io import load
from desc.examples import get
from desc.grid import LinearGrid

g = LinearGrid(rho=np.array([1.0]), theta=64, zeta=128, NFP=1, sym=False)
def fld(eq):
    a = float(eq.compute("a")["a"])
    return np.asarray(eq.compute("L_grad(B)", grid=g)["L_grad(B)"]).reshape(128, 64) / a

B = fld(get("precise_QA"))
print(f"{'run':10s} {'eps1':>8s} {'ratio p5':>9s} {'p50':>8s} {'p95':>8s} {'%worse':>8s} "
      f"{'%worse>10':>10s} {'min-region':>28s}")
for tag, path, e in [("aspin", "runs/live_eps1_L1_aspin/eq_lo.h5", "-10.31%"),
                     ("lsqfb", "runs/live_eps1_L1_lsqfb/eq_lo.h5", "-27.21%"),
                     ("aspin3", "runs/live_eps1_L1_aspin3/eq_lo.h5", "-32.91%")]:
    F = fld(load("922_stage1opt_results/" + path))
    r = F / B
    # where is the global minimum of L_gradB, and did it move?
    i0, j0 = np.unravel_index(np.argmin(F), F.shape)
    ib, jb = np.unravel_index(np.argmin(B), B.shape)
    zeta0, th0 = 360*i0/128, 360*j0/64
    zb, tb = 360*ib/128, 360*jb/64
    print(f"{tag:10s} {e:>8s} {np.percentile(r,5):9.3f} {np.percentile(r,50):8.3f} "
          f"{np.percentile(r,95):8.3f} {100*np.mean(r<1):7.0f}% {100*np.mean(r<0.9):9.0f}% "
          f"   min at (z={zeta0:5.1f},th={th0:5.1f}) vs base (z={zb:.1f},th={tb:.1f})")
print("\n  %worse    = fraction of the surface where L_gradB/a FELL (coils must come closer)")
print("  %worse>10 = fraction where it fell by more than 10%")
