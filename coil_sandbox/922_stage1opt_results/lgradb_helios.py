import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid

eq = load("sample_equilibria/helios_repro.h5")
if hasattr(eq, "__len__"): eq = eq[-1]
print("NFP", eq.NFP, "R0/a", eq.compute("R0/a")["R0/a"], "a", eq.compute("a")["a"])
g = LinearGrid(rho=1.0, M=64, N=64, NFP=eq.NFP, sym=False)
d = eq.compute(["L_grad(B)", "|B|", "K_vc", "|e_theta x e_zeta|", "theta", "zeta", "R", "Z"], grid=g)
L = np.asarray(d["L_grad(B)"]); B = np.asarray(d["|B|"])
K = np.linalg.norm(np.asarray(d["K_vc"]), axis=1)
dA = np.asarray(d["|e_theta x e_zeta|"]) * g.weights
a = float(eq.compute("a")["a"])
i = int(np.argmin(L))
print(f"a = {a:.4f} m")
print(f"min L_gradB = {L.min():.4f} m   ({L.min()/a:.3f} a)   at R={float(np.asarray(d['R'])[i]):.3f} Z={float(np.asarray(d['Z'])[i]):.3f} theta={float(np.asarray(d['theta'])[i]):.3f} zeta={float(np.asarray(d['zeta'])[i]):.3f}")
for q in [1,5,25,50]:
    print(f"  {q:3d}th pct L_gradB = {np.percentile(L,q):.4f} m  ({np.percentile(L,q)/a:.3f} a)")
print(f"  max  L_gradB = {L.max():.4f} m")
print(f"area-weighted mean L_gradB = {np.sum(L*dA)/np.sum(dA):.4f} m")
print(f"harmonic mean L_gradB      = {np.sum(dA)/np.sum(dA/L):.4f} m")
print(f"|K_vc|: min {K.min():.3e}  max {K.max():.3e} A/m, at max: R={float(np.asarray(d['R'])[int(np.argmax(K))]):.3f} Z={float(np.asarray(d['Z'])[int(np.argmax(K))]):.3f}")
