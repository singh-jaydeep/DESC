import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
for f in sys.argv[1:]:
    eq = load(f)
    for rho in (0.9, 0.95, 1.0):
        g = LinearGrid(rho=np.array([rho]), M=64, N=64, NFP=eq.NFP, sym=False)
        d = eq.compute(["sqrt(g)", "|B|", "f_C"], grid=g, helicity=(1, 0))
        s = np.asarray(d["sqrt(g)"])
        r = np.asarray(d["f_C"]) / np.asarray(d["|B|"])**3
        j = np.argmin(np.abs(s))
        print(f"{f} rho {rho}: sqrt(g) min {s.min():.3e} max {s.max():.3e} median {np.median(s):.3e}  at min |sqrt g|: "
              f"theta {g.nodes[j,1]:.3f} zeta {g.nodes[j,2]:.3f}; QS2 max {np.abs(r).max():.3e}")
    print(" boundary spectrum |R_lmn| tail by max(|m|,|n|):", [f"{k}:{np.max(np.abs(eq.Rb_lmn[np.max(np.abs(eq.surface.R_basis.modes[:,1:]),1)==k])):.1e}" for k in range(0, eq.M + 1, 2)])
