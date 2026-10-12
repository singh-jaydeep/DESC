import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
for f in sys.argv[1:]:
    eq = load(f)
    g = LinearGrid(rho=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["f_C", "|B|", "iota", "B_theta", "B_zeta"], grid=g, helicity=(1, 0))
    r = np.asarray(d["f_C"]) / np.asarray(d["|B|"]) ** 3
    n = g.nodes
    print(f, "|B| range", float(np.min(d["|B|"])), float(np.max(d["|B|"])), "B_zeta(G)", float(np.mean(d["B_zeta"])))
    for rr in np.unique(n[:, 0]):
        m = n[:, 0] == rr
        print(f"  rho {rr:.1f} rms {np.sqrt(np.mean(r[m]**2)):.3e} max {np.abs(r[m]).max():.3e} 99pct {np.percentile(np.abs(r[m]),99):.3e}")
    i = np.argsort(-np.abs(r))[:5]
    for j in i:
        print("   big row", n[j], r[j], float(d["|B|"][j]))
