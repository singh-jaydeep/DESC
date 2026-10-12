import sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
from desc.objectives import QuasisymmetryBoozer
eq = load("qa2best/ss7_b14_fb/eq_fb.h5")
for rho in (0.5, 1.0):
    g = LinearGrid(rho=np.array([rho]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)
    o = QuasisymmetryBoozer(eq, helicity=(1, 0), normalize=False, grid=g); o.build(verbose=0)
    f = np.asarray(o.compute_unscaled(eq.params_dict))
    d = eq.compute(["|B|_mn_B", "B modes"], grid=g, M_booz=2 * eq.M, N_booz=2 * eq.N)
    b = np.asarray(d["|B|_mn_B"]).ravel(); md = np.asarray(d["B modes"])
    br = b[md[:, 2] != 0]
    print(rho, "objective: n", f.size, "norm", np.linalg.norm(f), "max", np.abs(f).max(), "| direct: n", br.size, "norm",
          np.linalg.norm(br), "max", np.abs(br).max(), "M_booz obj", o._constants["transforms"]["B"].basis.M if "transforms" in o._constants else None,
          "B00", b[(md[:, 1] == 0) & (md[:, 2] == 0)])
    for M_, N_ in ((eq.M, eq.N), (3 * eq.M, 3 * eq.N)):
        d2 = eq.compute(["|B|_mn_B", "B modes"], grid=g, M_booz=M_, N_booz=N_)
        b2 = np.asarray(d2["|B|_mn_B"]).ravel(); m2 = np.asarray(d2["B modes"])
        print("   M_booz", M_, "breaking norm", np.linalg.norm(b2[m2[:, 2] != 0]), "max", np.abs(b2[m2[:, 2] != 0]).max())
